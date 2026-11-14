import os
import json
import math
from random import randint
import uuid

from quinine import QuinineArgumentParser
from munch import unmunchify
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from phase2 import validate_phase2_config
from runtime import (
    resolve_device, configure_reproducibility, configure_backend, capture_rng_state,
    restore_rng_state, runtime_signature,
)

import wandb


def train_step(model, xs, ys, optimizer, loss_func):
    if next(model.parameters()).device.type != "cuda" or xs.device.type != "cuda" or ys.device.type != "cuda":
        raise ValueError("Training requires model and inputs on CUDA")
    optimizer.zero_grad(set_to_none=False)
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    validate_phase2_config(args)
    device = next(model.parameters()).device
    resolve_device(device)
    configure_backend(args.deterministic)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Training equivalence requires FP32 model parameters")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.training.learning_rate, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0, amsgrad=False, foreach=False, fused=False,
    )
    curriculum = Curriculum(args.training.curriculum)
    model.train()

    # Only run length, output paths and checkpoint/logging intervals may change.
    signature = {
        "model": unmunchify(args.model),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "device": str(device),
        "training": {
            key: value for key, value in unmunchify(args.training).items()
            if key not in {"train_steps", "save_every_steps", "keep_every_steps", "resume_id"}
            and not (key == "data_kwargs" and not value)
        },
    }

    starting_step = 0
    state = None
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        # RNG byte tensors must remain on CPU; optimizer.load_state_dict moves
        # optimizer tensors to the parameter device.
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("checkpoint_version") != 2:
            raise ValueError("Legacy checkpoint lacks RNG/task-pool state; exact resume is unavailable")
        if state["training_signature"] != signature:
            raise ValueError("Training configuration changed; exact resume would not be equivalent")
        if state["runtime_signature"] != runtime_signature():
            raise ValueError("Execution stack changed; exact resume requires the same CUDA/runtime configuration")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"] + 1
        if args.training.train_steps < starting_step:
            raise ValueError("Requested training length precedes the checkpoint")
        curriculum.load_state_dict(state["curriculum_state"])
        if curriculum.step_count != starting_step:
            raise ValueError("Checkpoint curriculum and training step disagree")

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **(args.training.data_kwargs or {}))
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        pool_dict=state["task_pool_dict"] if state is not None else None,
        num_tasks=args.training.num_tasks if state is None else None,
        **args.training.task_kwargs,
    )
    # Restore only after constructors and all one-time initialization. In
    # particular, never regenerate a finite task pool after restoring RNG.
    if state is not None:
        restore_rng_state(state["rng_state"])
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples
    metrics_path = None
    if args.training.task == "phase2_classification" and not args.test_run:
        metrics_path = os.path.join(args.out_dir, "training_metrics.jsonl")
        # An interrupted run may have logged updates after the latest saved
        # checkpoint. Drop those stale rows before replaying the updates.
        if os.path.exists(metrics_path):
            rows = []
            with open(metrics_path, encoding="utf-8") as fp:
                for line in fp:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a partially written last line after a crash
                    if row["step"] < starting_step:
                        rows.append(row)
            with open(metrics_path + ".tmp", "w", encoding="utf-8") as fp:
                for row in rows:
                    fp.write(json.dumps(row, allow_nan=False) + "\n")
            os.replace(metrics_path + ".tmp", metrics_path)

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        log_now = i % args.wandb.log_every_steps == 0 or (metrics_path is not None and i == args.training.train_steps - 1)
        if log_now and not args.test_run:
            if args.training.task == "phase2_classification":
                log_metrics = {
                    "meta_train_bce": loss,
                    "meta_train_accuracy": 1 - point_wise_loss.mean().item(),
                    "pointwise/error": dict(zip(point_wise_tags, point_wise_loss.cpu().tolist())),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                }
                if task.corruption == "flip":
                    eta = task.eta
                    log_metrics["noise_entropy_floor"] = -eta * math.log(eta) - (1 - eta) * math.log1p(-eta)
            else:
                log_metrics = {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                }
            if metrics_path is not None:
                with open(metrics_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps({"step": i, **log_metrics}, allow_nan=False) + "\n")
            wandb.log(log_metrics, step=i)

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if (i % args.training.save_every_steps == 0 or i == args.training.train_steps - 1) and not args.test_run:
            training_state = {
                "checkpoint_version": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
                "last_meta_train_loss": loss,
                "curriculum_state": curriculum.state_dict(),
                "task_pool_dict": task_sampler.pool_dict,
                "rng_state": capture_rng_state(),
                "training_signature": signature,
                "runtime_signature": runtime_signature(),
            }
            temporary_path = state_path + ".tmp"
            torch.save(training_state, temporary_path)
            os.replace(temporary_path, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    validate_phase2_config(args)
    device = resolve_device(args.device)
    configure_reproducibility(args.seed, args.deterministic)
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            mode=args.wandb.mode,
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.to(device)
    model.train()

    train(model, args)

    if not args.test_run and args.eval_after_train:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
