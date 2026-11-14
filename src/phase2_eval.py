"""Frozen Phase II evaluation from bo.pdf, Eqs. (11)-(13).

Self-query appends x_k to the COMPLETE noisy context, retaining y_k at its
original position. Leave-one-out removes that one pair and is diagnostic only.
Unseen queries are the next state of the same stationary feature sequence.
Importance sampling is an optional, explicitly approximate Bayes diagnostic;
it never silently substitutes clean risk or oracle risk for excess risk.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path

from munch import unmunchify
import torch
import torch.nn.functional as F

from phase2 import Phase2Classification, binary_labels, validate_phase2_config
from runtime import capture_rng_state, restore_rng_state, configure_reproducibility, resolve_device, runtime_signature
from samplers import get_data_sampler

PROTOCOL_VERSION = "phase2-v1"


@dataclass
class EvaluationSettings:
    seed: int = 1000003
    num_tasks: int = 1024
    batch_size: int = 16
    query_chunk_size: int = 4
    leave_one_out: bool = True
    bayes_samples: int = 0
    bayes_min_ess: float = 100.0
    save_per_task: bool = True

    def __post_init__(self):
        for name in ("num_tasks", "batch_size", "query_chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("evaluation seed must be an integer in [0,2**32)")
        if not isinstance(self.bayes_samples, int) or self.bayes_samples < 0:
            raise ValueError("bayes_samples must be a nonnegative integer")
        if not math.isfinite(self.bayes_min_ess) or self.bayes_min_ess < 1:
            raise ValueError("bayes_min_ess must be finite and >=1")


def append_query(context_x, context_y, query_x):
    """Final y=0 is a dummy slot AFTER the queried x and cannot be attended to."""
    xs = torch.cat((context_x, query_x[:, None]), dim=1)
    ys = torch.cat((context_y, context_y.new_zeros((len(context_y), 1))), dim=1)
    return xs, ys


def self_query_batch(xs, ys, indices, leave_one_out=False):
    """Flatten task x selected-query pairs, retaining task-major ordering."""
    b, m, d = xs.shape
    contexts_x, contexts_y = [], []
    for k in indices:
        keep = torch.arange(m, device=xs.device) != k if leave_one_out else torch.ones(m, dtype=torch.bool, device=xs.device)
        contexts_x.append(xs[:, keep])
        contexts_y.append(ys[:, keep])
    cx = torch.stack(contexts_x, dim=1).reshape(b * len(indices), m - int(leave_one_out), d)
    cy = torch.stack(contexts_y, dim=1).reshape(b * len(indices), m - int(leave_one_out))
    q = xs[:, indices].reshape(b * len(indices), d)
    return append_query(cx, cy, q)


def self_query_logits(model, xs, ys, chunk_size, leave_one_out=False):
    pieces = []
    for start in range(0, xs.shape[1], chunk_size):
        indices = list(range(start, min(start + chunk_size, xs.shape[1])))
        px, py = self_query_batch(xs, ys, indices, leave_one_out)
        pieces.append(model(px, py, inds=[px.shape[1] - 1]).reshape(len(xs), len(indices)))
    return torch.cat(pieces, dim=1)


def exact_match_copy(context_x, context_y, queries):
    """Last exact match wins; unseen-feature fallback is the fixed class +1."""
    if context_x.shape[1] == 0:
        return context_y.new_ones(queries.shape[:2])
    match = (queries[:, :, None] == context_x[:, None]).all(dim=-1)
    positions = torch.arange(context_x.shape[1], device=context_x.device)
    last = torch.where(match, positions, -1).max(dim=-1).values
    copied = context_y.gather(1, last.clamp_min(0))
    return torch.where(last >= 0, copied, torch.ones_like(copied))


def importance_bayes_clean_probability(context_x, context_y, query_x, task, n_samples):
    """Posterior P(clean y_q=+1 | noisy C,x_q) under the stated Gaussian prior.

    Neither the true task weight nor clean context/query labels are inputs.
    Report ESS: the prior proposal can be very poor for large m or small eta.
    Markov feature likelihood cancels because features are independent of w.
    """
    b, _, d = context_x.shape
    weights = torch.randn(b, n_samples, d, device=context_x.device) / math.sqrt(d)
    scores = torch.einsum("bmd,bsd->bsm", context_x, weights) + task.bias
    if task.corruption == "flip":
        match = binary_labels(scores) == context_y[:, None]
        log_likelihood = torch.where(match, math.log1p(-task.eta), math.log(task.eta)).double().sum(-1)
    else:
        log_likelihood = torch.special.log_ndtr(context_y[:, None].double() * scores.double() / task.noise_std).sum(-1)
    posterior = log_likelihood.softmax(-1)
    query_scores = torch.einsum("bd,bsd->bs", query_x, weights) + task.bias
    positive_probability = (posterior * (query_scores >= 0)).sum(-1)
    ess = posterior.square().sum(-1).reciprocal()
    return positive_probability, ess


def mean_and_se(values):
    x = torch.as_tensor(values, dtype=torch.float64)
    return {"mean": x.mean().item(), "se_over_tasks": x.std().item() / math.sqrt(len(x)) if len(x) > 1 else None}


def subset_accuracy(records, correct_key, count_key):
    counts = torch.tensor([r[count_key] for r in records], dtype=torch.float64)
    correct = torch.tensor([r[correct_key] for r in records], dtype=torch.float64)
    if counts.sum() == 0:
        return {"accuracy": None, "count": 0, "se_over_tasks": None}
    ratio = (correct.sum() / counts.sum()).item()
    # Cluster delta-method SE: examples within a task need not be independent.
    residual = correct - ratio * counts
    se = residual.std().item() / (math.sqrt(len(records)) * counts.mean().item()) if len(records) > 1 else None
    return {"accuracy": ratio, "count": int(counts.sum()), "se_over_tasks": se}


def summarize(records, corruption, eta, settings):
    metric_keys = [key for key in records[0] if not key.endswith("_count") and not key.endswith("_correct") and not key.startswith("bayes_")]
    metrics = {key: mean_and_se([r[key] for r in records]) for key in metric_keys}
    metrics["corrupted_memorization"] = subset_accuracy(records, "corrupted_mem_correct", "corrupted_count")
    metrics["copy_corrupted_memorization"] = subset_accuracy(records, "copy_corrupted_mem_correct", "corrupted_count")
    if settings.leave_one_out:
        metrics["corrupted_leave_one_out"] = subset_accuracy(records, "corrupted_loo_correct", "corrupted_count")
    if corruption == "flip":
        metrics["flipped_memorization"] = dict(metrics["corrupted_memorization"])
    # This is deliberately null unless a checked Bayes approximation is requested.
    metrics["clean_excess_risk_estimate"] = None
    bayes = {"method": "not_computed", "certified": False}
    if settings.bayes_samples:
        ess = [r["bayes_ess"] for r in records]
        reliable = min(ess) >= settings.bayes_min_ess
        bayes = {
            "method": "prior_importance_sampling", "certified": False,
            "samples_per_task": settings.bayes_samples,
            "min_ess": min(ess), "mean_ess": sum(ess) / len(ess),
            "ess_threshold_met": reliable,
            "warning": "Finite-sample posterior approximation, not exact Bayes risk; ESS alone is not an accuracy certificate.",
            "clean_risk_plugin_estimate": mean_and_se([r["bayes_clean_risk_plugin"] for r in records]),
            "classifier_clean_error": mean_and_se([r["bayes_classifier_clean_error"] for r in records]),
        }
        if reliable:
            metrics["clean_excess_risk_estimate"] = mean_and_se([r["query_clean_error"] - r["bayes_clean_risk_plugin"] for r in records])
    return {
        "metrics": metrics,
        "bayes": bayes,
        "noise_entropy_floor": -eta * math.log(eta) - (1 - eta) * math.log1p(-eta) if corruption == "flip" else None,
        "interpretation": "Self-query exposes the observed label. Low memorization error alone does not establish benign overfitting. LOO is diagnostic, not the interpolation criterion. Bayes/excess estimates are approximate when present.",
    }


@torch.inference_mode()
def evaluate_phase2(model, conf, settings=None):
    if conf.training.task != "phase2_classification":
        raise ValueError("Phase II evaluation requires a Phase II classification checkpoint")
    validate_phase2_config(conf)
    settings = settings or EvaluationSettings(**dict(conf.get("phase2_eval") or {}))
    if settings.seed == conf.seed:
        raise ValueError("Use a distinct evaluation seed for disjoint fresh evaluation tasks")
    device = resolve_device(next(model.parameters()).device)
    old_rng, was_training = capture_rng_state(), model.training
    records = []
    try:
        configure_reproducibility(settings.seed, conf.deterministic)
        model.eval()
        n_dims, n_points = conf.model.n_dims, conf.training.curriculum.points.end
        sampler = get_data_sampler(conf.training.data, n_dims, **(conf.training.data_kwargs or {}))
        for start in range(0, settings.num_tasks, settings.batch_size):
            b = min(settings.batch_size, settings.num_tasks - start)
            task = Phase2Classification(n_dims, b, **conf.training.task_kwargs)
            # Draw once: the last state is the unseen Markov continuation.
            xs = sampler.sample_xs(n_points, b)
            labels = task.sample_labels(xs)
            xs = xs.to(device)
            ys, clean, corrupted = [value.to(device) for value in (labels.observed, labels.clean, labels.corrupted)]
            cx, cy, q = xs[:, :-1], ys[:, :-1], xs[:, -1]
            prefix_logits = model(xs, ys)
            fresh_bce = F.binary_cross_entropy_with_logits(prefix_logits, (ys + 1) / 2, reduction="none").mean(1)
            query_prediction = binary_labels(prefix_logits[:, -1])
            mem = binary_labels(self_query_logits(model, cx, cy, settings.query_chunk_size))
            copied = exact_match_copy(cx, cy, cx)
            copy_query = exact_match_copy(cx, cy, q[:, None])[:, 0]
            noisy_context = corrupted[:, :-1]
            loo = binary_labels(self_query_logits(model, cx, cy, settings.query_chunk_size, True)) if settings.leave_one_out else None
            probabilities, ess = (importance_bayes_clean_probability(cx, cy, q, task, settings.bayes_samples) if settings.bayes_samples else (None, None))
            for j in range(b):
                r = {
                    "fresh_prompt_bce": fresh_bce[j].item(),
                    "memorization_error": (mem[j] != cy[j]).float().mean().item(),
                    "query_clean_error": float(query_prediction[j] != clean[j, -1]),
                    "query_noisy_error": float(query_prediction[j] != ys[j, -1]),
                    "copy_memorization_error": (copied[j] != cy[j]).float().mean().item(),
                    "copy_query_clean_error": float(copy_query[j] != clean[j, -1]),
                    "copy_query_noisy_error": float(copy_query[j] != ys[j, -1]),
                    "corrupted_count": int(noisy_context[j].sum()),
                    "corrupted_mem_correct": int(((mem[j] == cy[j]) & noisy_context[j]).sum()),
                    "copy_corrupted_mem_correct": int(((copied[j] == cy[j]) & noisy_context[j]).sum()),
                }
                if task.corruption == "flip":
                    expected_noisy = task.eta + (1 - 2 * task.eta) * r["query_clean_error"]
                    r["query_noisy_identity_residual"] = r["query_noisy_error"] - expected_noisy
                if loo is not None:
                    r["leave_one_out_error"] = (loo[j] != cy[j]).float().mean().item()
                    r["corrupted_loo_correct"] = int(((loo[j] == cy[j]) & noisy_context[j]).sum())
                if probabilities is not None:
                    p = probabilities[j].item()
                    r["bayes_clean_risk_plugin"] = min(p, 1 - p)
                    r["bayes_classifier_clean_error"] = float((1 if p >= 0.5 else -1) != clean[j, -1].item())
                    r["bayes_ess"] = ess[j].item()
                records.append(r)
        result = summarize(records, task.corruption, task.eta, settings)
        result["metadata"] = {
            "protocol": PROTOCOL_VERSION, "source": "bo.pdf Phase II Eqs. (7)-(13)",
            "d": n_dims, "m": n_points - 1,
            "task_kwargs": unmunchify(conf.training.task_kwargs),
            "data_kwargs": unmunchify(conf.training.data_kwargs),
            "evaluation": asdict(settings), "runtime": runtime_signature(),
            "copy_unseen_fallback": "+1", "sign_at_zero": "+1",
        }
        if settings.save_per_task:
            result["per_task"] = records
        return result
    finally:
        model.train(was_training)
        restore_rng_state(old_rng)


def evaluate_run(run_path, step=-1, cache=True, **overrides):
    from eval import get_model_from_run
    run_path = Path(run_path)
    _, conf = get_model_from_run(run_path, only_conf=True)
    if conf.training.task != "phase2_classification":
        raise ValueError("Phase II evaluation requires a Phase II classification checkpoint")
    validate_phase2_config(conf)
    settings = EvaluationSettings(**{**dict(conf.get("phase2_eval") or {}), **overrides})
    checkpoint = run_path / ("state.pt" if step == -1 else f"model_{step}.pt")
    with checkpoint.open("rb") as fp:
        digest = hashlib.file_digest(fp, "sha256").hexdigest()
    config_digest = hashlib.sha256(json.dumps(unmunchify(conf), sort_keys=True).encode()).hexdigest()
    output = run_path / ("phase2_metrics.json" if step == -1 else f"phase2_metrics_{step}.json")
    if cache and output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        meta = saved.get("metadata", {})
        if (meta.get("checkpoint_sha256"), meta.get("config_sha256"), meta.get("protocol"), meta.get("evaluation")) == (digest, config_digest, PROTOCOL_VERSION, asdict(settings)):
            return saved
    old_rng = capture_rng_state()
    try:
        model, _ = get_model_from_run(run_path, step)
        result = evaluate_phase2(model, conf, settings)
    finally:
        restore_rng_state(old_rng)
    result["metadata"].update(checkpoint_sha256=digest, config_sha256=config_digest, checkpoint=str(checkpoint))
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    result["meta_training"] = {
        "last_batch_bce": state.get("last_meta_train_loss") if step == -1 else None,
        "train_step": state.get("train_step") if step == -1 else step,
        "note": "Last training-batch BCE is distinct from fresh_prompt_bce and memorization error.",
    }
    if cache:
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bayes-samples", type=int)
    args = vars(parser.parse_args())
    run_dir, step = args.pop("run_dir"), args.pop("step")
    result = evaluate_run(run_dir, step, **{k: v for k, v in args.items() if v is not None})
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
