"""Resume must reproduce the uninterrupted training trajectory on the same GPU/stack.

SDPA/eager parity is tolerance-based: fused attention changes rounding, so it is
not a promise of bitwise equivalence to PyTorch 1.11 or across hardware versions.
Run GPU acceptance checks with: pytest --require-cuda -m cuda -q
"""

import copy
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from curriculum import Curriculum
from models import TransformerModel
from runtime import capture_rng_state, restore_rng_state, configure_reproducibility, resolve_device
from tasks import get_task_sampler
from train import train, train_step
from test_compatibility import config


def draw_all_streams():
    return {
        "python": random.random(),
        "numpy": np.random.random(),
        "numpy_normal": np.random.normal(),
        "cpu": torch.rand(5),
        "cuda": [torch.rand(5, device=f"cuda:{i}").cpu() for i in range(torch.cuda.device_count())],
    }


def assert_identical(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_identical(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_identical(a, b)
    else:
        assert actual == expected


def test_rng_weights_only_roundtrip(tmp_path):
    configure_reproducibility(37)
    torch.randn(7)
    np.random.normal()  # includes NumPy's cached Gaussian state
    state = capture_rng_state()
    expected = draw_all_streams()
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    draw_all_streams()
    restore_rng_state(torch.load(path, weights_only=True, map_location="cpu"))
    assert_identical(draw_all_streams(), expected)


def test_gpu_only_guards():
    with pytest.raises(ValueError, match="CUDA"):
        resolve_device("cpu")
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device()
    model = TransformerModel(3, 3, n_embd=16, n_layer=1, n_head=2)
    assert model._backbone.config._attn_implementation == "sdpa"
    xs, ys = torch.randn(2, 3, 3), torch.randn(2, 3)
    with pytest.raises(ValueError, match="CUDA"):
        model(xs, ys)
    with pytest.raises(ValueError, match="CUDA"):
        train_step(model, xs, ys, torch.optim.Adam(model.parameters()), torch.nn.functional.mse_loss)


def test_curriculum_roundtrip():
    args = config("smoke.yaml").training.curriculum
    args.dims.interval, args.points.interval = 2, 3
    a, b = Curriculum(args), Curriculum(args)
    for _ in range(7):
        a.update()
    b.load_state_dict(a.state_dict())
    for _ in range(5):
        a.update()
        b.update()
        assert a.state_dict() == b.state_dict()


def test_finite_task_pool_roundtrip(tmp_path):
    configure_reproducibility(29)
    original = get_task_sampler("linear_regression", 5, 2, num_tasks=7)
    path = tmp_path / "pool.pt"
    torch.save({"pool": original.pool_dict, "rng": capture_rng_state()}, path)
    expected = original().w_b
    state = torch.load(path, weights_only=True)
    resumed = get_task_sampler("linear_regression", 5, 2, pool_dict=state["pool"])
    restore_rng_state(state["rng"])
    torch.testing.assert_close(resumed().w_b, expected, rtol=0, atol=0)


def test_sdpa_math_reference():
    # Test the attention equation itself on the available host. Production
    # Transformer/training entry points remain CUDA-only.
    configure_reproducibility(17)
    q, k, v = [torch.randn(2, 2, 7, 8, requires_grad=True) for _ in range(3)]
    reference_inputs = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    q_ref, k_ref, v_ref = reference_inputs
    logits = q_ref @ k_ref.transpose(-1, -2) / (8 ** 0.5)
    mask = torch.ones(7, 7, dtype=torch.bool).tril()
    expected = logits.masked_fill(~mask, float("-inf")).softmax(-1) @ v_ref
    actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for a, b in zip((q, k, v), reference_inputs):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_cuda_rng_roundtrip(tmp_path):
    test_rng_weights_only_roundtrip(tmp_path)


def run_segment(args, model):
    trajectory = []

    def observe(model, xs, ys, optimizer, loss_func):
        loss, pred = train_step(model, xs, ys, optimizer, loss_func)
        trajectory.append({
            "loss": loss, "prediction": pred.cpu(), "xs": xs.cpu(), "ys": ys.cpu(),
            "random_probes": draw_all_streams(),
        })
        return loss, pred

    with patch("train.train_step", side_effect=observe), patch("train.wandb.log"):
        train(model, args)
    return trajectory


def tiny_model():
    return TransformerModel(5, 11, n_embd=16, n_layer=1, n_head=2).cuda()


@pytest.mark.cuda
@pytest.mark.parametrize("num_tasks,num_examples", [(None, None), (7, None), (None, 19), (7, 19)])
def test_interrupted_matches_uninterrupted(tmp_path, num_tasks, num_examples):
    args = config("smoke.yaml")
    args.seed = 43
    args.training.num_tasks = num_tasks
    args.training.num_training_examples = num_examples
    args.training.train_steps = 6
    args.training.save_every_steps = 4  # final checkpoint must also save step 5
    args.training.curriculum.dims.start = 3
    args.training.curriculum.dims.interval = 2
    args.training.curriculum.points.start = 5
    args.training.curriculum.points.interval = 2
    args.out_dir = str(tmp_path / "continuous")
    (tmp_path / "continuous").mkdir()
    configure_reproducibility(args.seed)
    uninterrupted = tiny_model()
    expected_trace = run_segment(args, uninterrupted)
    expected_next_rng = draw_all_streams()
    expected_checkpoint = torch.load(tmp_path / "continuous" / "state.pt", weights_only=True)
    assert expected_checkpoint["train_step"] == 5

    args.out_dir = str(tmp_path / "resumed")
    (tmp_path / "resumed").mkdir()
    args.training.train_steps = 3
    configure_reproducibility(args.seed)
    interrupted = tiny_model()
    trace = run_segment(args, interrupted)

    # Simulate a process restart: all RNG streams and model initialization
    # change before restoring the checkpoint.
    configure_reproducibility(9876)
    draw_all_streams()
    resumed = tiny_model()
    args.training.train_steps = 6
    trace += run_segment(args, resumed)
    assert_identical(draw_all_streams(), expected_next_rng)
    assert_identical(trace, expected_trace)
    assert_identical(resumed.state_dict(), uninterrupted.state_dict())
    final = torch.load(tmp_path / "resumed" / "state.pt", weights_only=True)
    assert_identical(final, expected_checkpoint)

    # Refuse a silent change in the training experiment on resume.
    args.training.learning_rate *= 2
    with pytest.raises(ValueError, match="configuration changed"):
        run_segment(args, resumed)


@pytest.mark.cuda
def test_sdpa_eager_forward_backward_adam_parity():
    configure_reproducibility(41)
    sdpa = tiny_model().train()
    eager = copy.deepcopy(sdpa)
    eager._backbone.set_attn_implementation("eager")
    opts = [torch.optim.Adam(m.parameters(), lr=1e-4, foreach=False, fused=False) for m in (sdpa, eager)]
    with patch("torch.nn.functional.scaled_dot_product_attention", wraps=torch.nn.functional.scaled_dot_product_attention) as call:
        for _ in range(5):
            xs, ys = torch.randn(4, 11, 5, device="cuda"), torch.randn(4, 11, device="cuda")
            outputs = [m(xs, ys) for m in (sdpa, eager)]
            torch.testing.assert_close(*outputs, rtol=2e-4, atol=2e-5)
            for opt, pred in zip(opts, outputs):
                opt.zero_grad()
                torch.nn.functional.mse_loss(pred, ys).backward()
            for a, b in zip(sdpa.parameters(), eager.parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, rtol=2e-3, atol=2e-5)
            for opt in opts:
                opt.step()
            for a, b in zip(sdpa.parameters(), eager.parameters()):
                torch.testing.assert_close(a, b, rtol=2e-4, atol=5e-5)
        assert call.call_count == 5  # eager reference must not invoke SDPA
