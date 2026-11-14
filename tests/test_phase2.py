import copy
import json
import math
from unittest.mock import patch

from munch import Munch
import pytest
import torch
import torch.nn.functional as F
import yaml

from models import build_model
from phase2 import Phase2Classification, binary_labels, classification_bce, validate_phase2_config
from phase2_eval import (
    EvaluationSettings, append_query, self_query_batch, self_query_logits,
    exact_match_copy, importance_bayes_clean_probability, summarize,
    evaluate_phase2, evaluate_run,
)
from runtime import capture_rng_state, configure_reproducibility
from samplers import GaussianSampler
from test_compatibility import config
from test_reproducibility import assert_identical, draw_all_streams, run_segment


def test_confirmed_phase2_setup():
    args = config("phase2.yaml")
    assert (args.model.n_embd, args.model.n_layer, args.model.n_head) == (64, 3, 2)
    assert args.model.n_dims == 20
    assert args.model.n_positions == args.training.curriculum.points.end == 41
    assert args.training.train_steps == 100000
    assert args.training.data_kwargs.rho == 0
    assert args.training.task_kwargs.eta == 0.1
    assert args.training.num_tasks is args.training.num_training_examples is None
    state = capture_rng_state()
    validate_phase2_config(args)
    assert_identical(capture_rng_state(), state)
    probit = config("phase2_probit.yaml")
    assert probit.training.task_kwargs.eta is None
    assert probit.training.task_kwargs.noise_std == 1
    validate_phase2_config(probit)
    assert config("phase2_markov.yaml").training.data_kwargs.rho == 0.9


@pytest.mark.parametrize("change", ["pool", "examples", "curriculum", "scale", "positions"])
def test_reject_departure_from_phase2_protocol(change):
    args = config("phase2.yaml")
    if change == "pool":
        args.training.num_tasks = 100
    elif change == "examples":
        args.training.num_training_examples = 100
    elif change == "curriculum":
        args.training.curriculum.dims.start = 5
    elif change == "scale":
        args.training.data_kwargs.scale = 2
    else:
        args.model.n_positions = 40
    with pytest.raises(ValueError):
        validate_phase2_config(args)


def test_gaussian_prior_and_independent_flip_rate():
    configure_reproducibility(111)
    task = Phase2Classification(20, 12000, eta=0.2)
    assert abs(task.w_b.square().mean().item() - 1 / 20) < 0.001
    assert abs(task.w_b.mean().item()) < 0.002
    xs = torch.randn(12000, 8, 20)
    labels = task.sample_labels(xs)
    assert set(labels.clean.unique().tolist()) == {-1, 1}
    assert torch.equal(labels.clean, binary_labels((xs @ task.w_b)[:, :, 0]))
    assert abs(labels.corrupted.float().mean().item() - 0.2) < 0.006
    for mask in (labels.margins > 1, labels.margins < -1):
        assert abs(labels.corrupted[mask].float().mean().item() - 0.2) < 0.015


@pytest.mark.parametrize("kwargs", [
    {"eta": 0.5}, {"eta": 0}, {"eta": float("nan")},
    {"corruption": "flip", "noise_std": 1},
    {"corruption": "probit", "eta": 0.1},
    {"corruption": "probit", "noise_std": 0},
    {"seeds": [1]}, {"pool_dict": {"w": torch.zeros(1, 20, 1)}},
])
def test_invalid_corruption_or_nonfresh_tasks(kwargs):
    with pytest.raises(ValueError):
        Phase2Classification(20, 1, **kwargs)


def test_probit_is_sign_of_noisy_margin():
    task = Phase2Classification(2, 1, corruption="probit", noise_std=0.5)
    task.w_b = torch.tensor([[[1.0], [0.0]]])
    xs = torch.tensor([[[1.0, 0.0], [-0.2, 0.0], [0.1, 0.0]]])
    eps = torch.tensor([[-3.0, 1.0, -1.0]])
    with patch("torch.randn_like", return_value=eps):
        labels = task.sample_labels(xs)
    torch.testing.assert_close(labels.observed, binary_labels(labels.margins + 0.5 * eps))
    assert torch.equal(labels.corrupted, labels.observed != labels.clean)


@pytest.mark.parametrize("rho", [0.0, 0.6, 0.9])
def test_stationary_markov_marginals_and_lag_covariance(rho):
    torch.manual_seed(121)
    xs = GaussianSampler(3, rho=rho).sample_xs(10, 10000)
    assert abs(xs.mean().item()) < 0.02
    for t in (0, 1, 9):
        assert abs(xs[:, t].square().mean().item() - 1) < 0.025
    for lag in (1, 3):
        covariance = (xs[:, :-lag] * xs[:, lag:]).mean().item()
        assert abs(covariance - rho ** lag) < 0.025


def test_iid_sampler_keeps_original_draws_and_seeded_markov():
    torch.manual_seed(123)
    expected = torch.randn(2, 7, 3)
    torch.manual_seed(123)
    torch.testing.assert_close(GaussianSampler(3).sample_xs(7, 2), expected, rtol=0, atol=0)
    sampler = GaussianSampler(3, rho=0.8)
    torch.testing.assert_close(sampler.sample_xs(7, 2, seeds=[2, 3]), sampler.sample_xs(7, 2, seeds=[2, 3]), rtol=0, atol=0)
    for rho in (-0.1, 1, float("nan")):
        with pytest.raises(ValueError):
            GaussianSampler(3, rho=rho)


def test_bce_logits_matches_equation_10_and_is_stable():
    logits = torch.tensor([[-1e6, 1e6, -0.7, 0.2]], requires_grad=True)
    labels = torch.tensor([[1.0, -1.0, -1.0, 1.0]])
    actual = classification_bce(logits, labels)
    torch.testing.assert_close(actual, F.softplus(-labels * logits).mean())
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_self_query_retains_noisy_label_and_loo_removes_only_queried_pair():
    xs = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    ys = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    rng = capture_rng_state()
    for loo in (False, True):
        px, py = self_query_batch(xs, ys, [0, 2], leave_one_out=loo)
        assert px.shape == (4, 4 - int(loo), 2)
        assert torch.equal(py[:, -1], torch.zeros(4))
        for b in range(2):
            for j, k in enumerate([0, 2]):
                row = 2 * b + j
                torch.testing.assert_close(px[row, -1], xs[b, k])
                keep = [i for i in range(3) if i != k] if loo else [0, 1, 2]
                torch.testing.assert_close(px[row, :-1], xs[b, keep])
                torch.testing.assert_close(py[row, :-1], ys[b, keep])
    assert_identical(capture_rng_state(), rng)  # no re-sampling corruption


class CopyPredictor:
    def __call__(self, xs, ys, inds):
        assert inds == [xs.shape[1] - 1]
        assert torch.equal(ys[:, -1], torch.zeros(len(ys)))
        return exact_match_copy(xs[:, :-1], ys[:, :-1], xs[:, -1:])


def test_copy_baseline_exposes_self_query_shortcut_and_chunk_order():
    xs = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
    ys = torch.tensor([[-1., 1., -1., -1., 1.], [1., -1., -1., 1., -1.]])
    for chunk in (1, 2, 9):
        torch.testing.assert_close(self_query_logits(CopyPredictor(), xs, ys, chunk), ys)
        torch.testing.assert_close(self_query_logits(CopyPredictor(), xs, ys, chunk, True), torch.ones_like(ys))
    qx, qy = append_query(xs, ys, torch.tensor([[100., 101.], [102., 103.]]))
    torch.testing.assert_close(CopyPredictor()(qx, qy, [5]), torch.ones(2, 1))


@pytest.mark.parametrize("corruption", ["flip", "probit"])
def test_importance_bayes_likelihood_with_known_two_particle_posterior(corruption):
    options = {"eta": 0.1} if corruption == "flip" else {"noise_std": 1.0}
    task = Phase2Classification(1, 1, corruption=corruption, **options)
    particles = torch.tensor([[[1.0], [-1.0]]])
    with patch("torch.randn", return_value=particles):
        p, ess = importance_bayes_clean_probability(torch.ones(1, 1, 1), torch.ones(1, 1), torch.ones(1, 1), task, 2)
    expected = 0.9 if corruption == "flip" else torch.special.ndtr(torch.tensor(1.)).item()
    assert p.item() == pytest.approx(expected, abs=1e-7)
    assert ess.item() == pytest.approx(1 / (expected ** 2 + (1 - expected) ** 2), abs=1e-6)


def test_metrics_do_not_claim_exact_excess_and_handle_empty_flipped_subset():
    records = [{
        "memorization_error": 0., "query_clean_error": 0., "query_noisy_error": 0.,
        "corrupted_count": 0, "corrupted_mem_correct": 0, "copy_corrupted_mem_correct": 0,
    }]
    settings = EvaluationSettings(leave_one_out=False)
    result = summarize(records, "flip", 0.1, settings)
    assert result["metrics"]["flipped_memorization"]["accuracy"] is None
    assert result["metrics"]["clean_excess_risk_estimate"] is None
    assert result["noise_entropy_floor"] == pytest.approx(-0.1 * math.log(0.1) - 0.9 * math.log(0.9))
    records[0].update(bayes_ess=2., bayes_clean_risk_plugin=0.1, bayes_classifier_clean_error=0.)
    result = summarize(records, "flip", .1, EvaluationSettings(leave_one_out=False, bayes_samples=100, bayes_min_ess=10))
    assert not result["bayes"]["ess_threshold_met"]
    assert result["metrics"]["clean_excess_risk_estimate"] is None
    json.dumps(result, allow_nan=False)


class ReferenceCopyModule(torch.nn.Module):
    """Protocol-only reference for host tests; never replaces CUDA training."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, xs, ys, inds=None):
        indices = range(xs.shape[1]) if inds is None else inds
        return torch.cat([
            2 * exact_match_copy(xs[:, :k], ys[:, :k], xs[:, k:k+1])
            for k in indices
        ], dim=1)


@pytest.mark.parametrize("variant", ["phase2.yaml", "phase2_markov.yaml", "phase2_probit.yaml"])
def test_protocol_engine_with_reference_copy_predictor(variant):
    args = config(variant)
    model = ReferenceCopyModule().train()
    rng = capture_rng_state()
    # Only the device precondition is mocked for this non-neural reference.
    # Production Transformer forward and train_step still reject CPU tensors.
    with patch("phase2_eval.resolve_device", return_value=torch.device("cpu")):
        result = evaluate_phase2(model, args, EvaluationSettings(
            num_tasks=5, batch_size=2, query_chunk_size=3, bayes_samples=64, bayes_min_ess=1,
        ))
    assert model.training
    assert model.anchor.grad is None
    assert_identical(capture_rng_state(), rng)
    assert len(result["per_task"]) == 5
    assert result["metrics"]["memorization_error"]["mean"] == 0
    assert result["metrics"]["copy_memorization_error"]["mean"] == 0
    assert result["metrics"]["corrupted_memorization"]["accuracy"] == 1
    assert result["metrics"]["leave_one_out_error"]["mean"] > 0
    assert not result["bayes"]["certified"]
    if variant == "phase2_probit.yaml":
        assert result["noise_entropy_floor"] is None
        assert "query_noisy_identity_residual" not in result["metrics"]
    json.dumps(result, allow_nan=False)


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["phase2.yaml", "phase2_markov.yaml", "phase2_probit.yaml"])
def test_phase2_resume_is_identical(tmp_path, variant):
    args = config(variant)
    args.training.batch_size, args.training.train_steps = 2, 4
    args.training.keep_every_steps = -1
    args.training.save_every_steps = 3
    args.out_dir = str(tmp_path / "full")
    (tmp_path / "full").mkdir()
    configure_reproducibility(args.seed)
    full = build_model(args.model)
    full_trace = run_segment(args, full)
    full_rng = draw_all_streams()
    checkpoint = torch.load(tmp_path / "full" / "state.pt", weights_only=True)
    args.out_dir = str(tmp_path / "split")
    (tmp_path / "split").mkdir()
    args.training.train_steps = 2
    configure_reproducibility(args.seed)
    split = build_model(args.model)
    trace = run_segment(args, split)
    with (tmp_path / "split" / "training_metrics.jsonl").open("a") as fp:
        fp.write('{"step":999,"meta_train_bce":999}\n{"partial":')
    configure_reproducibility(999)
    split = build_model(args.model)
    args.training.train_steps = 4
    trace += run_segment(args, split)
    assert_identical(trace, full_trace)
    assert_identical(draw_all_streams(), full_rng)
    assert_identical(split.state_dict(), full.state_dict())
    assert_identical(torch.load(tmp_path / "split" / "state.pt", weights_only=True), checkpoint)
    logged = [json.loads(line) for line in (tmp_path / "split" / "training_metrics.jsonl").read_text().splitlines()]
    assert all(row["step"] <= 3 for row in logged)
    assert logged[-1]["step"] == 3


@pytest.mark.cuda
def test_phase2_train_and_frozen_evaluation_roundtrip(tmp_path):
    args = config("phase2_smoke.yaml")
    args.out_dir = str(tmp_path)
    configure_reproducibility(args.seed)
    model = build_model(args.model)
    run_segment(args, model)
    logged = [json.loads(line) for line in (tmp_path / "training_metrics.jsonl").read_text().splitlines()]
    assert logged[-1]["step"] == 2
    assert "meta_train_bce" in logged[-1]
    assert "excess_loss" not in logged[-1]
    before = copy.deepcopy(model.state_dict())
    rng = capture_rng_state()
    result = evaluate_phase2(model, args)
    assert model.training
    assert_identical(model.state_dict(), before)
    assert_identical(capture_rng_state(), rng)
    assert result["metadata"]["m"] == 40
    assert len(result["per_task"]) == 4
    assert result["metrics"]["copy_memorization_error"]["mean"] == 0
    assert result["metrics"]["clean_excess_risk_estimate"] is None
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(Munch.toDict(args)))
    saved = evaluate_run(tmp_path)
    assert saved == evaluate_run(tmp_path)
    assert saved["meta_training"]["train_step"] == 2
    assert saved["meta_training"]["last_batch_bce"] >= 0
    assert (tmp_path / "phase2_metrics.json").exists()
    # Last dummy label cannot influence the logit at the preceding query x.
    xs = torch.randn(2, 41, 20, device="cuda")
    ys = torch.ones(2, 41, device="cuda")
    with torch.no_grad():
        a = model(xs, ys, inds=[40])
        ys[:, -1] = -1
        torch.testing.assert_close(model(xs, ys, inds=[40]), a, rtol=0, atol=0)
