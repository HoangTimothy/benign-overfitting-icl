"""Scientific invariants and real train/save/resume/eval integration."""

import copy
from unittest.mock import patch
from pathlib import Path

import pytest
import torch
import yaml
from munch import Munch
from quinine import QuinineArgumentParser

from eval import eval_batch, get_model_from_run
from models import TransformerModel, get_relevant_baselines, GDModel
from base_models import NeuralNetwork
from schema import schema
from tasks import get_task_sampler
from train import train, train_step

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    with patch("sys.argv", ["train.py", "--config", str(ROOT / "src" / "conf" / name)]):
        return QuinineArgumentParser(schema=schema).parse_quinfig()


@pytest.mark.parametrize("path", sorted((ROOT / "src" / "conf").glob("*.yaml")))
def test_original_configs(path):
    if path.name not in {"base.yaml", "wandb.yaml"}:
        assert config(path.name).model.family == "gpt2"


@pytest.mark.cuda
def test_causal_interleaving_and_gradient():
    torch.manual_seed(7)
    model = TransformerModel(3, 5, n_embd=16, n_layer=1, n_head=2).cuda().eval()
    xs, ys = torch.randn(2, 5, 3, device="cuda"), torch.randn(2, 5, device="cuda")
    before = model(xs, ys)
    changed = ys.clone()
    changed[:, 2:] += 100
    torch.testing.assert_close(model(xs, changed)[:, :3], before[:, :3])
    changed_x = xs.clone()
    changed_x[:, 3:] += 100
    torch.testing.assert_close(model(changed_x, ys)[:, :3], before[:, :3])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = model._read_in.weight.detach().clone()
    loss, _ = train_step(model, xs, ys, optimizer, torch.nn.functional.mse_loss)
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(initial, model._read_in.weight)


@pytest.mark.cuda
def test_checkpoint_and_resume(tmp_path):
    import wandb
    run = wandb.init(mode="disabled")
    args = config("smoke.yaml")
    args.out_dir = str(tmp_path)
    model = TransformerModel(**{
        k: v for k, v in args.model.items() if k != "family"
    }).cuda()
    train(model, args)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(Munch.toDict(args)))
    loaded, _ = get_model_from_run(tmp_path)
    xs, ys = torch.randn(2, 5, 5, device="cuda"), torch.randn(2, 5, device="cuda")
    torch.testing.assert_close(model(xs, ys), loaded(xs, ys))
    # A completed run must perform zero additional updates when resumed.
    before = copy.deepcopy(loaded.state_dict())
    train(loaded, args)
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    metric = eval_batch(loaded.eval(), get_task_sampler("linear_regression", 5, 2), xs.cpu())
    assert metric.shape == (2, 5)
    assert torch.isfinite(metric).all()
    run.finish()


@pytest.mark.parametrize("task", ["linear_regression", "sparse_linear_regression", "linear_classification", "relu_2nn_regression", "decision_tree"])
@pytest.mark.cuda
def test_task_and_baselines(task):
    xs = torch.randn(2, 4, 20)
    ys = get_task_sampler(task, 20, 2)().evaluate(xs)
    for baseline in get_relevant_baselines(task):
        if isinstance(baseline, GDModel):
            baseline.num_steps = 2
        pred = baseline(xs, ys, inds=[0, 3])
        assert pred.shape == (2, 2)
        assert torch.isfinite(pred).all()


@pytest.mark.cuda
def test_gd_cuda():
    model = GDModel(NeuralNetwork, {"in_size": 3, "hidden_size": 4, "out_size": 1}, num_steps=2)
    prediction = model(torch.randn(2, 3, 3), torch.randn(2, 3))
    assert prediction.shape == (2, 3)
    assert prediction.device.type == "cuda"


def test_plot_api():
    import matplotlib
    matplotlib.use("Agg")
    import plot_utils
    fig, _ = plot_utils.basic_plot({"example": {"mean": [1, .5], "bootstrap_low": [.9, .4], "bootstrap_high": [1.1, .6]}})
    import matplotlib.pyplot as plt
    plt.close(fig)
