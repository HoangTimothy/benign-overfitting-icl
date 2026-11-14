"""bo.pdf Phase II, Eqs. (7)-(10): fresh noisy linear classification tasks.

Labels are sampled once per prompt and retained for every evaluation protocol.
The two corruption mechanisms are mutually exclusive. sign(0) is defined as +1;
ties have probability zero under the continuous data-generating distribution.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def binary_labels(scores):
    return torch.where(scores >= 0, torch.ones_like(scores), -torch.ones_like(scores))


def classification_error(logits, labels):
    return (binary_labels(logits) != labels).float()


def classification_bce(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, (labels + 1) / 2)


@dataclass
class ClassificationLabels:
    observed: torch.Tensor
    clean: torch.Tensor
    corrupted: torch.Tensor
    margins: torch.Tensor


class Phase2Classification:
    def __init__(
        self, n_dims, batch_size, pool_dict=None, seeds=None,
        corruption="flip", eta=None, noise_std=None, bias=0.0,
    ):
        if n_dims < 1 or batch_size < 1:
            raise ValueError("n_dims and batch_size must be positive")
        if pool_dict is not None or seeds is not None:
            raise ValueError("Phase II requires fresh independent tasks; finite pools/seeds are disabled")
        if not math.isfinite(bias):
            raise ValueError("bias must be finite")
        self.n_dims, self.b_size = n_dims, batch_size
        self.corruption, self.bias = corruption, float(bias)
        self.eta, self.noise_std = None, None
        if corruption == "flip":
            if noise_std is not None:
                raise ValueError("Do not combine label flips and probit noise")
            self.eta = 0.1 if eta is None else float(eta)
            if not 0 < self.eta < 0.5:
                raise ValueError("flip eta must lie strictly between 0 and 0.5")
        elif corruption == "probit":
            if eta is not None:
                raise ValueError("Do not combine probit noise and label flips")
            self.noise_std = 1.0 if noise_std is None else float(noise_std)
            if not math.isfinite(self.noise_std) or self.noise_std <= 0:
                raise ValueError("probit noise_std must be finite and positive")
        else:
            raise ValueError("corruption must be 'flip' or 'probit'")
        # Eq. (7): do not normalize to the sphere; the prior is Gaussian I_d/d.
        self.w_b = torch.randn(batch_size, n_dims, 1) / math.sqrt(n_dims)

    @staticmethod
    def generate_pool_dict(*args, **kwargs):
        raise ValueError("Phase II requires fresh independent tasks, not a finite task pool")

    def sample_labels(self, xs):
        if xs.shape[0] != self.b_size or xs.shape[-1] != self.n_dims:
            raise ValueError("Feature batch/dimension does not match the task")
        margins = (xs @ self.w_b.to(xs.device))[:, :, 0] + self.bias
        clean = binary_labels(margins)
        if self.corruption == "flip":
            flipped = torch.rand_like(margins) < self.eta
            observed = torch.where(flipped, -clean, clean)
        else:
            observed = binary_labels(margins + self.noise_std * torch.randn_like(margins))
        return ClassificationLabels(observed, clean, observed != clean, margins)

    def evaluate(self, xs):
        return self.sample_labels(xs).observed

    @staticmethod
    def get_metric():
        return classification_error

    @staticmethod
    def get_training_metric():
        return classification_bce


def validate_phase2_config(args):
    """The Phase II protocol has fixed d,m and fresh tasks at every update."""
    if args.training.task != "phase2_classification":
        return
    if args.training.data != "gaussian" or set(args.training.data_kwargs or {}) - {"rho"}:
        raise ValueError("Phase II uses stationary standard Gaussian features; only rho is configurable")
    rho = (args.training.data_kwargs or {}).get("rho", 0.0)
    if not 0 <= rho < 1:
        raise ValueError("Phase II rho must lie in [0,1)")
    if args.training.num_tasks is not None or args.training.num_training_examples is not None:
        raise ValueError("Phase II requires fresh tasks and features at every training update")
    dims, points = args.training.curriculum.dims, args.training.curriculum.points
    if dims.start != args.model.n_dims or dims.end != args.model.n_dims:
        raise ValueError("Phase II must use the full, fixed input dimension d")
    if points.start != points.end or points.end < 2:
        raise ValueError("Phase II requires a fixed sequence length m+1 >= 2")
    if points.end > args.model.n_positions:
        raise ValueError("Model positions must accommodate m context examples plus one query")
    # Fail on invalid/noise-combining options before a long run starts, without
    # consuming RNG that would change the initialized model or data stream.
    state = torch.get_rng_state()
    try:
        Phase2Classification(args.model.n_dims, 1, **args.training.task_kwargs)
    finally:
        torch.set_rng_state(state)
