"""CUDA-only execution and reproducible, weights-only-safe RNG checkpoints."""

import os
import random
import sys

import numpy as np
import torch
import transformers

# Must be set before the first CUDA BLAS operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_device(requested="cuda"):
    if requested == "auto":
        requested = "cuda"  # compatibility with older configs; never fall back
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError("This project requires a CUDA GPU; CPU execution is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install the locked CUDA build and use an NVIDIA GPU")
    return device


def configure_reproducibility(seed, deterministic=True):
    configure_backend(deterministic)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_backend(deterministic=True):
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    # Preserve full FP32 arithmetic; do not silently enable TF32 or AMP.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def capture_rng_state():
    algorithm, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (algorithm, keys.tolist(), position, has_gauss, cached_gauss),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    cuda_states = state["torch_cuda"]
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_states) != count:
        raise RuntimeError("Exact resume requires the same visible CUDA device count")
    random.setstate(state["python"])
    algorithm, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def runtime_signature():
    """Fail rather than promise exact resume across incompatible execution stacks."""
    return {
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "python": sys.version,
        "platform": sys.platform,
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "sdpa_backends": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
        "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
