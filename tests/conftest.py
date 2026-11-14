import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--require-cuda", action="store_true", help="Fail rather than skip GPU validation without CUDA")


def pytest_configure(config):
    if config.getoption("--require-cuda") and not torch.cuda.is_available():
        raise pytest.UsageError("GPU validation requires a real CUDA GPU; no CPU fallback")


def pytest_collection_modifyitems(items):
    if not torch.cuda.is_available():
        for item in items:
            if item.get_closest_marker("cuda"):
                item.add_marker(pytest.mark.skip(reason="CUDA GPU unavailable"))
