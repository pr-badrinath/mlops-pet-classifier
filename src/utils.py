"""Shared helper utilities used across the pipeline."""
import random
import yaml
import numpy as np
import torch


def load_params(params_path: str = "params.yaml") -> dict:
    """Load the central YAML config used by every stage of the pipeline."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible across numpy / torch / python's random module."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
