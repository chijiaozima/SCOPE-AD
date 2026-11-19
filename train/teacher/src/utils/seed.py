from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True, strict: bool = True) -> None:
    """Seed all RNGs and, by default, fail on non-deterministic Torch kernels."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required by deterministic CUDA matrix multiplication on CUDA >= 10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False


def seeded_generator(seed: int, device: str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))
