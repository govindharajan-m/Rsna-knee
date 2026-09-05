"""I provide explicit random-seed and deterministic-runtime controls."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

from .logging import get_logger

logger = get_logger("reproducibility")


@dataclass(frozen=True)
class SeedState:
    """I record which optional random-number generators I configured."""

    seed: int
    deterministic_requested: bool
    torch_configured: bool


def set_global_seed(seed: int, *, deterministic: bool = True) -> SeedState:
    """I seed Python, NumPy, and PyTorch when it is installed.

    I set ``PYTHONHASHSEED`` for child processes. Python applies that setting at
    interpreter startup, so callers should also set it before launching Python
    when hash-order reproducibility is required in the current process.
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        logger.info("PyTorch is unavailable; I seeded Python and NumPy only.")
        return SeedState(seed=seed, deterministic_requested=deterministic, torch_configured=False)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    return SeedState(seed=seed, deterministic_requested=deterministic, torch_configured=True)
