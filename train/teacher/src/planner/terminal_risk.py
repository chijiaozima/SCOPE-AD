from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


ORDINAL_DISTANCE = np.asarray([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])


def ordered_bayes_risk(
    posterior: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    classification_weight: float = 1.0,
    distance_weight: float = 0.5,
) -> np.ndarray | torch.Tensor:
    """Risk for DIAGNOSE_CN/MCI/AD under calibrated belief, never true labels."""
    if classification_weight < 0 or distance_weight < 0:
        raise ValueError("Risk weights must be non-negative")
    if isinstance(posterior, torch.Tensor):
        if posterior.shape[-1] != 3:
            raise ValueError("posterior last dimension must be 3")
        distance = torch.as_tensor(ORDINAL_DISTANCE, dtype=posterior.dtype, device=posterior.device)
        incorrect = (distance > 0).to(posterior.dtype)
        cost = classification_weight * incorrect + distance_weight * distance
        return posterior @ cost
    probabilities = np.asarray(posterior, dtype=float)
    if probabilities.shape[-1] != 3:
        raise ValueError("posterior last dimension must be 3")
    if np.any(probabilities < -1e-8) or not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("posterior must contain normalized probabilities")
    incorrect = (ORDINAL_DISTANCE > 0).astype(float)
    cost = classification_weight * incorrect + distance_weight * ORDINAL_DISTANCE
    return probabilities @ cost


def realized_ordinal_cost(
    true_label: int,
    diagnosis: int,
    *,
    classification_weight: float = 1.0,
    distance_weight: float = 0.5,
) -> float:
    """Privileged realized cost; use only in evaluation/oracle code."""
    if true_label not in (0, 1, 2) or diagnosis not in (0, 1, 2):
        raise ValueError("Labels must be 0, 1, or 2")
    return float(
        classification_weight * (true_label != diagnosis)
        + distance_weight * abs(true_label - diagnosis)
    )
