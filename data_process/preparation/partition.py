"""Seeded, ordered patient partition used to create the 4090pc split."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


class DataValidationError(ValueError):
    """Raised when input data violates an experiment invariant."""


def stratified_patient_split(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    seed: int,
    ratios: dict[str, float],
) -> dict[str, np.ndarray]:
    train_ratio = float(ratios["train"])
    val_ratio = float(ratios.get("validation", ratios.get("val", 0.0)))
    test_ratio = float(ratios["test"])
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise DataValidationError("Split ratios must sum to 1")
    indices = np.arange(len(patient_ids))
    train_idx, holdout_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    holdout_labels = labels[holdout_idx]
    val_fraction = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=val_fraction,
        random_state=seed,
        stratify=holdout_labels,
    )
    result = {
        "train": np.sort(train_idx),
        "validation": np.sort(val_idx),
        "test": np.sort(test_idx),
    }
    id_sets = [set(patient_ids[index]) for index in result.values()]
    if id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
        raise DataValidationError("Patient overlap detected between splits")
    return result
