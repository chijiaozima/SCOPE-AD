from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .schema import SchemaError, validate_patient_ids


@dataclass(frozen=True)
class CrossfitFold:
    fold: int
    fit_indices: np.ndarray
    holdout_indices: np.ndarray
    fit_patient_ids: tuple[str, ...]
    holdout_patient_ids: tuple[str, ...]

    def as_manifest(self) -> dict[str, object]:
        return {
            "fold": int(self.fold),
            "fit_patient_ids": list(self.fit_patient_ids),
            "holdout_patient_ids": list(self.holdout_patient_ids),
        }


def _patient_labels(
    patient_ids: Sequence[object], labels: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    ids = validate_patient_ids(patient_ids)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (len(ids),):
        raise SchemaError("labels must align with patient_ids")
    unique_ids = np.unique(ids)
    unique_labels: list[int] = []
    row_indices: dict[str, np.ndarray] = {}
    for patient_id in unique_ids:
        rows = np.flatnonzero(ids == patient_id)
        patient_y = np.unique(y[rows])
        if len(patient_y) != 1:
            raise SchemaError(f"Patient {patient_id} has inconsistent labels")
        row_indices[str(patient_id)] = rows
        unique_labels.append(int(patient_y[0]))
    return unique_ids.astype(str), np.asarray(unique_labels, dtype=np.int64), row_indices


def assert_disjoint_patient_sets(named_ids: Mapping[str, Iterable[object]]) -> None:
    normalized = {
        name: {str(value) for value in values}
        for name, values in named_ids.items()
    }
    names = list(normalized)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = normalized[left] & normalized[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise SchemaError(
                    f"Patient overlap between {left} and {right}; examples: {examples}"
                )


def assert_split_integrity(patient_ids: Sequence[object], splits: Sequence[str]) -> None:
    ids = validate_patient_ids(patient_ids)
    split_values = np.asarray(splits, dtype=str)
    if split_values.shape != (len(ids),):
        raise SchemaError("splits must align with patient_ids")
    unknown = set(split_values) - {"train", "validation", "test"}
    if unknown:
        raise SchemaError(f"Unknown split names: {sorted(unknown)}")
    named = {
        split: ids[split_values == split]
        for split in ("train", "validation", "test")
    }
    assert_disjoint_patient_sets(named)


def stratified_patient_split(
    patient_ids: Sequence[object],
    labels: Sequence[int],
    *,
    seed: int,
    ratios: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Split rows by patient and return disjoint train/validation/test indices."""

    unique_ids, unique_y, row_indices = _patient_labels(patient_ids, labels)
    train_ratio = float(ratios["train"])
    validation_ratio = float(ratios.get("validation", ratios.get("val", 0.0)))
    test_ratio = float(ratios["test"])
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise SchemaError("train, validation, and test ratios must all be positive")
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise SchemaError("Split ratios must sum to 1")
    patient_positions = np.arange(len(unique_ids))
    train_positions, holdout_positions = train_test_split(
        patient_positions,
        train_size=train_ratio,
        random_state=int(seed),
        stratify=unique_y,
    )
    validation_fraction = validation_ratio / (validation_ratio + test_ratio)
    validation_positions, test_positions = train_test_split(
        holdout_positions,
        train_size=validation_fraction,
        random_state=int(seed),
        stratify=unique_y[holdout_positions],
    )

    def rows_for(positions: np.ndarray) -> np.ndarray:
        selected = [row_indices[str(unique_ids[position])] for position in positions]
        return np.sort(np.concatenate(selected)).astype(np.int64)

    result = {
        "train": rows_for(train_positions),
        "validation": rows_for(validation_positions),
        "test": rows_for(test_positions),
    }
    ids = np.asarray(patient_ids, dtype=str)
    assert_disjoint_patient_sets({name: ids[index] for name, index in result.items()})
    return result


def patient_stratified_kfold(
    patient_ids: Sequence[object],
    labels: Sequence[int],
    *,
    n_splits: int = 5,
    seed: int = 42,
    shuffle: bool = True,
) -> list[CrossfitFold]:
    """Create patient-disjoint folds; repeated state rows stay with their patient."""

    unique_ids, unique_y, row_indices = _patient_labels(patient_ids, labels)
    if int(n_splits) != 5:
        raise SchemaError("Ordinal belief cross-fitting is fixed to 5 folds")
    class_counts = np.bincount(unique_y, minlength=3)
    if np.any(class_counts < n_splits):
        raise SchemaError(
            "Five-fold stratification requires at least five patients per class; "
            f"found {class_counts.tolist()}"
        )
    splitter = StratifiedKFold(
        n_splits=int(n_splits),
        shuffle=bool(shuffle),
        random_state=int(seed) if shuffle else None,
    )
    folds: list[CrossfitFold] = []
    holdout_seen: set[str] = set()
    for fold_id, (fit_positions, holdout_positions) in enumerate(
        splitter.split(unique_ids, unique_y)
    ):
        fit_ids = tuple(str(unique_ids[index]) for index in fit_positions)
        holdout_ids = tuple(str(unique_ids[index]) for index in holdout_positions)
        assert_disjoint_patient_sets({"fit": fit_ids, "holdout": holdout_ids})
        overlap = holdout_seen & set(holdout_ids)
        if overlap:
            raise SchemaError(f"Patients assigned to multiple OOF folds: {sorted(overlap)[:5]}")
        holdout_seen.update(holdout_ids)
        fit_rows = np.sort(
            np.concatenate([row_indices[patient_id] for patient_id in fit_ids])
        ).astype(np.int64)
        holdout_rows = np.sort(
            np.concatenate([row_indices[patient_id] for patient_id in holdout_ids])
        ).astype(np.int64)
        folds.append(
            CrossfitFold(
                fold=fold_id,
                fit_indices=fit_rows,
                holdout_indices=holdout_rows,
                fit_patient_ids=fit_ids,
                holdout_patient_ids=holdout_ids,
            )
        )
    if holdout_seen != set(unique_ids):
        missing = sorted(set(unique_ids) - holdout_seen)
        raise SchemaError(f"OOF folds do not cover every patient: {missing[:5]}")
    return folds


def inner_validation_split(
    indices: Sequence[int],
    labels: Sequence[int],
    *,
    seed: int,
    validation_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an early-stopping split without using the outer OOF holdout."""

    selected = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    fit, validation = train_test_split(
        selected,
        test_size=float(validation_fraction),
        random_state=int(seed),
        stratify=y[selected],
    )
    return np.sort(fit), np.sort(validation)
