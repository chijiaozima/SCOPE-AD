"""4090pc definitions for parsed baseline episodes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


LABELS: tuple[str, ...] = ("CN", "MCI", "AD")


LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


BASE_MODALITY = "demographic"


ACQUISITION_MODALITIES: tuple[str, ...] = (
    "cognitive",
    "apoe",
    "blood",
    "csf",
    "structural_mri",
    "amyloid_pet",
    "tau_pet",
)


ALL_MODALITIES: tuple[str, ...] = (BASE_MODALITY, *ACQUISITION_MODALITIES)


class SchemaError(ValueError):
    """Raised when episode data violates a non-negotiable schema invariant."""


def normalize_label(value: Any, pattern: str | None = None) -> int:
    """Return the ordered CN=0, MCI=1, AD=2 index."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise SchemaError("Diagnosis label is missing")
    text = str(value).strip().upper()
    if text in LABEL_TO_INDEX:
        return LABEL_TO_INDEX[text]
    if pattern:
        match = re.search(pattern, str(value), flags=re.IGNORECASE)
        if match and match.group(1).upper() in LABEL_TO_INDEX:
            return LABEL_TO_INDEX[match.group(1).upper()]
    raise SchemaError(f"Unrecognized ordered diagnosis label: {value!r}")


def validate_patient_ids(patient_ids: Sequence[Any]) -> np.ndarray:
    ids = np.asarray([str(value) for value in patient_ids], dtype=str)
    if ids.ndim != 1 or len(ids) == 0:
        raise SchemaError("patient_ids must be a non-empty one-dimensional sequence")
    if np.any(np.char.str_len(ids) == 0):
        raise SchemaError("patient_id cannot be empty")
    return ids


def assert_one_baseline_episode(patient_ids: Sequence[Any]) -> None:
    ids = validate_patient_ids(patient_ids)
    unique, counts = np.unique(ids, return_counts=True)
    duplicated = unique[counts > 1]
    if len(duplicated):
        raise SchemaError(
            "Expected one baseline episode per patient; duplicate patient_ids include "
            + ", ".join(duplicated[:5])
        )


def validate_availability_mask(mask: np.ndarray, patient_count: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    expected = (int(patient_count), len(ACQUISITION_MODALITIES))
    if result.shape != expected:
        raise SchemaError(f"available_mask must have shape {expected}; got {result.shape}")
    return result


@dataclass(frozen=True)
class ParsedEpisodes:
    """One baseline row per patient before numeric preprocessing."""

    patient_ids: np.ndarray
    labels: np.ndarray
    splits: np.ndarray
    modality_frames: Mapping[str, pd.DataFrame]
    available_mask: np.ndarray
    source_rows: int
    split_manifest_sha256: str = ""

    def __post_init__(self) -> None:
        ids = validate_patient_ids(self.patient_ids)
        assert_one_baseline_episode(ids)
        labels = np.asarray(self.labels, dtype=np.int64)
        splits = np.asarray(self.splits, dtype=str)
        if labels.shape != (len(ids),):
            raise SchemaError("labels must align one-to-one with patient_ids")
        if splits.shape != (len(ids),):
            raise SchemaError("splits must align one-to-one with patient_ids")
        if np.any((labels < 0) | (labels > 2)):
            raise SchemaError("labels must be ordered indices CN=0, MCI=1, AD=2")
        unknown_splits = set(splits) - {"train", "validation", "test"}
        if unknown_splits:
            raise SchemaError(f"Unknown split names: {sorted(unknown_splits)}")
        if tuple(self.modality_frames) != ALL_MODALITIES:
            raise SchemaError(
                f"modality_frames must be ordered exactly as {ALL_MODALITIES}; "
                f"got {tuple(self.modality_frames)}"
            )
        for name, frame in self.modality_frames.items():
            if len(frame) != len(ids):
                raise SchemaError(f"{name} features do not align with patient_ids")
        validate_availability_mask(self.available_mask, len(ids))

    def indices(self, split: str) -> np.ndarray:
        if split not in {"train", "validation", "test"}:
            raise SchemaError(f"Unknown split: {split}")
        return np.flatnonzero(self.splits == split)
