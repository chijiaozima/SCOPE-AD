from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


LABELS: tuple[str, ...] = ("CN", "MCI", "AD")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}

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


def validate_modality_order(modalities: Sequence[str]) -> None:
    actual = tuple(str(value) for value in modalities)
    if actual != ACQUISITION_MODALITIES:
        raise SchemaError(
            "Acquisition modalities must use the fixed order "
            f"{ACQUISITION_MODALITIES}; received {actual}"
        )


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


@dataclass(frozen=True)
class PreparedEpisodes:
    """Dense modality arrays used by the ordinal model."""

    patient_ids: np.ndarray
    labels: np.ndarray
    splits: np.ndarray
    features: Mapping[str, np.ndarray]
    available_mask: np.ndarray
    input_dims: Mapping[str, int]
    split_manifest_sha256: str = ""

    def __post_init__(self) -> None:
        ids = validate_patient_ids(self.patient_ids)
        assert_one_baseline_episode(ids)
        validate_availability_mask(self.available_mask, len(ids))
        if tuple(self.features) != ALL_MODALITIES:
            raise SchemaError(f"features must be ordered exactly as {ALL_MODALITIES}")
        for name, values in self.features.items():
            array = np.asarray(values)
            if array.ndim != 2 or array.shape[0] != len(ids):
                raise SchemaError(f"{name} must have shape [patients, features]")
            if int(self.input_dims.get(name, -1)) != array.shape[1]:
                raise SchemaError(f"input_dims[{name!r}] does not match feature array")

    def as_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "patient_ids": np.asarray(self.patient_ids),
            "labels": np.asarray(self.labels, dtype=np.int64),
            "splits": np.asarray(self.splits),
            "features": {key: np.asarray(value) for key, value in self.features.items()},
            "available_mask": np.asarray(self.available_mask, dtype=bool),
            "input_dims": {key: int(value) for key, value in self.input_dims.items()},
            "base_modality": BASE_MODALITY,
            "acquisition_modalities": list(ACQUISITION_MODALITIES),
            "split_manifest_sha256": self.split_manifest_sha256,
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "PreparedEpisodes":
        validate_modality_order(artifact.get("acquisition_modalities", ()))
        if artifact.get("base_modality") != BASE_MODALITY:
            raise SchemaError("Prepared artifact does not use demographic as the fixed base")
        return cls(
            patient_ids=np.asarray(artifact["patient_ids"], dtype=str),
            labels=np.asarray(artifact["labels"], dtype=np.int64),
            splits=np.asarray(artifact["splits"], dtype=str),
            features={name: np.asarray(artifact["features"][name]) for name in ALL_MODALITIES},
            available_mask=np.asarray(artifact["available_mask"], dtype=bool),
            input_dims={name: int(artifact["input_dims"][name]) for name in ALL_MODALITIES},
            split_manifest_sha256=str(artifact.get("split_manifest_sha256", "")),
        )
