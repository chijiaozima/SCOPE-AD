"""Check patient independence across partitions."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .schema import SchemaError, validate_patient_ids


def assert_disjoint_patient_sets(named_ids: Mapping[str, Iterable[object]]) -> None:
    normalized = {name: {str(value) for value in values} for name, values in named_ids.items()}
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
    named = {split: ids[split_values == split] for split in ("train", "validation", "test")}
    assert_disjoint_patient_sets(named)
