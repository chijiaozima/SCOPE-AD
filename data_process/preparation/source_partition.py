"""Reconstruct the three historical input shards before the availability filter."""

from __future__ import annotations

import math
import random
from collections import defaultdict


VALID_LABELS = ("CN", "MCI", "AD")


def parse_mapped_label(diagnose_text: str) -> str | None:
    for label in (*VALID_LABELS, "unmapped", "not available"):
        if f"mapped_label: {label}" in diagnose_text:
            return label
    return None


def split_by_label(
    rows: list[dict[str, str]],
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> dict[str, list[dict[str, str]]]:
    if not math.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = parse_mapped_label(row["diagnose"])
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label in filtered row_id={row.get('row_id')}: {label}")
        by_label[label].append(row)

    splits = {"train": [], "val": [], "test": []}
    for label in VALID_LABELS:
        label_rows = sorted(
            by_label[label],
            key=lambda row: (row["subject_id"], int(row["row_id"])),
        )
        rng.shuffle(label_rows)

        n_rows = len(label_rows)
        n_train = math.floor(n_rows * train_frac)
        n_val = math.floor(n_rows * val_frac)

        splits["train"].extend(label_rows[:n_train])
        splits["val"].extend(label_rows[n_train : n_train + n_val])
        splits["test"].extend(label_rows[n_train + n_val :])

    for split_rows in splits.values():
        split_rows.sort(key=lambda row: (row["subject_id"], int(row["row_id"])))
    return splits
