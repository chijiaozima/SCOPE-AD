from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from .action_space import MODALITIES


MODALITY_COUNT = len(MODALITIES)
LATTICE_SIZE = 1 << MODALITY_COUNT


def encode_mask(mask: Sequence[bool]) -> int:
    values = np.asarray(mask, dtype=bool)
    if values.shape != (MODALITY_COUNT,):
        raise ValueError(f"mask must have shape ({MODALITY_COUNT},)")
    return sum((1 << index) for index, value in enumerate(values) if value)


def decode_mask(code: int) -> np.ndarray:
    code = int(code)
    if code < 0 or code >= LATTICE_SIZE:
        raise ValueError(f"mask code must be in [0, {LATTICE_SIZE})")
    return np.asarray([bool(code & (1 << index)) for index in range(MODALITY_COUNT)])


def next_mask(mask: Sequence[bool], modality_index: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if result.shape != (MODALITY_COUNT,):
        raise ValueError(f"mask must have shape ({MODALITY_COUNT},)")
    modality_index = int(modality_index)
    if not 0 <= modality_index < MODALITY_COUNT:
        raise IndexError(modality_index)
    if result[modality_index]:
        raise ValueError("Cannot acquire an already observed modality")
    result[modality_index] = True
    return result


def iter_masks(
    available_mask: Sequence[bool] | None = None,
    *,
    descending: bool = False,
) -> Iterator[np.ndarray]:
    available = (
        np.ones(MODALITY_COUNT, dtype=bool)
        if available_mask is None
        else np.asarray(available_mask, dtype=bool)
    )
    if available.shape != (MODALITY_COUNT,):
        raise ValueError(f"available_mask must have shape ({MODALITY_COUNT},)")
    masks = [decode_mask(code) for code in range(LATTICE_SIZE)]
    masks = [mask for mask in masks if not np.any(mask & ~available)]
    masks.sort(key=lambda mask: (int(mask.sum()), encode_mask(mask)), reverse=descending)
    yield from masks


def levels(available_mask: Sequence[bool] | None = None, *, descending: bool = True) -> list[list[np.ndarray]]:
    grouped: dict[int, list[np.ndarray]] = {}
    for mask in iter_masks(available_mask):
        grouped.setdefault(int(mask.sum()), []).append(mask)
    return [grouped[count] for count in sorted(grouped, reverse=descending)]


def validate_lattice(masks: Iterable[Sequence[bool]], available_mask: Sequence[bool]) -> None:
    available = np.asarray(available_mask, dtype=bool)
    codes = {encode_mask(mask) for mask in masks}
    expected = {encode_mask(mask) for mask in iter_masks(available)}
    if codes != expected:
        missing, extra = sorted(expected - codes), sorted(codes - expected)
        raise ValueError(f"Incomplete subset lattice: missing={missing}, extra={extra}")
