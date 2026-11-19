from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import torch

from .schema import ACQUISITION_MODALITIES, SchemaError, validate_modality_order


MODALITY_COUNT = len(ACQUISITION_MODALITIES)


def validate_observed_mask(
    observed_mask: np.ndarray | torch.Tensor,
    available_mask: np.ndarray | torch.Tensor | None = None,
) -> None:
    if observed_mask.ndim != 2 or observed_mask.shape[1] != MODALITY_COUNT:
        raise SchemaError(
            f"observed_mask must have shape [batch, {MODALITY_COUNT}] in the fixed modality order"
        )
    if available_mask is not None:
        if available_mask.shape != observed_mask.shape:
            raise SchemaError("available_mask must have the same shape as observed_mask")
        if isinstance(observed_mask, torch.Tensor):
            invalid = observed_mask.bool() & ~available_mask.bool()
            has_invalid = bool(torch.any(invalid).item())
        else:
            invalid = np.asarray(observed_mask, dtype=bool) & ~np.asarray(
                available_mask, dtype=bool
            )
            has_invalid = bool(np.any(invalid))
        if has_invalid:
            raise SchemaError("observed_mask cannot include an unavailable modality")


def mask_to_modalities(mask: Sequence[bool]) -> tuple[str, ...]:
    values = np.asarray(mask, dtype=bool)
    if values.shape != (MODALITY_COUNT,):
        raise SchemaError(f"mask must have shape ({MODALITY_COUNT},)")
    return tuple(name for name, selected in zip(ACQUISITION_MODALITIES, values) if selected)


def modalities_to_mask(modalities: Sequence[str]) -> np.ndarray:
    unknown = set(modalities) - set(ACQUISITION_MODALITIES)
    if unknown:
        raise SchemaError(f"Unknown acquisition modalities: {sorted(unknown)}")
    selected = set(modalities)
    return np.asarray([name in selected for name in ACQUISITION_MODALITIES], dtype=bool)


def enumerate_valid_masks(
    available_mask: Sequence[bool],
    *,
    max_states: int = 128,
) -> np.ndarray:
    """Enumerate base-only through full-available states for one patient."""

    available = np.asarray(available_mask, dtype=bool)
    if available.shape != (MODALITY_COUNT,):
        raise SchemaError(f"available_mask must have shape ({MODALITY_COUNT},)")
    available_indices = np.flatnonzero(available).tolist()
    masks: list[np.ndarray] = []
    for count in range(len(available_indices) + 1):
        for selected in itertools.combinations(available_indices, count):
            mask = np.zeros(MODALITY_COUNT, dtype=bool)
            mask[list(selected)] = True
            masks.append(mask)
    if max_states <= 0:
        raise SchemaError("max_states must be positive")
    if len(masks) > max_states:
        # Deterministic, cardinality-spread subsampling while preserving endpoints.
        positions = np.linspace(0, len(masks) - 1, num=max_states, dtype=int)
        masks = [masks[position] for position in np.unique(positions)]
        if not np.array_equal(masks[0], np.zeros(MODALITY_COUNT, dtype=bool)):
            masks[0] = np.zeros(MODALITY_COUNT, dtype=bool)
        masks[-1] = available.copy()
    result = np.stack(masks).astype(bool)
    if np.any(result & ~available[None, :]):
        raise AssertionError("Generated a state containing an unavailable modality")
    return result


def expand_patient_partial_states(
    patient_indices: Sequence[int],
    available_mask: np.ndarray,
    *,
    max_states_per_patient: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic base/full/intermediate partial states for patients."""

    indices = np.asarray(patient_indices, dtype=np.int64)
    available = np.asarray(available_mask, dtype=bool)
    if available.ndim != 2 or available.shape[1] != MODALITY_COUNT:
        raise SchemaError(f"available_mask must have shape [patients, {MODALITY_COUNT}]")
    if np.any(indices < 0) or np.any(indices >= len(available)):
        raise SchemaError("patient_indices contains an out-of-range row")
    expanded_indices: list[int] = []
    observed_masks: list[np.ndarray] = []
    state_ids: list[int] = []
    for patient_index in indices:
        masks = enumerate_valid_masks(
            available[int(patient_index)], max_states=int(max_states_per_patient)
        )
        expanded_indices.extend([int(patient_index)] * len(masks))
        observed_masks.extend(masks)
        state_ids.extend(range(len(masks)))
    return (
        np.asarray(expanded_indices, dtype=np.int64),
        np.asarray(observed_masks, dtype=bool),
        np.asarray(state_ids, dtype=np.int64),
    )


def sample_random_masks(
    available_mask: np.ndarray,
    *,
    seed: int,
    base_only_probability: float = 0.15,
) -> np.ndarray:
    """Sample one random partial-state mask per row; demographic stays implicit."""

    available = np.asarray(available_mask, dtype=bool)
    if available.ndim != 2 or available.shape[1] != MODALITY_COUNT:
        raise SchemaError(f"available_mask must have shape [batch, {MODALITY_COUNT}]")
    if not 0.0 <= base_only_probability <= 1.0:
        raise SchemaError("base_only_probability must be in [0, 1]")
    rng = np.random.default_rng(int(seed))
    masks = available & (rng.random(available.shape) < 0.5)
    base_only_rows = rng.random(len(available)) < float(base_only_probability)
    masks[base_only_rows] = False
    return masks


def sample_random_masks_torch(
    available_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    base_only_probability: float = 0.15,
) -> torch.Tensor:
    available = available_mask.bool()
    if available.ndim != 2 or available.shape[1] != MODALITY_COUNT:
        raise SchemaError(f"available_mask must have shape [batch, {MODALITY_COUNT}]")
    draw = torch.rand(
        available.shape,
        generator=generator,
        device=available.device,
    )
    result = available & (draw < 0.5)
    base_draw = torch.rand(
        (available.shape[0],),
        generator=generator,
        device=available.device,
    )
    result[base_draw < float(base_only_probability)] = False
    return result


def assert_fixed_modality_order(modalities: Sequence[str]) -> None:
    validate_modality_order(modalities)
