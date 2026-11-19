from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np
import torch


MODALITIES = (
    "cognitive",
    "apoe",
    "blood",
    "csf",
    "structural_mri",
    "amyloid_pet",
    "tau_pet",
)
ACQUISITION_ACTIONS = tuple(f"REQUEST_{name.upper()}" for name in MODALITIES)
DIAGNOSIS_ACTIONS = ("DIAGNOSE_CN", "DIAGNOSE_MCI", "DIAGNOSE_AD")
ACTIONS = ACQUISITION_ACTIONS + DIAGNOSIS_ACTIONS
NUM_ACTIONS = len(ACTIONS)
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}
INDEX_TO_ACTION = dict(enumerate(ACTIONS))


def _validated_torch_mask(values: torch.Tensor, *, name: str) -> torch.Tensor:
    if values.layout != torch.strided or values.device.type == "meta":
        raise ValueError(f"{name} must be a materialized dense tensor")
    if values.dtype != torch.bool:
        try:
            torch.iinfo(values.dtype)
            numeric = True
        except TypeError:
            numeric = bool(values.is_floating_point()) and not bool(values.is_complex())
        if not numeric:
            raise ValueError(f"{name} must contain only booleans or numeric zero/one values")
        try:
            finite = bool(torch.isfinite(values).all().item())
            binary = bool(((values == 0) | (values == 1)).all().item())
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"{name} could not be validated as a binary mask") from exc
        if not finite or not binary:
            raise ValueError(
                f"{name} must contain only finite booleans or exact zero/one values"
            )
    return values.to(dtype=torch.bool)


def _validated_numpy_mask(values: object, *, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        tensor = _validated_torch_mask(values, name=name)
        return tensor.detach().to(device="cpu").numpy()
    try:
        result = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an array-like binary mask") from exc
    if (
        result.dtype.kind not in "bifu"
        or not np.isfinite(result).all()
        or not np.isin(result, [0, 1]).all()
    ):
        raise ValueError(
            f"{name} must contain only finite booleans or exact zero/one values"
        )
    return result.astype(bool, copy=False)


def _bool_mask(values: Sequence[bool], *, name: str) -> np.ndarray:
    result = _validated_numpy_mask(values, name=name)
    if result.shape != (len(MODALITIES),):
        raise ValueError(f"{name} must have shape ({len(MODALITIES)},)")
    return result


def _finite_non_negative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite non-negative numeric scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _strict_bounded_integer(
    value: object, *, name: str, minimum: int, maximum: int
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return result


def valid_action_mask(
    observed_mask: Sequence[bool],
    available_mask: Sequence[bool],
    *,
    cumulative_cost: float = 0.0,
    budget: float | None = None,
    raw_costs: Mapping[str, float] | None = None,
    acquisitions_used: int | None = None,
    max_acquisitions: int = 7,
) -> np.ndarray:
    """Return the hard controller mask; diagnosis actions are always valid."""
    observed = _bool_mask(observed_mask, name="observed_mask")
    available = _bool_mask(available_mask, name="available_mask")
    if np.any(observed & ~available):
        raise ValueError("observed_mask must be a subset of available_mask")
    cumulative = _finite_non_negative(cumulative_cost, name="cumulative_cost")
    maximum = _strict_bounded_integer(
        max_acquisitions,
        name="max_acquisitions",
        minimum=0,
        maximum=len(MODALITIES),
    )
    observed_count = int(observed.sum())
    if acquisitions_used is None:
        used = observed_count
    else:
        used = _strict_bounded_integer(
            acquisitions_used,
            name="acquisitions_used",
            minimum=0,
            maximum=len(MODALITIES),
        )
        if used != observed_count:
            raise ValueError("acquisitions_used must equal observed_mask.sum()")
    if observed_count > maximum:
        raise ValueError("observed acquisitions exceed max_acquisitions")

    costs: dict[str, float] | None = None
    if raw_costs is not None:
        if not isinstance(raw_costs, Mapping) or set(raw_costs) != set(MODALITIES):
            raise ValueError(
                "raw_costs must contain exactly the seven acquisition modalities"
            )
        costs = {
            modality: _finite_non_negative(
                raw_costs[modality], name=f"raw_costs[{modality}]"
            )
            for modality in MODALITIES
        }

    acquisitions = available & ~observed
    if used >= maximum:
        acquisitions[:] = False
    if budget is not None:
        episode_budget = _finite_non_negative(budget, name="budget")
        if episode_budget + 1e-12 < cumulative:
            raise ValueError("budget cannot be less than cumulative_cost")
        if costs is None:
            raise ValueError("raw_costs are required when budget is set")
        remaining = episode_budget - cumulative
        for index, modality in enumerate(MODALITIES):
            if acquisitions[index] and costs[modality] > remaining + 1e-12:
                acquisitions[index] = False
    result = np.ones(NUM_ACTIONS, dtype=bool)
    result[: len(MODALITIES)] = acquisitions
    return result


def validate_action(action: str | int, valid_mask: Sequence[bool]) -> int:
    if isinstance(action, str):
        if action not in ACTION_TO_INDEX:
            raise ValueError(f"Unknown action: {action}")
        index = ACTION_TO_INDEX[action]
    elif isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral):
        raise ValueError("action must be a known name or a non-boolean integer index")
    else:
        index = int(action)
    mask = _validated_numpy_mask(valid_mask, name="valid_mask")
    if mask.shape != (NUM_ACTIONS,):
        raise ValueError(f"valid_mask must have shape ({NUM_ACTIONS},)")
    if index < 0 or index >= NUM_ACTIONS or not mask[index]:
        raise ValueError(f"Invalid action: {action}")
    return index


def masked_softmax(
    logits: torch.Tensor,
    valid_mask: torch.Tensor | np.ndarray | Sequence[bool],
    temperature: float = 1.0,
) -> torch.Tensor:
    if isinstance(temperature, (bool, np.bool_)) or not isinstance(
        temperature, (int, float, np.integer, np.floating)
    ):
        raise ValueError("temperature must be a finite positive numeric scalar")
    temperature_value = float(temperature)
    if not np.isfinite(temperature_value) or temperature_value <= 0:
        raise ValueError("temperature must be finite and positive")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.layout != torch.strided
        or logits.device.type == "meta"
        or not logits.is_floating_point()
        or logits.is_complex()
        or logits.ndim < 1
        or logits.shape[-1] != NUM_ACTIONS
    ):
        raise ValueError("logits must be a dense floating tensor ending in ten actions")
    if isinstance(valid_mask, torch.Tensor):
        mask = _validated_torch_mask(valid_mask, name="valid_mask").to(
            device=logits.device
        )
    else:
        binary_mask = _validated_numpy_mask(valid_mask, name="valid_mask")
        mask = torch.as_tensor(binary_mask, dtype=torch.bool, device=logits.device)
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(0)
    try:
        mask = torch.broadcast_to(mask, logits.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"valid_mask with shape {tuple(mask.shape)} cannot broadcast to "
            f"logits shape {tuple(logits.shape)}"
        ) from exc
    if torch.isnan(logits).any().item():
        raise ValueError("logits cannot contain NaN")
    if torch.any(mask & ~torch.isfinite(logits)).item():
        raise ValueError("valid-action logits must be finite")
    if torch.any(~mask.any(dim=-1)):
        raise ValueError("Every row must contain at least one valid action")
    masked = (logits / temperature_value).masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(masked, dim=-1)
    return probabilities.masked_fill(~mask, 0.0)
