from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch

from src.states.action_space import masked_softmax


BellmanMode = Literal["soft", "hard"]


def _validated_mask(
    energies: torch.Tensor,
    valid_mask: torch.Tensor | np.ndarray | Sequence[bool],
) -> torch.Tensor:
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=energies.device)
    while mask.ndim < energies.ndim:
        mask = mask.unsqueeze(0)
    try:
        mask = torch.broadcast_to(mask, energies.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"valid_mask with shape {tuple(mask.shape)} cannot broadcast to "
            f"energies with shape {tuple(energies.shape)}"
        ) from exc
    if torch.any(~mask.any(dim=-1)):
        raise ValueError("Each state requires at least one valid action")
    return mask


def hard_value(
    energies: torch.Tensor,
    valid_mask: torch.Tensor | np.ndarray | Sequence[bool],
) -> torch.Tensor:
    """Masked hard Bellman value ``min_a E(s, a)``."""

    mask = _validated_mask(energies, valid_mask)
    return energies.masked_fill(~mask, torch.inf).amin(dim=-1)


def soft_value(
    energies: torch.Tensor,
    valid_mask: torch.Tensor | np.ndarray | Sequence[bool],
    *,
    temperature: float,
    bellman_mode: BellmanMode = "soft",
) -> torch.Tensor:
    """Masked Bellman value, using soft-min by default or an explicit hard min."""

    if bellman_mode not in {"soft", "hard"}:
        raise ValueError("bellman_mode must be 'soft' or 'hard'")
    if bellman_mode == "hard":
        return hard_value(energies, valid_mask)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    mask = _validated_mask(energies, valid_mask)
    scaled = (-energies / float(temperature)).masked_fill(~mask, -torch.inf)
    return -float(temperature) * torch.logsumexp(scaled, dim=-1)


def soft_value_numpy(
    energies: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray,
    *,
    temperature: float,
    bellman_mode: BellmanMode = "soft",
) -> float:
    tensor = torch.as_tensor(np.asarray(energies), dtype=torch.float64)
    value = soft_value(
        tensor,
        valid_mask,
        temperature=temperature,
        bellman_mode=bellman_mode,
    )
    return float(value.item())


def hard_value_numpy(
    energies: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray,
) -> float:
    tensor = torch.as_tensor(np.asarray(energies), dtype=torch.float64)
    return float(hard_value(tensor, valid_mask).item())


def bellman_value(
    energies: torch.Tensor,
    valid_mask: torch.Tensor | np.ndarray | Sequence[bool],
    *,
    temperature: float,
    bellman_mode: BellmanMode = "soft",
) -> torch.Tensor:
    """Explicitly named entry point for soft-versus-hard Bellman ablations."""

    return soft_value(
        energies,
        valid_mask,
        temperature=temperature,
        bellman_mode=bellman_mode,
    )


def energy_policy(
    energies: torch.Tensor,
    valid_mask: torch.Tensor | np.ndarray | Sequence[bool],
    *,
    temperature: float,
) -> torch.Tensor:
    return masked_softmax(-energies, valid_mask, temperature=temperature)
