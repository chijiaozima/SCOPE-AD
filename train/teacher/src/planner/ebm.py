from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from src.states.action_space import NUM_ACTIONS
from .soft_value import energy_policy


class EnergyNetwork(nn.Module):
    """State-conditioned 10-action energy model."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        layers: list[nn.Module] = []
        previous = int(state_dim)
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, int(width)),
                    nn.LayerNorm(int(width)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = int(width)
        layers.append(nn.Linear(previous, NUM_ACTIONS))
        self.network = nn.Sequential(*layers)
        self.state_dim = int(state_dim)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        if state_features.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {state_features.shape[-1]}")
        return self.network(state_features)

    def policy(
        self,
        state_features: torch.Tensor,
        valid_mask: torch.Tensor | np.ndarray,
        *,
        temperature: float,
    ) -> torch.Tensor:
        return energy_policy(self(state_features), valid_mask, temperature=temperature)


@torch.inference_mode()
def choose_action(
    model: EnergyNetwork,
    state_features: torch.Tensor,
    valid_mask: torch.Tensor | np.ndarray,
    *,
    temperature: float,
    stochastic: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    model.eval()
    energies = model(state_features)
    probabilities = energy_policy(energies, valid_mask, temperature=temperature)
    if stochastic:
        index = int(torch.multinomial(probabilities.reshape(-1), 1, generator=generator).item())
    else:
        index = int(probabilities.reshape(-1).argmax().item())
    return index, energies, probabilities
