"""Compact MLP student for soft planner-policy distillation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from ..states.action_space import ACTIONS, NUM_ACTIONS
from .action_head import ActionHeadOutput, StructuredActionHead


class MLPPolicy(nn.Module):
    """A feed-forward policy with the same structured head as the LLM student."""

    action_names = ACTIONS

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        *,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(num_actions) != NUM_ACTIONS:
            raise ValueError(f"This task has exactly {NUM_ACTIONS} actions")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        activations: dict[str, type[nn.Module]] = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_dims:
            width = int(width)
            if width <= 0:
                raise ValueError("All hidden dimensions must be positive")
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.LayerNorm(width),
                    activations[activation.lower()](),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = width
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.action_head = StructuredActionHead(previous)
        self.input_dim = int(input_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 1 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must end in dimension {self.input_dim}")
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw action logits; use :meth:`policy` to enforce a mask."""

        return self.action_head(self.encode(features))

    def policy(
        self,
        features: torch.Tensor,
        valid_action_mask: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> ActionHeadOutput:
        representation = self.encode(features)
        return self.action_head.policy(
            representation,
            valid_action_mask,
            temperature=temperature,
        )

    @torch.no_grad()
    def act(
        self,
        features: torch.Tensor,
        valid_action_mask: torch.Tensor,
        *,
        deterministic: bool = True,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        probabilities = self.policy(
            features,
            valid_action_mask,
            temperature=temperature,
        ).probabilities
        if deterministic:
            return probabilities.argmax(dim=-1)
        flat = probabilities.reshape(-1, NUM_ACTIONS)
        sampled = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
        return sampled.reshape(probabilities.shape[:-1])

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "MLPPolicy":
        """Create a policy from a YAML-style mapping."""

        if "input_dim" not in config:
            raise ValueError("MLP config requires input_dim")
        return cls(
            input_dim=int(config["input_dim"]),
            hidden_dims=tuple(int(value) for value in config.get("hidden_dims", (256, 128))),
            dropout=float(config.get("dropout", 0.1)),
            activation=str(config.get("activation", "gelu")),
            num_actions=int(config.get("num_actions", NUM_ACTIONS)),
        )


# Name retained from the earlier repository.
MaskAwareMLP = MLPPolicy


__all__ = ["MLPPolicy", "MaskAwareMLP"]
