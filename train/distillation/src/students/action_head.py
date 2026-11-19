"""Structured, mask-aware action head shared by the policy students."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..states.action_space import ACTIONS, NUM_ACTIONS, masked_softmax


def _broadcast_mask(valid_action_mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(valid_action_mask, dtype=torch.bool, device=logits.device)
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(0)
    try:
        mask = torch.broadcast_to(mask, logits.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"valid_action_mask with shape {tuple(mask.shape)} cannot be broadcast to "
            f"logits with shape {tuple(logits.shape)}"
        ) from exc
    if torch.any(~mask.any(dim=-1)):
        raise ValueError("Every sample must have at least one valid action")
    return mask


def masked_logits(logits: torch.Tensor, valid_action_mask: torch.Tensor) -> torch.Tensor:
    """Set invalid logits to ``-inf`` without changing valid logits."""

    if logits.ndim < 1 or logits.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"logits must end in {NUM_ACTIONS} actions, got {tuple(logits.shape)}")
    mask = _broadcast_mask(valid_action_mask, logits)
    return logits.masked_fill(~mask, -torch.inf)


def last_non_padding_hidden(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Select the last attended token for both left- and right-padded batches."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    mask = torch.as_tensor(attention_mask, device=hidden_states.device)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(hidden_states.shape[:2]):
        raise ValueError("attention_mask must have shape [batch, sequence]")
    mask = mask.to(dtype=torch.bool)
    if torch.any(~mask.any(dim=-1)):
        raise ValueError("Each sequence must contain at least one non-padding token")

    positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    last_indices = positions.unsqueeze(0).expand_as(mask).masked_fill(~mask, -1).amax(dim=-1)
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, last_indices]


# Backwards-compatible name used by the earlier prototype.
final_non_padding = last_non_padding_hidden


@dataclass(frozen=True)
class ActionHeadOutput:
    """Output of :meth:`StructuredActionHead.policy`."""

    logits: torch.Tensor
    masked_logits: torch.Tensor
    probabilities: torch.Tensor


class StructuredActionHead(nn.Module):
    """Map one state representation to the fixed ten-action vocabulary."""

    action_names = ACTIONS

    def __init__(self, hidden_size: int, *, bias: bool = True, num_actions: int = NUM_ACTIONS):
        super().__init__()
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        if int(num_actions) != NUM_ACTIONS:
            raise ValueError(f"This task has exactly {NUM_ACTIONS} actions in a fixed order")
        self.classifier = nn.Linear(int(hidden_size), NUM_ACTIONS, bias=bool(bias))

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        if representation.shape[-1] != self.classifier.in_features:
            raise ValueError(
                f"Expected representation size {self.classifier.in_features}, "
                f"got {representation.shape[-1]}"
            )
        return self.classifier(representation)

    def policy(
        self,
        representation: torch.Tensor,
        valid_action_mask: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> ActionHeadOutput:
        logits = self(representation)
        masked = masked_logits(logits, valid_action_mask)
        probabilities = masked_softmax(logits, valid_action_mask, temperature=temperature)
        return ActionHeadOutput(logits=logits, masked_logits=masked, probabilities=probabilities)


# Concise alias retained for scripts and checkpoints.
ActionHead = StructuredActionHead


__all__ = [
    "ActionHead",
    "ActionHeadOutput",
    "StructuredActionHead",
    "final_non_padding",
    "last_non_padding_hidden",
    "masked_logits",
]
