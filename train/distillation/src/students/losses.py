"""Losses for planner-policy distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from ..states.action_space import NUM_ACTIONS
from .action_head import masked_logits


Reduction = Literal["none", "mean", "sum"]


def _validated_inputs(
    student_logits: torch.Tensor,
    target_policy: torch.Tensor,
    valid_action_mask: torch.Tensor,
    *,
    strict_target_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if student_logits.ndim < 1 or student_logits.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"student_logits must end in {NUM_ACTIONS} actions")
    target = torch.as_tensor(target_policy, dtype=torch.float32, device=student_logits.device)
    mask = torch.as_tensor(valid_action_mask, dtype=torch.bool, device=student_logits.device)
    try:
        target = torch.broadcast_to(target, student_logits.shape)
        mask = torch.broadcast_to(mask, student_logits.shape)
    except RuntimeError as exc:
        raise ValueError("target_policy and valid_action_mask must match student_logits") from exc
    if not torch.isfinite(target).all() or torch.any(target < 0):
        raise ValueError("target_policy must contain finite, non-negative probabilities")
    if torch.any(~mask.any(dim=-1)):
        raise ValueError("Every sample must have at least one valid action")
    invalid_mass = target.masked_fill(mask, 0.0).sum(dim=-1)
    if strict_target_mask and torch.any(invalid_mass > 1e-7):
        raise ValueError("target_policy assigns probability to an invalid action")
    target = target.masked_fill(~mask, 0.0)
    normalizer = target.sum(dim=-1, keepdim=True)
    if torch.any(normalizer <= 0):
        raise ValueError("target_policy must have positive mass on at least one valid action")
    return student_logits, target / normalizer, mask


def _reduce(
    values: torch.Tensor,
    reduction: Reduction,
    sample_weight: torch.Tensor | None,
) -> torch.Tensor:
    if reduction == "none":
        if sample_weight is not None:
            weight = torch.as_tensor(sample_weight, dtype=values.dtype, device=values.device)
            values = values * torch.broadcast_to(weight, values.shape)
        return values
    if reduction == "sum":
        if sample_weight is not None:
            weight = torch.as_tensor(sample_weight, dtype=values.dtype, device=values.device)
            values = values * torch.broadcast_to(weight, values.shape)
        return values.sum()
    if reduction != "mean":
        raise ValueError("reduction must be one of: none, mean, sum")
    if sample_weight is None:
        return values.mean()
    weight = torch.as_tensor(sample_weight, dtype=values.dtype, device=values.device)
    weight = torch.broadcast_to(weight, values.shape)
    if torch.any(weight < 0) or not torch.isfinite(weight).all():
        raise ValueError("sample_weight must be finite and non-negative")
    return (values * weight).sum() / weight.sum().clamp_min(torch.finfo(values.dtype).eps)


def soft_kl_loss(
    student_logits: torch.Tensor,
    target_policy: torch.Tensor,
    valid_action_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    reduction: Reduction = "mean",
    sample_weight: torch.Tensor | None = None,
    strict_target_mask: bool = True,
) -> torch.Tensor:
    """Compute ``KL(target_policy || student_policy)`` over valid actions.

    The target is renormalized over valid actions. With the default strict mode,
    non-zero target mass on an invalid action is rejected instead of hidden.
    """

    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    logits, target, mask = _validated_inputs(
        student_logits,
        target_policy,
        valid_action_mask,
        strict_target_mask=strict_target_mask,
    )
    log_student = torch.log_softmax(masked_logits(logits / float(temperature), mask).float(), dim=-1)
    positive = target > 0
    log_target = torch.where(positive, torch.log(target.clamp_min(1e-12)), torch.zeros_like(target))
    per_sample = torch.where(positive, target * (log_target - log_student), 0.0).sum(dim=-1)
    # Roundoff can produce tiny negative values even though KL is non-negative.
    per_sample = per_sample.clamp_min(0.0)
    return _reduce(per_sample, reduction, sample_weight)


def soft_cross_entropy(
    student_logits: torch.Tensor,
    target_policy: torch.Tensor,
    valid_action_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    reduction: Reduction = "mean",
    sample_weight: torch.Tensor | None = None,
    strict_target_mask: bool = True,
) -> torch.Tensor:
    """Cross-entropy counterpart to :func:`soft_kl_loss`."""

    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    logits, target, mask = _validated_inputs(
        student_logits,
        target_policy,
        valid_action_mask,
        strict_target_mask=strict_target_mask,
    )
    log_student = torch.log_softmax(masked_logits(logits / float(temperature), mask).float(), dim=-1)
    safe_log_student = torch.where(target > 0, log_student, torch.zeros_like(log_student))
    per_sample = -(target * safe_log_student).sum(dim=-1)
    return _reduce(per_sample, reduction, sample_weight)


def hard_action_loss(
    student_logits: torch.Tensor,
    target_action: torch.Tensor,
    valid_action_mask: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Masked hard-action negative log likelihood."""

    logits = masked_logits(student_logits, valid_action_mask)
    actions = torch.as_tensor(target_action, dtype=torch.long, device=student_logits.device)
    expected_shape = student_logits.shape[:-1]
    if tuple(actions.shape) != tuple(expected_shape):
        raise ValueError(f"target_action must have shape {tuple(expected_shape)}")
    if torch.any((actions < 0) | (actions >= NUM_ACTIONS)):
        raise ValueError("target_action index is outside the fixed action vocabulary")
    mask = torch.broadcast_to(
        torch.as_tensor(valid_action_mask, dtype=torch.bool, device=student_logits.device),
        student_logits.shape,
    )
    selected_valid = torch.gather(mask, -1, actions.unsqueeze(-1)).squeeze(-1)
    if not bool(selected_valid.all()):
        raise ValueError("target_action contains an invalid action")
    loss = F.cross_entropy(
        logits.reshape(-1, NUM_ACTIONS).float(),
        actions.reshape(-1),
        reduction=reduction,
    )
    return loss.reshape(expected_shape) if reduction == "none" else loss


def energy_margin_preference_loss(
    student_logits: torch.Tensor,
    preferred_action: torch.Tensor,
    rejected_action: torch.Tensor,
    *,
    energy_margin: torch.Tensor | float = 0.0,
    valid_action_mask: torch.Tensor | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Weighted logistic ranking loss for planner energy preferences.

    A positive planner energy gap asks the preferred student logit to exceed the
    rejected logit by that gap. The detached gap also weights more certain pairs.
    """

    positive = torch.as_tensor(preferred_action, dtype=torch.long, device=student_logits.device)
    negative = torch.as_tensor(rejected_action, dtype=torch.long, device=student_logits.device)
    if positive.shape != student_logits.shape[:-1] or negative.shape != positive.shape:
        raise ValueError("preference action tensors must match the logits batch shape")
    if torch.any((positive < 0) | (positive >= NUM_ACTIONS) | (negative < 0) | (negative >= NUM_ACTIONS)):
        raise ValueError("preference action index is outside the fixed action vocabulary")
    if valid_action_mask is not None:
        mask = torch.broadcast_to(
            torch.as_tensor(valid_action_mask, dtype=torch.bool, device=student_logits.device),
            student_logits.shape,
        )
        if not bool(torch.gather(mask, -1, positive.unsqueeze(-1)).all()):
            raise ValueError("preferred_action contains an invalid action")
        if not bool(torch.gather(mask, -1, negative.unsqueeze(-1)).all()):
            raise ValueError("rejected_action contains an invalid action")
    preferred_logit = torch.gather(student_logits, -1, positive.unsqueeze(-1)).squeeze(-1)
    rejected_logit = torch.gather(student_logits, -1, negative.unsqueeze(-1)).squeeze(-1)
    margin = torch.as_tensor(energy_margin, dtype=student_logits.dtype, device=student_logits.device)
    margin = torch.broadcast_to(margin, positive.shape).clamp_min(0).detach()
    per_sample = F.softplus(-(preferred_logit - rejected_logit - margin))
    weights = torch.where(margin > 0, margin, torch.ones_like(margin))
    return _reduce(per_sample, reduction, weights)


@dataclass(frozen=True)
class DistillationLossOutput:
    loss: torch.Tensor
    soft_kl: torch.Tensor
    hard_nll: torch.Tensor


def distillation_loss(
    student_logits: torch.Tensor,
    target_policy: torch.Tensor,
    valid_action_mask: torch.Tensor,
    *,
    target_action: torch.Tensor | None = None,
    hard_weight: float = 0.0,
    temperature: float = 1.0,
) -> DistillationLossOutput:
    """Combine the primary soft KL with optional hard-action supervision."""

    if float(hard_weight) < 0:
        raise ValueError("hard_weight must be non-negative")
    soft = soft_kl_loss(
        student_logits,
        target_policy,
        valid_action_mask,
        temperature=temperature,
    )
    if target_action is None or float(hard_weight) == 0:
        hard = student_logits.sum() * 0.0
    else:
        hard = hard_action_loss(student_logits, target_action, valid_action_mask)
    return DistillationLossOutput(loss=soft + float(hard_weight) * hard, soft_kl=soft, hard_nll=hard)


# Earlier training scripts used this name.
target_kl = soft_kl_loss


__all__ = [
    "DistillationLossOutput",
    "distillation_loss",
    "energy_margin_preference_loss",
    "hard_action_loss",
    "soft_cross_entropy",
    "soft_kl_loss",
    "target_kl",
]
