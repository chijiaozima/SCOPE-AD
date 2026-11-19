from __future__ import annotations

import torch
from torch import nn


def ordinal_nll(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape [batch, 3]")
    labels = targets.long().reshape(-1)
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("targets must align with probabilities")
    if bool(torch.any((labels < 0) | (labels > 2)).item()):
        raise ValueError("targets must be ordered indices CN=0, MCI=1, AD=2")
    selected = probabilities.gather(1, labels[:, None]).squeeze(1)
    losses = -selected.clamp_min(1e-7).log()
    if class_weights is None:
        return losses.mean()
    weights = torch.as_tensor(
        class_weights, dtype=probabilities.dtype, device=probabilities.device
    )
    if weights.shape != (3,) or bool(torch.any(weights <= 0).item()):
        raise ValueError("class_weights must contain three positive values")
    sample_weights = weights[labels]
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-7)


def ordinal_expected_absolute_error(
    probabilities: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    classes = torch.arange(3, dtype=probabilities.dtype, device=probabilities.device)
    distances = (classes[None, :] - targets.reshape(-1, 1).to(probabilities.dtype)).abs()
    return (probabilities * distances).sum(dim=1).mean()


def brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=3).to(probabilities.dtype)
    return ((probabilities - one_hot) ** 2).sum(dim=1).mean()


class OrdinalBeliefLoss(nn.Module):
    def __init__(
        self,
        expected_mae_weight: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.expected_mae_weight = float(expected_mae_weight)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.shape != (3,) or bool(torch.any(weights <= 0).item()):
                raise ValueError("class_weights must contain three positive values")
            self.register_buffer("class_weights", weights)

    def forward(
        self, output: dict[str, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        probabilities = output["probabilities"]
        loss = ordinal_nll(probabilities, targets, self.class_weights)
        if self.expected_mae_weight:
            loss = loss + self.expected_mae_weight * ordinal_expected_absolute_error(
                probabilities, targets
            )
        return loss
