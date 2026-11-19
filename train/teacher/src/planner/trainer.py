from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .ebm import EnergyNetwork
from .soft_value import energy_policy


def planner_loss(
    predicted_energies: torch.Tensor,
    target_energies: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float,
    ranking_kl_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = valid_mask.bool() & torch.isfinite(target_energies)
    if torch.any(~valid.any(dim=-1)):
        raise ValueError("Every sample needs at least one finite valid target")
    td = F.huber_loss(predicted_energies[valid], target_energies[valid])
    target_policy = energy_policy(target_energies, valid, temperature=temperature)
    predicted_policy = energy_policy(predicted_energies, valid, temperature=temperature)
    kl = torch.sum(
        target_policy
        * (torch.log(target_policy.clamp_min(1e-12)) - torch.log(predicted_policy.clamp_min(1e-12))),
        dim=-1,
    ).mean()
    total = td + float(ranking_kl_weight) * kl
    return total, {"loss": total.detach(), "td": td.detach(), "ranking_kl": kl.detach()}


def polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1]")
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters(), strict=True):
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


@dataclass
class FitResult:
    model: EnergyNetwork
    target_model: EnergyNetwork
    history: list[dict[str, float]]
    best_epoch: int


class PlannerTrainer:
    def __init__(
        self,
        model: EnergyNetwork,
        *,
        temperature: float = 0.3,
        ranking_kl_weight: float = 1.0,
        target_update_tau: float = 0.01,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.target_model = copy.deepcopy(model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.temperature = float(temperature)
        self.ranking_kl_weight = float(ranking_kl_weight)
        self.target_update_tau = float(target_update_tau)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )

    def fit(
        self,
        train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        validation_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        epochs: int = 100,
        batch_size: int = 512,
        patience: int = 15,
        generator: torch.Generator | None = None,
    ) -> FitResult:
        dataset = TensorDataset(*train_tensors)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        best_state = copy.deepcopy(self.model.state_dict())
        best_epoch, best_validation, stale = 0, float("inf"), 0
        history: list[dict[str, float]] = []
        for epoch in range(1, int(epochs) + 1):
            self.model.train()
            totals = {"loss": 0.0, "td": 0.0, "ranking_kl": 0.0}
            seen = 0
            for features, targets, mask in loader:
                features, targets, mask = features.to(self.device), targets.to(self.device), mask.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss, components = planner_loss(
                    self.model(features), targets, mask,
                    temperature=self.temperature,
                    ranking_kl_weight=self.ranking_kl_weight,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                polyak_update(self.target_model, self.model, self.target_update_tau)
                batch_count = len(features)
                seen += batch_count
                for key in totals:
                    totals[key] += float(components[key]) * batch_count
            row = {"epoch": float(epoch), **{key: value / max(seen, 1) for key, value in totals.items()}}
            validation_loss = self.evaluate(validation_tensors) if validation_tensors is not None else row["loss"]
            row["validation_loss"] = validation_loss
            history.append(row)
            if validation_loss < best_validation - 1e-8:
                best_validation, best_epoch, stale = validation_loss, epoch, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if stale >= int(patience):
                    break
        self.model.load_state_dict(best_state)
        return FitResult(self.model, self.target_model, history, best_epoch)

    @torch.inference_mode()
    def evaluate(self, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        self.model.eval()
        features, targets, mask = (tensor.to(self.device) for tensor in tensors)
        loss, _ = planner_loss(
            self.model(features), targets, mask,
            temperature=self.temperature,
            ranking_kl_weight=self.ranking_kl_weight,
        )
        return float(loss)
