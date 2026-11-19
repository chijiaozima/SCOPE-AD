from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from src.data.masks import validate_observed_mask
from src.data.schema import ACQUISITION_MODALITIES, ALL_MODALITIES, BASE_MODALITY, SchemaError


def ordinal_probabilities_from_cumulative_logits(
    cumulative_logits: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Convert ordered logits for P(Y<=CN), P(Y<=MCI) to three probabilities."""

    if cumulative_logits.ndim != 2 or cumulative_logits.shape[1] != 2:
        raise ValueError("cumulative_logits must have shape [batch, 2]")
    cumulative = torch.sigmoid(cumulative_logits)
    p_cn = cumulative[:, 0]
    p_mci = cumulative[:, 1] - cumulative[:, 0]
    p_ad = 1.0 - cumulative[:, 1]
    probabilities = torch.stack((p_cn, p_mci, p_ad), dim=-1)
    probabilities = probabilities.clamp_min(eps)
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def ordinal_probabilities(
    severity: torch.Tensor,
    kappa1: torch.Tensor,
    kappa2: torch.Tensor,
    *,
    temperature: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    severity = severity.reshape(-1)
    temperature_tensor = torch.as_tensor(
        temperature, dtype=severity.dtype, device=severity.device
    ).clamp_min(1e-6)
    cumulative_logits = torch.stack(
        ((kappa1 - severity) / temperature_tensor, (kappa2 - severity) / temperature_tensor),
        dim=-1,
    )
    return ordinal_probabilities_from_cumulative_logits(cumulative_logits)


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(embedding_dim)),
            nn.LayerNorm(int(embedding_dim)),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class OrderedThresholdHead(nn.Module):
    def __init__(self, input_dim: int, threshold_epsilon: float = 1e-4):
        super().__init__()
        self.severity = nn.Linear(int(input_dim), 1)
        self.kappa1 = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        self.raw_gap = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.threshold_epsilon = float(threshold_epsilon)

    def thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
        kappa2 = self.kappa1 + F.softplus(self.raw_gap) + self.threshold_epsilon
        return self.kappa1, kappa2

    def forward(
        self,
        representation: torch.Tensor,
        *,
        temperature: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor]:
        z = self.severity(representation).squeeze(-1)
        kappa1, kappa2 = self.thresholds()
        probabilities = ordinal_probabilities(
            z, kappa1, kappa2, temperature=temperature
        )
        entropy = -(probabilities * probabilities.clamp_min(1e-7).log()).sum(dim=-1)
        distances = torch.stack(((z - kappa1).abs(), (z - kappa2).abs()), dim=-1)
        return {
            "z": z,
            "severity": z,
            "probabilities": probabilities,
            "ordinal_probabilities": probabilities,
            "entropy": entropy,
            "boundary_distances": distances,
            "distance_cn_mci": distances[:, 0],
            "distance_mci_ad": distances[:, 1],
            "kappa1": kappa1,
            "kappa2": kappa2,
            # Compatibility aliases for the copied legacy ordinal artifacts.
            "b1": kappa1,
            "b2": kappa2,
        }


class OrdinalBeliefModel(nn.Module):
    """Mask-aware ordinal model with demographic base evidence always active.

    The observed mask has exactly seven columns in ACQUISITION_MODALITIES order.
    APOE is an acquisition modality even though its raw ADNI source is stored
    beside demographic data. Unobserved modality rows never enter their encoder.
    """

    def __init__(
        self,
        input_dims: Mapping[str, int],
        *,
        encoder_hidden_dim: int = 64,
        embedding_dim: int = 64,
        latent_dim: int = 64,
        dropout: float = 0.2,
        threshold_epsilon: float = 1e-4,
        fusion_mode: str = "mean",
        fusion_depth: int = 1,
    ) -> None:
        super().__init__()
        if tuple(input_dims) != ALL_MODALITIES:
            raise SchemaError(
                f"input_dims must be ordered exactly as {ALL_MODALITIES}; got {tuple(input_dims)}"
            )
        if any(int(input_dims[name]) <= 0 for name in ALL_MODALITIES):
            raise SchemaError("Every modality input dimension must be positive")
        self.input_dims = {name: int(input_dims[name]) for name in ALL_MODALITIES}
        self.embedding_dim = int(embedding_dim)
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {
            "mean",
            "gated_mean",
            "concat",
            "mean_cognitive_residual",
        }:
            raise ValueError(
                "fusion_mode must be one of: mean, gated_mean, concat, "
                "mean_cognitive_residual"
            )
        if int(fusion_depth) not in {1, 2}:
            raise ValueError("fusion_depth must be 1 or 2")
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(
                    self.input_dims[name], encoder_hidden_dim, embedding_dim, dropout
                )
                for name in ALL_MODALITIES
            }
        )
        if self.fusion_mode == "concat":
            fused_dim = int(embedding_dim) * len(ALL_MODALITIES)
        elif self.fusion_mode == "mean_cognitive_residual":
            fused_dim = int(embedding_dim) * 2
        else:
            fused_dim = int(embedding_dim)
        if self.fusion_mode == "gated_mean":
            # One reliability weight per named modality. Equal initialization
            # reproduces the ordinary mean before training.
            self.modality_weight_logits = nn.Parameter(torch.zeros(len(ALL_MODALITIES)))
        fusion_layers: list[nn.Module] = [
            nn.Linear(fused_dim + len(ACQUISITION_MODALITIES), int(latent_dim)),
            nn.LayerNorm(int(latent_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        ]
        if int(fusion_depth) == 2:
            fusion_layers.extend(
                (
                    nn.Linear(int(latent_dim), int(latent_dim)),
                    nn.LayerNorm(int(latent_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                )
            )
        self.fusion = nn.Sequential(*fusion_layers)
        self.ordinal_head = OrderedThresholdHead(latent_dim, threshold_epsilon)

    def _check_features(
        self, features: Mapping[str, torch.Tensor], batch_size: int
    ) -> torch.Tensor:
        if BASE_MODALITY not in features:
            raise SchemaError("demographic base features are required on every forward pass")
        sample = features[BASE_MODALITY]
        if sample.ndim != 2 or sample.shape != (batch_size, self.input_dims[BASE_MODALITY]):
            raise SchemaError(
                f"demographic features must have shape [{batch_size}, "
                f"{self.input_dims[BASE_MODALITY]}]"
            )
        for name, values in features.items():
            if name not in self.input_dims:
                raise SchemaError(f"Unknown modality feature group: {name}")
            if values.ndim != 2 or values.shape[0] != batch_size:
                raise SchemaError(f"{name} features must have shape [batch, feature_dim]")
            if values.shape[1] != self.input_dims[name]:
                raise SchemaError(f"Unexpected feature dimension for {name}")
            if values.device != sample.device:
                raise SchemaError("All modality tensors must be on the same device")
        return sample

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        observed_mask: torch.Tensor,
        available_mask: torch.Tensor | None = None,
        *,
        temperature: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor]:
        validate_observed_mask(observed_mask, available_mask)
        batch_size = int(observed_mask.shape[0])
        sample = self._check_features(features, batch_size)
        observed = observed_mask.to(device=sample.device, dtype=torch.bool)

        demographic_embedding = self.encoders[BASE_MODALITY](features[BASE_MODALITY])
        embeddings = [demographic_embedding]
        for modality_index, modality in enumerate(ACQUISITION_MODALITIES):
            active = observed[:, modality_index]
            encoded_full = sample.new_zeros((batch_size, self.embedding_dim))
            if bool(torch.any(active).item()):
                if modality not in features:
                    raise SchemaError(f"Observed modality {modality} has no feature tensor")
                encoded_full[active] = self.encoders[modality](features[modality][active])
            embeddings.append(encoded_full)
        stacked = torch.stack(embeddings, dim=1)
        if self.fusion_mode == "concat":
            fused = stacked.reshape(batch_size, -1)
        else:
            active_mask = torch.cat(
                (
                    torch.ones((batch_size, 1), dtype=torch.bool, device=sample.device),
                    observed,
                ),
                dim=1,
            )
            if self.fusion_mode == "gated_mean":
                reliability = torch.nn.functional.softplus(self.modality_weight_logits)
                weights = active_mask.to(sample.dtype) * reliability.unsqueeze(0)
            else:
                weights = active_mask.to(sample.dtype)
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
            fused = fused / weights.sum(dim=1, keepdim=True).clamp_min(1e-7)
            if self.fusion_mode == "mean_cognitive_residual":
                # Cognitive is the first acquisition modality and is available
                # for every patient in the locked cohort. Keep its identity as
                # a residual while still using every observed modality.
                fused = torch.cat((fused, stacked[:, 1]), dim=-1)
        h = self.fusion(torch.cat((fused, observed.to(sample.dtype)), dim=-1))
        result = self.ordinal_head(h, temperature=temperature)
        result["h"] = h
        return result

    def thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ordinal_head.thresholds()


def build_ordinal_model(input_dims: Mapping[str, int], config: Mapping[str, object]) -> OrdinalBeliefModel:
    return OrdinalBeliefModel(
        input_dims,
        encoder_hidden_dim=int(config.get("encoder_hidden_dim", 64)),
        embedding_dim=int(config.get("embedding_dim", 64)),
        latent_dim=int(config.get("latent_dim", 64)),
        dropout=float(config.get("dropout", 0.2)),
        threshold_epsilon=float(config.get("threshold_epsilon", 1e-4)),
        fusion_mode=str(config.get("fusion_mode", "mean")),
        fusion_depth=int(config.get("fusion_depth", 1)),
    )
