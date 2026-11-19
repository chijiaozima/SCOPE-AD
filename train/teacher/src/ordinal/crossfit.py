from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.masks import expand_patient_partial_states, sample_random_masks_torch
from src.data.schema import PreparedEpisodes, SchemaError
from src.data.splits import CrossfitFold, patient_stratified_kfold
from src.utils.seed import set_seed

from .losses import OrdinalBeliefLoss
from .model import OrdinalBeliefModel
from .model import build_ordinal_model


def set_deterministic_seed(seed: int) -> None:
    """Compatibility wrapper for strict project-wide deterministic seeding."""
    set_seed(int(seed), deterministic=True, strict=True)


def build_seeded_ordinal_model(
    input_dims: Mapping[str, int],
    model_config: Mapping[str, object],
    *,
    seed: int,
) -> OrdinalBeliefModel:
    """Seed before parameter initialization and return a reproducible model."""

    set_deterministic_seed(int(seed))
    return build_ordinal_model(input_dims, model_config)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@dataclass(frozen=True)
class FitSummary:
    best_epoch: int
    best_validation_macro_f1: float
    history: tuple[dict[str, float | int], ...]


def _feature_batch(
    prepared: PreparedEpisodes,
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    numpy_indices = indices.detach().cpu().numpy()
    return {
        name: torch.from_numpy(np.asarray(values[numpy_indices], dtype=np.float32)).to(device)
        for name, values in prepared.features.items()
    }


def _loader(indices: Sequence[int], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    tensor = torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long)
    return DataLoader(
        TensorDataset(tensor),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=torch.Generator().manual_seed(int(seed)),
    )


def _use_bf16(device: torch.device, precision: str) -> bool:
    return precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()


def fit_ordinal_model(
    model: OrdinalBeliefModel,
    prepared: PreparedEpisodes,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device | None = None,
) -> FitSummary:
    """Train on random partial states; validation patients are never optimizer inputs."""

    set_deterministic_seed(seed)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if len(set(train_indices) & set(validation_indices)):
        raise SchemaError("Training and validation indices overlap")
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise SchemaError("Training and validation indices must be non-empty")
    device = device or resolve_device(str(config.get("device", "auto")))
    training = config.get("training", config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    criterion = OrdinalBeliefLoss(float(training.get("expected_mae_weight", 0.0)))
    batch_size = int(training.get("batch_size", 128))
    train_loader = _loader(train_indices, batch_size, True, seed)
    patience_limit = int(training.get("early_stopping_patience", 12))
    epochs = int(training.get("epochs", 80))
    base_only_probability = float(training.get("include_base_only_probability", 0.15))
    gradient_clip = float(training.get("gradient_clip_norm", 5.0))
    bf16 = _use_bf16(device, str(config.get("precision", "bf16")))
    mask_generator: torch.Generator | None = None
    if device.type in {"cpu", "cuda"}:
        mask_generator = torch.Generator(device=device.type)
        mask_generator.manual_seed(int(seed) + 9_973)
    validation_patients, validation_masks, _ = expand_patient_partial_states(
        validation_indices,
        prepared.available_mask,
        max_states_per_patient=int(training.get("validation_states_per_patient", 8)),
    )
    validation_state_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation_patients).long(),
            torch.from_numpy(validation_masks).bool(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    best_epoch = -1
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for (batch_indices,) in train_loader:
            batch_indices = batch_indices.long()
            available = torch.from_numpy(
                prepared.available_mask[batch_indices.numpy()].astype(bool)
            ).to(device)
            observed = sample_random_masks_torch(
                available,
                generator=mask_generator,
                base_only_probability=base_only_probability,
            )
            features = _feature_batch(prepared, batch_indices, device)
            targets = torch.from_numpy(prepared.labels[batch_indices.numpy()]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                output = model(features, observed, available)
                loss = criterion(output, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses: list[float] = []
        targets_all: list[np.ndarray] = []
        predictions_all: list[np.ndarray] = []
        with torch.no_grad():
            for batch_indices, batch_observed in validation_state_loader:
                batch_indices = batch_indices.long()
                available = torch.from_numpy(
                    prepared.available_mask[batch_indices.numpy()].astype(bool)
                ).to(device)
                observed = batch_observed.bool().to(device)
                features = _feature_batch(prepared, batch_indices, device)
                targets = torch.from_numpy(prepared.labels[batch_indices.numpy()]).long().to(device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=bf16
                ):
                    output = model(features, observed, available)
                    loss = criterion(output, targets)
                validation_losses.append(float(loss.detach().cpu()))
                targets_all.append(targets.cpu().numpy())
                predictions_all.append(output["probabilities"].argmax(dim=1).cpu().numpy())
        score = float(
            f1_score(
                np.concatenate(targets_all),
                np.concatenate(predictions_all),
                average="macro",
                labels=[0, 1, 2],
                zero_division=0,
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": float(np.mean(validation_losses)),
                "validation_macro_f1": score,
            }
        )
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break
    if best_state is None:
        raise RuntimeError("Ordinal training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return FitSummary(
        best_epoch=best_epoch,
        best_validation_macro_f1=best_score,
        history=tuple(history),
    )


def predict_ordinal_beliefs(
    model: OrdinalBeliefModel,
    prepared: PreparedEpisodes,
    patient_indices: Sequence[int],
    observed_masks: np.ndarray,
    *,
    batch_size: int = 512,
    temperature: float = 1.0,
    calibrated_thresholds: Sequence[float] | None = None,
    device: torch.device | None = None,
) -> dict[str, np.ndarray | float]:
    indices = np.asarray(patient_indices, dtype=np.int64)
    masks = np.asarray(observed_masks, dtype=bool)
    if masks.shape != (len(indices), prepared.available_mask.shape[1]):
        raise SchemaError("observed_masks must align with patient_indices")
    available_for_records = prepared.available_mask[indices]
    if np.any(masks & ~available_for_records):
        raise SchemaError("A prediction state observes an unavailable modality")
    device = device or next(model.parameters()).device
    effective_thresholds: tuple[float, float] | None = None
    if calibrated_thresholds is not None:
        threshold_values = np.asarray(calibrated_thresholds, dtype=np.float64)
        if (
            threshold_values.shape != (2,)
            or not np.all(np.isfinite(threshold_values))
            or not threshold_values[0] < threshold_values[1]
        ):
            raise SchemaError("calibrated_thresholds must contain two finite ordered values")
        effective_thresholds = (float(threshold_values[0]), float(threshold_values[1]))
    model.to(device)
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {
        "h": [],
        "z": [],
        "probabilities": [],
        "entropy": [],
        "boundary_distances": [],
    }
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            stop = min(start + int(batch_size), len(indices))
            batch_indices_numpy = indices[start:stop]
            batch_indices = torch.from_numpy(batch_indices_numpy).long()
            features = _feature_batch(prepared, batch_indices, device)
            observed = torch.from_numpy(masks[start:stop]).bool().to(device)
            available = torch.from_numpy(available_for_records[start:stop]).bool().to(device)
            output = model(
                features,
                observed,
                available,
                temperature=float(temperature),
            )
            if effective_thresholds is not None:
                from .model import ordinal_probabilities

                first = torch.as_tensor(
                    effective_thresholds[0], dtype=output["z"].dtype, device=device
                )
                second = torch.as_tensor(
                    effective_thresholds[1], dtype=output["z"].dtype, device=device
                )
                probabilities = ordinal_probabilities(
                    output["z"], first, second, temperature=float(temperature)
                )
                output["probabilities"] = probabilities
                output["ordinal_probabilities"] = probabilities
                output["entropy"] = -(
                    probabilities * probabilities.clamp_min(1e-7).log()
                ).sum(dim=-1)
                distances = torch.stack(
                    ((output["z"] - first).abs(), (output["z"] - second).abs()),
                    dim=-1,
                )
                output["boundary_distances"] = distances
            for key in arrays:
                arrays[key].append(output[key].float().cpu().numpy())
    result: dict[str, np.ndarray | float] = {
        key: np.concatenate(values, axis=0) for key, values in arrays.items()
    }
    if effective_thresholds is None:
        kappa1, kappa2 = model.thresholds()
        effective_thresholds = (
            float(kappa1.detach().cpu()),
            float(kappa2.detach().cpu()),
        )
    result["kappa1"], result["kappa2"] = effective_thresholds
    return result


@dataclass(frozen=True)
class GenericCrossfitResult:
    predictions: Mapping[str, np.ndarray]
    fold_id: np.ndarray
    folds: tuple[CrossfitFold, ...]


def patient_level_crossfit_predict(
    patient_ids: Sequence[object],
    labels: Sequence[int],
    *,
    model_factory: Callable[[int], object],
    fit_fn: Callable[[object, np.ndarray, int], None],
    predict_fn: Callable[[object, np.ndarray, int], Mapping[str, np.ndarray]],
    seed: int = 42,
    n_splits: int = 5,
) -> GenericCrossfitResult:
    """Generic testable OOF orchestration; a fold model never sees its holdout rows."""

    ids = np.asarray(patient_ids, dtype=str)
    y = np.asarray(labels, dtype=np.int64)
    folds = patient_stratified_kfold(
        ids, y, n_splits=int(n_splits), seed=int(seed), shuffle=True
    )
    assembled: dict[str, np.ndarray] = {}
    fold_id = np.full(len(ids), -1, dtype=np.int64)
    filled = np.zeros(len(ids), dtype=bool)
    for fold in folds:
        model = model_factory(fold.fold)
        fit_fn(model, fold.fit_indices.copy(), fold.fold)
        predicted = predict_fn(model, fold.holdout_indices.copy(), fold.fold)
        for key, values in predicted.items():
            array = np.asarray(values)
            if array.shape[0] != len(fold.holdout_indices):
                raise SchemaError(f"Prediction {key} does not align with fold holdout")
            if key not in assembled:
                assembled[key] = np.empty((len(ids), *array.shape[1:]), dtype=array.dtype)
            assembled[key][fold.holdout_indices] = array
        fold_id[fold.holdout_indices] = fold.fold
        filled[fold.holdout_indices] = True
    if not np.all(filled) or np.any(fold_id < 0):
        raise RuntimeError("Cross-fitting did not produce exactly one OOF prediction per row")
    return GenericCrossfitResult(
        predictions=assembled,
        fold_id=fold_id,
        folds=tuple(folds),
    )
