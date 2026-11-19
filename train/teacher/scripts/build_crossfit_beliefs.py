from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.masks import expand_patient_partial_states  # noqa: E402
from src.data.preprocessing import (  # noqa: E402
    build_preprocessor,
    load_and_parse,
    load_prepared,
)
from src.data.splits import inner_validation_split, patient_stratified_kfold  # noqa: E402
from src.evaluation.diagnosis import diagnosis_metrics  # noqa: E402
from src.ordinal.cache import (  # noqa: E402
    build_canonical_modality_vectors,
    canonical_input_schema_sha256,
    canonical_modality_vectors_sha256,
    canonical_ordinal_coordinates,
    canonical_representation_metadata,
    compose_canonical_h,
    save_belief_cache,
)
from src.ordinal.calibration import (  # noqa: E402
    TemperatureCalibration,
    evaluate_partial_calibration_gate,
    fit_validation_temperature,
)
from src.ordinal.crossfit import (  # noqa: E402
    build_seeded_ordinal_model,
    fit_ordinal_model,
    predict_ordinal_beliefs,
    resolve_device,
)
from src.ordinal.model import build_ordinal_model  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_joblib_dump(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    try:
        joblib.dump(value, temporary, compress=3)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _assert_prepared_exact_match(current: Any, frozen: Any) -> None:
    """Fail if current raw episodes do not reproduce the frozen final data artifact."""

    scalar_fields = ("split_manifest_sha256", "input_dims")
    for field in scalar_fields:
        if getattr(current, field) != getattr(frozen, field):
            raise RuntimeError(f"Current and frozen prepared data differ in {field}")
    array_fields = ("patient_ids", "labels", "splits", "available_mask")
    for field in array_fields:
        if not np.array_equal(getattr(current, field), getattr(frozen, field)):
            raise RuntimeError(f"Current and frozen prepared data differ in {field}")
    if tuple(current.features) != tuple(frozen.features):
        raise RuntimeError("Current and frozen prepared modality order differs")
    for modality in current.features:
        if not np.array_equal(
            np.asarray(current.features[modality]),
            np.asarray(frozen.features[modality]),
            equal_nan=True,
        ):
            raise RuntimeError(
                f"Current raw episodes do not reproduce frozen {modality} features"
            )


def _expanded_states(
    patient_indices: np.ndarray,
    available_mask: np.ndarray,
    *,
    max_states: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return expand_patient_partial_states(
        patient_indices,
        available_mask,
        max_states_per_patient=int(max_states),
    )


def _records_for_prediction(
    *,
    prepared: Any,
    patient_indices: np.ndarray,
    observed_masks: np.ndarray,
    state_ids: np.ndarray,
    prediction: Mapping[str, np.ndarray | float],
    canonical_h: np.ndarray,
    split: str,
    fold_id: int,
    prediction_source: str,
) -> dict[str, np.ndarray]:
    count = len(patient_indices)
    probabilities = np.asarray(prediction["probabilities"], dtype=np.float32)
    canonical_severity, canonical_boundaries = canonical_ordinal_coordinates(probabilities)
    if np.asarray(canonical_h).shape != (count, 64):
        raise RuntimeError("Canonical h must have shape [records, 64]")
    return {
        "patient_id": prepared.patient_ids[patient_indices].astype(str),
        # ``dtype=str`` creates a one-character Unicode array in NumPy and would
        # silently truncate ``train``/``validation``/``test``.  Object arrays
        # preserve the exact audited provenance strings through joblib.
        "split": np.full(count, split, dtype=object),
        "state_id": state_ids.astype(np.int64),
        "fold_id": np.full(count, int(fold_id), dtype=np.int64),
        "prediction_source": np.full(count, prediction_source, dtype=object),
        "observed_mask": observed_masks.astype(bool),
        "available_mask": prepared.available_mask[patient_indices].astype(bool),
        # Never cache a fold model's non-identifiable internal hidden space.
        "h": np.asarray(canonical_h, dtype=np.float32),
        "z": canonical_severity,
        "probabilities": probabilities,
        "entropy": np.asarray(prediction["entropy"], dtype=np.float32),
        "boundary_distances": canonical_boundaries,
    }


def _concatenate(records: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    fields = tuple(records[0])
    if any(tuple(record) != fields for record in records):
        raise RuntimeError("Belief record schemas differ")
    return {field: np.concatenate([record[field] for record in records], axis=0) for field in fields}


def _load_final_model(
    ordinal_config: Mapping[str, Any],
    *,
    ordinal_config_path: Path,
    data_config_path: Path,
    device: torch.device,
) -> tuple[Any, Any, Any, TemperatureCalibration, Mapping[str, Any]]:
    artifacts = ordinal_config["artifacts"]
    prepared_path = _project_path(artifacts["prepared_data"])
    preprocessor_path = _project_path(artifacts["preprocessor"])
    prepared = load_prepared(prepared_path)
    checkpoint_path = _project_path(artifacts["checkpoint"])
    calibration_path = _project_path(artifacts["calibration"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("prepared_sha256") != _sha256(prepared_path):
        raise RuntimeError("Final prepared data are not bound to the ordinal checkpoint")
    if checkpoint.get("preprocessor_sha256") != _sha256(preprocessor_path):
        raise RuntimeError("Final preprocessor is not bound to the ordinal checkpoint")
    if checkpoint.get("data_config_sha256") != _sha256(data_config_path):
        raise RuntimeError("Current data configuration is not bound to the ordinal checkpoint")
    if checkpoint.get("ordinal_config_sha256") != _sha256(ordinal_config_path):
        raise RuntimeError("Current ordinal configuration is not bound to the ordinal checkpoint")
    checkpoint_gate = checkpoint.get("calibration_acceptance")
    if not isinstance(checkpoint_gate, Mapping) or checkpoint_gate.get("passed") is not True:
        raise RuntimeError(
            "Final ordinal checkpoint did not pass the predeclared partial calibration gate"
        )
    if checkpoint.get("legacy_initialization_used") is not False:
        raise RuntimeError("Final ordinal checkpoint does not prove clean from-scratch training")
    if checkpoint.get("split_manifest_sha256") != prepared.split_manifest_sha256:
        raise RuntimeError(
            "Final ordinal checkpoint and prepared data use different locked split manifests"
        )
    model = build_ordinal_model(prepared.input_dims, checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    calibration_digest = _sha256(calibration_path)
    if checkpoint.get("calibration_sha256") != calibration_digest:
        raise RuntimeError(
            "Final calibration file is not the one bound to the ordinal checkpoint"
        )
    calibration = TemperatureCalibration.load(calibration_path)
    if calibration.method != str(ordinal_config["calibration"].get("method")):
        raise RuntimeError("Final calibration method differs from ordinal configuration")
    if not np.isclose(
        float(checkpoint.get("calibration_temperature", np.nan)),
        float(calibration.temperature),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Ordinal checkpoint and calibration temperature differ")
    for checkpoint_field, artifact_value in (
        ("calibration_kappa1", calibration.calibrated_kappa1),
        ("calibration_kappa2", calibration.calibrated_kappa2),
    ):
        checkpoint_value = checkpoint.get(checkpoint_field)
        if checkpoint_value is None or artifact_value is None or not np.isclose(
            float(checkpoint_value), float(artifact_value), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                f"Ordinal checkpoint and calibration artifact differ in {checkpoint_field}"
            )
    if checkpoint.get("calibration_method") != calibration.method:
        raise RuntimeError("Ordinal checkpoint and calibration method differ")
    preprocessor = joblib.load(preprocessor_path)
    if not hasattr(preprocessor, "transform") or not hasattr(
        preprocessor, "fit_patient_ids"
    ):
        raise RuntimeError("Final preprocessor artifact has an incompatible type")
    if list(checkpoint.get("fit_patient_ids", ())) != list(preprocessor.fit_patient_ids):
        raise RuntimeError("Final checkpoint and preprocessor fit-patient IDs differ")
    return prepared, preprocessor, model, calibration, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build auditable five-fold OOF ordinal belief states"
    )
    parser.add_argument("--config", default="configs/ordinal.yaml")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    ordinal_config_path = _project_path(args.config)
    ordinal_config = _load_yaml(ordinal_config_path)
    data_config_path = _project_path(ordinal_config["data_config"])
    data_config = _load_yaml(data_config_path)
    seed = int(args.seed if args.seed is not None else ordinal_config.get("seed", 42))
    crossfit_config = ordinal_config["crossfit"]
    if not bool(crossfit_config.get("require_oof_train_beliefs", True)):
        raise RuntimeError(
            "This pipeline refuses to build in-sample train beliefs; "
            "require_oof_train_beliefs must remain true"
        )
    if int(crossfit_config.get("folds", 0)) != 5:
        raise RuntimeError("Ordinal cross-fitting is fixed to five patient-level folds")

    episodes = load_and_parse(data_config, config_dir=PROJECT_ROOT)
    canonical_modality_vectors = build_canonical_modality_vectors(
        episodes.modality_frames
    )
    canonical_input_schema_digest = canonical_input_schema_sha256(
        episodes.modality_frames
    )
    canonical_input_vectors_digest = canonical_modality_vectors_sha256(
        canonical_modality_vectors
    )
    train_global = episodes.indices("train")
    validation_global = episodes.indices("validation")
    test_global = episodes.indices("test")
    local_folds = patient_stratified_kfold(
        episodes.patient_ids[train_global],
        episodes.labels[train_global],
        n_splits=5,
        seed=seed,
        shuffle=bool(crossfit_config.get("shuffle", True)),
    )
    device = resolve_device(str(ordinal_config.get("device", "auto")))
    (
        final_prepared,
        final_preprocessor,
        final_model,
        final_calibration,
        final_checkpoint,
    ) = _load_final_model(
        ordinal_config,
        ordinal_config_path=ordinal_config_path,
        data_config_path=data_config_path,
        device=device,
    )
    if int(final_checkpoint.get("seed", -1)) != seed:
        raise RuntimeError(
            "Final ordinal checkpoint seed does not match the requested cross-fit seed"
        )
    current_final_prepared = final_preprocessor.transform(episodes)
    _assert_prepared_exact_match(current_final_prepared, final_prepared)
    expected_final_fit_ids = episodes.patient_ids[train_global].astype(str).tolist()
    if list(final_preprocessor.fit_patient_ids) != expected_final_fit_ids:
        raise RuntimeError(
            "Final preprocessor was not fitted on exactly the locked training patients"
        )
    max_states = int(crossfit_config.get("max_states_per_patient", 128))
    batch_size = int(ordinal_config["training"].get("batch_size", 128))
    artifact_dir = _project_path(ordinal_config["artifacts"]["output_dir"])
    gate_config = ordinal_config["calibration"].get("acceptance_gate", {})
    final_gate_indices, final_gate_masks, _ = expand_patient_partial_states(
        validation_global,
        final_prepared.available_mask,
        max_states_per_patient=int(
            ordinal_config["calibration"].get("max_states_per_patient", 16)
        ),
    )
    final_gate_prediction = predict_ordinal_beliefs(
        final_model,
        final_prepared,
        final_gate_indices,
        final_gate_masks,
        batch_size=batch_size,
        temperature=final_calibration.temperature,
        calibrated_thresholds=final_calibration.resolved_thresholds(
            *final_model.thresholds()
        ),
        device=device,
    )
    final_gate_metrics = diagnosis_metrics(
        final_prepared.labels[final_gate_indices],
        probabilities=final_gate_prediction["probabilities"],
    )
    final_calibration_acceptance = evaluate_partial_calibration_gate(
        final_calibration,
        final_gate_metrics,
        gate_config,
    )
    if final_calibration_acceptance["passed"] is not True:
        raise RuntimeError(
            "Recomputed final-model validation calibration failed the predeclared gate"
        )
    checkpoint_acceptance = final_checkpoint["calibration_acceptance"]
    if (
        checkpoint_acceptance.get("version") != final_calibration_acceptance["version"]
        or checkpoint_acceptance.get("thresholds")
        != final_calibration_acceptance["thresholds"]
    ):
        raise RuntimeError("Checkpoint and current calibration acceptance contracts differ")

    record_parts: list[dict[str, np.ndarray]] = []
    fold_manifest: list[dict[str, object]] = []
    fold_temperatures: dict[str, float] = {}
    fold_calibration_acceptance: dict[str, Mapping[str, Any]] = {}
    for fold in local_folds:
        outer_fit = train_global[fold.fit_indices]
        outer_holdout = train_global[fold.holdout_indices]
        inner_fit, inner_validation = inner_validation_split(
            outer_fit,
            episodes.labels,
            seed=seed + fold.fold * 10_007,
            validation_fraction=0.10,
        )
        # Preprocessing itself is fit without the outer holdout patient.
        preprocessor = build_preprocessor(data_config)
        prepared_fold = preprocessor.fit_transform(episodes, outer_fit)
        fold_training_seed = seed + fold.fold * 100_003
        model = build_seeded_ordinal_model(
            prepared_fold.input_dims,
            ordinal_config["model"],
            seed=fold_training_seed,
        )
        fit_summary = fit_ordinal_model(
            model,
            prepared_fold,
            inner_fit,
            inner_validation,
            ordinal_config,
            seed=fold_training_seed,
            device=device,
        )

        # Each fold temperature is fitted only on the fixed external validation set.
        calibration_indices, calibration_masks, _ = expand_patient_partial_states(
            validation_global,
            prepared_fold.available_mask,
            max_states_per_patient=int(
                ordinal_config["calibration"].get("max_states_per_patient", 16)
            ),
        )
        validation_prediction = predict_ordinal_beliefs(
            model,
            prepared_fold,
            calibration_indices,
            calibration_masks,
            batch_size=batch_size,
            temperature=1.0,
            device=device,
        )
        calibration = fit_validation_temperature(
            validation_prediction["z"],
            prepared_fold.labels[calibration_indices],
            validation_prediction["kappa1"],
            validation_prediction["kappa2"],
            split_name="validation",
            max_iterations=int(ordinal_config["calibration"].get("max_iterations", 100)),
            initial_temperature=float(
                ordinal_config["calibration"].get("initial_temperature", 1.0)
            ),
            patient_count=len(validation_global),
            calibrate_thresholds=(
                str(ordinal_config["calibration"].get("method"))
                == "ordinal_temperature_thresholds"
            ),
        )
        if calibration.method != str(ordinal_config["calibration"].get("method")):
            raise RuntimeError(
                f"Cross-fit fold {fold.fold} produced the wrong calibration method"
            )
        fold_temperatures[str(fold.fold)] = calibration.temperature
        calibrated_thresholds = calibration.resolved_thresholds(
            validation_prediction["kappa1"], validation_prediction["kappa2"]
        )
        calibrated_validation_prediction = predict_ordinal_beliefs(
            model,
            prepared_fold,
            calibration_indices,
            calibration_masks,
            batch_size=batch_size,
            temperature=calibration.temperature,
            calibrated_thresholds=calibrated_thresholds,
            device=device,
        )
        fold_gate = evaluate_partial_calibration_gate(
            calibration,
            diagnosis_metrics(
                prepared_fold.labels[calibration_indices],
                probabilities=calibrated_validation_prediction["probabilities"],
            ),
            gate_config,
        )
        if fold_gate["passed"] is not True:
            raise RuntimeError(
                f"Cross-fit fold {fold.fold} failed the predeclared validation "
                "partial-state calibration gate"
            )
        fold_calibration_acceptance[str(fold.fold)] = fold_gate
        expanded, masks, state_ids = _expanded_states(
            outer_holdout,
            prepared_fold.available_mask,
            max_states=max_states,
        )
        prediction = predict_ordinal_beliefs(
            model,
            prepared_fold,
            expanded,
            masks,
            batch_size=max(batch_size, 256),
            temperature=calibration.temperature,
            calibrated_thresholds=calibrated_thresholds,
            device=device,
        )
        canonical_h = compose_canonical_h(
            canonical_modality_vectors, expanded, masks
        )
        record_parts.append(
            _records_for_prediction(
                prepared=prepared_fold,
                patient_indices=expanded,
                observed_masks=masks,
                state_ids=state_ids,
                prediction=prediction,
                canonical_h=canonical_h,
                split="train",
                fold_id=fold.fold,
                prediction_source="oof",
            )
        )
        fold_dir = artifact_dir / "crossfit" / f"fold_{fold.fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_preprocessor_path = fold_dir / "preprocessor.joblib"
        fold_model_path = fold_dir / "model.pt"
        fold_calibration_path = fold_dir / "temperature_calibration.json"
        _atomic_joblib_dump(preprocessor, fold_preprocessor_path)
        calibration.save(fold_calibration_path)
        fold_preprocessor_sha256 = _sha256(fold_preprocessor_path)
        fold_model_payload = {
                "schema_version": 1,
                "fold": fold.fold,
                "seed": fold_training_seed,
                "crossfit_seed": seed,
                "input_dims": dict(prepared_fold.input_dims),
                "model_config": dict(ordinal_config["model"]),
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "temperature": calibration.temperature,
                "calibration_method": calibration.method,
                "calibration_kappa1": calibration.calibrated_kappa1,
                "calibration_kappa2": calibration.calibrated_kappa2,
                "calibration_sha256": _sha256(fold_calibration_path),
                "calibration_acceptance": fold_gate,
                "preprocessor_sha256": fold_preprocessor_sha256,
                "data_config_sha256": _sha256(data_config_path),
                "ordinal_config_sha256": _sha256(ordinal_config_path),
                "calibration_state_count": len(calibration_indices),
                "fit_patient_ids": list(fold.fit_patient_ids),
                "holdout_patient_ids": list(fold.holdout_patient_ids),
                "optimizer_fit_patient_ids": episodes.patient_ids[inner_fit].astype(str).tolist(),
                "early_stopping_patient_ids": episodes.patient_ids[inner_validation]
                .astype(str)
                .tolist(),
                "preprocessor_fit_patient_ids": list(preprocessor.fit_patient_ids),
                "split_manifest_sha256": episodes.split_manifest_sha256,
                **canonical_representation_metadata(
                    input_schema_sha256=canonical_input_schema_digest,
                    input_vectors_sha256=canonical_input_vectors_digest,
                ),
                "best_epoch": fit_summary.best_epoch,
            }
        _atomic_torch_save(fold_model_payload, fold_model_path)
        fold_manifest.append(
            {
                **fold.as_manifest(),
                "model_sha256": _sha256(fold_model_path),
                "preprocessor_sha256": fold_preprocessor_sha256,
                "calibration_sha256": _sha256(fold_calibration_path),
                "calibration_acceptance": fold_gate,
                "training_seed": fold_training_seed,
            }
        )

    final_checkpoint_path = _project_path(ordinal_config["artifacts"]["checkpoint"])
    final_calibration_path = _project_path(ordinal_config["artifacts"]["calibration"])
    final_prepared_path = _project_path(ordinal_config["artifacts"]["prepared_data"])
    final_preprocessor_path = _project_path(ordinal_config["artifacts"]["preprocessor"])
    if final_prepared.split_manifest_sha256 != episodes.split_manifest_sha256:
        raise RuntimeError(
            "Final ordinal artifacts do not match the currently verified locked split manifest"
        )
    for split, patient_indices in (
        ("validation", validation_global),
        ("test", test_global),
    ):
        expanded, masks, state_ids = _expanded_states(
            patient_indices,
            final_prepared.available_mask,
            max_states=max_states,
        )
        prediction = predict_ordinal_beliefs(
            final_model,
            final_prepared,
            expanded,
            masks,
            batch_size=max(batch_size, 256),
            temperature=final_calibration.temperature,
            calibrated_thresholds=final_calibration.resolved_thresholds(
                *final_model.thresholds()
            ),
            device=device,
        )
        canonical_h = compose_canonical_h(
            canonical_modality_vectors, expanded, masks
        )
        record_parts.append(
            _records_for_prediction(
                prepared=final_prepared,
                patient_indices=expanded,
                observed_masks=masks,
                state_ids=state_ids,
                prediction=prediction,
                canonical_h=canonical_h,
                split=split,
                fold_id=-1,
                prediction_source="full_train_model",
            )
        )

    records = _concatenate(record_parts)
    metadata = {
        "prediction_mode": "patient_level_5fold_oof",
        "crossfit_folds": 5,
        "oof_verified": True,
        "fold_manifest": fold_manifest,
        "fold_temperatures": fold_temperatures,
        "fold_calibration_acceptance": fold_calibration_acceptance,
        "final_calibration_acceptance": final_calibration_acceptance,
        "validation_calibration": final_calibration.method,
        "final_calibration_parameters": {
            "temperature": final_calibration.temperature,
            "kappa1": final_calibration.calibrated_kappa1,
            "kappa2": final_calibration.calibrated_kappa2,
        },
        "validation_patient_ids": episodes.patient_ids[validation_global].astype(str).tolist(),
        "test_patient_ids": episodes.patient_ids[test_global].astype(str).tolist(),
        "seed": seed,
        "split_manifest_sha256": episodes.split_manifest_sha256,
        "final_checkpoint_sha256": _sha256(final_checkpoint_path),
        "final_calibration_sha256": _sha256(final_calibration_path),
        "final_prepared_sha256": _sha256(final_prepared_path),
        "final_preprocessor_sha256": _sha256(final_preprocessor_path),
        "data_config_sha256": _sha256(data_config_path),
        "ordinal_config_sha256": _sha256(ordinal_config_path),
        "legacy_initialization_used": final_checkpoint.get(
            "legacy_initialization_used"
        ),
        **canonical_representation_metadata(
            input_schema_sha256=canonical_input_schema_digest,
            input_vectors_sha256=canonical_input_vectors_digest,
        ),
        "note": (
            "Every train row was predicted by the fold that held out its patient. "
            "In-sample training beliefs are forbidden."
        ),
    }
    cache_path = _project_path(ordinal_config["artifacts"]["crossfit_cache"])
    save_belief_cache(cache_path, records, metadata)
    summary = {
        "cache": str(cache_path),
        "records": len(records["patient_id"]),
        "train_patients": len(np.unique(records["patient_id"][records["split"] == "train"])),
        "folds": 5,
        "oof_verified": True,
        "fold_temperatures": fold_temperatures,
        "fold_calibration_acceptance": fold_calibration_acceptance,
        "final_calibration_acceptance": final_calibration_acceptance,
        "split_manifest_sha256": episodes.split_manifest_sha256,
        "final_prepared_sha256": _sha256(final_prepared_path),
        "final_preprocessor_sha256": _sha256(final_preprocessor_path),
        "ordinal_config_sha256": _sha256(ordinal_config_path),
        "h_input_schema_sha256": canonical_input_schema_digest,
        "h_input_vectors_sha256": canonical_input_vectors_digest,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
