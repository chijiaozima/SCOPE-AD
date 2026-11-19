from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (  # noqa: E402
    build_preprocessor,
    load_and_parse,
    save_prepared,
)
from src.evaluation.diagnosis import diagnosis_metrics  # noqa: E402
from src.data.masks import expand_patient_partial_states  # noqa: E402
from src.ordinal.calibration import (  # noqa: E402
    evaluate_partial_calibration_gate,
    fit_validation_temperature,
)
from src.ordinal.crossfit import (  # noqa: E402
    build_seeded_ordinal_model,
    fit_ordinal_model,
    predict_ordinal_beliefs,
    resolve_device,
)
from src.utils.io import atomic_write_json  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and calibrate the ordinal belief model")
    parser.add_argument("--config", default="configs/ordinal.yaml")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    ordinal_config_path = _project_path(args.config)
    ordinal_config = _load_yaml(ordinal_config_path)
    data_config_path = _project_path(ordinal_config["data_config"])
    data_config = _load_yaml(data_config_path)
    seed = int(args.seed if args.seed is not None else ordinal_config.get("seed", 42))

    episodes = load_and_parse(data_config, config_dir=PROJECT_ROOT)
    train_indices = episodes.indices("train")
    validation_indices = episodes.indices("validation")
    test_indices = episodes.indices("test")
    if min(len(train_indices), len(validation_indices), len(test_indices)) == 0:
        raise RuntimeError("Source split files must all contain patients")

    preprocessor = build_preprocessor(data_config)
    prepared = preprocessor.fit_transform(episodes, train_indices)
    model = build_seeded_ordinal_model(
        prepared.input_dims,
        ordinal_config["model"],
        seed=seed,
    )

    device = resolve_device(str(ordinal_config.get("device", "auto")))
    fit_summary = fit_ordinal_model(
        model,
        prepared,
        train_indices,
        validation_indices,
        ordinal_config,
        seed=seed,
        device=device,
    )
    calibration_indices, calibration_masks, _ = expand_patient_partial_states(
        validation_indices,
        prepared.available_mask,
        max_states_per_patient=int(
            ordinal_config["calibration"].get("max_states_per_patient", 16)
        ),
    )
    validation_prediction = predict_ordinal_beliefs(
        model,
        prepared,
        calibration_indices,
        calibration_masks,
        batch_size=int(ordinal_config["training"].get("batch_size", 128)),
        temperature=1.0,
        device=device,
    )
    calibration_config = ordinal_config["calibration"]
    calibration = fit_validation_temperature(
        validation_prediction["z"],
        prepared.labels[calibration_indices],
        validation_prediction["kappa1"],
        validation_prediction["kappa2"],
        split_name=str(calibration_config["fit_split"]),
        max_iterations=int(calibration_config.get("max_iterations", 100)),
        initial_temperature=float(calibration_config.get("initial_temperature", 1.0)),
        patient_count=len(validation_indices),
        calibrate_thresholds=(
            str(calibration_config.get("method"))
            == "ordinal_temperature_thresholds"
        ),
    )
    if calibration.method != str(calibration_config.get("method")):
        raise RuntimeError("Produced ordinal calibration does not match configured method")
    calibrated_thresholds = calibration.resolved_thresholds(
        validation_prediction["kappa1"], validation_prediction["kappa2"]
    )
    calibrated_prediction = predict_ordinal_beliefs(
        model,
        prepared,
        calibration_indices,
        calibration_masks,
        batch_size=int(ordinal_config["training"].get("batch_size", 128)),
        temperature=calibration.temperature,
        calibrated_thresholds=calibrated_thresholds,
        device=device,
    )
    calibration_labels = prepared.labels[calibration_indices]
    partial_metrics_before = diagnosis_metrics(
        calibration_labels,
        probabilities=validation_prediction["probabilities"],
    )
    partial_metrics_after = diagnosis_metrics(
        calibration_labels,
        probabilities=calibrated_prediction["probabilities"],
    )
    calibration_acceptance = evaluate_partial_calibration_gate(
        calibration,
        partial_metrics_after,
        calibration_config.get("acceptance_gate", {}),
    )

    artifact_config: Mapping[str, str] = ordinal_config["artifacts"]
    prepared_path = _project_path(artifact_config["prepared_data"])
    preprocessor_path = _project_path(artifact_config["preprocessor"])
    save_prepared(
        prepared,
        preprocessor,
        prepared_path=prepared_path,
        preprocessor_path=preprocessor_path,
    )
    prepared_sha256 = _sha256(prepared_path)
    preprocessor_sha256 = _sha256(preprocessor_path)
    data_config_sha256 = _sha256(data_config_path)
    ordinal_config_sha256 = _sha256(ordinal_config_path)
    checkpoint_path = _project_path(artifact_config["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path = _project_path(artifact_config["calibration"])
    calibration.save(calibration_path)
    calibration_sha256 = _sha256(calibration_path)
    kappa1, kappa2 = model.thresholds()
    checkpoint_payload = {
        "schema_version": 1,
        "model_type": "OrdinalBeliefModel",
        "seed": seed,
        "input_dims": dict(prepared.input_dims),
        "model_config": dict(ordinal_config["model"]),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "thresholds": {
            "kappa1": float(kappa1.detach().cpu()),
            "kappa2": float(kappa2.detach().cpu()),
        },
        "calibration_temperature": calibration.temperature,
        "calibration_method": calibration.method,
        "calibration_kappa1": calibration.calibrated_kappa1,
        "calibration_kappa2": calibration.calibrated_kappa2,
        "calibration_sha256": calibration_sha256,
        "prepared_sha256": prepared_sha256,
        "preprocessor_sha256": preprocessor_sha256,
        "data_config_sha256": data_config_sha256,
        "ordinal_config_sha256": ordinal_config_sha256,
        "calibration_acceptance": calibration_acceptance,
        "selected_feature_sources": {
            "demographic": list(data_config["base_modality"]["source_columns"]),
            **{
                str(item["name"]): list(item["source_columns"])
                for item in data_config["acquisition_modalities"]
            },
        },
        "fit_patient_ids": list(preprocessor.fit_patient_ids),
        "split_manifest_sha256": prepared.split_manifest_sha256,
        "legacy_initialization_used": False,
        "test_used_for_training_or_selection": False,
    }
    _atomic_torch_save(checkpoint_payload, checkpoint_path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    summary = {
        "seed": seed,
        "patients": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "best_epoch": fit_summary.best_epoch,
        "validation_macro_f1": fit_summary.best_validation_macro_f1,
        "temperature": calibration.temperature,
        "calibration_method": calibration.method,
        "calibration_kappa1": calibration.calibrated_kappa1,
        "calibration_kappa2": calibration.calibrated_kappa2,
        "calibration_sha256": calibration_sha256,
        "prepared_sha256": prepared_sha256,
        "preprocessor_sha256": preprocessor_sha256,
        "data_config_sha256": data_config_sha256,
        "ordinal_config_sha256": ordinal_config_sha256,
        "calibration_state_count": len(calibration_indices),
        "validation_nll_before": calibration.nll_before,
        "validation_nll_after": calibration.nll_after,
        "validation_partial_state_calibration_before": {
            "ece": partial_metrics_before["ece"],
            "brier_score": partial_metrics_before["brier_score"],
        },
        "validation_partial_state_calibration_after": {
            "ece": partial_metrics_after["ece"],
            "brier_score": partial_metrics_after["brier_score"],
        },
        "calibration_acceptance": calibration_acceptance,
        "partial_state_calibration_acceptance_threshold_defined": True,
        "partial_state_calibration_acceptability_claimed": bool(
            calibration_acceptance["passed"]
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "split_manifest_sha256": prepared.split_manifest_sha256,
        "legacy_initialization_used": False,
        "test_used_for_training_or_selection": False,
    }
    summary_path = checkpoint_path.parent / "training_summary.json"
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if calibration_acceptance["passed"] is not True:
        raise RuntimeError(
            "Validation partial-state calibration failed the predeclared Stage-1 gate; "
            "cross-fitting is not authorized"
        )


if __name__ == "__main__":
    main()
