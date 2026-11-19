#!/usr/bin/env python3
"""Materialise leakage-safe planner states from the calibrated belief cache.

This module also contains the small artifact-gate helpers shared by the other
pipeline CLIs.  It intentionally imports optional project dependencies only
after argument parsing so ``--help`` works on a bare control machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_SUFFIX = ".metadata.json"


class ArtifactGateError(RuntimeError):
    """Raised when an upstream artifact cannot prove the required contract."""


def bootstrap_project() -> Path:
    root = PROJECT_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metadata_path(artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    return Path(f"{path}{METADATA_SUFFIX}")


def require_file(path: str | Path, *, description: str) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise ArtifactGateError(f"Missing {description}: {candidate}")
    if candidate.stat().st_size == 0:
        raise ArtifactGateError(f"Empty {description}: {candidate}")
    return candidate


def require_directory(path: str | Path, *, description: str) -> Path:
    candidate = Path(path)
    if not candidate.is_dir():
        raise ArtifactGateError(f"Missing {description}: {candidate}")
    return candidate


def load_sidecar(
    artifact_path: str | Path,
    *,
    expected_kind: str | None = None,
    require_deployable: bool | None = None,
    require_privileged: bool | None = None,
) -> dict[str, Any]:
    artifact = require_file(artifact_path, description=f"artifact for {artifact_path}")
    sidecar = require_file(
        metadata_path(artifact), description=f"metadata sidecar for {artifact}"
    )
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ArtifactGateError(f"Invalid metadata sidecar: {sidecar}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactGateError(f"Metadata root must be an object: {sidecar}")
    if int(value.get("schema_version", -1)) != 1:
        raise ArtifactGateError(f"Unsupported metadata schema_version in {sidecar}")
    recorded_digest = str(value.get("artifact_sha256", ""))
    actual_digest = sha256(artifact)
    if recorded_digest != actual_digest:
        raise ArtifactGateError(
            "Artifact/metadata digest mismatch (possibly an interrupted or mixed rerun): "
            f"artifact={artifact}, recorded={recorded_digest or '<missing>'}, "
            f"actual={actual_digest}"
        )
    if expected_kind is not None and value.get("kind") != expected_kind:
        raise ArtifactGateError(
            f"Expected metadata kind={expected_kind!r}, found {value.get('kind')!r}: {sidecar}"
        )
    if require_deployable is not None and value.get("deployable") is not require_deployable:
        raise ArtifactGateError(
            f"Expected deployable={require_deployable} in metadata: {sidecar}"
        )
    if require_privileged is not None and value.get("privileged") is not require_privileged:
        raise ArtifactGateError(
            f"Expected privileged={require_privileged} in metadata: {sidecar}"
        )
    return value


def write_json_atomic(path: str | Path, value: Mapping[str, Any] | Sequence[Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def write_parquet_atomic(frame: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".parquet", dir=destination.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    """Hash a resolved configuration mapping with stable JSON ordering."""

    import hashlib

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def planner_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact scientific/training knobs that bind planner artifacts."""

    planner = config.get("planner")
    cost = config.get("cost")
    risk = config.get("ordinal_risk")
    if not isinstance(planner, Mapping) or not isinstance(cost, Mapping) or not isinstance(
        risk, Mapping
    ):
        raise ArtifactGateError("planner, cost, and ordinal_risk mappings are required")
    required_planner = (
        "horizon",
        "gamma",
        "energy_temperature",
        "bellman_mode",
        "belief_mode",
        "ranking_kl_weight",
        "target_update_tau",
        "max_acquisitions",
        "budgets_usd",
        "hidden_dims",
        "dropout",
        "batch_size",
        "max_epochs",
        "early_stopping_patience",
        "learning_rate",
        "weight_decay",
        "td_loss",
    )
    missing = [key for key in required_planner if key not in planner]
    if missing:
        raise ArtifactGateError(f"Planner contract is missing fields: {missing}")
    if not isinstance(cost.get("raw_usd"), Mapping):
        raise ArtifactGateError("Planner contract requires cost.raw_usd")
    optional_planner = {}
    if "include_availability_in_predictive_state" in planner:
        optional_planner["include_availability_in_predictive_state"] = bool(
            planner["include_availability_in_predictive_state"]
        )
    return {
        "contract_version": 1,
        "planner": {
            **{key: planner[key] for key in required_planner},
            **optional_planner,
        },
        "ordinal_risk": dict(risk),
        "cost": {
            key: cost[key]
            for key in (
                "source",
                "transform",
                "lambda_usd",
                "lambda_step",
                "lambda_invasiveness",
                "lambda_time",
                "raw_usd",
                "full_reference_usd",
            )
            if key in cost
        },
        "seeds": list(config.get("seeds", ())),
        "device": str(config.get("device", "auto")),
    }


def directory_payload_sha256(
    path: str | Path, *, exclude: Iterable[str] = ()
) -> tuple[str, dict[str, str]]:
    """Hash every regular file and relative name in a deployable checkpoint tree."""

    import hashlib

    root = require_directory(path, description="checkpoint payload directory")
    excluded = {str(value) for value in exclude}
    files = [
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.relative_to(root).as_posix() not in excluded
    ]
    if not files:
        raise ArtifactGateError(f"Checkpoint payload directory has no files: {root}")
    file_digests: dict[str, str] = {}
    tree = hashlib.sha256()
    for candidate in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix()
        digest = sha256(candidate)
        file_digests[relative] = digest
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(bytes.fromhex(digest))
    return tree.hexdigest(), file_digests


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config_file(value: str | Path) -> dict[str, Any]:
    bootstrap_project()
    from src.utils.config import load_config

    path = require_file(resolve_config_path(value), description="configuration file")
    return load_config(path)


def resolve_artifact(config: Mapping[str, Any], value: str | Path) -> Path:
    bootstrap_project()
    from src.utils.config import resolve_path

    return resolve_path(config, value)


def artifact_value(
    config: Mapping[str, Any],
    key: str,
    *,
    fallback_config: Mapping[str, Any] | None = None,
    default: str | None = None,
) -> str:
    artifacts = config.get("artifacts", {})
    if isinstance(artifacts, Mapping) and artifacts.get(key):
        return str(artifacts[key])
    if fallback_config is not None:
        fallback = fallback_config.get("artifacts", {})
        if isinstance(fallback, Mapping) and fallback.get(key):
            return str(fallback[key])
    if default is not None:
        return default
    raise ArtifactGateError(f"Configuration is missing artifacts.{key}")


def audit_path(config: Mapping[str, Any], stage: str) -> Path:
    artifacts = config.get("artifacts", {})
    configured = artifacts.get("audit_dir") if isinstance(artifacts, Mapping) else None
    directory = resolve_artifact(config, configured or "artifacts/audit")
    return directory / f"{stage}.json"


def write_audit(
    config: Mapping[str, Any],
    stage: str,
    *,
    status: str,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any] | None = None,
    gates: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "kind": "pipeline_stage_audit",
        "stage": stage,
        "status": status,
        "created_at_utc": utc_now(),
        "config_path": config.get("_config_path"),
        "inputs": dict(inputs),
        "outputs": dict(outputs or {}),
        "gates": dict(gates or {}),
        "details": dict(details or {}),
    }
    return write_json_atomic(audit_path(config, stage), payload)


def parse_vector(value: Any, *, name: str, length: int | None = None) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ArtifactGateError(f"{name} is not valid JSON") from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ArtifactGateError(f"{name} must be an array")
    result = list(value)
    if length is not None and len(result) != int(length):
        raise ArtifactGateError(f"{name} must contain {length} values; found {len(result)}")
    return result


def require_columns(frame: Any, columns: Iterable[str], *, artifact: str | Path) -> None:
    missing = sorted(set(columns) - set(str(column) for column in frame.columns))
    if missing:
        raise ArtifactGateError(f"{artifact} is missing required columns: {missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cache every valid patient/subset planner state from the calibrated, "
            "patient-level cross-fitted ordinal belief artifact."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Experiment YAML (normally configs/experiments.yaml).",
    )
    return parser


def run(config: Mapping[str, Any]) -> Path:
    bootstrap_project()
    import numpy as np
    import pandas as pd

    from src.data.preprocessing import load_prepared
    from src.data.splits import assert_split_integrity
    from src.ordinal.cache import load_belief_cache
    from src.ordinal.calibration import TemperatureCalibration
    from src.planner.costs import RAW_USD_COSTS
    from src.states.state_builder import build_state
    from src.states.subset_lattice import encode_mask, validate_lattice

    ordinal_config = load_config_file("configs/ordinal.yaml")
    planner_config = load_config_file("configs/planner.yaml")
    data_config = load_config_file("configs/data.yaml")
    crossfit_path = resolve_artifact(
        ordinal_config,
        artifact_value(config, "crossfit_cache", fallback_config=ordinal_config),
    )
    calibration_path = resolve_artifact(
        ordinal_config,
        artifact_value(config, "calibration", fallback_config=ordinal_config),
    )
    prepared_path = resolve_artifact(
        ordinal_config,
        artifact_value(config, "prepared_data", fallback_config=ordinal_config),
    )
    preprocessor_path = resolve_artifact(
        ordinal_config,
        artifact_value(config, "preprocessor", fallback_config=ordinal_config),
    )
    ordinal_checkpoint_path = resolve_artifact(
        ordinal_config,
        artifact_value(config, "checkpoint", fallback_config=ordinal_config),
    )
    output_path = resolve_artifact(
        planner_config,
        artifact_value(config, "subset_states", fallback_config=planner_config),
    )

    require_file(crossfit_path, description="cross-fitted belief cache")
    require_file(calibration_path, description="validation calibration artifact")
    require_file(prepared_path, description="prepared baseline episodes")
    require_file(preprocessor_path, description="training-fitted data preprocessor")
    require_file(ordinal_checkpoint_path, description="frozen ordinal checkpoint")
    calibration = TemperatureCalibration.load(calibration_path)
    if (
        calibration.fit_split != "validation"
        or calibration.temperature <= 0
        or calibration.method != str(ordinal_config["calibration"].get("method"))
        or calibration.calibrated_kappa1 is None
        or calibration.calibrated_kappa2 is None
    ):
        raise ArtifactGateError(
            "Calibration gate failed: ordered temperature/threshold calibration must "
            "match config and be fitted on validation only"
        )
    cache = load_belief_cache(crossfit_path, require_oof_train=True)
    records = cache["records"]
    cache_metadata = cache["metadata"]
    ordinal_config_digest = sha256(PROJECT_ROOT / "configs/ordinal.yaml")
    if cache_metadata.get("ordinal_config_sha256") != ordinal_config_digest:
        raise ArtifactGateError(
            "Crossfit cache was not produced under the current ordinal configuration"
        )
    for digest_field in (
        "h_schema_sha256",
        "h_implementation_sha256",
        "h_input_schema_sha256",
        "h_input_vectors_sha256",
    ):
        if not cache_metadata.get(digest_field):
            raise ArtifactGateError(f"Crossfit cache lacks canonical {digest_field}")
    calibration_digest = sha256(calibration_path)
    checkpoint_digest = sha256(ordinal_checkpoint_path)
    prepared_digest = sha256(prepared_path)
    preprocessor_digest = sha256(preprocessor_path)
    if cache_metadata.get("final_calibration_sha256") != calibration_digest:
        raise ArtifactGateError(
            "Crossfit cache is not bound to the supplied final calibration artifact"
        )
    if cache_metadata.get("final_checkpoint_sha256") != checkpoint_digest:
        raise ArtifactGateError(
            "Crossfit cache is not bound to the supplied frozen ordinal checkpoint"
        )
    if cache_metadata.get("final_prepared_sha256") != prepared_digest:
        raise ArtifactGateError(
            "Crossfit cache is not bound to the supplied prepared-data artifact"
        )
    if cache_metadata.get("final_preprocessor_sha256") != preprocessor_digest:
        raise ArtifactGateError(
            "Crossfit cache is not bound to the supplied fitted preprocessor artifact"
        )
    prepared = load_prepared(prepared_path)
    assert_split_integrity(prepared.patient_ids, prepared.splits)
    cache_split_digest = str(cache_metadata.get("split_manifest_sha256", ""))
    if not prepared.split_manifest_sha256 or cache_split_digest != prepared.split_manifest_sha256:
        raise ArtifactGateError(
            "Locked split manifest mismatch between prepared data and crossfit cache"
        )
    configured_counts = data_config.get("locked_split_counts")
    if not isinstance(configured_counts, Mapping):
        raise ArtifactGateError("data.locked_split_counts is required")
    expected_counts = {
        split: int(configured_counts[split]) for split in ("train", "validation", "test")
    }
    prepared_counts = {
        split: int(np.sum(np.asarray(prepared.splits, dtype=str) == split))
        for split in ("train", "validation", "test")
    }
    cache_counts = {
        split: len(
            set(
                np.asarray(records["patient_id"], dtype=str)[
                    np.asarray(records["split"], dtype=str) == split
                ]
            )
        )
        for split in ("train", "validation", "test")
    }
    if prepared_counts != expected_counts or cache_counts != expected_counts:
        raise ArtifactGateError(
            "Locked patient split counts do not match data.yaml: "
            f"expected={expected_counts}, prepared={prepared_counts}, cache={cache_counts}"
        )

    patient_split = {
        str(patient_id): str(split)
        for patient_id, split in zip(prepared.patient_ids, prepared.splits, strict=True)
    }
    cache_patients = {str(value) for value in np.asarray(records["patient_id"], dtype=str)}
    unknown_patients = sorted(cache_patients - set(patient_split))
    if unknown_patients:
        raise ArtifactGateError(
            f"Belief cache contains patients absent from prepared data: {unknown_patients[:5]}"
        )

    output_rows: list[dict[str, Any]] = []
    count = len(records["patient_id"])
    for row_index in range(count):
        patient_id = str(records["patient_id"][row_index])
        split = str(records["split"][row_index])
        if patient_split[patient_id] != split:
            raise ArtifactGateError(
                f"Split mismatch for patient {patient_id}: cache={split}, prepared={patient_split[patient_id]}"
            )
        observed = tuple(
            bool(value)
            for value in parse_vector(
                records["observed_mask"][row_index], name="observed_mask", length=7
            )
        )
        available = tuple(
            bool(value)
            for value in parse_vector(
                records["available_mask"][row_index], name="available_mask", length=7
            )
        )
        cumulative_cost = float(
            sum(
                RAW_USD_COSTS[modality]
                for modality, is_observed in zip(
                    (
                        "cognitive",
                        "apoe",
                        "blood",
                        "csf",
                        "structural_mri",
                        "amyloid_pet",
                        "tau_pet",
                    ),
                    observed,
                    strict=True,
                )
                if is_observed
            )
        )
        state = build_state(
            belief={
                "h": records["h"][row_index],
                "severity": records["z"][row_index],
                "posterior": records["probabilities"][row_index],
                "entropy": records["entropy"][row_index],
                "boundary_distances": records["boundary_distances"][row_index],
            },
            patient_id=patient_id,
            observed_mask=observed,
            available_mask=available,
            cumulative_cost=cumulative_cost,
            remaining_budget=None,
        )
        subset_code = encode_mask(observed)
        output_rows.append(
            {
                "patient_id": patient_id,
                "split": split,
                "state_id": str(records["state_id"][row_index]),
                "fold_id": int(records["fold_id"][row_index]),
                "prediction_source": str(records["prediction_source"][row_index]),
                "subset_code": int(subset_code),
                "observed_count": int(sum(observed)),
                "observed_mask": list(observed),
                "available_mask": list(available),
                "h": list(state.h),
                "severity": float(state.severity),
                "posterior": list(state.posterior),
                "entropy": float(state.entropy),
                "boundary_distances": list(state.boundary_distances),
                "cumulative_cost_usd": cumulative_cost,
                "state_features_raw": state.feature_vector(belief_mode="raw").tolist(),
                "state_features_categorical": state.feature_vector(
                    belief_mode="categorical"
                ).tolist(),
                "state_features_ordinal": state.feature_vector(
                    belief_mode="ordinal"
                ).tolist(),
            }
        )

    frame = pd.DataFrame(output_rows)
    duplicate = frame.duplicated(["patient_id", "subset_code"], keep=False)
    if bool(duplicate.any()):
        examples = frame.loc[duplicate, ["patient_id", "subset_code"]].head().to_dict("records")
        raise ArtifactGateError(f"Duplicate patient/subset states: {examples}")
    for patient_id, patient_frame in frame.groupby("patient_id", sort=False):
        availability_rows = {
            tuple(bool(value) for value in parse_vector(row, name="available_mask", length=7))
            for row in patient_frame["available_mask"]
        }
        if len(availability_rows) != 1:
            raise ArtifactGateError(f"available_mask changes within patient {patient_id}")
        available = next(iter(availability_rows))
        masks = [
            parse_vector(value, name="observed_mask", length=7)
            for value in patient_frame["observed_mask"]
        ]
        validate_lattice(masks, available)

    write_parquet_atomic(frame, output_path)
    metadata = {
        "schema_version": 1,
        "kind": "subset_states",
        "created_at_utc": utc_now(),
        "deployable": False,
        "privileged": False,
        "offline_transition_lattice": True,
        "row_input_schema_deployable": True,
        "contains_future_derived_states": True,
        "contains_raw_hidden_values": False,
        "contains_true_label": False,
        "contains_hidden_modalities": False,
        "one_baseline_episode_per_patient": True,
        "acquisition_modalities": [
            "cognitive",
            "apoe",
            "blood",
            "csf",
            "structural_mri",
            "amyloid_pet",
            "tau_pet",
        ],
        "belief_source": "patient_level_5fold_oof_train_and_frozen_calibrated_eval",
        "h_representation": cache_metadata.get("h_representation"),
        "h_dimension": cache_metadata.get("h_dimension"),
        "h_projection_seed": cache_metadata.get("h_projection_seed"),
        "h_schema_sha256": cache_metadata.get("h_schema_sha256"),
        "h_implementation_sha256": cache_metadata.get("h_implementation_sha256"),
        "h_input_schema_sha256": cache_metadata.get("h_input_schema_sha256"),
        "h_input_vectors_sha256": cache_metadata.get("h_input_vectors_sha256"),
        "h_uses_observed_values_only": cache_metadata.get("h_uses_observed_values_only"),
        "severity_semantics": cache_metadata.get("severity_semantics"),
        "boundary_distance_semantics": cache_metadata.get(
            "boundary_distance_semantics"
        ),
        "split_manifest_sha256": prepared.split_manifest_sha256,
        "split_patient_counts": prepared_counts,
        "calibration_fit_split": calibration.fit_split,
        "calibration_temperature": float(calibration.temperature),
        "calibration_method": calibration.method,
        "calibration_kappa1": float(calibration.calibrated_kappa1),
        "calibration_kappa2": float(calibration.calibrated_kappa2),
        "ordinal_config_sha256": ordinal_config_digest,
        "final_calibration_acceptance": cache_metadata.get(
            "final_calibration_acceptance"
        ),
        "fold_calibration_acceptance": cache_metadata.get(
            "fold_calibration_acceptance"
        ),
        "final_calibration_sha256": calibration_digest,
        "final_checkpoint_sha256": checkpoint_digest,
        "final_prepared_sha256": prepared_digest,
        "final_preprocessor_sha256": preprocessor_digest,
        "source_crossfit_cache": str(crossfit_path),
        "source_crossfit_sha256": sha256(crossfit_path),
        "source_ordinal_checkpoint": str(ordinal_checkpoint_path),
        "source_prepared_data": str(prepared_path),
        "source_prepared_sha256": prepared_digest,
        "source_preprocessor": str(preprocessor_path),
        "source_preprocessor_sha256": preprocessor_digest,
        "row_count": int(len(frame)),
        "patient_count": int(frame["patient_id"].nunique()),
        "split_counts": {
            str(key): int(value) for key, value in frame.groupby("split").size().items()
        },
        "artifact_sha256": sha256(output_path),
    }
    write_json_atomic(metadata_path(output_path), metadata)
    audit = write_audit(
        config,
        "cache_subset_states",
        status="completed",
        inputs={
            "crossfit_cache": str(crossfit_path),
            "calibration": str(calibration_path),
            "prepared_data": str(prepared_path),
            "preprocessor": str(preprocessor_path),
            "ordinal_checkpoint": str(ordinal_checkpoint_path),
        },
        outputs={
            "subset_states": str(output_path),
            "metadata": str(metadata_path(output_path)),
        },
        gates={
            "training_beliefs_oof": True,
            "canonical_fold_invariant_h": True,
            "calibration_validation_only": True,
            "crossfit_bound_to_final_calibration": True,
            "crossfit_bound_to_final_checkpoint": True,
            "crossfit_bound_to_final_prepared_data": True,
            "crossfit_bound_to_final_preprocessor": True,
            "split_integrity": True,
            "locked_split_manifest_match": True,
            "locked_split_counts_match": True,
            "complete_patient_lattices": True,
            "row_input_schema_deployable": True,
            "offline_transition_lattice_non_deployable": True,
        },
        details={"rows": int(len(frame)), "patients": int(frame["patient_id"].nunique())},
    )
    print(f"Wrote {len(frame):,} leakage-safe subset states to {output_path}")
    print(f"Audit: {audit}")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config_file(args.config)
        run(config)
    except (ArtifactGateError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"cache_subset_states gate failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
