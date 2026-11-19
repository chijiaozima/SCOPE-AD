from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import joblib
import numpy as np

from src.data.schema import (
    ACQUISITION_MODALITIES,
    ALL_MODALITIES,
    BASE_MODALITY,
    SchemaError,
)


CACHE_SCHEMA_VERSION = 1
CANONICAL_H_DIM = 64
CANONICAL_H_SEED = 20260809
CANONICAL_H_REPRESENTATION = "frozen_canonical_ordinal_input_encoder_hash_v1"
CANONICAL_SEVERITY = "posterior_expectation_p_mci_plus_2p_ad"
CANONICAL_BOUNDARIES = "absolute_logit_calibrated_cumulative_probability"
_CANONICAL_SPEC = {
    "representation": CANONICAL_H_REPRESENTATION,
    "dimension": CANONICAL_H_DIM,
    "seed": CANONICAL_H_SEED,
    "modalities": list(ALL_MODALITIES),
    "numeric_transform": "signed_log1p_then_feature_hash",
    "categorical_transform": "normalized_value_feature_hash",
    "missing_transform": "omit_missing_fields_modality_presence_retained",
    "composition": "tanh(sum_observed_modality_vectors/sqrt(modality_count))",
}
CANONICAL_H_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(_CANONICAL_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
REQUIRED_RECORD_FIELDS = (
    "patient_id",
    "split",
    "state_id",
    "fold_id",
    "prediction_source",
    "observed_mask",
    "available_mask",
    "h",
    "z",
    "probabilities",
    "entropy",
    "boundary_distances",
)
FORBIDDEN_DEPLOYABLE_FIELDS = {
    "label",
    "labels",
    "true_label",
    "true_labels",
    "hidden_modalities",
    "oracle_q",
    "oracle_energy",
}


class InSampleBeliefError(RuntimeError):
    """Raised when train beliefs are not demonstrably patient-level OOF."""


def canonical_representation_metadata(
    *,
    input_schema_sha256: str | None = None,
    input_vectors_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the immutable, auditable planner-coordinate contract."""

    metadata = {
        "h_representation": CANONICAL_H_REPRESENTATION,
        "h_dimension": CANONICAL_H_DIM,
        "h_projection_seed": CANONICAL_H_SEED,
        "h_schema_sha256": CANONICAL_H_SCHEMA_SHA256,
        "h_implementation_sha256": canonical_implementation_sha256(),
        "h_uses_observed_values_only": True,
        "fold_internal_h_written_to_cache": False,
        "severity_semantics": CANONICAL_SEVERITY,
        "boundary_distance_semantics": CANONICAL_BOUNDARIES,
    }
    supplied = (input_schema_sha256 is not None, input_vectors_sha256 is not None)
    if any(supplied) and not all(supplied):
        raise SchemaError(
            "Canonical input schema and vector digests must be supplied together"
        )
    if all(supplied):
        for field, value in (
            ("h_input_schema_sha256", input_schema_sha256),
            ("h_input_vectors_sha256", input_vectors_sha256),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None:
                raise SchemaError(f"{field} must be a lowercase SHA-256 digest")
            metadata[field] = str(value)
    return metadata


def _hashed_coordinate(token: str, *, dimension: int) -> tuple[int, float]:
    digest = hashlib.sha256(f"{CANONICAL_H_SEED}|{token}".encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big", signed=False) % int(dimension)
    sign = 1.0 if digest[8] & 1 else -1.0
    return index, sign


def _fixed_value_token(
    modality: str, key: str, value: Any
) -> tuple[str, float] | None:
    """Map one raw parsed field without fitting statistics or vocabularies."""

    if value is None or (
        isinstance(value, (float, np.floating)) and not np.isfinite(float(value))
    ):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "none", "null", "unknown"}:
        return None
    numeric_text = text.replace(",", "")
    numeric_text = re.sub(r"^[<>]=?\s*", "", numeric_text)
    try:
        numeric = float(numeric_text)
    except ValueError:
        normalized = re.sub(r"\s+", " ", text.lower())
        return f"{modality}|{key}|categorical|{normalized}", 1.0
    if not np.isfinite(numeric):
        return None
    magnitude = float(np.sign(numeric) * np.log1p(abs(numeric)))
    return f"{modality}|{key}|numeric", float(np.clip(magnitude, -12.0, 12.0))


def build_canonical_modality_vectors(
    modality_frames: Mapping[str, Any],
    *,
    dimension: int = CANONICAL_H_DIM,
) -> dict[str, np.ndarray]:
    """Hash raw modality fields into a common, data-independent coordinate system.

    This adapter is deliberately not fitted.  Fold-specific ordinal encoders may
    rotate their internal hidden spaces, while these vectors remain identical for
    the same currently observed input in every fold and in the final model.
    """

    if int(dimension) != CANONICAL_H_DIM:
        raise SchemaError(f"canonical h dimension is fixed to {CANONICAL_H_DIM}")
    if tuple(modality_frames) != ALL_MODALITIES:
        raise SchemaError(f"modality_frames must be ordered exactly as {ALL_MODALITIES}")
    patient_count = len(modality_frames[BASE_MODALITY])
    result: dict[str, np.ndarray] = {}
    for modality in ALL_MODALITIES:
        frame = modality_frames[modality]
        if len(frame) != patient_count:
            raise SchemaError(f"{modality} rows do not align with the canonical adapter")
        columns = [str(value) for value in frame.columns]
        vectors = np.zeros((patient_count, CANONICAL_H_DIM), dtype=np.float32)
        for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
            token_count = 1
            presence_index, presence_sign = _hashed_coordinate(
                f"{modality}|present", dimension=CANONICAL_H_DIM
            )
            vectors[row_index, presence_index] += presence_sign
            for key, value in zip(columns, row, strict=True):
                encoded = _fixed_value_token(modality, key, value)
                if encoded is None:
                    continue
                token, magnitude = encoded
                coordinate, sign = _hashed_coordinate(token, dimension=CANONICAL_H_DIM)
                vectors[row_index, coordinate] += np.float32(sign * magnitude)
                token_count += 1
            vectors[row_index] /= np.float32(np.sqrt(token_count))
        result[modality] = vectors
    return result


def compose_canonical_h(
    modality_vectors: Mapping[str, np.ndarray],
    patient_indices: np.ndarray,
    observed_masks: np.ndarray,
) -> np.ndarray:
    """Compose h from demographic plus only the modalities observed at this state."""

    if tuple(modality_vectors) != ALL_MODALITIES:
        raise SchemaError(f"canonical modality vectors must use order {ALL_MODALITIES}")
    indices = np.asarray(patient_indices, dtype=np.int64)
    observed = np.asarray(observed_masks, dtype=bool)
    if observed.shape != (len(indices), len(ACQUISITION_MODALITIES)):
        raise SchemaError("observed_masks do not align with canonical h rows")
    patient_count = len(modality_vectors[BASE_MODALITY])
    if np.any(indices < 0) or np.any(indices >= patient_count):
        raise SchemaError("patient_indices contains an out-of-range canonical h row")
    h = np.asarray(modality_vectors[BASE_MODALITY][indices], dtype=np.float32).copy()
    modality_count = np.ones((len(indices), 1), dtype=np.float32)
    for modality_index, modality in enumerate(ACQUISITION_MODALITIES):
        active = observed[:, modality_index]
        if np.any(active):
            h[active] += np.asarray(
                modality_vectors[modality][indices[active]], dtype=np.float32
            )
            modality_count[active] += 1.0
    return np.tanh(h / np.sqrt(modality_count)).astype(np.float32)


def canonical_implementation_sha256() -> str:
    """Hash the exact executable coordinate formulas, not only a hand-written version tag."""

    payload = {
        "spec": _CANONICAL_SPEC,
        "functions": {
            function.__name__: inspect.getsource(function)
            for function in (
                _hashed_coordinate,
                _fixed_value_token,
                build_canonical_modality_vectors,
                compose_canonical_h,
                canonical_ordinal_coordinates,
            )
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonical_input_schema_sha256(modality_frames: Mapping[str, Any]) -> str:
    """Hash the ordered raw fields consumed by the frozen canonical encoder."""

    if tuple(modality_frames) != ALL_MODALITIES:
        raise SchemaError(f"modality_frames must be ordered exactly as {ALL_MODALITIES}")
    payload = [
        {
            "modality": modality,
            "columns": [str(column) for column in modality_frames[modality].columns],
        }
        for modality in ALL_MODALITIES
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonical_modality_vectors_sha256(
    modality_vectors: Mapping[str, np.ndarray],
) -> str:
    """Hash the exact per-patient canonical encoder inputs used to compose cached h."""

    if tuple(modality_vectors) != ALL_MODALITIES:
        raise SchemaError(f"canonical modality vectors must use order {ALL_MODALITIES}")
    digest = hashlib.sha256()
    patient_count: int | None = None
    for modality in ALL_MODALITIES:
        values = np.asarray(modality_vectors[modality], dtype="<f4", order="C")
        if values.ndim != 2 or values.shape[1] != CANONICAL_H_DIM:
            raise SchemaError(
                f"Canonical vector for {modality} must have shape [patients, {CANONICAL_H_DIM}]"
            )
        if patient_count is None:
            patient_count = int(values.shape[0])
        elif values.shape[0] != patient_count:
            raise SchemaError("Canonical modality vectors have inconsistent patient counts")
        digest.update(modality.encode("utf-8") + b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def canonical_ordinal_coordinates(
    probabilities: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Create fold-invariant severity and calibrated boundary evidence."""

    posterior = np.asarray(probabilities, dtype=np.float64)
    if posterior.ndim != 2 or posterior.shape[1] != 3:
        raise SchemaError("probabilities must have shape [records, 3]")
    if not np.all(np.isfinite(posterior)) or np.any(posterior < 0):
        raise SchemaError("probabilities must be finite and non-negative")
    if not np.allclose(posterior.sum(axis=1), 1.0, atol=1e-5):
        raise SchemaError("probabilities must sum to one")
    severity = posterior[:, 1] + 2.0 * posterior[:, 2]
    cumulative = np.stack((posterior[:, 0], posterior[:, 0] + posterior[:, 1]), axis=1)
    cumulative = np.clip(cumulative, float(eps), 1.0 - float(eps))
    boundary_evidence = np.abs(np.log(cumulative / (1.0 - cumulative)))
    return severity.astype(np.float32), boundary_evidence.astype(np.float32)


def _record_count(records: Mapping[str, Any]) -> int:
    if "patient_id" not in records:
        raise SchemaError("Belief records are missing patient_id")
    patient_ids = np.asarray(records["patient_id"])
    if patient_ids.ndim != 1 or len(patient_ids) == 0:
        raise SchemaError("Belief cache must contain at least one one-dimensional patient_id")
    return len(patient_ids)


def validate_belief_records(records: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in records]
    if missing:
        raise SchemaError("Belief records are missing fields: " + ", ".join(missing))
    forbidden = FORBIDDEN_DEPLOYABLE_FIELDS & set(records)
    if forbidden:
        raise SchemaError(
            "Deployable belief cache must not store labels, hidden modalities, or oracle values: "
            + ", ".join(sorted(forbidden))
        )
    count = _record_count(records)
    for field in REQUIRED_RECORD_FIELDS:
        if np.asarray(records[field]).shape[0] != count:
            raise SchemaError(f"Belief field {field} does not align with patient_id")
    split = np.asarray(records["split"], dtype=str)
    if set(split) - {"train", "validation", "test"}:
        raise SchemaError("Belief records contain an unknown split")
    observed = np.asarray(records["observed_mask"], dtype=bool)
    available = np.asarray(records["available_mask"], dtype=bool)
    expected_mask_shape = (count, len(ACQUISITION_MODALITIES))
    if observed.shape != expected_mask_shape or available.shape != expected_mask_shape:
        raise SchemaError(f"Belief masks must have shape {expected_mask_shape}")
    if np.any(observed & ~available):
        raise SchemaError("Belief cache contains an observed-but-unavailable modality")
    probabilities = np.asarray(records["probabilities"], dtype=np.float64)
    if probabilities.shape != (count, 3):
        raise SchemaError("probabilities must have shape [records, 3]")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise SchemaError("probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise SchemaError("probabilities must sum to one")
    severity = np.asarray(records["z"], dtype=np.float64)
    if severity.shape != (count,) or not np.all(np.isfinite(severity)):
        raise SchemaError("z must contain one finite canonical severity per record")
    if np.any(severity < -1e-6) or np.any(severity > 2.0 + 1e-6):
        raise SchemaError("canonical severity must lie in [0, 2]")
    entropy = np.asarray(records["entropy"], dtype=np.float64)
    if entropy.shape != (count,) or not np.all(np.isfinite(entropy)) or np.any(entropy < 0):
        raise SchemaError("entropy must contain one finite non-negative value per record")
    distances = np.asarray(records["boundary_distances"], dtype=np.float64)
    if (
        distances.shape != (count, 2)
        or not np.all(np.isfinite(distances))
        or np.any(distances < 0)
    ):
        raise SchemaError(
            "boundary_distances must have shape [records, 2] and be finite/non-negative"
        )
    h = np.asarray(records["h"], dtype=np.float64)
    if h.ndim != 2 or h.shape != (count, CANONICAL_H_DIM):
        raise SchemaError("h must have shape [records, latent_dim]")
    if not np.all(np.isfinite(h)):
        raise SchemaError("h must be finite")


def assert_training_beliefs_are_oof(
    records: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    """Prove every train prediction came from a fold that excluded that patient."""

    if metadata.get("prediction_mode") != "patient_level_5fold_oof":
        raise InSampleBeliefError(
            "Training beliefs must use prediction_mode='patient_level_5fold_oof'; "
            "in-sample train predictions are forbidden"
        )
    if int(metadata.get("crossfit_folds", 0)) != 5:
        raise InSampleBeliefError("Training beliefs must be generated by exactly five folds")
    if metadata.get("oof_verified") is not True:
        raise InSampleBeliefError("OOF cache metadata is not verified")
    manifests = metadata.get("fold_manifest")
    if not isinstance(manifests, list) or len(manifests) != 5:
        raise InSampleBeliefError("OOF cache must include five fold manifests")
    by_fold: dict[int, tuple[set[str], set[str]]] = {}
    all_holdout: set[str] = set()
    for manifest in manifests:
        fold = int(manifest["fold"])
        if fold in by_fold or fold not in range(5):
            raise InSampleBeliefError(f"OOF cache has an invalid/duplicate fold id: {fold}")
        fit_ids = {str(value) for value in manifest["fit_patient_ids"]}
        holdout_ids = {str(value) for value in manifest["holdout_patient_ids"]}
        if not fit_ids or not holdout_ids:
            raise InSampleBeliefError(f"Fold {fold} has an empty fit or holdout patient set")
        for digest_field in ("model_sha256", "preprocessor_sha256", "calibration_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(digest_field, ""))) is None:
                raise InSampleBeliefError(
                    f"Fold {fold} does not record a valid {digest_field}"
                )
        if fit_ids & holdout_ids:
            raise InSampleBeliefError(f"Fold {fold} has overlapping fit and holdout patients")
        if all_holdout & holdout_ids:
            raise InSampleBeliefError("A patient appears in multiple OOF holdout folds")
        all_holdout.update(holdout_ids)
        by_fold[fold] = (fit_ids, holdout_ids)
    if set(by_fold) != set(range(5)):
        raise InSampleBeliefError("OOF cache fold ids must be exactly 0,1,2,3,4")
    for fold, (fit_ids, holdout_ids) in by_fold.items():
        expected_fit = all_holdout - holdout_ids
        if fit_ids != expected_fit:
            raise InSampleBeliefError(
                f"Fold {fold} fit patients do not equal the other four holdout folds"
            )

    patient_ids = np.asarray(records["patient_id"], dtype=str)
    split = np.asarray(records["split"], dtype=str)
    fold_ids = np.asarray(records["fold_id"], dtype=np.int64)
    sources = np.asarray(records["prediction_source"], dtype=str)
    train_rows = np.flatnonzero(split == "train")
    if len(train_rows) == 0:
        raise InSampleBeliefError("OOF belief cache contains no training records")
    train_patients = set(patient_ids[train_rows])
    if train_patients != all_holdout:
        missing = sorted(all_holdout - train_patients)
        extra = sorted(train_patients - all_holdout)
        raise InSampleBeliefError(
            f"OOF patient coverage mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    for row in train_rows:
        patient_id = str(patient_ids[row])
        fold = int(fold_ids[row])
        if sources[row] != "oof" or fold not in by_fold:
            raise InSampleBeliefError(
                f"Training belief for {patient_id} is not marked as a valid OOF prediction"
            )
        fit_ids, holdout_ids = by_fold[fold]
        if patient_id in fit_ids or patient_id not in holdout_ids:
            raise InSampleBeliefError(
                f"Training belief for {patient_id} was produced by an in-sample fold"
            )


def assert_evaluation_beliefs_use_final_model(
    records: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    """Prove validation/test rows use the single frozen final ordinal model."""

    patient_ids = np.asarray(records["patient_id"], dtype=str)
    splits = np.asarray(records["split"], dtype=str)
    fold_ids = np.asarray(records["fold_id"], dtype=np.int64)
    sources = np.asarray(records["prediction_source"], dtype=str)
    declared: dict[str, set[str]] = {}
    for split_name in ("validation", "test"):
        field = f"{split_name}_patient_ids"
        raw_ids = metadata.get(field)
        if not isinstance(raw_ids, list) or not raw_ids:
            raise InSampleBeliefError(f"Belief cache must declare non-empty {field}")
        values = [str(value) for value in raw_ids]
        if len(values) != len(set(values)):
            raise InSampleBeliefError(f"Belief cache {field} contains duplicate patients")
        declared[split_name] = set(values)
        rows = np.flatnonzero(splits == split_name)
        actual = set(patient_ids[rows])
        if actual != declared[split_name]:
            missing = sorted(declared[split_name] - actual)
            extra = sorted(actual - declared[split_name])
            raise InSampleBeliefError(
                f"{split_name} patient coverage mismatch; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if np.any(fold_ids[rows] != -1) or np.any(sources[rows] != "full_train_model"):
            raise InSampleBeliefError(
                f"Every {split_name} belief must use fold_id=-1 and "
                "prediction_source='full_train_model'"
            )

    train_patients = set(patient_ids[splits == "train"])
    if train_patients & declared["validation"] or train_patients & declared["test"]:
        raise InSampleBeliefError("Train patients overlap validation/test belief rows")
    if declared["validation"] & declared["test"]:
        raise InSampleBeliefError("Validation and test patients overlap in belief cache")
    for patient_id in np.unique(patient_ids):
        patient_splits = set(splits[patient_ids == patient_id])
        if len(patient_splits) != 1:
            raise InSampleBeliefError(
                f"Patient {patient_id} appears in multiple belief-cache splits"
            )


def validate_belief_cache(
    artifact: Mapping[str, Any], *, require_oof_train: bool = True
) -> None:
    if int(artifact.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise SchemaError("Unsupported belief cache schema version")
    modalities = tuple(artifact.get("acquisition_modalities", ()))
    if modalities != ACQUISITION_MODALITIES:
        raise SchemaError("Belief cache modality order is incompatible")
    records = artifact.get("records")
    metadata = artifact.get("metadata")
    if not isinstance(records, Mapping) or not isinstance(metadata, Mapping):
        raise SchemaError("Belief cache must contain records and metadata mappings")
    split_manifest_sha256 = str(metadata.get("split_manifest_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", split_manifest_sha256) is None:
        raise SchemaError("Belief cache must record the locked split manifest SHA-256")
    for digest_field in (
        "final_checkpoint_sha256",
        "final_calibration_sha256",
        "final_prepared_sha256",
        "final_preprocessor_sha256",
        "data_config_sha256",
        "ordinal_config_sha256",
        "h_input_schema_sha256",
        "h_input_vectors_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(digest_field, ""))) is None:
            raise SchemaError(f"Belief cache must record {digest_field}")
    final_acceptance = metadata.get("final_calibration_acceptance")
    fold_acceptance = metadata.get("fold_calibration_acceptance")
    if not isinstance(final_acceptance, Mapping) or final_acceptance.get("passed") is not True:
        raise SchemaError("Final ordinal calibration must pass its predeclared gate")
    final_version = final_acceptance.get("version")
    final_thresholds = final_acceptance.get("thresholds")
    final_checks = final_acceptance.get("checks")
    if (
        not isinstance(final_version, str)
        or not final_version
        or final_acceptance.get("fit_split") != "validation"
        or not isinstance(final_thresholds, Mapping)
        or not isinstance(final_checks, Mapping)
        or not final_checks
        or any(value is not True for value in final_checks.values())
    ):
        raise SchemaError("Final calibration acceptance metadata is incomplete or failed")
    if not isinstance(fold_acceptance, Mapping) or set(fold_acceptance) != {
        str(index) for index in range(5)
    }:
        raise SchemaError("Belief cache must record passed calibration gates for folds 0..4")
    for fold_id, acceptance in fold_acceptance.items():
        if (
            not isinstance(acceptance, Mapping)
            or acceptance.get("passed") is not True
            or acceptance.get("fit_split") != "validation"
            or acceptance.get("version") != final_version
            or acceptance.get("thresholds") != final_thresholds
            or not isinstance(acceptance.get("checks"), Mapping)
            or not acceptance["checks"]
            or any(value is not True for value in acceptance["checks"].values())
        ):
            raise SchemaError(
                f"Cross-fit fold {fold_id} calibration acceptance is incomplete, "
                "failed, or uses a different contract"
            )
    manifest = metadata.get("fold_manifest")
    if not isinstance(manifest, list):
        raise SchemaError("Belief cache fold_manifest is missing")
    for entry in manifest:
        if not isinstance(entry, Mapping):
            raise SchemaError("Belief cache fold_manifest entry is malformed")
        fold_key = str(entry.get("fold"))
        if entry.get("calibration_acceptance") != fold_acceptance.get(fold_key):
            raise SchemaError(
                f"Fold {fold_key} manifest calibration gate differs from cache metadata"
            )
    if metadata.get("legacy_initialization_used") is not False:
        raise SchemaError(
            "Belief cache must prove the ordinal models did not use legacy initialization"
        )
    canonical_expected = canonical_representation_metadata()
    for key, expected in canonical_expected.items():
        if metadata.get(key) != expected:
            raise SchemaError(
                f"Belief cache canonical representation field {key!r} must be "
                f"{expected!r}; found {metadata.get(key)!r}"
            )
    validate_belief_records(records)
    if require_oof_train:
        assert_training_beliefs_are_oof(records, metadata)
        assert_evaluation_beliefs_use_final_model(records, metadata)


def save_belief_cache(
    path: str | Path,
    records: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    artifact = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "acquisition_modalities": list(ACQUISITION_MODALITIES),
        "records": {key: np.asarray(value) for key, value in records.items()},
        "metadata": dict(metadata),
    }
    validate_belief_cache(artifact, require_oof_train=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    os.close(descriptor)
    try:
        joblib.dump(artifact, temporary, compress=3)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_belief_cache(
    path: str | Path, *, require_oof_train: bool = True
) -> dict[str, Any]:
    artifact = joblib.load(Path(path))
    validate_belief_cache(artifact, require_oof_train=require_oof_train)
    return artifact
