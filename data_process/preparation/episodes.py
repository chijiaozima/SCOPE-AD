"""Read split CSVs and parse observable modalities with the 4090pc data loader."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .integrity import assert_split_integrity
from .schema import (
    ACQUISITION_MODALITIES,
    ALL_MODALITIES,
    BASE_MODALITY,
    ParsedEpisodes,
    SchemaError,
    normalize_label,
)


class ArtifactIntegrityError(RuntimeError):
    """Raised before deserialization when a legacy artifact fails its manifest."""


_EXACT_OUTCOME_ALIASES = frozenset(
    {
        "target",
        "outcome",
        "class",
        "response",
        "phenotype",
        "case_control",
        "disease",
        "stage",
        "status",
    }
)


def verify_legacy_bundle_file(
    path: str | Path,
    *,
    allowed_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify exact basename, byte count, and SHA-256 against manifest.json."""

    target = Path(path).resolve()
    root = Path(allowed_root).resolve() if allowed_root is not None else target.parent
    if not target.is_relative_to(root):
        raise ArtifactIntegrityError(f"Artifact is outside the allowed legacy bundle: {target}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(f"Legacy artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_record = manifest.get("files", {}).get(target.name)
    if not isinstance(file_record, dict):
        raise ArtifactIntegrityError(
            f"Legacy manifest has no exact basename entry for {target.name!r}"
        )
    if not target.is_file():
        raise ArtifactIntegrityError(f"Legacy artifact is missing: {target}")
    expected_bytes = int(file_record.get("bytes", -1))
    actual_bytes = target.stat().st_size
    if actual_bytes != expected_bytes:
        raise ArtifactIntegrityError(
            f"Legacy artifact byte-size mismatch: expected {expected_bytes}, got {actual_bytes}"
        )
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    expected_sha256 = str(file_record.get("sha256", "")).lower()
    actual_sha256 = digest.hexdigest()
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ArtifactIntegrityError(
            "Legacy artifact SHA-256 mismatch; refusing to use an unverified file"
        )
    return {
        "path": str(target),
        "basename": target.name,
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "manifest": str(manifest_path),
    }


def _normalized_name(value: object) -> str:
    result = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_").lower()
    return result or "unnamed"


def _is_unavailable(value: object, patterns: Iterable[str]) -> bool:
    if value is None or pd.isna(value) or not str(value).strip():
        return True
    text = str(value).lower()
    return any(str(pattern).lower() in text for pattern in patterns)


def parse_key_value_text(value: object, unavailable_patterns: Iterable[str]) -> dict[str, object]:
    if _is_unavailable(value, unavailable_patterns):
        return {}
    result: dict[str, object] = {}
    for raw_line in str(value).splitlines():
        line = raw_line.strip()
        if not line.startswith("-") or ":" not in line:
            continue
        raw_key, raw_value = line[1:].split(":", 1)
        key = _normalized_name(raw_key)
        field = raw_value.strip()
        if _is_unavailable(field, unavailable_patterns) or field.lower() in {
            "nan",
            "na",
            "n/a",
            "none",
            "null",
            "unknown",
        }:
            result[key] = None
        else:
            result[key] = field
    return result


def read_baseline_tables(path: str | Path, source_splits: Mapping[str, str]) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"ADNI input path does not exist: {source}")
    if source.is_file():
        frame = pd.read_csv(source)
        if "__split" not in frame:
            raise SchemaError("A single input table must contain a __split column")
        return frame
    frames: list[pd.DataFrame] = []
    for split in ("train", "validation", "test"):
        filename = source_splits.get(split)
        if not filename:
            raise SchemaError(f"source_splits is missing {split}")
        table_path = source / filename
        if not table_path.exists():
            raise FileNotFoundError(f"Missing {split} table: {table_path}")
        frame = pd.read_csv(table_path)
        frame["__split"] = split
        frame["__source_file"] = table_path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _select_one_baseline(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    patient_column = str(config["patient_id_column"])
    if patient_column not in frame:
        raise SchemaError(f"Missing patient id column: {patient_column}")
    split_counts = frame.groupby(patient_column, dropna=False)["__split"].nunique()
    across_splits = split_counts[split_counts > 1]
    if len(across_splits):
        raise SchemaError(
            "Patient overlap exists in source train/validation/test files; examples: "
            + ", ".join(str(value) for value in across_splits.index[:5])
        )
    if not frame[patient_column].duplicated().any():
        return frame.reset_index(drop=True)
    visit_column = str(config.get("baseline_visit_column", ""))
    if not visit_column or visit_column not in frame:
        raise SchemaError("Duplicate patients require a baseline_visit_column")
    priority = {
        str(value).lower(): rank
        for rank, value in enumerate(config.get("baseline_visit_priority", ("bl", "sc")))
    }
    ranked = frame.copy()
    ranked["__baseline_rank"] = (
        ranked[visit_column].astype(str).str.lower().map(priority).fillna(len(priority)).astype(int)
    )
    ranked = ranked.sort_values([patient_column, "__baseline_rank"], kind="stable")
    return (
        ranked.drop_duplicates(patient_column, keep="first")
        .drop(columns="__baseline_rank")
        .reset_index(drop=True)
    )


def _modality_config(config: Mapping[str, Any]) -> dict[str, list[str]]:
    base = config.get("base_modality", {})
    if base.get("name") != BASE_MODALITY:
        raise SchemaError("data config must define demographic as the fixed base modality")
    groups: dict[str, list[str]] = {
        BASE_MODALITY: [str(value) for value in base.get("source_columns", ())]
    }
    acquisition = config.get("acquisition_modalities", ())
    names = tuple(str(item.get("name")) for item in acquisition)
    if names != ACQUISITION_MODALITIES:
        raise SchemaError(
            f"data config acquisition order must be {ACQUISITION_MODALITIES}; got {names}"
        )
    for item in acquisition:
        groups[str(item["name"])] = [str(value) for value in item.get("source_columns", ())]
    selected_sources = {
        _normalized_name(source) for sources in groups.values() for source in sources
    }
    if "medical_history" in selected_sources:
        raise SchemaError(
            "medical_history is excluded from model inputs because its free text may "
            "state the diagnosis"
        )
    # APOE must never enter the always-visible base feature group.
    base_sources = set(groups[BASE_MODALITY])
    apoe_sources = set(groups["apoe"])
    if not apoe_sources or base_sources & apoe_sources or "apoe_genotype" in base_sources:
        raise SchemaError("APOE must be a non-empty, separately maskable demographic component")
    return groups


def apply_locked_split_manifest(
    baseline: pd.DataFrame,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Replace source-shard membership with a complete audited patient split."""

    patient_column = str(config["patient_id_column"])
    label_column = str(config["label_column"])
    required = {"patient_id", "label", "split"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise SchemaError("Locked split manifest is missing columns: " + ", ".join(missing))
    if manifest["patient_id"].astype(str).duplicated().any():
        raise SchemaError("Locked split manifest contains duplicate patient IDs")
    raw_ids = baseline[patient_column].astype(str)
    manifest_ids = manifest["patient_id"].astype(str)
    if raw_ids.duplicated().any():
        raise SchemaError("Baseline rows must be unique before applying the locked split")
    raw_set = set(raw_ids)
    manifest_set = set(manifest_ids)
    if raw_set != manifest_set:
        missing_ids = sorted(raw_set - manifest_set)
        extra_ids = sorted(manifest_set - raw_set)
        raise SchemaError(
            "Locked split manifest does not cover raw patients one-to-one; "
            f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
        )
    manifest_copy = manifest.copy()
    manifest_copy["patient_id"] = manifest_ids
    manifest_copy["split"] = manifest_copy["split"].astype(str)
    unknown = set(manifest_copy["split"]) - {"train", "validation", "test"}
    if unknown:
        raise SchemaError(f"Locked split manifest has unknown splits: {sorted(unknown)}")
    expected_counts = {
        str(name): int(value) for name, value in config.get("locked_split_counts", {}).items()
    }
    actual_counts = manifest_copy["split"].value_counts().to_dict()
    if expected_counts and actual_counts != expected_counts:
        raise SchemaError(
            f"Locked split counts differ: expected {expected_counts}, got {actual_counts}"
        )
    manifest_by_id = manifest_copy.set_index("patient_id")
    raw_label_by_id = {
        str(patient_id): normalize_label(value, str(config.get("label_regex", "")))
        for patient_id, value in zip(raw_ids, baseline[label_column])
    }
    mismatches: list[str] = []
    for patient_id in raw_ids:
        manifest_label = normalize_label(manifest_by_id.loc[patient_id, "label"])
        if manifest_label != raw_label_by_id[patient_id]:
            mismatches.append(patient_id)
    if mismatches:
        raise SchemaError(
            "Locked split manifest labels do not match raw mapped labels; examples: "
            + ", ".join(mismatches[:5])
        )
    result = baseline.copy()
    result["__split"] = [str(manifest_by_id.loc[patient_id, "split"]) for patient_id in raw_ids]
    return result


def parse_adni_episodes(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    locked_split_manifest: pd.DataFrame | None = None,
    split_manifest_sha256: str = "",
) -> ParsedEpisodes:
    patient_column = str(config["patient_id_column"])
    label_column = str(config["label_column"])
    groups = _modality_config(config)
    required = {patient_column, label_column, "__split"}
    required.update(source for sources in groups.values() for source in sources)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SchemaError("Missing required columns: " + ", ".join(missing))
    leakage = {_normalized_name(value) for value in config.get("leakage_columns", ())}
    value_patterns = tuple(
        re.compile(str(pattern), flags=re.IGNORECASE)
        for pattern in config.get(
            "leakage_value_patterns",
            (
                r"\balzheimer(?:'s)?(?: disease)?\b",
                r"\bmild cognitive impairment\b",
                r"\bcn\b",
                r"\bmci\b",
                r"\bad\b",
                r"\bdementia\b",
                r"\bcognitively normal\b",
                r"\bnormal cognition\b",
            ),
        )
    )
    selected_sources = {source for sources in groups.values() for source in sources}
    leaked = sorted(source for source in selected_sources if _normalized_name(source) in leakage)
    if leaked:
        raise SchemaError("Leakage columns requested as features: " + ", ".join(leaked))

    baseline = _select_one_baseline(frame, config)
    if config.get("locked_split_manifest"):
        if locked_split_manifest is None:
            raise SchemaError("Configured locked_split_manifest was not provided and verified")
        baseline = apply_locked_split_manifest(baseline, config, locked_split_manifest)
    patient_ids = baseline[patient_column].astype(str).to_numpy()
    labels = np.asarray(
        [
            normalize_label(value, str(config.get("label_regex", "")))
            for value in baseline[label_column]
        ],
        dtype=np.int64,
    )
    splits = baseline["__split"].astype(str).to_numpy()
    assert_split_integrity(patient_ids, splits)
    patterns = tuple(str(value) for value in config.get("unavailable_patterns", ()))

    modality_frames: dict[str, pd.DataFrame] = {}
    availability_columns: list[np.ndarray] = []
    for modality in ALL_MODALITIES:
        records: list[dict[str, object]] = []
        available: list[bool] = []
        for _, row in baseline.iterrows():
            record: dict[str, object] = {}
            has_value = False
            for source in groups[modality]:
                parsed = parse_key_value_text(row[source], patterns)
                if parsed:
                    has_value = True
                for key, value in parsed.items():
                    normalized_key = _normalized_name(key)
                    key_is_leakage = (
                        normalized_key in leakage
                        or normalized_key in _EXACT_OUTCOME_ALIASES
                        or re.search(
                            r"(^|_)(dx|diagnosis|diagnose|diagnostic|label|"
                            r"clinical_status|disease_stage|ground_truth|true_label|"
                            r"target_label|outcome_label|class_label|response_label)(_|$)",
                            normalized_key,
                        )
                    )
                    value_is_leakage = value is not None and any(
                        pattern.search(str(value)) for pattern in value_patterns
                    )
                    if key_is_leakage or value_is_leakage:
                        raise SchemaError(
                            "Diagnosis-like key/value detected in selected deployable features: "
                            f"patient={row[patient_column]!s}, source={source!r}, key={key!r}"
                        )
                    record[f"{_normalized_name(source)}__{key}"] = value
            if modality == BASE_MODALITY:
                # The constant also gives the base encoder a valid input if all
                # raw demographic fields happen to be absent.
                record["base_present"] = 1.0
                has_value = True
            records.append(record)
            available.append(has_value)
        frame_for_modality = pd.DataFrame.from_records(records)
        if frame_for_modality.shape[1] == 0:
            frame_for_modality = pd.DataFrame(
                {"no_observed_fields": np.zeros(len(records), dtype=np.float32)}
            )
        modality_frames[modality] = frame_for_modality.reindex(
            sorted(frame_for_modality.columns), axis=1
        ).reset_index(drop=True)
        if modality != BASE_MODALITY:
            availability_columns.append(np.asarray(available, dtype=bool))
    available_mask = np.stack(availability_columns, axis=1)
    return ParsedEpisodes(
        patient_ids=patient_ids,
        labels=labels,
        splits=splits,
        modality_frames=modality_frames,
        available_mask=available_mask,
        source_rows=len(frame),
        split_manifest_sha256=split_manifest_sha256,
    )


def load_and_parse(config: Mapping[str, Any], *, config_dir: str | Path = ".") -> ParsedEpisodes:
    input_path = Path(str(config["input_path"]))
    if not input_path.is_absolute():
        input_path = Path(config_dir) / input_path
    frame = read_baseline_tables(input_path.resolve(), config["source_splits"])
    locked_manifest: pd.DataFrame | None = None
    manifest_sha256 = ""
    if config.get("locked_split_manifest"):
        manifest_path = Path(str(config["locked_split_manifest"]))
        if not manifest_path.is_absolute():
            manifest_path = Path(config_dir) / manifest_path
        verification = verify_legacy_bundle_file(manifest_path.resolve())
        manifest_sha256 = str(verification["sha256"])
        locked_manifest = pd.read_csv(manifest_path)
    return parse_adni_episodes(
        frame,
        config,
        locked_split_manifest=locked_manifest,
        split_manifest_sha256=manifest_sha256,
    )
