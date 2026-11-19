"""Export the final split and validate it with the same episode parser as the planner."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .episodes import load_and_parse
from .modalities import OUTPUT_COLUMNS
from .partition import stratified_patient_split
from .schema import LABEL_TO_INDEX
from .source_partition import parse_mapped_label

SPLIT_FILES = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}
MANIFEST_NAME = "split_ids_seed42.csv"
LOADER_CONFIG = Path(__file__).with_name("loader_config.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def read_episodes(directory):
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name in (*SPLIT_FILES.values(), MANIFEST_NAME):
        data = (directory / name).read_bytes()
        expected = metadata["files"][name]
        if len(data) != expected["bytes"] or sha256(data) != expected["sha256"]:
            raise ValueError(f"File verification failed for {name}")
    config = json.loads(LOADER_CONFIG.read_text(encoding="utf-8"))
    config["locked_split_counts"] = metadata["counts"]
    return load_and_parse(config, config_dir=directory)


def export_splits(frame, directory, *, synthetic, source_hashes=None):
    if list(frame.columns) != OUTPUT_COLUMNS or not frame["visit"].eq("bl").all():
        raise ValueError("Expected the 13 baseline export columns and baseline visits")
    ids = frame["subject_id"].astype(str).to_numpy()
    if (
        frame["subject_id"].isna().any()
        or any(not item.strip() for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Expected one non-empty subject ID per row")
    labels = frame["diagnose"].map(parse_mapped_label)
    if set(labels) != set(LABEL_TO_INDEX):
        raise ValueError("Expected CN, MCI and AD baseline labels")
    indices = stratified_patient_split(
        ids,
        labels.map(LABEL_TO_INDEX).to_numpy(),
        42,
        {"train": 0.70, "validation": 0.15, "test": 0.15},
    )

    membership = []
    counts = {}
    by_class = {}
    for split, positions in indices.items():
        frame.iloc[positions].to_csv(
            directory / SPLIT_FILES[split], index=False, lineterminator="\n"
        )
        split_labels = labels.iloc[positions]
        membership.append(
            pd.DataFrame(
                {"patient_id": ids[positions], "label": split_labels.to_numpy(), "split": split}
            )
        )
        counts[split] = len(positions)
        by_class[split] = {label: int((split_labels == label).sum()) for label in LABEL_TO_INDEX}

    manifest = pd.concat(membership, ignore_index=True)
    manifest.to_csv(directory / MANIFEST_NAME, index=False, lineterminator="\n")
    files = {}
    for name in (*SPLIT_FILES.values(), MANIFEST_NAME):
        data = (directory / name).read_bytes()
        files[name] = {"bytes": len(data), "sha256": sha256(data)}
    metadata = {
        "synthetic": synthetic,
        "seed": 42,
        "counts": counts,
        "by_class": by_class,
        "files": files,
    }
    if source_hashes is not None:
        metadata["filtered_input_sha256"] = source_hashes
    write_json(directory / "manifest.json", metadata)

    # Check the actual CSVs and their split labels against the locked patient list.
    episodes = read_episodes(directory)
    if not (
        np.array_equal(episodes.patient_ids, manifest["patient_id"])
        and np.array_equal(episodes.splits, manifest["split"])
        and np.array_equal(episodes.labels, manifest["label"].map(LABEL_TO_INDEX))
    ):
        raise ValueError("Exported assignments or labels differ from the locked membership")
    return metadata
