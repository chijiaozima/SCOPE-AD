"""Reproduce the frozen 4090pc cohort from authorized local ADNI source files."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from preparation.dataset import MANIFEST_NAME, export_splits, sha256
from preparation.raw import check_inputs, rebuild_shards

PACKAGE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, required=True, help="Directory of the six 28May2026 source CSVs"
    )
    parser.add_argument("--dictionary", type=Path, required=True, help="ADNI DATADIC.csv")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--check-inputs",
        action="store_true",
        help="Check the seven source inputs without writing data",
    )
    destination.add_argument(
        "--output", type=Path, help="New private directory outside this package"
    )
    args = parser.parse_args()
    reference = json.loads((PACKAGE / "reference.json").read_text(encoding="utf-8"))
    raw = args.raw_dir.expanduser().resolve()
    dictionary = args.dictionary.expanduser().resolve()
    public_root = PACKAGE.parent if (PACKAGE.parent / "checkpoints").is_dir() else PACKAGE
    if any(p.is_relative_to(public_root) for p in (raw, dictionary)):
        parser.error("Keep research inputs outside the distributable directory")
    if args.check_inputs:
        check_inputs(raw, dictionary, reference)
        return
    output = args.output.expanduser().resolve()
    if output.is_relative_to(public_root):
        parser.error("Choose private input and output paths outside the distributable directory")
    if output.is_relative_to(raw) or output.exists() or args.output.expanduser().is_symlink():
        parser.error("Choose a new output directory outside the raw table directory")
    os.umask(0o077)
    output.parent.mkdir(parents=True, exist_ok=True)
    print("Rebuilding the three input shards from raw tables ...", flush=True)
    shards = rebuild_shards(raw, dictionary)
    hashes = {
        name: sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        for name, frame in shards.items()
    }
    for name, expected in reference["filtered_input_sha256"].items():
        if hashes[name] != expected:
            raise ValueError(
                f"Rebuilt {name} shard differs from the frozen export. "
                f"Got {len(shards[name])} rows, SHA-256 {hashes[name]}. Check table version, order and dictionary."
            )
    frame = pd.concat([shards[name] for name in ("train", "val", "test")], ignore_index=True)
    with TemporaryDirectory(prefix=".adni-build-", dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir(mode=0o700)
        metadata = export_splits(frame, stage, synthetic=False, source_hashes=hashes)
        if (
            metadata["files"] != reference["final_files"]
            or metadata["by_class"] != reference["by_class"]
        ):
            raise ValueError("Rebuilt seed-42 export differs from the verified 4090pc reference")
        stage.rename(output)
    print(f"Verified train/validation/test: {metadata['counts']}")
    print("Locked assignment SHA-256:", metadata["files"][MANIFEST_NAME]["sha256"])
    print(f"Private output: {output}")


if __name__ == "__main__":
    main()
