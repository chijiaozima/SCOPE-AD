"""Train the five-stage ordinal and Soft EBM teacher pipeline in a fresh run directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from src.utils.config import deep_merge

ROOT = Path(__file__).resolve().parent
STAGES = (
    ("train_ordinal", "ordinal"),
    ("build_crossfit_beliefs", "ordinal"),
    ("cache_subset_states", "experiments"),
    ("build_soft_bellman_targets", "planner"),
    ("train_ebm_planner", "planner"),
)
RESEARCH_ASSIGNMENT = "b8fbd61d5ba4e4d3455958dd118108c44b4ac3d94f151f1cdb512c9fc5c528f7"
SPLIT_FILE = "split_ids_seed42.csv"


def dataset_info(directory, profile):
    metadata = json.loads((directory / "manifest.json").read_text())
    if metadata["synthetic"] is not (profile == "example"):
        raise ValueError("Select the profile that matches the Step 1 dataset type.")
    for name in ("train.csv", "val.csv", "test.csv", SPLIT_FILE):
        with (directory / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != metadata["files"][name]["sha256"]:
            raise ValueError(f"Input checksum mismatch for {name}")
    if profile == "research" and metadata["files"][SPLIT_FILE]["sha256"] != RESEARCH_ASSIGNMENT:
        raise ValueError("Research mode requires the locked 4090pc seed-42 assignment.")
    return metadata


def prepare_run(directory, data, profile, counts):
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in ("scripts", "src", "configs"):
        shutil.copytree(
            ROOT / name, directory / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    cfg_dir = directory / "configs"
    data_path = cfg_dir / "data.yaml"
    settings = yaml.safe_load(data_path.read_text())
    if profile == "research" and settings["locked_split_counts"] != counts:
        raise ValueError("Research split counts differ from the original configuration.")
    settings.update(
        input_path=str(data),
        locked_split_manifest=str(data / SPLIT_FILE),
        locked_split_counts=counts,
    )
    data_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    if profile == "example":
        overrides = yaml.safe_load((ROOT / "example.yaml").read_text())
        for name, changes in overrides.items():
            path = cfg_dir / f"{name}.yaml"
            base = yaml.safe_load(path.read_text())
            path.write_text(yaml.safe_dump(deep_merge(base, changes), sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["example", "research"], default="example")
    parser.add_argument(
        "--data", type=Path, help="Step 1 output directory containing manifest.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New run directory outside the package and dataset",
    )
    args = parser.parse_args()
    if args.profile == "research" and args.data is None:
        parser.error("--data is required for the research profile")
    data = (args.data or ROOT.parents[1] / "data_process/pseudo_dataset").expanduser().resolve()
    run_dir = args.output.expanduser().resolve()
    if (
        run_dir.is_relative_to(ROOT)
        or run_dir.is_relative_to(ROOT.parents[1] / "data_process")
        or run_dir.is_relative_to(data)
        or run_dir.exists()
    ):
        parser.error("Choose a new output directory outside the source package and dataset.")
    metadata = dataset_info(data, args.profile)
    os.umask(0o077)
    prepare_run(run_dir, data, args.profile, metadata["counts"])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if args.profile == "example":
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log_dir = run_dir / "logs"
    log_dir.mkdir()
    for stage, config in STAGES:
        log = log_dir / f"{stage}.log"
        print(f"Running {stage} ...", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(
                [sys.executable, "-B", f"scripts/{stage}.py", "--config", f"configs/{config}.yaml"],
                cwd=run_dir,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise SystemExit(f"{stage} exited with status {result.returncode}. See {log}")
    if args.profile == "example":
        subprocess.run(
            [sys.executable, "-B", str(ROOT / "verify_example.py"), str(run_dir)],
            env=env,
            check=True,
        )
    print(f"Completed all five stages. Outputs: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
