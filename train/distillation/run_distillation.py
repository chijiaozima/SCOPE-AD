"""Distill the EBM teachers from a completed Step 2 run into action policies."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parent
STEP2 = ROOT.parent / "teacher"
sys.path.insert(0, str(STEP2))
from src.utils.config import deep_merge


def read_yaml(path):
    return yaml.safe_load(path.read_text())


def save_yaml(path, config):
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def check_teacher(teacher, profile):
    data = read_yaml(teacher / "configs/data.yaml")
    dataset = Path(data["input_path"]).resolve()
    metadata = json.loads((dataset / "manifest.json").read_text())
    if metadata["synthetic"] is not (profile == "example"):
        raise ValueError("Choose the profile matching the Step 1 dataset used by this teacher.")
    split = Path(data["locked_split_manifest"])
    assignment = hashlib.sha256(split.read_bytes()).hexdigest()
    if assignment != metadata["files"]["split_ids_seed42.csv"]["sha256"]:
        raise ValueError("Step 1 assignment checksum mismatch.")
    planner = read_yaml(teacher / "configs/planner.yaml")
    models = teacher / planner["artifacts"]["checkpoint_dir"]
    manifest = json.loads((models / "manifest.json").read_text())
    if (
        manifest["split_manifest_sha256"] != assignment
        or manifest["split_patient_counts"] != metadata["counts"]
        or sorted(item["seed"] for item in manifest["checkpoints"]) != [42, 123, 2026]
    ):
        raise ValueError("Teacher split or seeds do not match the expected Step 2 run.")
    for item in manifest["checkpoints"]:
        expected = (models / f"seed_{item['seed']}.pt").resolve()
        if Path(item["path"]).resolve() != expected or not expected.is_file():
            raise ValueError("Use the intact Step 2 run at its original path.")
    return dataset


def stage_sources(teacher, out, profile):
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("src", "scripts"):
        shutil.copytree(STEP2 / name, out / name, ignore=ignore)
        shutil.copytree(ROOT / name, out / name, dirs_exist_ok=True, ignore=ignore)
    configs = out / "configs"
    configs.mkdir()
    for name in ("data.yaml", "costs.yaml"):
        shutil.copyfile(teacher / "configs" / name, configs / name)
    for name, inputs in [
        ("ordinal", ("prepared_data", "preprocessor")),
        ("planner", ("subset_states", "checkpoint_dir")),
    ]:
        config = read_yaml(teacher / f"configs/{name}.yaml")
        for key in inputs:
            config["artifacts"][key] = str((teacher / config["artifacts"][key]).resolve())
        save_yaml(configs / f"{name}.yaml", config)
    student = read_yaml(ROOT / "configs/distillation.yaml")
    # The historical main run completed Qwen soft KL. The implicit hard run crashed.
    student["llm"]["train_objectives"] = ["soft_kl"]
    if profile == "example":
        student = deep_merge(student, read_yaml(ROOT / "example.yaml"))
    save_yaml(configs / "distillation.yaml", student)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("example", "research"), default="example")
    parser.add_argument(
        "--students",
        choices=("mlp", "qwen", "all"),
        help="Default: mlp for the CPU example, all for research.",
    )
    args = parser.parse_args()
    students = args.students or ("mlp" if args.profile == "example" else "all")
    teacher = args.teacher_run.expanduser().resolve()
    out = args.output.expanduser().resolve()
    dataset = check_teacher(teacher, args.profile)
    if any(
        out.is_relative_to(path)
        for path in (teacher, dataset, ROOT, STEP2, ROOT.parents[1] / "data_process")
    ):
        parser.error("Choose a new output directory outside the inputs and source packages.")
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    stage_sources(teacher, out, args.profile)
    logs = out / "logs"
    logs.mkdir()
    stages = [("generate_planner_trajectories", "planner")]
    if students in ("mlp", "all"):
        stages.append(("train_mlp_student", "distillation"))
    if students in ("qwen", "all"):
        stages.append(("train_llm_student", "distillation"))
    for stage, config in stages:
        log = logs / f"{stage}.log"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        if args.profile == "example":
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            if stage == "train_mlp_student":
                env["CUDA_VISIBLE_DEVICES"] = ""
        print(f"Running {stage}. Log: {log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(
                [sys.executable, "-B", f"scripts/{stage}.py", "--config", f"configs/{config}.yaml"],
                cwd=out,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise SystemExit(f"{stage} exited with status {result.returncode}. See {log}")
    if args.profile == "example":
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "verify_example.py"),
                str(out),
                "--students",
                students,
            ],
            check=True,
        )
    print(f"Completed distillation in {out}", flush=True)


if __name__ == "__main__":
    main()
