"""Check a synthetic distillation run and query each student type on one validation state."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_coverage(frame, run, meta):
    cfg = yaml.safe_load((run / "configs/data.yaml").read_text())
    data = Path(cfg["input_path"])
    assignment = Path(cfg["locked_split_manifest"])
    dataset = read_json(data / "manifest.json")
    if dataset["synthetic"] is not True or not (
        sha256(assignment)
        == dataset["files"][assignment.name]["sha256"]
        == meta["split_manifest_sha256"]
    ):
        raise ValueError("This check requires the synthetic split used by the teacher.")
    patients = pd.read_csv(assignment, dtype={"patient_id": str})
    expected = patients.loc[patients.split.isin(["train", "validation"]), ["patient_id", "split"]]
    expected_pairs = set(expected.itertuples(index=False, name=None))
    if sorted(frame.teacher_seed.unique()) != [42, 123, 2026]:
        raise ValueError("Expected three seeded teachers.")
    budgets = meta["training_budgets_usd"]
    for _, seeded in frame.groupby("teacher_seed"):
        if sorted(seeded.budget_usd.unique()) != sorted(budgets):
            raise ValueError("Missing teacher budget.")
        for _, group in seeded.groupby("budget_usd"):
            pairs = set(group[["patient_id", "split"]].itertuples(index=False, name=None))
            terminals = group.groupby("patient_id").terminal.sum()
            if pairs != expected_pairs or not terminals.eq(1).all():
                raise ValueError("Teacher trajectories do not cover exactly the expected episodes.")
    return expected.split.value_counts().to_dict()


def probe(kind, directory, case, info, prompt_chars):
    from src.students.mlp_policy import MLPPolicy

    row = case.iloc[0]
    valid = torch.tensor(np.array(row.valid_action_mask), dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        if kind == "qwen":
            from scripts.train_llm_student import _collator, _records
            from src.students.llm_action_policy import load_saved_policy

            model, tokenizer = load_saved_policy(
                info["base_model_snapshot"],
                directory / "seed_42",
                device_map=info["training_device_map"],
                torch_dtype=info["torch_dtype"],
            )
            batch = _collator(tokenizer, info["max_length"], torch)(
                _records(case, prompt_max_chars=prompt_chars)
            )
            device = model.backbone.get_input_embeddings().weight.device
            model.eval()
            probs = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device), valid.to(device)
            )["probabilities"]
        else:
            weights = torch.load(directory / "seed_42.pt", map_location="cpu", weights_only=True)
            model = MLPPolicy.from_config(weights)
            model.load_state_dict(weights["model_state_dict"])
            model.eval()
            features = torch.tensor(np.array(row.state_features), dtype=torch.float32).unsqueeze(0)
            probs = model.policy(features, valid).probabilities
    probs = probs.detach().float().cpu().numpy().flatten()
    if (
        probs.shape != (10,)
        or not np.isfinite(probs).all()
        or (probs < 0).any()
        or np.any(probs[~valid.numpy().flatten()] != 0)
        or not np.isclose(probs.sum(), 1, atol=1e-5)
    ):
        raise ValueError(f"Invalid student policy for {kind}.")
    return {"action": info["actions"][int(probs.argmax())], "probability": float(probs.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--students", choices=("mlp", "qwen", "all"), default="mlp")
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run))
    from src.states.action_space import ACTIONS

    base = run / "artifacts"
    trajectory = base / "trajectories/planner_records.parquet"
    meta = read_json(trajectory.with_suffix(".parquet.metadata.json"))
    source_digest = sha256(trajectory)
    teacher_digest = sha256(Path(meta["teacher_manifest"]))
    if (
        meta["artifact_sha256"] != source_digest
        or meta["teacher_manifest_sha256"] != teacher_digest
    ):
        raise ValueError("Trajectory lineage mismatch.")
    frame = pd.read_parquet(trajectory)
    counts = check_coverage(frame, run, meta)
    case = (
        frame.loc[
            (frame.split == "validation")
            & (frame.teacher_seed == 42)
            & (frame.trajectory_step == 0)
        ]
        .sort_values(["patient_id", "budget_usd"])
        .iloc[:1]
    )
    branches = {
        "mlp": ["mlp", "reactive_mlp"],
        "qwen": ["qwen"],
        "all": ["mlp", "reactive_mlp", "qwen"],
    }
    results = {}
    for kind in branches[args.students]:
        directory = base / "students" / kind
        info = read_json(directory / "manifest.json")
        if (
            info["source_trajectories_sha256"] != source_digest
            or info["teacher_manifest_sha256"] != teacher_digest
            or info["fit_split"] != "train"
            or info["checkpoint_selection_split"] != "validation"
            or info["test_used_for_training_or_selection"] is not False
            or info["actions"] != list(ACTIONS)
            or sorted(x["teacher_seed"] for x in info["checkpoints"]) != [42, 123, 2026]
        ):
            raise ValueError(f"Invalid student run or teacher lineage for {kind}.")
        seeds = []
        for entry in info["checkpoints"]:
            loss = float(entry["best_validation_loss"])
            if entry["best_epoch"] < 1 or not np.isfinite(loss):
                raise ValueError(f"Invalid validation selection for {kind}.")
            seeds.append(
                {
                    "seed": entry["teacher_seed"],
                    "best_epoch": entry["best_epoch"],
                    "validation_loss": loss,
                }
            )
        results[kind] = {
            "objective": info["objective"],
            "seeds": seeds,
            "seed42_prediction": probe(kind, directory, case, info, meta["prompt_max_chars"]),
        }
    result = {
        "dataset": "Step 1 synthetic",
        "patients": counts,
        "trajectory_rows": len(frame),
        "rows_by_split": frame.split.value_counts().to_dict(),
        "test_rows": int((frame.split == "test").sum()),
        "training_budgets_usd": meta["training_budgets_usd"],
        "teacher_manifest_sha256": teacher_digest,
        "validation_example": {
            "patient": str(case.iloc[0].patient_id),
            "budget_usd": float(case.iloc[0].budget_usd),
            "step": 0,
            "teacher_seed": 42,
            "teacher_action": str(case.iloc[0].chosen_action),
        },
        "students": results,
    }
    output = run / "example_result.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(frame)} teacher records and {', '.join(results)}. Result: {output}")


if __name__ == "__main__":
    main()
