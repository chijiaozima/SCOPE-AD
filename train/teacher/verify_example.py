"""Check the pseudo run's OOF coverage and query a trained teacher on a validation state."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.ordinal.cache import load_belief_cache
from src.planner.ebm import EnergyNetwork
from src.states.action_space import ACTIONS


def read_json(path):
    return json.loads(path.read_text())


def example_result(run_dir):
    settings = yaml.safe_load((run_dir / "configs/data.yaml").read_text())
    data = Path(settings["input_path"])
    if read_json(data / "manifest.json")["synthetic"] is not True:
        raise ValueError("Example verification requires the synthetic Step 1 dataset.")
    artifacts = run_dir / "artifacts"
    cache = load_belief_cache(artifacts / "ordinal/crossfit_beliefs.joblib", require_oof_train=True)
    meta = cache["metadata"]
    folds = meta["fold_manifest"]
    holdouts = [p for fold in folds for p in fold["holdout_patient_ids"]]
    train_ids = pd.read_csv(data / "train.csv")["subject_id"]
    if len(folds) != 5 or len(holdouts) != len(train_ids) or set(holdouts) != set(train_ids):
        raise ValueError("OOF coverage does not match the five-fold training cohort.")
    if any(set(f["fit_patient_ids"]) & set(f["holdout_patient_ids"]) for f in folds):
        raise ValueError("An OOF patient is in the fitting subset for its own prediction.")

    planner = artifacts / "planner"
    manifest = read_json(planner / "checkpoints/manifest.json")
    target_meta = read_json(planner / "soft_bellman_targets.parquet.metadata.json")
    audit = read_json(artifacts / "audit/train_ebm_planner.json")
    if (
        audit["status"] != "completed"
        or not all(audit["gates"].values())
        or manifest["test_used_for_training_or_selection"]
        or manifest["actions"] != list(ACTIONS)
    ):
        raise ValueError("EBM training or action-space audit failed.")

    split_rows = target_meta["split_counts"]
    details = audit["details"]
    if (
        details["train_rows"] != split_rows["train"]
        or details["validation_rows"] != split_rows["validation"]
        or details["test_rows_ignored"] != split_rows["test"]
    ):
        raise ValueError("The EBM row counts do not match the target partitions.")

    # Use demographics-only evidence for one validation episode. Never query its label.
    cases = pd.read_parquet(
        planner / "soft_bellman_targets.parquet",
        filters=[("split", "==", "validation"), ("subset_code", "==", 0)],
    )
    case = cases.sort_values(["patient_id", "budget_usd"]).iloc[0]
    features = torch.tensor(case.state_features, dtype=torch.float32)
    valid = torch.tensor(case.valid_action_mask, dtype=torch.bool)
    teachers = []
    for entry in manifest["checkpoints"]:
        model_file = planner / "checkpoints" / f"seed_{entry['seed']}.pt"
        params = torch.load(model_file, map_location="cpu", weights_only=True)
        model = EnergyNetwork(params["input_dim"], params["hidden_dims"], params["dropout"])
        model.load_state_dict(params["model_state_dict"])
        model.eval()
        with torch.inference_mode():
            probs = model.policy(features, valid, temperature=params["energy_temperature"]).numpy()
        if (
            probs.shape != (10,)
            or not np.isfinite(probs).all()
            or (probs < 0).any()
            or not np.isclose(probs.sum(), 1)
            or np.any(probs[~valid.numpy()] != 0)
        ):
            raise ValueError("Teacher did not return a normalized policy over valid actions.")
        teachers.append(
            {
                "seed": entry["seed"],
                "best_epoch": entry["best_epoch"],
                "best_validation_loss": entry["best_validation_loss"],
                "top_action": ACTIONS[int(probs.argmax())],
                "probabilities": dict(zip(ACTIONS, probs.tolist(), strict=True)),
            }
        )
    if [entry["seed"] for entry in teachers] != [42, 123, 2026]:
        raise ValueError("The example expects all three planner seeds.")
    ordinal = read_json(artifacts / "ordinal/training_summary.json")
    return {
        "profile": "synthetic_cpu",
        "split_counts": manifest["split_patient_counts"],
        "oof_holdout_sizes": [len(f["holdout_patient_ids"]) for f in folds],
        "oof_unique_patients": len(set(holdouts)),
        "calibration_method": ordinal["calibration_method"],
        "final_calibration": ordinal["calibration_acceptance"],
        "fold_calibration": [f["calibration_acceptance"]["measured"] for f in folds],
        "subset_states": read_json(artifacts / "states/subset_states.parquet.metadata.json")[
            "row_count"
        ],
        "bellman_targets_by_split": target_meta["split_counts"],
        "test_rows_used_for_training_or_selection": split_rows["test"]
        - details["test_rows_ignored"],
        "state_dim": manifest["state_dim"],
        "action_count": len(ACTIONS),
        "example_validation_id": str(case.patient_id),
        "example_budget_usd": float(case.budget_usd),
        "teachers": teachers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summary = example_result(args.run.resolve())
    output = args.run / "example_result.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Verified {summary['oof_unique_patients']} OOF patients, 5 folds, 3 trained teachers.")
    print(
        f"Policy on {summary['example_validation_id']}: {summary['teachers'][0]['top_action']} (seed 42)"
    )
    print(f"Example result: {output}")


if __name__ == "__main__":
    main()
