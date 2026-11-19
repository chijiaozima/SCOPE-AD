#!/usr/bin/env python3
"""Train seeded Soft EBM planners with validation-only checkpoint selection."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the ten-action energy network on train targets and select each seeded "
            "checkpoint on validation loss. Test targets are never loaded for selection."
        )
    )
    parser.add_argument("--config", required=True, help="Planner YAML configuration.")
    return parser


def _device(requested: str, torch: Any) -> Any:
    value = str(requested).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _atomic_torch_save(torch: Any, payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".pt", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _tensors(frame: Any, torch: Any) -> tuple[Any, Any, Any]:
    import numpy as np

    features = np.stack(
        [np.asarray(value, dtype=np.float32) for value in frame["state_features"]]
    )
    raw_masks = [np.asarray(value) for value in frame["valid_action_mask"]]
    if any(
        mask.dtype.kind not in "bifu"
        or not np.isfinite(mask).all()
        or not np.isin(mask, [0, 1]).all()
        for mask in raw_masks
    ):
        raise ValueError("Planner valid-action masks must contain exact finite zero/one values")
    masks = np.stack([mask.astype(bool, copy=False) for mask in raw_masks])
    parsed_targets: list[np.ndarray] = []
    for row_index, (values, mask) in enumerate(
        zip(frame["target_energies"], masks, strict=True)
    ):
        try:
            target = np.asarray(
                [np.nan if item is None else float(item) for item in values],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Planner target row {row_index} contains a non-numeric energy"
            ) from exc
        if target.shape != (10,):
            raise ValueError(
                f"Planner target row {row_index} must contain exactly ten energies"
            )
        if not np.all(np.isfinite(target[mask])):
            raise ValueError(
                f"Planner target row {row_index} contains a non-finite valid energy"
            )
        # Parquet represents null list entries as NaN.  Invalid actions are the
        # only positions allowed to be null/non-finite; restore their canonical
        # +inf sentinel before planner_loss/energy_policy sees the tensor.
        target[~mask] = np.inf
        parsed_targets.append(target.astype(np.float32))
    targets = np.stack(parsed_targets)
    if features.ndim != 2 or masks.shape != targets.shape or masks.shape[1] != 10:
        raise ValueError("Planner tensor artifact has incompatible dimensions")
    return (
        torch.from_numpy(features),
        torch.from_numpy(targets),
        torch.from_numpy(masks),
    )


def run(config: Mapping[str, Any]) -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import pandas as pd
    import torch

    from scripts.cache_subset_states import (
        ArtifactGateError,
        artifact_value,
        canonical_mapping_sha256,
        load_sidecar,
        metadata_path,
        planner_contract,
        require_columns,
        require_file,
        resolve_artifact,
        sha256,
        utc_now,
        write_audit,
        write_json_atomic,
    )
    from src.planner.ebm import EnergyNetwork
    from src.planner.trainer import PlannerTrainer
    from src.states.action_space import ACTIONS
    from src.utils.seed import set_seed

    target_path = resolve_artifact(config, artifact_value(config, "bellman_targets"))
    checkpoint_dir = resolve_artifact(config, artifact_value(config, "checkpoint_dir"))
    require_file(target_path, description="soft Bellman target artifact")
    target_metadata = load_sidecar(
        target_path,
        expected_kind="soft_bellman_targets",
        require_deployable=False,
        require_privileged=False,
    )
    if target_metadata.get("contains_true_label") is not False:
        raise ArtifactGateError("Planner targets do not prove contains_true_label=false")
    if target_metadata.get("terminal_source") != "calibrated_belief_risk":
        raise ArtifactGateError("Planner targets must use calibrated belief-risk terminals")
    for digest_field in (
        "h_schema_sha256",
        "h_implementation_sha256",
        "h_input_schema_sha256",
        "h_input_vectors_sha256",
    ):
        if not target_metadata.get(digest_field):
            raise ArtifactGateError(f"Planner targets lack canonical {digest_field}")
    for digest_field in (
        "source_subset_sha256",
        "source_prepared_sha256",
        "source_preprocessor_sha256",
    ):
        if not target_metadata.get(digest_field):
            raise ArtifactGateError(f"Planner targets lack {digest_field} provenance")
    if target_metadata.get("offline_training_only") is not True or target_metadata.get(
        "uses_hidden_outcomes_for_transitions"
    ) is not True:
        raise ArtifactGateError("Planner target artifact lacks offline-transition provenance")
    resolved_planner_contract = planner_contract(config)
    resolved_planner_contract_sha256 = canonical_mapping_sha256(
        resolved_planner_contract
    )
    if (
        target_metadata.get("planner_contract") != resolved_planner_contract
        or target_metadata.get("planner_contract_sha256")
        != resolved_planner_contract_sha256
    ):
        raise ArtifactGateError(
            "Bellman targets were generated under a different resolved planner contract"
        )
    training_budgets = target_metadata.get("training_budgets_usd")
    if target_metadata.get("budget_conditioned") is not True or not isinstance(
        training_budgets, list
    ) or not training_budgets:
        raise ArtifactGateError("Planner targets are not budget-conditioned")
    bellman_mode = str(target_metadata.get("bellman_mode", ""))
    if bellman_mode not in {"soft", "hard"}:
        raise ArtifactGateError("Planner targets lack an explicit Bellman mode")

    frame = pd.read_parquet(target_path)
    require_columns(
        frame,
        (
            "patient_id",
            "split",
            "budget_usd",
            "state_features",
            "valid_action_mask",
            "target_energies",
            "terminal_source",
            "bellman_mode",
        ),
        artifact=target_path,
    )
    if set(frame["split"]) - {"train", "validation", "test"}:
        raise ArtifactGateError("Bellman targets contain an unknown split")
    if not bool((frame["terminal_source"] == "calibrated_belief_risk").all()):
        raise ArtifactGateError("A planner target row uses a non-deployable terminal source")
    if set(frame["bellman_mode"].astype(str)) != {bellman_mode}:
        raise ArtifactGateError("Planner target rows have mixed/mismatched Bellman modes")
    patient_splits = frame.groupby("patient_id")["split"].nunique()
    if bool((patient_splits != 1).any()):
        raise ArtifactGateError("A patient appears in more than one data split")
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    if train_frame.empty or validation_frame.empty:
        raise ArtifactGateError("Both train and validation Bellman targets are required")
    # Intentionally do not construct test tensors here.
    train_tensors = _tensors(train_frame, torch)
    validation_tensors = _tensors(validation_frame, torch)
    input_dim = int(train_tensors[0].shape[1])
    if int(validation_tensors[0].shape[1]) != input_dim:
        raise ArtifactGateError("Train and validation state dimensions differ")

    planner = config.get("planner")
    if not isinstance(planner, Mapping):
        raise ArtifactGateError("planner mapping is required")
    seeds_raw = config.get("seeds", [42, 123, 2026])
    if not isinstance(seeds_raw, Sequence) or isinstance(seeds_raw, (str, bytes)):
        raise ArtifactGateError("seeds must be a sequence")
    seeds = [int(seed) for seed in seeds_raw]
    paired_seed42 = (
        isinstance(config.get("ablation"), Mapping)
        and config["ablation"].get("paired_seed42_ablation") is True
        and seeds == [42]
    )
    if len(seeds) < 3 and not paired_seed42:
        raise ArtifactGateError("At least three planner seeds are required")
    device = _device(str(config.get("device", "auto")), torch)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoints: list[dict[str, Any]] = []
    for seed in seeds:
        set_seed(seed, deterministic=True)
        model = EnergyNetwork(
            state_dim=input_dim,
            hidden_dims=tuple(int(value) for value in planner.get("hidden_dims", (256, 256, 128))),
            dropout=float(planner.get("dropout", 0.1)),
        )
        trainer = PlannerTrainer(
            model,
            temperature=float(planner.get("energy_temperature", 0.3)),
            ranking_kl_weight=float(planner.get("ranking_kl_weight", 1.0)),
            target_update_tau=float(planner.get("target_update_tau", 0.01)),
            learning_rate=float(planner.get("learning_rate", 1e-3)),
            weight_decay=float(planner.get("weight_decay", 1e-4)),
            device=device,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = trainer.fit(
            train_tensors,
            validation_tensors=validation_tensors,
            epochs=int(planner.get("max_epochs", 150)),
            batch_size=int(planner.get("batch_size", 512)),
            patience=int(planner.get("early_stopping_patience", 20)),
            generator=generator,
        )
        best_row = next(
            row for row in result.history if int(row["epoch"]) == int(result.best_epoch)
        )
        checkpoint_path = checkpoint_dir / f"seed_{seed}.pt"
        checkpoint_payload = {
            "schema_version": 1,
            "kind": "soft_ebm_planner_checkpoint",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in result.model.state_dict().items()
            },
            "input_dim": input_dim,
            "hidden_dims": [
                int(value) for value in planner.get("hidden_dims", (256, 256, 128))
            ],
            "dropout": float(planner.get("dropout", 0.1)),
            "seed": seed,
            "best_epoch": int(result.best_epoch),
            "best_validation_loss": float(best_row["validation_loss"]),
            "energy_temperature": float(planner.get("energy_temperature", 0.3)),
            "belief_mode": str(planner.get("belief_mode", "ordinal")),
            "include_availability_in_predictive_state": bool(
                planner.get("include_availability_in_predictive_state", True)
            ),
            "actions": list(ACTIONS),
            "terminal_source": "calibrated_belief_risk",
            "bellman_mode": bellman_mode,
            "contains_true_label": False,
            "fit_split": "train",
            "checkpoint_selection_split": "validation",
            "test_used_for_training_or_selection": False,
            "source_targets": str(target_path),
            "source_targets_sha256": sha256(target_path),
            "source_subset_sha256": target_metadata.get("source_subset_sha256"),
            "split_manifest_sha256": target_metadata.get("split_manifest_sha256"),
            "h_schema_sha256": target_metadata.get("h_schema_sha256"),
            "h_implementation_sha256": target_metadata.get("h_implementation_sha256"),
            "h_input_schema_sha256": target_metadata.get("h_input_schema_sha256"),
            "h_input_vectors_sha256": target_metadata.get("h_input_vectors_sha256"),
            "source_prepared_sha256": target_metadata.get("source_prepared_sha256"),
            "source_preprocessor_sha256": target_metadata.get(
                "source_preprocessor_sha256"
            ),
            "training_budgets_usd": training_budgets,
            "budget_conditioned": True,
            "planner_contract": resolved_planner_contract,
            "planner_contract_sha256": resolved_planner_contract_sha256,
            "history": result.history,
        }
        _atomic_torch_save(torch, checkpoint_payload, checkpoint_path)
        checkpoints.append(
            {
                "seed": seed,
                "path": str(checkpoint_path),
                "sha256": sha256(checkpoint_path),
                "best_epoch": int(result.best_epoch),
                "best_validation_loss": float(best_row["validation_loss"]),
            }
        )
        print(
            f"seed={seed}: epoch={result.best_epoch}, "
            f"validation_loss={float(best_row['validation_loss']):.6f}"
        )

    manifest_path = checkpoint_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": "soft_ebm_planner_checkpoint_manifest",
        "created_at_utc": utc_now(),
        "deployable": True,
        "privileged": False,
        "contains_true_label": False,
        "terminal_source": "calibrated_belief_risk",
        "bellman_mode": bellman_mode,
        "fit_split": "train",
        "checkpoint_selection_split": "validation",
        "held_out_split": "test",
        "test_used_for_training_or_selection": False,
        "state_dim": input_dim,
        "include_availability_in_predictive_state": bool(
            planner.get("include_availability_in_predictive_state", True)
        ),
        "actions": list(ACTIONS),
        "source_targets": str(target_path),
        "source_targets_sha256": sha256(target_path),
        "source_subset_sha256": target_metadata.get("source_subset_sha256"),
        "split_manifest_sha256": target_metadata.get("split_manifest_sha256"),
        "h_schema_sha256": target_metadata.get("h_schema_sha256"),
        "h_implementation_sha256": target_metadata.get("h_implementation_sha256"),
        "h_input_schema_sha256": target_metadata.get("h_input_schema_sha256"),
        "h_input_vectors_sha256": target_metadata.get("h_input_vectors_sha256"),
        "source_prepared_sha256": target_metadata.get("source_prepared_sha256"),
        "source_preprocessor_sha256": target_metadata.get(
            "source_preprocessor_sha256"
        ),
        "training_budgets_usd": training_budgets,
        "budget_conditioned": True,
        "planner_contract": resolved_planner_contract,
        "planner_contract_sha256": resolved_planner_contract_sha256,
        "final_calibration_sha256": target_metadata.get("final_calibration_sha256"),
        "ordinal_checkpoint_sha256": target_metadata.get("final_checkpoint_sha256"),
        "split_patient_counts": target_metadata.get("split_patient_counts"),
        "checkpoints": checkpoints,
    }
    write_json_atomic(manifest_path, manifest)
    # A directory cannot carry a normal artifact sidecar, so gate consumers use
    # the manifest itself; this sidecar makes the convention explicit as well.
    write_json_atomic(
        metadata_path(manifest_path),
        {
            **manifest,
            "kind": "soft_ebm_planner_checkpoint_manifest_metadata",
            "artifact_sha256": sha256(manifest_path),
        },
    )
    audit = write_audit(
        config,
        "train_ebm_planner",
        status="completed",
        inputs={"bellman_targets": str(target_path)},
        outputs={"checkpoint_dir": str(checkpoint_dir), "manifest": str(manifest_path)},
        gates={
            "fit_split_train_only": True,
            "checkpoint_selection_validation_only": True,
            "test_excluded_from_training_and_selection": True,
            "belief_risk_targets_only": True,
            "budget_conditioned_training": True,
            "minimum_three_seeds_or_explicit_paired_seed42_ablation": True,
        },
        details={
            "seeds": seeds,
            "paired_seed42_ablation": paired_seed42,
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "test_rows_ignored": int((frame["split"] == "test").sum()),
        },
    )
    print(f"Wrote {len(checkpoints)} planner checkpoints to {checkpoint_dir}")
    print(f"Audit: {audit}")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.cache_subset_states import ArtifactGateError, load_config_file

    try:
        run(load_config_file(args.config))
    except (ArtifactGateError, FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"train_ebm_planner gate failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
