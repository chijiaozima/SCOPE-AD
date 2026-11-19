#!/usr/bin/env python3
"""Train compact MLP students from masked Soft EBM teacher policies."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Distil each seeded planner into an MLP using train trajectories and "
            "validation-only early stopping; test trajectories are forbidden."
        )
    )
    parser.add_argument("--config", required=True, help="Distillation YAML configuration.")
    return parser


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


def _arrays(frame: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    import numpy as np

    features = np.stack(
        [np.asarray(value, dtype=np.float32) for value in frame["state_features"]]
    )
    policy = np.stack(
        [np.asarray(value, dtype=np.float32) for value in frame["planner_policy"]]
    )
    masks = np.stack(
        [np.asarray(value, dtype=bool) for value in frame["valid_action_mask"]]
    )
    energies = np.stack(
        [np.asarray(value, dtype=np.float32) for value in frame["ebm_energies"]]
    )
    actions = np.asarray(frame["chosen_action_index"], dtype=np.int64)
    if features.ndim != 2 or policy.shape != masks.shape or policy.shape != energies.shape:
        raise ValueError("Distillation arrays have incompatible shapes")
    if policy.shape[1] != 10 or actions.shape != (len(features),):
        raise ValueError("Distillation arrays must use exactly ten actions")
    if np.any(policy[~masks] > 1e-7) or not np.allclose(policy.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Teacher policies violate the valid-action mask")
    masked_energy = np.where(masks, energies, np.inf)
    order = np.argsort(masked_energy, axis=1, kind="stable")
    preferred = order[:, 0].astype(np.int64)
    rejected = order[:, 1].astype(np.int64)
    margins = (
        masked_energy[np.arange(len(order)), rejected]
        - masked_energy[np.arange(len(order)), preferred]
    ).astype(np.float32)
    return features, policy, masks, actions, preferred, rejected, margins


def _tensor_dataset(frame: Any, torch: Any) -> Any:
    from torch.utils.data import TensorDataset

    values = _arrays(frame)
    return TensorDataset(
        torch.from_numpy(values[0]),
        torch.from_numpy(values[1]),
        torch.from_numpy(values[2]),
        torch.from_numpy(values[3]),
        torch.from_numpy(values[4]),
        torch.from_numpy(values[5]),
        torch.from_numpy(values[6]),
    )


def _loss(
    model: Any,
    batch: Sequence[Any],
    *,
    device: Any,
    temperature: float,
    hard_weight: float,
    margin_weight: float,
    objective: str = "soft_kl",
) -> tuple[Any, dict[str, float]]:
    from src.students.losses import (
        distillation_loss,
        energy_margin_preference_loss,
        hard_action_loss,
    )

    features, target, mask, action, preferred, rejected, margin = [
        tensor.to(device) for tensor in batch
    ]
    logits = model(features)
    if objective == "hard_action":
        hard = hard_action_loss(logits, action, mask)
        return hard, {
            "loss": float(hard.detach()),
            "soft_kl": 0.0,
            "hard_nll": float(hard.detach()),
            "preference": 0.0,
        }
    components = distillation_loss(
        logits,
        target,
        mask,
        target_action=action,
        hard_weight=hard_weight,
        temperature=temperature,
    )
    preference = (
        energy_margin_preference_loss(
            logits,
            preferred,
            rejected,
            energy_margin=margin,
            valid_action_mask=mask,
        )
        if margin_weight > 0
        else logits.sum() * 0.0
    )
    total = components.loss + margin_weight * preference
    return total, {
        "loss": float(total.detach()),
        "soft_kl": float(components.soft_kl.detach()),
        "hard_nll": float(components.hard_nll.detach()),
        "preference": float(preference.detach()),
    }


def _fit_hard_reactive(
    *,
    train_dataset: Any,
    validation_dataset: Any,
    input_dim: int,
    mlp_config: Mapping[str, Any],
    distillation: Mapping[str, Any],
    device: Any,
    seed: int,
    torch: Any,
) -> tuple[Any, list[dict[str, float]], int, float]:
    """Fit the direct-action classification baseline without soft teacher labels."""

    from torch import nn
    from torch.utils.data import DataLoader

    from src.students.mlp_policy import MLPPolicy

    model = MLPPolicy(
        input_dim=input_dim,
        hidden_dims=tuple(int(value) for value in mlp_config.get("hidden_dims", (256, 128))),
        dropout=float(mlp_config.get("dropout", 0.1)),
        activation=str(mlp_config.get("activation", "gelu")),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(distillation.get("learning_rate", 1e-3)),
        weight_decay=float(distillation.get("weight_decay", 0.0)),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 100_003)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(distillation.get("batch_size", 256)),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(distillation.get("batch_size", 256)),
        shuffle=False,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_validation, stale = 0, float("inf"), 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(distillation.get("max_epochs", 100)) + 1):
        model.train()
        train_total, train_seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss(
                model,
                batch,
                device=device,
                temperature=1.0,
                hard_weight=1.0,
                margin_weight=0.0,
                objective="hard_action",
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.detach()) * len(batch[0])
            train_seen += len(batch[0])
        model.eval()
        validation_total, validation_seen, agreements = 0.0, 0, 0
        with torch.inference_mode():
            for batch in validation_loader:
                loss, _ = _loss(
                    model,
                    batch,
                    device=device,
                    temperature=1.0,
                    hard_weight=1.0,
                    margin_weight=0.0,
                    objective="hard_action",
                )
                features, _, mask, action, *_ = [tensor.to(device) for tensor in batch]
                predicted = model.policy(features, mask).probabilities.argmax(dim=-1)
                agreements += int((predicted == action).sum().item())
                validation_total += float(loss) * len(batch[0])
                validation_seen += len(batch[0])
        validation_loss = validation_total / max(validation_seen, 1)
        history.append(
            {
                "epoch": float(epoch),
                "hard_nll": train_total / max(train_seen, 1),
                "validation_hard_nll": validation_loss,
                "validation_top1_agreement": agreements / max(validation_seen, 1),
            }
        )
        if validation_loss < best_validation - 1e-8:
            best_validation, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= int(distillation.get("early_stopping_patience", 15)):
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_validation


def run(config: Mapping[str, Any]) -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from scripts.cache_subset_states import (
        ArtifactGateError,
        artifact_value,
        canonical_mapping_sha256,
        load_config_file,
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
    from src.states.action_space import ACTIONS
    from src.students.mlp_policy import MLPPolicy
    from src.utils.seed import set_seed

    trajectory_path = resolve_artifact(config, artifact_value(config, "planner_trajectories"))
    checkpoint_dir = resolve_artifact(config, artifact_value(config, "mlp_checkpoint_dir"))
    reactive_configured = config.get("artifacts", {}).get("reactive_mlp_checkpoint_dir")
    reactive_checkpoint_dir = (
        resolve_artifact(config, reactive_configured)
        if reactive_configured
        else checkpoint_dir.parent / "reactive_mlp"
    )
    require_file(trajectory_path, description="planner distillation trajectories")
    trajectory_metadata = load_sidecar(
        trajectory_path,
        expected_kind="planner_distillation_trajectories",
        require_deployable=False,
        require_privileged=False,
    )
    required_gates = {
        "contains_true_label": False,
        "contains_hidden_modalities": False,
        "contains_oracle_q": False,
        "test_used_for_training_or_selection": False,
        "offline_training_only": True,
        "student_input_projection_deployable": True,
        "budget_conditioned": True,
    }
    for key, expected in required_gates.items():
        if trajectory_metadata.get(key) != expected:
            raise ArtifactGateError(
                f"Trajectory metadata gate {key} expected {expected!r}"
            )
    for digest_field in (
        "h_schema_sha256",
        "h_implementation_sha256",
        "h_input_schema_sha256",
        "h_input_vectors_sha256",
    ):
        if not trajectory_metadata.get(digest_field):
            raise ArtifactGateError(f"Trajectory artifact lacks canonical {digest_field}")
    canonical_provenance = {
        field: trajectory_metadata[field]
        for field in (
            "h_schema_sha256",
            "h_implementation_sha256",
            "h_input_schema_sha256",
            "h_input_vectors_sha256",
        )
    }
    for digest_field in (
        "source_subset_sha256",
        "source_prepared_sha256",
        "source_preprocessor_sha256",
    ):
        if not trajectory_metadata.get(digest_field):
            raise ArtifactGateError(f"Trajectory artifact lacks {digest_field} provenance")
    training_budgets = trajectory_metadata.get("training_budgets_usd")
    if not isinstance(training_budgets, list) or not training_budgets:
        raise ArtifactGateError("Trajectory artifact lacks training_budgets_usd")
    expected_planner_contract = planner_contract(
        load_config_file("configs/planner.yaml")
    )
    expected_planner_contract_sha256 = canonical_mapping_sha256(
        expected_planner_contract
    )
    if (
        trajectory_metadata.get("planner_contract") != expected_planner_contract
        or trajectory_metadata.get("planner_contract_sha256")
        != expected_planner_contract_sha256
    ):
        raise ArtifactGateError(
            "Teacher trajectories use a different resolved planner contract"
        )
    teacher_manifest = require_file(
        trajectory_metadata.get("teacher_manifest", ""),
        description="trajectory teacher planner manifest",
    )
    teacher_manifest_sha256 = sha256(teacher_manifest)
    if trajectory_metadata.get("teacher_manifest_sha256") != teacher_manifest_sha256:
        raise ArtifactGateError("Teacher trajectory planner-manifest digest mismatch")
    teacher_provenance = {
        "planner_contract": expected_planner_contract,
        "planner_contract_sha256": expected_planner_contract_sha256,
        "teacher_manifest": str(teacher_manifest),
        "teacher_manifest_sha256": teacher_manifest_sha256,
    }

    frame = pd.read_parquet(trajectory_path)
    require_columns(
        frame,
        (
            "teacher_seed",
            "patient_id",
            "split",
            "budget_usd",
            "state_features",
            "valid_action_mask",
            "ebm_energies",
            "planner_policy",
            "chosen_action_index",
        ),
        artifact=trajectory_path,
    )
    if set(frame["split"]) != {"train", "validation"}:
        raise ArtifactGateError(
            "Distillation artifact must contain train/validation and no test rows"
        )
    forbidden = {"true_label", "label", "oracle_q", "hidden_modalities"} & set(frame.columns)
    if forbidden:
        raise ArtifactGateError(f"Forbidden student data columns: {sorted(forbidden)}")
    patient_splits = frame.groupby("patient_id")["split"].nunique()
    if bool((patient_splits != 1).any()):
        raise ArtifactGateError("A patient appears in multiple distillation splits")

    distillation = config.get("distillation")
    mlp_config = config.get("mlp")
    if not isinstance(distillation, Mapping) or not isinstance(mlp_config, Mapping):
        raise ArtifactGateError("distillation and mlp mappings are required")
    objective = str(distillation.get("objective", "soft_kl"))
    if objective not in {"soft_kl", "soft_kl_with_hard", "soft_kl_with_preference"}:
        raise ArtifactGateError(f"Unsupported MLP distillation objective: {objective}")
    hard_weight = float(distillation.get("hard_action_weight", 0.0))
    margin_weight = float(distillation.get("energy_margin_weight", 0.0))
    temperature = float(distillation.get("temperature", 1.0))
    if temperature <= 0 or hard_weight < 0 or margin_weight < 0:
        raise ArtifactGateError("Distillation temperature/weights are invalid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_seeds = sorted(int(value) for value in frame["teacher_seed"].unique())
    if len(teacher_seeds) < 3:
        raise ArtifactGateError("At least three teacher seeds are required")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reactive_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_records: list[dict[str, Any]] = []
    reactive_checkpoint_records: list[dict[str, Any]] = []
    for seed in teacher_seeds:
        set_seed(seed, deterministic=True)
        seed_frame = frame.loc[frame["teacher_seed"] == seed]
        train_frame = seed_frame.loc[seed_frame["split"] == "train"].reset_index(drop=True)
        validation_frame = seed_frame.loc[
            seed_frame["split"] == "validation"
        ].reset_index(drop=True)
        if train_frame.empty or validation_frame.empty:
            raise ArtifactGateError(f"Teacher seed {seed} lacks train or validation records")
        train_dataset = _tensor_dataset(train_frame, torch)
        validation_dataset = _tensor_dataset(validation_frame, torch)
        input_dim = int(train_dataset.tensors[0].shape[1])
        if int(validation_dataset.tensors[0].shape[1]) != input_dim:
            raise ArtifactGateError("MLP train/validation feature dimensions differ")
        model = MLPPolicy(
            input_dim=input_dim,
            hidden_dims=tuple(int(value) for value in mlp_config.get("hidden_dims", (256, 128))),
            dropout=float(mlp_config.get("dropout", 0.1)),
            activation=str(mlp_config.get("activation", "gelu")),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(distillation.get("learning_rate", 1e-3)),
            weight_decay=float(distillation.get("weight_decay", 0.0)),
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(distillation.get("batch_size", 256)),
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(distillation.get("batch_size", 256)),
            shuffle=False,
        )
        best_state = copy.deepcopy(model.state_dict())
        best_epoch, best_validation, stale = 0, float("inf"), 0
        history: list[dict[str, float]] = []
        max_epochs = int(distillation.get("max_epochs", 100))
        patience = int(distillation.get("early_stopping_patience", 15))
        for epoch in range(1, max_epochs + 1):
            model.train()
            totals = {"loss": 0.0, "soft_kl": 0.0, "hard_nll": 0.0, "preference": 0.0}
            seen = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, components = _loss(
                    model,
                    batch,
                    device=device,
                    temperature=temperature,
                    hard_weight=hard_weight,
                    margin_weight=margin_weight,
                    objective=objective,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                count = len(batch[0])
                seen += count
                for key in totals:
                    totals[key] += components[key] * count
            model.eval()
            validation_total, validation_seen = 0.0, 0
            agreements, validation_actions = 0, 0
            with torch.inference_mode():
                for batch in validation_loader:
                    loss, _ = _loss(
                        model,
                        batch,
                        device=device,
                        temperature=temperature,
                        hard_weight=hard_weight,
                        margin_weight=margin_weight,
                        objective=objective,
                    )
                    count = len(batch[0])
                    validation_total += float(loss) * count
                    validation_seen += count
                    features, _, mask, action, *_ = [tensor.to(device) for tensor in batch]
                    predicted = model.policy(features, mask).probabilities.argmax(dim=-1)
                    agreements += int((predicted == action).sum().item())
                    validation_actions += count
            validation_loss = validation_total / max(validation_seen, 1)
            row = {
                "epoch": float(epoch),
                **{key: value / max(seen, 1) for key, value in totals.items()},
                "validation_loss": validation_loss,
                "validation_top1_agreement": agreements / max(validation_actions, 1),
            }
            history.append(row)
            if validation_loss < best_validation - 1e-8:
                best_validation = validation_loss
                best_epoch = epoch
                stale = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        checkpoint_path = checkpoint_dir / f"seed_{seed}.pt"
        payload = {
            "schema_version": 1,
            "kind": "mlp_distilled_policy_checkpoint",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "input_dim": input_dim,
            "hidden_dims": [int(value) for value in mlp_config.get("hidden_dims", (256, 128))],
            "dropout": float(mlp_config.get("dropout", 0.1)),
            "activation": str(mlp_config.get("activation", "gelu")),
            "actions": list(ACTIONS),
            "teacher_seed": seed,
            "objective": objective,
            "temperature": temperature,
            "hard_action_weight": hard_weight,
            "energy_margin_weight": margin_weight,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "fit_split": "train",
            "checkpoint_selection_split": "validation",
            "test_used_for_training_or_selection": False,
            "contains_true_label": False,
            "source_trajectories": str(trajectory_path),
            "source_trajectories_sha256": sha256(trajectory_path),
            "source_subset_sha256": trajectory_metadata.get("source_subset_sha256"),
            **canonical_provenance,
            "source_prepared_sha256": trajectory_metadata.get("source_prepared_sha256"),
            "source_preprocessor_sha256": trajectory_metadata.get(
                "source_preprocessor_sha256"
            ),
            "training_budgets_usd": training_budgets,
            "budget_conditioned": True,
            **teacher_provenance,
            "history": history,
        }
        _atomic_torch_save(torch, payload, checkpoint_path)
        checkpoint_records.append(
            {
                "teacher_seed": seed,
                "path": str(checkpoint_path),
                "sha256": sha256(checkpoint_path),
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation,
            }
        )
        print(
            f"seed={seed}: epoch={best_epoch}, validation_loss={best_validation:.6f}"
        )

        reactive_model, reactive_history, reactive_epoch, reactive_validation = _fit_hard_reactive(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            input_dim=input_dim,
            mlp_config=mlp_config,
            distillation=distillation,
            device=device,
            seed=seed,
            torch=torch,
        )
        reactive_path = reactive_checkpoint_dir / f"seed_{seed}.pt"
        reactive_payload = {
            "schema_version": 1,
            "kind": "reactive_mlp_policy_checkpoint",
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in reactive_model.state_dict().items()
            },
            "input_dim": input_dim,
            "hidden_dims": [int(value) for value in mlp_config.get("hidden_dims", (256, 128))],
            "dropout": float(mlp_config.get("dropout", 0.1)),
            "activation": str(mlp_config.get("activation", "gelu")),
            "actions": list(ACTIONS),
            "teacher_seed": seed,
            "objective": "hard_direct_action_classification",
            "supervision_source": "planner_argmin_action_only",
            "uses_soft_policy_target": False,
            "best_epoch": reactive_epoch,
            "best_validation_loss": reactive_validation,
            "fit_split": "train",
            "checkpoint_selection_split": "validation",
            "test_used_for_training_or_selection": False,
            "contains_true_label": False,
            "source_trajectories": str(trajectory_path),
            "source_trajectories_sha256": sha256(trajectory_path),
            "source_subset_sha256": trajectory_metadata.get("source_subset_sha256"),
            "split_manifest_sha256": trajectory_metadata.get("split_manifest_sha256"),
            **canonical_provenance,
            "source_prepared_sha256": trajectory_metadata.get("source_prepared_sha256"),
            "source_preprocessor_sha256": trajectory_metadata.get(
                "source_preprocessor_sha256"
            ),
            "training_budgets_usd": training_budgets,
            "budget_conditioned": True,
            **teacher_provenance,
            "history": reactive_history,
        }
        _atomic_torch_save(torch, reactive_payload, reactive_path)
        reactive_checkpoint_records.append(
            {
                "teacher_seed": seed,
                "path": str(reactive_path),
                "sha256": sha256(reactive_path),
                "best_epoch": reactive_epoch,
                "best_validation_loss": reactive_validation,
            }
        )
        print(
            f"reactive seed={seed}: epoch={reactive_epoch}, "
            f"validation_hard_nll={reactive_validation:.6f}"
        )

    manifest_path = checkpoint_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": "mlp_distilled_policy_manifest",
        "created_at_utc": utc_now(),
        "deployable": True,
        "privileged": False,
        "contains_true_label": False,
        "contains_hidden_modalities": False,
        "contains_oracle_q": False,
        "fit_split": "train",
        "checkpoint_selection_split": "validation",
        "held_out_split": "test",
        "test_used_for_training_or_selection": False,
        "actions": list(ACTIONS),
        "objective": objective,
        "source_trajectories": str(trajectory_path),
        "source_trajectories_sha256": sha256(trajectory_path),
        "source_subset_sha256": trajectory_metadata.get("source_subset_sha256"),
        "split_manifest_sha256": trajectory_metadata.get("split_manifest_sha256"),
        **canonical_provenance,
        "source_prepared_sha256": trajectory_metadata.get("source_prepared_sha256"),
        "source_preprocessor_sha256": trajectory_metadata.get(
            "source_preprocessor_sha256"
        ),
        "training_budgets_usd": training_budgets,
        "budget_conditioned": True,
        **teacher_provenance,
        "checkpoints": checkpoint_records,
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        metadata_path(manifest_path),
        {
            **manifest,
            "kind": "mlp_distilled_policy_manifest_metadata",
            "artifact_sha256": sha256(manifest_path),
        },
    )
    reactive_manifest_path = reactive_checkpoint_dir / "manifest.json"
    reactive_manifest = {
        "schema_version": 1,
        "kind": "reactive_mlp_policy_manifest",
        "created_at_utc": utc_now(),
        "deployable": True,
        "privileged": False,
        "contains_true_label": False,
        "contains_hidden_modalities": False,
        "contains_oracle_q": False,
        "fit_split": "train",
        "checkpoint_selection_split": "validation",
        "held_out_split": "test",
        "test_used_for_training_or_selection": False,
        "actions": list(ACTIONS),
        "objective": "hard_direct_action_classification",
        "supervision_source": "planner_argmin_action_only",
        "uses_soft_policy_target": False,
        "source_trajectories": str(trajectory_path),
        "source_trajectories_sha256": sha256(trajectory_path),
        "source_subset_sha256": trajectory_metadata.get("source_subset_sha256"),
        "split_manifest_sha256": trajectory_metadata.get("split_manifest_sha256"),
        **canonical_provenance,
        "source_prepared_sha256": trajectory_metadata.get("source_prepared_sha256"),
        "source_preprocessor_sha256": trajectory_metadata.get(
            "source_preprocessor_sha256"
        ),
        "training_budgets_usd": training_budgets,
        "budget_conditioned": True,
        **teacher_provenance,
        "checkpoints": reactive_checkpoint_records,
    }
    write_json_atomic(reactive_manifest_path, reactive_manifest)
    write_json_atomic(
        metadata_path(reactive_manifest_path),
        {
            **reactive_manifest,
            "kind": "reactive_mlp_policy_manifest_metadata",
            "artifact_sha256": sha256(reactive_manifest_path),
        },
    )
    audit = write_audit(
        config,
        "train_mlp_student",
        status="completed",
        inputs={"planner_trajectories": str(trajectory_path)},
        outputs={
            "checkpoint_dir": str(checkpoint_dir),
            "manifest": str(manifest_path),
            "reactive_checkpoint_dir": str(reactive_checkpoint_dir),
            "reactive_manifest": str(reactive_manifest_path),
        },
        gates={
            "student_inputs_deployable": True,
            "fit_split_train_only": True,
            "checkpoint_selection_validation_only": True,
            "test_excluded": True,
            "invalid_actions_masked": True,
            "minimum_three_seeds": True,
            "reactive_hard_action_baseline_trained": True,
            "budget_conditioned_training": True,
        },
        details={"teacher_seeds": teacher_seeds, "objective": objective},
    )
    print(f"Wrote {len(checkpoint_records)} MLP checkpoints to {checkpoint_dir}")
    print(
        f"Wrote {len(reactive_checkpoint_records)} reactive MLP checkpoints to "
        f"{reactive_checkpoint_dir}"
    )
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
        raise SystemExit(f"train_mlp_student gate failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
