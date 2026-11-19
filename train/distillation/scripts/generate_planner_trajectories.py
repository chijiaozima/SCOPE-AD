#!/usr/bin/env python3
"""Generate leakage-safe receding-horizon teacher records for distillation."""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run each seeded Soft EBM teacher on train/validation patients only and save "
            "state, valid mask, energy, soft policy, chosen action, and transition records."
        )
    )
    parser.add_argument("--config", required=True, help="Planner YAML configuration.")
    return parser


def _state_from_row(row: Any) -> Any:
    from scripts.build_soft_bellman_targets import _state_from_row

    return _state_from_row(row)


def _load_manifest(path: Path) -> dict[str, Any]:
    from scripts.cache_subset_states import ArtifactGateError, load_sidecar, require_file

    require_file(path, description="planner checkpoint manifest")
    load_sidecar(
        path,
        expected_kind="soft_ebm_planner_checkpoint_manifest_metadata",
        require_deployable=True,
        require_privileged=False,
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactGateError(f"Invalid planner checkpoint manifest: {path}") from exc
    if value.get("kind") != "soft_ebm_planner_checkpoint_manifest":
        raise ArtifactGateError(f"Unexpected planner manifest kind: {value.get('kind')!r}")
    required = {
        "deployable": True,
        "privileged": False,
        "contains_true_label": False,
        "checkpoint_selection_split": "validation",
        "test_used_for_training_or_selection": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ArtifactGateError(
                f"Planner manifest gate {key} expected {expected!r}, found {value.get(key)!r}"
            )
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ArtifactGateError("Planner manifest contains no checkpoints")
    return value


def _observed_information(prepared: Any, patient_index: int, observed_mask: Sequence[bool]) -> dict[str, Any]:
    from src.states.action_space import MODALITIES

    information: dict[str, Any] = {
        "demographic": {
            "preprocessed_features": prepared.features["demographic"][patient_index].tolist()
        }
    }
    for modality, observed in zip(MODALITIES, observed_mask, strict=True):
        if observed:
            information[modality] = {
                "preprocessed_features": prepared.features[modality][patient_index].tolist()
            }
    return information


def run(config: Mapping[str, Any]) -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import numpy as np
    import pandas as pd
    import torch

    from scripts.cache_subset_states import (
        ArtifactGateError,
        artifact_value,
        canonical_mapping_sha256,
        load_config_file,
        load_sidecar,
        metadata_path,
        parse_vector,
        planner_contract,
        require_columns,
        require_file,
        resolve_artifact,
        sha256,
        utc_now,
        write_audit,
        write_json_atomic,
        write_parquet_atomic,
    )
    from src.data.preprocessing import load_prepared
    from src.planner.costs import RAW_USD_COSTS
    from src.planner.ebm import EnergyNetwork
    from src.planner.soft_value import energy_policy
    from src.states.action_space import ACTIONS, MODALITIES, valid_action_mask
    from src.states.subset_lattice import encode_mask
    from src.students.prompts import assert_no_forbidden_fields, build_structured_prompt

    distillation_config = (
        config
        if isinstance(config.get("llm"), Mapping)
        else load_config_file("configs/distillation.yaml")
    )
    ordinal_config = load_config_file("configs/ordinal.yaml")
    subset_path = resolve_artifact(config, artifact_value(config, "subset_states"))
    checkpoint_dir = resolve_artifact(config, artifact_value(config, "checkpoint_dir"))
    manifest_path = checkpoint_dir / "manifest.json"
    output_path = resolve_artifact(
        config,
        artifact_value(
            config,
            "planner_trajectories",
            fallback_config=distillation_config,
        ),
    )
    prepared_path = resolve_artifact(
        ordinal_config,
        artifact_value(config, "prepared_data", fallback_config=ordinal_config),
    )
    preprocessor_path = resolve_artifact(
        ordinal_config,
        artifact_value(config, "preprocessor", fallback_config=ordinal_config),
    )
    require_file(subset_path, description="subset states")
    require_file(prepared_path, description="prepared episodes")
    require_file(preprocessor_path, description="training-fitted data preprocessor")
    subset_metadata = load_sidecar(
        subset_path,
        expected_kind="subset_states",
        require_deployable=False,
        require_privileged=False,
    )
    for key, expected in {
        "offline_transition_lattice": True,
        "row_input_schema_deployable": True,
        "contains_future_derived_states": True,
        "contains_raw_hidden_values": False,
    }.items():
        if subset_metadata.get(key) is not expected:
            raise ArtifactGateError(f"Subset lattice metadata gate {key} failed")
    if subset_metadata.get("contains_true_label") is not False:
        raise ArtifactGateError("Subset lattice does not prove label-free row inputs")
    prepared_digest = sha256(prepared_path)
    preprocessor_digest = sha256(preprocessor_path)
    if subset_metadata.get("source_prepared_sha256") != prepared_digest:
        raise ArtifactGateError("Prepared data do not match subset-state provenance")
    if subset_metadata.get("source_preprocessor_sha256") != preprocessor_digest:
        raise ArtifactGateError("Preprocessor does not match subset-state provenance")
    manifest = _load_manifest(manifest_path)
    resolved_planner_contract = planner_contract(config)
    resolved_planner_contract_sha256 = canonical_mapping_sha256(
        resolved_planner_contract
    )
    if (
        manifest.get("planner_contract") != resolved_planner_contract
        or manifest.get("planner_contract_sha256")
        != resolved_planner_contract_sha256
    ):
        raise ArtifactGateError(
            "Planner checkpoint was produced under a different resolved planner contract"
        )
    subset_digest = sha256(subset_path)
    for digest_field in (
        "h_schema_sha256",
        "h_implementation_sha256",
        "h_input_schema_sha256",
        "h_input_vectors_sha256",
    ):
        expected = subset_metadata.get(digest_field)
        if not expected or manifest.get(digest_field) != expected:
            raise ArtifactGateError(
                f"Planner checkpoint and subset states have mismatched {digest_field}"
            )
    if manifest.get("source_subset_sha256") != subset_digest:
        raise ArtifactGateError("Planner checkpoint was trained on different subset states")
    for field, expected in (
        ("source_prepared_sha256", prepared_digest),
        ("source_preprocessor_sha256", preprocessor_digest),
    ):
        if manifest.get(field) != expected:
            raise ArtifactGateError(f"Planner checkpoint has mismatched {field}")
    training_budgets_raw = manifest.get("training_budgets_usd")
    if manifest.get("budget_conditioned") is not True or not isinstance(
        training_budgets_raw, list
    ) or not training_budgets_raw:
        raise ArtifactGateError("Planner manifest is not budget-conditioned")
    training_budgets = [float(value) for value in training_budgets_raw]

    frame = pd.read_parquet(subset_path)
    require_columns(
        frame,
        (
            "patient_id",
            "split",
            "subset_code",
            "observed_mask",
            "available_mask",
            "h",
            "severity",
            "posterior",
            "entropy",
            "boundary_distances",
            "cumulative_cost_usd",
        ),
        artifact=subset_path,
    )
    teacher_frame = frame.loc[frame["split"].isin(["train", "validation"])].copy()
    if teacher_frame.empty or set(teacher_frame["split"]) != {"train", "validation"}:
        raise ArtifactGateError("Teacher trajectories require both train and validation patients")
    if bool((teacher_frame["split"] == "test").any()):
        raise AssertionError("Test rows entered teacher trajectory generation")

    prepared = load_prepared(prepared_path)
    prepared_index = {
        str(patient_id): index for index, patient_id in enumerate(prepared.patient_ids)
    }
    missing = sorted(set(teacher_frame["patient_id"].astype(str)) - set(prepared_index))
    if missing:
        raise ArtifactGateError(f"Prepared features are missing patients: {missing[:5]}")

    planner = config.get("planner")
    if not isinstance(planner, Mapping):
        raise ArtifactGateError("planner mapping is required")
    temperature = float(planner.get("energy_temperature", 0.3))
    max_acquisitions = int(planner.get("max_acquisitions", 7))
    if max_acquisitions < 0 or max_acquisitions > 7:
        raise ArtifactGateError("planner.max_acquisitions must be between zero and seven")
    llm_config = distillation_config.get("llm", {})
    if not isinstance(llm_config, Mapping):
        raise ArtifactGateError("distillation llm mapping is required for safe prompt sizing")
    llm_max_length = int(llm_config.get("max_length", 768))
    prompt_max_chars = min(3000, llm_max_length * 4)
    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

    rows: list[dict[str, Any]] = []
    for checkpoint_entry in manifest["checkpoints"]:
        checkpoint_path = require_file(
            checkpoint_entry["path"], description="seeded planner checkpoint"
        )
        if checkpoint_entry.get("sha256") != sha256(checkpoint_path):
            raise ArtifactGateError(f"Planner checkpoint digest mismatch: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("contains_true_label") is not False or checkpoint.get(
            "test_used_for_training_or_selection"
        ) is not False:
            raise ArtifactGateError(f"Unsafe planner checkpoint metadata: {checkpoint_path}")
        if checkpoint.get("bellman_mode") != manifest.get("bellman_mode"):
            raise ArtifactGateError(f"Planner checkpoint Bellman mode mismatch: {checkpoint_path}")
        model = EnergyNetwork(
            state_dim=int(checkpoint["input_dim"]),
            hidden_dims=tuple(int(value) for value in checkpoint["hidden_dims"]),
            dropout=float(checkpoint["dropout"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        belief_mode = str(checkpoint.get("belief_mode", "ordinal"))
        seed = int(checkpoint["seed"])

        patient_groups = list(teacher_frame.groupby("patient_id", sort=True))
        for budget, (patient_id, patient_frame) in itertools.product(
            training_budgets, patient_groups
        ):
            patient_id = str(patient_id)
            split_values = set(str(value) for value in patient_frame["split"])
            if len(split_values) != 1:
                raise ArtifactGateError(f"Patient {patient_id} appears in multiple splits")
            split = next(iter(split_values))
            lattice = {
                int(row["subset_code"]): row for _, row in patient_frame.iterrows()
            }
            if 0 not in lattice:
                raise ArtifactGateError(f"Patient {patient_id} has no base-only subset")
            subset_code = 0
            history: tuple[str, ...] = ()
            terminated = False
            patient_index = prepared_index[patient_id]
            for step_index in range(max_acquisitions + 1):
                row = lattice[subset_code]
                base_state = _state_from_row(row)
                state = dataclasses.replace(
                    base_state,
                    action_history=history,
                    remaining_budget=max(0.0, budget - base_state.cumulative_cost),
                )
                valid = valid_action_mask(
                    state.observed_mask,
                    state.available_mask,
                    cumulative_cost=state.cumulative_cost,
                    budget=budget,
                    raw_costs=RAW_USD_COSTS,
                    max_acquisitions=max_acquisitions,
                )
                features = torch.from_numpy(
                    state.feature_vector(belief_mode=belief_mode)
                ).float().unsqueeze(0).to(device)
                if features.shape[1] != model.state_dim:
                    raise ArtifactGateError(
                        f"State dimension mismatch for checkpoint {checkpoint_path}: "
                        f"{features.shape[1]} != {model.state_dim}"
                    )
                with torch.inference_mode():
                    energies_tensor = model(features).squeeze(0)
                    probabilities_tensor = energy_policy(
                        energies_tensor,
                        valid,
                        temperature=temperature,
                    )
                energies = energies_tensor.float().cpu().numpy()
                probabilities = probabilities_tensor.float().cpu().numpy()
                action_index = int(probabilities.argmax())
                action = ACTIONS[action_index]
                observed_information = _observed_information(
                    prepared, patient_index, state.observed_mask
                )
                prompt_state = {
                    "belief_mode": belief_mode,
                    "observed_information": observed_information,
                    "h": list(state.h),
                    "severity": state.severity,
                    "posterior": list(state.posterior),
                    "entropy": state.entropy,
                    "distance_to_cn_mci_boundary": state.boundary_distances[0],
                    "distance_to_mci_ad_boundary": state.boundary_distances[1],
                    "observed_mask": list(state.observed_mask),
                    "available_mask": list(state.available_mask),
                    "cumulative_cost_usd": state.cumulative_cost,
                    "remaining_budget_usd": state.remaining_budget,
                    "candidate_costs_usd": dict(RAW_USD_COSTS),
                    "action_history": list(history),
                    "valid_action_mask": valid.tolist(),
                    "step_index": step_index,
                    "max_acquisitions": max_acquisitions,
                }
                assert_no_forbidden_fields(prompt_state)
                full_prompt = build_structured_prompt(prompt_state)
                prompt = build_structured_prompt(prompt_state, max_chars=prompt_max_chars)
                prompt_observations_truncated = prompt != full_prompt
                if "DEPLOYABLE_STATE=" not in prompt or "valid_actions" not in prompt:
                    raise ArtifactGateError("Safe prompt sizing removed mandatory action fields")
                next_subset_code: int | None = None
                terminal = action_index >= len(MODALITIES)
                if not terminal:
                    next_subset_code = subset_code | (1 << action_index)
                    if next_subset_code not in lattice:
                        raise ArtifactGateError(
                            f"Missing revealed child state {next_subset_code} for patient {patient_id}"
                        )
                rows.append(
                    {
                        "teacher_seed": seed,
                        "patient_id": patient_id,
                        "split": split,
                        "budget_usd": float(budget),
                        "trajectory_step": step_index,
                        "subset_code": int(subset_code),
                        "state_features": state.feature_vector(
                            belief_mode=belief_mode
                        ).tolist(),
                        "prompt_state_json": json.dumps(
                            prompt_state, sort_keys=True, separators=(",", ":")
                        ),
                        "prompt": prompt,
                        "prompt_char_count": len(prompt),
                        "prompt_observations_truncated": prompt_observations_truncated,
                        "valid_action_mask": valid.tolist(),
                        "ebm_energies": energies.tolist(),
                        "planner_policy": probabilities.tolist(),
                        "chosen_action_index": action_index,
                        "chosen_action": action,
                        "terminal": bool(terminal),
                        "next_subset_code": next_subset_code,
                    }
                )
                if terminal:
                    terminated = True
                    break
                history = history + (action,)
                subset_code = int(next_subset_code)
            if not terminated:
                raise ArtifactGateError(
                    f"Teacher failed to diagnose within acquisition limit for patient {patient_id}"
                )

    trajectory_frame = pd.DataFrame(rows)
    if trajectory_frame.empty:
        raise ArtifactGateError("Teacher trajectory generation produced no records")
    if set(trajectory_frame["split"]) - {"train", "validation"}:
        raise ArtifactGateError("Test records leaked into distillation trajectories")
    if {"true_label", "label", "oracle_q"} & set(trajectory_frame.columns):
        raise ArtifactGateError("Forbidden supervision leaked into distillation records")
    for index, row in trajectory_frame.iterrows():
        mask = np.asarray(parse_vector(row["valid_action_mask"], name="valid_action_mask", length=10), dtype=bool)
        policy = np.asarray(parse_vector(row["planner_policy"], name="planner_policy", length=10), dtype=float)
        if np.any(policy[~mask] > 1e-8) or not np.isclose(policy.sum(), 1.0, atol=1e-5):
            raise ArtifactGateError(f"Invalid masked teacher policy at row {index}")

    write_parquet_atomic(trajectory_frame, output_path)
    metadata = {
        "schema_version": 1,
        "kind": "planner_distillation_trajectories",
        "created_at_utc": utc_now(),
        "deployable": False,
        "privileged": False,
        "offline_training_only": True,
        "student_input_projection_deployable": True,
        "contains_true_label": False,
        "contains_hidden_modalities": False,
        "contains_oracle_q": False,
        "student_input_columns": ["state_features", "prompt", "valid_action_mask"],
        "teacher_target_columns": [
            "ebm_energies",
            "planner_policy",
            "chosen_action_index",
        ],
        "fit_splits": ["train"],
        "checkpoint_selection_splits": ["validation"],
        "excluded_splits": ["test"],
        "test_used_for_training_or_selection": False,
        "execution": "receding_horizon_one_action_then_replan",
        "teacher_bellman_mode": manifest.get("bellman_mode"),
        "teacher_type": "soft_ebm",
        "belief_mode": str(manifest.get("planner_contract", {}).get("planner", {}).get("belief_mode", "ordinal")),
        "budget_conditioned": True,
        "training_budgets_usd": training_budgets,
        "prompt_max_chars": prompt_max_chars,
        "teacher_manifest": str(manifest_path),
        "teacher_manifest_sha256": sha256(manifest_path),
        "planner_contract": resolved_planner_contract,
        "planner_contract_sha256": resolved_planner_contract_sha256,
        "split_manifest_sha256": subset_metadata.get("split_manifest_sha256"),
        "h_schema_sha256": subset_metadata.get("h_schema_sha256"),
        "h_implementation_sha256": subset_metadata.get("h_implementation_sha256"),
        "h_input_schema_sha256": subset_metadata.get("h_input_schema_sha256"),
        "h_input_vectors_sha256": subset_metadata.get("h_input_vectors_sha256"),
        "final_calibration_sha256": subset_metadata.get("final_calibration_sha256"),
        "ordinal_checkpoint_sha256": subset_metadata.get("final_checkpoint_sha256"),
        "source_prepared_sha256": prepared_digest,
        "source_preprocessor_sha256": preprocessor_digest,
        "split_patient_counts": subset_metadata.get("split_patient_counts"),
        "source_subset_states": str(subset_path),
        "source_subset_sha256": subset_digest,
        "row_count": int(len(trajectory_frame)),
        "patient_count": int(trajectory_frame["patient_id"].nunique()),
        "teacher_seeds": sorted(int(value) for value in trajectory_frame["teacher_seed"].unique()),
        "artifact_sha256": sha256(output_path),
    }
    write_json_atomic(metadata_path(output_path), metadata)
    audit = write_audit(
        config,
        "generate_planner_trajectories",
        status="completed",
        inputs={
            "subset_states": str(subset_path),
            "planner_manifest": str(manifest_path),
            "prepared_data": str(prepared_path),
            "preprocessor": str(preprocessor_path),
        },
        outputs={
            "planner_trajectories": str(output_path),
            "metadata": str(metadata_path(output_path)),
        },
        gates={
            "receding_horizon": True,
            "one_action_per_step": True,
            "budget_conditioned_trajectories": True,
            "prepared_and_preprocessor_provenance_match": True,
            "invalid_actions_zero_probability": True,
            "true_label_excluded": True,
            "oracle_excluded": True,
            "test_excluded": True,
        },
        details={
            "rows": int(len(trajectory_frame)),
            "patients": int(trajectory_frame["patient_id"].nunique()),
            "teacher_seeds": metadata["teacher_seeds"],
        },
    )
    print(f"Wrote {len(trajectory_frame):,} teacher trajectory records to {output_path}")
    print(f"Audit: {audit}")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.cache_subset_states import ArtifactGateError, load_config_file

    try:
        run(load_config_file(args.config))
    except (ArtifactGateError, FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"generate_planner_trajectories gate failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
