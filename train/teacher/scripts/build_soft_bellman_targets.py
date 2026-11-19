#!/usr/bin/env python3
"""Build belief-risk soft Bellman targets without accepting true labels."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve finite-horizon soft Bellman targets backward over each seven-modality "
            "subset lattice. Diagnosis targets use calibrated belief risk only."
        )
    )
    parser.add_argument("--config", required=True, help="Planner YAML configuration.")
    return parser


def _state_from_row(row: Any) -> Any:
    from scripts.cache_subset_states import parse_vector
    from src.states.state_builder import build_state

    remaining = row.get("remaining_budget_usd")
    if remaining is not None:
        try:
            import math

            if math.isnan(float(remaining)):
                remaining = None
        except (TypeError, ValueError):
            pass
    return build_state(
        belief={
            "h": parse_vector(row["h"], name="h"),
            "severity": float(row["severity"]),
            "posterior": parse_vector(row["posterior"], name="posterior", length=3),
            "entropy": float(row["entropy"]),
            "boundary_distances": parse_vector(
                row["boundary_distances"], name="boundary_distances", length=2
            ),
        },
        patient_id=str(row["patient_id"]),
        observed_mask=tuple(
            bool(value)
            for value in parse_vector(row["observed_mask"], name="observed_mask", length=7)
        ),
        available_mask=tuple(
            bool(value)
            for value in parse_vector(row["available_mask"], name="available_mask", length=7)
        ),
        cumulative_cost=float(row["cumulative_cost_usd"]),
        remaining_budget=None if remaining is None else float(remaining),
    )


def _cost_schedule(config: Mapping[str, Any]) -> Any:
    from scripts.cache_subset_states import ArtifactGateError
    from src.planner.costs import CostSchedule

    cost = config.get("cost")
    if not isinstance(cost, Mapping):
        raise ArtifactGateError("Planner configuration must include a cost mapping")
    if cost.get("source") != "raw_usd" or cost.get("transform") != "normalized_log":
        raise ArtifactGateError(
            "Planner costs must use source=raw_usd and transform=normalized_log"
        )
    if float(cost.get("lambda_invasiveness", 0.0)) != 0.0 or float(
        cost.get("lambda_time", 0.0)
    ) != 0.0:
        raise ArtifactGateError(
            "Invasiveness/time costs are unsupported without an externally sourced weight artifact"
        )
    raw = cost.get("raw_usd")
    if not isinstance(raw, Mapping):
        raise ArtifactGateError("cost.raw_usd is required")
    return CostSchedule(
        raw_usd={str(key): float(value) for key, value in raw.items()},
        lambda_usd=float(cost.get("lambda_usd", 1.0)),
        lambda_step=float(cost.get("lambda_step", 0.0)),
    )


def run(config: Mapping[str, Any]) -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import numpy as np
    import pandas as pd

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
        write_parquet_atomic,
    )
    from src.planner.backward_targets import build_backward_targets
    from src.states.action_space import MODALITIES

    subset_path = resolve_artifact(config, artifact_value(config, "subset_states"))
    output_path = resolve_artifact(config, artifact_value(config, "bellman_targets"))
    require_file(subset_path, description="subset-state artifact")
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
        raise ArtifactGateError("Subset states do not prove contains_true_label=false")
    for digest_field in (
        "h_schema_sha256",
        "h_implementation_sha256",
        "h_input_schema_sha256",
        "h_input_vectors_sha256",
    ):
        if not subset_metadata.get(digest_field):
            raise ArtifactGateError(f"Subset states lack canonical {digest_field}")

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
    if "true_label" in frame.columns or "label" in frame.columns:
        raise ArtifactGateError("Deployable subset artifact contains a label column")
    if frame.empty:
        raise ArtifactGateError("Subset-state artifact contains no rows")

    planner = config.get("planner")
    ordinal_risk = config.get("ordinal_risk")
    if not isinstance(planner, Mapping) or not isinstance(ordinal_risk, Mapping):
        raise ArtifactGateError("planner and ordinal_risk mappings are required")
    schedule = _cost_schedule(config)
    resolved_planner_contract = planner_contract(config)
    planner_contract_sha256 = canonical_mapping_sha256(resolved_planner_contract)
    temperature = float(planner.get("energy_temperature", 0.3))
    bellman_mode = str(planner.get("bellman_mode", ""))
    if bellman_mode not in {"soft", "hard"}:
        raise ArtifactGateError("planner.bellman_mode must be explicitly 'soft' or 'hard'")
    gamma = float(planner.get("gamma", 1.0))
    horizon = int(planner.get("horizon", 7))
    max_acquisitions = int(planner.get("max_acquisitions", len(MODALITIES)))
    belief_mode = str(planner.get("belief_mode", "ordinal"))
    experiment_config = load_config_file("configs/experiments.yaml")
    experiment_values = experiment_config.get("experiments", {})
    budgets_raw = planner.get("budgets_usd", experiment_values.get("budgets_usd", ()))
    budget_source = (
        "planner.budgets_usd"
        if planner.get("budgets_usd") is not None
        else "configs/experiments.yaml:experiments.budgets_usd"
    )
    if not isinstance(budgets_raw, Sequence) or isinstance(budgets_raw, (str, bytes)):
        raise ArtifactGateError("planner.budgets_usd must be a sequence")
    budgets = sorted({float(value) for value in budgets_raw})
    if not budgets or any(value <= 0 for value in budgets):
        raise ArtifactGateError("At least one positive predeclared planner budget is required")

    output_rows: list[dict[str, Any]] = []
    for patient_id, patient_frame in frame.groupby("patient_id", sort=True):
        splits = set(str(value) for value in patient_frame["split"])
        if len(splits) != 1:
            raise ArtifactGateError(f"Patient {patient_id} appears in multiple splits: {splits}")
        split = next(iter(splits))
        states: dict[int, Any] = {}
        for _, row in patient_frame.iterrows():
            code = int(row["subset_code"])
            if code in states:
                raise ArtifactGateError(f"Duplicate subset {code} for patient {patient_id}")
            states[code] = _state_from_row(row)
        for budget in budgets:
            budget_states = {
                code: dataclasses.replace(
                    state,
                    remaining_budget=max(0.0, budget - state.cumulative_cost),
                )
                for code, state in states.items()
            }
            # Deliberately no label argument exists on this call. Over-budget states
            # remain present only so lattice validation is exact; controller masks
            # prevent transitions into them, and they are filtered from the artifact.
            records = build_backward_targets(
                budget_states,
                cost_schedule=schedule,
                temperature=temperature,
                gamma=gamma,
                horizon=horizon,
                classification_weight=float(
                    ordinal_risk.get("classification_weight", 1.0)
                ),
                distance_weight=float(ordinal_risk.get("distance_weight", 0.5)),
                belief_mode=belief_mode,
                budget=budget,
                max_acquisitions=max_acquisitions,
                bellman_mode=bellman_mode,
                include_availability_in_predictive_state=bool(
                    planner.get("include_availability_in_predictive_state", True)
                ),
            )
            for record in records:
                state = budget_states[int(record.subset_code)]
                if state.cumulative_cost > budget + 1e-9:
                    continue
                values = record.to_dict()
                output_rows.append(
                    {
                        "patient_id": record.patient_id,
                        "split": split,
                        "budget_usd": float(budget),
                        "remaining_budget_usd": float(state.remaining_budget),
                        "subset_code": int(record.subset_code),
                        "state_features": list(record.state_features),
                        "valid_action_mask": list(record.valid_action_mask),
                        "target_energies": values["target_energies"],
                        "horizon": int(record.horizon),
                        "transition_source": record.transition_source,
                        "terminal_source": record.terminal_source,
                        "bellman_mode": record.bellman_mode,
                    }
                )

    target_frame = pd.DataFrame(output_rows)
    if target_frame.empty:
        raise ArtifactGateError("Backward target construction produced no rows")
    for row_index, row in target_frame.iterrows():
        mask = np.asarray(row["valid_action_mask"], dtype=bool)
        raw_targets = row["target_energies"]
        targets = np.asarray(
            [np.inf if value is None else float(value) for value in raw_targets], dtype=float
        )
        if mask.shape != (10,) or targets.shape != (10,):
            raise ArtifactGateError(f"Target row {row_index} does not contain ten actions")
        if not np.all(np.isfinite(targets[mask])):
            raise ArtifactGateError(f"Target row {row_index} has non-finite valid energies")
        if np.any(np.isfinite(targets[~mask])):
            raise ArtifactGateError(f"Target row {row_index} assigns energy to an invalid action")

    write_parquet_atomic(target_frame, output_path)
    metadata = {
        "schema_version": 1,
        "kind": "soft_bellman_targets",
        "created_at_utc": utc_now(),
        "deployable": False,
        "privileged": False,
        "offline_training_only": True,
        "contains_true_label": False,
        "contains_hidden_values": False,
        "uses_hidden_outcomes_for_transitions": True,
        "terminal_source": "calibrated_belief_risk",
        "transition_source": "offline_revealed_outcome",
        "test_used_for_training_or_selection": False,
        "action_count": 10,
        "horizon": horizon,
        "gamma": gamma,
        "energy_temperature": temperature,
        "bellman_mode": bellman_mode,
        "belief_mode": belief_mode,
        "include_availability_in_predictive_state": bool(
            planner.get("include_availability_in_predictive_state", True)
        ),
        "budget_conditioned": True,
        "training_budgets_usd": budgets,
        "training_budget_source": budget_source,
        "over_budget_states_excluded": True,
        "h_schema_sha256": subset_metadata.get("h_schema_sha256"),
        "h_implementation_sha256": subset_metadata.get("h_implementation_sha256"),
        "h_input_schema_sha256": subset_metadata.get("h_input_schema_sha256"),
        "h_input_vectors_sha256": subset_metadata.get("h_input_vectors_sha256"),
        "h_representation": subset_metadata.get("h_representation"),
        "severity_semantics": subset_metadata.get("severity_semantics"),
        "boundary_distance_semantics": subset_metadata.get(
            "boundary_distance_semantics"
        ),
        "final_calibration_sha256": subset_metadata.get("final_calibration_sha256"),
        "final_checkpoint_sha256": subset_metadata.get("final_checkpoint_sha256"),
        "source_prepared_sha256": subset_metadata.get("source_prepared_sha256"),
        "source_preprocessor_sha256": subset_metadata.get("source_preprocessor_sha256"),
        "split_manifest_sha256": subset_metadata.get("split_manifest_sha256"),
        "split_patient_counts": subset_metadata.get("split_patient_counts"),
        "planning_costs": schedule.planning,
        "raw_usd_costs": dict(schedule.raw_usd),
        "planner_contract": resolved_planner_contract,
        "planner_contract_sha256": planner_contract_sha256,
        "source_subset_states": str(subset_path),
        "source_subset_sha256": sha256(subset_path),
        "row_count": int(len(target_frame)),
        "patient_count": int(target_frame["patient_id"].nunique()),
        "split_counts": {
            str(key): int(value)
            for key, value in target_frame.groupby("split").size().items()
        },
        "artifact_sha256": sha256(output_path),
    }
    write_json_atomic(metadata_path(output_path), metadata)
    audit = write_audit(
        config,
        "build_soft_bellman_targets",
        status="completed",
        inputs={"subset_states": str(subset_path)},
        outputs={
            "bellman_targets": str(output_path),
            "metadata": str(metadata_path(output_path)),
        },
        gates={
            "subset_lattice_offline_non_deployable": True,
            "subset_row_input_schema_deployable": True,
            "target_artifact_offline_training_only": True,
            "true_label_not_accepted": True,
            "terminal_is_belief_risk": True,
            "all_valid_targets_finite": True,
            "budget_conditioned_targets": True,
            "bellman_mode_explicit": True,
            "over_budget_states_excluded": True,
            "test_not_used_for_selection": True,
        },
        details={
            "rows": int(len(target_frame)),
            "patients": int(target_frame["patient_id"].nunique()),
            "bellman_mode": bellman_mode,
        },
    )
    print(f"Wrote {len(target_frame):,} belief-risk Bellman targets to {output_path}")
    print(f"Audit: {audit}")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.cache_subset_states import ArtifactGateError, load_config_file

    try:
        run(load_config_file(args.config))
    except (ArtifactGateError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"build_soft_bellman_targets gate failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
