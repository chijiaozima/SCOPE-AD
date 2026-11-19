from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Mapping

import numpy as np

from src.states.action_space import MODALITIES, NUM_ACTIONS, valid_action_mask
from src.states.state_builder import PlannerState
from src.states.subset_lattice import decode_mask, encode_mask, next_mask, validate_lattice
from .costs import CostSchedule
from .soft_value import BellmanMode, soft_value_numpy
from .terminal_risk import ordered_bayes_risk


@dataclass(frozen=True)
class TargetRecord:
    patient_id: str
    subset_code: int
    state_features: tuple[float, ...]
    valid_action_mask: tuple[bool, ...]
    target_energies: tuple[float, ...]
    horizon: int
    transition_source: str = "offline_revealed_outcome"
    terminal_source: str = "calibrated_belief_risk"
    contains_true_label: bool = False
    bellman_mode: BellmanMode = "soft"

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["target_energies"] = [None if not np.isfinite(x) else x for x in self.target_energies]
        return result


def build_backward_targets(
    states_by_code: Mapping[int, PlannerState],
    *,
    cost_schedule: CostSchedule,
    temperature: float,
    gamma: float = 1.0,
    horizon: int = 7,
    classification_weight: float = 1.0,
    distance_weight: float = 0.5,
    belief_mode: str = "ordinal",
    budget: float | None = None,
    max_acquisitions: int = len(MODALITIES),
    bellman_mode: BellmanMode = "soft",
    include_availability_in_predictive_state: bool = True,
) -> list[TargetRecord]:
    """Solve belief-risk Bellman targets on one observed-subset lattice.

    The function deliberately has no label argument. Actual hidden modality values may
    have been used upstream to create each next observed state, but terminal targets
    are calculated only from calibrated deployable belief. ``bellman_mode='soft'``
    uses a temperature-controlled soft minimum; ``'hard'`` uses the masked minimum.
    """
    if not states_by_code:
        return []
    if bellman_mode not in {"soft", "hard"}:
        raise ValueError("bellman_mode must be 'soft' or 'hard'")
    if bellman_mode == "soft" and temperature <= 0:
        raise ValueError("temperature must be positive for soft Bellman targets")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if horizon < 0 or horizon > len(MODALITIES):
        raise ValueError("horizon must be between 0 and 7")
    if (
        isinstance(max_acquisitions, bool)
        or not isinstance(max_acquisitions, (int, np.integer))
        or int(max_acquisitions) < 0
        or int(max_acquisitions) > len(MODALITIES)
    ):
        raise ValueError("max_acquisitions must be an integer between 0 and 7")
    acquisition_limit = int(max_acquisitions)
    first = next(iter(states_by_code.values()))
    available = np.asarray(first.available_mask, dtype=bool)
    validate_lattice((decode_mask(code) for code in states_by_code), available)
    for code, state in states_by_code.items():
        if state.patient_id != first.patient_id:
            raise ValueError("A target lattice must contain exactly one patient")
        if encode_mask(state.observed_mask) != int(code):
            raise ValueError(f"State mask does not match lattice code {code}")
        if tuple(state.available_mask) != tuple(first.available_mask):
            raise ValueError("available_mask cannot vary within a patient lattice")
    planning_cost = cost_schedule.planning

    @lru_cache(maxsize=None)
    def solve(code: int, depth: int) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        state = states_by_code[code]
        controller_mask = valid_action_mask(
            state.observed_mask,
            state.available_mask,
            cumulative_cost=state.cumulative_cost,
            budget=budget,
            raw_costs=cost_schedule.raw_usd,
            max_acquisitions=acquisition_limit,
        )
        if depth == 0:
            controller_mask[: len(MODALITIES)] = False
        targets = np.full(NUM_ACTIONS, np.inf, dtype=float)
        targets[len(MODALITIES) :] = ordered_bayes_risk(
            state.posterior,
            classification_weight=classification_weight,
            distance_weight=distance_weight,
        )
        if depth > 0:
            for modality_index, modality in enumerate(MODALITIES):
                if not controller_mask[modality_index]:
                    continue
                child_code = encode_mask(next_mask(state.observed_mask, modality_index))
                if child_code not in states_by_code:
                    raise KeyError(f"Missing child state {child_code} for modality {modality}")
                child_targets, child_mask = solve(child_code, depth - 1)
                continuation = soft_value_numpy(
                    child_targets,
                    child_mask,
                    temperature=temperature,
                    bellman_mode=bellman_mode,
                )
                targets[modality_index] = planning_cost[modality] + gamma * continuation
        return tuple(float(x) for x in targets), tuple(bool(x) for x in controller_mask)

    records: list[TargetRecord] = []
    for code in sorted(states_by_code, key=lambda item: (decode_mask(item).sum(), item), reverse=True):
        state = states_by_code[code]
        # The complete lattice is required above for structural validation and
        # for feasible child transitions.  An over-budget node cannot be a
        # controller-reachable root, however, and valid_action_mask correctly
        # rejects such a root.  Do not ask the solver to materialize it.
        if budget is not None and state.cumulative_cost > float(budget) + 1e-12:
            continue
        if int(np.sum(state.observed_mask)) > acquisition_limit:
            continue
        remaining = int(np.sum(available & ~decode_mask(code)))
        depth = min(int(horizon), remaining)
        targets, mask = solve(int(code), depth)
        records.append(
            TargetRecord(
                patient_id=state.patient_id,
                subset_code=int(code),
                state_features=tuple(
                    float(x)
                    for x in state.feature_vector(
                        belief_mode=belief_mode,
                        include_availability_in_predictive_state=(
                            include_availability_in_predictive_state
                        ),
                    )
                ),
                valid_action_mask=mask,
                target_energies=targets,
                horizon=int(horizon),
                bellman_mode=bellman_mode,
            )
        )
    return records
