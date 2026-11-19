"""Leakage-safe structured prompts for the LLM action policy."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..states.action_space import (
    ACQUISITION_ACTIONS,
    ACTIONS,
    MODALITIES,
    NUM_ACTIONS,
    valid_action_mask,
)
from ..states.state_builder import assert_no_diagnosis_content, is_diagnosis_like_key


DEFAULT_RAW_COSTS: dict[str, float] = {
    "cognitive": 50.0,
    "apoe": 283.0,
    "blood": 232.0,
    "csf": 574.0,
    "structural_mri": 1431.0,
    "amyloid_pet": 2975.0,
    "tau_pet": 2975.0,
}

_FORBIDDEN_EXACT = {
    "label",
    "labels",
    "true_label",
    "ground_truth",
    "ground_truth_label",
    "target_label",
    "diagnosis_target",
    "hidden",
    "hidden_data",
    "hidden_modality",
    "hidden_modalities",
    "hidden_values",
    "unobserved_values",
    "future_observations",
    "oracle",
    "oracle_action",
    "oracle_q",
    "oracle_value",
    "planner_energy",
    "planner_energies",
    "ebm_energy",
    "ebm_energies",
    "planner_policy",
    "teacher_policy",
    "energy_vector",
    "q_value",
    "q_values",
    "soft_target",
    "target_policy",
    "chosen_action_target",
    "chosen_action",
    "chosen_action_index",
    "future_outcomes",
}
_FORBIDDEN_PATTERNS = (
    re.compile(r"(^|_)labels?(_|$)"),
    re.compile(r"(^|_)true(_|$)"),
    re.compile(r"(^|_)hidden(_|$)"),
    re.compile(r"(^|_)oracle(_|$)"),
    re.compile(r"(^|_)ground_truth(_|$)"),
    re.compile(r"(^|_)(ebm|planner|teacher)_(policy|q|energy|target|action)(_|$)"),
    re.compile(r"(^|_)planner_(q|energy|target)(_|$)"),
    re.compile(
        r"(^|_)(dx|diagnosis|diagnose|diagnostic|clinical_?status|disease_?stage)(_|$)"
    ),
)


class PromptSchemaError(ValueError):
    """Raised before unsafe or malformed state can enter a prompt."""


def _normalise_key(key: object) -> str:
    text = unicodedata.normalize("NFKC", str(key).strip())
    text = "".join(
        character for character in text if unicodedata.category(character) != "Cf"
    )
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def assert_no_forbidden_fields(
    value: Any, *, path: str = "state", _seen: set[int] | None = None
) -> None:
    """Recursively reject labels, hidden outcomes, oracle data, and planner targets."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    seen = set() if _seen is None else _seen
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise PromptSchemaError(f"Cyclic deployable prompt container: {path}")
        seen.add(identity)
        for raw_key, child in value.items():
            key = _normalise_key(raw_key)
            child_path = f"{path}.{raw_key}"
            if (
                key in _FORBIDDEN_EXACT
                or any(pattern.search(key) for pattern in _FORBIDDEN_PATTERNS)
                or is_diagnosis_like_key(raw_key)
            ):
                raise PromptSchemaError(f"Forbidden deployable prompt field: {child_path}")
            assert_no_forbidden_fields(child, path=child_path, _seen=seen)
        seen.remove(identity)
    elif isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            raise PromptSchemaError(f"Cyclic deployable prompt container: {path}")
        seen.add(identity)
        for index, child in enumerate(value):
            assert_no_forbidden_fields(
                child, path=f"{path}[{index}]", _seen=seen
            )
        seen.remove(identity)
    elif isinstance(value, np.ndarray):
        identity = id(value)
        if identity in seen:
            raise PromptSchemaError(f"Cyclic deployable prompt container: {path}")
        seen.add(identity)
        assert_no_forbidden_fields(value.tolist(), path=path, _seen=seen)
        seen.remove(identity)
    elif isinstance(value, np.generic):
        assert_no_forbidden_fields(value.item(), path=path, _seen=seen)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, PromptState):
        return asdict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise PromptSchemaError("Prompt state must be a mapping or dataclass instance")


def _first(mapping: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _finite_float(value: Any, name: str, *, non_negative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PromptSchemaError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (non_negative and result < 0):
        qualifier = "finite and non-negative" if non_negative else "finite"
        raise PromptSchemaError(f"{name} must be {qualifier}")
    return result


def _bool_vector(value: Any, name: str, size: int) -> tuple[bool, ...]:
    array = np.asarray(value)
    if array.shape != (size,):
        raise PromptSchemaError(f"{name} must have shape ({size},)")
    if (
        array.dtype.kind not in "bifu"
        or not np.isfinite(array).all()
        or not np.isin(array, [0, 1]).all()
    ):
        raise PromptSchemaError(f"{name} must contain only booleans")
    return tuple(bool(item) for item in array)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


@dataclass(frozen=True)
class PromptState:
    """Only deployable fields permitted in a student prompt."""

    observed_information: Mapping[str, Any]
    severity: float
    posterior: tuple[float, float, float]
    entropy: float
    distance_to_cn_mci_boundary: float
    distance_to_mci_ad_boundary: float
    observed_mask: tuple[bool, ...]
    available_mask: tuple[bool, ...]
    cumulative_cost_usd: float
    remaining_budget_usd: float | None
    latent_h: tuple[float, ...] | None = None
    candidate_costs_usd: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_RAW_COSTS))
    action_history: tuple[str, ...] = ()
    valid_action_mask: tuple[bool, ...] | None = None
    step_index: int = 0
    max_acquisitions: int = len(MODALITIES)
    belief_mode: str = "ordinal"


def validate_prompt_schema(state: Any) -> PromptState:
    """Validate and canonicalise a state into the safe prompt schema."""

    raw = _mapping(state)
    assert_no_forbidden_fields(raw)
    belief_mode = str(raw.get("belief_mode", "ordinal"))
    if belief_mode not in {"ordinal", "categorical"}:
        raise PromptSchemaError("belief_mode must be 'ordinal' or 'categorical'")
    belief_raw = _first(raw, ("belief", "ordinal_belief"), {})
    belief = dict(belief_raw) if isinstance(belief_raw, Mapping) else {}

    observed_information = _first(
        raw,
        ("observed_information", "observed_values", "observations", "patient_information"),
        {},
    )
    if not isinstance(observed_information, Mapping):
        raise PromptSchemaError("observed_information must be a mapping")
    assert_no_forbidden_fields(observed_information, path="state.observed_information")
    try:
        assert_no_diagnosis_content(
            observed_information, path="state.observed_information"
        )
    except ValueError as exc:
        raise PromptSchemaError(str(exc)) from exc

    posterior_value = _first(
        raw,
        ("posterior", "class_probabilities", "class_posterior"),
        _first(belief, ("posterior", "class_probabilities", "class_posterior")),
    )
    if posterior_value is None:
        raise PromptSchemaError("posterior/class_probabilities is required")
    posterior_array = np.asarray(posterior_value, dtype=float)
    if posterior_array.shape != (3,) or not np.isfinite(posterior_array).all() or np.any(posterior_array < 0):
        raise PromptSchemaError("posterior must be three finite, non-negative probabilities")
    if not np.isclose(posterior_array.sum(), 1.0, atol=1e-5):
        raise PromptSchemaError("posterior probabilities must sum to one")

    latent_value = _first(raw, ("latent_h", "h", "latent_representation"))
    if latent_value is None:
        latent_h = None
    else:
        latent_array = np.asarray(latent_value, dtype=float)
        if latent_array.ndim != 1 or latent_array.size == 0 or not np.isfinite(latent_array).all():
            raise PromptSchemaError("latent_h/h must be a non-empty finite one-dimensional vector")
        latent_h = tuple(float(value) for value in latent_array)

    severity = _finite_float(
        _first(raw, ("severity", "severity_score", "z"), _first(belief, ("severity", "severity_score", "z"))),
        "severity",
    )
    entropy = _finite_float(
        _first(raw, ("entropy", "predictive_entropy"), _first(belief, ("entropy", "predictive_entropy"))),
        "entropy",
        non_negative=True,
    )
    distance_cn = _finite_float(
        _first(
            raw,
            ("distance_to_cn_mci_boundary", "cn_mci_boundary_distance"),
            _first(belief, ("distance_to_cn_mci_boundary", "cn_mci_boundary_distance")),
        ),
        "distance_to_cn_mci_boundary",
    )
    distance_ad = _finite_float(
        _first(
            raw,
            ("distance_to_mci_ad_boundary", "mci_ad_boundary_distance"),
            _first(belief, ("distance_to_mci_ad_boundary", "mci_ad_boundary_distance")),
        ),
        "distance_to_mci_ad_boundary",
    )
    observed = _bool_vector(raw.get("observed_mask"), "observed_mask", len(MODALITIES))
    available = _bool_vector(raw.get("available_mask"), "available_mask", len(MODALITIES))
    if any(seen and not exists for seen, exists in zip(observed, available)):
        raise PromptSchemaError("observed_mask must be a subset of available_mask")
    allowed_information = {"demographic", *MODALITIES}
    unknown_information = set(observed_information) - allowed_information
    if unknown_information:
        raise PromptSchemaError(
            "observed_information has unsupported top-level field(s): "
            f"{sorted(unknown_information)}"
        )
    if "demographic" not in observed_information:
        raise PromptSchemaError("observed_information must contain demographic")
    expected_modalities = {
        modality
        for modality, is_observed in zip(MODALITIES, observed, strict=True)
        if is_observed
    }
    supplied_modalities = set(observed_information) - {"demographic"}
    if supplied_modalities != expected_modalities:
        raise PromptSchemaError(
            "observed_information acquisition modalities must match observed_mask exactly"
        )

    cumulative = _finite_float(
        _first(raw, ("cumulative_cost_usd", "cumulative_cost", "total_usd_cost"), 0.0),
        "cumulative_cost_usd",
        non_negative=True,
    )
    remaining_value = _first(raw, ("remaining_budget_usd", "remaining_budget"))
    if remaining_value is None and raw.get("budget") is not None:
        remaining_value = float(raw["budget"]) - cumulative
    remaining = None if remaining_value is None else _finite_float(
        remaining_value,
        "remaining_budget_usd",
        non_negative=True,
    )

    costs_raw = _first(raw, ("candidate_costs_usd", "raw_costs", "action_costs_usd"), DEFAULT_RAW_COSTS)
    if not isinstance(costs_raw, Mapping):
        raise PromptSchemaError("candidate_costs_usd must be a mapping")
    costs: dict[str, float] = {}
    for modality in MODALITIES:
        if modality not in costs_raw:
            raise PromptSchemaError(f"candidate_costs_usd is missing '{modality}'")
        costs[modality] = _finite_float(costs_raw[modality], f"cost[{modality}]", non_negative=True)

    history_raw = _first(raw, ("action_history", "history"), ())
    if isinstance(history_raw, str) or not isinstance(history_raw, Sequence):
        raise PromptSchemaError("action_history must be a sequence of action names")
    if any(not isinstance(action, str) for action in history_raw):
        raise PromptSchemaError("action_history entries must be action-name strings")
    history = tuple(history_raw)
    unknown = [action for action in history if action not in ACQUISITION_ACTIONS]
    if unknown:
        raise PromptSchemaError(
            "action_history may contain only acquisition actions; invalid="
            f"{unknown}"
        )
    if len(set(history)) != len(history):
        raise PromptSchemaError("action_history cannot contain duplicate acquisitions")
    history_modalities = {
        MODALITIES[ACQUISITION_ACTIONS.index(action)] for action in history
    }
    expected_history_modalities = {
        modality
        for modality, is_observed in zip(MODALITIES, observed, strict=True)
        if is_observed
    }
    if history_modalities != expected_history_modalities:
        raise PromptSchemaError("action_history must match observed_mask exactly")
    canonical_history = tuple(
        action
        for action, is_observed in zip(
            ACQUISITION_ACTIONS, observed, strict=True
        )
        if is_observed
    )

    expected_cost = sum(
        costs[MODALITIES[ACQUISITION_ACTIONS.index(action)]] for action in history
    )
    if not math.isclose(cumulative, expected_cost, rel_tol=1e-12, abs_tol=1e-8):
        raise PromptSchemaError(
            "cumulative_cost_usd must equal the candidate costs in action_history"
        )

    max_acquisitions_raw = raw.get("max_acquisitions", len(MODALITIES))
    if isinstance(max_acquisitions_raw, (bool, np.bool_)) or not isinstance(
        max_acquisitions_raw, (int, np.integer)
    ):
        raise PromptSchemaError("max_acquisitions must be an integer between zero and seven")
    max_acquisitions = int(max_acquisitions_raw)
    if max_acquisitions < 0 or max_acquisitions > len(MODALITIES):
        raise PromptSchemaError("max_acquisitions must be an integer between zero and seven")

    supplied_mask = raw.get("valid_action_mask")
    budget = None if remaining is None else cumulative + remaining
    derived = valid_action_mask(
        observed,
        available,
        cumulative_cost=cumulative,
        budget=budget,
        raw_costs=costs,
        acquisitions_used=len(history),
        max_acquisitions=max_acquisitions,
    )
    if supplied_mask is not None:
        prompt_mask = _bool_vector(supplied_mask, "valid_action_mask", NUM_ACTIONS)
        supplied = np.asarray(prompt_mask, dtype=bool)
        if not np.array_equal(supplied, derived):
            raise PromptSchemaError(
                "valid_action_mask must exactly equal the controller-derived mask"
            )
    else:
        prompt_mask = tuple(bool(item) for item in derived)

    step_index_raw = raw.get("step_index", len(history))
    if isinstance(step_index_raw, (bool, np.bool_)) or not isinstance(
        step_index_raw, (int, np.integer)
    ):
        raise PromptSchemaError("step_index must be an integer")
    step_index = int(step_index_raw)
    if step_index < 0:
        raise PromptSchemaError("step_index must be non-negative")
    if step_index != len(history):
        raise PromptSchemaError("step_index must equal the action_history length")

    return PromptState(
        observed_information=dict(observed_information),
        severity=severity,
        posterior=tuple(float(value) for value in posterior_array),
        entropy=entropy,
        distance_to_cn_mci_boundary=distance_cn,
        distance_to_mci_ad_boundary=distance_ad,
        observed_mask=observed,
        available_mask=available,
        cumulative_cost_usd=cumulative,
        remaining_budget_usd=remaining,
        latent_h=latent_h,
        candidate_costs_usd=costs,
        action_history=canonical_history,
        valid_action_mask=prompt_mask,
        step_index=step_index,
        max_acquisitions=max_acquisitions,
        belief_mode=belief_mode,
    )


def build_structured_prompt(state: Any, *, max_chars: int | None = None) -> str:
    """Render a deterministic prompt containing only validated deployable state."""

    safe = validate_prompt_schema(state)
    observed_names = [name for name, seen in zip(MODALITIES, safe.observed_mask) if seen]
    candidates = [
        {
            "action": ACTIONS[index],
            "cost_usd": safe.candidate_costs_usd[modality],
        }
        for index, modality in enumerate(MODALITIES)
        if safe.valid_action_mask and safe.valid_action_mask[index]
    ]
    valid_names = [
        action for action, is_valid in zip(ACTIONS, safe.valid_action_mask or ()) if is_valid
    ]
    belief_payload = {
        "posterior": {
            "CN": round(safe.posterior[0], 8),
            "MCI": round(safe.posterior[1], 8),
            "AD": round(safe.posterior[2], 8),
        },
        "entropy": round(safe.entropy, 8),
    }
    if safe.belief_mode == "ordinal":
        belief_payload = {
            "severity": round(safe.severity, 8),
            **belief_payload,
            "distance_to_CN_MCI_boundary": round(safe.distance_to_cn_mci_boundary, 8),
            "distance_to_MCI_AD_boundary": round(safe.distance_to_mci_ad_boundary, 8),
        }
    state_payload = {
        "step": safe.step_index,
        "max_acquisitions": safe.max_acquisitions,
        "observed_modalities": observed_names,
        "observed_information": _json_safe(safe.observed_information),
        "latent_h": None if safe.latent_h is None else [round(value, 6) for value in safe.latent_h],
        ("ordinal_belief" if safe.belief_mode == "ordinal" else "class_belief"): belief_payload,
        "observed_mask": list(safe.observed_mask),
        "available_mask": list(safe.available_mask),
        "cumulative_cost_usd": safe.cumulative_cost_usd,
        "remaining_budget_usd": safe.remaining_budget_usd,
        "candidate_requests": candidates,
        "action_history": list(safe.action_history),
        "valid_actions": valid_names,
    }
    body = json.dumps(state_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    prompt = (
        "You are a cost-aware sequential CN/MCI/AD diagnosis policy.\n"
        "Choose exactly one action from valid_actions. Use the fixed action names verbatim.\n"
        "Return only the action name; do not explain the choice.\n"
        f"DEPLOYABLE_STATE={body}"
    )
    if max_chars is not None:
        if int(max_chars) <= 0:
            raise ValueError("max_chars must be positive")
        if len(prompt) > int(max_chars):
            # Drop verbose observations first; never cut the rules, belief, costs, or mask.
            state_payload["observed_information"] = {
                key: {"omitted_due_to_length": True}
                for key in safe.observed_information
            }
            body = json.dumps(state_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            prompt = (
                "You are a cost-aware sequential CN/MCI/AD diagnosis policy.\n"
                "Choose exactly one action from valid_actions. Use the fixed action names verbatim.\n"
                "Return only the action name; do not explain the choice.\n"
                f"DEPLOYABLE_STATE={body}"
            )
            if len(prompt) > int(max_chars):
                raise PromptSchemaError(
                    "max_chars is too small for the mandatory safe prompt fields"
                )
    return prompt


class PromptFormatter:
    """Small callable wrapper used by tokenization datasets."""

    def __init__(self, max_chars: int | None = None):
        self.max_chars = max_chars

    def format(self, state: Any) -> str:
        return build_structured_prompt(state, max_chars=self.max_chars)

    __call__ = format


# Convenient names used by training scripts.
build_prompt = build_structured_prompt
format_prompt = build_structured_prompt
validate_no_leakage = assert_no_forbidden_fields


__all__ = [
    "DEFAULT_RAW_COSTS",
    "PromptFormatter",
    "PromptSchemaError",
    "PromptState",
    "assert_no_forbidden_fields",
    "build_prompt",
    "build_structured_prompt",
    "format_prompt",
    "validate_no_leakage",
    "validate_prompt_schema",
]
