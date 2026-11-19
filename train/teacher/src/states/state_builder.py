from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
import re
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np

from .action_space import ACQUISITION_ACTIONS, MODALITIES


FORBIDDEN_DEPLOYABLE_KEYS = {
    "label",
    "labels",
    "true_label",
    "ground_truth",
    "ground_truth_label",
    "target",
    "target_label",
    "hidden_modalities",
    "hidden_modality",
    "hidden_values",
    "unobserved_values",
    "future_observations",
    "oracle_q",
    "oracle_energy",
    "oracle_action",
    "oracle_value",
    "planner_energy",
    "planner_energies",
    "planner_policy",
    "teacher_policy",
    "target_policy",
    "future_outcomes",
}

_FORBIDDEN_DIAGNOSIS_KEY_PATTERN = re.compile(
    r"(^|_)(dx|diagnosis|diagnose|diagnostic|clinical_?status|disease_?stage)(_|$)"
)

_FORBIDDEN_DEPLOYABLE_PATTERNS = (
    re.compile(r"(^|_)labels?(_|$)"),
    re.compile(r"(^|_)true(_|$)"),
    re.compile(r"(^|_)hidden(_|$)"),
    re.compile(r"(^|_)oracle(_|$)"),
    re.compile(r"(^|_)ground_truth(_|$)"),
    re.compile(r"(^|_)(ebm|planner|teacher)_(policy|q|energy|target|action)(_|$)"),
    _FORBIDDEN_DIAGNOSIS_KEY_PATTERN,
)

_FORBIDDEN_DIAGNOSIS_VALUE_PATTERNS = (
    re.compile(r"\balzheimer(?:['’]?s)?(?:\s+disease)?\b", re.IGNORECASE),
    re.compile(r"\bmild\s+cognitive\s+impairment\b", re.IGNORECASE),
    re.compile(r"\bm(?:[\W_]*)c(?:[\W_]*)i\b", re.IGNORECASE),
    re.compile(r"\ba(?:[\W_]*)d\b", re.IGNORECASE),
    re.compile(r"\bc(?:[\W_]*)n\b", re.IGNORECASE),
    re.compile(r"\bdementia\b", re.IGNORECASE),
    re.compile(r"\bcognitively\s+normal\b", re.IGNORECASE),
    re.compile(r"\bnormal\s+cognition\b", re.IGNORECASE),
)


def _normalise_key(key: object) -> str:
    text = unicodedata.normalize("NFKC", str(key).strip())
    text = "".join(
        character for character in text if unicodedata.category(character) != "Cf"
    )
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def is_diagnosis_like_key(key: object) -> bool:
    normalized = _normalise_key(key)
    compact = normalized.replace("_", "")
    return bool(_FORBIDDEN_DIAGNOSIS_KEY_PATTERN.search(normalized)) or any(
        token in compact
        for token in ("diagnosis", "diagnose", "diagnostic", "clinicalstatus", "diseasestage")
    )


def _canonical_bool_mask(value: Any, *, name: str) -> tuple[bool, ...]:
    array = np.asarray(value)
    if array.shape != (len(MODALITIES),):
        raise ValueError(f"{name} must contain seven entries")
    if (
        array.dtype.kind not in "bifu"
        or not np.isfinite(array).all()
        or not np.isin(array, [0, 1]).all()
    ):
        raise ValueError(f"{name} must contain only booleans or exact zero/one values")
    return tuple(bool(item) for item in array)


def assert_deployable_mapping(
    value: Mapping[str, Any], *, path: str = "state", _seen: set[int] | None = None
) -> None:
    """Recursively reject privileged labels, hidden outcomes, and teacher/oracle data."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    seen = set() if _seen is None else _seen
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise ValueError(f"Deployable state contains a cyclic container: {path}")
        seen.add(identity)
        for raw_key, child in value.items():
            key = _normalise_key(raw_key)
            child_path = f"{path}.{raw_key}"
            if key in FORBIDDEN_DEPLOYABLE_KEYS or any(
                pattern.search(key) for pattern in _FORBIDDEN_DEPLOYABLE_PATTERNS
            ) or is_diagnosis_like_key(raw_key):
                raise ValueError(f"Deployable state contains forbidden field: {child_path}")
            if isinstance(child, Mapping):
                assert_deployable_mapping(child, path=child_path, _seen=seen)
            elif isinstance(child, (list, tuple)):
                _assert_deployable_sequence(child, path=child_path, _seen=seen)
            elif isinstance(child, np.ndarray):
                _assert_deployable_array(child, path=child_path, _seen=seen)
        seen.remove(identity)


def _assert_deployable_sequence(
    value: Sequence[Any], *, path: str, _seen: set[int]
) -> None:
    identity = id(value)
    if identity in _seen:
        raise ValueError(f"Deployable state contains a cyclic container: {path}")
    _seen.add(identity)
    for index, child in enumerate(value):
        child_path = f"{path}[{index}]"
        if isinstance(child, Mapping):
            assert_deployable_mapping(child, path=child_path, _seen=_seen)
        elif isinstance(child, (list, tuple)):
            _assert_deployable_sequence(child, path=child_path, _seen=_seen)
        elif isinstance(child, np.ndarray):
            _assert_deployable_array(child, path=child_path, _seen=_seen)
    _seen.remove(identity)


def _assert_deployable_array(
    value: np.ndarray, *, path: str, _seen: set[int]
) -> None:
    identity = id(value)
    if identity in _seen:
        raise ValueError(f"Deployable state contains a cyclic container: {path}")
    _seen.add(identity)
    _assert_deployable_sequence(value.tolist(), path=path, _seen=_seen)
    _seen.remove(identity)


def assert_no_diagnosis_content(
    value: Any, *, path: str, _seen: set[int] | None = None
) -> None:
    """Reject diagnosis-like keys or text in deployable observed measurements."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    seen = set() if _seen is None else _seen
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise ValueError(f"Deployable observations contain a cyclic container: {path}")
        seen.add(identity)
        for raw_key, child in value.items():
            key = _normalise_key(raw_key)
            child_path = f"{path}.{raw_key}"
            if is_diagnosis_like_key(raw_key):
                raise ValueError(f"Deployable observations contain diagnosis field: {child_path}")
            assert_no_diagnosis_content(child, path=child_path, _seen=seen)
        seen.remove(identity)
    elif isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            raise ValueError(f"Deployable observations contain a cyclic container: {path}")
        seen.add(identity)
        for index, child in enumerate(value):
            assert_no_diagnosis_content(
                child, path=f"{path}[{index}]", _seen=seen
            )
        seen.remove(identity)
    elif isinstance(value, np.ndarray):
        identity = id(value)
        if identity in seen:
            raise ValueError(f"Deployable observations contain a cyclic container: {path}")
        seen.add(identity)
        assert_no_diagnosis_content(value.tolist(), path=path, _seen=seen)
        seen.remove(identity)
    elif isinstance(value, np.generic):
        assert_no_diagnosis_content(value.item(), path=path, _seen=seen)
    elif isinstance(value, str) and any(
        pattern.search(value) for pattern in _FORBIDDEN_DIAGNOSIS_VALUE_PATTERNS
    ):
        raise ValueError(f"Deployable observations contain diagnosis-like text: {path}")
    elif value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                finite = bool(np.isfinite(value))
            except TypeError as exc:
                raise ValueError(
                    f"Deployable observations contain an unsupported numeric value: {path}"
                ) from exc
            if not finite:
                raise ValueError(
                    f"Deployable observations contain a non-finite number: {path}"
                )
    else:
        raise ValueError(
            f"Deployable observations contain an unsupported value type at {path}: "
            f"{type(value).__name__}"
        )


@dataclass(frozen=True)
class PlannerState:
    patient_id: str
    h: tuple[float, ...]
    severity: float
    posterior: tuple[float, float, float]
    entropy: float
    boundary_distances: tuple[float, float]
    observed_mask: tuple[bool, ...]
    available_mask: tuple[bool, ...]
    cumulative_cost: float = 0.0
    remaining_budget: float | None = None
    action_history: tuple[str, ...] = ()
    observed_values: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        h = np.asarray(self.h, dtype=float)
        if h.ndim != 1 or h.size == 0 or not np.isfinite(h).all():
            raise ValueError("h must be a non-empty finite one-dimensional vector")
        if not np.isfinite(float(self.severity)):
            raise ValueError("severity must be finite")
        if not np.isfinite(float(self.entropy)) or float(self.entropy) < 0:
            raise ValueError("entropy must be finite and non-negative")
        boundaries = np.asarray(self.boundary_distances, dtype=float)
        if (
            boundaries.shape != (2,)
            or not np.isfinite(boundaries).all()
            or np.any(boundaries < 0)
        ):
            raise ValueError("boundary_distances must contain two finite non-negative values")
        observed_mask = _canonical_bool_mask(
            self.observed_mask, name="observed_mask"
        )
        available_mask = _canonical_bool_mask(
            self.available_mask, name="available_mask"
        )
        object.__setattr__(self, "observed_mask", observed_mask)
        object.__setattr__(self, "available_mask", available_mask)
        if any(o and not a for o, a in zip(observed_mask, available_mask, strict=True)):
            raise ValueError("observed_mask must be a subset of available_mask")
        posterior = np.asarray(self.posterior, dtype=float)
        if (
            posterior.shape != (3,)
            or not np.isfinite(posterior).all()
            or np.any(posterior < 0)
            or not np.isclose(posterior.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError("posterior must be a normalized three-class probability vector")
        if not np.isfinite(float(self.cumulative_cost)) or self.cumulative_cost < 0:
            raise ValueError("cumulative_cost must be finite and non-negative")
        if self.remaining_budget is not None and (
            not np.isfinite(float(self.remaining_budget)) or self.remaining_budget < -1e-9
        ):
            raise ValueError("remaining_budget must be finite and non-negative")
        if not isinstance(self.action_history, (list, tuple)):
            raise ValueError("action_history must be a sequence of acquisition actions")
        history = tuple(self.action_history)
        if any(not isinstance(action, str) for action in history):
            raise ValueError("action_history entries must be acquisition action names")
        invalid_history = [
            action for action in history if action not in ACQUISITION_ACTIONS
        ]
        if invalid_history:
            raise ValueError(
                "action_history may contain only acquisition actions; invalid="
                f"{invalid_history}"
            )
        if len(set(history)) != len(history):
            raise ValueError("action_history cannot contain duplicate acquisitions")
        expected_history = tuple(
            action
            for action, observed in zip(
                ACQUISITION_ACTIONS, self.observed_mask, strict=True
            )
            if observed
        )
        if not history and expected_history:
            # Offline lattice snapshots have no temporal path. Materialise the only
            # safe history they can expose: the observed acquisitions in fixed order.
            history = expected_history
            object.__setattr__(self, "action_history", history)
        elif set(history) != set(expected_history):
            raise ValueError("action_history must match observed_mask exactly")
        elif history != self.action_history:
            object.__setattr__(self, "action_history", history)
        unknown = set(self.observed_values) - {
            modality for modality, observed in zip(MODALITIES, self.observed_mask, strict=True) if observed
        }
        if unknown:
            raise ValueError(f"observed_values contain hidden/unobserved modalities: {sorted(unknown)}")
        assert_deployable_mapping(self.observed_values, path="state.observed_values")
        assert_no_diagnosis_content(self.observed_values, path="state.observed_values")

    def feature_vector(
        self,
        *,
        belief_mode: str = "ordinal",
        cost_scale: float = 8520.0,
        include_availability_in_predictive_state: bool = True,
    ) -> np.ndarray:
        parts: list[np.ndarray] = [np.asarray(self.h, dtype=np.float32)]
        if belief_mode in {"categorical", "ordinal"}:
            parts.extend([np.asarray(self.posterior, dtype=np.float32), np.asarray([self.entropy], dtype=np.float32)])
        if belief_mode == "ordinal":
            parts.extend(
                [
                    np.asarray([self.severity], dtype=np.float32),
                    np.asarray(self.boundary_distances, dtype=np.float32),
                ]
            )
        elif belief_mode not in {"raw", "categorical"}:
            raise ValueError(f"Unknown belief_mode: {belief_mode}")
        # With no explicit episode budget, use the fixed full-reference residual.
        # Keeping this value at ``cost_scale`` for every subset would make the
        # training feature disagree with a full-budget rollout after one test.
        budget = (
            self.remaining_budget
            if self.remaining_budget is not None
            else max(0.0, float(cost_scale) - float(self.cumulative_cost))
        )
        predictive_availability = (
            np.asarray(self.available_mask, dtype=np.float32)
            if include_availability_in_predictive_state
            else np.zeros(len(self.available_mask), dtype=np.float32)
        )
        parts.extend(
            [
                np.asarray(self.observed_mask, dtype=np.float32),
                predictive_availability,
                np.asarray([self.cumulative_cost / cost_scale, float(budget) / cost_scale], dtype=np.float32),
            ]
        )
        return np.concatenate(parts)

    def public_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["observed_values"] = dict(self.observed_values)
        # Action order is not part of the Markov state and could otherwise be
        # supplied as a covert input. Expose only the canonical observed set.
        record["action_history"] = tuple(
            action
            for action, observed in zip(
                ACQUISITION_ACTIONS, self.observed_mask, strict=True
            )
            if observed
        )
        # Explicit aliases keep the LLM prompt schema stable without exposing
        # any additional information beyond the two deployable distances.
        record["distance_to_cn_mci_boundary"] = float(self.boundary_distances[0])
        record["distance_to_mci_ad_boundary"] = float(self.boundary_distances[1])
        assert_deployable_mapping(record)
        # ``observed_values`` may be backed by a caller-owned mutable mapping.
        # Recheck at the public boundary so post-construction mutation fails closed.
        assert_no_diagnosis_content(
            record["observed_values"], path="state.observed_values"
        )
        return record

    def with_belief(self, belief: Mapping[str, Any]) -> "PlannerState":
        assert_deployable_mapping(belief, path="belief")
        required = ("h", "severity", "posterior", "entropy", "boundary_distances")
        missing = [key for key in required if key not in belief]
        if missing:
            raise KeyError(f"Belief update missing fields: {missing}")
        return replace(
            self,
            h=tuple(float(x) for x in belief["h"]),
            severity=float(belief["severity"]),
            posterior=tuple(float(x) for x in belief["posterior"]),
            entropy=float(belief["entropy"]),
            boundary_distances=tuple(float(x) for x in belief["boundary_distances"]),
        )


def build_state(*, belief: Mapping[str, Any], **state_fields: Any) -> PlannerState:
    assert_deployable_mapping(belief, path="belief")
    assert_deployable_mapping(state_fields)
    return PlannerState(
        h=tuple(float(x) for x in belief["h"]),
        severity=float(belief["severity"]),
        posterior=tuple(float(x) for x in belief["posterior"]),
        entropy=float(belief["entropy"]),
        boundary_distances=tuple(float(x) for x in belief["boundary_distances"]),
        **state_fields,
    )
