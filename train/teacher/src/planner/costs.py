from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Mapping

from src.states.action_space import MODALITIES


RAW_USD_COSTS: dict[str, float] = {
    "cognitive": 50.0,
    "apoe": 283.0,
    "blood": 232.0,
    "csf": 574.0,
    "structural_mri": 1431.0,
    "amyloid_pet": 2975.0,
    "tau_pet": 2975.0,
}
FULL_REFERENCE_COST_USD = 8520.0


def validate_raw_costs(raw_costs: Mapping[str, float]) -> None:
    missing = sorted(set(MODALITIES) - set(raw_costs))
    extra = sorted(set(raw_costs) - set(MODALITIES))
    if missing or extra:
        raise ValueError(f"Cost keys must match modalities; missing={missing}, extra={extra}")
    if any(float(raw_costs[name]) <= 0 for name in MODALITIES):
        raise ValueError("All raw costs must be positive")


def normalized_log_costs(
    raw_costs: Mapping[str, float],
    *,
    reference_costs: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Apply the fixed-reference normalized log transform from the specification."""
    validate_raw_costs(raw_costs)
    reference = RAW_USD_COSTS if reference_costs is None else reference_costs
    validate_raw_costs(reference)
    low = min(log(float(value)) for value in reference.values())
    high = max(log(float(value)) for value in reference.values())
    if high <= low:
        raise ValueError("Reference cost range must be non-zero")
    return {name: (log(float(raw_costs[name])) - low) / (high - low) for name in MODALITIES}


@dataclass(frozen=True)
class CostSchedule:
    raw_usd: Mapping[str, float]
    lambda_usd: float = 1.0
    lambda_step: float = 0.0
    reference_usd: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        validate_raw_costs(self.raw_usd)
        if self.lambda_usd < 0 or self.lambda_step < 0:
            raise ValueError("Cost weights must be non-negative")

    @property
    def normalized(self) -> dict[str, float]:
        return normalized_log_costs(self.raw_usd, reference_costs=self.reference_usd)

    @property
    def planning(self) -> dict[str, float]:
        return {
            modality: self.lambda_usd * value + self.lambda_step
            for modality, value in self.normalized.items()
        }

    @property
    def full_raw_usd(self) -> float:
        return float(sum(float(self.raw_usd[name]) for name in MODALITIES))

    def shifted(self, changes: Mapping[str, float]) -> "CostSchedule":
        updated = dict(self.raw_usd)
        for modality, value in changes.items():
            if modality not in updated:
                raise KeyError(modality)
            updated[modality] = float(value)
        return CostSchedule(
            raw_usd=updated,
            lambda_usd=self.lambda_usd,
            lambda_step=self.lambda_step,
            reference_usd=self.reference_usd or self.raw_usd,
        )


def default_cost_schedule(*, lambda_usd: float = 1.0, lambda_step: float = 0.0) -> CostSchedule:
    schedule = CostSchedule(RAW_USD_COSTS, lambda_usd=lambda_usd, lambda_step=lambda_step)
    if schedule.full_raw_usd != FULL_REFERENCE_COST_USD:
        raise AssertionError("Configured ADNI modality costs must sum to $8,520")
    return schedule
