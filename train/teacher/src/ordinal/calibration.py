from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from src.utils.io import atomic_write_json

from .losses import ordinal_nll
from .model import ordinal_probabilities


class CalibrationError(ValueError):
    """Raised when calibration would violate validation-only fitting."""


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    fit_split: str
    state_count: int
    patient_count: int
    nll_before: float
    nll_after: float
    calibrated_kappa1: float | None = None
    calibrated_kappa2: float | None = None
    method: str = "ordinal_temperature"

    def __post_init__(self) -> None:
        numeric = (
            self.temperature,
            self.nll_before,
            self.nll_after,
        )
        if not all(np.isfinite(float(value)) for value in numeric):
            raise CalibrationError("Calibration scalars must be finite")
        if float(self.temperature) <= 0:
            raise CalibrationError("Calibration temperature must be positive")
        if self.fit_split != "validation":
            raise CalibrationError("Calibration artifact must be fitted on validation")
        if int(self.state_count) <= 0 or int(self.patient_count) <= 0:
            raise CalibrationError("Calibration counts must be positive")
        supplied = (self.calibrated_kappa1 is not None, self.calibrated_kappa2 is not None)
        if supplied[0] != supplied[1]:
            raise CalibrationError("Calibrated ordinal thresholds must be supplied as a pair")
        if supplied[0]:
            first = float(self.calibrated_kappa1)
            second = float(self.calibrated_kappa2)
            if not np.isfinite(first) or not np.isfinite(second) or not first < second:
                raise CalibrationError("Calibrated ordinal thresholds must be finite and ordered")

    def resolved_thresholds(
        self,
        kappa1: torch.Tensor | float,
        kappa2: torch.Tensor | float,
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        if self.calibrated_kappa1 is None:
            return kappa1, kappa2
        return float(self.calibrated_kappa1), float(self.calibrated_kappa2)

    def apply(
        self,
        severity: torch.Tensor,
        kappa1: torch.Tensor,
        kappa2: torch.Tensor,
    ) -> torch.Tensor:
        kappa1, kappa2 = self.resolved_thresholds(kappa1, kappa2)
        return ordinal_probabilities(
            severity, kappa1, kappa2, temperature=self.temperature
        )

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureCalibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def evaluate_partial_calibration_gate(
    calibration: TemperatureCalibration,
    calibrated_metrics: Mapping[str, Any],
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared validation-only Stage-1 calibration sanity gate."""

    required = {"version", "maximum_ece", "maximum_brier_score", "nll_tolerance"}
    missing = sorted(required - set(gate_config))
    if missing:
        raise CalibrationError(
            "Partial-state calibration gate is not fully predeclared; missing="
            + ", ".join(missing)
        )
    if calibration.fit_split != "validation":
        raise CalibrationError("Calibration acceptance may use validation only")
    maximum_ece = float(gate_config["maximum_ece"])
    maximum_brier = float(gate_config["maximum_brier_score"])
    nll_tolerance = float(gate_config["nll_tolerance"])
    ece = float(calibrated_metrics["ece"])
    brier = float(calibrated_metrics["brier_score"])
    values = (maximum_ece, maximum_brier, nll_tolerance, ece, brier)
    if not all(np.isfinite(value) for value in values):
        raise CalibrationError("Calibration gate thresholds and metrics must be finite")
    if not 0.0 <= maximum_ece <= 1.0:
        raise CalibrationError("maximum_ece must be between zero and one")
    if not 0.0 <= maximum_brier <= 2.0:
        raise CalibrationError("maximum_brier_score must be between zero and two")
    if nll_tolerance < 0.0:
        raise CalibrationError("nll_tolerance must be non-negative")
    checks = {
        "nll_not_worse": bool(
            calibration.nll_after <= calibration.nll_before + nll_tolerance
        ),
        "ece_within_limit": bool(ece <= maximum_ece),
        "brier_within_limit": bool(brier <= maximum_brier),
    }
    return {
        "version": str(gate_config["version"]),
        "fit_split": calibration.fit_split,
        "thresholds": {
            "maximum_ece": maximum_ece,
            "maximum_brier_score": maximum_brier,
            "nll_tolerance": nll_tolerance,
        },
        "measured": {
            "nll_before": float(calibration.nll_before),
            "nll_after": float(calibration.nll_after),
            "ece": ece,
            "brier_score": brier,
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def fit_validation_temperature(
    severity: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray | Sequence[int],
    kappa1: torch.Tensor | float,
    kappa2: torch.Tensor | float,
    *,
    split_name: str,
    max_iterations: int = 100,
    initial_temperature: float = 1.0,
    patient_count: int | None = None,
    calibrate_thresholds: bool = True,
) -> TemperatureCalibration:
    """Fit a validation-only ordered calibration map.

    By default this jointly fits one positive temperature and two ordered
    thresholds.  This remains a pure cumulative ordinal posterior, while
    allowing the validation cohort's class boundaries to be recalibrated.
    Set ``calibrate_thresholds=False`` only for legacy scalar-temperature
    compatibility.
    """

    if split_name != "validation":
        raise CalibrationError(
            "Temperature calibration must be fitted on the validation split only; "
            f"received {split_name!r}"
        )
    z = torch.as_tensor(severity, dtype=torch.float64).reshape(-1).detach()
    y = torch.as_tensor(labels, dtype=torch.long).reshape(-1).detach()
    if z.shape != y.shape or len(z) == 0:
        raise CalibrationError("severity and labels must be aligned and non-empty")
    if bool(torch.any((y < 0) | (y > 2)).item()):
        raise CalibrationError("labels must be CN=0, MCI=1, AD=2")
    threshold1 = torch.as_tensor(kappa1, dtype=torch.float64).detach()
    threshold2 = torch.as_tensor(kappa2, dtype=torch.float64).detach()
    if not bool((threshold1 < threshold2).item()):
        raise CalibrationError("Ordinal thresholds are not ordered")

    initial = max(float(initial_temperature), 1e-3)
    log_temperature = torch.tensor(np.log(initial), dtype=torch.float64, requires_grad=True)
    parameters = [log_temperature]
    calibrated_threshold1 = threshold1.clone()
    raw_gap: torch.Tensor | None = None
    if calibrate_thresholds:
        calibrated_threshold1 = threshold1.clone().requires_grad_(True)
        initial_gap = max(float((threshold2 - threshold1).item()), 1e-6)
        # Stable inverse softplus for the ordered threshold gap.
        inverse_gap = initial_gap + np.log(-np.expm1(-initial_gap))
        raw_gap = torch.tensor(inverse_gap, dtype=torch.float64, requires_grad=True)
        parameters.extend((calibrated_threshold1, raw_gap))
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=0.25,
        max_iter=int(max_iterations),
        line_search_fn="strong_wolfe",
    )
    with torch.no_grad():
        before = float(ordinal_nll(ordinal_probabilities(z, threshold1, threshold2), y))

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 100.0)
        fitted_threshold1 = calibrated_threshold1
        fitted_threshold2 = (
            fitted_threshold1 + F.softplus(raw_gap) + 1e-6
            if raw_gap is not None
            else threshold2
        )
        loss = ordinal_nll(
            ordinal_probabilities(
                z,
                fitted_threshold1,
                fitted_threshold2,
                temperature=temperature,
            ),
            y,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(1e-3, 100.0))
    fitted_threshold1 = calibrated_threshold1.detach()
    fitted_threshold2 = (
        fitted_threshold1 + F.softplus(raw_gap.detach()) + 1e-6
        if raw_gap is not None
        else threshold2
    )
    with torch.no_grad():
        after = float(
            ordinal_nll(
                ordinal_probabilities(
                    z,
                    fitted_threshold1,
                    fitted_threshold2,
                    temperature=temperature,
                ),
                y,
            )
        )
    # Optimizer line search can rarely finish a few ulps above the identity result.
    if after > before:
        temperature = 1.0
        after = before
        fitted_threshold1 = threshold1
        fitted_threshold2 = threshold2
    return TemperatureCalibration(
        temperature=temperature,
        fit_split="validation",
        state_count=len(y),
        patient_count=int(patient_count if patient_count is not None else len(y)),
        nll_before=before,
        nll_after=after,
        calibrated_kappa1=(
            float(fitted_threshold1) if calibrate_thresholds else None
        ),
        calibrated_kappa2=(
            float(fitted_threshold2) if calibrate_thresholds else None
        ),
        method=(
            "ordinal_temperature_thresholds"
            if calibrate_thresholds
            else "ordinal_temperature"
        ),
    )
