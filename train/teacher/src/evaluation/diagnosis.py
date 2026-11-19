"""Diagnostic and ordinal calibration metrics for CN/MCI/AD."""

from __future__ import annotations

from typing import Sequence

import numpy as np


CLASS_NAMES = ("CN", "MCI", "AD")
NUM_CLASSES = len(CLASS_NAMES)


def _labels(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array.astype(float)).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError(f"{name} must contain integer labels")
    array = array.astype(np.int64)
    if np.any((array < 0) | (array >= NUM_CLASSES)):
        raise ValueError(f"{name} labels must be CN=0, MCI=1, or AD=2")
    return array


def _probabilities(values: np.ndarray, n_samples: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (n_samples, NUM_CLASSES):
        raise ValueError(f"probabilities must have shape ({n_samples}, {NUM_CLASSES})")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every probability row must have positive mass")
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to one")
    return probabilities / row_sums


def confusion_matrix_3(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    true = _labels(y_true, "y_true")
    pred = _labels(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def classwise_sensitivity_specificity(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> tuple[dict[str, float], dict[str, float]]:
    matrix = confusion_matrix_3(y_true, y_pred)
    sensitivity: dict[str, float] = {}
    specificity: dict[str, float] = {}
    total = int(matrix.sum())
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = total - tp - fn - fp
        sensitivity[name] = float(tp / (tp + fn)) if tp + fn else float("nan")
        specificity[name] = float(tn / (tn + fp)) if tn + fp else float("nan")
    return sensitivity, specificity


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    sensitivity, _ = classwise_sensitivity_specificity(y_true, y_pred)
    values = np.asarray(list(sensitivity.values()), dtype=float)
    return float(np.nanmean(values))


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    matrix = confusion_matrix_3(y_true, y_pred)
    scores: list[float] = []
    for index in range(NUM_CLASSES):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        denominator = 2.0 * tp + fp + fn
        scores.append(2.0 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def ordinal_mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true = _labels(y_true, "y_true")
    pred = _labels(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.abs(true - pred).mean())


def extreme_error_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true = _labels(y_true, "y_true")
    pred = _labels(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float((np.abs(true - pred) == 2).mean())


def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Cohen's kappa with quadratic disagreement weights and fixed class order."""

    observed_counts = confusion_matrix_3(y_true, y_pred).astype(np.float64)
    n = observed_counts.sum()
    observed = observed_counts / n
    expected = np.outer(observed_counts.sum(axis=1), observed_counts.sum(axis=0)) / (n * n)
    indices = np.arange(NUM_CLASSES, dtype=float)
    weights = ((indices[:, None] - indices[None, :]) / (NUM_CLASSES - 1)) ** 2
    observed_disagreement = float((weights * observed).sum())
    expected_disagreement = float((weights * expected).sum())
    if expected_disagreement <= np.finfo(float).eps:
        return 1.0 if observed_disagreement <= np.finfo(float).eps else float("nan")
    return float(1.0 - observed_disagreement / expected_disagreement)


def expected_calibration_error(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Top-label ECE with equal-width confidence bins over ``[0, 1]``."""

    true = _labels(y_true, "y_true")
    probs = _probabilities(probabilities, len(true))
    if int(n_bins) <= 0:
        raise ValueError("n_bins must be positive")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == true
    bin_index = np.minimum((confidence * int(n_bins)).astype(int), int(n_bins) - 1)
    ece = 0.0
    for index in range(int(n_bins)):
        selected = bin_index == index
        if selected.any():
            ece += selected.mean() * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return float(ece)


def multiclass_brier_score(y_true: Sequence[int], probabilities: np.ndarray) -> float:
    """Mean summed squared probability error (range 0 to 2 for three classes)."""

    true = _labels(y_true, "y_true")
    probs = _probabilities(probabilities, len(true))
    one_hot = np.eye(NUM_CLASSES, dtype=float)[true]
    return float(np.square(probs - one_hot).sum(axis=1).mean())


def diagnosis_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int] | np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    *,
    n_bins: int = 15,
) -> dict[str, object]:
    """Return all mandatory diagnostic metrics in one audit-friendly mapping."""

    true = _labels(y_true, "y_true")
    if y_pred is not None and np.asarray(y_pred).ndim == 2 and probabilities is None:
        probabilities = np.asarray(y_pred)
        y_pred = None
    probs = None if probabilities is None else _probabilities(probabilities, len(true))
    if y_pred is None:
        if probs is None:
            raise ValueError("Provide y_pred or probabilities")
        pred = probs.argmax(axis=1)
    else:
        pred = _labels(y_pred, "y_pred")
    if pred.shape != true.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    sensitivity, specificity = classwise_sensitivity_specificity(true, pred)
    matrix = confusion_matrix_3(true, pred)
    result: dict[str, object] = {
        "balanced_accuracy": balanced_accuracy(true, pred),
        "macro_f1": macro_f1(true, pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ordinal_mae": ordinal_mae(true, pred),
        "extreme_error_rate": extreme_error_rate(true, pred),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(true, pred),
        "confusion_matrix": matrix.tolist(),
        "n_samples": int(len(true)),
        "class_counts": {name: int((true == index).sum()) for index, name in enumerate(CLASS_NAMES)},
    }
    if probs is not None:
        result["ece"] = expected_calibration_error(true, probs, n_bins=n_bins)
        result["brier_score"] = multiclass_brier_score(true, probs)
    return result


# Common short aliases.
brier_score = multiclass_brier_score
ece = expected_calibration_error


__all__ = [
    "CLASS_NAMES",
    "balanced_accuracy",
    "brier_score",
    "classwise_sensitivity_specificity",
    "confusion_matrix_3",
    "diagnosis_metrics",
    "ece",
    "expected_calibration_error",
    "extreme_error_rate",
    "macro_f1",
    "multiclass_brier_score",
    "ordinal_mae",
    "quadratic_weighted_kappa",
]
