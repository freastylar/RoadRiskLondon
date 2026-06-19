from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from roadrisk.utils.validation import DataValidationError


def _validated_binary_inputs(y_true: pd.Series, probabilities: np.ndarray) -> tuple[pd.Series, np.ndarray]:
    target = pd.Series(y_true).astype("Int64")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise DataValidationError("Metric target must contain only binary 0/1 values")
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1:
        raise DataValidationError("Metric probabilities must be a one-dimensional array")
    if len(target) != len(probs):
        raise DataValidationError("Metric target and probabilities must have the same length")
    if len(target) == 0:
        raise DataValidationError("Metric input must be non-empty")
    if len(set(target.astype(int))) != 2:
        raise DataValidationError("Metric target must contain both KSI and non-KSI classes")
    if not np.isfinite(probs).all():
        raise DataValidationError("Metric probabilities must be finite")
    if ((probs < 0) | (probs > 1)).any():
        raise DataValidationError("Metric probabilities must be between 0 and 1")
    return target.astype(int), probs


def binary_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    y_true, probabilities = _validated_binary_inputs(y_true, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "threshold": threshold,
    }
    return metrics


def calibration_table(y_true: pd.Series, probabilities: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true, probabilities = _validated_binary_inputs(y_true, probabilities)
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_probability": prob_pred, "observed_fraction": prob_true})


def threshold_metrics_table(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> list[dict]:
    y_true, probabilities = _validated_binary_inputs(y_true, probabilities)
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "predicted_positive_rate": float(predictions.mean()),
            }
        )
    return rows
