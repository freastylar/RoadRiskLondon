from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roadrisk.models.evaluate import binary_metrics, calibration_table, threshold_metrics_table
from roadrisk.utils.validation import DataValidationError


def test_binary_metrics_are_finite_for_valid_inputs():
    metrics = binary_metrics(pd.Series([0, 1, 0, 1]), np.array([0.1, 0.8, 0.4, 0.7]))
    assert {"roc_auc", "pr_auc", "precision", "recall", "f1", "confusion_matrix", "threshold"}.issubset(
        metrics
    )
    for key in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
        assert 0 <= metrics[key] <= 1


def test_calibration_and_threshold_tables_validate_metric_inputs():
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.7])
    calibration = calibration_table(y_true, probabilities)
    thresholds = threshold_metrics_table(y_true, probabilities)
    assert not calibration.empty
    assert calibration["mean_predicted_probability"].between(0, 1).all()
    assert calibration["observed_fraction"].between(0, 1).all()
    assert thresholds
    assert all(0 <= row["predicted_positive_rate"] <= 1 for row in thresholds)


@pytest.mark.parametrize(
    ("target", "probabilities", "message"),
    [
        ([1, 1, 1], [0.7, 0.8, 0.9], "both KSI and non-KSI"),
        ([0, 1], [0.2], "same length"),
        ([0, 1], [0.2, 1.2], "between 0 and 1"),
        ([0, 1], [0.2, np.nan], "finite"),
        ([0, 2], [0.2, 0.8], "binary"),
    ],
)
def test_binary_metrics_reject_invalid_inputs(target, probabilities, message):
    with pytest.raises(DataValidationError, match=message):
        binary_metrics(pd.Series(target), np.array(probabilities))
