from __future__ import annotations

import pytest

from roadrisk.models.monitor import _validate_monitoring_summary
from roadrisk.utils.validation import DataValidationError


def _valid_summary() -> dict:
    return {
        "model_name": "severity_model_mvp",
        "model_type": "logistic_regression_balanced",
        "split": {
            "train_years": [2020, 2021, 2022],
            "validation_years": [2023],
            "test_years": [2024],
        },
        "class_balance_by_year": [
            {"accident_year": 2020, "records": 100, "ksi_count": 10, "ksi_rate": 0.1},
            {"accident_year": 2021, "records": 120, "ksi_count": 18, "ksi_rate": 0.15},
        ],
        "missingness_by_year": [
            {"accident_year": 2020, "mean_missing_share": 0.02},
            {"accident_year": 2021, "mean_missing_share": 0.01},
        ],
        "test_metrics": {
            "roc_auc": 0.7,
            "pr_auc": 0.32,
            "precision": 0.25,
            "recall": 0.76,
            "f1": 0.38,
            "threshold": 0.5,
        },
        "known_limitations": ["Reported personal-injury collisions only."],
    }


def test_monitoring_summary_validation_accepts_valid_summary():
    _validate_monitoring_summary(_valid_summary())


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda summary: summary.pop("test_metrics"), "missing keys"),
        (lambda summary: summary["split"].update({"test_years": []}), "test_years"),
        (lambda summary: summary["class_balance_by_year"][0].update({"records": 0}), "records"),
        (
            lambda summary: summary["class_balance_by_year"][0].update({"ksi_count": 101}),
            "KSI counts",
        ),
        (
            lambda summary: summary["class_balance_by_year"][0].update({"ksi_rate": 1.2}),
            "KSI rates",
        ),
        (
            lambda summary: summary["missingness_by_year"][0].update({"mean_missing_share": -0.1}),
            "missingness",
        ),
        (lambda summary: summary["test_metrics"].update({"pr_auc": 1.2}), "pr_auc"),
    ],
)
def test_monitoring_summary_validation_rejects_invalid_summary(mutator, message):
    summary = _valid_summary()
    mutator(summary)
    with pytest.raises(DataValidationError, match=message):
        _validate_monitoring_summary(summary)
