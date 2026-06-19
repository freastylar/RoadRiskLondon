from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from roadrisk.utils.paths import app_data_dir, processed_dir, registry_dir, reports_dir
from roadrisk.utils.validation import DataValidationError, read_required_json, read_required_parquet


def _validate_monitoring_summary(summary: dict) -> None:
    required = {
        "model_name",
        "model_type",
        "split",
        "class_balance_by_year",
        "missingness_by_year",
        "test_metrics",
        "known_limitations",
    }
    missing = required - set(summary)
    if missing:
        raise DataValidationError(f"Monitoring summary missing keys: {sorted(missing)}")

    split = summary["split"] or {}
    for key in ["train_years", "validation_years", "test_years"]:
        if not split.get(key):
            raise DataValidationError(f"Monitoring summary split missing {key}")

    class_balance = pd.DataFrame(summary["class_balance_by_year"])
    if class_balance.empty:
        raise DataValidationError("Monitoring class balance must be non-empty")
    required_balance = {"accident_year", "records", "ksi_count", "ksi_rate"}
    if not required_balance.issubset(class_balance.columns):
        raise DataValidationError("Monitoring class balance missing required columns")
    if not class_balance["records"].gt(0).all():
        raise DataValidationError("Monitoring records must be positive")
    if not class_balance["ksi_count"].between(0, class_balance["records"]).all():
        raise DataValidationError("Monitoring KSI counts must be between 0 and records")
    if not class_balance["ksi_rate"].between(0, 1).all():
        raise DataValidationError("Monitoring KSI rates must be between 0 and 1")

    missingness = pd.DataFrame(summary["missingness_by_year"])
    if missingness.empty:
        raise DataValidationError("Monitoring missingness must be non-empty")
    if not {"accident_year", "mean_missing_share"}.issubset(missingness.columns):
        raise DataValidationError("Monitoring missingness missing required columns")
    if not missingness["mean_missing_share"].between(0, 1).all():
        raise DataValidationError("Monitoring missingness must be between 0 and 1")

    test_metrics = summary["test_metrics"] or {}
    metric_keys = {"roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"}
    if not metric_keys.issubset(test_metrics):
        raise DataValidationError("Monitoring test metrics missing required fields")
    for key in metric_keys:
        value = test_metrics[key]
        if not isinstance(value, int | float) or not 0 <= value <= 1:
            raise DataValidationError(f"Monitoring test metric {key} must be between 0 and 1")


def build_monitoring_summary(root: Path) -> dict:
    df = read_required_parquet(processed_dir(root) / "model_collision_severity.parquet")
    metrics = read_required_json(registry_dir(root) / "model_metrics.json")
    class_balance = (
        df.groupby("accident_year")["is_ksi"]
        .agg(records="count", ksi_rate="mean", ksi_count="sum")
        .reset_index()
        .to_dict(orient="records")
    )
    missingness = (
        df.groupby("accident_year")
        .apply(lambda group: group.isna().mean(numeric_only=False).mean(), include_groups=False)
        .rename("mean_missing_share")
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "model_name": metrics.get("model_name", "severity_model"),
        "model_type": metrics.get("model_type"),
        "split": metrics.get("split"),
        "class_balance_by_year": class_balance,
        "missingness_by_year": missingness,
        "test_metrics": metrics.get("test"),
        "known_limitations": [
            "Reported personal-injury collisions only.",
            "Severity reporting changed over time.",
            "Predictions are statistical associations, not causal estimates.",
        ],
    }
    _validate_monitoring_summary(summary)
    out = app_data_dir(root) / "monitoring_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (reports_dir(root) / "monitoring_report.md").write_text(
        "# Monitoring Summary\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return summary
