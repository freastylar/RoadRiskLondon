from __future__ import annotations

import json

import pandas as pd
import pytest

from roadrisk.data.build_app_tables import (
    _add_severity_flags,
    _validate_borough_summary,
    _validate_collision_points,
    _validate_safety_map_yearly,
    _validate_severity_drivers,
    _validate_vulnerable_summary,
    build_app_artifacts,
)
from roadrisk.data.build_historical_trends import build_historical_trends
from roadrisk.data.build_processed_data import build_processed_data
from roadrisk.features.build_collision_features import build_modeling_table
from roadrisk.models.train import train_model
from roadrisk.utils.validation import DataValidationError


def test_app_artifacts_exist_and_non_empty(sample_config):
    build_processed_data(sample_config)
    build_modeling_table(sample_config)
    train_model(sample_config.project_root, model_name="severity_model_mvp")
    build_historical_trends(sample_config)
    paths = build_app_artifacts(sample_config, include_monitoring=True)
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
    priority = pd.read_parquet(paths["priority"])
    safety_map_yearly = pd.read_parquet(paths["safety_map_yearly"])
    borough = pd.read_parquet(paths["borough"])
    severity_drivers = pd.read_parquet(paths["severity_drivers"])
    model_metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    monitoring = json.loads(paths["monitoring"].read_text(encoding="utf-8"))
    assert priority["priority_score"].between(0, 1).all()
    assert {
        "observed_ksi_rate",
        "observed_severity_component",
        "priority_band",
        "priority_reason",
        "audit_focus",
    }.issubset(priority.columns)
    assert priority["observed_ksi_rate"].between(0, 1).all()
    assert priority["priority_reason"].str.len().gt(0).all()
    assert not safety_map_yearly.empty
    assert {
        "accident_year",
        "total_collisions",
        "ksi_collisions",
        "ksi_rate",
        "vulnerable_records",
        "vulnerable_user_share",
    }.issubset(safety_map_yearly.columns)
    assert safety_map_yearly["total_collisions"].gt(0).all()
    assert safety_map_yearly["ksi_rate"].between(0, 1).all()
    assert safety_map_yearly["vulnerable_user_share"].between(0, 1).all()
    assert not borough.empty
    assert {"borough", "ksi_rate", "total_collisions", "ksi_collisions"}.issubset(borough.columns)
    assert not severity_drivers.empty
    assert {
        "condition",
        "condition_label",
        "category",
        "category_label",
        "total_collisions",
        "fatal_collisions",
        "serious_collisions",
        "slight_collisions",
        "ksi_collisions",
        "ksi_rate",
    }.issubset(severity_drivers.columns)
    assert severity_drivers["ksi_rate"].between(0, 1).all()
    assert (
        borough["fatal_collisions"] + borough["serious_collisions"] == borough["ksi_collisions"]
    ).all()
    assert (
        borough["fatal_collisions"]
        + borough["serious_collisions"]
        + borough["slight_collisions"]
        == borough["total_collisions"]
    ).all()
    assert (
        severity_drivers["fatal_collisions"] + severity_drivers["serious_collisions"]
        == severity_drivers["ksi_collisions"]
    ).all()
    assert (
        severity_drivers["fatal_collisions"]
        + severity_drivers["serious_collisions"]
        + severity_drivers["slight_collisions"]
        == severity_drivers["total_collisions"]
    ).all()
    assert {
        "split",
        "class_balance_by_year",
        "missingness_by_year",
        "test_metrics",
        "known_limitations",
    }.issubset(monitoring)
    assert monitoring["class_balance_by_year"]
    assert monitoring["missingness_by_year"]
    assert {"roc_auc", "pr_auc", "precision", "recall", "f1"}.issubset(monitoring["test_metrics"])
    assert model_metrics["test_thresholds"]
    assert model_metrics["test_slices"]
    assert model_metrics["test_borough_slices"]
    for row in model_metrics["test_thresholds"]:
        assert {"threshold", "precision", "recall", "f1", "predicted_positive_rate"}.issubset(row)
        assert 0 <= row["threshold"] <= 1
        assert 0 <= row["predicted_positive_rate"] <= 1
    for row in model_metrics["test_slices"]:
        assert {"slice", "records", "ksi_count", "ksi_rate", "roc_auc", "pr_auc"}.issubset(row)
        assert row["records"] > 0
    for row in model_metrics["test_borough_slices"]:
        assert {"borough_code", "borough", "records", "ksi_count", "ksi_rate", "roc_auc", "pr_auc"}.issubset(row)
        assert row["borough_code"].startswith("E09")
        assert row["records"] > 0


def test_app_severity_flags_reconcile_ksi_definition():
    flagged = _add_severity_flags(
        pd.DataFrame({"accident_severity_label": ["Fatal", "Serious", "Slight"]})
    )
    assert flagged["fatal_collision"].tolist() == [1, 0, 0]
    assert flagged["serious_collision"].tolist() == [0, 1, 0]
    assert flagged["slight_collision"].tolist() == [0, 0, 1]
    assert flagged["ksi_collision"].tolist() == [1, 1, 0]
    assert (
        flagged["fatal_collision"] + flagged["serious_collision"] + flagged["slight_collision"]
        == 1
    ).all()


def test_app_severity_flags_reject_unmapped_labels():
    with pytest.raises(DataValidationError, match="unmapped severity"):
        _add_severity_flags(pd.DataFrame({"accident_severity_label": ["Fatal", "Unknown"]}))


def test_borough_summary_validation_accepts_reconciled_summary():
    borough = pd.DataFrame(
        {
            "accident_year": [2024, 2024],
            "borough_code": ["E09000001", "E09000002"],
            "borough": ["City of London", "Barking and Dagenham"],
            "total_collisions": [10, 30],
            "fatal_collisions": [1, 2],
            "serious_collisions": [2, 4],
            "slight_collisions": [7, 24],
            "ksi_collisions": [3, 6],
            "ksi_rate": [0.3, 0.2],
            "share_of_london_collisions": [0.25, 0.75],
            "share_of_london_ksi": [1 / 3, 2 / 3],
        }
    )
    _validate_borough_summary(borough)


def test_borough_summary_validation_rejects_bad_shares():
    borough = pd.DataFrame(
        {
            "accident_year": [2024, 2024],
            "borough_code": ["E09000001", "E09000002"],
            "borough": ["City of London", "Barking and Dagenham"],
            "total_collisions": [10, 30],
            "fatal_collisions": [1, 2],
            "serious_collisions": [2, 4],
            "slight_collisions": [7, 24],
            "ksi_collisions": [3, 6],
            "ksi_rate": [0.3, 0.2],
            "share_of_london_collisions": [0.2, 0.7],
            "share_of_london_ksi": [1 / 3, 2 / 3],
        }
    )
    with pytest.raises(DataValidationError, match="collision shares"):
        _validate_borough_summary(borough)


def test_vulnerable_summary_validation_accepts_valid_summary():
    summary = pd.DataFrame(
        {
            "accident_year": [2024, 2024],
            "road_user_group": ["pedestrian", "cyclist"],
            "records": [10, 20],
            "ksi_count": [3, 4],
            "ksi_rate": [0.3, 0.2],
        }
    )
    _validate_vulnerable_summary(summary)


def test_vulnerable_summary_validation_rejects_unknown_group():
    summary = pd.DataFrame(
        {
            "accident_year": [2024],
            "road_user_group": ["invented_group"],
            "records": [10],
            "ksi_count": [3],
            "ksi_rate": [0.3],
        }
    )
    with pytest.raises(DataValidationError, match="unknown groups"):
        _validate_vulnerable_summary(summary)


def test_vulnerable_summary_validation_rejects_bad_ksi_rate():
    summary = pd.DataFrame(
        {
            "accident_year": [2024],
            "road_user_group": ["pedestrian"],
            "records": [10],
            "ksi_count": [3],
            "ksi_rate": [0.9],
        }
    )
    with pytest.raises(DataValidationError, match="KSI rates"):
        _validate_vulnerable_summary(summary)


def test_collision_points_validation_accepts_valid_points():
    points = pd.DataFrame(
        {
            "collision_id": ["a", "b"],
            "accident_year": [2024, 2024],
            "longitude": [-0.1, -0.2],
            "latitude": [51.5, 51.55],
            "predicted_ksi_risk": [0.2, 0.8],
            "is_ksi": [0, 1],
        }
    )
    _validate_collision_points(points)


def test_collision_points_validation_rejects_bad_coordinates():
    points = pd.DataFrame(
        {
            "collision_id": ["a"],
            "accident_year": [2024],
            "longitude": [-3.0],
            "latitude": [51.5],
            "predicted_ksi_risk": [0.2],
            "is_ksi": [0],
        }
    )
    with pytest.raises(DataValidationError, match="coordinates"):
        _validate_collision_points(points)


def test_safety_map_yearly_validation_accepts_valid_summary():
    summary = pd.DataFrame(
        {
            "accident_year": [2024],
            "total_collisions": [100],
            "ksi_collisions": [20],
            "mean_predicted_ksi_risk": [0.4],
            "vulnerable_records": [50],
            "ksi_rate": [0.2],
            "vulnerable_user_share": [0.5],
        }
    )
    _validate_safety_map_yearly(summary)


def test_safety_map_yearly_validation_rejects_bad_rate():
    summary = pd.DataFrame(
        {
            "accident_year": [2024],
            "total_collisions": [100],
            "ksi_collisions": [20],
            "mean_predicted_ksi_risk": [0.4],
            "vulnerable_records": [50],
            "ksi_rate": [0.9],
            "vulnerable_user_share": [0.5],
        }
    )
    with pytest.raises(DataValidationError, match="KSI rates"):
        _validate_safety_map_yearly(summary)


def _valid_severity_drivers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "accident_year": [2024, 2024],
            "condition": ["weather_conditions", "weather_conditions"],
            "condition_label": ["Weather", "Weather"],
            "category": ["1", "2"],
            "category_label": ["Fine", "Rain"],
            "total_collisions": [10, 30],
            "fatal_collisions": [1, 2],
            "serious_collisions": [2, 4],
            "slight_collisions": [7, 24],
            "ksi_collisions": [3, 6],
            "ksi_rate": [0.3, 0.2],
            "serious_harm_score": [16, 32],
            "harm_score_per_collision": [1.6, 32 / 30],
            "share_of_condition_year": [0.25, 0.75],
        }
    )


def test_severity_drivers_validation_accepts_valid_summary():
    _validate_severity_drivers(_valid_severity_drivers())


def test_severity_drivers_validation_rejects_bad_harm_score():
    drivers = _valid_severity_drivers()
    drivers.loc[0, "serious_harm_score"] = 999
    with pytest.raises(DataValidationError, match="harm scores"):
        _validate_severity_drivers(drivers)


def test_severity_drivers_validation_rejects_bad_condition_year_shares():
    drivers = _valid_severity_drivers()
    drivers.loc[0, "share_of_condition_year"] = 0.4
    with pytest.raises(DataValidationError, match="shares"):
        _validate_severity_drivers(drivers)
