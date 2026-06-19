from __future__ import annotations

import json

import pandas as pd
import pytest

from roadrisk.data.build_processed_data import build_processed_data
from roadrisk.features.build_collision_features import build_modeling_table
from roadrisk.models.train import (
    Split,
    _validate_training_input,
    train_model,
    validate_modeling_year_range,
)
from roadrisk.utils.validation import DataValidationError


def test_model_training_outputs_artifacts(sample_config):
    build_processed_data(sample_config)
    build_modeling_table(sample_config)
    metrics = train_model(sample_config.project_root, model_name="severity_model_mvp")
    assert metrics["test"]["pr_auc"] >= 0
    assert (sample_config.project_root / "models/registry/severity_model.joblib").exists()
    metrics_path = sample_config.project_root / "models/registry/model_metrics.json"
    importance_path = sample_config.project_root / "models/registry/feature_importance.parquet"
    calibration_path = sample_config.project_root / "models/registry/calibration_data.parquet"
    assert metrics_path.exists()
    assert importance_path.exists()
    assert calibration_path.exists()

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for split in ["validation", "test", "baseline_validation", "baseline_test"]:
        assert {"roc_auc", "pr_auc", "precision", "recall", "f1", "confusion_matrix"}.issubset(
            saved_metrics[split]
        )
        assert len(saved_metrics[split]["confusion_matrix"]) == 2
        assert len(saved_metrics[split]["confusion_matrix"][0]) == 2
    assert "test_slices" in saved_metrics
    assert saved_metrics["test_slices"]
    assert "test_borough_slices" in saved_metrics
    assert saved_metrics["test_borough_slices"]
    assert "test_thresholds" in saved_metrics
    assert saved_metrics["test_thresholds"]
    for row in saved_metrics["test_thresholds"]:
        assert {"threshold", "precision", "recall", "f1", "predicted_positive_rate"}.issubset(row)
        assert 0 <= row["threshold"] <= 1
        assert 0 <= row["precision"] <= 1
        assert 0 <= row["recall"] <= 1
        assert 0 <= row["f1"] <= 1
        assert 0 <= row["predicted_positive_rate"] <= 1
    for row in saved_metrics["test_slices"]:
        assert {"slice", "records", "ksi_count", "ksi_rate", "roc_auc", "pr_auc", "precision", "recall", "f1"}.issubset(row)
        assert row["records"] > 0
        assert 0 <= row["ksi_rate"] <= 1
    for row in saved_metrics["test_borough_slices"]:
        assert {"borough_code", "borough", "records", "ksi_count", "ksi_rate", "roc_auc", "pr_auc", "precision", "recall", "f1"}.issubset(row)
        assert row["borough_code"].startswith("E09")
        assert row["borough"]
        assert row["records"] > 0
        assert 0 <= row["ksi_rate"] <= 1

    importance = pd.read_parquet(importance_path)
    calibration = pd.read_parquet(calibration_path)
    assert not importance.empty
    assert {"feature", "importance"}.issubset(importance.columns)
    assert not calibration.empty
    assert {"mean_predicted_probability", "observed_fraction"}.issubset(calibration.columns)
    assert calibration["mean_predicted_probability"].between(0, 1).all()
    assert calibration["observed_fraction"].between(0, 1).all()


def test_validate_modeling_year_range_matches_feature_artifacts(sample_config):
    build_processed_data(sample_config)
    build_modeling_table(sample_config)
    validate_modeling_year_range(sample_config.project_root, 2020, 2024)
    with pytest.raises(DataValidationError, match="Requested training years"):
        validate_modeling_year_range(sample_config.project_root, 2021, 2024)


def _valid_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "collision_id": ["a", "b", "c", "d", "e", "f"],
            "accident_year": [2020, 2020, 2021, 2021, 2022, 2022],
            "is_ksi": [0, 1, 0, 1, 0, 1],
        }
    )


def test_training_input_validation_accepts_binary_target_and_unique_ids():
    split = Split(train_years=[2020], validation_years=[2021], test_years=[2022])
    _validate_training_input(_valid_training_frame(), split)


def test_training_input_validation_rejects_non_binary_target():
    split = Split(train_years=[2020], validation_years=[2021], test_years=[2022])
    df = _valid_training_frame()
    df.loc[0, "is_ksi"] = 2
    with pytest.raises(DataValidationError, match="binary"):
        _validate_training_input(df, split)


def test_training_input_validation_rejects_duplicate_collision_ids():
    split = Split(train_years=[2020], validation_years=[2021], test_years=[2022])
    df = _valid_training_frame()
    df.loc[1, "collision_id"] = "a"
    with pytest.raises(DataValidationError, match="unique"):
        _validate_training_input(df, split)


def test_training_input_validation_rejects_one_class_split():
    split = Split(train_years=[2020], validation_years=[2021], test_years=[2022])
    df = _valid_training_frame()
    df.loc[df["accident_year"] == 2022, "is_ksi"] = 0
    with pytest.raises(DataValidationError, match="test split"):
        _validate_training_input(df, split)
