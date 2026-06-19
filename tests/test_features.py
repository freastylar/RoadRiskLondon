from __future__ import annotations

import json

import pytest

from roadrisk.data.build_processed_data import build_processed_data
from roadrisk.features.build_collision_features import (
    FORBIDDEN_FEATURES,
    build_modeling_table,
    validate_feature_schema,
)
from roadrisk.utils.validation import DataValidationError


def test_modeling_table_and_feature_schema(sample_config):
    build_processed_data(sample_config)
    df = build_modeling_table(sample_config)
    assert not df.empty
    assert df["is_ksi"].nunique() == 2
    schema_path = sample_config.project_root / "models/registry/feature_schema.json"
    assert schema_path.exists()
    assert not (set(df.columns) & FORBIDDEN_FEATURES - {"is_ksi", "collision_id"})
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate_feature_schema(schema, df)


def test_feature_schema_validation_rejects_duplicate_features(sample_config):
    build_processed_data(sample_config)
    df = build_modeling_table(sample_config)
    schema_path = sample_config.project_root / "models/registry/feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["boolean_features"] = [schema["numeric_features"][0]]
    schema["feature_columns"] = [
        *schema["numeric_features"],
        *schema["categorical_features"],
        *schema["boolean_features"],
    ]
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_feature_schema(schema, df)


def test_feature_schema_validation_rejects_leakage_feature(sample_config):
    build_processed_data(sample_config)
    df = build_modeling_table(sample_config)
    schema_path = sample_config.project_root / "models/registry/feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["numeric_features"] = [*schema["numeric_features"], "accident_severity"]
    schema["feature_columns"] = [
        *schema["numeric_features"],
        *schema["categorical_features"],
        *schema["boolean_features"],
    ]
    with pytest.raises(DataValidationError, match="leakage"):
        validate_feature_schema(schema, df)


def test_feature_schema_validation_rejects_missing_modeling_table_column(sample_config):
    build_processed_data(sample_config)
    df = build_modeling_table(sample_config)
    schema_path = sample_config.project_root / "models/registry/feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    with pytest.raises(DataValidationError, match="missing schema columns"):
        validate_feature_schema(schema, df.drop(columns=[schema["feature_columns"][0]]))
