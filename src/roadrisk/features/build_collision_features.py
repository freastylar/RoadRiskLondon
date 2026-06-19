from __future__ import annotations

import json

import pandas as pd

from roadrisk.config import PipelineConfig
from roadrisk.data.decode_categories import is_ksi_from_severity
from roadrisk.features.build_casualty_features import build_casualty_flags
from roadrisk.features.spatial_features import add_simple_grid
from roadrisk.utils.paths import processed_dir, registry_dir, reports_dir
from roadrisk.utils.validation import DataValidationError, read_required_parquet, require_non_empty

FORBIDDEN_FEATURES = {
    "accident_index",
    "accident_reference",
    "accident_severity",
    "accident_severity_label",
    "casualty_severity",
    "is_ksi",
    "collision_id",
    "date",
    "time",
}

NUMERIC_CANDIDATES = [
    "accident_year",
    "month",
    "hour",
    "number_of_vehicles",
    "number_of_casualties",
    "speed_limit",
    "casualty_record_count",
    "grid_lon",
    "grid_lat",
]

CATEGORICAL_CANDIDATES = [
    "day_of_week",
    "road_type",
    "junction_detail",
    "junction_control",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "urban_or_rural_area",
    "local_authority_ons_district",
    "local_authority_district",
    "local_authority_highway",
]

BOOLEAN_CANDIDATES = [
    "weekend",
    "has_pedestrian",
    "has_cyclist",
    "has_motorcyclist",
    "has_child",
    "has_elderly",
]


def validate_feature_schema(schema: dict, df: pd.DataFrame | None = None) -> None:
    required_keys = {
        "target",
        "id_column",
        "numeric_features",
        "categorical_features",
        "boolean_features",
        "feature_columns",
        "forbidden_features",
        "year_range",
    }
    missing_keys = required_keys - set(schema)
    if missing_keys:
        raise DataValidationError(f"Feature schema missing keys: {sorted(missing_keys)}")
    numeric = list(schema["numeric_features"])
    categorical = list(schema["categorical_features"])
    boolean = list(schema["boolean_features"])
    feature_columns = list(schema["feature_columns"])
    combined = [*numeric, *categorical, *boolean]
    if len(combined) != len(set(combined)):
        raise DataValidationError("Feature schema has duplicate feature names")
    if feature_columns != combined:
        raise DataValidationError("Feature schema feature_columns must match grouped feature lists")
    leakage = sorted(set(feature_columns) & FORBIDDEN_FEATURES)
    if leakage:
        raise DataValidationError(f"Feature schema includes leakage columns: {leakage}")
    if schema["target"] != "is_ksi":
        raise DataValidationError("Feature schema target must be is_ksi")
    if schema["id_column"] != "collision_id":
        raise DataValidationError("Feature schema id_column must be collision_id")
    if len(schema["year_range"]) != 2 or schema["year_range"][0] > schema["year_range"][1]:
        raise DataValidationError("Feature schema year_range must be [start_year, end_year]")
    if df is not None:
        missing_columns = [column for column in ["collision_id", "is_ksi", *feature_columns] if column not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Modeling table missing schema columns: {missing_columns}")


def build_modeling_table(config: PipelineConfig) -> pd.DataFrame:
    proc = processed_dir(config.project_root)
    collisions = read_required_parquet(proc / "collisions_london.parquet")
    casualties = read_required_parquet(proc / "casualties_london.parquet")
    collisions = collisions[
        collisions["accident_year"].between(config.year_range.start_year, config.year_range.end_year)
    ].copy()
    require_non_empty(collisions, "model collision input")

    if "accident_severity_label" in collisions.columns:
        target_source = collisions["accident_severity_label"]
    elif "accident_severity" in collisions.columns:
        target_source = collisions["accident_severity"]
    else:
        raise DataValidationError("No verified accident severity column for target")
    collisions["is_ksi"] = is_ksi_from_severity(target_source)
    collisions = collisions[collisions["is_ksi"].isin([0, 1])].copy()
    if collisions["is_ksi"].nunique() != 2:
        raise DataValidationError("Target is_ksi must contain both classes")

    if "day_of_week" in collisions.columns:
        collisions["weekend"] = pd.to_numeric(collisions["day_of_week"], errors="coerce").isin([1, 7])

    flags = build_casualty_flags(casualties)
    model_df = collisions.merge(flags, on="collision_id", how="left")
    bool_cols = [column for column in BOOLEAN_CANDIDATES if column in model_df.columns]
    for column in bool_cols:
        model_df[column] = model_df[column].fillna(False).astype(int)
    if "casualty_record_count" in model_df.columns:
        model_df["casualty_record_count"] = model_df["casualty_record_count"].fillna(0)
    model_df = add_simple_grid(model_df)

    numeric = [column for column in NUMERIC_CANDIDATES if column in model_df.columns]
    categorical = [column for column in CATEGORICAL_CANDIDATES if column in model_df.columns]
    boolean = [column for column in BOOLEAN_CANDIDATES if column in model_df.columns]
    feature_columns = [*numeric, *categorical, *boolean]
    leakage = sorted(set(feature_columns) & FORBIDDEN_FEATURES)
    if leakage:
        raise DataValidationError(f"Leakage columns selected as features: {leakage}")
    if not feature_columns:
        raise DataValidationError("No verified feature columns available")

    output = model_df[["collision_id", "is_ksi", *feature_columns]].copy()
    out_path = proc / "model_collision_severity.parquet"
    output.to_parquet(out_path, index=False)

    schema = {
        "target": "is_ksi",
        "id_column": "collision_id",
        "numeric_features": numeric,
        "categorical_features": categorical,
        "boolean_features": boolean,
        "feature_columns": feature_columns,
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
        "year_range": [config.year_range.start_year, config.year_range.end_year],
    }
    validate_feature_schema(schema, output)
    reg = registry_dir(config.project_root)
    (reg / "feature_schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    report = "\n".join(
        [
            "# Modeling Table Report",
            "",
            f"- Rows: {len(output)}",
            f"- KSI rows: {int(output['is_ksi'].sum())}",
            f"- Slight rows: {int((output['is_ksi'] == 0).sum())}",
            f"- Years: {config.year_range.start_year}-{config.year_range.end_year}",
            f"- Numeric features: {numeric}",
            f"- Categorical features: {categorical}",
            f"- Boolean features: {boolean}",
            "",
        ]
    )
    (reports_dir(config.project_root) / "modeling_table_report.md").write_text(report, encoding="utf-8")
    return output
