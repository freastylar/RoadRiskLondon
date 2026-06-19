from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from roadrisk.config import PipelineConfig
from roadrisk.data.borough_boundaries import ensure_borough_boundaries
from roadrisk.data.decode_categories import (
    CASUALTY_TYPE_GROUPS,
    CONDITION_FIELD_LABELS,
    condition_category_label,
    condition_field_label,
    is_ksi_from_severity,
)
from roadrisk.data.filter_london import LONDON_BOROUGH_BY_ONS
from roadrisk.features.priority_score import compute_priority_scores
from roadrisk.features.spatial_features import add_simple_grid
from roadrisk.models.monitor import build_monitoring_summary
from roadrisk.models.predict import predict_probabilities
from roadrisk.utils.paths import app_data_dir, processed_dir, registry_dir
from roadrisk.utils.validation import (
    DataValidationError,
    read_required_json,
    read_required_parquet,
    require_columns,
    require_file,
    require_non_empty,
)


def _copy_json(source: Path, destination: Path) -> None:
    require_file(source)
    shutil.copyfile(source, destination)


def _add_severity_flags(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["accident_severity_label"], "severity summary input")
    flagged = df.copy()
    decoded_target = is_ksi_from_severity(flagged["accident_severity_label"])
    if decoded_target.isna().any():
        bad_values = sorted(
            flagged.loc[decoded_target.isna(), "accident_severity_label"].astype(str).unique()
        )
        msg = f"Severity summary input contains unmapped severity labels: {bad_values}"
        raise DataValidationError(msg)
    flagged["fatal_collision"] = flagged["accident_severity_label"].eq("Fatal").astype(int)
    flagged["serious_collision"] = flagged["accident_severity_label"].eq("Serious").astype(int)
    flagged["slight_collision"] = flagged["accident_severity_label"].eq("Slight").astype(int)
    flagged["ksi_collision"] = decoded_target.astype(int)
    return flagged


def _validate_borough_summary(borough: pd.DataFrame) -> None:
    require_non_empty(borough, "borough summary")
    required = {
        "accident_year",
        "borough_code",
        "borough",
        "total_collisions",
        "fatal_collisions",
        "serious_collisions",
        "slight_collisions",
        "ksi_collisions",
        "ksi_rate",
        "share_of_london_collisions",
        "share_of_london_ksi",
    }
    missing = required - set(borough.columns)
    if missing:
        raise DataValidationError(f"Borough summary missing columns: {sorted(missing)}")
    unknown_codes = sorted(set(borough["borough_code"]) - set(LONDON_BOROUGH_BY_ONS))
    if unknown_codes:
        raise DataValidationError(f"Borough summary has non-London borough codes: {unknown_codes}")
    if not borough["total_collisions"].gt(0).all():
        raise DataValidationError("Borough summary total collisions must be positive")
    if not (
        borough["fatal_collisions"] + borough["serious_collisions"] == borough["ksi_collisions"]
    ).all():
        raise DataValidationError("Borough summary KSI totals must equal Fatal plus Serious")
    if not (
        borough["fatal_collisions"]
        + borough["serious_collisions"]
        + borough["slight_collisions"]
        == borough["total_collisions"]
    ).all():
        raise DataValidationError("Borough summary severity totals must equal total collisions")
    expected_rate = borough["ksi_collisions"] / borough["total_collisions"]
    if not expected_rate.round(12).eq(borough["ksi_rate"].round(12)).all():
        raise DataValidationError("Borough summary KSI rates do not match counts")
    for column in ["ksi_rate", "share_of_london_collisions", "share_of_london_ksi"]:
        if not borough[column].between(0, 1).all():
            raise DataValidationError(f"Borough summary {column} must be between 0 and 1")
    collision_shares = borough.groupby("accident_year")["share_of_london_collisions"].sum()
    if not collision_shares.round(12).eq(1).all():
        raise DataValidationError("Borough collision shares must sum to 1 by year")
    ksi_shares = borough.groupby("accident_year")["share_of_london_ksi"].sum()
    if not ksi_shares.round(12).eq(1).all():
        raise DataValidationError("Borough KSI shares must sum to 1 by year")


def _build_borough_summary(collisions: pd.DataFrame) -> pd.DataFrame:
    if "local_authority_ons_district" not in collisions.columns:
        msg = "Borough summary requires local_authority_ons_district"
        raise DataValidationError(msg)
    summary_input = collisions.copy()
    summary_input["borough_code"] = summary_input["local_authority_ons_district"].astype(str).str.upper()
    summary_input["borough"] = summary_input["borough_code"].map(LONDON_BOROUGH_BY_ONS)
    summary_input = summary_input.dropna(subset=["borough"]).copy()
    require_non_empty(summary_input, "borough summary input")
    summary_input = _add_severity_flags(summary_input)
    borough = (
        summary_input.groupby(["accident_year", "borough_code", "borough"], dropna=False)
        .agg(
            total_collisions=("collision_id", "count"),
            fatal_collisions=("fatal_collision", "sum"),
            serious_collisions=("serious_collision", "sum"),
            slight_collisions=("slight_collision", "sum"),
            ksi_collisions=("ksi_collision", "sum"),
        )
        .reset_index()
    )
    borough["ksi_rate"] = borough["ksi_collisions"] / borough["total_collisions"]
    london_totals = borough.groupby("accident_year")[["total_collisions", "ksi_collisions"]].transform(
        "sum"
    )
    borough["share_of_london_collisions"] = (
        borough["total_collisions"] / london_totals["total_collisions"]
    )
    borough["share_of_london_ksi"] = borough["ksi_collisions"] / london_totals["ksi_collisions"]
    borough = borough.sort_values(["accident_year", "borough"])
    _validate_borough_summary(borough)
    return borough


def _build_severity_drivers(collisions: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        collisions,
        ["collision_id", "accident_year", "accident_severity_label"],
        "severity driver input",
    )
    available_conditions = [field for field in CONDITION_FIELD_LABELS if field in collisions.columns]
    if not available_conditions:
        msg = "No configured severity-driver condition fields exist in processed collisions"
        raise ValueError(msg)

    base = _add_severity_flags(collisions)
    summaries: list[pd.DataFrame] = []
    for condition in available_conditions:
        condition_input = base[
            [
                "accident_year",
                "collision_id",
                condition,
                "fatal_collision",
                "serious_collision",
                "slight_collision",
                "ksi_collision",
            ]
        ].copy()
        condition_input["condition"] = condition
        condition_input["condition_label"] = condition_field_label(condition)
        condition_input["category"] = condition_input[condition].astype("string").fillna("Missing")
        condition_input["category_label"] = condition_input[condition].map(
            lambda value, field=condition: condition_category_label(field, value)
        )
        grouped = (
            condition_input.groupby(
                ["accident_year", "condition", "condition_label", "category", "category_label"],
                dropna=False,
            )
            .agg(
                total_collisions=("collision_id", "count"),
                fatal_collisions=("fatal_collision", "sum"),
                serious_collisions=("serious_collision", "sum"),
                slight_collisions=("slight_collision", "sum"),
                ksi_collisions=("ksi_collision", "sum"),
            )
            .reset_index()
        )
        grouped["ksi_rate"] = grouped["ksi_collisions"] / grouped["total_collisions"]
        grouped["serious_harm_score"] = grouped["fatal_collisions"] * 10 + grouped["serious_collisions"] * 3
        grouped["harm_score_per_collision"] = grouped["serious_harm_score"] / grouped["total_collisions"]
        totals = grouped.groupby(["accident_year", "condition"])["total_collisions"].transform("sum")
        grouped["share_of_condition_year"] = grouped["total_collisions"] / totals
        summaries.append(grouped)

    drivers = pd.concat(summaries, ignore_index=True)
    drivers = drivers.sort_values(["condition_label", "accident_year", "category_label"])
    _validate_severity_drivers(drivers)
    return drivers


def _validate_severity_drivers(drivers: pd.DataFrame) -> None:
    require_non_empty(drivers, "severity drivers summary")
    required = {
        "accident_year",
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
        "serious_harm_score",
        "harm_score_per_collision",
        "share_of_condition_year",
    }
    missing = required - set(drivers.columns)
    if missing:
        raise DataValidationError(f"Severity drivers missing columns: {sorted(missing)}")
    if not drivers["total_collisions"].gt(0).all():
        raise DataValidationError("Severity drivers total collisions must be positive")
    if not (
        drivers["fatal_collisions"] + drivers["serious_collisions"] == drivers["ksi_collisions"]
    ).all():
        raise DataValidationError("Severity drivers KSI totals must equal Fatal plus Serious")
    if not (
        drivers["fatal_collisions"]
        + drivers["serious_collisions"]
        + drivers["slight_collisions"]
        == drivers["total_collisions"]
    ).all():
        raise DataValidationError("Severity drivers severity totals must equal total collisions")
    expected_ksi_rate = drivers["ksi_collisions"] / drivers["total_collisions"]
    expected_harm = drivers["fatal_collisions"] * 10 + drivers["serious_collisions"] * 3
    expected_harm_per_collision = expected_harm / drivers["total_collisions"]
    if not expected_ksi_rate.round(12).eq(drivers["ksi_rate"].round(12)).all():
        raise DataValidationError("Severity drivers KSI rates do not match counts")
    if not expected_harm.eq(drivers["serious_harm_score"]).all():
        raise DataValidationError("Severity drivers harm scores do not match severity counts")
    if not expected_harm_per_collision.round(12).eq(drivers["harm_score_per_collision"].round(12)).all():
        raise DataValidationError("Severity drivers harm-per-collision values do not match counts")
    for column in ["ksi_rate", "share_of_condition_year"]:
        if not drivers[column].between(0, 1).all():
            raise DataValidationError(f"Severity drivers {column} must be between 0 and 1")
    share_sums = drivers.groupby(["accident_year", "condition"])["share_of_condition_year"].sum()
    if not share_sums.round(12).eq(1).all():
        raise DataValidationError("Severity drivers condition-year shares must sum to 1")


def _validate_vulnerable_summary(summary: pd.DataFrame) -> None:
    require_non_empty(summary, "vulnerable user summary")
    required = {"accident_year", "road_user_group", "records", "ksi_count", "ksi_rate"}
    missing = required - set(summary.columns)
    if missing:
        raise DataValidationError(f"Vulnerable user summary missing columns: {sorted(missing)}")
    known_groups = set(CASUALTY_TYPE_GROUPS.values())
    unknown_groups = sorted(set(summary["road_user_group"].astype(str)) - known_groups)
    if unknown_groups:
        raise DataValidationError(f"Vulnerable user summary has unknown groups: {unknown_groups}")
    if not summary["records"].gt(0).all():
        raise DataValidationError("Vulnerable user summary records must be positive")
    if not summary["ksi_count"].between(0, summary["records"]).all():
        raise DataValidationError("Vulnerable user summary KSI counts must be between 0 and records")
    expected_rate = summary["ksi_count"] / summary["records"]
    if not expected_rate.round(12).eq(summary["ksi_rate"].round(12)).all():
        raise DataValidationError("Vulnerable user summary KSI rates do not match counts")
    if not summary["ksi_rate"].between(0, 1).all():
        raise DataValidationError("Vulnerable user summary KSI rates must be between 0 and 1")


def _validate_collision_points(points: pd.DataFrame) -> None:
    require_non_empty(points, "collision point sample")
    required = {
        "collision_id",
        "accident_year",
        "longitude",
        "latitude",
        "predicted_ksi_risk",
        "is_ksi",
    }
    missing = required - set(points.columns)
    if missing:
        raise DataValidationError(f"Collision point sample missing columns: {sorted(missing)}")
    if points["collision_id"].astype(str).duplicated().any():
        raise DataValidationError("Collision point sample collision_id values must be unique")
    if points[["longitude", "latitude"]].isna().any().any():
        raise DataValidationError("Collision point sample coordinates must be non-missing")
    if not points["longitude"].between(-0.6, 0.4).all() or not points["latitude"].between(51.2, 51.8).all():
        raise DataValidationError("Collision point sample coordinates must be inside broad London bounds")
    if not points["predicted_ksi_risk"].between(0, 1).all():
        raise DataValidationError("Collision point sample predicted_ksi_risk must be between 0 and 1")
    if not set(pd.to_numeric(points["is_ksi"], errors="coerce").dropna().astype(int)).issubset({0, 1}):
        raise DataValidationError("Collision point sample is_ksi must be binary")


def _validate_safety_map_yearly(summary: pd.DataFrame) -> None:
    require_non_empty(summary, "safety map yearly summary")
    required = {
        "accident_year",
        "total_collisions",
        "ksi_collisions",
        "mean_predicted_ksi_risk",
        "vulnerable_records",
        "ksi_rate",
        "vulnerable_user_share",
    }
    missing = required - set(summary.columns)
    if missing:
        raise DataValidationError(f"Safety map yearly summary missing columns: {sorted(missing)}")
    if not summary["total_collisions"].gt(0).all():
        raise DataValidationError("Safety map yearly total collisions must be positive")
    if not summary["ksi_collisions"].between(0, summary["total_collisions"]).all():
        raise DataValidationError("Safety map yearly KSI collisions must be between 0 and total collisions")
    if not summary["vulnerable_records"].between(0, summary["total_collisions"]).all():
        raise DataValidationError("Safety map yearly vulnerable records must be between 0 and total collisions")
    expected_ksi_rate = summary["ksi_collisions"] / summary["total_collisions"]
    expected_vulnerable_share = summary["vulnerable_records"] / summary["total_collisions"]
    if not expected_ksi_rate.round(12).eq(summary["ksi_rate"].round(12)).all():
        raise DataValidationError("Safety map yearly KSI rates do not match counts")
    if not expected_vulnerable_share.round(12).eq(summary["vulnerable_user_share"].round(12)).all():
        raise DataValidationError("Safety map yearly vulnerable-user shares do not match counts")
    for column in ["mean_predicted_ksi_risk", "ksi_rate", "vulnerable_user_share"]:
        if not summary[column].between(0, 1).all():
            raise DataValidationError(f"Safety map yearly {column} must be between 0 and 1")


def build_app_artifacts(config: PipelineConfig, include_monitoring: bool = False) -> dict[str, Path]:
    app_dir = app_data_dir(config.project_root)
    proc = processed_dir(config.project_root)
    reg = registry_dir(config.project_root)

    collisions = read_required_parquet(proc / "collisions_london.parquet")
    model_df = read_required_parquet(proc / "model_collision_severity.parquet")
    casualties = read_required_parquet(proc / "casualties_london.parquet")
    metrics = read_required_json(reg / "model_metrics.json")
    collisions = collisions[
        collisions["accident_year"].between(config.year_range.start_year, config.year_range.end_year)
    ].copy()
    require_non_empty(collisions, "app collision data")
    model_df = model_df[
        model_df["accident_year"].between(config.year_range.start_year, config.year_range.end_year)
    ].copy()
    require_non_empty(model_df, "app model data")
    model_df["predicted_ksi_risk"] = predict_probabilities(config.project_root, model_df)
    collisions = add_simple_grid(collisions)
    enriched = collisions.merge(
        model_df[
            [
                "collision_id",
                "is_ksi",
                "predicted_ksi_risk",
                "has_pedestrian",
                "has_cyclist",
                "has_motorcyclist",
                "has_child",
                "has_elderly",
            ]
        ],
        on="collision_id",
        how="inner",
    )
    require_non_empty(enriched, "app collision enriched data")

    point_columns = [
        column
        for column in [
            "collision_id",
            "accident_year",
            "longitude",
            "latitude",
            "accident_severity_label",
            "predicted_ksi_risk",
            "is_ksi",
            "has_pedestrian",
            "has_cyclist",
            "has_motorcyclist",
        ]
        if column in enriched.columns
    ]
    points = enriched.dropna(subset=["longitude", "latitude"])[point_columns].copy()
    if len(points) > 25000:
        points = points.sample(25000, random_state=42)
    _validate_collision_points(points)

    safety_map_yearly = (
        enriched.groupby("accident_year", dropna=False)
        .agg(
            total_collisions=("collision_id", "count"),
            ksi_collisions=("is_ksi", "sum"),
            mean_predicted_ksi_risk=("predicted_ksi_risk", "mean"),
            vulnerable_records=(
                "has_pedestrian",
                lambda s: int(
                    (
                        enriched.loc[s.index, ["has_pedestrian", "has_cyclist", "has_motorcyclist"]]
                        .fillna(False)
                        .any(axis=1)
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )
    safety_map_yearly["ksi_rate"] = (
        safety_map_yearly["ksi_collisions"] / safety_map_yearly["total_collisions"]
    )
    safety_map_yearly["vulnerable_user_share"] = (
        safety_map_yearly["vulnerable_records"] / safety_map_yearly["total_collisions"]
    )
    _validate_safety_map_yearly(safety_map_yearly)

    grid = (
        enriched.dropna(subset=["grid_id"])
        .groupby(["grid_id", "grid_lon", "grid_lat"], dropna=False)
        .agg(
            record_count=("collision_id", "count"),
            recent_ksi_count=("is_ksi", "sum"),
            predicted_ksi_risk=("predicted_ksi_risk", "mean"),
            vulnerable_records=(
                "has_pedestrian",
                lambda s: int(
                    (
                        enriched.loc[s.index, ["has_pedestrian", "has_cyclist", "has_motorcyclist"]]
                        .fillna(False)
                        .any(axis=1)
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )
    grid["vulnerable_user_share"] = grid["vulnerable_records"] / grid["record_count"]
    require_non_empty(grid, "grid risk summary")
    priority = compute_priority_scores(grid)

    vulnerable_summary = pd.DataFrame()
    if "road_user_group" in casualties.columns:
        casualty_base = casualties.drop(columns=["accident_year"], errors="ignore")
        vulnerable_summary = casualty_base.merge(
            collisions[["collision_id", "accident_year", "accident_severity_label"]], on="collision_id", how="inner"
        )
        vulnerable_summary["is_ksi"] = vulnerable_summary["accident_severity_label"].isin(["Fatal", "Serious"])
        vulnerable_summary = (
            vulnerable_summary.groupby(["accident_year", "road_user_group"], dropna=False)
            .agg(records=("collision_id", "count"), ksi_count=("is_ksi", "sum"))
            .reset_index()
        )
        vulnerable_summary["ksi_rate"] = vulnerable_summary["ksi_count"] / vulnerable_summary["records"]
    _validate_vulnerable_summary(vulnerable_summary)
    borough_summary = _build_borough_summary(collisions)
    severity_drivers = _build_severity_drivers(collisions)
    boundary_path = ensure_borough_boundaries(config.project_root)

    paths = {
        "points": app_dir / "collision_points_sample.parquet",
        "safety_map_yearly": app_dir / "safety_map_yearly.parquet",
        "grid": app_dir / "hex_or_grid_risk_summary.parquet",
        "priority": app_dir / "priority_locations.parquet",
        "vulnerable": app_dir / "vulnerable_user_summary.parquet",
        "borough": app_dir / "borough_severity_yearly.parquet",
        "severity_drivers": app_dir / "severity_drivers_yearly.parquet",
        "borough_boundaries": boundary_path,
        "metrics": app_dir / "model_metrics.json",
        "monitoring": app_dir / "monitoring_summary.json",
    }
    points.to_parquet(paths["points"], index=False)
    safety_map_yearly.to_parquet(paths["safety_map_yearly"], index=False)
    grid.to_parquet(paths["grid"], index=False)
    priority.to_parquet(paths["priority"], index=False)
    vulnerable_summary.to_parquet(paths["vulnerable"], index=False)
    borough_summary.to_parquet(paths["borough"], index=False)
    severity_drivers.to_parquet(paths["severity_drivers"], index=False)
    _copy_json(reg / "model_metrics.json", paths["metrics"])
    if include_monitoring:
        build_monitoring_summary(config.project_root)
    elif not paths["monitoring"].exists():
        summary = {
            "model_name": metrics.get("model_name", "severity_model"),
            "split": metrics.get("split"),
            "test_metrics": metrics.get("test"),
            "known_limitations": [
                "Monitoring summary was generated from static training artifacts.",
                "Run with --include-monitoring for full yearly summaries.",
            ],
        }
        paths["monitoring"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return paths
