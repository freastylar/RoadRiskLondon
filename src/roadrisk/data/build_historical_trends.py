from __future__ import annotations

from pathlib import Path

import pandas as pd

from roadrisk.config import PipelineConfig
from roadrisk.data.clean_collisions import add_collision_id, normalise_columns
from roadrisk.data.decode_categories import casualty_group, decode_severity
from roadrisk.data.filter_london import LONDON_BOROUGH_NAMES, LONDON_ONS_DISTRICTS
from roadrisk.data.inspect_schema import table_files_from_manifest
from roadrisk.utils.paths import app_data_dir, reports_dir
from roadrisk.utils.validation import DataValidationError, require_non_empty

HISTORIC_LONDON_DISTRICT_CODES = set(range(1, 34))


def _validate_historical_yearly(yearly: pd.DataFrame) -> None:
    require_non_empty(yearly, "historical severity yearly artifact")
    required = {
        "accident_year",
        "total_collisions",
        "ksi_collisions",
        "fatal_collisions",
        "serious_collisions",
        "slight_collisions",
        "ksi_rate",
    }
    missing = required - set(yearly.columns)
    if missing:
        raise DataValidationError(f"Historical yearly artifact missing columns: {sorted(missing)}")
    if not yearly["total_collisions"].gt(0).all():
        raise DataValidationError("Historical yearly totals must be positive")
    if not (
        yearly["fatal_collisions"] + yearly["serious_collisions"] == yearly["ksi_collisions"]
    ).all():
        raise DataValidationError("Historical yearly KSI totals must equal Fatal plus Serious")
    if not (
        yearly["fatal_collisions"]
        + yearly["serious_collisions"]
        + yearly["slight_collisions"]
        == yearly["total_collisions"]
    ).all():
        raise DataValidationError("Historical yearly severity totals must equal total collisions")
    expected_rate = yearly["ksi_collisions"] / yearly["total_collisions"]
    if not expected_rate.round(12).eq(yearly["ksi_rate"].round(12)).all():
        raise DataValidationError("Historical yearly KSI rates do not match counts")


def _validate_road_user_shares(road_user: pd.DataFrame) -> None:
    require_non_empty(road_user, "historical road-user severity artifact")
    required = {
        "accident_year",
        "road_user_group",
        "severity_label",
        "casualty_count",
        "share_of_group_year",
        "share_of_all_groups_year",
    }
    missing = required - set(road_user.columns)
    if missing:
        raise DataValidationError(f"Historical road-user artifact missing columns: {sorted(missing)}")
    if (road_user["casualty_count"] < 0).any():
        raise DataValidationError("Historical road-user casualty counts must be non-negative")
    for column in ["share_of_group_year", "share_of_all_groups_year"]:
        if not road_user[column].between(0, 1).all():
            raise DataValidationError(f"Historical road-user {column} must be between 0 and 1")
    group_sums = road_user.groupby(["accident_year", "road_user_group"])["share_of_group_year"].sum()
    if not group_sums.round(12).eq(1).all():
        raise DataValidationError("Historical road-user group-year shares must sum to 1")
    year_sums = road_user.groupby("accident_year")["share_of_all_groups_year"].sum()
    if not year_sums.round(12).eq(1).all():
        raise DataValidationError("Historical road-user all-group yearly shares must sum to 1")


def _chunk_london_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    has_filter_column = False
    if "local_authority_ons_district" in df.columns:
        has_filter_column = True
        mask = mask | df["local_authority_ons_district"].astype(str).str.upper().isin(
            LONDON_ONS_DISTRICTS
        )
    if "local_authority_district" in df.columns:
        has_filter_column = True
        district = pd.to_numeric(df["local_authority_district"], errors="coerce")
        mask = mask | district.isin(HISTORIC_LONDON_DISTRICT_CODES)
    for column in ["local_authority_district", "local_authority_highway", "borough_name"]:
        if column in df.columns:
            has_filter_column = True
            values = df[column].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
            mask = mask | values.isin(LONDON_BOROUGH_NAMES)
    if has_filter_column:
        return mask
    raise DataValidationError("No verified London filter column found in collision data")


def _severity_counts_for_file(path: Path, config: PipelineConfig, chunksize: int = 500_000) -> pd.DataFrame:
    counts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk = normalise_columns(chunk)
        required = {"accident_year", "accident_severity"}
        missing = required - set(chunk.columns)
        if missing:
            raise DataValidationError(f"Historical collisions missing required columns: {sorted(missing)}")

        chunk["accident_year"] = pd.to_numeric(chunk["accident_year"], errors="coerce")
        chunk = chunk[
            chunk["accident_year"].between(config.year_range.start_year, config.year_range.end_year)
        ].copy()
        if chunk.empty:
            continue

        chunk = chunk.loc[_chunk_london_mask(chunk)].copy()
        if chunk.empty:
            continue

        chunk["severity_label"] = decode_severity(chunk["accident_severity"])
        chunk = chunk[chunk["severity_label"].isin(["Fatal", "Serious", "Slight"])].copy()
        if chunk.empty:
            continue

        grouped = (
            chunk.groupby(["accident_year", "severity_label"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        counts.append(grouped)

    if not counts:
        raise DataValidationError(f"No London severity rows found in {path}")
    return pd.concat(counts, ignore_index=True)


def _london_collision_ids_for_file(
    path: Path, config: PipelineConfig, chunksize: int = 500_000
) -> set[str]:
    collision_ids: set[str] = set()
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk = normalise_columns(chunk)
        if "accident_year" not in chunk.columns:
            raise DataValidationError("Historical collisions missing required column: accident_year")
        chunk["accident_year"] = pd.to_numeric(chunk["accident_year"], errors="coerce")
        chunk = chunk[
            chunk["accident_year"].between(config.year_range.start_year, config.year_range.end_year)
        ].copy()
        if chunk.empty:
            continue
        chunk = chunk.loc[_chunk_london_mask(chunk)].copy()
        if chunk.empty:
            continue
        chunk = add_collision_id(chunk)
        collision_ids.update(chunk["collision_id"].astype(str).tolist())
    if not collision_ids:
        raise DataValidationError("No London collision IDs found for historical road-user chart")
    return collision_ids


def _road_user_severity_counts_for_file(
    path: Path, config: PipelineConfig, london_collision_ids: set[str], chunksize: int = 500_000
) -> pd.DataFrame:
    counts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk = normalise_columns(chunk)
        required = {"accident_year", "casualty_type", "casualty_severity"}
        missing = required - set(chunk.columns)
        if missing:
            raise DataValidationError(f"Historical casualties missing required columns: {sorted(missing)}")
        chunk["accident_year"] = pd.to_numeric(chunk["accident_year"], errors="coerce")
        chunk = chunk[
            chunk["accident_year"].between(config.year_range.start_year, config.year_range.end_year)
        ].copy()
        if chunk.empty:
            continue
        chunk = add_collision_id(chunk)
        chunk = chunk[chunk["collision_id"].astype(str).isin(london_collision_ids)].copy()
        if chunk.empty:
            continue
        chunk["road_user_group"] = casualty_group(chunk["casualty_type"])
        chunk["severity_label"] = decode_severity(chunk["casualty_severity"])
        chunk = chunk[chunk["severity_label"].isin(["Fatal", "Serious", "Slight"])].copy()
        if chunk.empty:
            continue
        grouped = (
            chunk.groupby(["accident_year", "road_user_group", "severity_label"], dropna=False)
            .size()
            .rename("casualty_count")
            .reset_index()
        )
        counts.append(grouped)

    if not counts:
        raise DataValidationError(f"No London road-user severity rows found in {path}")
    return pd.concat(counts, ignore_index=True)


def build_historical_trends(config: PipelineConfig) -> dict[str, Path]:
    app_dir = app_data_dir(config.project_root)
    table_files = table_files_from_manifest(config.project_root, "collisions")
    raw_counts = pd.concat(
        [_severity_counts_for_file(path, config) for path in table_files.paths], ignore_index=True
    )
    yearly_long = (
        raw_counts.groupby(["accident_year", "severity_label"], dropna=False)["count"]
        .sum()
        .reset_index()
    )
    yearly = (
        yearly_long.pivot(index="accident_year", columns="severity_label", values="count")
        .fillna(0)
        .reset_index()
    )
    yearly.columns.name = None
    yearly = yearly.rename(
        columns={
            "Fatal": "fatal_collisions",
            "Serious": "serious_collisions",
            "Slight": "slight_collisions",
        }
    )
    for column in ["fatal_collisions", "serious_collisions", "slight_collisions"]:
        if column not in yearly.columns:
            yearly[column] = 0
        yearly[column] = yearly[column].astype(int)
    yearly["accident_year"] = yearly["accident_year"].astype(int)
    yearly["ksi_collisions"] = yearly["fatal_collisions"] + yearly["serious_collisions"]
    yearly["total_collisions"] = (
        yearly["fatal_collisions"] + yearly["serious_collisions"] + yearly["slight_collisions"]
    )
    yearly["ksi_rate"] = yearly["ksi_collisions"] / yearly["total_collisions"]
    yearly = yearly[
        [
            "accident_year",
            "total_collisions",
            "ksi_collisions",
            "fatal_collisions",
            "serious_collisions",
            "slight_collisions",
            "ksi_rate",
        ]
    ].sort_values("accident_year")
    _validate_historical_yearly(yearly)

    london_collision_ids: set[str] = set()
    for path in table_files.paths:
        london_collision_ids.update(_london_collision_ids_for_file(path, config))
    casualty_files = table_files_from_manifest(config.project_root, "casualties")
    road_user_counts = pd.concat(
        [
            _road_user_severity_counts_for_file(path, config, london_collision_ids)
            for path in casualty_files.paths
        ],
        ignore_index=True,
    )
    road_user = (
        road_user_counts.groupby(["accident_year", "road_user_group", "severity_label"], dropna=False)[
            "casualty_count"
        ]
        .sum()
        .reset_index()
    )
    group_totals = road_user.groupby(["accident_year", "road_user_group"])["casualty_count"].transform(
        "sum"
    )
    year_totals = road_user.groupby("accident_year")["casualty_count"].transform("sum")
    road_user["share_of_group_year"] = road_user["casualty_count"] / group_totals
    road_user["share_of_all_groups_year"] = road_user["casualty_count"] / year_totals
    road_user["accident_year"] = road_user["accident_year"].astype(int)
    road_user = road_user.sort_values(["road_user_group", "severity_label", "accident_year"])
    _validate_road_user_shares(road_user)

    paths = {
        "severity_yearly": app_dir / "historical_severity_yearly.parquet",
        "road_user_severity_yearly": app_dir / "historical_road_user_severity_yearly.parquet",
        "yearly": app_dir / "historical_trends_yearly.parquet",
    }
    yearly.to_parquet(paths["severity_yearly"], index=False)
    road_user.to_parquet(paths["road_user_severity_yearly"], index=False)
    yearly.to_parquet(paths["yearly"], index=False)

    report = "\n".join(
        [
            "# Historical Severity Trend Report",
            "",
            f"- Years: {int(yearly['accident_year'].min())}-{int(yearly['accident_year'].max())}",
            f"- Rows: {len(yearly)}",
            f"- Total collisions: {int(yearly['total_collisions'].sum())}",
            f"- KSI collisions: {int(yearly['ksi_collisions'].sum())}",
            f"- Road-user severity rows: {len(road_user)}",
            "- Source: raw DfT collision file(s), London rows only.",
            "- London filter: ONS London borough codes where available; historical numeric district codes 1-33 for earlier records.",
            "",
        ]
    )
    (reports_dir(config.project_root) / "historical_trends_report.md").write_text(report, encoding="utf-8")
    return paths
