from __future__ import annotations

from pathlib import Path

import pandas as pd

from roadrisk.config import PipelineConfig
from roadrisk.data.clean_casualties import clean_casualties
from roadrisk.data.clean_collisions import clean_london_collisions
from roadrisk.data.clean_vehicles import clean_vehicles
from roadrisk.data.inspect_schema import table_files_from_manifest
from roadrisk.utils.paths import processed_dir, reports_dir
from roadrisk.utils.validation import DataValidationError, require_columns, require_non_empty


def _read_table(root: Path, table: str) -> pd.DataFrame:
    table_files = table_files_from_manifest(root, table)
    frames = [pd.read_csv(path, low_memory=False) for path in table_files.paths]
    df = pd.concat(frames, ignore_index=True)
    require_non_empty(df, f"raw {table}")
    return df


def _validate_processed_links(
    collisions: pd.DataFrame,
    vehicles: pd.DataFrame,
    casualties: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> None:
    require_columns(collisions, ["collision_id", "accident_year"], "processed collisions")
    require_columns(vehicles, ["collision_id"], "processed vehicles")
    require_columns(casualties, ["collision_id"], "processed casualties")
    if collisions["collision_id"].astype(str).duplicated().any():
        raise DataValidationError("Processed collisions must have unique collision_id values")
    collision_ids = set(collisions["collision_id"].astype(str))
    for name, table in [("vehicles", vehicles), ("casualties", casualties)]:
        linked_ids = set(table["collision_id"].astype(str))
        missing_ids = sorted(linked_ids - collision_ids)
        if missing_ids:
            preview = missing_ids[:5]
            raise DataValidationError(
                f"Processed {name} contain collision IDs not found in collisions: {preview}"
            )
        if "accident_year" in table.columns:
            years = pd.to_numeric(table["accident_year"], errors="coerce")
            if years.isna().any():
                raise DataValidationError(f"Processed {name} accident_year must be numeric")
            if not years.between(start_year, end_year).all():
                raise DataValidationError(f"Processed {name} years must be inside configured range")
    collision_years = pd.to_numeric(collisions["accident_year"], errors="coerce")
    if collision_years.isna().any():
        raise DataValidationError("Processed collisions accident_year must be numeric")
    if not collision_years.between(start_year, end_year).all():
        raise DataValidationError("Processed collision years must be inside configured range")


def build_processed_data(config: PipelineConfig, allow_coordinate_fallback: bool = False) -> dict[str, Path]:
    output_dir = processed_dir(config.project_root)
    collisions_raw = _read_table(config.project_root, "collisions")
    collisions = clean_london_collisions(
        collisions_raw,
        config.year_range.start_year,
        config.year_range.end_year,
        allow_coordinate_fallback=allow_coordinate_fallback,
    )
    collision_ids = set(collisions["collision_id"])

    vehicles = clean_vehicles(
        _read_table(config.project_root, "vehicles"),
        config.year_range.start_year,
        config.year_range.end_year,
    )
    vehicles = vehicles[vehicles["collision_id"].isin(collision_ids)].copy()
    require_non_empty(vehicles, "London vehicles")

    casualties = clean_casualties(
        _read_table(config.project_root, "casualties"),
        config.year_range.start_year,
        config.year_range.end_year,
    )
    casualties = casualties[casualties["collision_id"].isin(collision_ids)].copy()
    require_non_empty(casualties, "London casualties")
    _validate_processed_links(
        collisions,
        vehicles,
        casualties,
        config.year_range.start_year,
        config.year_range.end_year,
    )

    paths = {
        "collisions": output_dir / "collisions_london.parquet",
        "vehicles": output_dir / "vehicles_london.parquet",
        "casualties": output_dir / "casualties_london.parquet",
    }
    collisions.to_parquet(paths["collisions"], index=False)
    vehicles.to_parquet(paths["vehicles"], index=False)
    casualties.to_parquet(paths["casualties"], index=False)

    report = "\n".join(
        [
            "# Processed Data Report",
            "",
            f"- Collision rows: {len(collisions)}",
            f"- Vehicle rows: {len(vehicles)}",
            f"- Casualty rows: {len(casualties)}",
            f"- Years: {config.year_range.start_year}-{config.year_range.end_year}",
            f"- Collision output: `{paths['collisions']}`",
            f"- Vehicle output: `{paths['vehicles']}`",
            f"- Casualty output: `{paths['casualties']}`",
            "",
        ]
    )
    (reports_dir(config.project_root) / "processed_data_report.md").write_text(report, encoding="utf-8")
    return paths
