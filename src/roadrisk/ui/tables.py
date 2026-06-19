from __future__ import annotations

import json

import pandas as pd

from roadrisk.config import project_root_from
from roadrisk.utils.validation import DataValidationError, read_required_json, read_required_parquet

ROOT = project_root_from()
APP_DIR = ROOT / "data" / "app"
REGISTRY_DIR = ROOT / "models" / "registry"


REQUIRED_APP_ARTIFACTS = [
    APP_DIR / "collision_points_sample.parquet",
    APP_DIR / "safety_map_yearly.parquet",
    APP_DIR / "hex_or_grid_risk_summary.parquet",
    APP_DIR / "borough_boundaries.geojson",
    APP_DIR / "historical_severity_yearly.parquet",
    APP_DIR / "historical_road_user_severity_yearly.parquet",
    APP_DIR / "historical_trends_yearly.parquet",
    APP_DIR / "borough_severity_yearly.parquet",
    APP_DIR / "severity_drivers_yearly.parquet",
    APP_DIR / "vulnerable_user_summary.parquet",
    APP_DIR / "priority_locations.parquet",
    APP_DIR / "model_metrics.json",
    APP_DIR / "monitoring_summary.json",
    REGISTRY_DIR / "severity_model.joblib",
    REGISTRY_DIR / "feature_schema.json",
    REGISTRY_DIR / "feature_importance.parquet",
    REGISTRY_DIR / "calibration_data.parquet",
]


def _validate_readable_artifact(path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        read_required_parquet(path)
    elif suffix in {".json", ".geojson"}:
        read_required_json(path)


def assert_app_ready() -> None:
    missing = [path for path in REQUIRED_APP_ARTIFACTS if not path.exists() or path.stat().st_size == 0]
    if missing:
        paths = "\n".join(f"- `{path}`" for path in missing)
        raise DataValidationError(
            "RoadRisk app artifacts are not ready. Run the pipeline scripts first:\n" + paths
        )
    invalid = []
    for path in REQUIRED_APP_ARTIFACTS:
        try:
            _validate_readable_artifact(path)
        except Exception as exc:
            invalid.append(f"- `{path}` ({exc})")
    if invalid:
        paths = "\n".join(invalid)
        raise DataValidationError(
            "RoadRisk app artifacts exist but are not readable or non-empty:\n" + paths
        )


def load_parquet(name: str) -> pd.DataFrame:
    return read_required_parquet(APP_DIR / name)


def load_json(name: str) -> dict:
    return read_required_json(APP_DIR / name)


def load_registry_json(name: str) -> dict:
    path = REGISTRY_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_registry_parquet(name: str) -> pd.DataFrame:
    return read_required_parquet(REGISTRY_DIR / name)
