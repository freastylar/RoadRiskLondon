from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


class DataValidationError(RuntimeError):
    """Raised when a pipeline artifact is missing or invalid."""


def require_file(path: Path) -> Path:
    if not path.exists():
        raise DataValidationError(f"Missing required file: {path}")
    if path.stat().st_size == 0:
        raise DataValidationError(f"Required file is empty: {path}")
    return path


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise DataValidationError(f"{context} missing required columns: {missing}")


def require_non_empty(df: pd.DataFrame, context: str) -> None:
    if df.empty:
        raise DataValidationError(f"{context} is empty")


def read_required_parquet(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_parquet(path)
    require_non_empty(df, str(path))
    return df


def read_required_json(path: Path) -> dict:
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not data:
        raise DataValidationError(f"JSON artifact is empty: {path}")
    return data


def assert_year_range(df: pd.DataFrame, start_year: int, end_year: int, column: str = "accident_year") -> None:
    require_columns(df, [column], "year range check")
    years = set(pd.to_numeric(df[column], errors="coerce").dropna().astype(int))
    outside = [year for year in years if year < start_year or year > end_year]
    if outside:
        raise DataValidationError(f"Found years outside configured range: {outside}")


def validate_coordinates(df: pd.DataFrame, lon: str = "longitude", lat: str = "latitude") -> None:
    if lon not in df.columns or lat not in df.columns:
        return
    coords = df[[lon, lat]].dropna()
    if coords.empty:
        return
    bad = coords[
        (coords[lon] < -8.7)
        | (coords[lon] > 2.1)
        | (coords[lat] < 49.7)
        | (coords[lat] > 60.9)
    ]
    if not bad.empty:
        raise DataValidationError("Found coordinates outside broad Great Britain bounds")
