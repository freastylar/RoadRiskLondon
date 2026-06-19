from __future__ import annotations

import pandas as pd

from roadrisk.data.decode_categories import decode_severity
from roadrisk.data.filter_london import filter_london_collisions
from roadrisk.utils.validation import DataValidationError, require_columns, validate_coordinates


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in df.columns
    }
    normalised = df.rename(columns=renamed)
    aliases = {
        "collision_index": "accident_index",
        "collision_year": "accident_year",
        "collision_ref_no": "accident_reference",
        "collision_severity": "accident_severity",
    }
    return normalised.rename(columns={old: new for old, new in aliases.items() if old in normalised.columns})


def add_collision_id(df: pd.DataFrame) -> pd.DataFrame:
    if "accident_index" in df.columns:
        df["collision_id"] = df["accident_index"].astype(str)
        return df
    if {"accident_year", "accident_reference"}.issubset(df.columns):
        df["collision_id"] = (
            pd.to_numeric(df["accident_year"], errors="coerce").astype("Int64").astype(str)
            + "_"
            + df["accident_reference"].astype(str)
        )
        return df
    raise DataValidationError("No verified collision linking key found")


def clean_collisions(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    df = normalise_columns(df.copy())
    if "accident_year" not in df.columns and "date" in df.columns:
        df["accident_year"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.year
    require_columns(df, ["accident_year", "accident_severity"], "collisions")
    df["accident_year"] = pd.to_numeric(df["accident_year"], errors="coerce").astype("Int64")
    df = df[df["accident_year"].between(start_year, end_year)].copy()
    if df.empty:
        raise DataValidationError("No collisions remain after year filtering")
    df = add_collision_id(df)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["month"] = df["date"].dt.month
    if "time" in df.columns:
        parsed_time = pd.to_datetime(df["time"], format="%H:%M", errors="coerce")
        df["hour"] = parsed_time.dt.hour
    for column in ["longitude", "latitude", "number_of_vehicles", "number_of_casualties"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["accident_severity_label"] = decode_severity(df["accident_severity"])
    validate_coordinates(df)
    return df


def clean_london_collisions(
    df: pd.DataFrame, start_year: int, end_year: int, allow_coordinate_fallback: bool = False
) -> pd.DataFrame:
    cleaned = clean_collisions(df, start_year, end_year)
    return filter_london_collisions(cleaned, allow_coordinate_fallback=allow_coordinate_fallback)
