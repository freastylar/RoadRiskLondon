from __future__ import annotations

import pandas as pd

from roadrisk.data.clean_collisions import add_collision_id, normalise_columns
from roadrisk.utils.validation import DataValidationError


def clean_vehicles(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    df = normalise_columns(df.copy())
    if "accident_year" in df.columns:
        df["accident_year"] = pd.to_numeric(df["accident_year"], errors="coerce").astype("Int64")
        df = df[df["accident_year"].between(start_year, end_year)].copy()
    df = add_collision_id(df)
    if df.empty:
        raise DataValidationError("No vehicles remain after cleaning")
    for column in ["vehicle_reference", "vehicle_type", "age_of_driver", "age_band_of_driver"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
