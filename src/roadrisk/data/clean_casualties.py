from __future__ import annotations

import pandas as pd

from roadrisk.data.clean_collisions import add_collision_id, normalise_columns
from roadrisk.data.decode_categories import casualty_group
from roadrisk.utils.validation import DataValidationError


def clean_casualties(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    df = normalise_columns(df.copy())
    if "accident_year" in df.columns:
        df["accident_year"] = pd.to_numeric(df["accident_year"], errors="coerce").astype("Int64")
        df = df[df["accident_year"].between(start_year, end_year)].copy()
    df = add_collision_id(df)
    if df.empty:
        raise DataValidationError("No casualties remain after cleaning")
    for column in [
        "vehicle_reference",
        "casualty_reference",
        "casualty_type",
        "age_of_casualty",
        "age_band_of_casualty",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "casualty_type" in df.columns:
        df["road_user_group"] = casualty_group(df["casualty_type"])
    if "age_of_casualty" in df.columns:
        df["is_child"] = df["age_of_casualty"].between(0, 15)
        df["is_elderly"] = df["age_of_casualty"].ge(65)
    return df
