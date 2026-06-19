from __future__ import annotations

from pathlib import Path

import pandas as pd

from roadrisk.data.decode_categories import (
    CONDITION_FIELD_LABELS,
    condition_category_label,
    condition_field_label,
)
from roadrisk.data.filter_london import LONDON_BOROUGH_BY_ONS
from roadrisk.utils.paths import registry_dir
from roadrisk.utils.validation import read_required_parquet

FEATURE_LABELS = {
    "accident_year": "Collision year",
    "month": "Month",
    "hour": "Hour of day",
    "number_of_vehicles": "Number of vehicles",
    "number_of_casualties": "Number of casualties",
    "speed_limit": "Speed limit",
    "casualty_record_count": "Casualty records",
    "grid_lon": "Grid longitude",
    "grid_lat": "Grid latitude",
    "weekend": "Weekend",
    "has_pedestrian": "Pedestrian involved",
    "has_cyclist": "Cyclist involved",
    "has_motorcyclist": "Motorcyclist involved",
    "has_child": "Child casualty involved",
    "has_elderly": "Older casualty involved",
}

DAY_OF_WEEK_LABELS = {
    "1": "Sunday",
    "2": "Monday",
    "3": "Tuesday",
    "4": "Wednesday",
    "5": "Thursday",
    "6": "Friday",
    "7": "Saturday",
}


def load_feature_importance(root: Path) -> pd.DataFrame:
    return read_required_parquet(registry_dir(root) / "feature_importance.parquet")


def feature_display_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for field in CONDITION_FIELD_LABELS:
        prefix = f"{field}_"
        if feature.startswith(prefix):
            value = feature.removeprefix(prefix)
            return f"{condition_field_label(field)}: {condition_category_label(field, value)}"
    if feature.startswith("day_of_week_"):
        value = feature.removeprefix("day_of_week_")
        return f"Day of week: {DAY_OF_WEEK_LABELS.get(value, value)}"
    for prefix in ["local_authority_ons_district_", "local_authority_highway_"]:
        if feature.startswith(prefix):
            code = feature.removeprefix(prefix)
            return f"Borough: {LONDON_BOROUGH_BY_ONS.get(code, code)}"
    if feature.startswith("local_authority_district_"):
        code = feature.removeprefix("local_authority_district_")
        return f"Local authority district code {code}"
    return feature.replace("_", " ").title()


def add_feature_display_labels(df: pd.DataFrame) -> pd.DataFrame:
    labelled = df.copy()
    labelled["feature_label"] = labelled["feature"].map(feature_display_label)
    return labelled
