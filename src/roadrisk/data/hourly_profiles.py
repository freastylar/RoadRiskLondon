"""Hourly traffic exposure profiles for the trip-risk estimate.

DfT raw count sessions in London only cover daytime weekday hours (about 07:00-
18:00). Where we have those real measurements we use the observed hour-of-day
shape; for night hours and weekends, which DfT does not count, we fall back to a
standard UK profile shape. The result is a normalised 24-hour profile per road
type and day type, with a column flagging which hours are measured vs assumed.

The raw aggregation lives in
``data/raw/traffic/london_hourly_profiles.parquet`` (built once from the raw
counts file); this module turns it into the normalised profiles the trip-risk
builder consumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from roadrisk.utils.paths import raw_dir
from roadrisk.utils.validation import DataValidationError, read_required_parquet, require_columns

PROFILE_FILE = "traffic/london_hourly_profiles.parquet"

# Standard fallback shape used only for hours DfT does not count (relative flow).
FALLBACK_WEEKDAY = np.array(
    [0.006, 0.004, 0.003, 0.003, 0.005, 0.014, 0.035, 0.062, 0.073, 0.060,
     0.050, 0.050, 0.052, 0.052, 0.054, 0.060, 0.072, 0.080, 0.064, 0.044,
     0.032, 0.024, 0.018, 0.012]
)
FALLBACK_WEEKEND = np.array(
    [0.012, 0.008, 0.006, 0.005, 0.005, 0.008, 0.014, 0.022, 0.034, 0.048,
     0.060, 0.068, 0.072, 0.072, 0.070, 0.066, 0.062, 0.056, 0.048, 0.040,
     0.034, 0.028, 0.022, 0.016]
)


def _measured_shape(profile: pd.DataFrame, road_type: str, day_type: str) -> dict[int, float]:
    sub = profile[(profile["road_type"] == road_type) & (profile["day_type"] == day_type)]
    return {int(r.hour): float(r.motor_mean) for r in sub.itertuples()}


def build_hourly_profiles(root: Path) -> pd.DataFrame:
    """Normalised 24h profiles per (road_type, day_type) with a measured/assumed flag."""
    raw = read_required_parquet(raw_dir(root) / PROFILE_FILE)
    require_columns(raw, ["road_type", "day_type", "hour", "motor_mean"], "hourly profiles")

    road_types = sorted(raw["road_type"].dropna().unique().tolist()) or ["Major", "Minor"]
    rows = []
    for road_type in road_types:
        for day_type, fallback in [("Weekday", FALLBACK_WEEKDAY), ("Weekend", FALLBACK_WEEKEND)]:
            measured = _measured_shape(raw, road_type, day_type)
            shape = fallback.astype(float).copy()
            is_measured = np.zeros(24, dtype=bool)
            if measured:
                # Scale the fallback so it is continuous with measured hours, then
                # overwrite measured hours with the observed relative flow.
                measured_total = sum(measured.values())
                fallback_at_measured = fallback[list(measured.keys())].sum()
                scale = (fallback_at_measured / measured_total) if measured_total > 0 else 1.0
                for hour, value in measured.items():
                    shape[hour] = value * scale
                    is_measured[hour] = True
            shape = shape / shape.sum()
            for hour in range(24):
                rows.append(
                    {
                        "road_type": road_type,
                        "day_type": day_type,
                        "hour": hour,
                        "flow_fraction": float(shape[hour]),
                        "is_measured": bool(is_measured[hour]),
                    }
                )

    out = pd.DataFrame(rows)
    _validate(out)
    return out


def _validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataValidationError("Hourly profiles are empty")
    totals = df.groupby(["road_type", "day_type"])["flow_fraction"].sum()
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-6):
        raise DataValidationError("Each hourly profile must sum to 1.0")
    if (df["flow_fraction"] < 0).any():
        raise DataValidationError("flow_fraction must be non-negative")
