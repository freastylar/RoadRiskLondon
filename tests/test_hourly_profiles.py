from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roadrisk.data.hourly_profiles import build_hourly_profiles
from roadrisk.utils.validation import DataValidationError


def _write_raw(root: Path, hours: range = range(7, 19)) -> None:
    rows = []
    for road_type in ["Major", "Minor"]:
        for hour in hours:
            rows.append(
                {
                    "road_type": road_type,
                    "day_type": "Weekday",
                    "hour": hour,
                    "motor_mean": 100.0 + hour,  # rising daytime shape
                    "bike_mean": 5.0,
                }
            )
    out = root / "data/raw/traffic/london_hourly_profiles.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)


def test_profiles_sum_to_one_per_group(tmp_path: Path):
    _write_raw(tmp_path)
    df = build_hourly_profiles(tmp_path)
    totals = df.groupby(["road_type", "day_type"])["flow_fraction"].sum()
    assert np.allclose(totals.to_numpy(), 1.0, atol=1e-6)
    # 24 hours x 2 road types x 2 day types.
    assert len(df) == 24 * 2 * 2


def test_measured_hours_flagged_and_others_assumed(tmp_path: Path):
    _write_raw(tmp_path)
    df = build_hourly_profiles(tmp_path)
    weekday_major = df[(df.road_type == "Major") & (df.day_type == "Weekday")]
    measured = set(weekday_major[weekday_major.is_measured]["hour"])
    assert measured == set(range(7, 19))  # only daytime hours measured
    # Weekend has no measured hours (raw data was weekday-only).
    weekend = df[df.day_type == "Weekend"]
    assert not weekend["is_measured"].any()


def test_rejects_missing_file(tmp_path: Path):
    (tmp_path / "data/raw/traffic").mkdir(parents=True, exist_ok=True)
    with pytest.raises(DataValidationError):
        build_hourly_profiles(tmp_path)
