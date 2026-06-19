from __future__ import annotations

import pandas as pd
import pytest

from roadrisk.data.build_historical_trends import (
    _validate_historical_yearly,
    _validate_road_user_shares,
    build_historical_trends,
)
from roadrisk.data.build_processed_data import build_processed_data
from roadrisk.utils.validation import DataValidationError


def test_historical_trends_non_empty(sample_config):
    build_processed_data(sample_config)
    paths = build_historical_trends(sample_config)
    yearly = pd.read_parquet(paths["severity_yearly"])
    road_user = pd.read_parquet(paths["road_user_severity_yearly"])
    assert yearly["accident_year"].nunique() > 1
    assert yearly["ksi_collisions"].notna().all()
    assert {
        "total_collisions",
        "fatal_collisions",
        "serious_collisions",
        "slight_collisions",
        "ksi_collisions",
    }.issubset(yearly.columns)
    assert (yearly["fatal_collisions"] + yearly["serious_collisions"] == yearly["ksi_collisions"]).all()
    assert (
        yearly["fatal_collisions"]
        + yearly["serious_collisions"]
        + yearly["slight_collisions"]
        == yearly["total_collisions"]
    ).all()
    assert not road_user.empty
    assert {
        "accident_year",
        "road_user_group",
        "severity_label",
        "casualty_count",
        "share_of_group_year",
        "share_of_all_groups_year",
    }.issubset(road_user.columns)
    _validate_historical_yearly(yearly)
    _validate_road_user_shares(road_user)


def test_historical_yearly_validation_rejects_bad_totals():
    yearly = pd.DataFrame(
        {
            "accident_year": [2020],
            "total_collisions": [10],
            "ksi_collisions": [4],
            "fatal_collisions": [1],
            "serious_collisions": [2],
            "slight_collisions": [6],
            "ksi_rate": [0.4],
        }
    )
    with pytest.raises(DataValidationError, match="KSI totals"):
        _validate_historical_yearly(yearly)


def test_historical_road_user_validation_rejects_bad_share_denominator():
    road_user = pd.DataFrame(
        {
            "accident_year": [2020, 2020],
            "road_user_group": ["pedestrian", "pedestrian"],
            "severity_label": ["Serious", "Slight"],
            "casualty_count": [2, 8],
            "share_of_group_year": [0.1, 0.8],
            "share_of_all_groups_year": [0.2, 0.8],
        }
    )
    with pytest.raises(DataValidationError, match="group-year shares"):
        _validate_road_user_shares(road_user)
