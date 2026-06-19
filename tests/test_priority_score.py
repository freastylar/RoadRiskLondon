from __future__ import annotations

import pandas as pd
import pytest

from roadrisk.features.priority_score import compute_priority_scores
from roadrisk.utils.validation import DataValidationError


def test_priority_scores_reject_empty_input():
    df = pd.DataFrame(
        columns=[
            "predicted_ksi_risk",
            "recent_ksi_count",
            "vulnerable_user_share",
            "record_count",
        ]
    )
    with pytest.raises(DataValidationError, match="non-empty"):
        compute_priority_scores(df)


def test_priority_scores_are_bounded():
    df = pd.DataFrame(
        {
            "predicted_ksi_risk": [0.1, 0.8],
            "recent_ksi_count": [1, 3],
            "vulnerable_user_share": [0.2, 0.9],
            "record_count": [5, 20],
        }
    )
    scored = compute_priority_scores(df)
    assert scored["priority_score"].between(0, 1).all()
    assert scored["observed_ksi_rate"].between(0, 1).all()
    assert scored["data_confidence"].between(0.15, 1).all()
    assert {"priority_band", "priority_reason", "audit_focus"}.issubset(scored.columns)
    assert scored["priority_reason"].str.len().gt(0).all()
    assert scored["audit_focus"].str.len().gt(0).all()
    assert scored.iloc[0]["priority_score"] >= scored.iloc[1]["priority_score"]
    assert scored["priority_score"].is_monotonic_decreasing


def test_priority_bands_follow_score_thresholds():
    df = pd.DataFrame(
        {
            "predicted_ksi_risk": [0.1, 0.5, 0.9],
            "recent_ksi_count": [0, 5, 10],
            "vulnerable_user_share": [0.1, 0.5, 0.9],
            "record_count": [10, 20, 30],
        }
    )
    scored = compute_priority_scores(df)
    for _, row in scored.iterrows():
        if row["priority_score"] >= 0.75:
            assert row["priority_band"] == "Very high"
        elif row["priority_score"] >= 0.60:
            assert row["priority_band"] == "High"
        elif row["priority_score"] >= 0.40:
            assert row["priority_band"] == "Medium"
        else:
            assert row["priority_band"] == "Lower"


def test_priority_scores_reject_zero_record_count():
    df = pd.DataFrame(
        {
            "predicted_ksi_risk": [0.1],
            "recent_ksi_count": [0],
            "vulnerable_user_share": [0.2],
            "record_count": [0],
        }
    )
    with pytest.raises(DataValidationError, match="record_count"):
        compute_priority_scores(df)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("predicted_ksi_risk", 1.2, "predicted_ksi_risk"),
        ("predicted_ksi_risk", -0.1, "predicted_ksi_risk"),
        ("vulnerable_user_share", 1.1, "vulnerable_user_share"),
        ("vulnerable_user_share", -0.2, "vulnerable_user_share"),
        ("recent_ksi_count", -1, "recent_ksi_count"),
    ],
)
def test_priority_scores_reject_invalid_component_values(column, value, message):
    df = pd.DataFrame(
        {
            "predicted_ksi_risk": [0.1],
            "recent_ksi_count": [0],
            "vulnerable_user_share": [0.2],
            "record_count": [5],
        }
    )
    df.loc[0, column] = value
    with pytest.raises(DataValidationError, match=message):
        compute_priority_scores(df)


def test_priority_scores_reject_ksi_count_above_record_count():
    df = pd.DataFrame(
        {
            "predicted_ksi_risk": [0.1],
            "recent_ksi_count": [6],
            "vulnerable_user_share": [0.2],
            "record_count": [5],
        }
    )
    with pytest.raises(DataValidationError, match="cannot exceed"):
        compute_priority_scores(df)


def test_priority_scores_reject_nonnumeric_inputs():
    df = pd.DataFrame(
        {
            "predicted_ksi_risk": ["not-a-number"],
            "recent_ksi_count": [0],
            "vulnerable_user_share": [0.2],
            "record_count": [5],
        }
    )
    with pytest.raises(DataValidationError, match="numeric"):
        compute_priority_scores(df)
