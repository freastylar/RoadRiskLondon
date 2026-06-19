from __future__ import annotations

import pytest

from roadrisk.config import Mode, PipelineConfig, YearRange


def test_year_range_validation(sample_project):
    assert YearRange(2020, 2024).validate().years == [2020, 2021, 2022, 2023, 2024]
    with pytest.raises(ValueError):
        YearRange(2025, 2024).validate()
    with pytest.raises(ValueError):
        YearRange(1978, 2024).validate()
    with pytest.raises(ValueError):
        YearRange(2020, 2025).validate()
    with pytest.raises(ValueError):
        PipelineConfig(Mode.MVP, YearRange(2019, 2024), sample_project).validate()
