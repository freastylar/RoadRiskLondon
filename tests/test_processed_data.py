from __future__ import annotations

import pandas as pd
import pytest

from roadrisk.data.build_processed_data import _validate_processed_links, build_processed_data
from roadrisk.utils.validation import DataValidationError


def test_processed_data_is_non_empty(sample_config):
    paths = build_processed_data(sample_config)
    collisions = pd.read_parquet(paths["collisions"])
    vehicles = pd.read_parquet(paths["vehicles"])
    casualties = pd.read_parquet(paths["casualties"])
    assert not collisions.empty
    assert not vehicles.empty
    assert not casualties.empty
    assert {"collision_id", "accident_year", "accident_severity_label"}.issubset(collisions.columns)
    _validate_processed_links(collisions, vehicles, casualties, 2020, 2024)


def _valid_processed_tables():
    collisions = pd.DataFrame({"collision_id": ["a", "b"], "accident_year": [2020, 2021]})
    vehicles = pd.DataFrame({"collision_id": ["a", "b"], "accident_year": [2020, 2021]})
    casualties = pd.DataFrame({"collision_id": ["a"], "accident_year": [2020]})
    return collisions, vehicles, casualties


def test_processed_link_validation_accepts_linked_tables():
    collisions, vehicles, casualties = _valid_processed_tables()
    _validate_processed_links(collisions, vehicles, casualties, 2020, 2024)


def test_processed_link_validation_rejects_unlinked_vehicle_rows():
    collisions, vehicles, casualties = _valid_processed_tables()
    vehicles.loc[0, "collision_id"] = "missing"
    with pytest.raises(DataValidationError, match="not found"):
        _validate_processed_links(collisions, vehicles, casualties, 2020, 2024)


def test_processed_link_validation_rejects_duplicate_collision_ids():
    collisions, vehicles, casualties = _valid_processed_tables()
    collisions.loc[1, "collision_id"] = "a"
    with pytest.raises(DataValidationError, match="unique"):
        _validate_processed_links(collisions, vehicles, casualties, 2020, 2024)


def test_processed_link_validation_rejects_out_of_range_years():
    collisions, vehicles, casualties = _valid_processed_tables()
    casualties.loc[0, "accident_year"] = 2030
    with pytest.raises(DataValidationError, match="years"):
        _validate_processed_links(collisions, vehicles, casualties, 2020, 2024)
