from __future__ import annotations

import pandas as pd

from roadrisk.data.decode_categories import (
    casualty_group,
    casualty_type_label,
    is_ksi_from_severity,
)


def test_ksi_target_definition():
    target = is_ksi_from_severity(pd.Series([1, 2, 3, "Fatal", "Serious", "Slight"]))
    assert target.tolist() == [1, 1, 0, 1, 1, 0]


def test_historical_casualty_type_mapping():
    groups = casualty_group(pd.Series([109, 108, 104, 106, 110, 113, 9]))
    assert groups.tolist() == [
        "car_occupant",
        "car_occupant",
        "motorcyclist",
        "motorcyclist",
        "bus_or_coach",
        "other",
        "car_occupant",
    ]
    labels = casualty_type_label(pd.Series([109, 104]))
    assert labels.tolist() == [
        "Car including private hire cars, 1979-2004",
        "Motorcycle, 1979-1998",
    ]
