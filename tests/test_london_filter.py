from __future__ import annotations

import pandas as pd

from roadrisk.data.filter_london import filter_london_collisions


def test_london_filter_uses_ons_codes():
    df = pd.DataFrame(
        {
            "collision_id": ["a", "b"],
            "local_authority_ons_district": ["E09000001", "E06000001"],
        }
    )
    filtered = filter_london_collisions(df)
    assert filtered["collision_id"].tolist() == ["a"]
