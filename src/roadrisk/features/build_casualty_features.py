from __future__ import annotations

import pandas as pd


def build_casualty_flags(casualties: pd.DataFrame) -> pd.DataFrame:
    if "collision_id" not in casualties.columns:
        raise ValueError("casualties missing collision_id")
    grouped = casualties.groupby("collision_id", dropna=False)
    features = grouped.size().rename("casualty_record_count").to_frame()
    if "road_user_group" in casualties.columns:
        for group in ["pedestrian", "cyclist", "motorcyclist"]:
            features[f"has_{group}"] = grouped["road_user_group"].apply(lambda s, g=group: bool((s == g).any()))
    if "is_child" in casualties.columns:
        features["has_child"] = grouped["is_child"].max().astype(bool)
    if "is_elderly" in casualties.columns:
        features["has_elderly"] = grouped["is_elderly"].max().astype(bool)
    return features.reset_index()
