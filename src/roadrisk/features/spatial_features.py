from __future__ import annotations

import pandas as pd


def add_simple_grid(df: pd.DataFrame, size: float = 0.01) -> pd.DataFrame:
    result = df.copy()
    if {"longitude", "latitude"}.issubset(result.columns):
        lon = pd.to_numeric(result["longitude"], errors="coerce")
        lat = pd.to_numeric(result["latitude"], errors="coerce")
        result["grid_lon"] = (lon / size).round().mul(size)
        result["grid_lat"] = (lat / size).round().mul(size)
        result["grid_id"] = result["grid_lon"].round(3).astype(str) + "_" + result["grid_lat"].round(3).astype(str)
    return result
