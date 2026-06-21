"""Traffic exposure features for the trip-risk estimate.

Two raw inputs, both filtered to London:

- ``data/raw/traffic/london_aadf_by_point_year.parquet`` - DfT AADF counts
  aggregated per count point per year, with a ``grid_id`` already attached.
  This is the primary, cell-level exposure source.
- ``data/raw/traffic/london_la_traffic.parquet`` - DfT local-authority traffic
  (vehicle-miles per borough per year). Used to fill grid cells that have no
  count point so every cell gets an exposure value.

A grid cell's traffic is the mean count-point flow in that cell for the year
when a count point exists; otherwise it falls back to a borough-level proxy and
is flagged with ``traffic_is_imputed = True``. Exposure is never silently faked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from roadrisk.utils.paths import raw_dir
from roadrisk.utils.validation import (
    DataValidationError,
    read_required_parquet,
    require_columns,
)

AADF_FILE = "traffic/london_aadf_by_point_year.parquet"
LA_FILE = "traffic/london_la_traffic.parquet"


def _load_point_traffic(root: Path) -> pd.DataFrame:
    df = read_required_parquet(raw_dir(root) / AADF_FILE)
    require_columns(
        df,
        ["grid_id", "year", "local_authority_code", "all_motor_vehicles"],
        "AADF point traffic",
    )
    df = df.copy()
    df["all_motor_vehicles"] = pd.to_numeric(df["all_motor_vehicles"], errors="coerce")
    return df.dropna(subset=["grid_id", "all_motor_vehicles"])


def _load_borough_traffic(root: Path) -> pd.DataFrame:
    df = read_required_parquet(raw_dir(root) / LA_FILE)
    require_columns(df, ["local_authority_code", "year", "all_motor_vehicles"], "borough traffic")
    df = df.copy()
    df["all_motor_vehicles"] = pd.to_numeric(df["all_motor_vehicles"], errors="coerce")
    return df.dropna(subset=["all_motor_vehicles"])


def build_cell_traffic(root: Path, start_year: int, end_year: int) -> pd.DataFrame:
    """Per (grid_id, year) traffic exposure with an imputation flag.

    Returns columns: grid_id, year, traffic_flow, traffic_is_imputed; the
    per-borough fallback used to fill missing cells is stashed in
    ``df.attrs["borough_fallback"]``.
    """
    points = _load_point_traffic(root)
    points = points[points["year"].between(start_year, end_year)]
    if points.empty:
        raise DataValidationError("No AADF point traffic in the requested year range")

    # One row per (grid_id, year). A cell may contain count points from more than
    # one borough, so aggregate over boroughs here to avoid fan-out on merge.
    cell = (
        points.groupby(["grid_id", "year"], as_index=False)
        .agg(traffic_flow=("all_motor_vehicles", "mean"))
    )
    cell["traffic_is_imputed"] = False

    # Borough-level fallback: a per-borough-year scale used for cells with no
    # count point. We rescale borough vehicle-miles to the same magnitude as the
    # observed per-cell flows so the offset stays on one scale.
    borough = _load_borough_traffic(root)
    borough = borough[borough["year"].between(start_year, end_year)]
    borough_proxy = borough.groupby(["local_authority_code", "year"], as_index=False).agg(
        borough_vehicle_miles=("all_motor_vehicles", "sum")
    )
    observed_cell_median = float(cell["traffic_flow"].median())
    borough_median = float(borough_proxy["borough_vehicle_miles"].median())
    scale = observed_cell_median / borough_median if borough_median > 0 else 1.0
    borough_proxy["borough_flow"] = borough_proxy["borough_vehicle_miles"] * scale

    if (cell["traffic_flow"] < 0).any():
        raise DataValidationError("Traffic flow must be non-negative")
    cell = cell.copy()
    cell.attrs["borough_fallback"] = borough_proxy[
        ["local_authority_code", "year", "borough_flow"]
    ].copy()
    return cell


def fill_panel_traffic(panel: pd.DataFrame, cell_traffic: pd.DataFrame) -> pd.DataFrame:
    """Attach traffic_flow + traffic_is_imputed to every panel (grid_id, borough, year) row.

    Cells with an observed count point use it; the rest fall back to the
    borough-year proxy and are flagged imputed.
    """
    fallback = cell_traffic.attrs.get("borough_fallback")
    if fallback is None:
        raise DataValidationError("cell_traffic is missing its borough_fallback attribute")

    merged = panel.merge(
        cell_traffic[["grid_id", "year", "traffic_flow", "traffic_is_imputed"]],
        on=["grid_id", "year"],
        how="left",
    )
    merged = merged.merge(fallback, on=["local_authority_code", "year"], how="left")

    observed = merged["traffic_flow"].notna()
    merged["traffic_is_imputed"] = (~observed).astype(bool)
    merged["traffic_flow"] = merged["traffic_flow"].where(observed, merged["borough_flow"])

    # Any cell-year still missing both (no borough proxy either) gets the global
    # median so log-exposure stays finite; flag it imputed.
    global_median = float(cell_traffic["traffic_flow"].median())
    still_missing = merged["traffic_flow"].isna()
    merged.loc[still_missing, "traffic_flow"] = global_median
    merged.loc[still_missing, "traffic_is_imputed"] = True

    merged["traffic_flow"] = merged["traffic_flow"].clip(lower=1.0)
    merged["log_traffic_exposure"] = np.log(merged["traffic_flow"])
    return merged.drop(columns=["borough_flow"])
