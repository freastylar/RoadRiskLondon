from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from roadrisk.config import Mode, PipelineConfig, YearRange
from roadrisk.features.road_risk import build_road_risk_table

START, END = 2020, 2024
# A road running east-west near these crashes.
ROAD = LineString([(-0.12, 51.50), (-0.08, 51.50)])
CRASH_LON, CRASH_LAT = -0.10, 51.5001  # ~10 m from the road line


def _write_roads(root: Path) -> None:
    # Realistic OSM-style columns: a street name plus a road number.
    gdf = gpd.GeoDataFrame(
        {"name": ["Western Avenue"], "ref": ["A40"], "highway": ["primary"]},
        geometry=[ROAD],
        crs="EPSG:4326",
    )
    out = root / "data/raw/roads/london_roads.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out)


def _write_collisions(root: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        rows.append(
            {
                "collision_id": f"C{i + 1}",
                "accident_year": int(rng.integers(START, END + 1)),
                "hour": int(rng.integers(0, 24)),
                "day_of_week": int(rng.integers(1, 8)),
                "weather_conditions": int(rng.integers(1, 9)),
                "longitude": CRASH_LON + rng.normal(0, 0.0002),
                "latitude": CRASH_LAT + rng.normal(0, 0.0002),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "data/processed/collisions_london.parquet", index=False)
    veh = []
    for r in rows:
        veh.append({"collision_id": r["collision_id"], "vehicle_type": 9})   # car
        if int(r["collision_id"][1:]) % 2 == 0:
            veh.append({"collision_id": r["collision_id"], "vehicle_type": 1})  # bike
    pd.DataFrame(veh).to_parquet(root / "data/processed/vehicles_london.parquet", index=False)


def _write_traffic(root: Path) -> None:
    grid_id = f"{round(CRASH_LON / 0.01) * 0.01}_{round(CRASH_LAT / 0.01) * 0.01}"
    pd.DataFrame(
        [{"grid_id": grid_id, "year": y, "local_authority_code": "E09000001",
          "all_motor_vehicles": 800.0} for y in range(START, END + 1)]
    ).to_parquet(root / "data/raw/traffic/london_aadf_by_point_year.parquet", index=False)
    pd.DataFrame(
        [{"local_authority_code": "E09000001", "year": y, "all_motor_vehicles": 1_000_000.0}
         for y in range(START, END + 1)]
    ).to_parquet(root / "data/raw/traffic/london_la_traffic.parquet", index=False)
    pd.DataFrame(
        [{"count_point_id": 1, "year": START, "local_authority_code": "E09000001",
          "road_type": "Major", "latitude": CRASH_LAT, "longitude": CRASH_LON,
          "all_motor_vehicles": 800.0, "all_HGVs": 50.0, "pedal_cycles": 10.0}]
    ).to_parquet(root / "data/raw/traffic/london_raw_counts.parquet", index=False)
    prof = []
    for road_type in ["Major", "Minor"]:
        for hour in range(7, 19):
            prof.append({"road_type": road_type, "day_type": "Weekday", "hour": hour,
                         "motor_mean": 100.0 + hour, "bike_mean": 5.0})
    pd.DataFrame(prof).to_parquet(root / "data/raw/traffic/london_hourly_profiles.parquet", index=False)


@pytest.fixture()
def road_project(tmp_path: Path) -> PipelineConfig:
    for d in ["data/processed", "data/app", "data/raw/traffic", "data/raw/roads", "models/reports"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    _write_roads(tmp_path)
    _write_collisions(tmp_path)
    _write_traffic(tmp_path)
    return PipelineConfig(
        mode=Mode.MODELING, year_range=YearRange(START, END), project_root=tmp_path
    ).validate()


def test_build_road_risk_snaps_to_named_road(road_project):
    df = build_road_risk_table(road_project)
    assert not df.empty
    # Crashes were all near the road line; label prefers the street NAME over the number.
    assert (df["road_label"] == "Western Avenue").all()
    assert df["risk_per_trip"].between(0, 1).all()
    assert (df["estimated_trips"] > 0).all()
    assert df["road_lon"].notna().all() and df["road_lat"].notna().all()
    assert df["confidence"].isin(["Low", "Medium", "High"]).all()


def test_road_risk_artifacts_written(road_project):
    build_road_risk_table(road_project)
    assert (road_project.project_root / "data/app/road_risk_table.parquet").exists()
    assert (road_project.project_root / "data/app/road_risk_meta.json").exists()


def test_modes_limited_to_known(road_project):
    df = build_road_risk_table(road_project)
    assert set(df["mode"].unique()).issubset(
        {"Bicycle", "Car", "Motorcycle", "Van / goods", "Bus / coach"}
    )
