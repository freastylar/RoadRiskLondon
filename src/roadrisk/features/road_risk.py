"""Road-based trip-risk: estimated collision risk per trip, by real road segment.

Crashes are snapped to the nearest OpenStreetMap road centreline, then grouped
by road x hour-bucket x day-type x weather x mode. Exposure (estimated trips) is
derived from the nearest traffic count and standard hourly profiles, the same way
as the grid version. The output is a per-road risk table the app looks up and
draws as coloured points on a 2D map.

As with the grid version, this is an ESTIMATE from reported collisions and
assumed exposure, not a measured per-trip probability. Thin slices are smoothed
toward the London average and flagged with a confidence label.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd

from roadrisk.config import PipelineConfig
from roadrisk.data.filter_london import LONDON_BOROUGH_BY_ONS
from roadrisk.data.hourly_profiles import build_hourly_profiles
from roadrisk.data.traffic import build_cell_traffic
from roadrisk.features.spatial_features import add_simple_grid
from roadrisk.utils.paths import app_data_dir, processed_dir, raw_dir, reports_dir
from roadrisk.utils.validation import (
    DataValidationError,
    read_required_parquet,
    require_columns,
)

ROADS_FILE = "roads/london_roads.parquet"
BNG = "EPSG:27700"  # British National Grid (metres) for accurate nearest-road snapping
WGS84 = "EPSG:4326"
MAX_SNAP_METRES = 60.0  # crashes farther than this from any road are dropped
DAYS_PER_YEAR = 365.0
SMOOTHING_TRIPS = 200_000.0

# Road classes drawn in the grey "all roads / no crash data" base layer.
BASE_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link",
}

# DfT vehicle-type codes grouped into the travel modes the user can pick.
MODE_BY_VEHICLE_TYPE = {
    "Bicycle": {1},
    "Motorcycle": {2, 3, 4, 5, 23, 97},
    "Car": {8, 9},
    "Van / goods": {10, 11, 19, 20, 21, 98},
    "Bus / coach": {16, 17, 18},
}
# Rough share of motor-vehicle flow per mode (documented assumptions, not measured).
MODE_EXPOSURE_SHARE = {
    "Car": 0.78, "Van / goods": 0.15, "Motorcycle": 0.012,
    "Bus / coach": 0.013, "Bicycle": 0.02,
}
HOUR_BUCKETS = {
    "Night (00-06)": range(0, 6),
    "Morning (06-10)": range(6, 10),
    "Midday (10-16)": range(10, 16),
    "Evening (16-20)": range(16, 20),
    "Late (20-24)": range(20, 24),
}
WEATHER_LABELS = {
    1: "Fine", 2: "Rain", 3: "Snow", 4: "Fine", 5: "Rain",
    6: "Snow", 7: "Fog/mist", 8: "Other", 9: "Other",
}
# Fallback hourly profile (normalised) used only when measured profiles are missing.
_FALLBACK_WEEKDAY = np.array(
    [0.006, 0.004, 0.003, 0.003, 0.005, 0.014, 0.035, 0.062, 0.073, 0.060,
     0.050, 0.050, 0.052, 0.052, 0.054, 0.060, 0.072, 0.080, 0.064, 0.044,
     0.032, 0.024, 0.018, 0.012]
)
_FALLBACK_WEEKEND = np.array(
    [0.012, 0.008, 0.006, 0.005, 0.005, 0.008, 0.014, 0.022, 0.034, 0.048,
     0.060, 0.068, 0.072, 0.072, 0.070, 0.066, 0.062, 0.056, 0.048, 0.040,
     0.034, 0.028, 0.022, 0.016]
)
_FALLBACK_WEEKDAY = _FALLBACK_WEEKDAY / _FALLBACK_WEEKDAY.sum()
_FALLBACK_WEEKEND = _FALLBACK_WEEKEND / _FALLBACK_WEEKEND.sum()


def _hour_bucket(hour: pd.Series) -> pd.Series:
    out = pd.Series(index=hour.index, dtype="object")
    for label, hours in HOUR_BUCKETS.items():
        out[hour.isin(list(hours))] = label
    return out


def _vehicle_mode(vehicle_type: pd.Series) -> pd.Series:
    out = pd.Series("Other", index=vehicle_type.index, dtype="object")
    for mode, codes in MODE_BY_VEHICLE_TYPE.items():
        out[vehicle_type.isin(list(codes))] = mode
    return out


def _measured_bucket_fraction(
    profiles: pd.DataFrame, road_type: str, hour_bucket: str, day_type: str
) -> float:
    """Fraction of a year's traffic in this slice, from measured hourly profiles.

    Falls back to a standard profile shape when no measured profile is available.
    """
    hours = list(HOUR_BUCKETS[hour_bucket])
    day_weight = (5 / 7) if day_type == "Weekday" else (2 / 7)
    sub = profiles[(profiles["road_type"] == road_type) & (profiles["day_type"] == day_type)]
    if sub.empty:
        fallback = _FALLBACK_WEEKDAY if day_type == "Weekday" else _FALLBACK_WEEKEND
        return float(fallback[hours].sum()) * day_weight
    in_bucket = float(sub[sub["hour"].isin(hours)]["flow_fraction"].sum())
    return in_bucket * day_weight


def _load_roads(config: PipelineConfig) -> gpd.GeoDataFrame:
    path = raw_dir(config.project_root) / ROADS_FILE
    if not path.exists():
        raise DataValidationError(
            f"Road network not found: {path}. Run scripts/14_fetch_london_roads.py first."
        )
    roads = gpd.read_parquet(path)
    roads = roads.copy()

    # Prefer the human street NAME (e.g. "Brixton Road"); fall back to the road
    # number (A23), then the road class. This is what users recognise.
    def _label(row) -> str | None:
        name = row.get("name")
        ref = row.get("ref")
        if isinstance(name, str) and name and name != "nan":
            return name
        if isinstance(ref, str) and ref and ref != "nan":
            return ref
        highway = row.get("highway")
        return f"{str(highway).title()} road" if isinstance(highway, str) and highway else None

    roads["road_label"] = roads.apply(_label, axis=1)
    roads = roads[roads["road_label"].notna()].copy()
    roads = roads.to_crs(BNG)
    roads["road_id"] = np.arange(len(roads))
    return roads[["road_id", "road_label", "highway", "geometry"]]


def _snap_collisions_to_roads(config: PipelineConfig, roads: gpd.GeoDataFrame) -> pd.DataFrame:
    proc = processed_dir(config.project_root)
    collisions = read_required_parquet(proc / "collisions_london.parquet")
    require_columns(
        collisions,
        ["collision_id", "accident_year", "hour", "day_of_week", "weather_conditions",
         "longitude", "latitude"],
        "road-risk collision input",
    )
    start, end = config.year_range.start_year, config.year_range.end_year
    collisions = collisions[collisions["accident_year"].between(start, end)].copy()
    collisions = collisions.dropna(subset=["hour", "day_of_week", "longitude", "latitude"])
    collisions["hour"] = collisions["hour"].astype(int)
    collisions["hour_bucket"] = _hour_bucket(collisions["hour"])
    collisions["day_type"] = np.where(collisions["day_of_week"].isin([1, 7]), "Weekend", "Weekday")
    collisions["weather"] = collisions["weather_conditions"].map(WEATHER_LABELS).fillna("Other")
    # Borough name for a readable per-area split (e.g. "Brixton Road (Lambeth)").
    if "local_authority_ons_district" in collisions.columns:
        codes = collisions["local_authority_ons_district"].astype(str).str.upper()
        collisions["borough"] = codes.map(LONDON_BOROUGH_BY_ONS).fillna("Other London")
    else:
        collisions["borough"] = "London"
    # Grid id is kept so each road can borrow the nearest traffic-count exposure.
    collisions = add_simple_grid(collisions)

    points = gpd.GeoDataFrame(
        collisions,
        geometry=gpd.points_from_xy(collisions["longitude"], collisions["latitude"]),
        crs=WGS84,
    ).to_crs(BNG)
    snapped = gpd.sjoin_nearest(points, roads, how="left", distance_col="snap_m")
    snapped = snapped[snapped["snap_m"] <= MAX_SNAP_METRES].copy()
    if snapped.empty:
        raise DataValidationError("No collisions snapped within the road distance threshold")
    return snapped


def _vehicle_modes(config: PipelineConfig) -> pd.DataFrame:
    vehicles = read_required_parquet(processed_dir(config.project_root) / "vehicles_london.parquet")
    require_columns(vehicles, ["collision_id", "vehicle_type"], "road-risk vehicle input")
    vehicles = vehicles.copy()
    vehicles["mode"] = _vehicle_mode(pd.to_numeric(vehicles["vehicle_type"], errors="coerce").fillna(-1))
    modes = vehicles.loc[vehicles["mode"] != "Other", ["collision_id", "mode"]].drop_duplicates()
    if modes.empty:
        raise DataValidationError("No collisions with a recognised travel mode")
    return modes


def _road_area_paths(df: pd.DataFrame, roads: gpd.GeoDataFrame) -> pd.DataFrame:
    """A JSON list of [lon, lat] line paths per road_area, for the map PathLayer.

    Collects the geometry of the (deduplicated) road segments that had crashes in
    each area, simplified and reprojected to lon/lat.
    """
    geom_wgs = roads.set_index("road_id").geometry.to_crs(WGS84)
    pairs = df[["road_area", "road_id"]].drop_duplicates()
    rows = []
    for road_area, group in pairs.groupby("road_area"):
        paths: list[list[list[float]]] = []
        for road_id in group["road_id"].unique():
            geom = geom_wgs.get(road_id)
            if geom is None or geom.is_empty:
                continue
            line = geom.simplify(0.0001, preserve_topology=False)
            coords = [[round(x, 6), round(y, 6)] for x, y in line.coords]
            if len(coords) >= 2:
                paths.append(coords)
        rows.append({"road_area": road_area, "road_path_json": json.dumps(paths)})
    return pd.DataFrame(rows)


def _write_base_roads(config: PipelineConfig, roads: gpd.GeoDataFrame) -> int:
    """Write all major/named London roads as simplified lon/lat paths (grey base).

    This lets the map show every real road, with crash-based risk drawn on top.
    A grey road means 'no recorded crashes here', not necessarily 'safe'.
    """
    base = roads.copy()
    # Major + named through-roads only (skip residential/unclassified/service) so the
    # base layer stays light enough to render across all of London at once.
    base = base[base["highway"].isin(BASE_HIGHWAYS)]
    geom = base.geometry.to_crs(WGS84).simplify(0.0001, preserve_topology=False)
    rows = []
    for label, line in zip(base["road_label"].to_numpy(), geom.to_numpy(), strict=True):
        if line is None or line.is_empty:
            continue
        coords = [[round(x, 6), round(y, 6)] for x, y in line.coords]
        if len(coords) >= 2:
            # Store as JSON so the nested list survives the parquet round-trip
            # (plain lists come back as numpy arrays that PathLayer cannot read).
            rows.append({"road_label": label, "path_json": json.dumps(coords)})
    base_df = pd.DataFrame(rows)
    base_df.to_parquet(app_data_dir(config.project_root) / "road_base_geometry.parquet", index=False)
    return len(base_df)


def build_road_risk_table(config: PipelineConfig) -> pd.DataFrame:
    start, end = config.year_range.start_year, config.year_range.end_year
    years = end - start + 1
    roads = _load_roads(config)
    _write_base_roads(config, roads)
    snapped = _snap_collisions_to_roads(config, roads)
    modes = _vehicle_modes(config)
    df = snapped.merge(modes, on="collision_id", how="inner")

    # OSM splits each named road into many tiny segments, so aggregate by the road
    # LABEL (e.g. all "A40" pieces together) within a borough-scale area to get
    # meaningful crash counts.
    df["road_area"] = df["road_label"].astype(str) + " (" + df["borough"].astype(str) + ")"
    centroid_bng = roads.set_index("road_id").geometry.centroid
    cx = centroid_bng.x.to_dict()
    cy = centroid_bng.y.to_dict()
    df["seg_x"] = df["road_id"].map(cx)
    df["seg_y"] = df["road_id"].map(cy)

    paths = _road_area_paths(df, roads)

    group_cols = ["road_area", "road_label", "highway", "hour_bucket", "day_type", "weather", "mode"]
    counts = df.groupby(group_cols, as_index=False).agg(
        crash_count=("collision_id", "nunique"),
        grid_id=("grid_id", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        seg_x=("seg_x", "mean"),
        seg_y=("seg_y", "mean"),
    )
    # Convert the averaged BNG centroid back to lon/lat for the map marker.
    loc = gpd.GeoSeries(gpd.points_from_xy(counts["seg_x"], counts["seg_y"]), crs=BNG).to_crs(WGS84)
    counts["road_lon"] = loc.x
    counts["road_lat"] = loc.y
    counts = counts.merge(paths, on="road_area", how="left")

    cell_traffic = build_cell_traffic(config.project_root, start, end)
    yearly_flow = cell_traffic.groupby("grid_id", as_index=False).agg(aadf=("traffic_flow", "mean"))
    london_aadf = float(yearly_flow["aadf"].median())
    counts = counts.merge(yearly_flow, on="grid_id", how="left")
    counts["aadf"] = counts["aadf"].fillna(london_aadf)

    counts["road_type"] = np.where(
        counts["highway"].astype(str).str.contains("motorway|trunk|primary", case=False, na=False),
        "Major", "Minor",
    )

    profiles = build_hourly_profiles(config.project_root)
    bucket_fraction = counts.apply(
        lambda r: _measured_bucket_fraction(profiles, r["road_type"], r["hour_bucket"], r["day_type"]),
        axis=1,
    )
    mode_share = counts["mode"].map(MODE_EXPOSURE_SHARE).fillna(0.02)
    counts["estimated_trips"] = (
        counts["aadf"] * DAYS_PER_YEAR * years * bucket_fraction * mode_share
    ).clip(lower=1.0)

    global_rate = float(df["collision_id"].nunique()) / float(
        (yearly_flow["aadf"].sum() * DAYS_PER_YEAR * years) + 1.0
    )
    counts["risk_per_trip"] = (
        (counts["crash_count"] + SMOOTHING_TRIPS * global_rate)
        / (counts["estimated_trips"] + SMOOTHING_TRIPS)
    )
    counts["one_in_n"] = (1.0 / counts["risk_per_trip"]).round().astype("Int64")
    counts["confidence"] = pd.cut(
        counts["crash_count"], bins=[-1, 4, 19, np.inf], labels=["Low", "Medium", "High"]
    ).astype(str)
    rel = 1.0 / np.sqrt(counts["crash_count"].clip(lower=1))
    counts["one_in_n_low"] = (1.0 / (counts["risk_per_trip"] * (1 + rel))).round().astype("Int64")
    counts["one_in_n_high"] = (1.0 / (counts["risk_per_trip"] * (1 - rel).clip(lower=0.05))).round().astype("Int64")

    out = counts[[
        "road_area", "road_label", "highway", "road_lon", "road_lat", "road_path_json",
        "hour_bucket", "day_type", "weather", "mode",
        "crash_count", "estimated_trips", "risk_per_trip",
        "one_in_n", "one_in_n_low", "one_in_n_high", "confidence",
    ]].copy()
    _validate(out)

    out.to_parquet(processed_dir(config.project_root) / "road_risk_table.parquet", index=False)
    out.to_parquet(app_data_dir(config.project_root) / "road_risk_table.parquet", index=False)
    _write_meta(config, out, start, end, global_rate)
    _write_report(config, out, start, end)
    return out


def _validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataValidationError("Road-risk table is empty")
    if not df["risk_per_trip"].between(0, 1).all():
        raise DataValidationError("risk_per_trip must be a probability in [0, 1]")
    if (df["estimated_trips"] <= 0).any():
        raise DataValidationError("estimated_trips must be positive")
    if df["road_lon"].isna().any() or df["road_lat"].isna().any():
        raise DataValidationError("every road row must have a map location")


def _write_meta(config: PipelineConfig, df: pd.DataFrame, start: int, end: int, global_rate: float) -> None:
    meta = {
        "year_range": [start, end],
        "rows": int(len(df)),
        "roads": int(df["road_area"].nunique()),
        "road_labels": int(df["road_label"].nunique()),
        "modes": sorted(df["mode"].unique().tolist()),
        "hour_buckets": list(HOUR_BUCKETS.keys()),
        "weather_categories": sorted(df["weather"].unique().tolist()),
        "global_risk_per_trip": global_rate,
        "assumptions": [
            "Crashes snapped to the nearest OSM road centreline within 60 m.",
            "Exposure uses nearest traffic count x standard hourly profile x fixed mode share.",
            "Reported personal-injury collisions only; not a guaranteed personal probability.",
            "Thin slices smoothed toward the London average and flagged Low confidence.",
        ],
    }
    (app_data_dir(config.project_root) / "road_risk_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_report(config: PipelineConfig, df: pd.DataFrame, start: int, end: int) -> None:
    report = "\n".join(
        [
            "# Road Risk Table Report",
            "",
            f"- Rows: {len(df)}",
            f"- Distinct road areas: {df['road_area'].nunique()}",
            f"- Distinct road labels: {df['road_label'].nunique()}",
            f"- Years: {start}-{end}",
            f"- Modes: {sorted(df['mode'].unique().tolist())}",
            f"- Low-confidence rows: {(df['confidence'] == 'Low').mean():.1%}",
            "",
        ]
    )
    (reports_dir(config.project_root) / "road_risk_report.md").write_text(report, encoding="utf-8")
