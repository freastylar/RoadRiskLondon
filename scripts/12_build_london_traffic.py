"""Download DfT road-traffic data and build the London traffic artifacts.

End-to-end, reproducible source for the traffic files the Trip Risk page needs.
It downloads two public DfT datasets, filters them to the 33 London boroughs,
and writes the derived parquets:

- ``london_raw_counts.parquet``        - London rows of the raw counts (slim columns)
- ``london_aadf_by_point_year.parquet``- mean flow per count point per year (with grid)
- ``london_la_traffic.parquet``        - local-authority traffic for London boroughs
- ``london_hourly_profiles.parquet``   - mean flow by road type x weekday/weekend x hour

The raw counts CSV is large (~1 GB), so it is streamed in chunks and skipped if
already present.

Usage:
    python scripts/12_build_london_traffic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadrisk.features.spatial_features import add_simple_grid  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "traffic"

# Public DfT road-traffic statistics downloads (roadtraffic.dft.gov.uk).
RAW_COUNTS_URL = "https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/dft_traffic_counts_raw_counts.zip"
RAW_COUNTS_CSV = RAW_DIR / "dft_traffic_counts_raw_counts.csv"
LA_TRAFFIC_URL = "https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/local_authority_traffic.csv"
LA_TRAFFIC_CSV = RAW_DIR / "local_authority_traffic.csv"

LONDON = {f"E09{n:06d}" for n in range(1, 34)}  # E09000001..E09000033
RAW_USECOLS = [
    "count_point_id", "year", "local_authority_code", "road_type",
    "latitude", "longitude", "link_length_km",
    "pedal_cycles", "all_HGVs", "all_motor_vehicles",
]
PROFILE_USECOLS = [
    "local_authority_code", "road_type", "hour", "count_date",
    "all_motor_vehicles", "pedal_cycles",
]


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists, skipping: {dest.name}")
        return
    print(f"  downloading {url} ...")
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if url.endswith(".zip"):
        import io
        import zipfile

        with urllib.request.urlopen(url, timeout=600) as resp:  # noqa: S310
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(name) as src, dest.open("wb") as out:
                out.write(src.read())
    else:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def _build_london_raw_counts() -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(RAW_COUNTS_CSV, usecols=RAW_USECOLS, chunksize=200_000):
        chunks.append(chunk[chunk["local_authority_code"].isin(LONDON)])
    raw = pd.concat(chunks, ignore_index=True)
    raw.to_parquet(RAW_DIR / "london_raw_counts.parquet", index=False)
    print(f"  london_raw_counts.parquet: {len(raw):,} rows")
    return raw


def _build_aadf(raw: pd.DataFrame) -> None:
    agg = (
        raw.groupby(["count_point_id", "year", "local_authority_code", "road_type"], as_index=False)
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            link_length_km=("link_length_km", "first"),
            all_motor_vehicles=("all_motor_vehicles", "mean"),
            all_HGVs=("all_HGVs", "mean"),
            pedal_cycles=("pedal_cycles", "mean"),
        )
    )
    agg = add_simple_grid(agg)
    agg.to_parquet(RAW_DIR / "london_aadf_by_point_year.parquet", index=False)
    print(f"  london_aadf_by_point_year.parquet: {len(agg):,} point-years")


def _build_la_traffic() -> None:
    la = pd.read_csv(LA_TRAFFIC_CSV)
    la_london = la[la["local_authority_code"].isin(LONDON)].copy()
    la_london.to_parquet(RAW_DIR / "london_la_traffic.parquet", index=False)
    print(f"  london_la_traffic.parquet: {len(la_london):,} rows")


def _build_hourly_profiles() -> None:
    sums: dict[tuple, list] = {}
    bikes: dict[tuple, list] = {}
    for chunk in pd.read_csv(RAW_COUNTS_CSV, usecols=PROFILE_USECOLS, chunksize=300_000):
        chunk = chunk[chunk["local_authority_code"].isin(LONDON)].dropna(subset=["hour"])
        if chunk.empty:
            continue
        chunk["hour"] = chunk["hour"].astype(int)
        dow = pd.to_datetime(chunk["count_date"], errors="coerce").dt.dayofweek
        chunk["day_type"] = np.where(dow.isin([5, 6]), "Weekend", "Weekday")
        chunk["rt"] = chunk["road_type"].fillna("Minor")
        for key, sub in chunk.groupby(["rt", "day_type", "hour"]):
            s = sums.setdefault(key, [0.0, 0])
            s[0] += float(sub["all_motor_vehicles"].sum())
            s[1] += len(sub)
            b = bikes.setdefault(key, [0.0, 0])
            b[0] += float(sub["pedal_cycles"].sum())
            b[1] += len(sub)
    rows = [
        {"road_type": rt, "day_type": dt, "hour": hr,
         "motor_mean": tot / n if n else 0.0,
         "bike_mean": bikes[(rt, dt, hr)][0] / bikes[(rt, dt, hr)][1] if bikes[(rt, dt, hr)][1] else 0.0}
        for (rt, dt, hr), (tot, n) in sums.items()
    ]
    out = pd.DataFrame(rows).sort_values(["road_type", "day_type", "hour"])
    out.to_parquet(RAW_DIR / "london_hourly_profiles.parquet", index=False)
    print(f"  london_hourly_profiles.parquet: {len(out)} rows")


def main() -> None:
    print("Downloading DfT traffic data...")
    _download(LA_TRAFFIC_URL, LA_TRAFFIC_CSV)
    _download(RAW_COUNTS_URL, RAW_COUNTS_CSV)
    print("Building London traffic artifacts...")
    raw = _build_london_raw_counts()
    _build_aadf(raw)
    _build_la_traffic()
    _build_hourly_profiles()
    print("Done. Now run scripts/15_build_road_risk.py to build the Trip Risk table.")


if __name__ == "__main__":
    main()
