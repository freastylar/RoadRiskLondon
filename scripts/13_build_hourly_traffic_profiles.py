"""Aggregate DfT raw count sessions into London hourly traffic profiles.

Streams the large raw counts CSV, keeps London rows, and aggregates mean flow by
(road_type, weekday/weekend, hour). DfT counts only cover daytime weekday hours,
so the output is partial; ``roadrisk.data.hourly_profiles`` fills the gaps with a
standard profile shape and flags which hours are measured.

Usage:
    python scripts/13_build_hourly_traffic_profiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "traffic" / "dft_traffic_counts_raw_counts.csv"
OUT = PROJECT_ROOT / "data" / "raw" / "traffic" / "london_hourly_profiles.parquet"

LONDON = {f"E09{n:06d}" for n in range(1, 34)}
USECOLS = ["local_authority_code", "road_type", "hour", "count_date", "all_motor_vehicles", "pedal_cycles"]


def main() -> None:
    if not RAW_CSV.exists():
        sys.exit(f"Raw counts file not found: {RAW_CSV}")
    sums: dict[tuple, list] = {}
    bikes: dict[tuple, list] = {}
    for chunk in pd.read_csv(RAW_CSV, usecols=USECOLS, chunksize=300_000):
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

    rows = []
    for (rt, dt, hr), (tot, n) in sums.items():
        bt, bn = bikes[(rt, dt, hr)]
        rows.append(
            {
                "road_type": rt,
                "day_type": dt,
                "hour": hr,
                "motor_mean": tot / n if n else 0.0,
                "bike_mean": bt / bn if bn else 0.0,
            }
        )
    out = pd.DataFrame(rows).sort_values(["road_type", "day_type", "hour"])
    out.to_parquet(OUT, index=False)
    print(f"Wrote {len(out)} profile rows to {OUT}")
    print("Hours present:", sorted(out["hour"].unique().tolist()))


if __name__ == "__main__":
    main()
