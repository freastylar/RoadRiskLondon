from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from roadrisk.config import Mode, PipelineConfig, YearRange


@pytest.fixture()
def sample_project(tmp_path: Path) -> Path:
    for directory in [
        "data/raw",
        "data/interim",
        "data/processed",
        "data/app",
        "models/registry",
        "models/reports",
    ]:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)

    collisions = []
    vehicles = []
    casualties = []
    severities = [1, 2, 3, 3]
    casualty_types = [0, 1, 3, 9]
    for year in range(2020, 2025):
        for idx, severity in enumerate(severities, start=1):
            collision_id = f"{year}{idx:04d}"
            collisions.append(
                {
                    "accident_index": collision_id,
                    "accident_year": year,
                    "accident_reference": f"{idx:04d}",
                    "longitude": -0.12 + idx * 0.01,
                    "latitude": 51.50 + idx * 0.01,
                    "local_authority_ons_district": "E09000001",
                    "accident_severity": severity,
                    "number_of_vehicles": 1 + idx % 2,
                    "number_of_casualties": 1,
                    "date": f"0{idx}/02/{year}",
                    "time": f"0{idx}:30",
                    "day_of_week": idx,
                    "road_type": idx,
                    "speed_limit": 20 + idx * 10,
                    "junction_detail": idx,
                    "junction_control": idx,
                    "light_conditions": idx,
                    "weather_conditions": idx,
                    "road_surface_conditions": idx,
                    "urban_or_rural_area": 1,
                }
            )
            vehicles.append(
                {
                    "accident_index": collision_id,
                    "accident_year": year,
                    "accident_reference": f"{idx:04d}",
                    "vehicle_reference": 1,
                    "vehicle_type": 9,
                }
            )
            casualties.append(
                {
                    "accident_index": collision_id,
                    "accident_year": year,
                    "accident_reference": f"{idx:04d}",
                    "vehicle_reference": 1,
                    "casualty_reference": 1,
                    "casualty_severity": severity,
                    "casualty_type": casualty_types[idx - 1],
                    "age_of_casualty": 10 if idx == 1 else 70 if idx == 2 else 30,
                }
            )

    raw = tmp_path / "data" / "raw"
    files = {
        "collisions": raw / "collisions_sample.csv",
        "vehicles": raw / "vehicles_sample.csv",
        "casualties": raw / "casualties_sample.csv",
    }
    pd.DataFrame(collisions).to_csv(files["collisions"], index=False)
    pd.DataFrame(vehicles).to_csv(files["vehicles"], index=False)
    pd.DataFrame(casualties).to_csv(files["casualties"], index=False)
    manifest = {
        "mode": "mvp",
        "start_year": 2020,
        "end_year": 2024,
        "latest_final_year": 2024,
        "files": [
            {
                "table": table,
                "path": str(path),
                "label": f"{table} sample",
                "url": "file://sample",
                "filename": path.name,
                "start_year": 2020,
                "end_year": 2024,
                "final": True,
                "provisional": False,
                "bytes": path.stat().st_size,
                "sha256": "sample",
            }
            for table, path in files.items()
        ],
    }
    (raw / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    boundary_source = Path(__file__).resolve().parents[1] / "data/app/borough_boundaries.geojson"
    if boundary_source.exists():
        shutil.copyfile(boundary_source, tmp_path / "data/app/borough_boundaries.geojson")
    return tmp_path


@pytest.fixture()
def sample_config(sample_project: Path) -> PipelineConfig:
    return PipelineConfig(
        mode=Mode.MVP,
        year_range=YearRange(2020, 2024),
        project_root=sample_project,
    ).validate()
