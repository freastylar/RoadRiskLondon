# RoadRisk London

RoadRisk London is a data-first Streamlit application for understanding long-term London road-safety trends, modeling recent KSI collision severity risk, and ranking locations for safety review.

The project follows the pipeline:

```text
raw source files
-> schema inspection reports
-> processed London Parquet tables
-> historical trend tables
-> recent-window modeling table
-> trained model and evaluation artifacts
-> small app-ready artifacts
-> Streamlit app
```

The app reads only from `data/app/` and `models/registry/`. It does not download raw data, clean raw files, or train models at runtime.

## Current App Pages

- **Safety Map**: point and heatmap views. Metric cards use full processed London data; point rendering uses a sample for performance.
- **Historical Trends**: long-run London collision severity trends and road-user casualty trends.
- **Borough Overview**: borough ranking, table, and borough boundary map.
- **Collision Severity Factors**: descriptive condition analysis for speed, road type, junctions, light, weather, and road surface.
- **Model Evaluation**: chronological model metrics, confusion matrix, calibration, feature importance, threshold tradeoffs, vulnerable-user slices, and borough performance slices.
- **Model Monitoring**: yearly KSI class balance, missingness summary, test metrics, and limitations.
- **Methodology**: source, target, modelling, priority-score, and limitation notes.
- **Trip Risk by Road**: relative collision risk along real London roads by hour, day, weather, and travel mode. Crashes are snapped to OpenStreetMap road centrelines; risk is shown as a multiple of the typical road (a relative index, not an absolute per-trip probability).

## Setup

```bash
python -m pip install -r requirements.txt
```

## Setup for collaborators

After cloning, the app needs prebuilt data artifacts that are **not** stored in
git (the parquet/JSON files under `data/app/` and `models/registry/` are
gitignored). You have two options:

1. **Share the artifacts directly** (fastest): copy the `data/app/` and
   `models/registry/` folders from a teammate who has already built them. The
   app then runs immediately with `streamlit run app.py`.
2. **Rebuild from source**: run the pipeline below to regenerate everything from
   the raw DfT downloads.

The **Trip Risk by Road** page additionally needs road and traffic artifacts:

```bash
python scripts/14_fetch_london_roads.py                                  # OSM road network (needs network)
python scripts/13_build_hourly_traffic_profiles.py                       # hourly exposure profiles
python scripts/15_build_road_risk.py --mode modeling --start-year 2015 --end-year 2024
```

This reads the London-filtered traffic counts in `data/raw/traffic/` and writes
`road_risk_table.parquet`, `road_risk_meta.json`, and `road_base_geometry.parquet`
into `data/app/`.

## MVP Pipeline

```bash
python scripts/01_download_data.py --mode mvp --start-year 2020 --end-year 2024
python scripts/02_inspect_schema.py --mode mvp --start-year 2020 --end-year 2024
python scripts/03_build_processed_data.py --mode mvp --start-year 2020 --end-year 2024
python scripts/05_build_features.py --mode mvp --start-year 2020 --end-year 2024
python scripts/06_train_model.py --start-year 2020 --end-year 2024 --model-name severity_model_mvp
python scripts/04_build_historical_trends.py --mode mvp --start-year 2020 --end-year 2024
python scripts/07_build_app_artifacts.py --mode mvp --start-year 2020 --end-year 2024 --include-monitoring
ruff check .
pytest -q
streamlit run app.py
```

`scripts/06_train_model.py` validates that the existing modelling table and feature schema match the requested year range. If it fails, rebuild the modelling table with `scripts/05_build_features.py` for the same years before training.

## Required App Artifacts

Before Streamlit starts, the app checks that these prebuilt artifacts exist and are non-empty:

```text
data/app/collision_points_sample.parquet
data/app/safety_map_yearly.parquet
data/app/hex_or_grid_risk_summary.parquet
data/app/historical_severity_yearly.parquet
data/app/historical_road_user_severity_yearly.parquet
data/app/historical_trends_yearly.parquet
data/app/borough_severity_yearly.parquet
data/app/severity_drivers_yearly.parquet
data/app/vulnerable_user_summary.parquet
data/app/priority_locations.parquet
data/app/model_metrics.json
data/app/monitoring_summary.json
models/registry/severity_model.joblib
models/registry/feature_schema.json
models/registry/feature_importance.parquet
models/registry/calibration_data.parquet
```

If any are missing, rebuild the pipeline artifacts rather than adding fallback or demo data.

## Full-History Trends

```bash
python scripts/01_download_data.py --mode trends --start-year 1979 --end-year 2024
python scripts/02_inspect_schema.py --mode trends --start-year 1979 --end-year 2024
python scripts/03_build_processed_data.py --mode trends --start-year 1979 --end-year 2024
python scripts/04_build_historical_trends.py --mode trends --start-year 1979 --end-year 2024
```

Full-history files are large, so scripts use chunked reads where practical and aggregate early.

## Modeling Window

```bash
python scripts/05_build_features.py --mode modeling --start-year 2015 --end-year 2024
python scripts/06_train_model.py --start-year 2015 --end-year 2024 --model-name severity_model
python scripts/07_build_app_artifacts.py --mode modeling --start-year 2015 --end-year 2024 --include-monitoring
```

## Validation

```bash
python scripts/08_run_all_checks.py
```

This currently runs:

```bash
ruff check .
pytest -q
```

To also verify that all required app and model artifacts exist before launching Streamlit, run:

```bash
python scripts/08_run_all_checks.py --include-artifacts
```

## Limitations

DfT road-safety open data covers reported personal-injury collisions on public roads. It excludes unreported collisions, near misses, private-road incidents, and sensitive contributory factors. Injury severity reporting changed over time, so long-term severity trends require care. Model outputs are statistical decision-support signals, not causal proof or automatic policy recommendations.
