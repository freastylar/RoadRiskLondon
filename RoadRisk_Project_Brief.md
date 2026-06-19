# RoadRisk London — Project Brief

## 1. Project Title

**RoadRisk London: Explainable ML for Collision Severity, Long-Term Safety Trends, and Safer-Street Prioritization**

## 2. Project Context

This project is for a data-science / model deployment course focused on the development, evaluation, monitoring, and deployment of models.

The goal is to build an interactive web application that uses official UK road-safety open data to understand, model, and prioritize road-safety risk in London.

The application should not be a generic crash map. It should be a decision-support and insight tool that helps answer:

> How has reported road-safety risk changed over decades, where are collisions most likely to become serious or fatal today, why are those locations risky, and which areas should be prioritized for safety review?

## 3. Core Insight

The main insight is:

> The places with the most collisions are not necessarily the places where collisions are most severe. A good safety-priority system should consider severity risk, vulnerable road-user involvement, recent serious/fatal collision density, long-term change, and data confidence.

The project should focus on explainable, useful data science rather than flashy but shallow visuals.

## 4. Scope Philosophy

The project must use the long historical horizon where it adds insight, but it must not let full-history complexity break the core pipeline.

There should be two related but separate analytical products:

```text
Full-history trend module
1979–latest available final year
→ historical trends, decade comparison, long-term road-safety change

Recent-window modeling module
configurable recent years, e.g. 2015–latest available final year
→ severity model, risk prediction, intervention priority, monitoring
```

This separation is important because decades of historical data are valuable for trend analysis, but not all historical data should automatically be treated as one homogeneous modeling population. London’s road environment, reporting practices, vehicle fleets, emergency response, cycling levels, and infrastructure have changed over time.

## 5. Year-Range Strategy

### 5.1 Configurable Year Ranges

The pipeline must support configurable year ranges. Do not hard-code one fixed year range throughout the project.

At minimum, support:

```bash
--start-year 1979 --end-year <latest_final_year> --mode trends
--start-year 2015 --end-year <latest_final_year> --mode modeling
--start-year 2020 --end-year <latest_final_year> --mode mvp
```

The implementation should determine the latest available final year from the available downloaded files or configuration.

### 5.2 Recommended Development Sequence

The first implementation may use the latest-five-year files or a recent subset to prove that the pipeline works.

However, the final project should aim to include:

1. A **full-history trend module** using all feasible years from 1979 onward.
2. A **recent-window severity model** using a configurable period such as 2015 onward.
3. A **sensitivity comparison** showing how model results change when the modeling window changes, for example:
   - latest 5 years
   - 2015 onward
   - all available years, if feasible

### 5.3 Why Not Use All Years for Everything Immediately?

Older data is not bad. It is valuable. But it should be handled carefully because:

- full-history files are larger and increase implementation risk;
- published fields and coding may vary over decades;
- injury severity reporting practices have changed;
- old road environments may be less relevant for current intervention prioritization;
- a model trained on old and recent periods together may learn period-specific reporting and infrastructure patterns rather than current risk.

Therefore:

- use full history for historical insight;
- use a recent configurable window for the primary predictive model;
- optionally compare modeling windows as a sensitivity study.

## 6. Primary Scope

### MVP Scope

The MVP should prove the full pipeline using a recent manageable subset first.

Recommended MVP:

- geography: London only;
- years: latest five final years or another recent configurable subset;
- target: KSI collision, where KSI means killed or seriously injured;
- main model: binary classifier predicting whether a reported injury collision is serious/fatal versus slight;
- main app output: safety-priority ranking and map.

### Final Desired Scope

After the MVP works:

- add historical trend analysis using all available final years from 1979 onward, if computationally feasible;
- expand the model window to a longer recent period such as 2015 onward;
- add a comparison between year-window choices;
- add optional traffic-adjusted risk only after the core safety model and app work.

## 7. Fixed Technology Stack

Use a Python-first stack.

### Required

- Python
- Streamlit
- Pandas
- GeoPandas
- PyArrow / Parquet
- Plotly
- PyDeck
- scikit-learn
- pytest
- ruff

### Optional

- DuckDB
- SHAP
- imbalanced-learn

### Do Not Use

Do not introduce these unless explicitly requested later:

- React
- Next.js
- FastAPI
- Dash
- Observable
- complex JavaScript animation frameworks
- live backend services
- app-time model training
- app-time raw data downloads

The project should be simple, correct, and data-first.

## 8. Data Sources

### 8.1 Main Dataset: UK DfT Road Safety Open Data

Use the UK Department for Transport road-safety open data.

The core data consists of three linked record-level tables:

1. **Collisions**
2. **Vehicles**
3. **Casualties**

These are linked using collision identifiers such as `accident_index`, `accident_year`, and/or `accident_reference`, depending on the schema version.

The data is coded, so category decoding must use the official DfT data guide / lookup tables.

### 8.2 Collision Table

Expected collision-level fields may include, but must be verified through schema inspection:

- accident_index
- accident_year
- accident_reference
- longitude
- latitude
- police_force
- accident_severity
- number_of_vehicles
- number_of_casualties
- date
- day_of_week
- time
- local_authority_district
- local_authority_ons_district
- local_authority_highway
- road_type
- speed_limit
- junction_detail
- junction_control
- light_conditions
- weather_conditions
- road_surface_conditions
- special_conditions_at_site
- carriageway_hazards
- urban_or_rural_area
- lsoa_of_accident_location

Do not assume these columns exist. Inspect actual source files first.

### 8.3 Vehicle Table

Expected vehicle-level fields may include, but must be verified:

- accident_index
- accident_year
- accident_reference
- vehicle_reference
- vehicle_type
- vehicle_manoeuvre
- vehicle_direction_from
- vehicle_direction_to
- junction_location
- skidding_and_overturning
- hit_object_in_carriageway
- vehicle_leaving_carriageway
- first_point_of_impact
- journey_purpose_of_driver
- sex_of_driver
- age_of_driver
- age_band_of_driver
- age_of_vehicle
- propulsion_code
- generic_make_model
- driver_imd_decile
- driver_home_area_type
- lsoa_of_driver

Do not assume these columns exist. Inspect actual source files first.

### 8.4 Casualty Table

Expected casualty-level fields may include, but must be verified:

- accident_index
- accident_year
- accident_reference
- vehicle_reference
- casualty_reference
- casualty_class
- casualty_severity
- sex_of_casualty
- age_of_casualty
- age_band_of_casualty
- pedestrian_location
- pedestrian_movement
- casualty_type
- casualty_home_area_type
- casualty_imd_decile
- lsoa_of_casualty

Do not assume these columns exist. Inspect actual source files first.

### 8.5 Optional Dataset: DfT Road Traffic / AADF Data

The traffic data is optional and should be added only after the core MVP works.

Potential uses:

- traffic-adjusted risk;
- collisions per traffic flow;
- confidence flag based on counted vs estimated traffic flow;
- road exposure denominator.

Do not attempt this in the first implementation phase.

### 8.6 Optional Dataset: ONS / London Boundaries

Optional later:

- London borough boundaries;
- LSOA boundaries;
- local authority lookup tables.

Use only if needed for maps or equity analysis.

## 9. Key Data Limitations

The project must state these limitations clearly:

1. The DfT road-safety dataset covers reported personal-injury collisions on public roads.
2. It does not include all crashes, near misses, private-road incidents, or unreported collisions.
3. Sensitive contributory factors may not be available in the public data.
4. Injury severity reporting practices have changed over time, so long-term severity trends must be interpreted carefully.
5. The model predicts statistical association, not causal impact.
6. Safety-priority rankings are decision-support outputs, not automatic policy recommendations.
7. If traffic-adjusted risk is added, estimated traffic-flow values should be treated as lower-confidence than counted values.
8. Older historical records are useful for trends, but may not represent current road environments.

## 10. Data Pipeline Requirements

The Streamlit app must not download, clean, join, or train on raw data at runtime.

The correct pipeline is:

```text
raw source files
    ↓
schema inspection reports
    ↓
clean processed Parquet tables
    ↓
historical trend tables
    ↓
recent-window modeling table
    ↓
trained model + evaluation artifacts
    ↓
small app-ready artifacts
    ↓
Streamlit app
```

The app should only read from:

- `data/app/`
- `models/registry/`

## 11. Repository Structure

Use this structure:

```text
roadrisk-london/
│
├── app.py
├── pages/
│   ├── 1_Safety_Map.py
│   ├── 2_Historical_Trends.py
│   ├── 3_Vulnerable_Users.py
│   ├── 4_Severity_Model.py
│   ├── 5_Intervention_Priority.py
│   ├── 6_Model_Monitoring.py
│   └── 7_Methodology.py
│
├── src/roadrisk/
│   ├── __init__.py
│   ├── config.py
│   ├── data_sources.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── inspect_schema.py
│   │   ├── clean_collisions.py
│   │   ├── clean_vehicles.py
│   │   ├── clean_casualties.py
│   │   ├── decode_categories.py
│   │   ├── filter_london.py
│   │   ├── build_historical_trends.py
│   │   └── build_app_tables.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_collision_features.py
│   │   ├── build_casualty_features.py
│   │   ├── spatial_features.py
│   │   └── priority_score.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   ├── explain.py
│   │   └── monitor.py
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── theme.py
│   │   ├── maps.py
│   │   ├── charts.py
│   │   ├── cards.py
│   │   └── tables.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── paths.py
│       └── validation.py
│
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_inspect_schema.py
│   ├── 03_build_processed_data.py
│   ├── 04_build_historical_trends.py
│   ├── 05_build_features.py
│   ├── 06_train_model.py
│   ├── 07_build_app_artifacts.py
│   └── 08_run_all_checks.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── app/
│
├── models/
│   ├── registry/
│   └── reports/
│
├── tests/
├── requirements.txt
├── pyproject.toml
├── AGENTS.md
├── README.md
└── RoadRisk_Project_Brief.md
```

## 12. Implementation Phases

### Phase 0 — Planning Only

Codex must first produce a plan. It should not implement code immediately.

The plan must include:

- files to create;
- data sources;
- configurable year-range strategy;
- tests;
- success criteria;
- risks;
- exact commands to run.

### Phase 1 — Data Ingestion and Schema Inspection

Goal:

Prove that the data can be downloaded or loaded, parsed, filtered to London, and saved as non-empty processed tables.

Start with a manageable recent subset or the latest-five-year files. The code must still be written so that the year range is configurable.

Required outputs:

- raw files in `data/raw/`;
- schema reports in `models/reports/`;
- London processed Parquet files in `data/processed/`.

Required reports:

- `models/reports/schema_collisions.md`
- `models/reports/schema_vehicles.md`
- `models/reports/schema_casualties.md`

Each schema report must include:

- row count;
- columns;
- dtypes;
- missingness;
- date range;
- available years;
- severity distribution;
- key categorical values;
- London row count;
- warnings / issues;
- notes about schema differences if multiple year files are inspected.

Required tests:

- raw or local input files exist;
- processed files are non-empty;
- key linking columns exist;
- London filtering is non-empty;
- latitude/longitude are valid where available;
- accident severity exists;
- date/time columns parse;
- year range configuration works.

### Phase 2 — Full-History Trend Tables

Goal:

Create app-ready historical trend tables from all feasible years.

This phase can be implemented after the recent-pipeline proof works. It should aggregate rather than expose full raw records to Streamlit.

Output examples:

- `data/app/historical_trends_yearly.parquet`
- `data/app/historical_trends_decade.parquet`
- `data/app/historical_trends_road_user.parquet`
- `data/app/historical_trends_london_vs_gb.parquet`, if GB comparison is feasible

Recommended aggregations:

- year;
- decade;
- severity;
- KSI count;
- KSI rate;
- road-user group;
- road type;
- speed limit;
- light conditions;
- weather conditions;
- London borough / local authority, if available.

Required tests:

- trend tables are non-empty;
- multiple years are present;
- KSI counts are non-null;
- yearly totals are internally consistent with processed data;
- long-term tables do not require Streamlit to load raw full-history data.

### Phase 3 — Modeling Table

Goal:

Create a clean collision-level modeling table from a configurable recent modeling window.

Output:

- `data/processed/model_collision_severity.parquet`
- `models/reports/modeling_table_report.md`

Target:

```text
is_ksi = 1 if accident_severity is Fatal or Serious, else 0
```

Feature examples, only if verified in schema:

- year;
- month;
- hour;
- day_of_week;
- weekend;
- speed_limit;
- road_type;
- junction_detail;
- junction_control;
- light_conditions;
- weather_conditions;
- road_surface_conditions;
- urban_or_rural_area;
- number_of_vehicles;
- number_of_casualties;
- has_pedestrian;
- has_cyclist;
- has_motorcyclist;
- has_child;
- has_elderly.

Do not include target leakage fields.

Required tests:

- modeling table is non-empty;
- target exists;
- target has both classes;
- modeling year range matches configuration;
- feature schema is saved;
- no target leakage columns are used.

### Phase 4 — Model Training and Evaluation

Goal:

Train and evaluate an explainable severity-risk model.

Required models:

1. Baseline model.
2. Main model using either:
   - logistic regression;
   - random forest;
   - HistGradientBoostingClassifier;
   - another scikit-learn model justified in the report.

Use a chronological split:

- train on earlier years;
- validate on a later year if enough years exist;
- test on the latest final year in the modeling window.

Do not use a random split as the primary evaluation.

Required metrics:

- ROC-AUC;
- PR-AUC;
- precision;
- recall;
- F1;
- confusion matrix;
- calibration curve data;
- performance by year;
- performance by vulnerable-user group if available.

Save:

- `models/registry/severity_model.joblib`
- `models/registry/feature_schema.json`
- `models/registry/model_metrics.json`
- `models/registry/model_card.md`
- `models/registry/feature_importance.parquet`
- `models/registry/calibration_data.parquet`

Required tests:

- model artifact exists;
- metrics JSON exists;
- metrics are valid numbers;
- feature schema matches training data;
- prediction function returns probabilities between 0 and 1.

### Phase 5 — Modeling-Window Sensitivity Analysis

Goal:

Compare how the severity model changes under different year windows.

This phase is a stretch goal but strongly recommended if time allows.

Compare at least two of:

- latest five years;
- 2015 onward;
- 2021 onward, to reduce COVID-period effects;
- all available years, if computationally feasible and schemas are stable.

Outputs:

- `models/reports/model_window_sensitivity.md`
- `data/app/model_window_comparison.parquet`

Compare:

- class balance;
- ROC-AUC;
- PR-AUC;
- KSI recall;
- calibration;
- top features;
- priority-ranking stability.

This should answer:

> Does adding older historical data improve the current severity model, or does it introduce instability because road conditions and reporting practices changed?

### Phase 6 — App Artifacts

Goal:

Create small, fast-loading files for Streamlit.

Required outputs:

- `data/app/collision_points_sample.parquet`
- `data/app/hex_or_grid_risk_summary.parquet`
- `data/app/historical_trends_yearly.parquet`
- `data/app/vulnerable_user_summary.parquet`
- `data/app/priority_locations.parquet`
- `data/app/model_metrics.json`
- `data/app/monitoring_summary.json`

Optional:

- `data/app/model_window_comparison.parquet`

Required tests:

- all app artifacts exist;
- all app artifacts are non-empty;
- map files contain valid coordinates or geometry;
- historical trend files contain multiple years;
- priority file contains a priority score;
- model metrics file is readable.

### Phase 7 — Streamlit MVP

Goal:

Build a simple, useful app from app artifacts only.

Pages:

1. Safety Map
2. Historical Trends
3. Vulnerable Users
4. Severity Model
5. Intervention Priority
6. Model Monitoring
7. Methodology

Rules:

- Do not download data in the app.
- Do not train models in the app.
- Do not process raw files in the app.
- Do not silently fall back to empty demo data.
- If demo data is used, label it clearly.

### Phase 8 — Intervention Priority

Goal:

Convert the model into a decision-support ranking.

Priority score should combine:

- predicted KSI risk;
- recent KSI density;
- vulnerable-user share;
- data confidence.

Optional later:

- traffic-adjusted risk;
- deprivation weighting.

The page must show:

- ranked table;
- map layer;
- explanation of the risk drivers;
- disclaimer that this is decision support, not causal proof.

### Phase 9 — Model Monitoring

Goal:

Show that the model is being treated as a deployed model.

Monitoring should include:

- model version;
- training years;
- validation/test years;
- modeling window;
- KSI class balance;
- missingness by year;
- performance by year;
- performance by road-user group;
- feature drift summary;
- calibration drift if available;
- known data limitations;
- sensitivity comparison across modeling windows, if implemented.

### Phase 10 — Polish and Deployment

Only after the data, model, and app work:

- improve layout;
- improve map styling;
- improve legends;
- improve cards;
- improve README;
- add deployment instructions;
- prepare final presentation notes.

## 13. App Page Details

### Page 1 — Safety Map

Purpose:

Show where London’s reported injury collisions and KSI collisions are concentrated.

Features:

- map of collision points or grid/hex bins;
- filters for year, severity, road user, weather, lighting, speed limit;
- summary cards:
  - total collisions;
  - KSI collisions;
  - KSI rate;
  - pedestrian/cyclist/motorcyclist involvement;
- top areas by KSI density.

### Page 2 — Historical Trends

Purpose:

Use the long historical horizon to show how road-safety patterns changed over decades.

Features:

- yearly KSI trend;
- decade comparison;
- London versus GB comparison, if feasible;
- road-user group trends;
- changes by road type, speed limit, lighting, or weather;
- annotation or warning for known reporting/specification changes.

This page should use aggregated historical artifacts only, not raw full-history data.

### Page 3 — Vulnerable Users

Purpose:

Explore how severity differs for pedestrians, cyclists, motorcyclists, children, and elderly road users.

Charts:

- KSI rate by road-user group;
- KSI rate by hour;
- KSI rate by light conditions;
- KSI rate by speed limit;
- KSI rate by weather / road surface;
- map filtered to selected group.

### Page 4 — Severity Model

Purpose:

Make the ML model transparent.

Include:

- target definition;
- modeling year window;
- train/validation/test split;
- model metrics;
- confusion matrix;
- calibration plot;
- feature importance;
- example risk explanation;
- optional model-window sensitivity results.

Emphasize PR-AUC and recall because KSI is an imbalanced target.

### Page 5 — Intervention Priority

Purpose:

Turn the model into a decision-support tool.

Include:

- ranked high-priority locations;
- map of priority areas;
- selected-location explanation;
- suggested audit category, based on observed risk pattern.

Possible audit suggestions:

- lighting review;
- junction safety review;
- pedestrian crossing review;
- cyclist infrastructure review;
- speed-management review;
- road-surface / wet-weather safety review.

These must be phrased as suggestions, not guaranteed interventions.

### Page 6 — Model Monitoring

Purpose:

Show model health and deployment awareness.

Include:

- model version;
- training and test years;
- modeling window;
- missingness summary;
- feature drift summary;
- performance by year;
- performance by road-user group;
- performance by borough or local authority if available;
- calibration drift;
- sensitivity results if implemented.

### Page 7 — Methodology

Purpose:

Explain data, model, limitations, and ethics.

Must include:

- dataset description;
- long-history versus recent-modeling-window distinction;
- target definition;
- feature engineering;
- model choice;
- evaluation strategy;
- limitations;
- fairness concerns;
- non-causal interpretation;
- warning about historical reporting/specification changes.

## 14. Testing Requirements

Use `pytest`.

Every phase must add tests.

At minimum, test:

- processed data is non-empty;
- key columns exist;
- coordinates are valid;
- London filtering works;
- year-range configuration works;
- historical trend artifacts contain multiple years;
- target definition works;
- feature schema is stable;
- model artifacts exist;
- app artifacts exist;
- prediction outputs are valid;
- priority scores are bounded and non-null.

Use `ruff` for linting.

Required commands:

```bash
ruff check .
pytest -q
```

## 15. Success Criteria

The project is successful if:

1. The data pipeline creates non-empty London datasets.
2. The pipeline supports configurable year ranges.
3. The app includes a long-term historical trend module if computationally feasible.
4. The model predicts KSI risk using a recent configurable modeling window.
5. The model is evaluated with meaningful metrics, especially PR-AUC, recall, and calibration.
6. The app presents risk, historical trends, vulnerable-user patterns, model results, intervention priorities, and monitoring.
7. The app runs using `streamlit run app.py`.
8. The app does not process raw data or train models at runtime.
9. The methodology clearly states limitations and avoids causal overclaims.
10. The README explains how to reproduce the pipeline and run the app.

## 16. Non-Goals

Do not attempt these in the first version:

- fully animated maps;
- real-time road safety data;
- exact causal intervention estimates;
- full GB interactive app interface;
- deep learning;
- complex geospatial road-network snapping;
- heavy live computation inside Streamlit;
- production backend;
- React frontend.

## 17. Guiding Principle

This project should prioritize:

```text
correct data pipeline
→ configurable historical scope
→ clear modeling target
→ honest evaluation
→ long-term trend insight
→ useful intervention ranking
→ simple but effective app
```

The goal is not to build the most visually impressive dashboard. The goal is to build a credible data-science application that produces meaningful, explainable urban-safety insights.
