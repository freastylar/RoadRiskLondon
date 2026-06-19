# RoadRisk Improvement Log

## 2026-06-19 - Cycle 1: Priority Ranking Explainability

### What was inspected
- Confirmed real model artifacts exist in `models/registry/`, including `severity_model.joblib`, `model_metrics.json`, `feature_importance.parquet`, and `calibration_data.parquet`.
- Confirmed app artifacts exist in `data/app/`, including `priority_locations.parquet`, map samples, historical tables, borough tables, severity-factor tables, and monitoring summary.
- Reviewed `src/roadrisk/features/priority_score.py` and found the score used real model/grid data but explained little beyond numeric components.
- Reviewed KSI target construction in `src/roadrisk/features/build_collision_features.py` and `src/roadrisk/data/decode_categories.py`.
- Reviewed existing priority tests and found they only checked score bounds and sort order.

### What changed
- Added `observed_ksi_rate`, `observed_severity_component`, `priority_band`, `priority_reason`, and `audit_focus` to priority scoring.
- Changed `data_confidence` to use a log-scaled record count, so high-count cells still gain confidence but do not dominate linearly.
- Changed priority score to combine modelled KSI risk, observed KSI count, observed KSI rate, vulnerable-user share, and data confidence.
- Updated the home page to show a readable priority table instead of raw internal columns.
- Rebuilt `data/app/priority_locations.parquet` through `scripts/07_build_app_artifacts.py`.
- Added tests for explanation columns, observed KSI rate, confidence bounds, and invalid zero-record input.

### Why it improves the project
- Makes the intervention-priority ranking more transparent and useful for decision support.
- Reduces the chance that users treat a raw score as unexplained truth.
- Improves data integrity by rejecting impossible zero-record priority cells.
- Keeps the app data-first: the Streamlit page still reads only prebuilt app artifacts.

### Files changed
- `app.py`
- `src/roadrisk/features/priority_score.py`
- `tests/test_priority_score.py`
- `tests/test_app_artifacts.py`
- `data/app/priority_locations.parquet`
- `IMPROVEMENT_LOG.md`

### Checks run
- `python scripts/07_build_app_artifacts.py --mode mvp --start-year 2020 --end-year 2024 --include-monitoring` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 13 tests

### Remaining issues
- The score weights are simple and transparent but still heuristic.
- Priority locations are grid cells, not road segments or junctions.
- No traffic exposure data is used, so rankings remain decision-support signals rather than causal or exposure-adjusted risk.

## 2026-06-19 - Cycle 2: Model Evaluation Page

### What was inspected
- Confirmed real model artifacts exist in `models/registry/`, including metrics, calibration data, feature schema, feature importance, and serialized model files.
- Reviewed `models/registry/model_metrics.json` and confirmed it contains chronological train/validation/test years, baseline metrics, and model metrics.
- Reviewed `models/registry/feature_importance.parquet` and `models/registry/calibration_data.parquet`.
- Reviewed existing model tests and found they checked model creation but not the full set of evaluation artifacts used by the app.

### What changed
- Added a new Streamlit page: `pages/5_Model_Evaluation.py`.
- The page shows chronological split years, ROC-AUC, PR-AUC, baseline PR-AUC, precision, recall, confusion matrix, calibration curve, and feature importance.
- Added `load_registry_parquet()` to `src/roadrisk/ui/tables.py` so pages can load validated registry Parquet artifacts.
- Expanded `tests/test_model_training.py` to verify saved metrics, confusion matrix shape, feature importance, and calibration artifacts.

### Why it improves the project
- Makes the data-science and evaluation story visible in the app instead of hiding it in JSON files.
- Shows that the model is evaluated chronologically, not with a random split.
- Presents the model honestly as a ranking signal with high recall and modest precision.
- Improves artifact integrity tests for the files that support the model evaluation page.

### Files changed
- `pages/5_Model_Evaluation.py`
- `src/roadrisk/ui/tables.py`
- `tests/test_model_training.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 13 tests
- Streamlit render check at `http://localhost:8505/Model_Evaluation` - passed

### Remaining issues
- Feature importance labels are still one-hot encoded feature names, so some categorical features are technical rather than reader-friendly.
- The page does not yet show performance slices by borough or vulnerable-user group.
- Calibration is shown for the test year only.

## 2026-06-19 - Cycle 3: Decoded Model Feature Labels

### What was inspected
- Rendered the new model evaluation page and found the feature-importance chart still exposed one-hot encoded feature names such as `road_type_9` and `junction_detail_-1`.
- Reviewed `src/roadrisk/models/explain.py`, which only loaded feature importance and did not add display labels.
- Confirmed existing decoded STATS19 condition labels were already available in `src/roadrisk/data/decode_categories.py`.

### What changed
- Added `feature_display_label()` and `add_feature_display_labels()` to `src/roadrisk/models/explain.py`.
- Decoded condition features such as road type, junction detail/control, light, weather, road surface, and urban/rural values.
- Decoded borough ONS/highway codes where possible and common boolean/numeric feature names.
- Updated `pages/5_Model_Evaluation.py` so the feature-importance chart uses readable labels while preserving the original raw feature name in hover data.
- Added `tests/test_model_explain.py` for decoded condition, borough, boolean, and preservation behavior.

### Why it improves the project
- Avoids repeating the earlier `Code X` usability problem in the model interpretation page.
- Makes feature importance understandable to non-technical reviewers.
- Keeps raw model feature names available for traceability while improving the user-facing labels.

### Files changed
- `src/roadrisk/models/explain.py`
- `pages/5_Model_Evaluation.py`
- `tests/test_model_explain.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 16 tests
- Streamlit render check at `http://localhost:8505/Model_Evaluation` - passed

### Remaining issues
- `local_authority_district_*` feature values are still shown as local-authority district codes because the current artifact does not include a verified lookup for those numeric values.
- Feature importance remains logistic-regression association, not causal importance.

## 2026-06-19 - Cycle 4: Model Monitoring Page

### What was inspected
- Confirmed `data/app/monitoring_summary.json` exists and contains split metadata, yearly class balance, yearly missingness, test metrics, and limitations.
- Reviewed the current page list and found monitoring was not visible in the app.
- Reviewed `tests/test_app_artifacts.py` and found it checked artifact existence but not the monitoring JSON fields used by a page.

### What changed
- Added `pages/6_Model_Monitoring.py`.
- The page shows train/validation/test years, latest KSI rate, KSI-rate change over the modelling window, yearly KSI-rate trend, yearly record/KSI counts, missingness trend, current test-year metrics, and known limitations.
- Expanded `tests/test_app_artifacts.py` so monitoring summary fields required by the page are validated.

### Why it improves the project
- Makes static model monitoring visible rather than hidden in JSON.
- Helps users see whether the target rate shifts across years.
- Reinforces that the model is evaluated and monitored from prebuilt artifacts, not retrained inside Streamlit.

### Files changed
- `pages/6_Model_Monitoring.py`
- `tests/test_app_artifacts.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 16 tests
- Streamlit render check at `http://localhost:8505/Model_Monitoring` - passed

### Remaining issues
- Monitoring is static and based on the current artifact build only.
- The monitoring summary does not yet include performance slices by borough or vulnerable-user group.
- Missingness is aggregated as a mean share, which can hide feature-specific missingness changes.

## 2026-06-19 - Cycle 5: Severity Aggregate Invariant Tests

### What was inspected
- Reviewed `tests/test_app_artifacts.py` and found app artifacts were checked for existence, bounds, and required columns, but not all severity arithmetic invariants.
- Reviewed `tests/test_historical_trends.py` and found historical severity tables were checked for required columns but not whether severity totals reconciled.

### What changed
- Added invariant checks for borough severity artifacts:
  - fatal + serious = KSI
  - fatal + serious + slight = total collisions
- Added the same checks for severity-factor artifacts.
- Added the same checks for historical yearly severity artifacts.

### Why it improves the project
- Protects core KSI calculations used across the historical, borough, and severity-factor pages.
- Reduces the chance of silently displaying internally inconsistent aggregate tables.
- Improves test coverage for key calculations rather than only file existence.

### Files changed
- `tests/test_app_artifacts.py`
- `tests/test_historical_trends.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 16 tests

### Remaining issues
- Tests verify internal arithmetic consistency, not agreement with external official aggregate publications.
- More slice-specific tests could be added for borough shares and road-user casualty shares.

## 2026-06-19 - Cycle 6: Vulnerable-User Model Performance Slices

### What was inspected
- Reviewed `models/registry/model_metrics.json` and found aggregate validation/test metrics but no subgroup performance metrics.
- Reviewed `data/processed/model_collision_severity.parquet` and confirmed vulnerable-user flags exist in the modelling table.
- Reviewed `pages/5_Model_Evaluation.py` and found it could display aggregate model metrics but not whether performance differs for vulnerable-user slices.

### What changed
- Added vulnerable-user slice metric generation in `src/roadrisk/models/train.py`.
- Test-year and validation-year metrics now include slices where both KSI and non-KSI examples exist:
  - any vulnerable road user involved
  - no vulnerable road user flag
  - pedestrian involved
  - cyclist involved
  - motorcyclist involved
  - child casualty involved
  - older casualty involved
- Updated `pages/5_Model_Evaluation.py` to show a visible test-year performance slice table.
- Expanded `tests/test_model_training.py` to require slice metrics in saved model artifacts.
- Regenerated `models/registry/model_metrics.json` and rebuilt app artifacts so `data/app/model_metrics.json` includes the new slices.

### Why it improves the project
- Strengthens the data-science evaluation story beyond aggregate metrics.
- Helps reveal whether the model behaves differently for vulnerable-user groups.
- Makes model limitations and subgroup behavior more transparent for decision support.

### Files changed
- `src/roadrisk/models/train.py`
- `pages/5_Model_Evaluation.py`
- `tests/test_model_training.py`
- `models/registry/model_metrics.json`
- `models/registry/severity_model.joblib`
- `models/registry/severity_model_mvp.joblib`
- `models/registry/feature_importance.parquet`
- `models/registry/calibration_data.parquet`
- `data/app/model_metrics.json`
- `data/app/monitoring_summary.json`
- `IMPROVEMENT_LOG.md`

### Checks run
- `python scripts/06_train_model.py --start-year 2020 --end-year 2024 --model-name severity_model_mvp` - passed
- `python scripts/07_build_app_artifacts.py --mode mvp --start-year 2020 --end-year 2024 --include-monitoring` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 16 tests
- Streamlit render check at `http://localhost:8505/Model_Evaluation` - passed

### Remaining issues
- Slice metrics are still observational evaluation summaries, not fairness guarantees.
- Smaller slices can be noisier than aggregate test metrics.
- Borough-level performance slices are still not implemented.

## 2026-06-19 - Cycle 7: Model Threshold Tradeoff

### What was inspected
- Reviewed the model evaluation page and metrics artifacts after adding vulnerable-user slices.
- Found that the page still emphasized the default `0.5` threshold, even though KSI is imbalanced and threshold choice strongly changes recall and precision.
- Confirmed existing model probabilities and labels are sufficient to compute threshold tradeoff metrics without retraining inside Streamlit.

### What changed
- Added `threshold_metrics_table()` to `src/roadrisk/models/evaluate.py`.
- Updated `src/roadrisk/models/train.py` so validation and test metrics include precision, recall, F1, and predicted-positive rate at thresholds 0.3, 0.4, 0.5, 0.6, and 0.7.
- Updated `pages/5_Model_Evaluation.py` with a Threshold Tradeoff section containing a Plotly line chart and visible threshold table.
- Expanded `tests/test_model_training.py` so threshold metrics are required and bounded.
- Regenerated model artifacts and rebuilt app artifacts.

### Why it improves the project
- Makes the imbalanced-class evaluation more honest and useful.
- Shows that the model can be tuned for higher recall or higher precision depending on the intervention workflow.
- Avoids misleading users into treating the default `0.5` threshold as the only meaningful operating point.

### Files changed
- `src/roadrisk/models/evaluate.py`
- `src/roadrisk/models/train.py`
- `pages/5_Model_Evaluation.py`
- `tests/test_model_training.py`
- `models/registry/model_metrics.json`
- `models/registry/severity_model.joblib`
- `models/registry/severity_model_mvp.joblib`
- `models/registry/feature_importance.parquet`
- `models/registry/calibration_data.parquet`
- `data/app/model_metrics.json`
- `data/app/monitoring_summary.json`
- `IMPROVEMENT_LOG.md`

### Checks run
- `python scripts/06_train_model.py --start-year 2020 --end-year 2024 --model-name severity_model_mvp` - passed
- `python scripts/07_build_app_artifacts.py --mode mvp --start-year 2020 --end-year 2024 --include-monitoring` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 16 tests
- Streamlit render check at `http://localhost:8505/Model_Evaluation` - passed

### Remaining issues
- Thresholds are fixed reference points, not optimized for a specific operational cost function.
- No recommended threshold is selected yet because the project has not defined a policy preference between recall and precision.

## 2026-06-19 - Cycle 8: Safety Map Full-Data Metric Integrity

### What was inspected
- Reviewed `pages/1_Safety_Map.py` and found metric cards were calculated from `collision_points_sample.parquet`.
- Reviewed `src/roadrisk/data/build_app_tables.py` and confirmed the point artifact is intentionally sampled to 25,000 rows for map performance.
- Compared the point sample with full processed app data and confirmed the sample is much smaller than the full 111,462-collision recent-window dataset.

### What changed
- Added `data/app/safety_map_yearly.parquet`, built from the full enriched collision data before point sampling.
- Updated `src/roadrisk/ui/tables.py` so the new summary artifact is required before app launch.
- Updated `pages/1_Safety_Map.py` so metric cards use full-data yearly totals, while the map still uses the sampled point layer.
- Added a page caption explaining that metrics use all processed London collisions and the point map is sampled.
- Expanded `tests/test_app_artifacts.py` to validate the new Safety Map summary artifact and its rate bounds.

### Why it improves the project
- Fixes a potentially misleading metric display where map cards could undercount collisions because they were based on a sampled visualization artifact.
- Preserves map performance while keeping summary numbers truthful.
- Strengthens the app's data-first separation between display samples and aggregate facts.

### Files changed
- `src/roadrisk/data/build_app_tables.py`
- `src/roadrisk/ui/tables.py`
- `pages/1_Safety_Map.py`
- `tests/test_app_artifacts.py`
- `data/app/safety_map_yearly.parquet`
- `IMPROVEMENT_LOG.md`

### Checks run
- `python scripts/07_build_app_artifacts.py --mode mvp --start-year 2020 --end-year 2024 --include-monitoring` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 16 tests
- Streamlit render check at `http://localhost:8505/Safety_Map` - passed

### Remaining issues
- The map point layer is still sampled, so visual density is approximate in point mode.
- A future improvement could use aggregated grid layers as the default map for full-data spatial density.

## 2026-06-19 - Cycle 9: Training Window Guard

### What was inspected
- Reviewed `scripts/06_train_model.py` and found it accepted `--start-year` and `--end-year`.
- Reviewed `src/roadrisk/models/train.py` and confirmed `train_model()` uses the existing modelling table and feature schema rather than rebuilding or filtering by CLI arguments.
- Identified that a user could request one training window while silently training on a different existing modelling artifact.

### What changed
- Added `validate_modeling_year_range()` to `src/roadrisk/models/train.py`.
- The validation checks both `models/registry/feature_schema.json` and `data/processed/model_collision_severity.parquet`.
- Updated `scripts/06_train_model.py` to fail loudly when requested years do not match the existing modelling artifacts.
- Added a unit test covering matching and mismatched modelling windows.

### Why it improves the project
- Prevents silent model training on the wrong year range.
- Makes the CLI behavior match the data-first pipeline: the modelling table must be built for the requested window before training.
- Improves reproducibility and reduces the risk of misleading model reports.

### Files changed
- `src/roadrisk/models/train.py`
- `scripts/06_train_model.py`
- `tests/test_model_training.py`
- `models/registry/model_metrics.json`
- `models/registry/severity_model.joblib`
- `models/registry/severity_model_mvp.joblib`
- `models/registry/feature_importance.parquet`
- `models/registry/calibration_data.parquet`
- `data/app/model_metrics.json`
- `data/app/monitoring_summary.json`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 17 tests
- `python scripts/06_train_model.py --start-year 2020 --end-year 2024 --model-name severity_model_mvp` - passed
- `python scripts/07_build_app_artifacts.py --mode mvp --start-year 2020 --end-year 2024 --include-monitoring` - passed

### Remaining issues
- The training script still does not build the modelling table itself; users must run the feature-building step first.
- A future pipeline runner could enforce the full ordered sequence automatically.

## 2026-06-19 - Cycle 10: Borough-Level Model Performance Slices

### What was inspected
- Reviewed `data/processed/model_collision_severity.parquet` and confirmed verified London borough ONS codes are present in `local_authority_ons_district`.
- Confirmed the 2024 chronological test year has both KSI and non-KSI records for all 33 London boroughs.
- Reviewed `models/registry/model_metrics.json` and found vulnerable-user slices existed but borough-level model performance was still absent.

### What changed
- Added borough-level slice metric generation in `src/roadrisk/models/train.py`.
- The model metrics now include validation/test borough slices with records, KSI count, KSI rate, ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrix.
- Updated `pages/5_Model_Evaluation.py` with a Test-Year Borough Performance section.
- Expanded `tests/test_model_training.py` to require valid borough-slice metrics in saved model artifacts.
- Regenerated model artifacts and rebuilt app artifacts so `data/app/model_metrics.json` includes the borough slices.

### Why it improves the project
- Strengthens the model evaluation story by showing geographic performance differences, not only aggregate scores.
- Uses verified ONS borough codes already present in the modelling table.
- Helps identify boroughs where model behavior may be weaker or stronger before using the ranking for decision support.

### Files changed
- `src/roadrisk/models/train.py`
- `pages/5_Model_Evaluation.py`
- `tests/test_model_training.py`
- `models/registry/model_metrics.json`
- `models/registry/severity_model.joblib`
- `models/registry/severity_model_mvp.joblib`
- `models/registry/feature_importance.parquet`
- `models/registry/calibration_data.parquet`
- `data/app/model_metrics.json`
- `data/app/monitoring_summary.json`
- `IMPROVEMENT_LOG.md`

### Checks run
- `python scripts/06_train_model.py --start-year 2020 --end-year 2024 --model-name severity_model_mvp` - passed
- `python scripts/07_build_app_artifacts.py --mode mvp --start-year 2020 --end-year 2024 --include-monitoring` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 17 tests
- Streamlit render check at `http://localhost:8505/Model_Evaluation` - passed

### Remaining issues
- Borough slices are based on one test year, so small-borough estimates can be noisy.
- The page shows borough slice metrics but does not yet visualize them as a map or chart.

## 2026-06-19 - Cycle 11: Streamlit Boundary Tests

### What was inspected
- Reviewed all Streamlit entrypoints: `app.py` and `pages/*.py`.
- Confirmed the pages currently load prebuilt app/registry artifacts and do not directly train models or download raw data.
- Found there was no automated test protecting that architectural boundary.

### What changed
- Added `tests/test_streamlit_boundaries.py`.
- The new tests parse Streamlit entrypoints with `ast` and fail if pages import pipeline/training/download modules.
- The new tests also fail if pages call pipeline, training, raw CSV read, Parquet/CSV write, prediction, or network download functions.

### Why it improves the project
- Protects a hard project rule: no training, raw downloading, or raw processing inside Streamlit.
- Prevents future regressions as more pages are added.
- Reinforces the data-first design where Streamlit reads only prebuilt artifacts.

### Files changed
- `tests/test_streamlit_boundaries.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 19 tests

### Remaining issues
- The boundary test is static and conservative; it catches direct imports/calls but not every possible indirect side effect.
- Future page helpers should keep using `roadrisk.ui.*` loaders or similarly narrow read-only APIs.

## 2026-06-19 - Cycle 12: Borough Performance Visualization

### What was inspected
- Reviewed the new borough-level model performance slices added in Cycle 10.
- Found the Model Evaluation page exposed the borough metrics as a table only.
- Confirmed the existing metrics artifact contains enough data to visualize borough PR-AUC and KSI rate without recomputing model predictions in Streamlit.

### What changed
- Added a horizontal borough performance chart to `pages/5_Model_Evaluation.py`.
- The chart ranks boroughs by test-year PR-AUC and colors them by test-year KSI rate.
- The underlying table remains available below the chart.

### Why it improves the project
- Makes geographic model-performance differences easier to scan.
- Improves the data-science evaluation story while still using only prebuilt artifacts.
- Helps users see where the model may be more or less reliable before interpreting priority outputs.

### Files changed
- `pages/5_Model_Evaluation.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 19 tests
- Streamlit render check at `http://localhost:8505/Model_Evaluation` - passed

### Remaining issues
- The borough chart uses one test year only.
- It is a performance chart, not an intervention-priority map.

## 2026-06-19 - Cycle 13: README Reproducibility Update

### What was inspected
- Reviewed `README.md` and found the MVP app-artifact command did not include `--mode mvp` or `--include-monitoring`.
- Reviewed `src/roadrisk/ui/tables.py` and found the required app artifacts had expanded since the README was first written.
- Reviewed current pages and found the README did not describe the newer Model Evaluation and Model Monitoring content.
- Reviewed `scripts/08_run_all_checks.py` to confirm what validation command currently runs.

### What changed
- Updated `README.md` with the current app page list.
- Updated MVP and modelling-window commands to include the correct `scripts/07_build_app_artifacts.py` mode and monitoring flags.
- Added a note explaining that model training validates the existing modelling table and feature schema year range.
- Added the current required app/registry artifact list.
- Added a validation section documenting `python scripts/08_run_all_checks.py`.

### Why it improves the project
- Makes the project easier to reproduce from documented commands.
- Reduces the chance users run Streamlit before required artifacts exist.
- Aligns documentation with the current data-first app boundary and artifact gates.

### Files changed
- `README.md`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 19 tests

### Remaining issues
- `scripts/08_run_all_checks.py` does not yet verify artifact existence directly; Streamlit does that at app startup.
- Deployment-specific instructions are still minimal.

## 2026-06-19 - Cycle 14: Optional Artifact Readiness Check

### What was inspected
- Reviewed `scripts/08_run_all_checks.py` and found it only ran ruff and pytest.
- Reviewed `src/roadrisk/ui/tables.py` and confirmed `assert_app_ready()` is the central app artifact gate.
- Reviewed README validation instructions and found no documented command for checking artifacts before Streamlit launch.

### What changed
- Added `--include-artifacts` to `scripts/08_run_all_checks.py`.
- The optional check calls `assert_app_ready()` and verifies required `data/app/` and `models/registry/` artifacts are present and non-empty.
- Added `tests/test_run_all_checks.py` to verify the artifact check uses the app readiness gate.
- Updated `README.md` with the optional full validation command.

### Why it improves the project
- Lets users verify artifact readiness before launching Streamlit.
- Reuses the same artifact gate as the app, avoiding a duplicate list of required files.
- Improves reproducibility while keeping ordinary lint/test checks usable on fresh development runs.

### Files changed
- `scripts/08_run_all_checks.py`
- `tests/test_run_all_checks.py`
- `README.md`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 20 tests
- `python scripts/08_run_all_checks.py --include-artifacts` - passed

### Remaining issues
- Deployment-specific instructions are still minimal.
- Artifact readiness checks only verify existence/non-empty status; deeper artifact content checks live in pytest.

## 2026-06-19 - Cycle 15: App Metrics Artifact Content Validation

### What was inspected
- Reviewed `tests/test_model_training.py` and confirmed registry model metrics are checked for thresholds and performance slices.
- Reviewed `tests/test_app_artifacts.py` and found the copied app metrics artifact was not checked for the newer threshold, vulnerable-user slice, and borough slice fields.
- Confirmed `pages/5_Model_Evaluation.py` reads `data/app/model_metrics.json`, not the registry JSON directly.

### What changed
- Expanded `tests/test_app_artifacts.py` to validate `data/app/model_metrics.json`.
- The test now requires:
  - test threshold metrics,
  - test vulnerable-user slice metrics,
  - test borough slice metrics,
  - bounded thresholds and positive slice record counts.

### Why it improves the project
- Ensures the app-facing metrics artifact contains the richer evaluation content the page depends on.
- Catches failures where registry metrics are updated but the app artifact copy is stale or incomplete.
- Strengthens end-to-end artifact validation for the Model Evaluation page.

### Files changed
- `tests/test_app_artifacts.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 20 tests

### Remaining issues
- App artifact tests still run on a small synthetic dataset, so they validate structure and invariants rather than real-world metric quality.

## 2026-06-19 - Cycle 16: Collision Severity Factors Clarity Pass

### What was inspected
- Reviewed the existing pages from a user and teacher perspective, with no new pages added.
- Found that `pages/4_Collision_Severity_Factors.py` presented useful data, but the wording made the page harder to teach:
  - "severity signal" sounded model-like and vague,
  - "burden" was not immediately understandable,
  - "factor family" sounded technical,
  - the page needed a stronger reminder that these are severity patterns after reported collisions, not proof of collision causation.

### What changed
- Renamed controls and metrics on the Collision Severity Factors page to plainer language:
  - "Factor view" became "Compare".
  - "Factor family" became "Factor type".
  - "Severity per collision" became "Average harm per collision".
  - "Serious-harm burden" became "Total serious harm".
  - "Strongest severity signal" became "Highest average harm".
- Added a short interpretation note near the top explaining how to read the page.
- Kept the existing data, calculations, filters, and artifacts unchanged.

### Why it improves the project
- Makes the page easier to understand for someone evaluating the application quickly.
- Reduces the risk that users read condition categories as causal proof or exposure-adjusted risk.
- Keeps the data-science insight while removing avoidable jargon.

### Files changed
- `pages/4_Collision_Severity_Factors.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 20 tests
- Streamlit render check at `http://localhost:8505/Collision_Severity_Factors` - passed

### Remaining issues
- The page is still dense because it combines filters, summary metrics, ranking, composition, and a table.
- The serious-harm score is a heuristic severity summary, not an official DfT metric.
- There is still no exposure denominator, so the page should not be read as true collision risk by traffic volume.

## 2026-06-19 - Cycle 17: Model Monitoring Readability Pass

### What was inspected
- Reviewed the existing page set from a user and teacher perspective, without adding any pages.
- Found that `pages/6_Model_Monitoring.py` contained useful monitoring diagnostics, but the page did not clearly explain what each chart was meant to prove or warn about.
- The most confusing parts were the purpose of monitoring, the meaning of missingness, and how a reader should interpret class-balance changes over time.

### What changed
- Added a plain-language note at the top of the Model Monitoring page explaining why monitoring matters.
- Added question-style section headings before each diagnostic chart:
  - "Is the outcome stable over time?"
  - "Are there enough recent records?"
  - "Is the input data quality changing?"
- Added captions explaining what to look for in each chart.
- Renamed "Current Test-Year Metrics" to "Latest holdout-year performance".
- Added plain definitions for recall and precision near the metrics table.

### Why it improves the project
- Makes the monitoring page easier to present and defend in an evaluation.
- Turns the page from a set of technical diagnostics into a clearer argument about model reliability.
- Keeps the data-first boundary intact because the page still reads only prebuilt monitoring artifacts.

### Files changed
- `pages/6_Model_Monitoring.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `ruff check .` - passed
- `pytest -q` - passed, 20 tests
- Streamlit render check at `http://localhost:8505/Model_Monitoring` - passed

### Remaining issues
- The page still assumes the reader accepts the existing model; it does not compare alternative models.
- The monitoring artifacts are static and do not represent live production monitoring.
- Missingness is summarized as an average across model inputs, so it can hide feature-specific issues.

## 2026-06-19 - Cycle 18: Priority Score Input Validation

### What was inspected
- Reviewed `src/roadrisk/features/priority_score.py`, `src/roadrisk/data/build_app_tables.py`, `tests/test_priority_score.py`, and `tests/test_app_artifacts.py`.
- Scored candidate improvements:
  - Validate priority-score inputs before ranking: +3 correctness, +2 priority usefulness = 5.
  - Add more explanations to Model Evaluation: +2 user understanding = 2.
  - Polish the app overview table: +1 polish, +2 user understanding = 3.
- Chose priority-score input validation because impossible component values could otherwise produce plausible-looking priority rankings.

### What changed
- Added numeric/non-missing validation for priority-score inputs.
- Added explicit bounds checks:
  - `predicted_ksi_risk` must be between 0 and 1.
  - `vulnerable_user_share` must be between 0 and 1.
  - `record_count` must be positive.
  - `recent_ksi_count` must be non-negative and cannot exceed `record_count`.
- Added targeted tests for out-of-range probabilities, out-of-range vulnerable-user shares, negative KSI counts, KSI counts above record counts, and nonnumeric inputs.

### Why it improves the project
- Prevents impossible data from silently producing a decision-support ranking.
- Strengthens the intervention-priority story by making score inputs more trustworthy.
- Improves data integrity without changing the existing scoring formula or app pages.

### Files changed
- `src/roadrisk/features/priority_score.py`
- `tests/test_priority_score.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_priority_score.py -q` - passed, 9 tests
- `ruff check src/roadrisk/features/priority_score.py tests/test_priority_score.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 27 tests

### Remaining issues
- The priority score is still a heuristic, not a causal estimate.
- The score still does not include traffic exposure or road-network context.
- Component weights are fixed and should be described as decision-support choices rather than learned parameters.

## 2026-06-19 - Cycle 19: Severity Summary Reconciliation Guard

### What was inspected
- Audited KSI and severity definitions across `src/`, `pages/`, `tests/`, and `app.py`.
- Reviewed `src/roadrisk/features/build_collision_features.py`, `src/roadrisk/data/build_app_tables.py`, `src/roadrisk/data/clean_collisions.py`, and `tests/test_target.py`.
- Scored candidate improvements:
  - Enforce severity reconciliation in app summary artifacts: +3 correctness, +3 avoids misleading KSI/severity totals = 6.
  - Add more KSI wording to Methodology: +2 user understanding = 2.
  - Add a small Model Evaluation explainer: +2 user understanding = 2.
- Chose the severity reconciliation guard because it protects borough, severity-factor, and app summary outputs from quiet KSI/count inconsistencies.

### What changed
- Updated `_add_severity_flags()` in `src/roadrisk/data/build_app_tables.py` to use the central `is_ksi_from_severity()` decoder.
- Added a validation failure if app-summary input contains severity labels that cannot be mapped to Fatal, Serious, or Slight.
- Added tests proving that:
  - Fatal and Serious are KSI, Slight is non-KSI.
  - Fatal, Serious, and Slight bucket flags reconcile exactly to one severity bucket per row.
  - Unmapped severity labels fail loudly instead of being silently treated as non-KSI.

### Why it improves the project
- Avoids misleading totals in app artifacts if an unexpected severity label enters the processed data.
- Keeps KSI definitions consistent between modelling and app summaries.
- Strengthens data integrity before values are shown in charts, tables, maps, or borough summaries.

### Files changed
- `src/roadrisk/data/build_app_tables.py`
- `tests/test_app_artifacts.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_app_artifacts.py -q` - passed, 3 tests
- `ruff check src/roadrisk/data/build_app_tables.py tests/test_app_artifacts.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 29 tests

### Remaining issues
- Some KSI labels still appear directly in page text; the source of truth is in data code and tests, not a shared UI glossary.
- Historical trend processing uses its own aggregation path, although it also decodes to Fatal, Serious, and Slight before aggregating.
- The app still reports severity outcomes for reported injury collisions only, not all road risk or near misses.

## 2026-06-19 - Cycle 20: Model Metric Input Validation

### What was inspected
- Reviewed `src/roadrisk/models/train.py`, `src/roadrisk/models/evaluate.py`, `src/roadrisk/models/predict.py`, `pages/5_Model_Evaluation.py`, and model-related tests.
- Confirmed real model artifacts exist under `models/registry/` and app-facing metrics exist under `data/app/model_metrics.json`.
- Scored candidate improvements:
  - Add explicit validation for metric inputs: +3 correctness, +3 model/evaluation value = 6.
  - Add prediction schema mismatch test: +3 correctness, +1 maintainability = 4.
  - Improve Model Evaluation captions: +2 user understanding = 2.
- Chose metric-input validation because all model cards, charts, and evaluation tables depend on these metrics being well-defined.

### What changed
- Added `_validated_binary_inputs()` in `src/roadrisk/models/evaluate.py`.
- `binary_metrics()`, `calibration_table()`, and `threshold_metrics_table()` now reject:
  - empty inputs,
  - one-class targets,
  - non-binary targets,
  - length mismatches,
  - non-finite probabilities,
  - probabilities outside `[0, 1]`.
- Added `tests/test_model_evaluate.py` to cover valid metric outputs and invalid input failures.

### Why it improves the project
- Prevents invalid model-evaluation artifacts from being generated silently.
- Makes the Model Evaluation page more trustworthy because metrics cannot be computed from malformed targets or probabilities.
- Provides direct test coverage for a core data-science calculation layer.

### Files changed
- `src/roadrisk/models/evaluate.py`
- `tests/test_model_evaluate.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_model_evaluate.py -q` - passed, 7 tests
- `ruff check src/roadrisk/models/evaluate.py tests/test_model_evaluate.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 36 tests

### Remaining issues
- The model still uses one main balanced logistic-regression approach; no alternative model comparison is shown in-app.
- Metric validation guards the calculation inputs, but it does not prove the model is substantively strong.
- Threshold choices are still fixed reporting thresholds, not optimized intervention-policy thresholds.

## 2026-06-19 - Cycle 21: Prediction-Time Probability Validation

### What was inspected
- Reviewed `src/roadrisk/models/predict.py`, `tests/test_prediction.py`, and the app artifact builder path that uses predicted KSI probabilities.
- Confirmed `predict_probabilities()` already checked for missing feature columns but did not validate the model's returned probabilities.
- Scored candidate improvements:
  - Validate prediction-time schema and probability outputs: +3 correctness, +3 model/evaluation integrity, +2 priority usefulness = 8.
  - Add only a schema-mismatch test: +3 correctness, +1 maintainability = 4.
  - Improve prediction wording in pages: +2 user understanding = 2.
- Chose prediction-time validation because predicted probabilities feed map summaries, grid risk, and priority scoring.

### What changed
- Added prediction output validation in `src/roadrisk/models/predict.py`.
- `predict_probabilities()` now fails if:
  - the model output length does not match the number of input rows,
  - any predicted probability is non-finite,
  - any predicted probability is outside `[0, 1]`.
- Added tests for:
  - real trained-model probability output,
  - schema mismatch when a required feature is missing,
  - invalid model outputs using a lightweight fake pipeline.

### Why it improves the project
- Prevents bad model outputs from silently entering app artifacts.
- Protects downstream priority scores, map summaries, and risk tables from impossible predicted-risk values.
- Strengthens the boundary between trained model artifacts and app-ready decision-support data.

### Files changed
- `src/roadrisk/models/predict.py`
- `tests/test_prediction.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_prediction.py -q` - passed, 5 tests
- `ruff check src/roadrisk/models/predict.py tests/test_prediction.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 40 tests

### Remaining issues
- The prediction function checks probability validity, but it does not assess whether the probabilities are well calibrated.
- Feature drift is monitored separately and remains static, not live.
- The app still presents predicted KSI risk as decision support, not a causal or exposure-adjusted risk estimate.

## 2026-06-19 - Cycle 22: Monitoring Summary Validation

### What was inspected
- Reviewed `src/roadrisk/models/monitor.py`, `pages/6_Model_Monitoring.py`, `tests/test_app_artifacts.py`, and `scripts/07_build_app_artifacts.py`.
- Confirmed the monitoring artifact contains class balance, missingness, split metadata, and test metrics, but the generator did not validate those values before writing JSON.
- Scored candidate improvements:
  - Validate monitoring summary rows and test metrics before writing JSON: +3 correctness, +3 model/evaluation value = 6.
  - Add only stronger app-artifact assertions for monitoring JSON: +3 correctness, +1 maintainability = 4.
  - Improve Monitoring page wording further: +2 user understanding = 2.
- Chose monitoring-summary validation because the Model Monitoring page depends directly on this artifact.

### What changed
- Added `_validate_monitoring_summary()` in `src/roadrisk/models/monitor.py`.
- The monitoring generator now fails before writing if:
  - required top-level keys are missing,
  - train/validation/test split years are missing,
  - class-balance rows are empty or have impossible records, KSI counts, or KSI rates,
  - missingness rows are empty or outside `[0, 1]`,
  - required test metrics are missing or outside `[0, 1]`.
- Added `tests/test_model_monitor.py` for valid monitoring summaries and invalid summary failures.

### Why it improves the project
- Prevents impossible monitoring values from reaching the app.
- Strengthens the data-science reliability story by validating the artifact behind the monitoring page.
- Keeps the Streamlit app data-first because validation happens during artifact generation, not at page runtime.

### Files changed
- `src/roadrisk/models/monitor.py`
- `tests/test_model_monitor.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_model_monitor.py -q` - passed, 8 tests
- `ruff check src/roadrisk/models/monitor.py tests/test_model_monitor.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 48 tests

### Remaining issues
- Monitoring remains static and artifact-based; it is not live production monitoring.
- Mean missingness can hide individual feature drift.
- The monitoring page reports current reliability checks, but it does not decide whether the model should be retired or retrained.

## 2026-06-19 - Cycle 23: App Artifact Readability Gate

### What was inspected
- Reviewed `src/roadrisk/ui/tables.py`, `tests/test_app_artifacts.py`, `tests/test_run_all_checks.py`, and `scripts/08_run_all_checks.py`.
- Confirmed the app readiness gate checked required artifact existence and file size, but did not prove Parquet/JSON artifacts were readable or non-empty after parsing.
- Found that `borough_boundaries.geojson` is used by the borough overview page but was not part of the required readiness list.
- Scored candidate improvements:
  - Make `assert_app_ready()` load every required Parquet/JSON artifact and reject empty tables/dicts: +3 correctness, +2 user understanding through clearer startup errors = 5.
  - Add more tests only around existing artifact creation: +1 maintainability = 1.
  - Improve README wording for artifact checks: +1 documentation = 1.

### What changed
- Added `borough_boundaries.geojson` to the required app artifact list.
- Updated `assert_app_ready()` so it validates readable/non-empty Parquet, JSON, and GeoJSON artifacts, not just file size.
- Added `tests/test_app_readiness.py` to verify:
  - readable artifacts pass,
  - empty Parquet artifacts fail,
  - invalid JSON artifacts fail.

### Why it improves the project
- Prevents Streamlit from launching with corrupted or empty app-facing data files.
- Catches app artifact problems at startup with a clear validation error rather than later chart/table crashes.
- Strengthens the data-first contract: UI pages only render after app-ready artifacts are genuinely readable.

### Files changed
- `src/roadrisk/ui/tables.py`
- `tests/test_app_readiness.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_app_readiness.py -q` - passed, 3 tests
- `ruff check src/roadrisk/ui/tables.py tests/test_app_readiness.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 51 tests

### Remaining issues
- The readiness gate checks readability and non-empty content, but most schema-specific validation still lives in pipeline tests.
- The model `.joblib` artifact is checked for existence/non-empty status, not loaded during Streamlit startup.
- GeoJSON structure is currently checked only as readable JSON, not as a full geographic schema.

## 2026-06-19 - Cycle 24: Historical Trend Denominator Validation

### What was inspected
- Reviewed `src/roadrisk/data/build_historical_trends.py`, `tests/test_historical_trends.py`, and `pages/2_Historical_Trends.py`.
- Inspected current historical app artifacts:
  - `historical_trends_yearly.parquet`
  - `historical_severity_yearly.parquet`
  - `historical_road_user_severity_yearly.parquet`
- Confirmed the existing tests checked basic non-empty outputs and severity-count reconciliation, but did not validate road-user share denominators.
- Scored candidate improvements:
  - Validate historical severity totals and road-user share denominators before writing artifacts: +3 correctness, +3 avoids misleading trend interpretation = 6.
  - Add only stronger tests around existing artifacts: +3 correctness, +1 maintainability = 4.
  - Add more explanatory captions to the historical page: +2 user understanding = 2.

### What changed
- Added `_validate_historical_yearly()` in `src/roadrisk/data/build_historical_trends.py`.
- Added `_validate_road_user_shares()` in `src/roadrisk/data/build_historical_trends.py`.
- Historical trend generation now fails if:
  - yearly severity columns are missing,
  - yearly totals are non-positive,
  - Fatal + Serious does not equal KSI,
  - Fatal + Serious + Slight does not equal total collisions,
  - KSI rates do not match counts,
  - road-user shares are outside `[0, 1]`,
  - road-user severity shares do not sum to 1 by group/year or all groups/year.
- Added focused tests for bad yearly totals and bad road-user share denominators.

### Why it improves the project
- Prevents the historical trend page from displaying misleading severity totals or percentage shares.
- Strengthens the most visible descriptive-data part of the project.
- Keeps validation in the preprocessing/artifact layer rather than inside Streamlit.

### Files changed
- `src/roadrisk/data/build_historical_trends.py`
- `tests/test_historical_trends.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_historical_trends.py -q` - passed, 3 tests
- `ruff check src/roadrisk/data/build_historical_trends.py tests/test_historical_trends.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 53 tests

### Remaining issues
- Historical processing still depends on the source schema exposing usable London filter fields.
- Road-user historical grouping is intentionally broad and may hide subcategory changes across eras.
- The historical page remains descriptive only; it does not explain causal reasons for long-run trend changes.

## 2026-06-19 - Cycle 25: Borough Summary Validation

### What was inspected
- Reviewed `pages/3_Borough_Overview.py`, `src/roadrisk/data/build_app_tables.py`, `src/roadrisk/data/borough_boundaries.py`, and borough-related tests.
- Confirmed the borough page depends on `borough_severity_yearly.parquet` for the chart, table, and map coloring.
- Found that borough summaries were created from verified ONS borough codes, but the artifact builder did not explicitly validate borough-code membership, severity reconciliation, KSI rates, or yearly London share totals before writing the parquet.
- Scored candidate improvements:
  - Validate borough summary counts, rates, shares, and London ONS codes before writing artifact: +3 correctness, +2 intervention/geographic usefulness = 5.
  - Validate GeoJSON feature schema beyond readable JSON: +3 correctness, +1 maintainability = 4.
  - Remove duplicated borough map helper from page: +1 maintainability, -2 superficial-only = -1.

### What changed
- Added `_validate_borough_summary()` in `src/roadrisk/data/build_app_tables.py`.
- `_build_borough_summary()` now validates the borough artifact before returning it.
- The validator checks:
  - required borough summary columns,
  - borough codes are known London ONS district codes,
  - total collisions are positive,
  - Fatal + Serious equals KSI,
  - Fatal + Serious + Slight equals total collisions,
  - KSI rates match counts,
  - KSI rates and London share columns are bounded in `[0, 1]`,
  - borough collision shares and KSI shares sum to 1 by year.
- Added tests for a valid borough summary and a bad yearly share total.

### Why it improves the project
- Protects the Borough Overview graph, table, and choropleth map from misleading borough aggregates.
- Strengthens the geographic decision-support layer without adding new UI.
- Keeps validation in artifact generation instead of Streamlit page code.

### Files changed
- `src/roadrisk/data/build_app_tables.py`
- `tests/test_app_artifacts.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_app_artifacts.py -q` - passed, 5 tests
- `ruff check src/roadrisk/data/build_app_tables.py tests/test_app_artifacts.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 55 tests

### Remaining issues
- The borough map still assumes the GeoJSON uses `LAD24CD` and `LAD24NM` properties.
- Borough shares are based on reported injury collisions only, not exposure or traffic volume.
- The page aggregates selected years in Streamlit, so page-level selected-year share calculations are not separately tested.

## 2026-06-19 - Cycle 26: Priority Ranking Output Validation

### What was inspected
- Reviewed `src/roadrisk/features/priority_score.py`, `tests/test_priority_score.py`, `app.py`, and the current `data/app/priority_locations.parquet` artifact.
- Confirmed priority inputs were already validated, but the scorer did not explicitly reject empty inputs or validate the final ranked output before returning it.
- Scored candidate improvements:
  - Add explicit priority-output validation and reject empty priority inputs: +3 correctness, +2 priority/actionability = 5.
  - Improve reason wording in the overview table: +2 user understanding = 2.
  - Add visual emphasis to top priorities: +1 polish, -2 superficial-only = -1.

### What changed
- Added `_validate_priority_output()` in `src/roadrisk/features/priority_score.py`.
- `compute_priority_scores()` now rejects empty inputs.
- The final priority output is validated for:
  - bounded predicted risk, observed rates, component scores, data confidence, and priority score,
  - descending `priority_score` sort order,
  - valid priority bands,
  - non-empty priority reasons and audit focus labels.
- Added tests for empty input rejection, sorted output, and priority-band threshold behavior.

### Why it improves the project
- Prevents malformed priority rankings from reaching app artifacts.
- Strengthens the intervention-priority layer, which is the most decision-support-oriented part of the app.
- Adds direct tests for the scoring output contract, not just the scoring input contract.

### Files changed
- `src/roadrisk/features/priority_score.py`
- `tests/test_priority_score.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_priority_score.py -q` - passed, 11 tests
- `ruff check src/roadrisk/features/priority_score.py tests/test_priority_score.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 57 tests

### Remaining issues
- Priority scores remain heuristic and do not include traffic exposure.
- Component weights are fixed by design rather than optimized against intervention outcomes.
- Priority reasons identify the largest normalized component, not a causal explanation.

## 2026-06-19 - Cycle 27: Feature Schema Contract Validation

### What was inspected
- Reviewed `src/roadrisk/features/build_collision_features.py`, `src/roadrisk/models/train.py`, `tests/test_features.py`, and `tests/test_model_training.py`.
- Confirmed the modelling table and feature schema existed, but schema validation was split across a few local checks and did not fully guard against duplicate features, leakage features, missing schema keys, or schema/table mismatch.
- Scored candidate improvements:
  - Add a reusable feature-schema validator and use it before writing/training: +3 correctness, +3 model/evaluation integrity = 6.
  - Add only more assertions to `test_features.py`: +3 correctness, +1 maintainability = 4.
  - Improve feature-schema text in Model Evaluation page: +2 user understanding = 2.

### What changed
- Added `validate_feature_schema()` in `src/roadrisk/features/build_collision_features.py`.
- The modelling-table builder validates the schema before writing `feature_schema.json`.
- `train_model()` now validates the feature schema against `model_collision_severity.parquet` before fitting.
- The validator checks:
  - required schema keys,
  - duplicate feature names,
  - `feature_columns` exactly matching numeric + categorical + boolean features,
  - leakage/forbidden feature columns,
  - expected `target` and `id_column`,
  - valid year-range shape,
  - required schema columns are present in the modelling table.
- Added tests for valid generated schemas, duplicate features, leakage features, and schema/table mismatch.

### Why it improves the project
- Prevents feature leakage or stale schema/data mismatch before model training.
- Makes the model-evaluation story more credible because training now depends on a validated schema contract.
- Reduces duplicated schema checks by centralizing them in one reusable validator.

### Files changed
- `src/roadrisk/features/build_collision_features.py`
- `src/roadrisk/models/train.py`
- `tests/test_features.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_features.py tests/test_model_training.py -q` - passed, 6 tests
- `ruff check src/roadrisk/features/build_collision_features.py src/roadrisk/models/train.py tests/test_features.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 60 tests

### Remaining issues
- Feature usefulness is still evaluated through downstream model metrics, not feature-level causal analysis.
- Optional features are included when present; there is no feature-selection search.
- The model remains a single interpretable baseline-style classifier rather than a full model comparison suite.

## 2026-06-19 - Cycle 28: Training Input Target Validation

### What was inspected
- Reviewed `src/roadrisk/models/train.py`, `tests/test_model_training.py`, `tests/conftest.py`, and the current `data/processed/model_collision_severity.parquet`.
- Confirmed the current real modelling table has binary `is_ksi`, unique `collision_id`, and years 2020-2024.
- Found that training validated schema/table alignment, but did not explicitly reject duplicate collision IDs, non-binary target values, nonnumeric years, or one-class chronological splits through one reusable training-input check.
- Scored candidate improvements:
  - Validate the modelling table target and IDs before training: +3 correctness, +3 model/evaluation integrity = 6.
  - Add only another training test for malformed `is_ksi`: +3 correctness, +1 maintainability = 4.
  - Add page wording about the target: +2 user understanding = 2.

### What changed
- Added `_validate_training_input()` in `src/roadrisk/models/train.py`.
- `train_model()` now validates the modelling table before fitting.
- The validator checks:
  - required `collision_id`, `is_ksi`, and `accident_year` columns,
  - unique collision IDs,
  - binary/non-missing `is_ksi` target values,
  - numeric/non-missing accident years,
  - observed years match the chronological split,
  - train, validation, and test splits are non-empty and contain both classes.
- Added tests for valid training input, non-binary target values, duplicate collision IDs, and one-class test splits.

### Why it improves the project
- Prevents malformed modelling tables from silently corrupting model training.
- Strengthens the chronological evaluation story by validating split contents before fitting.
- Adds a clearer training-input contract on top of the feature-schema contract.

### Files changed
- `src/roadrisk/models/train.py`
- `tests/test_model_training.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_model_training.py -q` - passed, 6 tests
- `ruff check src/roadrisk/models/train.py tests/test_model_training.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 64 tests

### Remaining issues
- The target remains collision-level severity, not collision occurrence risk.
- The model still assumes the latest chronological year is the holdout test period.
- Duplicate-row validation happens at training time; earlier processed-data checks still focus on linking and non-empty artifacts.

## 2026-06-19 - Cycle 29: Processed Table Link Validation

### What was inspected
- Reviewed `src/roadrisk/data/build_processed_data.py`, `src/roadrisk/data/clean_vehicles.py`, `src/roadrisk/data/clean_casualties.py`, and `tests/test_processed_data.py`.
- Confirmed vehicles and casualties are filtered to London collision IDs, but the processed-data builder did not explicitly validate post-filter links, duplicate collision IDs, or configured year ranges before writing parquet artifacts.
- Scored candidate improvements:
  - Add processed-data relationship validation before parquet writes: +3 correctness, +3 protects downstream modelling/features = 6.
  - Add only more assertions to `test_processed_data.py`: +3 correctness, +1 maintainability = 4.
  - Improve processed report wording: +1 documentation = 1.

### What changed
- Added `_validate_processed_links()` in `src/roadrisk/data/build_processed_data.py`.
- `build_processed_data()` now validates processed collisions, vehicles, and casualties before writing parquet files.
- The validator checks:
  - required linking columns,
  - unique collision IDs in processed collisions,
  - every vehicle/casualty collision ID exists in processed collisions,
  - numeric accident years where present,
  - collision, vehicle, and casualty years stay within the configured range.
- Added tests for valid processed links, unlinked vehicle rows, duplicate collision IDs, and out-of-range casualty years.

### Why it improves the project
- Prevents broken processed artifacts from feeding feature engineering, modelling, maps, and summaries.
- Strengthens the data-first pipeline at the raw-to-processed boundary.
- Makes linking assumptions explicit and tested instead of implicit in filtering code.

### Files changed
- `src/roadrisk/data/build_processed_data.py`
- `tests/test_processed_data.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_processed_data.py -q` - passed, 5 tests
- `ruff check src/roadrisk/data/build_processed_data.py tests/test_processed_data.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 68 tests

### Remaining issues
- Vehicle and casualty duplicate-row semantics are not deeply validated because source record identity can vary by table/year.
- The processed report remains a simple row-count report rather than a full data-quality dashboard.
- Coordinate and London-filter quality are validated elsewhere and could still be summarized more clearly.

## 2026-06-19 - Cycle 30: Vulnerable-User Summary Validation

### What was inspected
- Reviewed `src/roadrisk/data/build_app_tables.py`, `src/roadrisk/features/build_casualty_features.py`, `tests/test_features.py`, and the current `data/app/vulnerable_user_summary.parquet`.
- Confirmed the vulnerable-user summary is built from decoded casualty `road_user_group` values and collision severity labels.
- Found that the artifact builder required the summary to be non-empty, but did not explicitly validate group values, positive record counts, KSI count bounds, or KSI-rate reconciliation before writing.
- Scored candidate improvements:
  - Validate vulnerable-user summary counts/rates/groups before writing: +3 correctness, +2 decision-support usefulness = 5.
  - Add only tests around existing generated artifact: +3 correctness, +1 maintainability = 4.
  - Add explanatory captions to a page using the artifact: +2 understanding = 2.

### What changed
- Added `_validate_vulnerable_summary()` in `src/roadrisk/data/build_app_tables.py`.
- `build_app_artifacts()` now validates `vulnerable_user_summary.parquet` content before writing.
- The validator checks:
  - required columns,
  - road-user groups are known decoded casualty groups,
  - records are positive,
  - KSI counts are between 0 and records,
  - KSI rates match `ksi_count / records`,
  - KSI rates are bounded in `[0, 1]`.
- Added tests for a valid vulnerable-user summary, an unknown group, and a bad KSI rate.

### Why it improves the project
- Prevents misleading vulnerable-user summaries from reaching app artifacts.
- Strengthens the vulnerable-user evidence used to contextualize safety risk and priority scoring.
- Keeps validation in artifact generation rather than Streamlit page code.

### Files changed
- `src/roadrisk/data/build_app_tables.py`
- `tests/test_app_artifacts.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_app_artifacts.py -q` - passed, 8 tests
- `ruff check src/roadrisk/data/build_app_tables.py tests/test_app_artifacts.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 71 tests

### Remaining issues
- Vulnerable-user summaries count casualty records by group, not exposure-adjusted risk.
- The app currently uses broad road-user groups; subcategory detail is intentionally simplified.
- The summary is app-artifact validation, not a substitute for broader casualty data-quality reporting.

## 2026-06-19 - Cycle 31: Safety Map Artifact Validation

### What was inspected
- Reviewed `src/roadrisk/data/build_app_tables.py`, `pages/1_Safety_Map.py`, `src/roadrisk/ui/maps.py`, and the current safety-map artifacts:
  - `data/app/collision_points_sample.parquet`
  - `data/app/safety_map_yearly.parquet`
- Confirmed the safety map uses sampled point data for display and yearly summaries for metric cards.
- Found that the artifacts were required to be non-empty, but the builder did not explicitly validate point coordinates, predicted-risk bounds, binary KSI flags, yearly count bounds, or rate reconciliation before writing.
- Scored candidate improvements:
  - Validate point and yearly safety-map artifacts before writing: +3 correctness, +2 user-facing map reliability = 5.
  - Add only tests around the existing generated artifacts: +3 correctness, +1 maintainability = 4.
  - Improve map page empty-selection handling: +2 user understanding, +1 polish = 3.

### What changed
- Added `_validate_collision_points()` in `src/roadrisk/data/build_app_tables.py`.
- Added `_validate_safety_map_yearly()` in `src/roadrisk/data/build_app_tables.py`.
- `build_app_artifacts()` now validates the point sample and yearly safety-map summary before writing.
- The validators check:
  - required columns,
  - unique point collision IDs,
  - non-missing coordinates inside broad London bounds,
  - predicted KSI risk bounded in `[0, 1]`,
  - binary `is_ksi`,
  - positive yearly total collisions,
  - KSI and vulnerable-user counts bounded by total collisions,
  - KSI rates and vulnerable-user shares matching counts.
- Added tests for valid points, bad coordinates, valid yearly summary, and bad yearly KSI rate.

### Why it improves the project
- Prevents invalid point or yearly summary data from reaching the Safety Map page.
- Protects the first user-facing page from misleading counts, rates, or map locations.
- Keeps map validation in the artifact-generation layer rather than the Streamlit page.

### Files changed
- `src/roadrisk/data/build_app_tables.py`
- `tests/test_app_artifacts.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_app_artifacts.py -q` - passed, 12 tests
- `ruff check src/roadrisk/data/build_app_tables.py tests/test_app_artifacts.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 75 tests

### Remaining issues
- The point map remains a sampled layer for performance, not a complete collision layer.
- The broad London coordinate bounds catch obvious errors but do not prove every point is inside an exact borough polygon.
- The heatmap remains a visual density aid, not a statistical hotspot model.

## 2026-06-19 - Cycle 32: Severity Driver Artifact Validation

### What was inspected
- Reviewed `src/roadrisk/data/build_app_tables.py`, `pages/4_Collision_Severity_Factors.py`, `tests/test_app_artifacts.py`, and the current `data/app/severity_drivers_yearly.parquet`.
- Confirmed the Collision Severity Factors page ranks categories using KSI rate, KSI counts, total serious harm, and average harm per collision.
- Found that the artifact existed and basic tests checked severity totals, but the builder did not explicitly validate harm-score formulas or condition-year share denominators before writing.
- Scored candidate improvements:
  - Validate severity-driver artifact counts/rates/harm scores/share denominators before writing: +3 correctness, +2 user understanding by avoiding misleading rankings = 5.
  - Add only app-artifact assertions for the existing severity-driver output: +3 correctness, +1 maintainability = 4.
  - Improve terminology on the page again: +2 understanding = 2.

### What changed
- Added `_validate_severity_drivers()` in `src/roadrisk/data/build_app_tables.py`.
- `_build_severity_drivers()` now validates the severity-driver artifact before returning it.
- The validator checks:
  - required columns,
  - positive total collisions,
  - Fatal + Serious equals KSI,
  - Fatal + Serious + Slight equals total collisions,
  - KSI rates match counts,
  - serious-harm score matches `Fatal * 10 + Serious * 3`,
  - harm per collision matches the harm score divided by total collisions,
  - KSI rates and condition-year shares are bounded in `[0, 1]`,
  - condition-year shares sum to 1 for each year and condition.
- Added tests for valid severity drivers, bad harm scores, and bad condition-year shares.

### Why it improves the project
- Prevents misleading condition rankings from reaching the Collision Severity Factors page.
- Protects the page’s most interpretive metrics: KSI rate, total serious harm, and average harm per collision.
- Keeps validation in the artifact-generation layer rather than relying on Streamlit calculations.

### Files changed
- `src/roadrisk/data/build_app_tables.py`
- `tests/test_app_artifacts.py`
- `IMPROVEMENT_LOG.md`

### Checks run
- `pytest tests/test_app_artifacts.py -q` - passed, 15 tests
- `ruff check src/roadrisk/data/build_app_tables.py tests/test_app_artifacts.py` - passed
- `ruff check .` - passed
- `pytest -q` - passed, 78 tests

### Remaining issues
- The serious-harm score remains a project heuristic, not an official DfT measure.
- Condition categories overlap, so all-factor rows are for ranking and comparison rather than additive totals.
- The page does not include exposure denominators, so it should not be read as collision causation.
