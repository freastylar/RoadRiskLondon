from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from roadrisk.data.filter_london import LONDON_BOROUGH_BY_ONS
from roadrisk.features.build_collision_features import validate_feature_schema
from roadrisk.models.evaluate import binary_metrics, calibration_table, threshold_metrics_table
from roadrisk.utils.paths import processed_dir, registry_dir, reports_dir
from roadrisk.utils.validation import DataValidationError, read_required_parquet

VULNERABLE_SLICE_COLUMNS = {
    "Pedestrian involved": "has_pedestrian",
    "Cyclist involved": "has_cyclist",
    "Motorcyclist involved": "has_motorcyclist",
    "Child casualty involved": "has_child",
    "Older casualty involved": "has_elderly",
}


@dataclass(frozen=True)
class Split:
    train_years: list[int]
    validation_years: list[int]
    test_years: list[int]


def chronological_split(years: list[int]) -> Split:
    unique = sorted(set(years))
    if len(unique) < 3:
        raise DataValidationError("Need at least three years for chronological train/validation/test split")
    if len(unique) >= 5:
        return Split(train_years=unique[:-2], validation_years=[unique[-2]], test_years=[unique[-1]])
    return Split(train_years=unique[:-2], validation_years=[unique[-2]], test_years=[unique[-1]])


def _build_pipeline(schema: dict) -> Pipeline:
    numeric = [*schema["numeric_features"], *schema["boolean_features"]]
    categorical = schema["categorical_features"]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _feature_names(pipeline: Pipeline, schema: dict) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    names: list[str] = []
    names.extend([*schema["numeric_features"], *schema["boolean_features"]])
    if schema["categorical_features"]:
        onehot = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        names.extend(onehot.get_feature_names_out(schema["categorical_features"]).tolist())
    return names


def _slice_metrics(subset: pd.DataFrame, probabilities: pd.Series) -> list[dict]:
    slices: list[tuple[str, pd.Series]] = []
    available = [column for column in VULNERABLE_SLICE_COLUMNS.values() if column in subset.columns]
    if available:
        slices.append(("Any vulnerable road user involved", subset[available].fillna(0).astype(bool).any(axis=1)))
        slices.append(("No vulnerable road user flag", ~subset[available].fillna(0).astype(bool).any(axis=1)))
    for label, column in VULNERABLE_SLICE_COLUMNS.items():
        if column in subset.columns:
            slices.append((label, subset[column].fillna(0).astype(bool)))

    rows = []
    y_true = subset["is_ksi"].astype(int)
    for label, mask in slices:
        slice_y = y_true[mask]
        if slice_y.empty or slice_y.nunique() != 2:
            continue
        values = binary_metrics(slice_y, probabilities.loc[slice_y.index].to_numpy())
        rows.append(
            {
                "slice": label,
                "records": int(len(slice_y)),
                "ksi_count": int(slice_y.sum()),
                "ksi_rate": float(slice_y.mean()),
                **values,
            }
        )
    return rows


def _borough_slice_metrics(subset: pd.DataFrame, probabilities: pd.Series) -> list[dict]:
    if "local_authority_ons_district" not in subset.columns:
        return []
    rows = []
    y_true = subset["is_ksi"].astype(int)
    borough_codes = subset["local_authority_ons_district"].astype(str).str.upper()
    for borough_code, indices in borough_codes.groupby(borough_codes).groups.items():
        if borough_code not in LONDON_BOROUGH_BY_ONS:
            continue
        slice_y = y_true.loc[indices]
        if slice_y.empty or slice_y.nunique() != 2:
            continue
        values = binary_metrics(slice_y, probabilities.loc[slice_y.index].to_numpy())
        rows.append(
            {
                "borough_code": borough_code,
                "borough": LONDON_BOROUGH_BY_ONS[borough_code],
                "records": int(len(slice_y)),
                "ksi_count": int(slice_y.sum()),
                "ksi_rate": float(slice_y.mean()),
                **values,
            }
        )
    return sorted(rows, key=lambda row: row["records"], reverse=True)


def _validate_training_input(df: pd.DataFrame, split: Split) -> None:
    required = {"collision_id", "is_ksi", "accident_year"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Training input missing columns: {sorted(missing)}")
    if df["collision_id"].astype(str).duplicated().any():
        raise DataValidationError("Training input collision_id values must be unique")
    target = pd.to_numeric(df["is_ksi"], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise DataValidationError("Training target is_ksi must contain only binary 0/1 values")
    years = pd.to_numeric(df["accident_year"], errors="coerce")
    if years.isna().any():
        raise DataValidationError("Training input accident_year must be numeric and non-missing")
    expected_years = set(split.train_years + split.validation_years + split.test_years)
    observed_years = set(years.astype(int))
    if observed_years != expected_years:
        raise DataValidationError("Training input years must match chronological split years")
    for name, split_years in [
        ("train", split.train_years),
        ("validation", split.validation_years),
        ("test", split.test_years),
    ]:
        subset = target[years.astype(int).isin(split_years)]
        if subset.empty or subset.nunique() != 2:
            raise DataValidationError(f"{name} split must be non-empty and contain both classes")


def validate_modeling_year_range(root: Path, start_year: int, end_year: int) -> None:
    reg = registry_dir(root)
    schema_path = reg / "feature_schema.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    schema_range = schema.get("year_range")
    requested = [start_year, end_year]
    if schema_range != requested:
        raise DataValidationError(
            f"Requested training years {requested} do not match feature schema year_range {schema_range}. "
            "Rebuild the modelling table for the requested range before training."
        )

    df = read_required_parquet(processed_dir(root) / "model_collision_severity.parquet")
    observed_years = pd.to_numeric(df["accident_year"], errors="coerce").dropna().astype(int)
    observed_range = [int(observed_years.min()), int(observed_years.max())]
    if observed_range != requested:
        raise DataValidationError(
            f"Requested training years {requested} do not match modelling table years {observed_range}. "
            "Rebuild the modelling table for the requested range before training."
        )


def train_model(root: Path, model_name: str = "severity_model") -> dict:
    reg = registry_dir(root)
    schema_path = reg / "feature_schema.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    df = read_required_parquet(processed_dir(root) / "model_collision_severity.parquet")
    validate_feature_schema(schema, df)
    feature_columns = schema["feature_columns"]
    if "accident_year" not in df.columns:
        raise DataValidationError("Modeling table must include accident_year for chronological split")
    split = chronological_split(pd.to_numeric(df["accident_year"], errors="coerce").dropna().astype(int).tolist())
    _validate_training_input(df, split)
    train = df[df["accident_year"].isin(split.train_years)]
    validation = df[df["accident_year"].isin(split.validation_years)]
    test = df[df["accident_year"].isin(split.test_years)]

    x_train = train[feature_columns]
    y_train = train["is_ksi"].astype(int)
    pipeline = _build_pipeline(schema)
    pipeline.fit(x_train, y_train)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(x_train, y_train)

    metrics = {
        "model_name": model_name,
        "model_type": "logistic_regression_balanced",
        "split": {
            "train_years": split.train_years,
            "validation_years": split.validation_years,
            "test_years": split.test_years,
        },
        "class_balance": {
            "train_ksi_rate": float(y_train.mean()),
            "validation_ksi_rate": float(validation["is_ksi"].mean()),
            "test_ksi_rate": float(test["is_ksi"].mean()),
        },
    }
    for subset_name, subset in [("validation", validation), ("test", test)]:
        probs = pipeline.predict_proba(subset[feature_columns])[:, 1]
        metrics[subset_name] = binary_metrics(subset["is_ksi"].astype(int), probs)
        metrics[f"{subset_name}_thresholds"] = threshold_metrics_table(
            subset["is_ksi"].astype(int), probs
        )
        metrics[f"{subset_name}_slices"] = _slice_metrics(
            subset, pd.Series(probs, index=subset.index)
        )
        metrics[f"{subset_name}_borough_slices"] = _borough_slice_metrics(
            subset, pd.Series(probs, index=subset.index)
        )
        base_probs = dummy.predict_proba(subset[feature_columns])[:, 1]
        metrics[f"baseline_{subset_name}"] = binary_metrics(subset["is_ksi"].astype(int), base_probs)

    model_path = reg / f"{model_name}.joblib"
    joblib.dump({"pipeline": pipeline, "feature_schema": schema}, model_path)
    if model_name != "severity_model":
        joblib.dump({"pipeline": pipeline, "feature_schema": schema}, reg / "severity_model.joblib")

    metrics_path = reg / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    test_probs = pipeline.predict_proba(test[feature_columns])[:, 1]
    calibration = calibration_table(test["is_ksi"].astype(int), test_probs)
    calibration.to_parquet(reg / "calibration_data.parquet", index=False)

    model = pipeline.named_steps["model"]
    importance = pd.DataFrame(
        {"feature": _feature_names(pipeline, schema), "importance": model.coef_[0]}
    ).sort_values("importance", key=lambda s: s.abs(), ascending=False)
    importance.to_parquet(reg / "feature_importance.parquet", index=False)

    card = "\n".join(
        [
            "# Severity Model Card",
            "",
            f"- Model: {model_name}",
            "- Type: balanced logistic regression",
            f"- Train years: {split.train_years}",
            f"- Validation years: {split.validation_years}",
            f"- Test years: {split.test_years}",
            "- Target: KSI collision, Fatal or Serious versus Slight.",
            "- Interpretation: statistical association, not causal impact.",
            "",
        ]
    )
    (reg / "model_card.md").write_text(card, encoding="utf-8")
    (reports_dir(root) / "evaluation_report.md").write_text(card + json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
