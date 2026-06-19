from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from roadrisk.utils.paths import registry_dir
from roadrisk.utils.validation import DataValidationError


def load_model(root: Path, model_name: str = "severity_model.joblib") -> dict:
    path = registry_dir(root) / model_name
    if not path.exists():
        raise DataValidationError(f"Missing model artifact: {path}")
    return joblib.load(path)


def predict_probabilities(root: Path, df: pd.DataFrame, model_name: str = "severity_model.joblib") -> pd.Series:
    artifact = load_model(root, model_name=model_name)
    schema = artifact["feature_schema"]
    missing = [column for column in schema["feature_columns"] if column not in df.columns]
    if missing:
        raise DataValidationError(f"Prediction input missing feature columns: {missing}")
    probabilities = artifact["pipeline"].predict_proba(df[schema["feature_columns"]])[:, 1]
    if len(probabilities) != len(df):
        raise DataValidationError("Prediction output length does not match input rows")
    if not np.isfinite(probabilities).all():
        raise DataValidationError("Prediction probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise DataValidationError("Prediction probabilities must be between 0 and 1")
    return pd.Series(probabilities, index=df.index, name="predicted_ksi_risk")
