from __future__ import annotations

import math

import pandas as pd

from roadrisk.utils.validation import DataValidationError


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    low = values.min()
    high = values.max()
    if high == low:
        return pd.Series(0.5, index=series.index)
    return (values - low) / (high - low)


def _priority_band(score: float) -> str:
    if score >= 0.75:
        return "Very high"
    if score >= 0.60:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Lower"


def _priority_reason(row: pd.Series) -> str:
    components = {
        "modelled KSI risk": row["risk_component"],
        "observed KSI count": row["density_component"],
        "observed KSI rate": row["observed_severity_component"],
        "vulnerable-user share": row["vulnerable_user_component"],
    }
    strongest = max(components, key=components.get)
    return f"Highest component: {strongest}"


def _audit_focus(row: pd.Series) -> str:
    if row["vulnerable_user_share"] >= 0.5:
        return "Vulnerable-user safety review"
    if row["recent_ksi_count"] >= 5 and row["observed_ksi_rate"] >= 0.2:
        return "KSI cluster review"
    if row["predicted_ksi_risk"] >= 0.6:
        return "Model-flagged severity review"
    return "Collision pattern review"


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise DataValidationError(f"Priority input {column} must be numeric and non-missing")
    return values


def _validate_priority_output(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataValidationError("Priority output must be non-empty")
    bounded_columns = [
        "predicted_ksi_risk",
        "vulnerable_user_share",
        "observed_ksi_rate",
        "risk_component",
        "density_component",
        "observed_severity_component",
        "vulnerable_user_component",
        "data_confidence",
        "priority_score",
    ]
    for column in bounded_columns:
        if not df[column].between(0, 1).all():
            raise DataValidationError(f"Priority output {column} must be between 0 and 1")
    if not df["priority_score"].is_monotonic_decreasing:
        raise DataValidationError("Priority output must be sorted by descending priority_score")
    valid_bands = {"Very high", "High", "Medium", "Lower"}
    invalid_bands = sorted(set(df["priority_band"]) - valid_bands)
    if invalid_bands:
        raise DataValidationError(f"Priority output has invalid bands: {invalid_bands}")
    if df["priority_reason"].astype(str).str.len().eq(0).any():
        raise DataValidationError("Priority output priority_reason must be non-empty")
    if df["audit_focus"].astype(str).str.len().eq(0).any():
        raise DataValidationError("Priority output audit_focus must be non-empty")


def compute_priority_scores(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "predicted_ksi_risk",
        "recent_ksi_count",
        "vulnerable_user_share",
        "record_count",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(f"Priority input missing columns: {missing}")
    if df.empty:
        raise DataValidationError("Priority input must be non-empty")
    result = df.copy()
    predicted_risk = _numeric_column(result, "predicted_ksi_risk")
    recent_ksi_count = _numeric_column(result, "recent_ksi_count")
    vulnerable_user_share = _numeric_column(result, "vulnerable_user_share")
    record_count = _numeric_column(result, "record_count")
    if (record_count <= 0).any():
        raise DataValidationError("Priority input record_count must be positive")
    if (~predicted_risk.between(0, 1)).any():
        raise DataValidationError("Priority input predicted_ksi_risk must be between 0 and 1")
    if (recent_ksi_count < 0).any():
        raise DataValidationError("Priority input recent_ksi_count must be non-negative")
    if (recent_ksi_count > record_count).any():
        raise DataValidationError("Priority input recent_ksi_count cannot exceed record_count")
    if (~vulnerable_user_share.between(0, 1)).any():
        raise DataValidationError("Priority input vulnerable_user_share must be between 0 and 1")
    result["predicted_ksi_risk"] = predicted_risk
    result["recent_ksi_count"] = recent_ksi_count
    result["vulnerable_user_share"] = vulnerable_user_share
    result["record_count"] = record_count
    result["observed_ksi_rate"] = result["recent_ksi_count"] / result["record_count"]
    result["risk_component"] = minmax(result["predicted_ksi_risk"])
    result["density_component"] = minmax(result["recent_ksi_count"])
    result["observed_severity_component"] = minmax(result["observed_ksi_rate"])
    result["vulnerable_user_component"] = minmax(result["vulnerable_user_share"])
    log_record_count = pd.to_numeric(result["record_count"], errors="coerce").map(math.log1p)
    result["data_confidence"] = minmax(log_record_count).clip(0.15, 1.0)
    base_score = (
        0.30 * result["risk_component"]
        + 0.30 * result["density_component"]
        + 0.25 * result["observed_severity_component"]
        + 0.15 * result["vulnerable_user_component"]
    )
    result["priority_score"] = (
        base_score * (0.50 + 0.50 * result["data_confidence"])
    ).clip(0, 1)
    result["priority_band"] = result["priority_score"].map(_priority_band)
    result["priority_reason"] = result.apply(_priority_reason, axis=1)
    result["audit_focus"] = result.apply(_audit_focus, axis=1)
    result = result.sort_values("priority_score", ascending=False)
    _validate_priority_output(result)
    return result
