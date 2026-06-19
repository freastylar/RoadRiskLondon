from __future__ import annotations

import json

import pandas as pd
import pytest

from roadrisk.ui import tables
from roadrisk.utils.validation import DataValidationError


def _write_ready_artifacts(tmp_path):
    parquet_path = tmp_path / "artifact.parquet"
    json_path = tmp_path / "artifact.json"
    geojson_path = tmp_path / "artifact.geojson"
    model_path = tmp_path / "artifact.joblib"
    pd.DataFrame({"value": [1]}).to_parquet(parquet_path, index=False)
    json_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature"}]}),
        encoding="utf-8",
    )
    model_path.write_text("placeholder", encoding="utf-8")
    return [parquet_path, json_path, geojson_path, model_path]


def test_app_ready_gate_accepts_readable_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "REQUIRED_APP_ARTIFACTS", _write_ready_artifacts(tmp_path))
    tables.assert_app_ready()


def test_app_ready_gate_rejects_empty_parquet_artifact(tmp_path, monkeypatch):
    artifacts = _write_ready_artifacts(tmp_path)
    empty_parquet = tmp_path / "empty.parquet"
    pd.DataFrame({"value": []}).to_parquet(empty_parquet, index=False)
    artifacts[0] = empty_parquet
    monkeypatch.setattr(tables, "REQUIRED_APP_ARTIFACTS", artifacts)
    with pytest.raises(DataValidationError, match="not readable or non-empty"):
        tables.assert_app_ready()


def test_app_ready_gate_rejects_invalid_json_artifact(tmp_path, monkeypatch):
    artifacts = _write_ready_artifacts(tmp_path)
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{not-json", encoding="utf-8")
    artifacts[1] = invalid_json
    monkeypatch.setattr(tables, "REQUIRED_APP_ARTIFACTS", artifacts)
    with pytest.raises(DataValidationError, match="not readable or non-empty"):
        tables.assert_app_ready()
