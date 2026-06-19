from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roadrisk.data.build_processed_data import build_processed_data
from roadrisk.features.build_collision_features import build_modeling_table
from roadrisk.models import predict as predict_module
from roadrisk.models.predict import predict_probabilities
from roadrisk.models.train import train_model
from roadrisk.utils.validation import DataValidationError


def test_predictions_are_probabilities(sample_config):
    build_processed_data(sample_config)
    df = build_modeling_table(sample_config)
    train_model(sample_config.project_root, model_name="severity_model_mvp")
    probs = predict_probabilities(sample_config.project_root, df)
    assert isinstance(probs, pd.Series)
    assert probs.between(0, 1).all()


def test_predictions_reject_schema_mismatch(sample_config):
    build_processed_data(sample_config)
    df = build_modeling_table(sample_config)
    train_model(sample_config.project_root, model_name="severity_model_mvp")
    with pytest.raises(DataValidationError, match="missing feature columns"):
        predict_probabilities(sample_config.project_root, df.drop(columns=["accident_year"]))


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.array([[0.1, 1.2], [0.4, 0.6]]), "between 0 and 1"),
        (np.array([[0.1, np.nan], [0.4, 0.6]]), "finite"),
        (np.array([[0.1, 0.9]]), "length"),
    ],
)
def test_predictions_reject_invalid_model_outputs(tmp_path, monkeypatch, probabilities, message):
    class FakePipeline:
        def predict_proba(self, _df):
            return probabilities

    monkeypatch.setattr(
        predict_module,
        "load_model",
        lambda _root, model_name="severity_model.joblib": {
            "feature_schema": {"feature_columns": ["feature"]},
            "pipeline": FakePipeline(),
        },
    )
    with pytest.raises(DataValidationError, match=message):
        predict_probabilities(tmp_path, pd.DataFrame({"feature": [1, 2]}))
