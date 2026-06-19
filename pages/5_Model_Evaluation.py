from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from roadrisk.models.explain import add_feature_display_labels
from roadrisk.ui.tables import (
    assert_app_ready,
    load_json,
    load_registry_json,
    load_registry_parquet,
)
from roadrisk.ui.theme import configure_page, show_limitations
from roadrisk.utils.validation import DataValidationError


def _year_text(years: list[int]) -> str:
    return ", ".join(str(year) for year in years)


def _confusion_matrix_df(matrix: list[list[int]]) -> pd.DataFrame:
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual slight", "Actual KSI"],
        columns=["Predicted slight", "Predicted KSI"],
    )


def _metric_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for split in ["validation", "test"]:
        values = metrics[split]
        baseline = metrics[f"baseline_{split}"]
        rows.append(
            {
                "Split": split.title(),
                "ROC-AUC": values["roc_auc"],
                "PR-AUC": values["pr_auc"],
                "Baseline PR-AUC": baseline["pr_auc"],
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1": values["f1"],
            }
        )
    return pd.DataFrame(rows)


configure_page("RoadRisk London - Model Evaluation")
try:
    assert_app_ready()
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

metrics = load_json("model_metrics.json")
feature_schema = load_registry_json("feature_schema.json")
feature_importance = load_registry_parquet("feature_importance.parquet")
calibration = load_registry_parquet("calibration_data.parquet")

st.header("Model Evaluation")
st.caption(
    "Collision-level KSI severity model evaluation from prebuilt artifacts. "
    "The model estimates whether a reported injury collision is fatal or serious, not whether a collision will occur."
)

split = metrics["split"]
card_1, card_2, card_3, card_4 = st.columns(4)
card_1.metric("Model", metrics.get("model_type", "model"))
card_2.metric("Train years", _year_text(split["train_years"]))
card_3.metric("Validation years", _year_text(split["validation_years"]))
card_4.metric("Test years", _year_text(split["test_years"]))

test_metrics = metrics["test"]
baseline_test = metrics["baseline_test"]
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Test ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
metric_2.metric("Test PR-AUC", f"{test_metrics['pr_auc']:.3f}", f"baseline {baseline_test['pr_auc']:.3f}")
metric_3.metric("Test recall", f"{test_metrics['recall']:.1%}")
metric_4.metric("Test precision", f"{test_metrics['precision']:.1%}")

st.info(
    "The model is useful mainly as a ranking signal: recall is high, but precision is modest. "
    "That means it catches many KSI collisions while also flagging many collisions that are not KSI."
)

metrics_df = _metric_table(metrics)
st.dataframe(
    metrics_df.style.format(
        {
            "ROC-AUC": "{:.3f}",
            "PR-AUC": "{:.3f}",
            "Baseline PR-AUC": "{:.3f}",
            "Precision": "{:.1%}",
            "Recall": "{:.1%}",
            "F1": "{:.3f}",
        }
    ),
    width="stretch",
    hide_index=True,
)

left, right = st.columns(2)
with left:
    confusion = _confusion_matrix_df(test_metrics["confusion_matrix"])
    fig = px.imshow(
        confusion,
        text_auto=True,
        color_continuous_scale="Reds",
        labels={"x": "", "y": "", "color": "Collisions"},
    )
    fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
    st.plotly_chart(fig, width="stretch")

with right:
    calibration_fig = go.Figure()
    calibration_fig.add_trace(
        go.Scatter(
            x=calibration["mean_predicted_probability"],
            y=calibration["observed_fraction"],
            mode="lines+markers",
            name="Observed",
        )
    )
    calibration_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration")
    )
    calibration_fig.update_layout(
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed KSI fraction",
        yaxis_tickformat=".0%",
        xaxis_tickformat=".0%",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    st.plotly_chart(calibration_fig, width="stretch")

threshold_rows = metrics.get("test_thresholds", [])
if threshold_rows:
    st.subheader("Threshold Tradeoff")
    st.caption(
        "KSI is imbalanced, so the probability threshold changes the recall/precision tradeoff. "
        "Lower thresholds catch more KSI collisions but flag more non-KSI collisions too."
    )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_fig = px.line(
        threshold_table,
        x="threshold",
        y=["precision", "recall", "f1", "predicted_positive_rate"],
        markers=True,
        labels={"threshold": "Probability threshold", "value": "Rate", "variable": "Metric"},
    )
    threshold_fig.update_layout(
        yaxis_tickformat=".0%",
        xaxis_tickformat=".0%",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    st.plotly_chart(threshold_fig, width="stretch")
    st.table(
        threshold_table[
            ["threshold", "precision", "recall", "f1", "predicted_positive_rate"]
        ].style.format(
            {
                "threshold": "{:.0%}",
                "precision": "{:.1%}",
                "recall": "{:.1%}",
                "f1": "{:.3f}",
                "predicted_positive_rate": "{:.1%}",
            }
        )
    )

importance = add_feature_display_labels(feature_importance)
importance["direction"] = importance["importance"].map(lambda value: "Raises KSI probability" if value > 0 else "Lowers KSI probability")
top_importance = importance.reindex(importance["importance"].abs().sort_values(ascending=False).index).head(20)
importance_fig = px.bar(
    top_importance.sort_values("importance"),
    x="importance",
    y="feature_label",
    color="direction",
    orientation="h",
    hover_data={"feature": True, "feature_label": False, "importance": ":.3f"},
    labels={"importance": "Logistic-regression coefficient", "feature": ""},
    color_discrete_map={
        "Raises KSI probability": "#b91c1c",
        "Lowers KSI probability": "#2563eb",
    },
)
importance_fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
st.plotly_chart(importance_fig, width="stretch")

test_slices = metrics.get("test_slices", [])
if test_slices:
    st.subheader("Test-Year Performance Slices")
    st.caption(
        "Slices are evaluated on the chronological test year only. "
        "Rows are shown only when a slice contains both KSI and non-KSI collisions."
    )
    slice_table = pd.DataFrame(test_slices)[
        ["slice", "records", "ksi_count", "ksi_rate", "roc_auc", "pr_auc", "precision", "recall", "f1"]
    ].rename(
        columns={
            "slice": "Slice",
            "records": "Records",
            "ksi_count": "KSI",
            "ksi_rate": "KSI rate",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
        }
    )
    st.table(
        slice_table.style.format(
            {
                "KSI rate": "{:.1%}",
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}",
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "F1": "{:.3f}",
            }
        )
    )

borough_slices = metrics.get("test_borough_slices", [])
if borough_slices:
    st.subheader("Test-Year Borough Performance")
    st.caption(
        "Borough slices use verified London ONS borough codes. "
        "Rows are shown only when the borough has both KSI and non-KSI collisions in the test year."
    )
    borough_table = pd.DataFrame(borough_slices)[
        ["borough", "records", "ksi_count", "ksi_rate", "roc_auc", "pr_auc", "precision", "recall", "f1"]
    ].rename(
        columns={
            "borough": "Borough",
            "records": "Records",
            "ksi_count": "KSI",
            "ksi_rate": "KSI rate",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
        }
    )
    borough_fig = px.bar(
        borough_table.sort_values("PR-AUC", ascending=True),
        x="PR-AUC",
        y="Borough",
        orientation="h",
        color="KSI rate",
        color_continuous_scale="Reds",
        hover_data={
            "Records": ":,",
            "KSI": ":,",
            "KSI rate": ":.1%",
            "ROC-AUC": ":.3f",
            "PR-AUC": ":.3f",
            "Recall": ":.1%",
            "Precision": ":.1%",
        },
        labels={"PR-AUC": "Test PR-AUC", "KSI rate": "KSI rate"},
    )
    borough_fig.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        coloraxis_colorbar={"title": "KSI rate"},
    )
    st.plotly_chart(borough_fig, width="stretch")
    st.dataframe(
        borough_table.style.format(
            {
                "KSI rate": "{:.1%}",
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}",
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "F1": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

st.caption(
    f"Feature schema: {len(feature_schema['feature_columns'])} input features "
    f"({len(feature_schema['numeric_features'])} numeric, "
    f"{len(feature_schema['categorical_features'])} categorical, "
    f"{len(feature_schema['boolean_features'])} boolean). "
    "Categorical features are one-hot encoded before modelling."
)
show_limitations()
