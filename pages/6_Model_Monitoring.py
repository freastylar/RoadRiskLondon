from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from roadrisk.ui.tables import assert_app_ready, load_json
from roadrisk.ui.theme import configure_page, show_limitations
from roadrisk.utils.validation import DataValidationError


def _year_text(years: list[int]) -> str:
    return ", ".join(str(year) for year in years)


configure_page("RoadRisk London - Model Monitoring")
try:
    assert_app_ready()
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

monitoring = load_json("monitoring_summary.json")

st.header("Model Monitoring")
st.caption(
    "Static monitoring summary from the current modelling window. "
    "This page checks whether the target rate and missingness change over time; it does not retrain the model."
)
st.info(
    "Use this page to check whether the model is being judged on data that still looks like the data it was built on. "
    "Large changes in the KSI rate, record volume, or missingness would make the model results less trustworthy."
)

split = monitoring["split"]
card_1, card_2, card_3 = st.columns(3)
card_1.metric("Train years", _year_text(split["train_years"]))
card_2.metric("Validation years", _year_text(split["validation_years"]))
card_3.metric("Test years", _year_text(split["test_years"]))

class_balance = pd.DataFrame(monitoring["class_balance_by_year"])
missingness = pd.DataFrame(monitoring["missingness_by_year"])
latest = class_balance.sort_values("accident_year").iloc[-1]
first = class_balance.sort_values("accident_year").iloc[0]

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Latest KSI rate", f"{latest['ksi_rate']:.1%}")
metric_2.metric(
    "KSI-rate change",
    f"{latest['ksi_rate'] - first['ksi_rate']:+.1%}",
    f"{int(first['accident_year'])} to {int(latest['accident_year'])}",
)
metric_3.metric("Latest records", f"{int(latest['records']):,}")

st.subheader("Is the outcome stable over time?")
st.caption(
    "A mostly steady KSI rate means the model target is reasonably comparable across years. "
    "A sharp jump or drop would be a warning that reporting, traffic patterns, or the data extract may have changed."
)
balance_fig = px.line(
    class_balance,
    x="accident_year",
    y="ksi_rate",
    markers=True,
    labels={"accident_year": "Year", "ksi_rate": "KSI rate"},
)
balance_fig.update_layout(
    yaxis_tickformat=".0%",
    margin={"l": 20, "r": 20, "t": 20, "b": 20},
)
st.plotly_chart(balance_fig, width="stretch")

st.subheader("Are there enough recent records?")
st.caption(
    "The model is most useful when the latest year still has enough records and enough KSI cases. "
    "The KSI bars are expected to be much smaller because fatal and serious collisions are rarer than slight collisions."
)
records_fig = px.bar(
    class_balance,
    x="accident_year",
    y=["ksi_count", "records"],
    barmode="group",
    labels={"accident_year": "Year", "value": "Records", "variable": "Measure"},
)
records_fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
st.plotly_chart(records_fig, width="stretch")

st.subheader("Is the input data quality changing?")
st.caption(
    "Missingness is the average share of model inputs that are missing in each year. "
    "A rising line would mean the model is receiving less complete information than before."
)
missing_fig = px.line(
    missingness,
    x="accident_year",
    y="mean_missing_share",
    markers=True,
    labels={"accident_year": "Year", "mean_missing_share": "Mean missingness"},
)
missing_fig.update_layout(
    yaxis_tickformat=".2%",
    margin={"l": 20, "r": 20, "t": 20, "b": 20},
)
st.plotly_chart(missing_fig, width="stretch")

test_metrics = monitoring["test_metrics"]
st.subheader("Latest holdout-year performance")
st.caption(
    "These metrics come from the chronological test year only. "
    "Recall is the share of actual KSI collisions caught by the model; precision is the share of flagged collisions that were actually KSI."
)
metric_table = pd.DataFrame(
    [
        {
            "ROC-AUC": test_metrics["roc_auc"],
            "PR-AUC": test_metrics["pr_auc"],
            "Precision": test_metrics["precision"],
            "Recall": test_metrics["recall"],
            "F1": test_metrics["f1"],
            "Threshold": test_metrics["threshold"],
        }
    ]
)
st.dataframe(
    metric_table.style.format(
        {
            "ROC-AUC": "{:.3f}",
            "PR-AUC": "{:.3f}",
            "Precision": "{:.1%}",
            "Recall": "{:.1%}",
            "F1": "{:.3f}",
            "Threshold": "{:.2f}",
        }
    ),
    width="stretch",
    hide_index=True,
)

st.subheader("Known Limitations")
for limitation in monitoring.get("known_limitations", []):
    st.write(f"- {limitation}")
show_limitations()
