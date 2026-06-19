from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from roadrisk.ui.tables import assert_app_ready, load_json, load_parquet
from roadrisk.ui.theme import configure_page, show_limitations
from roadrisk.utils.validation import DataValidationError

configure_page("RoadRisk London")

st.title("RoadRisk London")

try:
    assert_app_ready()
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

metrics = load_json("model_metrics.json")
priority = load_parquet("priority_locations.parquet")
trends = load_parquet("historical_trends_yearly.parquet")

st.subheader("Current Model")
col1, col2, col3 = st.columns(3)
col1.metric("Model", metrics.get("model_name", "severity_model"))
col2.metric("Test PR-AUC", round(metrics.get("test", {}).get("pr_auc", 0), 3))
col3.metric("Priority Areas", len(priority))

st.subheader("Latest Trend Year")
latest = trends.sort_values("accident_year").iloc[-1]
col4, col5, col6 = st.columns(3)
col4.metric("Year", int(latest["accident_year"]))
col5.metric("Collisions", int(latest["total_collisions"]))
col6.metric("KSI rate", f"{latest['ksi_rate']:.1%}")

st.subheader("Highest Priority Grid Cells")
st.caption(
    "Priority combines modelled KSI severity risk, recent observed KSI concentration, "
    "observed KSI rate, vulnerable-user share, and data confidence. It is a triage signal "
    "for further review, not proof that a condition caused collisions."
)
priority_table = priority.head(20)[
    [
        "grid_id",
        "priority_band",
        "priority_score",
        "priority_reason",
        "audit_focus",
        "record_count",
        "recent_ksi_count",
        "observed_ksi_rate",
        "predicted_ksi_risk",
        "vulnerable_user_share",
        "data_confidence",
    ]
].copy()
priority_table = priority_table.rename(
    columns={
        "grid_id": "Grid",
        "priority_band": "Band",
        "priority_score": "Priority",
        "priority_reason": "Reason",
        "audit_focus": "Suggested review",
        "record_count": "Records",
        "recent_ksi_count": "KSI",
        "observed_ksi_rate": "Observed KSI rate",
        "predicted_ksi_risk": "Modelled KSI risk",
        "vulnerable_user_share": "Vulnerable-user share",
        "data_confidence": "Data confidence",
    }
)
for column in ["Priority", "Observed KSI rate", "Modelled KSI risk", "Vulnerable-user share", "Data confidence"]:
    priority_table[column] = priority_table[column].map(lambda value: f"{value:.1%}")
st.dataframe(priority_table, width="stretch", hide_index=True)
show_limitations()
