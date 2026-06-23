from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from roadrisk.ui.cards import metric_row
from roadrisk.ui.maps import heatmap, point_map
from roadrisk.ui.tables import assert_app_ready, load_parquet
from roadrisk.ui.theme import configure_page, show_limitations
from roadrisk.utils.validation import DataValidationError

configure_page("RoadRisk London - Safety Map")
try:
    assert_app_ready()
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

st.header("Safety Map")

points = load_parquet("collision_points_sample.parquet")
yearly_summary = load_parquet("safety_map_yearly.parquet")
years = sorted(yearly_summary["accident_year"].dropna().astype(int).unique())
selected_years = st.multiselect("Year", years, default=years[:] if len(years) >= 3 else years)
filtered = points[points["accident_year"].isin(selected_years)] if selected_years else points
summary = (
    yearly_summary[yearly_summary["accident_year"].isin(selected_years)]
    if selected_years
    else yearly_summary
)
total_collisions = int(summary["total_collisions"].sum())
ksi_collisions = int(summary["ksi_collisions"].sum())
ksi_rate = ksi_collisions / total_collisions if total_collisions else 0

metric_row(
    [
        ("Collisions", total_collisions),
        ("KSI collisions", ksi_collisions),
        ("KSI rate", f"{ksi_rate:.1%}"),
    ]
)
st.caption(
    "Metric cards use all processed London collisions for the selected years. "
    "The point map uses a sampled point layer for performance."
)
map_style = st.segmented_control("Map style", ["Points", "Heatmap"], default="Points")
if map_style == "Heatmap":
    heatmap(filtered)
else:
    point_map(filtered)
show_limitations()
