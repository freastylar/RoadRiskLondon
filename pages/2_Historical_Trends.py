from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plotly.graph_objects as go
import streamlit as st

from roadrisk.ui.tables import assert_app_ready, load_parquet
from roadrisk.ui.theme import configure_page, show_limitations
from roadrisk.utils.validation import DataValidationError

configure_page("RoadRisk London - Historical Trends")
try:
    assert_app_ready()
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

yearly = load_parquet("historical_severity_yearly.parquet")
road_user = load_parquet("historical_road_user_severity_yearly.parquet")

st.header("Reported London Injury Collisions by Severity")
use_log_scale = st.toggle("Use log scale", value=False)

fig = go.Figure()
series = [
    ("total_collisions", "Total collisions", "#263238", 4),
    ("ksi_collisions", "KSI collisions", "#c62828", 4),
    ("fatal_collisions", "Fatal collisions", "#6a1b9a", 2),
    ("serious_collisions", "Serious collisions", "#ef6c00", 2),
    ("slight_collisions", "Slight collisions", "#1565c0", 2),
]
for column, label, color, width in series:
    fig.add_trace(
        go.Scatter(
            x=yearly["accident_year"],
            y=yearly[column],
            mode="lines",
            name=label,
            line={"color": color, "width": width},
            hovertemplate="%{x}<br>" + label + ": %{y:,}<extra></extra>",
        )
    )
fig.update_layout(
    xaxis_title="Year",
    xaxis_dtick=1,  # Forces the x-axis to show only whole years
    yaxis_title="Collision count",
    yaxis_type="log" if use_log_scale else "linear",
    hovermode="x unified",
    legend_title_text="Click to hide/show",
    margin={"l": 20, "r": 20, "t": 20, "b": 20},
)
st.plotly_chart(fig, width="stretch")

st.header("Casualties by Road-User Group and Severity")

group_labels = {
    "pedestrian": "Pedestrian",
    "cyclist": "Cyclist",
    "motorcyclist": "Motorcyclist",
    "car_occupant": "Car occupant",
    "bus_or_coach": "Bus or coach",
    "other": "Other",
}
available_groups = sorted(road_user["road_user_group"].dropna().unique().tolist())
default_groups = [
    group
    for group in ["pedestrian", "cyclist", "motorcyclist", "car_occupant"]
    if group in available_groups
]
selected_group_labels = st.multiselect(
    "Road-user groups",
    options=[group_labels.get(group, group.replace("_", " ").title()) for group in available_groups],
    default=[group_labels[group] for group in default_groups],
)
label_to_group = {
    group_labels.get(group, group.replace("_", " ").title()): group for group in available_groups
}
selected_groups = [label_to_group[label] for label in selected_group_labels]
metric = st.radio(
    "Metric",
    ["Count", "Share of group-year total", "Share of all road-user casualties that year"],
    horizontal=True,
)

group_palette = {
    "pedestrian": {"Fatal": "#67000d", "Serious": "#cb181d", "Slight": "#fcae91"},
    "cyclist": {"Fatal": "#00441b", "Serious": "#238b45", "Slight": "#a1d99b"},
    "motorcyclist": {"Fatal": "#7f2704", "Serious": "#d94801", "Slight": "#fdae6b"},
    "car_occupant": {"Fatal": "#08306b", "Serious": "#2171b5", "Slight": "#9ecae1"},
    "bus_or_coach": {"Fatal": "#004c6d", "Serious": "#2c7fb8", "Slight": "#a6bddb"},
    "other": {"Fatal": "#252525", "Serious": "#737373", "Slight": "#cccccc"},
}

group_fig = go.Figure()
filtered_road_user = road_user[road_user["road_user_group"].isin(selected_groups)]
y_column = {
    "Count": "casualty_count",
    "Share of group-year total": "share_of_group_year",
    "Share of all road-user casualties that year": "share_of_all_groups_year",
}[metric]
y_title = {
    "Count": "Casualty count",
    "Share of group-year total": "Share of group-year total",
    "Share of all road-user casualties that year": "Share of all road-user casualties that year",
}[metric]
hover_suffix = "" if metric == "Count" else "%"

for group in selected_groups:
    group_data = filtered_road_user[filtered_road_user["road_user_group"] == group]
    for severity in ["Fatal", "Serious", "Slight"]:
        trace_data = group_data[group_data["severity_label"] == severity]
        if trace_data.empty:
            continue
        y_values = trace_data[y_column] if metric == "Count" else trace_data[y_column] * 100
        group_fig.add_trace(
            go.Scatter(
                x=trace_data["accident_year"],
                y=y_values,
                mode="lines",
                name=f"{group_labels.get(group, group)} - {severity}",
                line={
                    "color": group_palette.get(group, group_palette["other"])[severity],
                    "width": 3 if severity != "Slight" else 2,
                },
                hovertemplate=(
                    "%{x}<br>"
                    + group_labels.get(group, group)
                    + " - "
                    + severity
                    + f": %{{y:,.2f}}{hover_suffix}<extra></extra>"
                ),
            )
        )

group_fig.update_layout(
    xaxis_title="Year",
    xaxis_dtick=1,
    yaxis_title=y_title,
    hovermode="x unified",
    legend_title_text="Group - severity",
    margin={"l": 20, "r": 20, "t": 20, "b": 20},
)
st.plotly_chart(group_fig, width="stretch")
if selected_groups:
    ranges = (
        filtered_road_user.groupby("road_user_group")["accident_year"]
        .agg(["min", "max"])
        .reset_index()
    )
    range_text = ", ".join(
        f"{group_labels.get(row['road_user_group'], row['road_user_group'])}: {int(row['min'])}-{int(row['max'])}"
        for _, row in ranges.iterrows()
    )
    st.caption(f"Loaded year range by selected group: {range_text}")
show_limitations()