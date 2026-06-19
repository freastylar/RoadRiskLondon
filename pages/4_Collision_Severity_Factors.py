from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from roadrisk.ui.tables import assert_app_ready, load_parquet
from roadrisk.ui.theme import configure_page, show_limitations
from roadrisk.utils.validation import DataValidationError


def _aggregate_condition(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["condition", "condition_label", "category", "category_label"], dropna=False)
        .agg(
            total_collisions=("total_collisions", "sum"),
            fatal_collisions=("fatal_collisions", "sum"),
            serious_collisions=("serious_collisions", "sum"),
            slight_collisions=("slight_collisions", "sum"),
            ksi_collisions=("ksi_collisions", "sum"),
            serious_harm_score=("serious_harm_score", "sum"),
        )
        .reset_index()
    )
    summary["ksi_rate"] = summary["ksi_collisions"] / summary["total_collisions"]
    summary["harm_score_per_collision"] = summary["serious_harm_score"] / summary["total_collisions"]
    summary["category_display"] = summary["condition_label"] + ": " + summary["category_label"]
    total_collisions = summary["total_collisions"].sum()
    total_ksi = summary["ksi_collisions"].sum()
    summary["share_of_selected_collisions"] = summary["total_collisions"] / total_collisions
    summary["relative_ksi_rate"] = summary["ksi_rate"] / (total_ksi / total_collisions)
    return summary


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df[
        [
            "condition_label",
            "category_label",
            "total_collisions",
            "fatal_collisions",
            "serious_collisions",
            "slight_collisions",
            "ksi_collisions",
            "ksi_rate",
            "relative_ksi_rate",
            "serious_harm_score",
            "harm_score_per_collision",
        ]
    ].copy()
    table = table.rename(
        columns={
            "condition_label": "Factor",
            "category_label": "Category",
            "total_collisions": "Total collisions",
            "fatal_collisions": "Fatal",
            "serious_collisions": "Serious",
            "slight_collisions": "Slight",
            "ksi_collisions": "KSI",
            "ksi_rate": "KSI rate",
            "relative_ksi_rate": "Vs average KSI rate",
            "serious_harm_score": "Total serious-harm score",
            "harm_score_per_collision": "Average harm per collision",
        }
    )
    table["KSI rate"] = table["KSI rate"].map(lambda value: f"{value:.1%}")
    table["Vs average KSI rate"] = table["Vs average KSI rate"].map(lambda value: f"{value:.2f}x")
    table["Average harm per collision"] = table["Average harm per collision"].map(
        lambda value: f"{value:.2f}"
    )
    return table


configure_page("RoadRisk London - Collision Severity Factors")
try:
    assert_app_ready()
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

drivers = load_parquet("severity_drivers_yearly.parquet")

st.header("Collision Severity Factors")
st.caption(
    "This page describes reported London injury collisions. It highlights conditions linked with "
    "more serious outcomes after a collision has happened; it does not measure true exposure-adjusted "
    "collision risk."
)
st.info(
    "Read this as severity after a reported collision, not as proof that a condition causes collisions. "
    "Average harm per collision highlights categories where individual collisions tend to be more severe; "
    "total serious harm highlights categories where the overall KSI burden is largest."
)

years = sorted(drivers["accident_year"].dropna().astype(int).unique().tolist())
condition_options = (
    drivers[["condition", "condition_label"]]
    .drop_duplicates()
    .sort_values("condition_label")
    .set_index("condition_label")["condition"]
    .to_dict()
)

control_1, control_2, control_3, control_4 = st.columns([1.4, 1.25, 1, 1])
with control_1:
    selected_years = st.multiselect(
        "Years",
        years,
        default=years[-5:] if len(years) >= 5 else years,
    )
with control_2:
    factor_view = st.selectbox("Compare", ["All categories", "One factor type"])
with control_3:
    min_records = st.slider("Minimum collisions", min_value=1, max_value=2000, value=500, step=50)
with control_4:
    top_n = st.slider("Show top", min_value=5, max_value=25, value=12, step=1)

selected_condition_label = None
if factor_view == "One factor type":
    selected_condition_label = st.selectbox("Factor type", list(condition_options))

filter_1, filter_2 = st.columns(2)
with filter_1:
    include_unknown = st.toggle("Include unknown or missing values", value=False)
with filter_2:
    include_urban_rural = st.toggle("Include urban/rural in all-factor view", value=False)

rank_label = st.segmented_control(
    "Rank by",
    [
        "Average harm per collision",
        "KSI rate",
        "KSI collisions",
        "Total serious harm",
        "Fatal collisions",
        "Total collisions",
    ],
    default="Average harm per collision",
)
metric_columns = {
    "Average harm per collision": "harm_score_per_collision",
    "Total serious harm": "serious_harm_score",
    "KSI collisions": "ksi_collisions",
    "KSI rate": "ksi_rate",
    "Fatal collisions": "fatal_collisions",
    "Total collisions": "total_collisions",
}
metric_column = metric_columns[rank_label]

year_filtered = drivers[drivers["accident_year"].isin(selected_years)].copy() if selected_years else drivers.copy()
if selected_condition_label is not None:
    selected_condition = condition_options[selected_condition_label]
    year_filtered = year_filtered[year_filtered["condition"] == selected_condition].copy()
elif not include_urban_rural:
    year_filtered = year_filtered[year_filtered["condition"] != "urban_or_rural_area"].copy()

if not include_unknown:
    hidden_categories = {"Data missing or out of range", "Unknown", "Unallocated"}
    year_filtered = year_filtered[~year_filtered["category_label"].isin(hidden_categories)].copy()

summary = _aggregate_condition(year_filtered)
summary = summary[summary["total_collisions"] >= min_records].copy()

if summary.empty:
    st.warning("No categories match the selected filters.")
    st.stop()

top_by_severity = summary.sort_values("harm_score_per_collision", ascending=False).iloc[0]
top_by_burden = summary.sort_values("ksi_collisions", ascending=False).iloc[0]
top_by_rate = summary.sort_values("ksi_rate", ascending=False).iloc[0]
top_by_volume = summary.sort_values("total_collisions", ascending=False).iloc[0]
selected_ksi_rate = summary["ksi_collisions"].sum() / summary["total_collisions"].sum()

card_1, card_2, card_3, card_4 = st.columns(4)
card_1.metric("Highest average harm", str(top_by_severity["category_display"]))
card_2.metric("Most KSI collisions", str(top_by_burden["category_display"]), f"{int(top_by_burden['ksi_collisions']):,}")
card_3.metric("Highest KSI rate", str(top_by_rate["category_display"]), f"{top_by_rate['ksi_rate']:.1%}")
card_4.metric("Selected KSI rate", f"{selected_ksi_rate:.1%}")

current_top = summary.sort_values(metric_column, ascending=False).iloc[0]
st.info(
    f"{current_top['category_display']} ranks highest by {rank_label.lower()}: "
    f"{int(current_top['ksi_collisions']):,} KSI collisions from "
    f"{int(current_top['total_collisions']):,} reported injury collisions, "
    f"with a {current_top['ksi_rate']:.1%} KSI rate."
)

ranked = summary.sort_values(metric_column, ascending=False).head(top_n)
chart = ranked.sort_values(metric_column, ascending=True)
fig = px.bar(
    chart,
    x=metric_column,
    y="category_display",
    orientation="h",
    color="ksi_rate",
    color_continuous_scale="Reds",
    hover_data={
        "condition_label": True,
        "total_collisions": ":,",
        "fatal_collisions": ":,",
        "serious_collisions": ":,",
        "slight_collisions": ":,",
        "ksi_collisions": ":,",
        "ksi_rate": ":.1%",
        "relative_ksi_rate": ":.2f",
        "category_display": False,
    },
    labels={
        metric_column: rank_label,
        "category_display": "",
        "condition_label": "Factor",
        "total_collisions": "Total collisions",
        "ksi_rate": "KSI rate",
        "relative_ksi_rate": "Vs selected average",
    },
)
fig.update_layout(
    margin={"l": 20, "r": 20, "t": 20, "b": 20},
    coloraxis_colorbar={"title": "KSI rate"},
    yaxis_title=None,
)
st.plotly_chart(fig, width="stretch")

composition = ranked[
    [
        "category_display",
        "fatal_collisions",
        "serious_collisions",
        "slight_collisions",
        "total_collisions",
    ]
].copy()
for column in ["fatal_collisions", "serious_collisions", "slight_collisions"]:
    composition[column] = composition[column] / composition["total_collisions"]
composition = composition.melt(
    id_vars=["category_display"],
    value_vars=["fatal_collisions", "serious_collisions", "slight_collisions"],
    var_name="severity",
    value_name="share",
)
severity_labels = {
    "fatal_collisions": "Fatal",
    "serious_collisions": "Serious",
    "slight_collisions": "Slight",
}
composition["severity"] = composition["severity"].map(severity_labels)

composition_fig = px.bar(
    composition,
    x="category_display",
    y="share",
    color="severity",
    color_discrete_map={"Fatal": "#7f1d1d", "Serious": "#ef4444", "Slight": "#fecaca"},
    labels={
        "category_display": "",
        "share": "Share of collisions",
        "severity": "Severity",
    },
)
composition_fig.update_layout(
    barmode="stack",
    yaxis_tickformat=".0%",
    xaxis_tickangle=-35,
    margin={"l": 20, "r": 20, "t": 20, "b": 140},
)
st.plotly_chart(composition_fig, width="stretch")

st.dataframe(_format_table(ranked), width="stretch", hide_index=True)

if factor_view == "All categories":
    st.caption(
        "Categories from different factor types overlap. For example, the same collision can be "
        "counted once under speed limit and once under light conditions, so the rows are for ranking "
        "and comparison, not additive totals."
    )

show_limitations()
