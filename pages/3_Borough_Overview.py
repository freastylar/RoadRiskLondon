from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plotly.express as px
import pydeck as pdk
import streamlit as st

from roadrisk.ui.tables import APP_DIR, assert_app_ready, load_parquet
from roadrisk.ui.theme import configure_page, show_limitations
from roadrisk.utils.validation import DataValidationError


def borough_choropleth_map(
    boundary_path: Path, borough_metrics, metric_column: str
) -> None:
    with boundary_path.open("r", encoding="utf-8") as handle:
        geojson = json.load(handle)
    values = borough_metrics.set_index("borough_code")[metric_column].to_dict()
    names = borough_metrics.set_index("borough_code")["borough"].to_dict()
    max_value = max(values.values()) if values else 0
    for feature in geojson["features"]:
        code = feature["properties"]["LAD24CD"]
        value = float(values.get(code, 0))
        normalized = value / max_value if max_value else 0
        feature["properties"]["metric_value"] = value
        feature["properties"]["borough"] = names.get(code, feature["properties"].get("LAD24NM", code))
        feature["properties"]["fill_color"] = [
            255,
            int(245 - 185 * normalized),
            int(235 - 197 * normalized),
            80 + int(140 * normalized),
        ]

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[55, 55, 55, 180],
        line_width_min_pixels=1,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=51.5074, longitude=-0.1278, zoom=9),
        tooltip={"text": "{borough}\nValue: {metric_value}"},
    )
    st.pydeck_chart(deck)


configure_page("RoadRisk London - Borough Overview")
try:
    assert_app_ready()
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

borough_yearly = load_parquet("borough_severity_yearly.parquet")

st.header("Borough Overview")

years = sorted(borough_yearly["accident_year"].dropna().astype(int).unique().tolist())
selected_years = st.multiselect("Years", years, default=years[-5:] if len(years) >= 5 else years)
metric_label = st.selectbox(
    "Rank by",
    [
        "KSI rate",
        "KSI collisions",
        "Total collisions",
        "Fatal collisions",
        "Serious collisions",
    ],
)
show_top_10 = st.toggle("Show top 10 only", value=False)

metric_columns = {
    "KSI rate": "ksi_rate",
    "KSI collisions": "ksi_collisions",
    "Total collisions": "total_collisions",
    "Fatal collisions": "fatal_collisions",
    "Serious collisions": "serious_collisions",
}
metric_column = metric_columns[metric_label]

filtered = (
    borough_yearly[borough_yearly["accident_year"].isin(selected_years)]
    if selected_years
    else borough_yearly
)
summary = (
    filtered.groupby(["borough_code", "borough"], dropna=False)
    .agg(
        total_collisions=("total_collisions", "sum"),
        fatal_collisions=("fatal_collisions", "sum"),
        serious_collisions=("serious_collisions", "sum"),
        slight_collisions=("slight_collisions", "sum"),
        ksi_collisions=("ksi_collisions", "sum"),
    )
    .reset_index()
)
summary["ksi_rate"] = summary["ksi_collisions"] / summary["total_collisions"]
summary["share_of_london_collisions"] = (
    summary["total_collisions"] / summary["total_collisions"].sum()
)
summary["share_of_london_ksi"] = summary["ksi_collisions"] / summary["ksi_collisions"].sum()
summary["year_range"] = (
    f"{min(selected_years)}-{max(selected_years)}" if selected_years else f"{min(years)}-{max(years)}"
)
summary = summary.sort_values(metric_column, ascending=False)
display_summary = summary.head(10) if show_top_10 else summary

fig = px.bar(
    display_summary,
    x="borough",
    y=metric_column,
    color="ksi_collisions",
    color_continuous_scale="Reds",
    labels={
        "borough": "Borough",
        metric_column: metric_label,
        "ksi_collisions": "KSI collisions",
    },
)
fig.update_layout(
    xaxis_tickangle=-45,
    margin={"l": 20, "r": 20, "t": 20, "b": 120},
)
st.plotly_chart(fig, width="stretch")

st.subheader("Map")
boundary_path = APP_DIR / "borough_boundaries.geojson"
if boundary_path.exists() and boundary_path.stat().st_size > 0:
    borough_choropleth_map(boundary_path, summary, metric_column)
else:
    st.warning(
        "Borough boundary file is missing. Rebuild app artifacts to enable the map: "
        "`python scripts/07_build_app_artifacts.py --mode mvp --start-year 2020 --end-year 2024 --include-monitoring`"
    )

table = summary[
    [
        "borough",
        "total_collisions",
        "fatal_collisions",
        "serious_collisions",
        "slight_collisions",
        "ksi_collisions",
        "ksi_rate",
        "share_of_london_collisions",
        "share_of_london_ksi",
        "year_range",
    ]
].copy()
table["ksi_rate"] = table["ksi_rate"].map(lambda value: f"{value:.1%}")
table["share_of_london_collisions"] = table["share_of_london_collisions"].map(
    lambda value: f"{value:.1%}"
)
table["share_of_london_ksi"] = table["share_of_london_ksi"].map(lambda value: f"{value:.1%}")
st.dataframe(table, width="stretch", hide_index=True)
show_limitations()
