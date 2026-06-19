from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st


def point_map(df: pd.DataFrame, risk_column: str = "predicted_ksi_risk") -> None:
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_radius=35,
        get_fill_color=f"[255, 80, 40, 60 + {risk_column} * 190]",
        pickable=True,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10),
        tooltip={"text": "Risk: {" + risk_column + "}"},
    )
    st.pydeck_chart(deck)


def heatmap(df: pd.DataFrame, weight_column: str = "is_ksi") -> None:
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_weight=weight_column,
        radius_pixels=30,
        intensity=1.1,
        threshold=0.08,
        color_range=[
            [0, 0, 255, 45],      # blue outer glow
            [0, 255, 255, 90],    # cyan
            [0, 255, 80, 130],    # green
            [255, 255, 0, 170],   # yellow
            [255, 128, 0, 210],   # orange
            [255, 0, 0, 245],     # red core
        ],
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10),
    )
    st.pydeck_chart(deck)


def grid_map(df: pd.DataFrame, score_column: str = "priority_score") -> None:
    center_lat = float(df["grid_lat"].mean())
    center_lon = float(df["grid_lon"].mean())
    layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position="[grid_lon, grid_lat]",
        get_elevation=f"{score_column} * 1000",
        elevation_scale=1,
        radius=350,
        get_fill_color=f"[220, 55, 45, 80 + {score_column} * 175]",
        pickable=True,
        auto_highlight=True,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10, pitch=35),
        tooltip={"text": "Priority: {" + score_column + "}\nRecords: {record_count}"},
    )
    st.pydeck_chart(deck)


def borough_choropleth_map(boundary_path: Path, borough_metrics: pd.DataFrame, metric_column: str) -> None:
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
