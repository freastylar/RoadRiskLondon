from __future__ import annotations

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import pydeck as pdk

from roadrisk.ui.theme import configure_page, show_limitations
from roadrisk.ui.tables import assert_app_ready, load_parquet
from roadrisk.utils.validation import DataValidationError

configure_page("RoadRisk London - Collision Typologies")

try:
    assert_app_ready()
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

summary_path = Path("data/app/typology_summary.json")
if not summary_path.exists():
    st.error("Required typology artifacts missing. Please run: `uv run scripts/09_train_typologies.py` first.")
    st.stop()

with open(summary_path, "r") as f:
    typology_summary = json.load(f)

df_points = load_parquet("typology_map_points.parquet")

st.title("Systemic Collision Typologies")
st.markdown("""
    Unsupervised K-Modes clustering partition of regional accident records. 
    Profile definitions and intervention targets are generated procedurally based on data constraints.
""")

st.sidebar.header("Profile Selection")
menu_options = {k: v["dynamic_title"] for k, v in typology_summary.items()}
selected_title = st.sidebar.selectbox("Select Target Profile", options=list(menu_options.values()))

cluster_id = [k for k, v in menu_options.items() if v == selected_title][0]
cluster_meta = typology_summary[cluster_id]
modes = cluster_meta['modes']

st.sidebar.header("Map Layer Constraints")
view_mode = st.sidebar.radio("Mapping View Mode", options=["All Incidents", "Prioritized Hotspots"])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Profile Volume", f"{cluster_meta['total_incidents']:,}")
with col2:
    st.metric("KSI Proportion", f"{cluster_meta['ksi_ratio']:.2f}%")
with col3:
    st.metric("Dominant Weather Condition", modes.get('weather_conditions', 'Various'))

st.subheader("🛠️ Targeted Infrastructure Interventions")
audit_recommendations = []

if cluster_id == "-1":
    audit_recommendations.append("- **Custom Direct Audit Required:** These incidents represent unique environmental or spatial outliers that deviate heavily from standard systemic profiles.")
else:
    if "dark" in modes.get('light_conditions', '').lower():
        audit_recommendations.append("- **Visibility/Lighting:** Schedule structural night audits, replace aging fixtures, or implement high-intensity LED systems.")
    speed_digits = ''.join(filter(str.isdigit, modes.get('speed_limit', '')))
    if speed_digits and int(speed_digits) >= 40:
        audit_recommendations.append("- **Speed Management:** Evaluate corridor-segregation barriers or deploy speed-enforcement cameras.")
    if "junction" in modes.get('junction_detail', '').lower() or "crossroads" in modes.get('junction_detail', '').lower():
        audit_recommendations.append("- **Intersection Safety:** Adjust signal timing offsets or upgrade conflict-zone line markings.")
    if any(x in modes.get('weather_conditions', '').lower() for x in ["rain", "wet", "damp"]):
        audit_recommendations.append("- **Surface Treatment:** Apply high-friction anti-skid resurfacing overlays at high-density coordinates.")

if not audit_recommendations:
    audit_recommendations.append("- **General Safety Review:** Implement standard local traffic calming and density mitigation measures.")

st.info("\n".join(audit_recommendations))

st.subheader("Statistical Profile Archetype")
modes_df = pd.DataFrame.from_dict(modes, orient='index', columns=['Dominant Context Metric'])
st.table(modes_df)

df_filtered = df_points[df_points['typology_cluster'] == int(cluster_id)].copy()

if view_mode == "Prioritized Hotspots":
    df_filtered = df_filtered[df_filtered['is_hotspot'] == True]
    min_accidents = st.sidebar.slider("Minimum incidents per hotspot location", min_value=1, max_value=20, value=3)
    df_filtered = df_filtered[df_filtered['weight'] >= min_accidents]
    st.subheader(f"Prioritized Hotspots for Profile: {selected_title}")
    st.caption(f"Showing aggregated risk locations containing a minimum of {min_accidents} recurring profile incidents.")
    get_radius_val = "weight * 25"
else:
    df_filtered = df_filtered[df_filtered['is_hotspot'] == False]
    st.subheader(f"All Incident Locations: {selected_title}")
    st.caption("Displaying unaggregated pinpoint coordinates matching this profile archetype.")
    get_radius_val = "50"

if cluster_id == "-1":
    df_filtered['color_r'] = 140
    df_filtered['color_g'] = 140
    df_filtered['color_b'] = 145
else:
    df_filtered['color_r'] = df_filtered['ksi_count'].apply(lambda x: 220 if x > 0 else 255)
    df_filtered['color_g'] = df_filtered['ksi_count'].apply(lambda x: 50 if x > 0 else 140)
    df_filtered['color_b'] = df_filtered['ksi_count'].apply(lambda x: 50 if x > 0 else 0)

st.pydeck_chart(pdk.Deck(
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    initial_view_state=pdk.ViewState(
        latitude=51.5074,
        longitude=-0.1278,
        zoom=10.5,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            df_filtered,
            get_position="[longitude, latitude]",
            get_fill_color="[color_r, color_g, color_b, 180]",
            get_radius=get_radius_val,
            pickable=True,
        ),
    ],
))

show_limitations()