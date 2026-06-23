from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from roadrisk.ui.tables import assert_app_ready
from roadrisk.ui.theme import configure_page, show_limitations
from roadrisk.utils.validation import DataValidationError

configure_page("RoadRisk London - Home")

st.markdown(
    """
    <style>
    div[data-testid="stContainer"] {
        background-color: #f4f8fb;
        border-left: 6px solid #1f77b4;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
    }
    .stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    assert_app_ready()
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

st.title("RoadRisk London")
st.subheader("Data-Driven Road Safety Analytics")

st.divider()

st.markdown(
    "**RoadRisk London** is an analytical suite designed to provide deep insights into road safety across Greater London. "
    "By moving beyond basic collision counts, the project integrates exposure metrics, environmental conditions, and geospatial "
    "network data to identify systemic collision patterns and assess historical infrastructure risk. The goal is to provide "
    "decision-makers with actionable intelligence for safety interventions."
)

st.divider()

st.header("Underlying Data Sources")
with st.container(border = True):
    st.markdown(
        "- **Reported Collisions:** Official Department for Transport STATS19 data covering reported personal-injury road collisions, vehicles, and casualties.\n"
        "- **Traffic & Exposure:** Department for Transport raw traffic counts and Annual Average Daily Flow data used to construct hourly traffic profiles.\n"
        "- **Road Networks:** High-resolution drivable road segment data fetched from OpenStreetMap, enriched with infrastructure attributes.\n"
        "- **Administrative Boundaries:** Geographic perimeters of the 33 London Boroughs for regional performance tracking."
    )

st.header("Explore the Dashboard")

col1, col2 = st.columns(2)

with col1:
    with st.container(border = True):
        st.subheader("Safety Map")
        st.markdown("Geographic exploration of historical collision volumes and severity densities.")
        if st.button("Open Safety Map", use_container_width=True, key="btn_safety_map"):
            st.switch_page("pages/1_Safety_Map.py")
        
    with st.container(border = True):
        st.subheader("Historical Trends")
        st.markdown("Time-series analysis tracking safety performance and road-user specific casualties over time.")
        if st.button("Open Historical Trends", use_container_width=True, key="btn_hist_trends"):
            st.switch_page("pages/2_Historical_Trends.py")
        
    with st.container(border = True):
        st.subheader("Borough Overview")
        st.markdown("Administrative performance rankings and regional safety burden distributions.")
        if st.button("Open Borough Overview", use_container_width=True, key="btn_borough_overview"):
            st.switch_page("pages/3_Borough_Overview.py")

    with st.container(border = True):
        st.subheader("Collision Severity Factors")
        st.markdown("Analysis of environmental and infrastructural conditions that impact collision severity.")
        if st.button("Open Collision Severity Factors", use_container_width=True, key="btn_severity_factors"):
            st.switch_page("pages/4_Collision_Severity_Factors.py")

with col2:
    with st.container(border = True):
        st.subheader("Trip Risk")
        st.markdown("Exposure-adjusted relative risk mapping for specific travel modes, times, and weather conditions.")
        if st.button("Open Trip Risk", use_container_width=True, key="btn_trip_risk"):
            st.switch_page("pages/5_Trip_Risk.py")

    with st.container(border = True):
        st.subheader("Collision Typologies")
        st.markdown("Systemic incident profiling to identify and cluster intervention targets.")
        if st.button("Open Collision Typologies", use_container_width=True, key="btn_typologies"):
            st.switch_page("pages/6_Collision_Typologies.py")

    with st.container(border = True):
        st.subheader("Methodology")
        st.markdown("Detailed documentation of the analytical approach, pipeline architecture, and data limitations.")
        if st.button("Open Methodology", use_container_width=True, key="btn_methodology"):
            st.switch_page("pages/7_Methodology.py")

st.divider()
show_limitations()