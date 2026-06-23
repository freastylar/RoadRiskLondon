from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from roadrisk.ui.tables import assert_app_ready
from roadrisk.ui.theme import configure_page
from roadrisk.utils.validation import DataValidationError

configure_page("RoadRisk London - Methodology")

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
    </style>
    """,
    unsafe_allow_html=True
)

try:
    assert_app_ready()
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

st.title("Methodology & Architecture")
st.markdown("An in-depth technical overview of the data sources, analytical methodologies, and software architecture powering RoadRisk London.")

tab_data, tab_approach, tab_tech, tab_limitations = st.tabs([
    "General Info & Data Sources", 
    "Analytical Approach", 
    "Technologies & Pipeline", 
    "Known Limitations"
])

with tab_data:
    st.header("General Information & Data Sources")
    st.markdown("The platform integrates multiple high-volume datasets, transforming raw, disparate records into a unified analytical schema.")
    
    with st.container():
        st.subheader("1. STATS19 Reported Collisions (DfT)")
        st.markdown(
            "The foundational dataset is the Department for Transport's STATS19 database, which logs personal-injury road collisions reported to the police.\n"
            "- **Collision Data:** Contains geospatial coordinates, severity classifications (Slight, Serious, Fatal), environmental conditions (weather, light, road surface), and temporal data.\n"
            "- **Casualty & Vehicle Data:** Highly granular tables linked to individual collisions, allowing us to parse out specific road-user groups (pedestrians, cyclists, motorcyclists) and vehicle maneuvers."
        )
        
    with st.container():
        st.subheader("2. Traffic & Exposure Data (DfT)")
        st.markdown(
            "To move beyond absolute collision counts, we use DfT traffic flow data to calculate base exposure rates.\n"
            "- **AADF (Annual Average Daily Flow):** Provides baseline vehicular volume across major road links.\n"
            "- **Raw Hourly Counts:** Used to generate dynamic hourly flow profiles, enabling the Trip Risk assessment to understand how traffic volume changes by the hour, day type (weekday vs. weekend), and road classification."
        )
        
    with st.container():
        st.subheader("3. Geospatial Road Networks (OpenStreetMap)")
        st.markdown(
            "The physical road network is mapped using OpenStreetMap (OSM) data for Greater London.\n"
            "- **Network Graphs:** Extracts drivable nodes and edges, filtering out inaccessible paths.\n"
            "- **Infrastructure Tags:** Appends critical physical attributes to road segments, including speed limits, road hierarchy, lanes, street lighting, and cycleway infrastructure."
        )

    with st.container():
        st.subheader("4. Administrative Boundaries")
        st.markdown("GeoJSON definitions of the 33 London Boroughs, utilized for choropleth mapping and spatial aggregation in the Borough Overview.")

with tab_approach:
    st.header("Analytical Approach")
    st.markdown("RoadRisk London bridges the gap between raw historical reporting and actionable risk management.")
    
    with st.container():
        st.subheader("Full-History Descriptive Analysis")
        st.markdown(
            "Utilizes final historical data to evaluate temporal trends, geographical concentrations, and administrative performance across Greater London. This provides a macroscopic view of road safety burdens over time. The approach aggregates millions of individual records into yearly metrics, smoothing out short-term variance to identify macroscopic shifts in road safety."
        )
        
    with st.container():
        st.subheader("Exposure-Adjusted Trip Risk")
        st.markdown(
            "Absolute collision counts map where people drive, not necessarily where roads are dangerous. The Trip Risk algorithm calculates a Relative Risk Index.\n"
            "- **Mechanism:** Collisions are spatially snapped to OSM road segments using nearest-neighbor geospatial joins. DfT traffic volumes are then applied to these segments using matched road-class hierarchies.\n"
            "- **Output:** The risk score represents collisions per million miles driven under specific conditions, highlighting roads that are disproportionately dangerous relative to their traffic load."
        )

    with st.container():
        st.subheader("Systemic Priority Ranking & Typologies")
        st.markdown(
            "Grid cells and locations are ranked for infrastructure review using a composite score based on:\n"
            "1. **Recent KSI Count:** Acute, recent historical evidence of Killed or Seriously Injured collisions.\n"
            "2. **Vulnerable-User Share:** Prioritizing areas disproportionately dangerous to pedestrians and cyclists.\n"
            "3. **Data Confidence:** Ensuring low-volume statistical anomalies do not skew spatial interventions.\n\n"
            "Furthermore, unsupervised clustering algorithms group locations into distinct environmental and infrastructural profiles (typologies) to suggest systemic, rather than isolated, interventions."
        )

with tab_tech:
    st.header("Technologies & Pipeline Scripts")
    st.markdown("The project is built entirely in Python, utilizing modern data engineering and geospatial libraries.")
    
    with st.container():
        st.subheader("Core Technology Stack")
        st.markdown(
            "- **Frontend:** Streamlit for UI, interactivity, and dynamic routing.\n"
            "- **Data Processing:** Pandas, Numpy, and PyArrow for saving outputs as highly compressed Parquet files to ensure performant application loading.\n"
            "- **Geospatial:** GeoPandas, Shapely for spatial joins and buffer generation, Folium for interactive mapping, and OSMnx/NetworkX for fetching and parsing the OSM road graph."
        )

    with st.container():
        st.subheader("Data Pipeline Scripts (Backend)")
        st.markdown(
            "The backend relies on a strictly numbered execution pipeline to transform raw data into app-ready artifacts:\n"
            "- **01 - 04 (Ingestion & ETL):** Fetches STATS19 data, standardizes schemas across multiple years, handles missing or improperly encoded values, and builds the aggregated historical trend tables.\n"
            "- **05 - 08 (Feature Engineering):** Defines the feature engineering pipeline, executes chronological data splits, and builds the final Streamlit artifacts based on environmental attributes and localized severity drivers.\n"
            "- **09 - 10 (Clustering & Typologies):** Executes clustering on environmental attributes using frequency-based distance metrics appropriate for categorical data, and assigns typology profiles to geographical grid cells to segment systemic collision archetypes.\n"
            "- **12 - 15 (Spatial & Exposure Routing):** Downloads raw DfT traffic counts, builds hourly flow matrices, queries OSMnx for the London network, and performs complex spatial snapping using GeoPandas spatial indexing to bind collisions to specific road edges to calculate relative risk."
        )

with tab_limitations:
    st.header("Known Limitations")
    st.warning("All analytical outputs must be contextualized by the limitations of the data generation process.")
    
    with st.container():
        st.markdown(
            "- **Police Under-Reporting:** STATS19 only captures collisions reported to the police. Minor collisions, damage-only events, and certain active-travel falls are heavily under-represented.\n"
            "- **Lagging Indicator:** The dashboards rely on historical injuries to assess risk. The platform currently lacks telematics, hard-braking, or near-miss data which would serve as leading indicators.\n"
            "- **Temporal Environment Changes:** A collision from a previous decade occurred on a road layout that may have been entirely redesigned. Long-term historical data may not reflect current physical infrastructure.\n"
            "- **Reporting Standard Shifts:** Changes in how police forces record severity caused artificial spikes in Serious classifications, altering longitudinal KSI trends.\n"
            "- **Private Roads:** Incidents on private land, retail parks, or off the public highway network are excluded from the spatial mapping."
        )