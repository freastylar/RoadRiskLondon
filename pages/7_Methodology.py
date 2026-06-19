from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from roadrisk.ui.tables import assert_app_ready
from roadrisk.ui.theme import configure_page
from roadrisk.utils.validation import DataValidationError

configure_page("RoadRisk London - Methodology")
try:
    assert_app_ready()
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

st.header("Methodology")
st.markdown(
    """
RoadRisk London uses official DfT road-safety open data for reported personal-injury road collisions, vehicles, and casualties.

The project separates two analytical products:

- Full-history descriptive trend analysis, using as much final historical data as feasible.
- Recent-window predictive modeling, using a configurable current window for KSI severity risk and intervention-priority ranking.

The KSI target is Fatal or Serious collision severity versus Slight. Evaluation uses chronological train, validation, and test splits, with PR-AUC, recall, F1, ROC-AUC, confusion matrix, and calibration artifacts.

The priority ranking combines predicted KSI risk, recent observed KSI density, vulnerable-user share, and data confidence. It is decision support for safety review, not causal proof and not an automatic intervention recommendation.

Known limitations include under-reporting, exclusion of near misses and private-road incidents, missing sensitive contributory factors, historical severity-reporting changes, and changing road environments across decades.
"""
)
