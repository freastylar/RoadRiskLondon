from __future__ import annotations

import streamlit as st


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, page_icon="RoadRisk", layout="wide")


def show_limitations() -> None:
    st.caption(
        "Reported personal-injury collisions only. Model outputs are statistical decision-support signals, not causal recommendations."
    )
