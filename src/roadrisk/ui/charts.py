from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def yearly_ksi_chart(df: pd.DataFrame) -> None:
    fig = px.line(df, x="accident_year", y=["total_collisions", "ksi_count"], markers=True)
    st.plotly_chart(fig, width="stretch")


def ksi_rate_chart(df: pd.DataFrame, x: str, color: str | None = None) -> None:
    fig = px.line(df, x=x, y="ksi_rate", color=color, markers=True)
    st.plotly_chart(fig, width="stretch")


def bar_chart(df: pd.DataFrame, x: str, y: str, color: str | None = None) -> None:
    fig = px.bar(df, x=x, y=y, color=color)
    st.plotly_chart(fig, width="stretch")
