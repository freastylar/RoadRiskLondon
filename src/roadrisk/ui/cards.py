from __future__ import annotations

import streamlit as st


def metric_row(metrics: list[tuple[str, str | int | float]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics, strict=True):
        column.metric(label, value)
