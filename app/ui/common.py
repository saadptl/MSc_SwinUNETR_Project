from pathlib import Path
from typing import Any

import streamlit as st


PAGES = [
    "🏠 Dashboard",
    "🩻 MRI Analysis",
    "📊 Results",
    "🔬 Explainable AI",
    "📄 Report",
    "ℹ️ About Project",
    "⚙️ System & Deployment",
]

CLASS_ORDER = ["Normal/Mild", "Moderate", "Severe"]
CLASS_ICONS = {"Normal/Mild": "🟢", "Moderate": "🟠", "Severe": "🔴"}


def navigate(page: str) -> None:
    """Queue navigation for the next Streamlit rerun.

    Streamlit does not allow changing the value of a widget-backed
    session-state key after that widget has been instantiated. The
    sidebar radio uses the ``navigation`` key, so buttons must write to
    a separate pending key instead. ``main.py`` consumes that key
    before creating the radio widget on the next run.
    """
    if page in PAGES:
        st.session_state.pending_navigation = page
        st.rerun()


def session():
    return st.session_state.get("analysis_session")


def pct(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value <= 1.0:
        value *= 100.0
    return max(0.0, min(100.0, value))


def filename(value: Any, fallback="N/A") -> str:
    if not value:
        return fallback
    try:
        return Path(str(value)).name
    except Exception:
        return str(value)


def safe(value: Any, fallback="N/A") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def section(title: str, subtitle: str | None = None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def status_chip(label: str, ready: bool):
    if ready:
        st.success(f"✓ {label}")
    else:
        st.info(f"○ {label}")
