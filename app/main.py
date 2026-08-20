import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import APP_NAME, APP_VERSION
from core.analysis_session import AnalysisSession
from ui.home import render_home
from ui.analysis import render_analysis
from ui.results import render_results
from ui.xai import render_xai
from ui.report import render_report
from ui.about import render_about
from ui.system import render_system
from ui.common import PAGES


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    path = APP_DIR / "assets" / "style.css"
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def initialize_state():
    # ``navigation`` is owned by the sidebar radio widget. It may be
    # initialized here, before the widget exists, but must never be
    # changed after the widget is created.
    if "navigation" not in st.session_state:
        st.session_state.navigation = PAGES[0]
    if "pending_navigation" not in st.session_state:
        st.session_state.pending_navigation = None
    if "analysis_session" not in st.session_state:
        st.session_state.analysis_session = AnalysisSession()

    defaults = {
        "analysis_result": None,
        "last_analysis": None,
        "selected_middle_index": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def consume_navigation():
    """Apply queued navigation before the sidebar radio is instantiated."""
    pending = st.session_state.get("pending_navigation")
    if pending in PAGES:
        st.session_state.navigation = pending
    st.session_state.pending_navigation = None


def render_sidebar():
    # Deliberately use native Streamlit components here. This prevents
    # raw HTML from ever appearing in the sidebar.
    st.sidebar.title("🩻 Lumbar Spine AI")
    st.sidebar.caption("Research & Explainability Dashboard")
    st.sidebar.divider()

    page = st.sidebar.radio("Navigation", PAGES, key="navigation")

    st.sidebar.divider()
    s = st.session_state.analysis_session

    st.sidebar.markdown("### System Status")
    st.sidebar.success("✓ Inference Complete" if s.has_prediction else "○ No Analysis")
    st.sidebar.success("✓ XAI Available" if s.has_xai else "○ XAI Pending")
    st.sidebar.success("✓ Report Available" if s.has_report else "○ Report Pending")

    st.sidebar.divider()
    st.sidebar.markdown("### Model")
    st.sidebar.write("**Architecture:** Swin Transformer")
    st.sidebar.write("**Classes:** 3")
    st.sidebar.write("**Runtime:** " + (s.device or "Not initialized"))
    st.sidebar.write(f"**Version:** {APP_VERSION}")
    st.sidebar.divider()
    st.sidebar.caption("Academic research prototype • Not for clinical diagnosis")
    return page


def render_page(page):
    routes = {
        "🏠 Dashboard": render_home,
        "🩻 MRI Analysis": render_analysis,
        "📊 Results": render_results,
        "🔬 Explainable AI": render_xai,
        "📄 Report": render_report,
        "ℹ️ About Project": render_about,
        "⚙️ System & Deployment": render_system,
    }
    routes.get(page, render_home)()


def main():
    load_css()
    initialize_state()
    # IMPORTANT: consume pending navigation before the radio widget is
    # instantiated. This avoids StreamlitAPIException.
    consume_navigation()
    page = render_sidebar()
    render_page(page)


if __name__ == "__main__":
    main()
