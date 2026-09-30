"""
Lumbar Spine AI — Professional Medical Research Dashboard
Main entry point with redesigned sidebar and navigation.
"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Ensure APP_DIR is at the very beginning of sys.path
if str(APP_DIR) in sys.path:
    sys.path.remove(str(APP_DIR))
sys.path.insert(0, str(APP_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from app.config import APP_NAME, APP_VERSION
except ImportError:
    try:
        from config import APP_NAME, APP_VERSION
    except ImportError:
        APP_NAME    = "Lumbar Spine AI"
        APP_VERSION = "1.0.0"

from core.analysis_session import AnalysisSession
from ui.home      import render_home
from ui.analysis  import render_analysis
from ui.segmentation import render_segmentation
from ui.results   import render_results
from ui.xai       import render_xai
from ui.report    import render_report
from ui.about     import render_about
from ui.system    import render_system
from ui.common    import PAGES

# ── Paths ───────────────────────────────────────────────────────────────────
_SEG_ROOT = PROJECT_ROOT / "outputs" / "segmentation"
_OUTPUTS  = PROJECT_ROOT / "outputs"


st.set_page_config(
    page_title="Lumbar Spine AI — Research Dashboard",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    path = APP_DIR / "assets" / "style.css"
    if path.exists():
        st.markdown(
            f"<style>{path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    # Critical override: force Streamlit's internal CSS variables to light-mode
    # This prevents OS dark-mode from making text white-on-light-background
    st.markdown("""
    <style>
    /* Override Streamlit's internal CSS custom properties */
    :root, [data-theme="light"], [data-theme="dark"], [data-theme="auto"] {
        --text-color: #0F1B3D !important;
        --background-color: #F4F6FB !important;
        --secondary-background-color: #FFFFFF !important;
        --primary-color: #2563EB !important;
        color-scheme: light !important;
    }

    /* Streamlit uses these internal vars for text - override all of them */
    .stApp, .main, [data-testid="stAppViewContainer"] {
        --text-color: #0F1B3D !important;
        color: #0F1B3D !important;
        background-color: #F4F6FB !important;
    }

    /* Target Streamlit's generated class names for text */
    .st-emotion-cache-* { color: #0F1B3D !important; }

    /* Force all paragraph and header tags to be visible dark */
    .stApp p:not([style*="color"]),
    .stApp span:not([style*="color"]),
    .stApp div:not([style*="color"]):not([class*="sidebar"]) {
        color: #0F1B3D;
    }

    /* Info, warning, error text */
    [data-testid="stAlert"] p { color: #0F1B3D !important; }
    [data-testid="stAlert"] strong { color: #0F1B3D !important; }

    /* Ensure Streamlit success, info, warning text stays readable */
    .stSuccess p  { color: #065F46 !important; }
    .stInfo p     { color: #1E40AF !important; }
    .stWarning p  { color: #92400E !important; }
    .stError p    { color: #991B1B !important; }

    /* Top bar background */
    header[data-testid="stHeader"] {
        background-color: #0F1B3D !important;
    }

    /* Tab text visibility */
    [data-baseweb="tab"] p {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    [data-baseweb="tab"][aria-selected="true"] p {
        color: #1D4ED8 !important;
    }

    /* Selectbox / multiselect text */
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] div {
        color: #0F1B3D !important;
    }

    /* Number input, text input */
    input, textarea, select {
        color: #0F1B3D !important;
        background: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_state():
    if "navigation" not in st.session_state:
        st.session_state.navigation = PAGES[0]

    if "pending_navigation" not in st.session_state:
        st.session_state.pending_navigation = None

    if "analysis_session" not in st.session_state:
        st.session_state.analysis_session = AnalysisSession()
    else:
        # Patch any session instance created before new attributes were added
        sess = st.session_state.analysis_session
        if not hasattr(sess, "measurements") or sess.measurements is None:
            sess.measurements = []
        if not hasattr(sess, "detected_lesion_metrics") or sess.detected_lesion_metrics is None:
            sess.detected_lesion_metrics = {}
        if not hasattr(sess, "level_morphometrics") or sess.level_morphometrics is None:
            sess.level_morphometrics = {}

    defaults = {
        "analysis_result":       None,
        "last_analysis":         None,
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


def _check_xai_available(path_name: str) -> bool:
    return (_SEG_ROOT / path_name).exists()


def _check_classification_available() -> bool:
    return (PROJECT_ROOT / "models" / "best_model.pth").exists()


def _check_localization_available() -> bool:
    ckpt = (
        _SEG_ROOT
        / "rsna_part33_balanced_symmetry_refinement"
        / "checkpoints"
        / "part33_best_development_macro.pth"
    )
    return ckpt.exists()


def _check_report_available() -> bool:
    s = st.session_state.get("analysis_session")
    return bool(s and s.has_report)


def render_sidebar():
    # ── Branding ────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style="padding:0.8rem 0 1rem 0; border-bottom:1px solid rgba(255,255,255,0.12); margin-bottom:0.8rem;">
        <div style="font-size:1.15rem; font-weight:800; color:#FFFFFF !important; letter-spacing:-0.01em; -webkit-text-fill-color:#FFFFFF !important;">
            🧬 Lumbar Spine AI
        </div>
        <div style="font-size:0.7rem; color:#7C91C7 !important; -webkit-text-fill-color:#7C91C7 !important; margin-top:3px; font-weight:500;">
            Research &amp; Explainability Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation via buttons ────────────────────────────────────────────
    # Using buttons instead of radio to get reliable text rendering
    current_page = st.session_state.get("navigation", PAGES[0])

    nav_items = [
        ("🏠 Dashboard",          "🏠", "Dashboard"),
        ("🩻 MRI Analysis",       "🩻", "MRI Analysis"),
        ("🧠 3D Segmentation",    "🧠", "3D Segmentation"),
        ("📊 Results",            "📊", "Results"),
        ("🔬 Explainable AI",     "🔬", "Explainable AI"),
        ("📄 Report",             "📄", "Report"),
        ("ℹ️ About Project",      "ℹ️", "About Project"),
        ("⚙️ System & Deployment","⚙️", "System & Deployment"),
    ]

    selected_page = current_page
    for page_key, emoji, label in nav_items:
        is_active = (current_page == page_key)
        # Inject per-button active highlight via inline CSS keyed to the label
        if is_active:
            st.sidebar.markdown(
                f'<style>[data-testid="stSidebar"] [data-testid="baseButton-secondary"][kind="secondary"] p {{ color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; }}</style>',
                unsafe_allow_html=True
            )
        if st.sidebar.button(
            f"{emoji}  {label}",
            key=f"nav_{page_key}",
            width="stretch",
            type="primary" if is_active else "secondary",
        ):
            selected_page = page_key
            st.session_state.navigation = page_key
            st.rerun()

    st.session_state.navigation = selected_page

    st.sidebar.markdown(
        '<hr style="border:none; border-top:1px solid rgba(255,255,255,0.1); margin:0.8rem 0;"/>',
        unsafe_allow_html=True
    )

    # ── System Status ────────────────────────────────────────────────────
    cls_avail = _check_classification_available()
    loc_avail = _check_localization_available()
    xai_avail = any(
        _check_xai_available(d) for d in [
            "rsna_part45_3d_xai",
            "rsna_part46_point_targeted_3d_xai",
            "rsna_part46b_multi_point_3d_xai",
            "rsna_part46c_xai_summary",
            "rsna_part46d_xai_visual_inspection",
        ]
    )
    rpt_avail = _check_report_available()

    st.sidebar.markdown(
        '<div style="font-size:0.68rem; font-weight:700; color:#7C91C7 !important;'
        ' -webkit-text-fill-color:#7C91C7 !important; letter-spacing:0.1em;'
        ' text-transform:uppercase; margin-bottom:0.5rem;">System Status</div>',
        unsafe_allow_html=True
    )

    status_items = [
        ("Classification",  cls_avail),
        ("3D Localization", loc_avail),
        ("XAI Modules",     xai_avail),
        ("Report Ready",    rpt_avail),
    ]
    status_html = ""
    for label, avail in status_items:
        dot   = "#34D399" if avail else "#94A3B8"
        txt   = "Available" if avail else "Pending"
        status_html += (
            f'<div style="display:flex; align-items:center; gap:8px; margin:5px 0;'
            f' padding:6px 10px; background:rgba(255,255,255,0.05);'
            f' border-radius:7px;">'
            f'<div style="width:9px; height:9px; border-radius:50%;'
            f' background:{dot}; flex-shrink:0;"></div>'
            f'<span style="font-size:0.8rem; color:#CBD5E8 !important;'
            f' -webkit-text-fill-color:#CBD5E8 !important; font-weight:500;">'
            f'{label} <span style="color:{dot} !important;'
            f' -webkit-text-fill-color:{dot} !important;'
            f' font-size:0.72rem; font-weight:600;">· {txt}</span></span>'
            f'</div>'
        )
    st.sidebar.markdown(status_html, unsafe_allow_html=True)

    st.sidebar.markdown(
        '<hr style="border:none; border-top:1px solid rgba(255,255,255,0.1); margin:0.8rem 0;"/>',
        unsafe_allow_html=True
    )

    # ── Model Info ────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div style="font-size:0.68rem; font-weight:700; color:#7C91C7 !important;'
        ' -webkit-text-fill-color:#7C91C7 !important; letter-spacing:0.1em;'
        ' text-transform:uppercase; margin-bottom:0.4rem;">Architecture</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<div style="font-size:0.78rem; color:#A8B8D8 !important;'
        ' -webkit-text-fill-color:#A8B8D8 !important; line-height:1.7;">'
        '<span style="color:#E8EDF6 !important; -webkit-text-fill-color:#E8EDF6 !important; font-weight:600;">Classification:</span>'
        ' Swin Transformer<br/>'
        '<span style="color:#E8EDF6 !important; -webkit-text-fill-color:#E8EDF6 !important; font-weight:600;">Localization:</span>'
        ' 3D Swin-UNETR<br/>'
        '<span style="color:#E8EDF6 !important; -webkit-text-fill-color:#E8EDF6 !important; font-weight:600;">Conditions:</span>'
        ' SCS · LFNN · RFNN · LSS · RSS</div>',
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        '<div style="font-size:0.7rem; color:#6B7FA3 !important;'
        ' -webkit-text-fill-color:#6B7FA3 !important;'
        f' margin-top:1rem;">v{APP_VERSION} · MSc Major Project</div>',
        unsafe_allow_html=True
    )

    return selected_page


def render_page(page):
    import importlib
    if page == "🧠 3D Segmentation":
        import ui.segmentation
        importlib.reload(ui.segmentation)
        ui.segmentation.render_segmentation()
        return

    if page == "📄 Report":
        import ui.report
        importlib.reload(ui.report)
        ui.report.render_report()
        return

    if page == "📊 Results":
        import ui.results
        importlib.reload(ui.results)
        ui.results.render_results()
        return

    routes = {
        "🏠 Dashboard":         render_home,
        "🩻 MRI Analysis":      render_analysis,
        "🧠 3D Segmentation":   render_segmentation,
        "📊 Results":           render_results,
        "🔬 Explainable AI":    render_xai,
        "📄 Report":            render_report,
        "ℹ️ About Project":     render_about,
        "⚙️ System & Deployment": render_system,
    }
    routes.get(page, render_home)()


def main():
    load_css()
    initialize_state()
    consume_navigation()
    page = render_sidebar()
    render_page(page)


if __name__ == "__main__":
    main()
