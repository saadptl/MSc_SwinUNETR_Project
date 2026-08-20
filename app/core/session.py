import streamlit as st


SESSION_KEYS_TO_CLEAR = [
    "last_analysis",
    "xai_result",
    "xai_paths",
    "report_path",
    "report_summary",
]


def reset_analysis_session():
    """Clear only dashboard inference/report state."""
    for key in SESSION_KEYS_TO_CLEAR:
        st.session_state.pop(key, None)

    st.session_state["dashboard_reset_notice"] = True


def consume_reset_notice():
    """Return and clear the one-time reset notification."""
    value = st.session_state.pop(
        "dashboard_reset_notice",
        False,
    )
    return value
