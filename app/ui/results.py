from pathlib import Path
import streamlit as st
from textwrap import dedent

from ui.common import CLASS_ORDER, CLASS_ICONS, navigate, filename, safe, pct, section


def _session():
    s = st.session_state.get("analysis_session")
    return s if s and s.has_prediction else None


def _prediction_card(s):
    prediction = safe(s.predicted_class, "Unknown")
    confidence = pct(s.confidence)
    icon = CLASS_ICONS.get(prediction, "🔵")

    st.markdown("### 🎯 Primary Prediction")
    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.markdown(
            dedent(f"""
            <div class="result-hero">
                <div class="eyebrow">MODEL PREDICTION</div>
                <div class="result-title">{icon} {prediction}</div>
                <div class="muted">Highest-probability severity category from the existing trained model.</div>
            </div>
            """),
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Confidence", f"{confidence:.2f}%")
        st.caption("Model probability for the selected class")


def _probabilities(s):
    st.markdown("### 📈 Classification Probabilities")
    probs = s.probabilities or {}
    cols = st.columns(3)
    for col, name in zip(cols, CLASS_ORDER):
        value = pct(probs.get(name, 0.0))
        with col:
            st.markdown(f"**{CLASS_ICONS[name]} {name}**")
            st.metric("Probability", f"{value:.2f}%")
            st.progress(value / 100.0)


def _mri_info(s):
    st.markdown("---")
    st.markdown("### 🩻 MRI Study Information")
    meta = s.series_metadata if isinstance(s.series_metadata, dict) else {}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Study ID**")
        st.code(safe(s.study_id, "Not available from upload"))
    with c2:
        st.markdown("**Series ID**")
        st.code(safe(s.series_id, "Not available from upload"))
    with c3:
        st.metric("DICOM Slices", meta.get("number_of_slices", "N/A"))

    c1, c2 = st.columns(2)
    with c1:
        st.info(f"First slice: **{filename(meta.get('first_file'))}**")
    with c2:
        st.info(f"Last slice: **{filename(meta.get('last_file'))}**")


def _model_info(s):
    st.markdown("---")
    st.markdown("### 🤖 Model & Inference Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model", safe(s.model_name, "Swin Transformer"))
    with c2:
        st.metric("Runtime", safe(s.device))
    with c3:
        st.metric("Input", safe(s.tensor_shape))
    st.caption("Existing trained checkpoint used for inference. No retraining or checkpoint modification occurs in the dashboard.")


def _input_slices(s):
    files = s.selected_files or []
    if not files:
        return
    st.markdown("---")
    st.markdown("### 🧠 Model Input Construction")
    st.caption("Three adjacent slices are used as Channel 0 / Channel 1 / Channel 2.")
    cols = st.columns(3)
    roles = [("Channel 0", "Previous"), ("Channel 1", "Middle"), ("Channel 2", "Next")]
    for col, path, (channel, role) in zip(cols, files, roles):
        with col:
            st.markdown(f"**{channel} — {role}**")
            st.code(filename(path))


def _actions(s):
    st.markdown("---")
    st.markdown("### 🔗 Continue Workflow")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🩻 Analyze Another MRI", use_container_width=True, key="results_another"):
            s.reset()
            st.session_state.last_analysis = None
            st.session_state.analysis_result = None
            navigate("🩻 MRI Analysis")
    with c2:
        if st.button("🔬 Open Explainable AI", type="primary", use_container_width=True, key="results_xai"):
            navigate("🔬 Explainable AI")
    with c3:
        if st.button("📄 Open Report Generator", use_container_width=True, key="results_report"):
            navigate("📄 Report")


def render_results():
    s = _session()
    st.title("📊 Analysis Results")
    st.caption("AI-assisted lumbar spine MRI classification using the existing trained model.")

    if s is None:
        st.info("No completed MRI analysis is available in the current dashboard session.")
        st.markdown("### Start an Analysis")
        st.write("Upload a DICOM series in MRI Analysis, run inference, then return here.")
        if st.button("🩻 Go to MRI Analysis", type="primary", use_container_width=True):
            navigate("🩻 MRI Analysis")
        return

    _prediction_card(s)
    st.markdown("---")
    _probabilities(s)
    _mri_info(s)
    _model_info(s)
    _input_slices(s)
    _actions(s)

    st.markdown("---")
    st.markdown("### 🧪 Research & Deployment Status")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.success("✓ Inference Complete")
    with c2: st.success("✓ XAI Ready") if s.has_xai else st.info("○ XAI Pending")
    with c3: st.success("✓ Report Ready") if s.has_report else st.info("○ Report Pending")
    with c4: st.success("✓ Model Available")

    st.markdown("---")
    st.warning("**Academic Research Prototype — Not for Clinical Diagnosis**\n\nPredictions and explainability visualizations are for academic research, software demonstration, and project evaluation only.")
