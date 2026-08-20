import streamlit as st
from textwrap import dedent
from ui.common import navigate


def render_home():
    st.title("🩻 Lumbar Spine AI")
    st.caption("Automated lumbar spine MRI disease detection, severity classification, explainability, and research reporting.")

    st.markdown(
        dedent("""
        <div class="hero">
            <div class="eyebrow">MSC COMPUTER SCIENCE • DATA SCIENCE</div>
            <h1>Lumbar Spine AI</h1>
            <p>Professional research dashboard for DICOM MRI analysis using an existing trained Swin Transformer classifier.</p>
        </div>
        """),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("AI Model", "Swin Transformer")
    with c2: st.metric("Input", "3 DICOM slices")
    with c3: st.metric("Output", "3 severity classes")
    with c4: st.metric("XAI", "Grad-CAM")

    st.markdown("## Professional Analysis Workflow")
    cols = st.columns(6)
    steps = [
        ("01", "DICOM", "Upload MRI series"),
        ("02", "Viewer", "Review selected slices"),
        ("03", "Inference", "Classify severity"),
        ("04", "XAI", "Explain model output"),
        ("05", "Results", "Review probabilities"),
        ("06", "Report", "Generate PDF"),
    ]
    for col, (n, title, text) in zip(cols, steps):
        with col:
            st.markdown(f"**{n}**")
            st.subheader(title)
            st.caption(text)

    st.markdown("---")
    if st.button("🩻 Start MRI Analysis", type="primary", use_container_width=True):
        navigate("🩻 MRI Analysis")

    st.warning("Academic Research Prototype — Not for Clinical Diagnosis")
