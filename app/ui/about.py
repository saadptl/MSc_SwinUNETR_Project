import streamlit as st


def render_about():
    st.title("ℹ️ About Project")
    st.caption("Automated Lumbar Spine Disease Detection and Classification from MRI Images")

    st.markdown("### Project Objective")
    st.write("Develop a research-oriented AI workflow that accepts lumbar spine MRI DICOM data, constructs the trained three-slice model input, predicts severity, provides Grad-CAM explainability, and produces a professional research report.")

    st.markdown("### Technology Stack")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Deep Learning**")
        st.write("PyTorch • Swin Transformer")
    with c2:
        st.markdown("**Medical Imaging**")
        st.write("DICOM • pydicom • OpenCV")
    with c3:
        st.markdown("**Dashboard**")
        st.write("Streamlit • Grad-CAM • ReportLab")

    st.markdown("### Classification")
    st.write("The dashboard presents three severity classes: Normal/Mild, Moderate, and Severe.")

    st.markdown("### Research Scope")
    st.write("The dashboard is a project demonstration and research prototype. It is not a clinical diagnostic system.")
