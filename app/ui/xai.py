from pathlib import Path
import json

import streamlit as st

from ui.common import navigate, filename, safe, pct


def _session():
    s = st.session_state.get("analysis_session")
    return s if s and s.has_prediction else None


def _find_paths(result):
    outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
    return {k: str(v) for k, v in outputs.items() if v}


def _generate(s):
    if len(s.selected_files) != 3:
        raise RuntimeError("XAI requires exactly three adjacent DICOM slices from the current analysis.")

    # Existing read-only XAI service. It loads the existing checkpoint and
    # performs inference/Grad-CAM only; it does not train or modify weights.
    from xai_dashboard import explain_dicom_series

    result = explain_dicom_series(
        [str(p) for p in s.selected_files],
        target_class=s.predicted_class_id,
        output_name="dashboard_current_analysis",
    )
    s.xai_result = result
    s.xai_paths = _find_paths(result)
    s.gradcam_generated = bool(s.xai_paths)
    return result


def render_xai():
    s = _session()
    st.title("🔬 Explainable AI")
    st.caption("Grad-CAM explanation for the existing trained Swin Transformer model.")

    st.info("Grad-CAM is presented as a model-attention visualization. It is not a disease segmentation map and is not a clinically validated localization method.")

    if s is None:
        st.warning("Run an MRI analysis first. XAI uses the exact three slices selected during inference.")
        if st.button("🩻 Go to MRI Analysis", type="primary", use_container_width=True):
            navigate("🩻 MRI Analysis")
        return

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Prediction", safe(s.predicted_class))
    with c2: st.metric("Confidence", f"{pct(s.confidence):.2f}%")
    with c3: st.metric("Target Class", safe(s.predicted_class))

    st.markdown("---")
    st.markdown("### Generate Grad-CAM")

    if s.has_xai:
        st.success("Grad-CAM explanation is already available for this analysis.")
    else:
        st.info("Click the button to generate the explanation. This does not retrain the model.")

    if st.button("🔬 Generate XAI Explanation", type="primary", use_container_width=True, key="generate_xai"):
        with st.spinner("Generating Grad-CAM from the existing trained model..."):
            try:
                _generate(s)
                st.success("Grad-CAM generated successfully.")
                st.rerun()
            except Exception as exc:
                st.error("XAI generation failed. The model checkpoint was not modified.")
                st.exception(exc)

    if not s.has_xai:
        return

    st.markdown("---")
    st.markdown("### 🖼️ Grad-CAM Visualization")

    paths = s.xai_paths
    middle = paths.get("middle_slice")
    heatmap = paths.get("heatmap")
    overlay = paths.get("overlay")
    panel = paths.get("panel")

    cols = st.columns(3)
    for col, path, title in zip(
        cols,
        [middle, heatmap, overlay],
        ["MRI Middle Slice", "Grad-CAM Heatmap", "Model Attention Overlay"],
    ):
        with col:
            st.markdown(f"**{title}**")
            if path and Path(path).exists():
                st.image(path, use_container_width=True)
            else:
                st.warning("Image not found.")

    if panel and Path(panel).exists():
        st.markdown("### 📋 XAI Summary Panel")
        st.image(panel, use_container_width=True)

    xai = s.xai_result.get("xai", {}) if isinstance(s.xai_result, dict) else {}
    if xai:
        st.markdown("### 📐 XAI Statistics")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Feature Shape", str(xai.get("feature_shape", "N/A")))
        with c2: st.metric("CAM Mean", f"{float(xai.get('cam_mean', 0)):.4f}")
        with c3: st.metric("CAM Std", f"{float(xai.get('cam_std', 0)):.4f}")

    st.success("✓ XAI Ready")
    st.caption("Interpretation: brighter CAM regions indicate stronger model-attention contribution for the selected target class; they should not be interpreted as definitive disease localization.")

    st.markdown("---")
    if st.button("📄 Continue to Report", type="primary", use_container_width=True):
        navigate("📄 Report")
