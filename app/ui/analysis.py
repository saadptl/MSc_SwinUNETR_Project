from pathlib import Path
import shutil

import streamlit as st

from inference.dashboard_inference import analyze_series
from ui.common import navigate, filename, safe, section


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "dashboard" / "uploads" / "current_series"


def _save_uploads(uploaded_files):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for item in UPLOAD_DIR.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    paths = []
    for item in uploaded_files:
        destination = UPLOAD_DIR / Path(item.name).name
        destination.write_bytes(item.getbuffer())
        paths.append(destination)
    return paths


def _extract_result(result):
    prediction = result.get("prediction", {}) or {}
    series = result.get("series", {}) or {}
    selected = result.get("selected", {}) or {}

    selected_files = []
    for key in ("channel_0_previous", "channel_1_middle", "channel_2_next"):
        item = selected.get(key)
        if isinstance(item, dict) and item.get("path"):
            selected_files.append(str(item["path"]))

    # Fallback for inference implementations that return paths directly.
    if len(selected_files) != 3:
        selected_files = [str(p) for p in result.get("selected_files", [])]

    return prediction, series, selected, selected_files


def _store_session(result):
    prediction, series, selected, selected_files = _extract_result(result)
    session = st.session_state.analysis_session

    session.study_id = safe(result.get("study_id"), None)
    session.series_id = safe(result.get("series_id"), None)
    session.selected_files = selected_files
    session.selected_metadata = selected
    session.series_metadata = series
    session.middle_slice = (
        selected.get("channel_1_middle", {}).get("path")
        if isinstance(selected.get("channel_1_middle"), dict)
        else None
    )
    session.tensor_shape = safe(result.get("tensor_shape"))
    session.device = safe(prediction.get("device"), "cuda:0")
    session.model_name = safe(result.get("model_name"), "Swin Transformer")
    session.model_checkpoint = safe(result.get("model_checkpoint"), None)
    session.predicted_class = safe(prediction.get("class_name"), "Unknown")
    session.predicted_class_id = int(prediction.get("class_id", 0) or 0)
    session.confidence = float(prediction.get("confidence", prediction.get("confidence_percent", 0.0)) or 0.0)
    session.probabilities = {
        str(k): float(v) for k, v in (prediction.get("probabilities", {}) or {}).items()
    }
    session.processed_images = result.get("processed_images", []) or []
    session.gradcam_generated = False
    session.xai_paths = {}
    session.xai_result = None
    session.report_generated = False
    session.report_path = None
    session.report_summary = None

    st.session_state.last_analysis = result
    st.session_state.analysis_result = result


def _render_viewer(result):
    images = result.get("processed_images") or []
    selected = result.get("selected") or {}
    if not images:
        return

    section("🩻 Three-Slice MRI Viewer", "These are the preprocessed channels supplied to the classifier.")
    labels = [
        ("Channel 0", "Previous", "channel_0_previous"),
        ("Channel 1", "Middle", "channel_1_middle"),
        ("Channel 2", "Next", "channel_2_next"),
    ]
    cols = st.columns(3)
    for col, image, (channel, role, key) in zip(cols, images, labels):
        item = selected.get(key, {}) if isinstance(selected, dict) else {}
        with col:
            st.markdown(f"**{channel} — {role}**")
            st.image(image, caption=filename(item.get("filename")), use_container_width=True, clamp=True)
            st.caption(f"Instance: {item.get('instance_number', 'N/A')}")


def render_analysis():
    st.title("🩻 MRI Analysis")
    st.caption("Upload a DICOM series and run inference with the existing trained Swin Transformer model.")

    st.info("Research workflow: DICOM series → 3 adjacent slices → preprocessing → inference → Results → XAI → Report. No retraining is performed.")

    uploaded = st.file_uploader(
        "Upload DICOM slices from one MRI series",
        type=["dcm"],
        accept_multiple_files=True,
        key="mri_dicom_uploader",
    )

    if not uploaded:
        st.markdown("### Start a New Analysis")
        st.write("Upload the DICOM files belonging to a single lumbar-spine MRI series.")
        if st.session_state.analysis_session.has_prediction:
            st.success("A previous analysis is still available in Results.")
            if st.button("📊 View Previous Results", use_container_width=True):
                navigate("📊 Results")
        return

    st.success(f"{len(uploaded)} DICOM file(s) selected.")

    saved = _save_uploads(uploaded)

    try:
        from inference.dicom_loader import read_series
        slices = read_series(UPLOAD_DIR)
    except Exception as exc:
        st.error("The uploaded files could not be read as a valid DICOM series.")
        st.exception(exc)
        return

    if len(slices) < 3:
        st.error("At least 3 readable DICOM slices are required because the trained classifier expects three adjacent channels.")
        return

    st.success(f"Readable DICOM slices: {len(slices)}")

    max_middle = len(slices) - 1
    default_middle = len(slices) // 2
    middle_index = st.slider(
        "Select middle slice",
        1 if len(slices) > 2 else 0,
        max_middle - 1 if len(slices) > 2 else max_middle,
        min(default_middle, max_middle - 1),
        help="The selected slice is Channel 1; its immediate neighbors become Channels 0 and 2.",
    )

    middle_name = filename(slices[middle_index].get("path") if isinstance(slices[middle_index], dict) else slices[middle_index])
    st.caption(f"Selected middle slice: {middle_name}")

    if st.button("🚀 Analyze Selected MRI", type="primary", use_container_width=True, key="analyze_selected_mri"):
        with st.spinner("Preprocessing MRI and running the existing trained model..."):
            try:
                result = analyze_series(str(UPLOAD_DIR), middle_index=middle_index)
                _store_session(result)
                st.success("MRI analysis completed successfully.")
                navigate("📊 Results")
            except Exception as exc:
                st.error("MRI analysis failed. The trained model was not modified.")
                st.exception(exc)
                return

    result = st.session_state.get("last_analysis")
    if result:
        st.markdown("---")
        _render_viewer(result)
