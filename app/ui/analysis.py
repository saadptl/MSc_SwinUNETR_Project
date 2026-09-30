"""
MRI Analysis Page — Professional redesign
Upload DICOM series, view MRI slices, run classification inference.
Preserves existing inference pipeline.
"""

from pathlib import Path
import shutil
import streamlit as st

from inference.dashboard_inference import analyze_series
from ui.common import navigate, filename, safe, section, CLASS_ORDER, CLASS_ICONS, pct

_APP_DIR   = Path(__file__).resolve().parents[1]
_PROJ_ROOT = _APP_DIR.parent
UPLOAD_DIR = _PROJ_ROOT / "outputs" / "dashboard" / "uploads" / "current_series"


def _save_uploads(uploaded_files):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for item in UPLOAD_DIR.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    paths = []
    for item in uploaded_files:
        dest = UPLOAD_DIR / Path(item.name).name
        dest.write_bytes(item.getbuffer())
        paths.append(dest)
    return paths


def _extract_result(result):
    prediction = result.get("prediction", {}) or {}
    series     = result.get("series",     {}) or {}
    selected   = result.get("selected",   {}) or {}

    selected_files = []
    for key in ("channel_0_previous", "channel_1_middle", "channel_2_next"):
        item = selected.get(key)
        if isinstance(item, dict) and item.get("path"):
            selected_files.append(str(item["path"]))

    if len(selected_files) != 3:
        selected_files = [str(p) for p in result.get("selected_files", [])]

    return prediction, series, selected, selected_files


def _store_session(result):
    prediction, series, selected, selected_files = _extract_result(result)
    session = st.session_state.analysis_session

    session.study_id         = safe(result.get("study_id"), None)
    session.series_id        = safe(result.get("series_id"), None)
    session.selected_files   = selected_files
    session.selected_metadata = selected
    session.series_metadata  = series
    session.middle_slice     = (
        selected.get("channel_1_middle", {}).get("path")
        if isinstance(selected.get("channel_1_middle"), dict) else None
    )
    session.tensor_shape     = safe(result.get("tensor_shape"))
    session.device           = safe(prediction.get("device"), "cuda:0")
    session.model_name       = safe(result.get("model_name"), "Swin Transformer")
    session.model_checkpoint = safe(result.get("model_checkpoint"), None)
    session.predicted_class  = safe(prediction.get("class_name"), "Unknown")
    session.predicted_class_id = int(prediction.get("class_id", 0) or 0)
    session.confidence       = float(
        prediction.get("confidence", prediction.get("confidence_percent", 0.0)) or 0.0
    )
    session.probabilities    = {
        str(k): float(v) for k, v in (prediction.get("probabilities", {}) or {}).items()
    }
    session.processed_images = result.get("processed_images", []) or []
    session.gradcam_generated = False
    session.xai_paths        = {}
    session.xai_result       = None
    session.report_generated = False
    session.report_path      = None
    session.report_summary   = None

    st.session_state.last_analysis   = result
    st.session_state.analysis_result = result


def _page_header():
    st.markdown("""
    <div class="page-title-row">
        <h2>🩻 MRI Analysis</h2>
        <p>Upload or select an MRI series to run classification and localization analysis</p>
    </div>
    """, unsafe_allow_html=True)


def _render_result_panel(session):
    """Show classification result in a professional right-panel style."""
    prediction = safe(session.predicted_class, "Unknown")
    confidence = pct(session.confidence)
    probs      = session.probabilities or {}
    icon       = CLASS_ICONS.get(prediction, "🔵")

    sev_color = {
        "Normal/Mild": "#34D399",
        "Moderate":    "#F59E0B",
        "Severe":      "#F87171",
    }.get(prediction, "#3B82F6")

    st.markdown(f"""
    <div style="background:#FFFFFF; border-radius:14px; padding:1.4rem;
                border:1px solid #E8EDF6; box-shadow:0 2px 10px rgba(15,27,61,0.07);
                border-top:4px solid {sev_color};">
        <div style="font-size:0.7rem; font-weight:600; color:#6B7FA3;
                    text-transform:uppercase; margin-bottom:0.6rem;">
            CLASSIFICATION RESULT
        </div>
        <div style="font-size:1.4rem; font-weight:800; color:#0F1B3D; margin-bottom:0.3rem;">
            {icon} {prediction}
        </div>
        <div style="font-size:0.82rem; color:#6B7FA3; margin-bottom:1rem;">
            Confidence: <strong style="color:#0F1B3D;">{confidence:.2f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("**Probability Distribution**")

    prob_colors = {
        "Normal/Mild": "#34D399",
        "Moderate":    "#F59E0B",
        "Severe":      "#F87171",
    }
    for cls in CLASS_ORDER:
        val  = pct(probs.get(cls, 0.0))
        c    = prob_colors.get(cls, "#3B82F6")
        icon_c = CLASS_ICONS.get(cls, "•")
        st.markdown(f"""
        <div style="margin:0.4rem 0;">
            <div style="display:flex; justify-content:space-between;
                        font-size:0.77rem; color:#4B5E8A; margin-bottom:3px;">
                <span>{icon_c} {cls}</span><span><strong>{val:.1f}%</strong></span>
            </div>
            <div style="height:8px; background:#F0F4FF; border-radius:4px; overflow:hidden;">
                <div style="width:{val}%; height:100%; background:{c}; border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    meta = session.series_metadata if isinstance(session.series_metadata, dict) else {}
    study_id  = safe(session.study_id,  "Upload")
    series_id = safe(session.series_id, "Upload")
    slices    = meta.get("number_of_slices", "N/A")
    img_size  = meta.get("image_size", "N/A")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem; font-weight:600; color:#6B7FA3;
                text-transform:uppercase; margin-bottom:0.5rem;">SERIES INFO</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.79rem; color:#4B5E8A; line-height:1.9;">
        Study ID: <code style="font-size:0.73rem;">{study_id}</code><br/>
        Series ID: <code style="font-size:0.73rem;">{series_id}</code><br/>
        Total Slices: <strong>{slices}</strong><br/>
        Image Size: <strong>{img_size}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("📊 Full Results →", type="primary", use_container_width=True, key="analysis_results_btn"):
        navigate("📊 Results")
    if st.button("🧠 3D Localization →", use_container_width=True, key="analysis_seg_btn"):
        navigate("🧠 3D Segmentation")
    if st.button("🔬 Explainable AI →", use_container_width=True, key="analysis_xai_btn"):
        navigate("🔬 Explainable AI")


def _render_mri_viewer(result):
    """Display the MRI viewer with preprocessed slices."""
    images   = result.get("processed_images") or []
    selected = result.get("selected")         or {}

    if not images:
        return

    st.markdown('<div class="section-title" style="margin-bottom:0.8rem;">🩻 MRI Slice Viewer</div>', unsafe_allow_html=True)
    st.caption("Three adjacent DICOM slices supplied to the classifier (Channel 0 / 1 / 2).")

    labels = [
        ("Ch. 0", "Previous", "channel_0_previous"),
        ("Ch. 1", "Middle",   "channel_1_middle"),
        ("Ch. 2", "Next",     "channel_2_next"),
    ]

    cols = st.columns(3)
    for col, image, (ch, role, key) in zip(cols, images, labels):
        item = selected.get(key, {}) if isinstance(selected, dict) else {}
        with col:
            st.markdown(f"""
            <div style="background:#F8FAFF; border-radius:10px; padding:4px;
                        border:1px solid #CBD5E1; margin-bottom:6px;">
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption(f"**{ch} — {role}** · Instance: {item.get('instance_number', 'N/A')}")

    # Thumbnail strip
    series_metadata = result.get("series", {}) or {}
    if isinstance(series_metadata, dict):
        st.markdown('<div style="font-size:0.75rem; color:#6B7FA3; margin-top:0.5rem;">Thumbnail Strip</div>', unsafe_allow_html=True)
        all_files = result.get("all_files", [])
        if all_files:
            thumb_cols = st.columns(min(8, len(all_files)))
            for i, (tc, fp) in enumerate(zip(thumb_cols, all_files[:8])):
                with tc:
                    try:
                        import pydicom
                        import numpy as np
                        dcm = pydicom.dcmread(str(fp))
                        arr = dcm.pixel_array.astype(float)
                        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
                        tc.image(arr.astype("uint8"), use_container_width=True)
                    except Exception:
                        tc.markdown(f"<div style='font-size:0.65rem; color:#8A9DC4;'>{i}</div>", unsafe_allow_html=True)


def render_analysis():
    _page_header()

    # Research info banner
    st.markdown("""
    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-left:4px solid #3B82F6;
                border-radius:8px; padding:0.8rem 1.2rem; font-size:0.82rem; color:#1E40AF; margin-bottom:1.2rem;">
        <strong>Research Workflow:</strong> DICOM series → 3 adjacent slices → preprocessing → 
        inference → Results → XAI → Report. No retraining is performed on the existing checkpoint.
    </div>
    """, unsafe_allow_html=True)

    session = st.session_state.analysis_session

    # ── Layout ────────────────────────────────────────────────────────────
    left_col, center_col, right_col = st.columns([1, 1.8, 1.1], gap="medium")

    with left_col:
        st.markdown("""
        <div class="section-title" style="margin-bottom:0.8rem;">📂 Select MRI Study</div>
        """, unsafe_allow_html=True)

        sample_cases = {
            "Case 1: Patient 44036939 / Series 2828203845 (Sagittal T2)": _PROJ_ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification" / "test_images" / "44036939" / "2828203845",
            "Case 2: Patient 44036939 / Series 3481971518 (Axial T2)": _PROJ_ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification" / "test_images" / "44036939" / "3481971518",
            "Case 3: Patient 44036939 / Series 3844393089 (Sagittal T1)": _PROJ_ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification" / "test_images" / "44036939" / "3844393089",
        }

        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0F1B3D; margin-bottom:4px;'>⚡ Option A: RSNA Benchmark Cases</div>", unsafe_allow_html=True)
        selected_sample_label = st.selectbox(
            "Select Evaluation Case",
            list(sample_cases.keys()),
            key="analysis_sample_choice",
        )
        if st.button("⚡ Load Selected Case", use_container_width=True, key="load_sample_case_btn"):
            target_dir = sample_cases[selected_sample_label]
            if target_dir.exists():
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                for item in UPLOAD_DIR.iterdir():
                    if item.is_file() or item.is_symlink(): item.unlink()
                for dcm_f in target_dir.glob("*.dcm"):
                    shutil.copy2(dcm_f, UPLOAD_DIR / dcm_f.name)
                st.session_state["sample_case_loaded"] = selected_sample_label
                st.success(f"Loaded {len(list(UPLOAD_DIR.glob('*.dcm')))} DICOM slices!")
                st.rerun()

        st.markdown("<div style='margin:0.8rem 0; border-top:1px dashed #CBD5E1;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0F1B3D; margin-bottom:4px;'>📁 Option B: Custom DICOM Upload</div>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload DICOM slices",
            type=["dcm"],
            accept_multiple_files=True,
            key="mri_dicom_uploader",
            help="Upload all DICOM files from one lumbar spine MRI series.",
        )

        if session.has_prediction:
            st.markdown("""
            <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:8px;
                        padding:0.6rem 0.8rem; font-size:0.78rem; color:#065F46; margin-top:0.6rem;">
                ✅ Previous analysis available
            </div>
            """, unsafe_allow_html=True)
            if st.button("📊 View Previous Results", use_container_width=True, key="prev_results"):
                navigate("📊 Results")

        has_local_series = UPLOAD_DIR.exists() and any(UPLOAD_DIR.glob("*.dcm"))
        if not uploaded and not has_local_series:
            if not session.has_prediction:
                st.markdown("""
                <div style="margin-top:1rem; padding:1rem; background:#F8FAFF; border-radius:10px;
                            border:1px dashed #C7D4EC; text-align:center; color:#8A9DC4;">
                    <div style="font-size:1.5rem;">🩻</div>
                    <div style="font-size:0.8rem; margin-top:0.4rem;">
                        Load a benchmark case above<br/>or upload DICOM files
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Center: Viewer or upload response ────────────────────────────────
    with center_col:
        st.markdown("""
        <div class="section-title" style="margin-bottom:0.8rem;">🖥️ MRI Viewer</div>
        """, unsafe_allow_html=True)

        if uploaded:
            st.success(f"✅ {len(uploaded)} custom DICOM file(s) selected.")
            _save_uploads(uploaded)
        elif has_local_series:
            active_lbl = st.session_state.get("sample_case_loaded", "Preloaded RSNA Test Case")
            st.info(f"📂 Active Study: **{active_lbl}**")

        if uploaded or has_local_series:
            try:
                from inference.dicom_loader import read_series
                slices = read_series(UPLOAD_DIR)
            except Exception as exc:
                st.error("The uploaded files could not be read as a valid DICOM series.")
                with st.expander("Technical details"):
                    st.exception(exc)
                return

            if len(slices) < 3:
                st.error("At least 3 readable DICOM slices are required (classifier expects 3 adjacent channels).")
                return

            st.caption(f"Readable DICOM slices: **{len(slices)}**")

            max_m  = len(slices) - 1
            def_m  = len(slices) // 2
            middle = st.slider(
                "Select middle slice (Channel 1)",
                1 if len(slices) > 2 else 0,
                max_m - 1 if len(slices) > 2 else max_m,
                min(def_m, max_m - 1),
                help="Slice selected as Channel 1; its immediate neighbors become Channels 0 and 2.",
            )

            fn = filename(
                slices[middle].get("path") if isinstance(slices[middle], dict) else slices[middle]
            )
            st.caption(f"Middle slice: **{fn}**")

            if st.button(
                "🚀 Run Analysis",
                type="primary",
                use_container_width=True,
                key="analyze_selected_mri",
            ):
                with st.spinner("Preprocessing MRI and running the existing trained model..."):
                    try:
                        result = analyze_series(str(UPLOAD_DIR), middle_index=middle)
                        _store_session(result)
                        st.success("✅ MRI analysis completed successfully.")
                        navigate("📊 Results")
                    except Exception as exc:
                        st.error("MRI analysis failed. The trained model was not modified.")
                        with st.expander("Technical details"):
                            st.exception(exc)
                        return

        result = st.session_state.get("last_analysis")
        if result:
            _render_mri_viewer(result)
        elif not uploaded:
            st.markdown("""
            <div style="padding:3rem 2rem; background:#F8FAFF; border-radius:12px;
                        border:1px solid #CBD5E1; text-align:center;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">&#x1FA7B;</div>
                <div style="font-size:0.9rem; font-weight:600; color:#0F1B3D;">MRI Viewer</div>
                <div style="font-size:0.8rem; margin-top:0.3rem; color:#475569;">Upload DICOM files to view MRI slices</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Right: Current Analysis ──────────────────────────────────────────
    with right_col:
        st.markdown("""
        <div class="section-title" style="margin-bottom:0.8rem;">📋 Current Analysis</div>
        """, unsafe_allow_html=True)

        if session.has_prediction:
            _render_result_panel(session)
        else:
            st.markdown("""
            <div style="padding:2rem 1rem; background:#F8FAFF; border-radius:12px;
                        border:1px solid #CBD5E1; text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">&#x1F4CA;</div>
                <div style="font-size:0.82rem; font-weight:600; color:#0F1B3D;">No analysis yet</div>
                <div style="font-size:0.75rem; margin-top:0.3rem; color:#475569;">Upload DICOM and run analysis</div>
            </div>
            """, unsafe_allow_html=True)
