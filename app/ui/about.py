"""
About Project Page — Professional academic project showcase
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from ui.common import navigate

_APP_DIR   = Path(__file__).resolve().parents[1]
_PROJ_ROOT = _APP_DIR.parent
_OUTPUTS   = _PROJ_ROOT / "outputs"


def _page_header():
    st.markdown("""
    <div class="page-title-row">
        <h2>ℹ️ About the Project</h2>
        <p>Academic research project information, dataset, methodology, and limitations</p>
    </div>
    """, unsafe_allow_html=True)


def _project_title_section():
    st.markdown("""
    <div class="section-card">
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:0.7rem; font-weight:600; letter-spacing:0.15em; text-transform:uppercase;
                        color:#3B82F6; margin-bottom:0.7rem;">
                MSc Computer Science (Data Science) · Major Project
            </div>
            <h2 style="font-size:1.5rem; font-weight:800; color:#0F1B3D; line-height:1.3; margin:0 0 1rem 0;">
                Explainable Swin-UNETR Framework for<br/>
                Automated Lumbar Spine Disease Detection<br/>
                and Classification from MRI Images
            </h2>
            <div style="display:flex; justify-content:center; gap:1rem; flex-wrap:wrap;">
                <span style="background:#EFF6FF; color:#2563EB; border:1px solid #BFDBFE; border-radius:20px; padding:4px 14px; font-size:0.78rem; font-weight:600;">
                    🧠 Swin Transformer
                </span>
                <span style="background:#F5F3FF; color:#7C3AED; border:1px solid #DDD6FE; border-radius:20px; padding:4px 14px; font-size:0.78rem; font-weight:600;">
                    🧊 Swin-UNETR
                </span>
                <span style="background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; border-radius:20px; padding:4px 14px; font-size:0.78rem; font-weight:600;">
                    🔬 XAI
                </span>
                <span style="background:#FFF1F2; color:#9F1239; border:1px solid #FECDD3; border-radius:20px; padding:4px 14px; font-size:0.78rem; font-weight:600;">
                    🩻 MRI Analysis
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _objective_section():
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">🎯 Project Objective</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div style="background:#FFFFFF; border-radius:12px; padding:1.4rem;
                    border:1px solid #E8EDF6; border-left:4px solid #3B82F6;">
            <div style="font-size:0.85rem; color:#4B5E8A; line-height:1.8;">
            This project implements an end-to-end medical AI framework for automated 
            lumbar spine degenerative disease detection. The system combines:
            <br/><br/>
            <strong style="color:#0F1B3D;">• Swin Transformer</strong> — Severity classification 
            (Normal/Mild · Moderate · Severe) from 2D DICOM MRI slices.<br/>
            <strong style="color:#0F1B3D;">• 3D Swin-UNETR</strong> — Point-supervised disease 
            localization across five degenerative conditions.<br/>
            <strong style="color:#0F1B3D;">• Explainable AI</strong> — Five-module XAI pipeline 
            including Grad-CAM, 3D attribution, quantitative analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#FFFFFF; border-radius:12px; padding:1.4rem;
                    border:1px solid #E8EDF6; border-left:4px solid #8B5CF6;">
            <div style="font-size:0.85rem; color:#4B5E8A; line-height:1.8;">
            <strong style="color:#0F1B3D;">Research Contributions:</strong><br/>
            • Point-to-pseudomask anatomical radius supervision strategy<br/>
            • Balanced LFNN/RFNN symmetry refinement training<br/>
            • Multi-scale 3D gradient attribution XAI pipeline<br/>
            • Quantitative XAI metrics (centroid distance, R10 concentration)<br/>
            • Clinical anatomy guard system (C2-to-S1 Level Verification)
            </div>
        </div>
        """, unsafe_allow_html=True)


def _dataset_section():
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">📁 Dataset</div>', unsafe_allow_html=True)

    # Load real data
    try:
        pt_df = pd.read_csv(_OUTPUTS / "patient_statistics.csv")
        patients = int(pt_df[pt_df["Statistic"] == "Unique Patients"]["Value"].iloc[0])
    except Exception:
        patients = 1975

    try:
        mri_df = pd.read_csv(_OUTPUTS / "mri_series_statistics.csv")
        axial_count    = int(mri_df[mri_df["MRI Series"] == "Axial T2"]["Count"].iloc[0])     if "Axial T2"       in mri_df["MRI Series"].values else 0
        sag_t1_count   = int(mri_df[mri_df["MRI Series"] == "Sagittal T1"]["Count"].iloc[0])  if "Sagittal T1"    in mri_df["MRI Series"].values else 0
        sag_t2_count   = int(mri_df[mri_df["MRI Series"] == "Sagittal T2/STIR"]["Count"].iloc[0]) if "Sagittal T2/STIR" in mri_df["MRI Series"].values else 0
        total_series   = axial_count + sag_t1_count + sag_t2_count
    except Exception:
        axial_count, sag_t1_count, sag_t2_count, total_series = 2340, 1980, 1974, 6294

    try:
        ann_df   = pd.read_csv(_OUTPUTS / "disease_annotation_summary.csv")
        total_ann = int(ann_df["Annotations"].sum())
    except Exception:
        total_ann = 48692

    st.markdown(f"""
    <div class="section-card">
        <div style="font-weight:700; color:#0F1B3D; font-size:1rem; margin-bottom:0.8rem;">
            📦 RSNA 2024 Lumbar Spine Degenerative Classification
        </div>
        <div style="display:flex; gap:2rem; flex-wrap:wrap; margin-bottom:1rem;">
            <div style="text-align:center;">
                <div style="font-size:1.6rem; font-weight:800; color:#3B82F6;">{patients:,}</div>
                <div style="font-size:0.75rem; color:#6B7FA3; text-transform:uppercase;">Patients</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.6rem; font-weight:800; color:#8B5CF6;">{total_series:,}</div>
                <div style="font-size:0.75rem; color:#6B7FA3; text-transform:uppercase;">MRI Series</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.6rem; font-weight:800; color:#0D9488;">{total_ann:,}</div>
                <div style="font-size:0.75rem; color:#6B7FA3; text-transform:uppercase;">Annotations</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.6rem; font-weight:800; color:#EC4899;">5</div>
                <div style="font-size:0.75rem; color:#6B7FA3; text-transform:uppercase;">Disease Classes</div>
            </div>
        </div>
        <div style="font-size:0.8rem; color:#6B7FA3;">
            MRI Series: Axial T2 ({axial_count:,}) · Sagittal T1 ({sag_t1_count:,}) · Sagittal T2/STIR ({sag_t2_count:,})
        </div>
        <div style="font-size:0.78rem; color:#8A9DC4; margin-top:0.4rem;">
            Diseases: Spinal Canal Stenosis · Left/Right Neural Foraminal Narrowing · Left/Right Subarticular Stenosis
        </div>
    </div>
    """, unsafe_allow_html=True)


def _methodology_pipeline():
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">⚙️ Research Methodology</div>', unsafe_allow_html=True)

    steps = [
        ("🩻", "MRI Input",              "DICOM series upload and metadata extraction"),
        ("⚙️", "Preprocessing",          "Normalization, resampling, canonical grid alignment"),
        ("🧠", "Swin Transformer",        "3-channel adjacent-slice severity classification"),
        ("🧊", "3D MRI Processing",       "64×96×96 canonical volume construction"),
        ("📍", "Point Supervision",       "Point-to-pseudomask radius annotation conversion"),
        ("🏥", "Swin-UNETR Localization", "5-class disease + lumbar level prediction"),
        ("💡", "XAI Pipeline",            "Grad-CAM · 3D attribution · quantitative metrics"),
        ("📄", "Results & Report",        "Evaluation metrics · PDF report generation"),
    ]
    for i, (icon, title, desc) in enumerate(steps):
        connector = "↓" if i < len(steps) - 1 else ""
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:0.8rem; padding:0.6rem 1rem;
                    background:#FFFFFF; border-radius:10px; border:1px solid #E8EDF6; margin-bottom:4px;">
            <div style="width:32px; height:32px; background:#EFF6FF; border-radius:50%;
                        display:flex; align-items:center; justify-content:center; font-size:1rem; flex-shrink:0;">
                {icon}
            </div>
            <div>
                <div style="font-size:0.85rem; font-weight:700; color:#0F1B3D;">{i+1}. {title}</div>
                <div style="font-size:0.75rem; color:#6B7FA3;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if connector:
            st.markdown(f'<div style="text-align:center; color:#CBD5E8; font-size:1.1rem; margin:0;">{connector}</div>', unsafe_allow_html=True)


def _technology_stack():
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">🛠️ Technology Stack</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="section-card" style="height:100%;">
            <div style="font-weight:700; color:#3B82F6; font-size:0.9rem; margin-bottom:0.8rem;">
                🧠 Deep Learning
            </div>
            <div style="font-size:0.8rem; color:#4B5E8A; line-height:2;">
                Python · PyTorch · MONAI<br/>
                Swin Transformer · Swin-UNETR<br/>
                Point Supervision · Pseudo-masks
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="section-card" style="height:100%;">
            <div style="font-weight:700; color:#8B5CF6; font-size:0.9rem; margin-bottom:0.8rem;">
                🩻 Medical Imaging
            </div>
            <div style="font-size:0.8rem; color:#4B5E8A; line-height:2;">
                pydicom · nibabel · OpenCV<br/>
                NumPy · Pandas · Matplotlib<br/>
                Multi-planar Reconstruction
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="section-card" style="height:100%;">
            <div style="font-weight:700; color:#0D9488; font-size:0.9rem; margin-bottom:0.8rem;">
                📊 Platform
            </div>
            <div style="font-size:0.8rem; color:#4B5E8A; line-height:2;">
                Streamlit · ReportLab<br/>
                NVIDIA RTX 2050 (4 GB)<br/>
                Windows · Python venv
            </div>
        </div>
        """, unsafe_allow_html=True)


def _clinical_doctrine():
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">🩺 Clinical Doctrine</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#F0FDF4; border-left:4px solid #16A34A; border-radius:8px; padding:1rem 1.4rem; margin-bottom:1rem;">
        <strong style="color:#15803D; font-size:0.88rem;">
            "Count from C2 · Trace the Anatomy · Verify the Level"
        </strong>
        <p style="font-size:0.8rem; color:#166534; line-height:1.7; margin:0.5rem 0 0 0;">
        In clinical spine practice, ~10–15% of patients have Lumbosacral Transitional Vertebrae (LSTV)
        causing off-by-one level numbering shifts that geometric metrics cannot detect.
        The C2-to-S1 Protocol Guard is implemented to ensure anatomical level verification 
        before finalizing any clinical-use output.
        </p>
    </div>
    """, unsafe_allow_html=True)


def _limitations():
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">⚠️ Research Limitations</div>', unsafe_allow_html=True)

    limitations = [
        "Trained on RSNA 2024 point-supervised annotations, not manual voxel-level clinical masks.",
        "Transitional anatomy variants (LSTV) cannot be definitively ruled out from localized lumbar FOV scans without full-spine C2 scout views.",
        "Model outputs, localization probabilities, and Grad-CAM heatmaps are research evaluation artifacts.",
        "This application is an academic prototype developed for MSc evaluation — NOT approved as a diagnostic medical device.",
        "XAI attribution maps are model-evidence patterns; they do not constitute clinically validated lesion boundaries.",
    ]

    for lim in limitations:
        st.markdown(f"""
        <div style="display:flex; gap:0.6rem; align-items:flex-start; padding:0.5rem 0; border-bottom:1px solid #F0F4FF;">
            <span style="color:#F59E0B; font-size:0.9rem; flex-shrink:0;">⚠</span>
            <span style="font-size:0.82rem; color:#4B5E8A;">{lim}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="research-disclaimer" style="margin-top:1rem;">
        <strong>⚠️ Clinical Warning</strong>
        This software is for academic research and educational demonstration only. It must not be used for 
        patient diagnosis or clinical treatment planning without oversight by certified medical professionals.
    </div>
    """, unsafe_allow_html=True)


def render_about():
    _page_header()
    _project_title_section()
    _objective_section()
    _dataset_section()

    col_pipe, col_method = st.columns([1, 1.2], gap="large")
    with col_pipe:
        _methodology_pipeline()
    with col_method:
        _technology_stack()
        st.markdown("<br/>", unsafe_allow_html=True)
        _clinical_doctrine()

    _limitations()

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📊 View Results", key="about_results", use_container_width=True, type="primary"):
            navigate("📊 Results")
    with c2:
        if st.button("🔬 Explainable AI", key="about_xai", use_container_width=True):
            navigate("🔬 Explainable AI")
    with c3:
        if st.button("⚙️ System Info", key="about_sys", use_container_width=True):
            navigate("⚙️ System & Deployment")

    st.caption("Lumbar Spine AI · MSc Major Project · Swin-UNETR Framework")
