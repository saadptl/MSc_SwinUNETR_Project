"""
Dashboard Home Page — Explainable Swin-UNETR Lumbar Spine AI
Professional Medical Research Dashboard
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from ui.common import navigate

# ── Project root paths ──────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _APP_DIR.parent
_OUTPUTS = _PROJECT_ROOT / "outputs"
_SEG_OUTPUTS = _OUTPUTS / "segmentation"


def _read_csv_value(path: Path, metric_col: str, value_col: str, metric: str, fallback="N/A"):
    """Safe CSV single-value reader."""
    try:
        df = pd.read_csv(path)
        row = df[df[metric_col].str.strip() == metric]
        if not row.empty:
            return row[value_col].iloc[0]
    except Exception:
        pass
    return fallback


def _hero_section():
    st.markdown("""
    <div class="page-hero">
        <div class="eyebrow">MSc Computer Science &bull; Data Science &bull; Major Project</div>
        <h1>Explainable Swin-UNETR Framework</h1>
        <p class="subtitle">for Automated Lumbar Spine Disease Detection and Classification from MRI Images</p>
        <p class="description">Research &amp; Explainability Dashboard &mdash; 3D Swin-UNETR Localization &bull; Severity Classification &bull; XAI &bull; Quantitative Analysis</p>
    </div>
    """, unsafe_allow_html=True)


def _kpi_cards():
    # Load actual stats
    patients = "N/A"
    try:
        df = pd.read_csv(_OUTPUTS / "patient_statistics.csv")
        row = df[df["Statistic"] == "Unique Patients"]
        if not row.empty:
            patients = f"{int(row['Value'].iloc[0]):,}"
    except Exception:
        pass

    mri_series = "N/A"
    try:
        df = pd.read_csv(_OUTPUTS / "mri_series_statistics.csv")
        total = df["Count"].sum()
        mri_series = f"{int(total):,}"
    except Exception:
        pass

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-icon">👥</span>
            <div class="kpi-label">Patients</div>
            <div class="kpi-value">{patients}</div>
            <div class="kpi-sublabel">RSNA 2024 Dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-icon">🩻</span>
            <div class="kpi-label">MRI Series</div>
            <div class="kpi-value">{mri_series}</div>
            <div class="kpi-sublabel">Axial T2 · Sagittal T1/T2</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="kpi-card">
            <span class="kpi-icon">🧬</span>
            <div class="kpi-label">Disease Classes</div>
            <div class="kpi-value">5</div>
            <div class="kpi-sublabel">SCS · LFNN · RFNN · LSS · RSS</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        xai_modules = 5
        xai_dirs = [
            _SEG_OUTPUTS / "rsna_part45_3d_xai",
            _SEG_OUTPUTS / "rsna_part46_point_targeted_3d_xai",
            _SEG_OUTPUTS / "rsna_part46b_multi_point_3d_xai",
            _SEG_OUTPUTS / "rsna_part46c_xai_summary",
            _SEG_OUTPUTS / "rsna_part46d_xai_visual_inspection",
        ]
        available_count = sum(1 for d in xai_dirs if d.exists())
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-icon">🔬</span>
            <div class="kpi-label">XAI Modules</div>
            <div class="kpi-value">{available_count}/{xai_modules}</div>
            <div class="kpi-sublabel">Part 4.5 &ndash; 4.6D Available</div>
        </div>
        """, unsafe_allow_html=True)


def _project_overview_and_distribution():
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🎯 Project Overview</div>
            <p class="section-subtitle" style="margin-bottom:1rem;">Academic deep-learning framework for lumbar spine disease detection.</p>
            <p style="font-size:0.83rem; color:#4B5E8A; line-height:1.7; margin:0;">
            An end-to-end medical AI framework combining <strong>Swin Transformer</strong> severity 
            classification, <strong>3D Swin-UNETR</strong> disease localization, and a five-module 
            <strong>Explainable AI</strong> pipeline — including 3D gradient attribution, 
            point-targeted XAI, multi-point analysis, quantitative XAI metrics, and 
            visual inspection — over the <em>RSNA 2024 Lumbar Spine Degenerative Classification</em> dataset.
            </p>
            <br/>
            <p style="font-size:0.78rem; color:#6B7FA3;">
            ✦ MRI Analysis &nbsp;·&nbsp; ✦ Severity Classification &nbsp;·&nbsp; ✦ 3D Swin-UNETR Localization<br/>
            ✦ Grad-CAM XAI &nbsp;·&nbsp; ✦ Quantitative Attribution &nbsp;·&nbsp; ✦ Research Reporting
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("ℹ️ Learn More →", key="home_learn_more"):
            navigate("ℹ️ About Project")

    with right:
        _disease_distribution_chart()


def _disease_distribution_chart():
    diseases = []
    counts = []
    try:
        df = pd.read_csv(_OUTPUTS / "disease_annotation_summary.csv")
        for _, row in df.iterrows():
            diseases.append(str(row["condition"]))
            counts.append(int(row["Annotations"]))
    except Exception:
        diseases = [
            "Spinal Canal Stenosis",
            "Left Neural Foraminal Narrowing",
            "Right Neural Foraminal Narrowing",
            "Left Subarticular Stenosis",
            "Right Subarticular Stenosis",
        ]
        counts = [0, 0, 0, 0, 0]

    if not any(counts):
        st.info("Disease annotation data not available.")
        return

    short_names = {
        "Spinal Canal Stenosis": "SCS",
        "Left Neural Foraminal Narrowing": "LFNN",
        "Right Neural Foraminal Narrowing": "RFNN",
        "Left Subarticular Stenosis": "LSS",
        "Right Subarticular Stenosis": "RSS",
    }
    labels = [short_names.get(d, d) for d in diseases]
    colors = ["#3B82F6", "#8B5CF6", "#0D9488", "#EC4899", "#F59E0B"]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFF")

    bars = ax.barh(labels, counts, color=colors, height=0.55, edgecolor="none")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            ha="left",
            fontsize=8,
            color="#4B5E8A",
            fontweight="600",
        )

    ax.set_xlabel("Annotation Count", fontsize=8, color="#6B7FA3")
    ax.set_title("Disease Distribution", fontsize=10, fontweight="700", color="#0F1B3D", pad=8)
    ax.tick_params(labelsize=8, colors="#4B5E8A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E8EDF6")
    ax.spines["bottom"].set_color("#E8EDF6")
    ax.set_xlim(0, max(counts) * 1.15 if counts else 1)
    plt.tight_layout(pad=1.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _module_cards():
    st.markdown("---")
    st.markdown('<div class="section-title" style="margin-bottom:1rem;">🚀 Core Analysis Modules</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="feature-card blue-accent">
            <span class="feature-icon">🧠</span>
            <div class="feature-title">Severity Classification</div>
            <div class="feature-description">
                Swin Transformer-based lumbar spine severity classification from DICOM MRI series.
                Outputs Normal/Mild · Moderate · Severe with confidence scores.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("View Analysis →", key="home_cls_btn", use_container_width=True, type="primary"):
            navigate("🩻 MRI Analysis")

    with c2:
        st.markdown("""
        <div class="feature-card teal-accent">
            <span class="feature-icon">🧊</span>
            <div class="feature-title">3D Swin-UNETR Localization</div>
            <div class="feature-description">
                Disease and lumbar-level prediction from 3D MRI volumes. Multi-planar 
                localization visualization with axial, coronal and sagittal views.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("View Localization →", key="home_seg_btn", use_container_width=True, type="primary"):
            navigate("🧠 3D Segmentation")

    with c3:
        st.markdown("""
        <div class="feature-card purple-accent">
            <span class="feature-icon">💡</span>
            <div class="feature-title">Explainable AI</div>
            <div class="feature-description">
                Grad-CAM, 3D gradient attribution, point-targeted XAI and quantitative 
                explainability analysis across diseases and lumbar levels.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("View XAI →", key="home_xai_btn", use_container_width=True, type="primary"):
            navigate("🔬 Explainable AI")


def _system_status_row():
    st.markdown("---")
    st.markdown('<div class="section-title" style="margin-bottom:1rem;">⚡ Research Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("MRI Input", "DICOM Upload"),
        ("Preprocessing", "Normalization"),
        ("Classification", "Swin Transformer"),
        ("3D Processing", "64×96×96 Grid"),
        ("Localization", "Swin-UNETR"),
        ("Prediction", "Disease + Level"),
        ("XAI", "Attribution"),
        ("Report", "PDF Export"),
    ]

    cols = st.columns(len(steps))
    colors = ["#3B82F6","#0D9488","#8B5CF6","#3B82F6","#0D9488","#EC4899","#8B5CF6","#3B82F6"]
    for col, (title, sub), color in zip(cols, steps, colors):
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:0.7rem 0.3rem; background:#FFFFFF; 
                        border-radius:10px; border:1px solid #E8EDF6; border-top:3px solid {color};">
                <div style="font-size:0.8rem; font-weight:700; color:#0F1B3D;">{title}</div>
                <div style="font-size:0.7rem; color:#8A9DC4; margin-top:2px;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)


def _disclaimer():
    st.markdown("""
    <div class="research-disclaimer">
        <strong>⚠️ Academic Research Prototype — Not for Clinical Diagnosis</strong>
        This dashboard presents model-derived outputs for academic research and project evaluation only.
        XAI outputs are model-attribution patterns and must not be interpreted as clinically validated 
        lesion boundaries or diagnostic evidence.
    </div>
    """, unsafe_allow_html=True)


def render_home():
    _hero_section()
    _kpi_cards()
    st.markdown("<br/>", unsafe_allow_html=True)
    _project_overview_and_distribution()
    _module_cards()
    _system_status_row()
    st.markdown("<br/>", unsafe_allow_html=True)
    _disclaimer()
