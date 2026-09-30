"""
System & Deployment Page — Technical runtime and project structure information
"""

from pathlib import Path
import sys
import json
import streamlit as st
import torch

from ui.common import navigate

_APP_DIR   = Path(__file__).resolve().parents[1]
_PROJ_ROOT = _APP_DIR.parent
_SEG       = _PROJ_ROOT / "outputs" / "segmentation"

_XAI_DIRS = {
    "Part 4.5 — 3D XAI":              _SEG / "rsna_part45_3d_xai",
    "Part 4.6 — Point-Targeted XAI":  _SEG / "rsna_part46_point_targeted_3d_xai",
    "Part 4.6B — Multi-Point XAI":    _SEG / "rsna_part46b_multi_point_3d_xai",
    "Part 4.6C — Quantitative XAI":   _SEG / "rsna_part46c_xai_summary",
    "Part 4.6D — Visual Inspection":  _SEG / "rsna_part46d_xai_visual_inspection",
}


def _page_header():
    st.markdown("""
    <div class="page-title-row">
        <h2>⚙️ System &amp; Deployment</h2>
        <p>Runtime environment, model checkpoint, technology stack, and project structure</p>
    </div>
    """, unsafe_allow_html=True)


def _system_info_cards():
    st.markdown('<div class="section-title" style="margin-bottom:1rem;">🖥️ Runtime Environment</div>', unsafe_allow_html=True)

    cuda_avail = torch.cuda.is_available()
    gpu_name   = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    gpu_mem    = (
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        if cuda_avail else "N/A"
    )

    cols = st.columns(4)
    items = [
        ("Python",   sys.version.split()[0],         "#3B82F6"),
        ("PyTorch",  torch.__version__,               "#8B5CF6"),
        ("CUDA",     torch.version.cuda or "CPU",     "#0D9488"),
        ("GPU Mem",  gpu_mem,                         "#EC4899"),
    ]
    for col, (label, val, color) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div style="background:#FFFFFF; border-radius:12px; padding:1.2rem 1.4rem;
                        border:1px solid #E8EDF6; border-top:3px solid {color};
                        box-shadow:0 2px 8px rgba(15,27,61,0.06); text-align:center;">
                <div style="font-size:0.72rem; font-weight:600; color:#6B7FA3; text-transform:uppercase; margin-bottom:0.4rem;">{label}</div>
                <div style="font-size:1.3rem; font-weight:800; color:#0F1B3D;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    if cuda_avail:
        st.markdown(f"""
        <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:8px; padding:0.7rem 1.2rem;
                    margin-top:0.8rem; font-size:0.82rem; color:#065F46;">
            ✅ GPU Detected: <strong>{gpu_name}</strong> — CUDA inference enabled
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#FFF7ED; border:1px solid #FED7AA; border-radius:8px; padding:0.7rem 1.2rem;
                    margin-top:0.8rem; font-size:0.82rem; color:#92400E;">
            ⚠️ No GPU detected — running on CPU
        </div>
        """, unsafe_allow_html=True)


def _model_checkpoints():
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">🔑 Model Checkpoints</div>', unsafe_allow_html=True)

    checkpoints = {
        "Classification (Swin Transformer)":
            _PROJ_ROOT / "models" / "best_model.pth",
        "Localization (Swin-UNETR Part 3.3)":
            _PROJ_ROOT / "outputs" / "segmentation" / "rsna_part33_balanced_symmetry_refinement"
            / "checkpoints" / "part33_best_development_macro.pth",
    }

    for label, ckpt_path in checkpoints.items():
        exists = ckpt_path.exists()
        size   = f"{ckpt_path.stat().st_size / 1e6:.1f} MB" if exists else "N/A"
        status_html = (
            f'<span style="background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; border-radius:12px; padding:2px 10px; font-size:0.72rem; font-weight:600;">✅ Found &bull; {size}</span>'
            if exists else
            '<span style="background:#FFF1F2; color:#9F1239; border:1px solid #FECDD3; border-radius:12px; padding:2px 10px; font-size:0.72rem; font-weight:600;">✗ Not Found</span>'
        )
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.7rem 1rem;
                    background:#FFFFFF; border-radius:8px; border:1px solid #E8EDF6; margin-bottom:6px;">
            <div>
                <div style="font-size:0.85rem; font-weight:600; color:#0F1B3D;">{label}</div>
                <div style="font-size:0.7rem; color:#8A9DC4; margin-top:1px;">{ckpt_path.name}</div>
            </div>
            {status_html}
        </div>
        """, unsafe_allow_html=True)

    st.caption("Dashboard is inference-only. No training loop, optimizer, or checkpoint-writing operation runs here.")


def _artifact_status_table():
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">📦 Research Artifact Status</div>', unsafe_allow_html=True)

    rows = []
    for label, path in _XAI_DIRS.items():
        exists = path.exists()
        if exists:
            try:
                pngs = list(path.rglob("*.png"))
                csvs = list(path.rglob("*.csv"))
                jsons = list(path.rglob("*.json"))
                counts = f"{len(pngs)} PNG · {len(csvs)} CSV · {len(jsons)} JSON"
            except Exception:
                counts = "Available"
        else:
            counts = "—"

        rows.append({
            "Module":    label,
            "Status":    "✅ Available" if exists else "⚠️ Missing",
            "Artifacts": counts,
        })

    df = __import__("pandas").DataFrame(rows)
    st.dataframe(df.set_index("Module"), use_container_width=True)


def _architecture_diagram():
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">🏗️ System Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px;
                padding:1.2rem 1.4rem; overflow-x:auto; margin-top:0.5rem;">
    <pre style="font-family:'Courier New',Courier,monospace; font-size:0.7rem;
                color:#0F1B3D; -webkit-text-fill-color:#0F1B3D;
                line-height:1.55; margin:0; background:transparent; white-space:pre;">  ┌─────────────────────────────────────────────────────────────┐
  │                    Streamlit Dashboard                       │
  │  ┌───────────┐  ┌──────────┐  ┌─────────────────────────┐  │
  │  │ MRI Upload│  │ Analysis │  │ Results / Report / XAI  │  │
  │  └─────┬─────┘  └────┬─────┘  └───────────┬─────────────┘  │
  └────────┼─────────────┼───────────────────────────────────┘
           │             │
  ┌────────▼─────────────▼────────────────────────────────────┐
  │                  Inference Pipeline                        │
  │  ┌────────────────┐         ┌───────────────────────────┐ │
  │  │ DICOM Loader   │         │  Preprocessing Pipeline   │ │
  │  │ (pydicom)      │────────►│  Normalization / Resize   │ │
  │  └────────────────┘         └──────────────┬────────────┘ │
  │                                            │               │
  │  ┌─────────────────────┐   ┌──────────────────────────┐   │
  │  │ Swin Transformer    │   │ 3D Swin-UNETR            │   │
  │  │ Classification      │   │ Disease Localization      │   │
  │  │ (best_model.pth)    │   │ (Part 3.3 Checkpoint)    │   │
  │  └──────────┬──────────┘   └──────────────┬───────────┘   │
  │             │                             │               │
  │  ┌──────────▼─────────────────────────────▼──────────┐   │
  │  │               XAI Pipeline                         │   │
  │  │  Grad-CAM · 3D Attribution · Quantitative Metrics │   │
  │  └──────────────────────────┬───────────────────────┘   │
  │                             │                           │
  │  ┌──────────────────────────▼───────────────────────┐  │
  │  │         Report Generator (ReportLab PDF)          │  │
  │  └──────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────┘</pre>
    </div>
    """, unsafe_allow_html=True)



def _project_structure():
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">📁 Project Structure</div>', unsafe_allow_html=True)

    tree = {
        "app/": {
            "main.py":      "Dashboard entry point",
            "config.py":    "Application configuration",
            "ui/":          "All page modules",
            "core/":        "Analysis session management",
            "inference/":   "Model inference pipeline",
            "assets/":      "CSS stylesheets",
        },
        "models/":           "best_model.pth (Swin Transformer)",
        "outputs/": {
            "classification_report.csv": "Classification metrics",
            "evaluation_metrics.csv":    "Overall metrics",
            "confusion_matrix.csv":      "Confusion matrix",
            "disease_annotation_summary.csv": "Disease annotations",
            "segmentation/": {
                "rsna_part33_balanced_symmetry_refinement/checkpoints/": "Swin-UNETR checkpoint",
                "rsna_part33_untouched_test_evaluation/":  "Test evaluation",
                "rsna_part45_3d_xai/":         "Part 4.5 XAI",
                "rsna_part46_point_targeted_3d_xai/": "Part 4.6 XAI",
                "rsna_part46b_multi_point_3d_xai/":  "Part 4.6B XAI",
                "rsna_part46c_xai_summary/":    "Part 4.6C Quantitative XAI",
                "rsna_part46d_xai_visual_inspection/": "Part 4.6D Visual XAI",
            },
        },
        "dataset/":          "RSNA 2024 Lumbar Spine dataset",
    }

    def render_tree(d, indent=0):
        for key, val in d.items():
            prefix = "  " * indent
            if isinstance(val, dict):
                st.markdown(f"`{prefix}📂 {key}`")
                render_tree(val, indent + 1)
            else:
                st.markdown(f"`{prefix}📄 {key}` — {val}")

    with st.expander("🌲 Expand Project Tree", expanded=False):
        render_tree(tree)


def render_system():
    _page_header()
    _system_info_cards()
    _model_checkpoints()

    col_art, col_arch = st.columns([1, 1.4], gap="large")
    with col_art:
        _artifact_status_table()
    with col_arch:
        _architecture_diagram()

    _project_structure()

    st.markdown("---")
    st.markdown("""
    <div class="research-disclaimer">
        <strong>Deployment Mode: Inference-Only</strong>
        The dashboard does not contain an optimizer, training loop, or checkpoint-writing operation.
        All model weights are loaded read-only from existing validated checkpoints.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("ℹ️ About Project", key="sys_about", width="stretch"):
            navigate("ℹ️ About Project")
    with c2:
        if st.button("🏠 Dashboard", key="sys_home", width="stretch", type="primary"):
            navigate("🏠 Dashboard")
