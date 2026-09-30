"""
Results Page — Model Performance Results
Displays classification metrics, confusion matrix, and 3D localization performance.
All values loaded from actual project CSV/JSON artifacts.
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

from ui.common import CLASS_ORDER, CLASS_ICONS, navigate, filename, safe, pct, section

# ── Paths ───────────────────────────────────────────────────────────────────
_APP_DIR   = Path(__file__).resolve().parents[1]
_PROJ_ROOT = _APP_DIR.parent
_OUTPUTS   = _PROJ_ROOT / "outputs"
_SEG       = _OUTPUTS / "segmentation"
_TEST_EVAL = _SEG / "rsna_part33_untouched_test_evaluation"


def _session():
    s = st.session_state.get("analysis_session")
    return s if s and s.has_prediction else None


# ── Page Header ─────────────────────────────────────────────────────────────

def _page_header():
    st.markdown("""
    <div class="page-title-row">
        <h2>📊 Model Performance Results</h2>
        <p>Evaluation metrics for classification and disease localization — loaded from project artifacts</p>
    </div>
    """, unsafe_allow_html=True)


# ── Classification Performance ───────────────────────────────────────────────

def _load_classification_metrics():
    try:
        df = pd.read_csv(_OUTPUTS / "evaluation_metrics.csv")
        data = dict(zip(df["Metric"].str.strip(), df["Value"].astype(float)))
        return data
    except Exception:
        return {}


def _load_classification_report():
    try:
        df = pd.read_csv(_OUTPUTS / "classification_report.csv", index_col=0)
        return df
    except Exception:
        return None


def _load_confusion_matrix():
    try:
        df = pd.read_csv(_OUTPUTS / "confusion_matrix.csv", index_col=0)
        return df
    except Exception:
        return None


def _classification_kpi_cards(metrics: dict):
    st.markdown('<div class="section-title" style="margin-bottom:1rem;">📈 Classification Performance</div>', unsafe_allow_html=True)

    acc  = metrics.get("Accuracy", None)
    prec = metrics.get("Precision", None)
    rec  = metrics.get("Recall", None)
    f1   = metrics.get("F1 Score", None)

    cols = st.columns(4)
    items = [
        ("Accuracy",  acc,  "#3B82F6"),
        ("Precision", prec, "#8B5CF6"),
        ("Recall",    rec,  "#0D9488"),
        ("F1 Score",  f1,   "#EC4899"),
    ]
    for col, (label, val, color) in zip(cols, items):
        display = f"{val*100:.1f}%" if val is not None else "N/A"
        with col:
            st.markdown(f"""
            <div style="background:#FFFFFF; border-radius:12px; padding:1.2rem 1.4rem;
                        border:1px solid #E8EDF6; border-top:3px solid {color};
                        box-shadow:0 2px 8px rgba(15,27,61,0.06); text-align:center;">
                <div style="font-size:0.72rem; font-weight:600; color:#6B7FA3;
                            text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;">{label}</div>
                <div style="font-size:2rem; font-weight:800; color:#0F1B3D;">{display}</div>
            </div>
            """, unsafe_allow_html=True)


def _confusion_matrix_plot(df_cm: pd.DataFrame):
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">🔲 Confusion Matrix (Severity Classification)</div>', unsafe_allow_html=True)

    labels = list(df_cm.index)
    cm = df_cm.values.astype(float)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#FFFFFF")

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted Label", fontsize=8.5, color="#4B5E8A")
    ax.set_ylabel("True Label", fontsize=8.5, color="#4B5E8A")
    ax.set_title("Severity Classification Confusion Matrix", fontsize=9, fontweight="700",
                 color="#0F1B3D", pad=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = int(cm[i, j])
            color = "white" if cm[i, j] > cm.max() * 0.6 else "#0F1B3D"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=9, fontweight="700", color=color)

    plt.tight_layout(pad=1.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _class_wise_metrics_chart(df_report: pd.DataFrame):
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">📋 Class-wise Metrics</div>', unsafe_allow_html=True)

    classes = ["Normal/Mild", "Moderate", "Severe"]
    avail = [c for c in classes if c in df_report.index]
    if not avail:
        st.info("Class-wise metrics not available in the classification report.")
        return

    prec_vals  = [df_report.loc[c, "precision"]  if c in df_report.index else 0 for c in classes]
    rec_vals   = [df_report.loc[c, "recall"]      if c in df_report.index else 0 for c in classes]
    f1_vals    = [df_report.loc[c, "f1-score"]    if c in df_report.index else 0 for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFF")

    b1 = ax.bar(x - width, prec_vals, width, label="Precision", color="#3B82F6", edgecolor="none")
    b2 = ax.bar(x,         rec_vals,  width, label="Recall",    color="#0D9488", edgecolor="none")
    b3 = ax.bar(x + width, f1_vals,   width, label="F1 Score",  color="#EC4899", edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=8, color="#6B7FA3")
    ax.set_title("Class-wise Metrics", fontsize=10, fontweight="700", color="#0F1B3D", pad=6)
    ax.legend(fontsize=7.5, framealpha=0.9)
    ax.tick_params(labelsize=8, colors="#4B5E8A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E8EDF6")
    ax.spines["bottom"].set_color("#E8EDF6")

    plt.tight_layout(pad=1.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Table
    rows = []
    for cls in classes:
        if cls in df_report.index:
            rows.append({
                "Class": cls,
                "Precision": f"{df_report.loc[cls,'precision']:.4f}",
                "Recall":    f"{df_report.loc[cls,'recall']:.4f}",
                "F1 Score":  f"{df_report.loc[cls,'f1-score']:.4f}",
                "Support":   f"{int(df_report.loc[cls,'support']):,}" if "support" in df_report.columns else "N/A",
            })
        else:
            rows.append({"Class": cls, "Precision": "N/A", "Recall": "N/A", "F1 Score": "N/A", "Support": "N/A"})

    df_table = pd.DataFrame(rows).set_index("Class")
    st.dataframe(df_table, use_container_width=True)


# ── 3D Localization Performance ──────────────────────────────────────────────

def _load_localization_summary():
    summary_path = _TEST_EVAL / "part33_untouched_test_summary.json"
    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _load_disease_metrics():
    try:
        return pd.read_csv(_TEST_EVAL / "part33_test_disease_metrics.csv")
    except Exception:
        return None


def _localization_kpi_cards(summary: dict):
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">🧊 3D Localization Performance (Part 3.3)</div>', unsafe_allow_html=True)

    overall = summary.get("test_overall", None)
    macro   = summary.get("test_macro", None)
    mtp     = summary.get("test_mean_probability", None)
    hit05   = summary.get("test_hit_050", None)

    items = [
        ("Overall Score",         overall,  "#3B82F6"),
        ("Macro Disease Score",   macro,    "#8B5CF6"),
        ("Mean True Probability", mtp,      "#0D9488"),
        ("Hit @ 0.50",            hit05,    "#EC4899"),
    ]

    cols = st.columns(4)
    for col, (label, val, color) in zip(cols, items):
        display = f"{val:.4f}" if val is not None else "N/A"
        with col:
            st.markdown(f"""
            <div style="background:#FFFFFF; border-radius:12px; padding:1.2rem 1.4rem;
                        border:1px solid #E8EDF6; border-top:3px solid {color};
                        box-shadow:0 2px 8px rgba(15,27,61,0.06); text-align:center;">
                <div style="font-size:0.7rem; font-weight:600; color:#6B7FA3;
                            text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;">{label}</div>
                <div style="font-size:1.8rem; font-weight:800; color:#0F1B3D;">{display}</div>
            </div>
            """, unsafe_allow_html=True)

    st.caption(
        f"Untouched test evaluation · {summary.get('test_cases','N/A')} cases · "
        f"{summary.get('test_points','N/A')} annotation points · "
        f"Checkpoint: Part 3.3 Balanced LFNN/RFNN Symmetry Refinement"
    )


def _disease_performance_table(df_dm: pd.DataFrame):
    st.markdown('<div class="section-title" style="margin:1.5rem 0 0.8rem 0;">🦠 Disease-wise Localization Performance</div>', unsafe_allow_html=True)

    rename = {
        "class_name":       "Disease",
        "count":            "Points",
        "accuracy":         "Accuracy",
        "mean_probability": "Mean Prob.",
        "hit_050":          "Hit @ 0.50",
    }
    display_cols = ["Disease", "Points", "Accuracy", "Mean Prob.", "Hit @ 0.50"]

    df = df_dm.rename(columns=rename)
    for col in ["Accuracy", "Mean Prob.", "Hit @ 0.50"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: f"{float(v):.4f}" if v is not None else "N/A")

    st.dataframe(df[display_cols] if all(c in df.columns for c in display_cols) else df,
                 use_container_width=True)


def _disease_performance_chart(df_dm: pd.DataFrame):
    """Horizontal bar chart of localization accuracy per disease."""
    short = {
        "Spinal Canal Stenosis":           "SCS",
        "Left Neural Foraminal Narrowing": "LFNN",
        "Right Neural Foraminal Narrowing":"RFNN",
        "Left Subarticular Stenosis":      "LSS",
        "Right Subarticular Stenosis":     "RSS",
    }

    labels   = [short.get(str(r["class_name"]), str(r["class_name"])) for _, r in df_dm.iterrows()]
    accuracy = [float(r["accuracy"]) for _, r in df_dm.iterrows()]
    hit      = [float(r["hit_050"])  for _, r in df_dm.iterrows()]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFF")

    ax.bar(x - 0.18, accuracy, 0.32, label="Accuracy",    color="#3B82F6", edgecolor="none")
    ax.bar(x + 0.18, hit,      0.32, label="Hit @ 0.50",  color="#EC4899", edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=8, color="#6B7FA3")
    ax.set_title("Disease-wise Localization Performance", fontsize=10, fontweight="700",
                 color="#0F1B3D", pad=6)
    ax.legend(fontsize=7.5, framealpha=0.9)
    ax.tick_params(labelsize=8, colors="#4B5E8A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E8EDF6")
    ax.spines["bottom"].set_color("#E8EDF6")
    plt.tight_layout(pad=1.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Live Classification Result ───────────────────────────────────────────────

def _live_result_section(s):
    """Show live session classification result if available."""
    st.markdown("---")
    st.markdown('<div class="section-title" style="margin-bottom:1rem;">🎯 Current Session Analysis Result</div>', unsafe_allow_html=True)

    prediction  = safe(s.predicted_class, "Unknown")
    confidence  = pct(s.confidence)
    icon        = CLASS_ICONS.get(prediction, "🔵")
    probs       = s.probabilities or {}

    left, right = st.columns([1, 1.4])

    with left:
        sev_class = "severity-normal" if "Normal" in prediction else (
            "severity-moderate" if "Moderate" in prediction else "severity-severe"
        )
        st.markdown(f"""
        <div style="background:#FFFFFF; border-radius:14px; padding:1.5rem;
                    border:1px solid #E8EDF6; box-shadow:0 2px 10px rgba(15,27,61,0.07);">
            <div style="font-size:0.7rem; font-weight:600; color:#6B7FA3; text-transform:uppercase; margin-bottom:0.6rem;">
                CLASSIFICATION RESULT
            </div>
            <div class="{sev_class}" style="margin-bottom:1rem;">{icon} {prediction}</div>
            <div style="font-size:0.82rem; color:#4B5E8A;">
                Confidence: <strong style="color:#0F1B3D;">{confidence:.2f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("**Probability Distribution**")
        prob_colors = {"Normal/Mild": "#34D399", "Moderate": "#F59E0B", "Severe": "#F87171"}
        for cls in CLASS_ORDER:
            val = pct(probs.get(cls, 0.0))
            bar_color = prob_colors.get(cls, "#3B82F6")
            st.markdown(f"""
            <div style="margin:0.4rem 0;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#4B5E8A; margin-bottom:3px;">
                    <span>{CLASS_ICONS.get(cls,'')} {cls}</span><span>{val:.1f}%</span>
                </div>
                <div style="height:8px; background:#F0F4FF; border-radius:4px; overflow:hidden;">
                    <div style="width:{val}%; height:100%; background:{bar_color}; border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🩻 New Analysis", key="results_another", use_container_width=True):
            s.reset()
            navigate("🩻 MRI Analysis")
    with c2:
        if st.button("🧠 3D Localization", key="results_seg", use_container_width=True, type="primary"):
            navigate("🧠 3D Segmentation")
    with c3:
        if st.button("🔬 Explainable AI", key="results_xai", use_container_width=True):
            navigate("🔬 Explainable AI")
    with c4:
        if st.button("📄 Generate Report", key="results_report", use_container_width=True):
            navigate("📄 Report")


# ── Disclaimer ───────────────────────────────────────────────────────────────

def _disclaimer():
    st.markdown("""
    <div class="research-disclaimer">
        <strong>⚠️ Research Interpretation Only</strong>
        Classification metrics are computed on the RSNA 2024 validation split.
        Localization metrics are from the untouched test evaluation (Part 3.3).
        No metrics are fabricated. Values shown are actual project artifacts.
    </div>
    """, unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def render_results():
    _page_header()

    # ── Classification Performance ──────────────────────────────
    metrics   = _load_classification_metrics()
    df_report = _load_classification_report()
    df_cm     = _load_confusion_matrix()

    if metrics:
        _classification_kpi_cards(metrics)
    else:
        st.warning("Classification metrics file not found.")

    col_cm, col_cw = st.columns([1, 1.3], gap="large")
    with col_cm:
        if df_cm is not None:
            _confusion_matrix_plot(df_cm)
        else:
            st.info("Confusion matrix CSV not available.")
    with col_cw:
        if df_report is not None:
            _class_wise_metrics_chart(df_report)
        else:
            st.info("Classification report CSV not available.")

    # ── 3D Localization Performance ─────────────────────────────
    st.markdown("---")
    summary = _load_localization_summary()
    df_dm   = _load_disease_metrics()

    if summary:
        _localization_kpi_cards(summary)
    else:
        st.warning("Localization evaluation summary not found (expects Part 3.3 test evaluation JSON).")

    if df_dm is not None:
        col_chart, col_tbl = st.columns([1.2, 1], gap="large")
        with col_chart:
            _disease_performance_chart(df_dm)
        with col_tbl:
            _disease_performance_table(df_dm)

    # ── Live session result ─────────────────────────────────────
    s = _session()
    if s is not None:
        _live_result_section(s)
    else:
        st.markdown("---")
        st.info("No live analysis session active. Upload a DICOM series in MRI Analysis to see real-time classification results here.")
        if st.button("🩻 Go to MRI Analysis", type="primary", use_container_width=False, key="results_goto_mri"):
            navigate("🩻 MRI Analysis")

    _disclaimer()
