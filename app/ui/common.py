from pathlib import Path
from typing import Any

import streamlit as st


# ============================================================================
# APPLICATION NAVIGATION
# ============================================================================

PAGES = [
    "🏠 Dashboard",
    "🩻 MRI Analysis",
    "🧠 3D Segmentation",
    "📊 Results",
    "🔬 Explainable AI",
    "📄 Report",
    "ℹ️ About Project",
    "⚙️ System & Deployment",
]


# ============================================================================
# CLASSIFICATION MODEL
# ============================================================================

CLASSIFICATION_CLASS_ORDER = [
    "Normal/Mild",
    "Moderate",
    "Severe",
]


CLASSIFICATION_CLASS_ICONS = {
    "Normal/Mild": "🟢",
    "Moderate": "🟡",
    "Severe": "🔴",
}


CLASSIFICATION_MODEL_NAME = "Swin Transformer"

CLASSIFICATION_TASK = "Severity Classification"


# ============================================================================
# SWIN-UNETR DISEASE LOCALIZATION MODEL
# ============================================================================

LOCALIZATION_CLASS_ORDER = [
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]


LOCALIZATION_CLASS_IDS = {
    "Spinal Canal Stenosis": 1,
    "Left Neural Foraminal Narrowing": 2,
    "Right Neural Foraminal Narrowing": 3,
    "Left Subarticular Stenosis": 4,
    "Right Subarticular Stenosis": 5,
}


LOCALIZATION_CLASS_SHORT_NAMES = {
    "Spinal Canal Stenosis": "SCS",
    "Left Neural Foraminal Narrowing": "LFNN",
    "Right Neural Foraminal Narrowing": "RFNN",
    "Left Subarticular Stenosis": "LSS",
    "Right Subarticular Stenosis": "RSS",
}


LOCALIZATION_CLASS_ICONS = {
    "Spinal Canal Stenosis": "🦴",
    "Left Neural Foraminal Narrowing": "◀️",
    "Right Neural Foraminal Narrowing": "▶️",
    "Left Subarticular Stenosis": "🔹",
    "Right Subarticular Stenosis": "🔸",
}


LOCALIZATION_MODEL_NAME = "Swin-UNETR"

LOCALIZATION_TASK = "3D Disease Localization"

LOCALIZATION_NUM_CLASSES = 6

LOCALIZATION_INPUT_SHAPE = "64 × 96 × 96"

LOCALIZATION_PARAMETER_COUNT = 4_078_116

LOCALIZATION_MODEL_VARIANT = (
    "Part 3.3 Balanced LFNN/RFNN Symmetry Refinement"
)


# ============================================================================
# BACKWARD-COMPATIBILITY ALIASES
# ============================================================================
#
# Existing classification pages currently import CLASS_ORDER and CLASS_ICONS.
# Keep these aliases so the existing classification workflow continues to
# work while the dashboard is being migrated to the combined architecture.
#

CLASS_ORDER = CLASSIFICATION_CLASS_ORDER

CLASS_ICONS = CLASSIFICATION_CLASS_ICONS


# ============================================================================
# FINAL MODEL ALIASES
# ============================================================================
#
# These names are used by the newer Swin-UNETR dashboard components.
#

FINAL_MODEL_NAME = LOCALIZATION_MODEL_NAME

FINAL_MODEL_VARIANT = LOCALIZATION_MODEL_VARIANT

FINAL_INPUT_SHAPE = LOCALIZATION_INPUT_SHAPE

FINAL_NUM_CLASSES = LOCALIZATION_NUM_CLASSES

FINAL_PARAMETER_COUNT = LOCALIZATION_PARAMETER_COUNT


# ============================================================================
# CLASS ID → NAME
# ============================================================================

def class_name(class_id: int) -> str:
    """
    Return the Swin-UNETR output class name.

    Class 0:
        Background

    Classes 1–5:
        Lumbar-spine disease localization classes
    """

    names = {
        0: "Background",
        1: "Spinal Canal Stenosis",
        2: "Left Neural Foraminal Narrowing",
        3: "Right Neural Foraminal Narrowing",
        4: "Left Subarticular Stenosis",
        5: "Right Subarticular Stenosis",
    }

    try:
        class_id = int(class_id)
    except (TypeError, ValueError):
        return "Unknown"

    return names.get(class_id, "Unknown")


# ============================================================================
# CLASS ID → SHORT NAME
# ============================================================================

def class_short_name(class_id: int) -> str:
    """
    Return the compact disease abbreviation used in plots and dashboard cards.
    """

    try:
        class_id = int(class_id)
    except (TypeError, ValueError):
        return "Unknown"

    short_names = {
        0: "BG",
        1: "SCS",
        2: "LFNN",
        3: "RFNN",
        4: "LSS",
        5: "RSS",
    }

    return short_names.get(class_id, "Unknown")


# ============================================================================
# NAVIGATION
# ============================================================================

def navigate(page: str) -> None:
    """
    Queue navigation for the next Streamlit rerun.
    """

    if page in PAGES:
        st.session_state.pending_navigation = page
        st.rerun()


# ============================================================================
# SESSION ACCESS
# ============================================================================

def session():
    """
    Return the centralized AnalysisSession instance.
    """

    return st.session_state.get("analysis_session")


# ============================================================================
# PERCENTAGE HELPER
# ============================================================================

def pct(value: Any) -> float:
    """
    Convert either a probability/fraction or percentage into percentage form.

    Examples
    --------
    0.62  -> 62.0
    62.0  -> 62.0
    """

    try:
        value = float(value)

    except (TypeError, ValueError):
        return 0.0

    if value <= 1.0:
        value *= 100.0

    return max(
        0.0,
        min(
            100.0,
            value,
        ),
    )


# ============================================================================
# FILENAME HELPER
# ============================================================================

def filename(
    value: Any,
    fallback: str = "N/A",
) -> str:
    """
    Return only the filename portion of a path-like value.
    """

    if not value:
        return fallback

    try:
        return Path(str(value)).name

    except Exception:
        return str(value)


# ============================================================================
# SAFE DISPLAY HELPER
# ============================================================================

def safe(
    value: Any,
    fallback: str = "N/A",
) -> str:
    """
    Convert a value into a safe non-empty display string.
    """

    if value is None:
        return fallback

    text = str(value).strip()

    return text or fallback


# ============================================================================
# SECTION HELPER
# ============================================================================

def section(
    title: str,
    subtitle: str | None = None,
):
    """
    Render a consistent dashboard section heading.
    """

    st.markdown(
        f"## {title}"
    )

    if subtitle:
        st.caption(subtitle)


# ============================================================================
# STATUS CHIP
# ============================================================================

def status_chip(
    label: str,
    ready: bool,
):
    """
    Display a simple system status indicator.
    """

    if ready:
        st.success(
            f"✓ {label}"
        )

    else:
        st.info(
            f"○ {label}"
        )


# ============================================================================
# CLASSIFICATION MODEL SUMMARY
# ============================================================================

def classification_model_summary() -> dict:
    """
    Return the classification branch configuration for dashboard display.
    """

    return {
        "architecture": CLASSIFICATION_MODEL_NAME,
        "task": CLASSIFICATION_TASK,
        "classes": len(
            CLASSIFICATION_CLASS_ORDER
        ),
        "class_names": list(
            CLASSIFICATION_CLASS_ORDER
        ),
    }


# ============================================================================
# LOCALIZATION MODEL SUMMARY
# ============================================================================

def localization_model_summary() -> dict:
    """
    Return the Swin-UNETR localization branch configuration.
    """

    return {
        "architecture": LOCALIZATION_MODEL_NAME,
        "variant": LOCALIZATION_MODEL_VARIANT,
        "task": LOCALIZATION_TASK,
        "classes": LOCALIZATION_NUM_CLASSES,
        "disease_classes": len(
            LOCALIZATION_CLASS_ORDER
        ),
        "input_shape": LOCALIZATION_INPUT_SHAPE,
        "parameters": LOCALIZATION_PARAMETER_COUNT,
        "class_names": list(
            LOCALIZATION_CLASS_ORDER
        ),
    }


# ============================================================================
# SCIENTIFIC DISCLAIMER
# ============================================================================

def research_disclaimer():
    """
    Display the standard project research limitation.
    """

    st.warning(
        "**Academic Research Prototype — Not for Clinical Diagnosis**\n\n"
        "The dashboard presents model-derived classification and "
        "disease-localization outputs for academic research and project "
        "evaluation. The Swin-UNETR localization outputs are probability/"
        "predicted-class representations derived from point/localization "
        "annotations; they must not be interpreted as clinically validated "
        "voxel-level segmentation masks."
    )