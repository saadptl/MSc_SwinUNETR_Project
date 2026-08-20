from pathlib import Path

# ============================================================
# Dashboard configuration
# Phase 3 - Part 1
# ============================================================

APP_NAME = "Lumbar Spine AI"
APP_VERSION = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.pth"

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES_DIR = DATASET_DIR / "train_images"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dashboard"
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "dashboard" / "uploads"
XAI_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dashboard" / "xai"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dashboard" / "reports"

CLASS_NAMES = [
    "Normal/Mild",
    "Moderate",
    "Severe",
]

INPUT_SIZE = 224
INPUT_CHANNELS = 3

PROJECT_TITLE = (
    "Explainable Swin-UNETR Framework for "
    "Automated Lumbar Spine Disease Detection "
    "and Classification from MRI Images"
)

SHORT_TITLE = "Automated Lumbar Spine Disease Detection"

DISCLAIMER = (
    "This application is an academic research prototype. "
    "Predictions and Grad-CAM visualizations are not a substitute "
    "for professional medical diagnosis."
)

for directory in (
    OUTPUT_DIR,
    UPLOAD_DIR,
    XAI_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
