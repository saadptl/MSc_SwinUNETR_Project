# ============================================================
# xai_dashboard.py
# Phase 2 - Part 4 XAI
# Dashboard-ready XAI service for the trained SwinClassifier
#
# READ-ONLY:
#   - Uses existing trained checkpoint
#   - No retraining
#   - No optimizer
#   - No checkpoint modification
#
# IMPORTANT:
# Grad-CAM is presented as a MODEL-ATTENTION VISUALIZATION.
# It is NOT treated as proof of disease location.
#
# Input convention matches the project's training loader:
#   3 adjacent DICOM slices -> channels C,H,W
#   CLAHE -> resize 224x224 -> /255
#
# Outputs:
#   original middle slice
#   heatmap
#   overlay
#   side-by-side XAI panel
#   prediction JSON
# ============================================================

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model.pth"
)

# Use the corrected checkpoint only if the user explicitly
# changes this value. Default is the established trained model.
# No training is performed here.
XAI_MODEL_PATH = MODEL_PATH

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "app"
    / "outputs"
    / "xai_dashboard"
)

HEATMAP_DIR = OUTPUT_ROOT / "heatmaps"
OVERLAY_DIR = OUTPUT_ROOT / "overlays"
PANEL_DIR = OUTPUT_ROOT / "panels"
MIDDLE_DIR = OUTPUT_ROOT / "middle_slices"

for directory in (
    HEATMAP_DIR,
    OVERLAY_DIR,
    PANEL_DIR,
    MIDDLE_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Normal/Mild",
    "Moderate",
    "Severe",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model


# ============================================================
# MODEL
# ============================================================

_model = None


def load_model():
    """
    Load the already-trained model once.

    This function NEVER trains or changes the checkpoint.
    """
    global _model

    if _model is not None:
        return _model

    if not XAI_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{XAI_MODEL_PATH}"
        )

    checkpoint = torch.load(
        XAI_MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = create_model(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=3,
        pretrained=False,
        dropout=0.30,
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    _model = model
    return _model


# ============================================================
# DICOM
# ============================================================

def read_dicom(path: str):
    ds = pydicom.dcmread(str(path))

    image = ds.pixel_array.astype(np.float32)

    # Correct MONOCHROME1 if present.
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    return image


def normalize_and_clahe(image):
    image = image.astype(np.float32)

    image -= image.min()
    denominator = image.max()

    if denominator > 0:
        image /= denominator

    image = np.uint8(image * 255.0)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)

    return image


def preprocess_dicom(path: str, size: int = 224):
    image = read_dicom(path)
    image = normalize_and_clahe(image)

    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    return image


def build_three_slice_input(
    dicom_paths: List[str],
    size: int = 224,
):
    """
    dicom_paths must contain exactly 3 adjacent slices:
        [previous, middle, next]

    Returns:
        tensor: [1,3,224,224]
        display_middle: original/processed middle slice
    """

    if len(dicom_paths) != 3:
        raise ValueError(
            "Dashboard XAI requires exactly 3 adjacent DICOM slices."
        )

    images = [
        preprocess_dicom(path, size)
        for path in dicom_paths
    ]

    array = np.stack(
        images,
        axis=0
    )

    tensor = torch.tensor(
        array,
        dtype=torch.float32
    ) / 255.0

    tensor = tensor.unsqueeze(0)

    return tensor, images[1]


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    tensor,
    target_class: Optional[int] = None,
):
    """
    Generate class-specific Grad-CAM from the Swin feature map.

    Swin output from this project:
        [B,7,7,768]

    The CAM is therefore a 7x7 spatial explanation resized to
    the 224x224 input space.
    """

    model.eval()

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    # Remove old gradient references.
    model.zero_grad(set_to_none=True)

    logits = model(tensor)

    probabilities = F.softmax(
        logits,
        dim=1
    )

    predicted_class = int(
        torch.argmax(
            probabilities,
            dim=1
        ).item()
    )

    if target_class is None:
        target_class = predicted_class

    if not 0 <= int(target_class) < 3:
        raise ValueError("target_class must be 0, 1, or 2.")

    features = model.get_activations()

    if features is None:
        raise RuntimeError(
            "Model did not expose Swin activations."
        )

    # IMPORTANT:
    # retain_grad is required because features are not leaf tensors.
    features.retain_grad()

    score = logits[:, int(target_class)].sum()
    score.backward()

    gradients = features.grad

    if gradients is None:
        raise RuntimeError(
            "Gradients were not captured from the Swin feature map."
        )

    # Expected shape:
    # [1, 7, 7, 768]
    if features.ndim != 4:
        raise RuntimeError(
            f"Unexpected Swin feature shape: {features.shape}"
        )

    # Gradient weights per feature channel.
    weights = gradients.mean(
        dim=(1, 2),
        keepdim=True
    )

    # Weighted feature map.
    cam = (
        features * weights
    ).sum(dim=-1)

    cam = F.relu(cam)

    cam = cam[0].detach().cpu().numpy()

    cam = cv2.resize(
        cam,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    # Robust normalization.
    cam -= cam.min()

    maximum = cam.max()

    if maximum > 1e-8:
        cam /= maximum

    cam = np.clip(
        cam,
        0.0,
        1.0
    ).astype(np.float32)

    return {
        "logits": logits.detach().cpu().numpy()[0],
        "probabilities": probabilities.detach().cpu().numpy()[0],
        "predicted_class": predicted_class,
        "target_class": int(target_class),
        "cam": cam,
        "feature_shape": list(features.shape),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def make_heatmap(cam):
    heatmap = np.uint8(
        np.clip(cam, 0, 1) * 255
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    return heatmap


def make_overlay(middle_gray, cam):
    """
    Overlay CAM on the central/visualization slice.

    The 3-channel model input is preserved for inference.
    The middle slice is used only for human-readable display.
    """

    middle_bgr = cv2.cvtColor(
        middle_gray,
        cv2.COLOR_GRAY2BGR
    )

    heatmap = make_heatmap(cam)

    overlay = cv2.addWeighted(
        middle_bgr,
        0.60,
        heatmap,
        0.40,
        0
    )

    return overlay


def make_panel(
    middle_gray,
    heatmap,
    overlay,
    prediction,
    confidence,
):
    """
    Create one dashboard-friendly 3-panel image.

    Labels intentionally describe CAM as model attention,
    not as disease segmentation.
    """

    original = cv2.cvtColor(
        middle_gray,
        cv2.COLOR_GRAY2BGR
    )

    images = [
        original,
        heatmap,
        overlay,
    ]

    labels = [
        "MRI - Middle Slice",
        "Grad-CAM Heatmap",
        "Model Attention Overlay",
    ]

    panels = []

    for image, label in zip(images, labels):
        panel = image.copy()

        cv2.rectangle(
            panel,
            (0, 0),
            (224, 34),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            panel,
            label,
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        panels.append(panel)

    panel = np.hstack(panels)

    # Footer.
    footer_height = 62

    canvas = np.zeros(
        (
            panel.shape[0] + footer_height,
            panel.shape[1],
            3
        ),
        dtype=np.uint8
    )

    canvas[:panel.shape[0]] = panel

    text_1 = (
        f"Prediction: {prediction}"
        f"   Confidence: {confidence:.2f}%"
    )

    text_2 = (
        "Grad-CAM: model-attention visualization"
        " | Not a disease segmentation"
    )

    cv2.putText(
        canvas,
        text_1,
        (10, panel.shape[0] + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        text_2,
        (10, panel.shape[0] + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return canvas


# ============================================================
# DASHBOARD SERVICE
# ============================================================

def explain_dicom_series(
    dicom_paths: List[str],
    target_class: Optional[int] = None,
    output_name: str = "dashboard_xai",
) -> Dict:
    """
    Main function for the dashboard.

    Example:

        result = explain_dicom_series(
            [
                "7.dcm",
                "8.dcm",
                "9.dcm",
            ],
            output_name="study_4003253"
        )

    Returns a JSON-friendly dictionary.
    """

    model = load_model()

    tensor, middle_slice = build_three_slice_input(
        dicom_paths
    )

    tensor = tensor.to(DEVICE)

    result = generate_gradcam(
        model,
        tensor,
        target_class
    )

    probabilities = result["probabilities"]

    prediction_id = result["predicted_class"]

    prediction_name = CLASS_NAMES[
        prediction_id
    ]

    confidence = float(
        probabilities[prediction_id] * 100.0
    )

    cam = result["cam"]

    heatmap = make_heatmap(cam)

    overlay = make_overlay(
        middle_slice,
        cam
    )

    panel = make_panel(
        middle_slice,
        heatmap,
        overlay,
        prediction_name,
        confidence,
    )

    safe_name = str(output_name)

    middle_path = (
        MIDDLE_DIR
        / f"{safe_name}_middle.png"
    )

    heatmap_path = (
        HEATMAP_DIR
        / f"{safe_name}_gradcam.png"
    )

    overlay_path = (
        OVERLAY_DIR
        / f"{safe_name}_overlay.png"
    )

    panel_path = (
        PANEL_DIR
        / f"{safe_name}_xai_panel.png"
    )

    cv2.imwrite(
        str(middle_path),
        middle_slice
    )

    cv2.imwrite(
        str(heatmap_path),
        heatmap
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )

    cv2.imwrite(
        str(panel_path),
        panel
    )

    # CAM statistics.
    cam_mean = float(cam.mean())
    cam_std = float(cam.std())
    cam_max = float(cam.max())

    result_json = {
        "prediction": {
            "class_id": prediction_id,
            "class_name": prediction_name,
            "confidence_percent": confidence,
            "probabilities_percent": {
                CLASS_NAMES[i]: float(
                    probabilities[i] * 100.0
                )
                for i in range(3)
            },
        },
        "xai": {
            "method": "Grad-CAM",
            "target_class": prediction_name,
            "feature_shape": result["feature_shape"],
            "cam_shape": list(cam.shape),
            "cam_min": float(cam.min()),
            "cam_max": cam_max,
            "cam_mean": cam_mean,
            "cam_std": cam_std,
            "interpretation": (
                "Grad-CAM visualizes image regions contributing "
                "to the model prediction. It should not be "
                "interpreted as a definitive disease localization "
                "or clinical diagnosis."
            ),
        },
        "input": {
            "channels": [
                str(path)
                for path in dicom_paths
            ],
            "visualization_slice": str(
                dicom_paths[1]
            ),
            "tensor_shape": list(
                tensor.shape
            ),
        },
        "outputs": {
            "middle_slice": str(middle_path),
            "heatmap": str(heatmap_path),
            "overlay": str(overlay_path),
            "panel": str(panel_path),
        },
        "model": {
            "checkpoint": str(XAI_MODEL_PATH),
            "device": str(DEVICE),
        },
    }

    return result_json


# ============================================================
# Optional command-line test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PHASE 2 - PART 4 XAI")
    print("DASHBOARD-READY INFERENCE + GRAD-CAM")
    print("=" * 70)

    print("Model:", XAI_MODEL_PATH)
    print("Device:", DEVICE)

    # Change these three paths for a manual test.
    TEST_DICOM = [
        PROJECT_ROOT
        / "dataset"
        / "rsna-2024-lumbar-spine-degenerative-classification"
        / "train_images"
        / "4003253"
        / "702807833"
        / "7.dcm",

        PROJECT_ROOT
        / "dataset"
        / "rsna-2024-lumbar-spine-degenerative-classification"
        / "train_images"
        / "4003253"
        / "702807833"
        / "8.dcm",

        PROJECT_ROOT
        / "dataset"
        / "rsna-2024-lumbar-spine-degenerative-classification"
        / "train_images"
        / "4003253"
        / "702807833"
        / "9.dcm",
    ]

    for path in TEST_DICOM:
        if not path.exists():
            raise FileNotFoundError(
                f"Test DICOM not found:\n{path}"
            )

    output = explain_dicom_series(
        [str(p) for p in TEST_DICOM],
        output_name="phase4_dashboard_test"
    )

    print("\n" + "=" * 70)
    print("PREDICTION")
    print("=" * 70)

    print(
        "Class:",
        output["prediction"]["class_name"]
    )

    print(
        "Confidence:",
        f"{output['prediction']['confidence_percent']:.2f}%"
    )

    print(
        "\nProbabilities:"
    )

    for name, value in output[
        "prediction"
    ][
        "probabilities_percent"
    ].items():
        print(
            f"  {name:<12}: {value:.4f}%"
        )

    print("\nXAI")
    print(
        "Feature shape:",
        output["xai"]["feature_shape"]
    )

    print(
        "CAM mean:",
        output["xai"]["cam_mean"]
    )

    print(
        "CAM std:",
        output["xai"]["cam_std"]
    )

    print("\nGenerated files:")

    for name, path in output[
        "outputs"
    ].items():
        print(
            f"{name:<15}: {path}"
        )

    print("\nPhase 2 - Part 4 XAI completed.")
