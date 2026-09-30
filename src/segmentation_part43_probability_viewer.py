"""
PART 4.3
INTERACTIVE SWIN-UNETR DISEASE-LOCALIZATION VIEWER

Purpose
-------
Interactive inspection of the REAL Part 4.2 Swin-UNETR output.

Displays:
    1. Canonical MRI
    2. Disease probability
    3. Predicted disease class

Supports:
    - Axial
    - Coronal
    - Sagittal
    - Disease selection by number or name
    - Slice navigation
    - Probability threshold

IMPORTANT
---------
This is a disease-localization probability viewer.

It does NOT claim validated voxel-level segmentation because
manual voxel ground truth is unavailable.

No:
    - training
    - checkpoint modification
    - dashboard modification
    - fabricated voxel masks
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CASE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part42_final_dicom_inference"
    / "study_7143189_series_3219733239"
)


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ============================================================================
# LOAD CASE
# ============================================================================

def load_case(case_dir: Path):

    npz_path = (
        case_dir
        / "swinunetr_inference.npz"
    )

    metadata_path = (
        case_dir
        / "inference_metadata.json"
    )

    if not npz_path.exists():

        raise FileNotFoundError(
            "Missing Part 4.2 inference file:\n"
            f"{npz_path}"
        )

    data = np.load(
        npz_path
    )

    required_arrays = [
        "image",
        "probabilities",
        "predicted_class",
    ]

    for name in required_arrays:

        if name not in data:

            raise KeyError(
                f"Required array '{name}' "
                "is missing from the inference file."
            )

    image = data[
        "image"
    ]

    probabilities = data[
        "probabilities"
    ]

    predicted_class = data[
        "predicted_class"
    ]

    metadata = {}

    if metadata_path.exists():

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:

            metadata = json.load(
                f
            )

    return (
        image,
        probabilities,
        predicted_class,
        metadata,
    )


# ============================================================================
# NORMALIZE IMAGE FOR DISPLAY
# ============================================================================

def normalize_display(
    image,
):

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    low = np.percentile(
        image,
        1,
    )

    high = np.percentile(
        image,
        99,
    )

    if high <= low:

        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    normalized = (
        image - low
    ) / (
        high - low
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


# ============================================================================
# GET ORIENTATION SLICE
# ============================================================================

def get_orientation_slice(
    image,
    probabilities,
    predicted_class,
    orientation,
    slice_index,
    disease_class,
):

    if orientation == "Axial":

        mri = image[
            slice_index,
            :,
            :,
        ]

        probability = probabilities[
            disease_class,
            slice_index,
            :,
            :,
        ]

        prediction = predicted_class[
            slice_index,
            :,
            :
        ]

        axis_text = (
            f"Axial z = {slice_index}"
        )

    elif orientation == "Coronal":

        mri = image[
            :,
            slice_index,
            :,
        ]

        probability = probabilities[
            disease_class,
            :,
            slice_index,
            :,
        ]

        prediction = predicted_class[
            :,
            slice_index,
            :
        ]

        axis_text = (
            f"Coronal y = {slice_index}"
        )

    elif orientation == "Sagittal":

        mri = image[
            :,
            :,
            slice_index,
        ]

        probability = probabilities[
            disease_class,
            :,
            :,
            slice_index,
        ]

        prediction = predicted_class[
            :,
            :,
            slice_index,
        ]

        axis_text = (
            f"Sagittal x = {slice_index}"
        )

    else:

        raise ValueError(
            f"Unknown orientation: {orientation}"
        )

    return (
        mri,
        probability,
        prediction,
        axis_text,
    )


# ============================================================================
# DISPLAY VIEW
# ============================================================================

def show_view(
    image,
    probabilities,
    predicted_class,
    orientation,
    slice_index,
    disease_class,
    threshold,
    metadata,
):

    (
        mri,
        probability,
        prediction,
        axis_text,
    ) = get_orientation_slice(
        image,
        probabilities,
        predicted_class,
        orientation,
        slice_index,
        disease_class,
    )

    # ------------------------------------------------------------------------
    # Threshold probability
    # ------------------------------------------------------------------------

    probability_thresholded = np.where(
        probability >= threshold,
        probability,
        np.nan,
    )

    # ------------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------------

    fig = plt.figure(
        figsize=(
            18,
            6,
        )
    )

    # ------------------------------------------------------------------------
    # PANEL 1 — MRI
    # ------------------------------------------------------------------------

    ax1 = fig.add_subplot(
        1,
        3,
        1,
    )

    ax1.imshow(
        mri,
        cmap="gray",
        interpolation="nearest",
    )

    ax1.set_title(
        "MRI\n"
        f"{axis_text}",
        fontsize=13,
    )

    ax1.axis(
        "off"
    )

    # ------------------------------------------------------------------------
    # PANEL 2 — PROBABILITY
    # ------------------------------------------------------------------------

    ax2 = fig.add_subplot(
        1,
        3,
        2,
    )

    ax2.imshow(
        mri,
        cmap="gray",
        interpolation="nearest",
    )

    probability_image = ax2.imshow(
        probability_thresholded,
        cmap="hot",
        alpha=0.55,
        vmin=threshold,
        vmax=1.0,
        interpolation="nearest",
    )

    ax2.set_title(
        "Disease Probability\n"
        f"{CLASS_NAMES[disease_class]}",
        fontsize=13,
    )

    ax2.axis(
        "off"
    )

    colorbar = fig.colorbar(
        probability_image,
        ax=ax2,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Model probability"
    )

    # ------------------------------------------------------------------------
    # PANEL 3 — PREDICTED CLASS
    # ------------------------------------------------------------------------

    ax3 = fig.add_subplot(
        1,
        3,
        3,
    )

    ax3.imshow(
        mri,
        cmap="gray",
        interpolation="nearest",
    )

    class_mask = np.where(
        prediction == disease_class,
        1.0,
        np.nan,
    )

    ax3.imshow(
        class_mask,
        cmap="viridis",
        alpha=0.45,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax3.set_title(
        "Predicted Disease Class\n"
        f"{CLASS_NAMES[disease_class]}",
        fontsize=13,
    )

    ax3.axis(
        "off"
    )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    study_id = metadata.get(
        "study_id",
        "Unknown",
    )

    series_id = metadata.get(
        "series_id",
        "Unknown",
    )

    fig.suptitle(
        "Part 4.3 — Swin-UNETR Disease Localization",
        fontsize=16,
    )

    fig.text(
        0.5,
        0.015,
        (
            f"Study {study_id} | "
            f"Series {series_id} | "
            f"{orientation} | "
            f"Slice {slice_index} | "
            f"Threshold {threshold:.2f} | "
            "Probability visualization — "
            "not validated voxel segmentation"
        ),
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(
        rect=(
            0,
            0.05,
            1,
            0.94,
        )
    )

    plt.show()


# ============================================================================
# GET INTEGER INPUT
# ============================================================================

def get_integer(
    prompt,
    minimum,
    maximum,
    default,
):

    while True:

        value = input(
            prompt
        ).strip()

        if value == "":

            return default

        try:

            number = int(
                value
            )

        except ValueError:

            print(
                "Please enter a number."
            )

            continue

        if (
            number < minimum
            or
            number > maximum
        ):

            print(
                f"Please enter a value "
                f"from {minimum} to {maximum}."
            )

            continue

        return number


# ============================================================================
# GET FLOAT INPUT
# ============================================================================

def get_float(
    prompt,
    minimum,
    maximum,
    default,
):

    while True:

        value = input(
            prompt
        ).strip()

        if value == "":

            return default

        try:

            number = float(
                value
            )

        except ValueError:

            print(
                "Please enter a number."
            )

            continue

        if (
            number < minimum
            or
            number > maximum
        ):

            print(
                f"Please enter a value "
                f"from {minimum:.2f} "
                f"to {maximum:.2f}."
            )

            continue

        return number


# ============================================================================
# DISEASE SELECTION
# ============================================================================

def select_disease():

    print()
    print(
        "Available diseases:"
    )

    for class_id in range(
        1,
        6,
    ):

        print(
            f"{class_id}. "
            f"{CLASS_NAMES[class_id]}"
        )

    print()

    disease_lookup = {}

    for class_id in range(
        1,
        6,
    ):

        disease_name = (
            CLASS_NAMES[
                class_id
            ]
        )

        disease_lookup[
            str(class_id)
        ] = class_id

        disease_lookup[
            disease_name.lower()
        ] = class_id

    while True:

        value = input(
            "Select disease "
            "[1-5 or disease name]: "
        ).strip()

        key = value.lower()

        if key in disease_lookup:

            return disease_lookup[
                key
            ]

        print(
            "Invalid disease. "
            "Please enter 1-5 or the disease name."
        )


# ============================================================================
# ORIENTATION SELECTION
# ============================================================================

def select_orientation():

    print()
    print(
        "Orientations:"
    )

    print(
        "1. Axial"
    )

    print(
        "2. Coronal"
    )

    print(
        "3. Sagittal"
    )

    print()

    orientation_lookup = {
        "1": "Axial",
        "2": "Coronal",
        "3": "Sagittal",
        "axial": "Axial",
        "coronal": "Coronal",
        "sagittal": "Sagittal",
    }

    while True:

        value = input(
            "Select orientation "
            "[1-3 or axial/coronal/sagittal]: "
        ).strip()

        key = value.lower()

        if key in orientation_lookup:

            return orientation_lookup[
                key
            ]

        print(
            "Invalid orientation. "
            "Please enter 1-3 or axial/coronal/sagittal."
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)

    print(
        "PART 4.3"
    )

    print(
        "INTERACTIVE SWIN-UNETR DISEASE-LOCALIZATION VIEWER"
    )

    print("=" * 80)

    print()
    print(
        "No training."
    )

    print(
        "No checkpoint modification."
    )

    print(
        "No fabricated voxel masks."
    )

    print(
        "Voxel-level segmentation claim: DISABLED"
    )

    # ------------------------------------------------------------------------
    # Case directory
    # ------------------------------------------------------------------------

    print()

    case_input = input(
        "Enter case directory "
        "(press Enter for the current Part 4.2 case): "
    ).strip()

    if case_input:

        case_dir = Path(
            case_input
        )

        if not case_dir.is_absolute():

            case_dir = (
                PROJECT_ROOT
                / case_dir
            )

    else:

        case_dir = (
            DEFAULT_CASE_DIR
        )

    case_dir = case_dir.resolve()

    print()

    print(
        "Case directory:"
    )

    print(
        case_dir
    )

    if not case_dir.exists():

        raise FileNotFoundError(
            f"Case directory does not exist:\n"
            f"{case_dir}"
        )

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    (
        image,
        probabilities,
        predicted_class,
        metadata,
    ) = load_case(
        case_dir
    )

    print()

    print(
        f"MRI shape: "
        f"{image.shape}"
    )

    print(
        f"Probability shape: "
        f"{probabilities.shape}"
    )

    print(
        f"Prediction shape: "
        f"{predicted_class.shape}"
    )

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    if image.ndim != 3:

        raise RuntimeError(
            "Expected MRI shape "
            "(D,H,W), received "
            f"{image.shape}"
        )

    if probabilities.ndim != 4:

        raise RuntimeError(
            "Expected probability shape "
            "(C,D,H,W), received "
            f"{probabilities.shape}"
        )

    if probabilities.shape[0] != 6:

        raise RuntimeError(
            "Expected exactly 6 classes, "
            f"received {probabilities.shape[0]}"
        )

    if predicted_class.ndim != 3:

        raise RuntimeError(
            "Expected predicted class shape "
            "(D,H,W), received "
            f"{predicted_class.shape}"
        )

    if image.shape != predicted_class.shape:

        raise RuntimeError(
            "MRI and prediction shapes do not match."
        )

    if probabilities.shape[1:] != image.shape:

        raise RuntimeError(
            "Probability and MRI shapes do not match."
        )

    # ------------------------------------------------------------------------
    # Normalize display
    # ------------------------------------------------------------------------

    image = normalize_display(
        image
    )

    depth, height, width = (
        image.shape
    )

    # ------------------------------------------------------------------------
    # Case information
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "CASE INFORMATION"
    )

    print("=" * 80)

    print(
        f"Study ID: "
        f"{metadata.get('study_id', 'Unknown')}"
    )

    print(
        f"Series ID: "
        f"{metadata.get('series_id', 'Unknown')}"
    )

    print(
        f"Canonical volume: "
        f"{depth} × {height} × {width}"
    )

    # ------------------------------------------------------------------------
    # Disease
    # ------------------------------------------------------------------------

    disease_class = select_disease()

    # ------------------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------------------

    orientation = select_orientation()

    # ------------------------------------------------------------------------
    # Slice range
    # ------------------------------------------------------------------------

    if orientation == "Axial":

        maximum_slice = (
            depth - 1
        )

    elif orientation == "Coronal":

        maximum_slice = (
            height - 1
        )

    else:

        maximum_slice = (
            width - 1
        )

    default_slice = (
        maximum_slice // 2
    )

    slice_index = get_integer(
        (
            f"\nSlice index "
            f"[0-{maximum_slice}] "
            f"(default {default_slice}): "
        ),
        0,
        maximum_slice,
        default_slice,
    )

    # ------------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------------

    threshold = get_float(
        (
            "\nProbability threshold "
            "(0.00-1.00, default 0.50): "
        ),
        0.0,
        1.0,
        0.50,
    )

    # ------------------------------------------------------------------------
    # Show
    # ------------------------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "DISPLAY SETTINGS"
    )

    print(
        "=" * 80
    )

    print(
        f"Disease: "
        f"{CLASS_NAMES[disease_class]}"
    )

    print(
        f"Orientation: "
        f"{orientation}"
    )

    print(
        f"Slice: "
        f"{slice_index}"
    )

    print(
        f"Threshold: "
        f"{threshold:.2f}"
    )

    # ------------------------------------------------------------------------
    # View
    # ------------------------------------------------------------------------

    show_view(
        image,
        probabilities,
        predicted_class,
        orientation,
        slice_index,
        disease_class,
        threshold,
        metadata,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "PART 4.3 VIEW COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "The displayed probability map is a "
        "model disease-localization output."
    )

    print(
        "It is NOT a validated voxel-level segmentation."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()