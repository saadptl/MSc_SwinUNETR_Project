"""
PART 4.4
PHYSICAL-SPACE SWIN-UNETR MPR VIEWER

Purpose
-------
Professional physical-space visualization of the final
Part 3.3 Swin-UNETR disease-localization output.

Pipeline
--------
RSNA DICOM series
        |
        v
Part 2.20B physical-space geometry
        |
        v
Canonical MRI: 64 x 96 x 96
        |
        +----------------------+
        |                      |
        v                      v
MRI canonical volume    Part 3.3 probability volume
        |                      |
        +----------+-----------+
                   |
                   v
        Physical-space MPR
       /        |         \
   Axial     Coronal    Sagittal

Important
---------
This is a physical-space disease-localization viewer.

The model was trained with point supervision, not manual
voxel segmentation masks.

Therefore:
    - probability maps are displayed as model localization
    - no validated voxel Dice claim is made
    - no fabricated masks are created
    - no checkpoint is modified
    - no dashboard is modified

The canonical volume is displayed using the physical extents
provided by the Part 2.20B geometry pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# PROJECT
# ============================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


DEFAULT_CASE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part42_final_dicom_inference"
    / "study_7143189_series_3219733239"
)


# ============================================================================
# DISEASE CLASSES
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
# LOAD PART 4.2 OUTPUT
# ============================================================================

def load_inference_case(
    case_dir: Path,
):

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
            "Part 4.2 inference output not found:\n"
            f"{npz_path}"
        )

    data = np.load(
        npz_path
    )

    image = data[
        "image"
    ].astype(
        np.float32
    )

    probabilities = data[
        "probabilities"
    ].astype(
        np.float32
    )

    predicted_class = data[
        "predicted_class"
    ].astype(
        np.uint8
    )

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
# LOAD PHYSICAL GEOMETRY
# ============================================================================

def load_geometry(
    study_id,
    series_id,
):

    print()
    print(
        "Loading Part 2.20B physical-space geometry..."
    )

    import segmentation_rsna_part220b_geometry_corrected_training as part220b

    manifest = (
        part220b.load_manifest()
    )

    study_id = str(
        study_id
    )

    series_id = str(
        series_id
    )

    point_df = manifest[
        (
            manifest["study_id"]
            .astype(str)
            == study_id
        )
        &
        (
            manifest["series_id"]
            .astype(str)
            == series_id
        )
    ].copy()

    if point_df.empty:

        raise ValueError(
            "Study/series not found in Part 2.20B manifest:\n"
            f"study_id={study_id}\n"
            f"series_id={series_id}"
        )

    result = part220b.load_case(
        study_id,
        series_id,
        point_df,
    )

    if len(result) == 3:

        _image, _points, geometry = result

    elif len(result) == 4:

        _image, _points, geometry, _extra = result

    else:

        raise RuntimeError(
            "Unexpected Part 2.20B load_case return length: "
            f"{len(result)}"
        )

    return geometry


# ============================================================================
# GEOMETRY HELPERS
# ============================================================================

def get_geometry_value(
    geometry,
    key,
    default=None,
):

    if isinstance(
        geometry,
        dict,
    ):

        return geometry.get(
            key,
            default,
        )

    if hasattr(
        geometry,
        key,
    ):

        return getattr(
            geometry,
            key,
            default,
        )

    return default


def as_array(
    value,
    dtype=np.float64,
):

    if value is None:

        return None

    return np.asarray(
        value,
        dtype=dtype,
    )


# ============================================================================
# BUILD PHYSICAL EXTENTS
# ============================================================================

def build_physical_extents(
    geometry,
    image_shape,
):

    """
    Returns physical coordinate extents for the canonical volume.

    Expected geometry information:
        patient_min
        patient_max

    If patient_min/max are unavailable, the function falls back
    to canonical voxel coordinates.
    """

    patient_min = get_geometry_value(
        geometry,
        "patient_min",
    )

    patient_max = get_geometry_value(
        geometry,
        "patient_max",
    )

    patient_min = as_array(
        patient_min
    )

    patient_max = as_array(
        patient_max
    )

    depth, height, width = (
        image_shape
    )

    if (
        patient_min is not None
        and
        patient_max is not None
        and
        patient_min.size >= 3
        and
        patient_max.size >= 3
    ):

        xmin = float(
            min(
                patient_min[0],
                patient_max[0],
            )
        )

        xmax = float(
            max(
                patient_min[0],
                patient_max[0],
            )
        )

        ymin = float(
            min(
                patient_min[1],
                patient_max[1],
            )
        )

        ymax = float(
            max(
                patient_min[1],
                patient_max[1],
            )
        )

        zmin = float(
            min(
                patient_min[2],
                patient_max[2],
            )
        )

        zmax = float(
            max(
                patient_min[2],
                patient_max[2],
            )
        )

        return {
            "x": (
                xmin,
                xmax,
            ),
            "y": (
                ymin,
                ymax,
            ),
            "z": (
                zmin,
                zmax,
            ),
        }

    # ------------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------------

    print(
        "WARNING: patient_min/patient_max unavailable."
    )

    print(
        "Using canonical voxel coordinates as fallback."
    )

    return {
        "x": (
            0.0,
            float(width - 1),
        ),
        "y": (
            0.0,
            float(height - 1),
        ),
        "z": (
            0.0,
            float(depth - 1),
        ),
    }


# ============================================================================
# NORMALIZE IMAGE
# ============================================================================

def normalize_image(
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
            image
        )

    image = (
        image - low
    ) / (
        high - low
    )

    return np.clip(
        image,
        0.0,
        1.0,
    )


# ============================================================================
# GET PHYSICAL MPR SLICE
# ============================================================================

def get_mpr_slice(
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
            :
        ]

        probability = probabilities[
            disease_class,
            slice_index,
            :,
            :
        ]

        prediction = predicted_class[
            slice_index,
            :,
            :
        ]

    elif orientation == "Coronal":

        mri = image[
            :,
            slice_index,
            :
        ]

        probability = probabilities[
            disease_class,
            :,
            slice_index,
            :
        ]

        prediction = predicted_class[
            :,
            slice_index,
            :
        ]

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

    else:

        raise ValueError(
            f"Unknown orientation: {orientation}"
        )

    return (
        mri,
        probability,
        prediction,
    )


# ============================================================================
# DISPLAY EXTENT
# ============================================================================

def get_extent(
    orientation,
    physical_extent,
):

    x_min, x_max = (
        physical_extent["x"]
    )

    y_min, y_max = (
        physical_extent["y"]
    )

    z_min, z_max = (
        physical_extent["z"]
    )

    if orientation == "Axial":

        return (
            x_min,
            x_max,
            y_min,
            y_max,
        )

    if orientation == "Coronal":

        return (
            x_min,
            x_max,
            z_min,
            z_max,
        )

    if orientation == "Sagittal":

        return (
            y_min,
            y_max,
            z_min,
            z_max,
        )

    raise ValueError(
        f"Unknown orientation: {orientation}"
    )


# ============================================================================
# AXIS LABELS
# ============================================================================

def get_axis_labels(
    orientation,
):

    if orientation == "Axial":

        return (
            "Patient X (mm)",
            "Patient Y (mm)",
        )

    if orientation == "Coronal":

        return (
            "Patient X (mm)",
            "Patient Z (mm)",
        )

    if orientation == "Sagittal":

        return (
            "Patient Y (mm)",
            "Patient Z (mm)",
        )

    raise ValueError(
        f"Unknown orientation: {orientation}"
    )


# ============================================================================
# CREATE VIEW
# ============================================================================

def create_view(
    image,
    probabilities,
    predicted_class,
    physical_extent,
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
    ) = get_mpr_slice(
        image,
        probabilities,
        predicted_class,
        orientation,
        slice_index,
        disease_class,
    )

    probability_display = np.where(
        probability >= threshold,
        probability,
        np.nan,
    )

    extent = get_extent(
        orientation,
        physical_extent,
    )

    xlabel, ylabel = (
        get_axis_labels(
            orientation
        )
    )

    # ------------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------------

    fig = plt.figure(
        figsize=(
            19,
            7,
        )
    )

    # ------------------------------------------------------------------------
    # MRI
    # ------------------------------------------------------------------------

    ax1 = fig.add_subplot(
        1,
        3,
        1,
    )

    ax1.imshow(
        mri,
        cmap="gray",
        extent=extent,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )

    ax1.set_title(
        f"Physical MRI\n"
        f"{orientation}",
        fontsize=13,
    )

    ax1.set_xlabel(
        xlabel
    )

    ax1.set_ylabel(
        ylabel
    )

    # ------------------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------------------

    ax2 = fig.add_subplot(
        1,
        3,
        2,
    )

    ax2.imshow(
        mri,
        cmap="gray",
        extent=extent,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )

    probability_image = ax2.imshow(
        probability_display,
        cmap="hot",
        extent=extent,
        origin="lower",
        aspect="equal",
        alpha=0.55,
        vmin=threshold,
        vmax=1.0,
        interpolation="nearest",
    )

    ax2.set_title(
        "Disease Localization Probability\n"
        f"{CLASS_NAMES[disease_class]}",
        fontsize=13,
    )

    ax2.set_xlabel(
        xlabel
    )

    ax2.set_ylabel(
        ylabel
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
    # Predicted class
    # ------------------------------------------------------------------------

    ax3 = fig.add_subplot(
        1,
        3,
        3,
    )

    ax3.imshow(
        mri,
        cmap="gray",
        extent=extent,
        origin="lower",
        aspect="equal",
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
        extent=extent,
        origin="lower",
        aspect="equal",
        alpha=0.45,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax3.set_title(
        "Predicted Disease Class\n"
        f"{CLASS_NAMES[disease_class]}",
        fontsize=13,
    )

    ax3.set_xlabel(
        xlabel
    )

    ax3.set_ylabel(
        ylabel
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
        "Part 4.4 — Physical-Space Swin-UNETR Disease Localization",
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
            "Physical-space probability visualization — "
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

    return fig


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

    lookup = {}

    for class_id in range(
        1,
        6,
    ):

        lookup[
            str(class_id)
        ] = class_id

        lookup[
            CLASS_NAMES[
                class_id
            ].lower()
        ] = class_id

    while True:

        value = input(
            "\nSelect disease "
            "[1-5 or disease name]: "
        ).strip().lower()

        if value in lookup:

            return lookup[
                value
            ]

        print(
            "Invalid disease selection."
        )


# ============================================================================
# ORIENTATION
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

    lookup = {
        "1": "Axial",
        "2": "Coronal",
        "3": "Sagittal",
        "axial": "Axial",
        "coronal": "Coronal",
        "sagittal": "Sagittal",
    }

    while True:

        value = input(
            "\nSelect orientation "
            "[1-3 or axial/coronal/sagittal]: "
        ).strip().lower()

        if value in lookup:

            return lookup[
                value
            ]

        print(
            "Invalid orientation selection."
        )


# ============================================================================
# INTEGER INPUT
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

            value = int(
                value
            )

        except ValueError:

            print(
                "Please enter an integer."
            )

            continue

        if (
            value < minimum
            or
            value > maximum
        ):

            print(
                f"Please enter a value "
                f"between {minimum} and {maximum}."
            )

            continue

        return value


# ============================================================================
# FLOAT INPUT
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

            value = float(
                value
            )

        except ValueError:

            print(
                "Please enter a numeric value."
            )

            continue

        if (
            value < minimum
            or
            value > maximum
        ):

            print(
                f"Please enter a value "
                f"between {minimum:.2f} and "
                f"{maximum:.2f}."
            )

            continue

        return value


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)

    print(
        "PART 4.4"
    )

    print(
        "PHYSICAL-SPACE SWIN-UNETR MPR VIEWER"
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

    case_input = input(
        "\nEnter Part 4.2 case directory "
        "(press Enter for default): "
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

    if not case_dir.exists():

        raise FileNotFoundError(
            f"Case directory does not exist:\n"
            f"{case_dir}"
        )

    print()
    print(
        "Case directory:"
    )

    print(
        case_dir
    )

    # ------------------------------------------------------------------------
    # Load inference
    # ------------------------------------------------------------------------

    (
        image,
        probabilities,
        predicted_class,
        metadata,
    ) = load_inference_case(
        case_dir
    )

    # ------------------------------------------------------------------------
    # Validate arrays
    # ------------------------------------------------------------------------

    if image.ndim != 3:

        raise RuntimeError(
            f"Expected image shape (D,H,W), "
            f"received {image.shape}"
        )

    if probabilities.ndim != 4:

        raise RuntimeError(
            f"Expected probability shape (C,D,H,W), "
            f"received {probabilities.shape}"
        )

    if probabilities.shape[0] != 6:

        raise RuntimeError(
            "Expected six model classes."
        )

    if probabilities.shape[1:] != image.shape:

        raise RuntimeError(
            "Probability volume does not match "
            "MRI volume."
        )

    if predicted_class.shape != image.shape:

        raise RuntimeError(
            "Predicted class volume does not match "
            "MRI volume."
        )

    # ------------------------------------------------------------------------
    # Normalize image
    # ------------------------------------------------------------------------

    image = normalize_image(
        image
    )

    depth, height, width = (
        image.shape
    )

    print()
    print(
        f"Canonical volume: "
        f"{depth} × {height} × {width}"
    )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    study_id = metadata.get(
        "study_id"
    )

    series_id = metadata.get(
        "series_id"
    )

    if study_id is None:

        raise RuntimeError(
            "study_id missing from Part 4.2 metadata."
        )

    if series_id is None:

        raise RuntimeError(
            "series_id missing from Part 4.2 metadata."
        )

    print(
        f"Study ID: {study_id}"
    )

    print(
        f"Series ID: {series_id}"
    )

    # ------------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------------

    geometry = load_geometry(
        study_id,
        series_id,
    )

    physical_extent = (
        build_physical_extents(
            geometry,
            image.shape,
        )
    )

    print()
    print(
        "Physical extents:"
    )

    print(
        f"X: {physical_extent['x']}"
    )

    print(
        f"Y: {physical_extent['y']}"
    )

    print(
        f"Z: {physical_extent['z']}"
    )

    # ------------------------------------------------------------------------
    # Disease
    # ------------------------------------------------------------------------

    disease_class = (
        select_disease()
    )

    # ------------------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------------------

    orientation = (
        select_orientation()
    )

    # ------------------------------------------------------------------------
    # Slice
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
    # Print configuration
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "PHYSICAL MPR SETTINGS"
    )

    print("=" * 80)

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
    # Display
    # ------------------------------------------------------------------------

    fig = create_view(
        image,
        probabilities,
        predicted_class,
        physical_extent,
        orientation,
        slice_index,
        disease_class,
        threshold,
        metadata,
    )

    plt.show()

    # ------------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "PART 4.4 COMPLETE"
    )

    print("=" * 80)

    print(
        "Physical-space MPR displayed successfully."
    )

    print(
        "The probability layer represents "
        "model disease localization."
    )

    print(
        "It is NOT validated voxel-level segmentation."
    )

    print(
        "Dashboard modified: NO"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()