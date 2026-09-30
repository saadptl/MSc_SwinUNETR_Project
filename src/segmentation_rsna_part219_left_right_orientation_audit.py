"""
PART 2.19
Anatomical Left/Right Spatial Orientation Audit

Purpose
-------
Investigate the persistent confusion between:

    Class 2 = Left Neural Foraminal Narrowing
    Class 3 = Right Neural Foraminal Narrowing

This is an AUDIT ONLY.

It does NOT:
    - train a model
    - modify any checkpoint
    - modify Part104
    - modify Part216
    - modify Part218
    - create voxel-wise ground truth
    - change the dashboard

The audit examines:

    1. RSNA annotation coordinates
    2. Native DICOM geometry
    3. ImageOrientationPatient
    4. ImagePositionPatient
    5. PixelSpacing
    6. Slice direction / normal
    7. Native x/y coordinate distributions
    8. Model-grid coordinate distributions
    9. Left-vs-right coordinate separation
   10. DICOM physical-space projection
   11. Qualitative visualizations

Scientific limitation
---------------------
RSNA provides point/localization annotations rather than
manual voxel-wise segmentation masks.

This audit therefore evaluates coordinate and geometry
consistency only.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# PROJECT PATHS
# ================================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part213_point_supervision_manifest"
    / "part213_point_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part219_left_right_orientation_audit"
)

VIS_DIR = OUTPUT_DIR / "visualizations"

REPORT_DIR = OUTPUT_DIR / "reports"

for directory in [
    OUTPUT_DIR,
    VIS_DIR,
    REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

NUM_VALIDATION_SERIES = 25

MODEL_SHAPE = (
    64,
    96,
    96,
)

LEFT_FORAMINAL = 2

RIGHT_FORAMINAL = 3

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(SEED)

np.random.seed(SEED)


# ================================================================
# IMPORT ROBUST PART11 DICOM LOADER
# ================================================================

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from segmentation_rsna_part11_controlled_pilot_training_corrected import (
    read_dicom_series_robust,
    resolve_series_dir,
)


# ================================================================
# MANIFEST
# ================================================================

def load_manifest():

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Part 2.13 manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    df = pd.read_csv(
        MANIFEST_PATH
    )

    df["study_id"] = (
        df["study_id"]
        .astype(str)
    )

    df["series_id"] = (
        df["series_id"]
        .astype(str)
    )

    df["class_id"] = (
        df["class_id"]
        .astype(int)
    )

    for column in [
        "native_z",
        "native_y",
        "native_x",
        "model_z_float",
        "model_y_float",
        "model_x_float",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# ================================================================
# RECREATE THE SAME VALIDATION SERIES USED IN PART 2.17
# ================================================================

def get_validation_series(
    manifest_df,
):

    studies = sorted(
        manifest_df[
            "study_id"
        ].unique()
    )

    rng = random.Random(
        SEED
    )

    studies = list(
        studies
    )

    rng.shuffle(
        studies
    )

    split_index = int(
        len(studies) * 0.80
    )

    train_studies = set(
        studies[:split_index]
    )

    validation_studies = set(
        studies[split_index:]
    )

    validation_df = manifest_df[
        manifest_df[
            "study_id"
        ].isin(
            validation_studies
        )
    ].copy()

    validation_series = (
        validation_df[
            [
                "study_id",
                "series_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "study_id",
                "series_id",
            ]
        )
        .head(
            NUM_VALIDATION_SERIES
        )
    )

    overlap = (
        set(
            validation_series[
                "study_id"
            ]
        )
        &
        train_studies
    )

    if overlap:

        raise RuntimeError(
            "Study leakage detected."
        )

    return (
        validation_series,
        train_studies,
        validation_studies,
    )


# ================================================================
# DICOM GEOMETRY HELPERS
# ================================================================

def safe_float_array(
    value,
):

    if value is None:

        return None

    try:

        array = np.asarray(
            value,
            dtype=np.float64,
        )

        return array

    except Exception:

        return None


def normalize_vector(
    vector,
):

    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        vector
    )

    if norm < 1e-12:

        return vector

    return vector / norm


def calculate_normal(
    row_direction,
    column_direction,
):

    row_direction = normalize_vector(
        row_direction
    )

    column_direction = normalize_vector(
        column_direction
    )

    normal = np.cross(
        row_direction,
        column_direction,
    )

    return normalize_vector(
        normal
    )


def vector_angle_degrees(
    a,
    b,
):

    a = normalize_vector(a)
    b = normalize_vector(b)

    value = np.clip(
        np.dot(a, b),
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                value
            )
        )
    )


def extract_geometry(
    records,
):

    if not records:

        return {
            "geometry_valid":
                False
        }

    first = records[0]

    iop = getattr(
        first,
        "ImageOrientationPatient",
        None,
    )

    if iop is None:

        return {
            "geometry_valid":
                False,

            "reason":
                "ImageOrientationPatient missing",
        }

    if len(iop) != 6:

        return {
            "geometry_valid":
                False,

            "reason":
                "ImageOrientationPatient length != 6",
        }

    row_direction = np.asarray(
        iop[:3],
        dtype=np.float64,
    )

    column_direction = np.asarray(
        iop[3:6],
        dtype=np.float64,
    )

    row_direction = normalize_vector(
        row_direction
    )

    column_direction = normalize_vector(
        column_direction
    )

    normal = calculate_normal(
        row_direction,
        column_direction,
    )

    positions = []

    instance_numbers = []

    pixel_spacing = getattr(
        first,
        "PixelSpacing",
        None,
    )

    if pixel_spacing is not None:

        try:

            pixel_spacing = [
                float(pixel_spacing[0]),
                float(pixel_spacing[1]),
            ]

        except Exception:

            pixel_spacing = None

    else:

        pixel_spacing = None

    for ds in records:

        position = getattr(
            ds,
            "ImagePositionPatient",
            None,
        )

        if position is not None:

            try:

                positions.append(
                    np.asarray(
                        position,
                        dtype=np.float64,
                    )
                )

            except Exception:

                pass

        instance = getattr(
            ds,
            "InstanceNumber",
            None,
        )

        try:

            instance_numbers.append(
                int(instance)
            )

        except Exception:

            instance_numbers.append(
                None
            )

    if positions:

        positions_array = np.stack(
            positions
        )

    else:

        positions_array = np.empty(
            (0, 3),
            dtype=np.float64,
        )

    # Projection of slice positions onto normal.
    if len(positions_array) >= 2:

        slice_projection = (
            positions_array
            @
            normal
        )

        order = np.argsort(
            slice_projection
        )

        sorted_projection = (
            slice_projection[
                order
            ]
        )

        differences = np.diff(
            sorted_projection
        )

        nonzero_differences = (
            differences[
                np.abs(
                    differences
                )
                >
                1e-6
            ]
        )

        if len(
            nonzero_differences
        ):

            median_slice_spacing = (
                float(
                    np.median(
                        np.abs(
                            nonzero_differences
                        )
                    )
                )
            )

        else:

            median_slice_spacing = 0.0

        first_projection = float(
            sorted_projection[0]
        )

        last_projection = float(
            sorted_projection[-1]
        )

        physical_extent = abs(
            last_projection
            -
            first_projection
        )

    else:

        slice_projection = np.array(
            []
        )

        median_slice_spacing = None

        first_projection = None

        last_projection = None

        physical_extent = None

    # Estimate normal direction from first-to-last slice.
    if (
        len(positions_array)
        >= 2
    ):

        first_position = (
            positions_array[0]
        )

        last_position = (
            positions_array[-1]
        )

        observed_direction = (
            last_position
            -
            first_position
        )

        observed_direction = normalize_vector(
            observed_direction
        )

        normal_alignment = float(
            np.dot(
                observed_direction,
                normal,
            )
        )

        normal_angle = (
            vector_angle_degrees(
                observed_direction,
                normal,
            )
        )

    else:

        observed_direction = None

        normal_alignment = None

        normal_angle = None

    return {
        "geometry_valid":
            True,

        "row_direction":
            row_direction,

        "column_direction":
            column_direction,

        "normal":
            normal,

        "pixel_spacing":
            pixel_spacing,

        "positions":
            positions_array,

        "instance_numbers":
            instance_numbers,

        "slice_projection":
            slice_projection,

        "median_slice_spacing":
            median_slice_spacing,

        "physical_slice_extent":
            physical_extent,

        "observed_slice_direction":
            observed_direction,

        "normal_alignment":
            normal_alignment,

        "normal_angle":
            normal_angle,
    }


# ================================================================
# POINT PHYSICAL COORDINATE
# ================================================================

def native_pixel_to_patient(
    ds,
    row,
    column,
):

    """
    DICOM patient-coordinate calculation for a pixel.

    DICOM convention:

        ImagePositionPatient
        + row * row_spacing * row_direction
        + column * column_spacing * column_direction
    """

    position = getattr(
        ds,
        "ImagePositionPatient",
        None,
    )

    iop = getattr(
        ds,
        "ImageOrientationPatient",
        None,
    )

    spacing = getattr(
        ds,
        "PixelSpacing",
        None,
    )

    if (
        position is None
        or iop is None
        or spacing is None
    ):

        return None

    try:

        position = np.asarray(
            position,
            dtype=np.float64,
        )

        row_direction = normalize_vector(
            np.asarray(
                iop[:3],
                dtype=np.float64,
            )
        )

        column_direction = normalize_vector(
            np.asarray(
                iop[3:6],
                dtype=np.float64,
            )
        )

        row_spacing = float(
            spacing[0]
        )

        column_spacing = float(
            spacing[1]
        )

        patient_position = (
            position
            +
            row
            * row_spacing
            * row_direction
            +
            column
            * column_spacing
            * column_direction
        )

        return patient_position

    except Exception:

        return None


# ================================================================
# FIND CORRESPONDING DICOM SLICE
# ================================================================

def find_slice_for_native_z(
    records,
    native_z,
):

    if not records:

        return None

    z_index = int(
        round(
            float(native_z)
        )
    )

    z_index = max(
        0,
        min(
            z_index,
            len(records) - 1,
        ),
    )

    return records[
        z_index
    ]


# ================================================================
# BUILD POINT AUDIT
# ================================================================

def build_point_audit(
    case_df,
    records,
    geometry,
):

    rows = []

    if not geometry.get(
        "geometry_valid",
        False,
    ):

        return rows

    row_direction = (
        geometry[
            "row_direction"
        ]
    )

    column_direction = (
        geometry[
            "column_direction"
        ]
    )

    normal = (
        geometry[
            "normal"
        ]
    )

    positions = (
        geometry[
            "positions"
        ]
    )

    for _, point in case_df.iterrows():

        class_id = int(
            point[
                "class_id"
            ]
        )

        if class_id not in [
            LEFT_FORAMINAL,
            RIGHT_FORAMINAL,
        ]:

            continue

        native_z = float(
            point[
                "native_z"
            ]
        )

        native_y = float(
            point[
                "native_y"
            ]
        )

        native_x = float(
            point[
                "native_x"
            ]
        )

        model_z = float(
            point[
                "model_z_float"
            ]
        )

        model_y = float(
            point[
                "model_y_float"
            ]
        )

        model_x = float(
            point[
                "model_x_float"
            ]
        )

        ds = find_slice_for_native_z(
            records,
            native_z,
        )

        patient_coordinate = None

        if ds is not None:

            patient_coordinate = (
                native_pixel_to_patient(
                    ds,
                    native_y,
                    native_x,
                )
            )

        if (
            patient_coordinate
            is not None
        ):

            patient_x = float(
                patient_coordinate[0]
            )

            patient_y = float(
                patient_coordinate[1]
            )

            patient_z = float(
                patient_coordinate[2]
            )

            row_projection = float(
                np.dot(
                    patient_coordinate,
                    row_direction,
                )
            )

            column_projection = float(
                np.dot(
                    patient_coordinate,
                    column_direction,
                )
            )

            normal_projection = float(
                np.dot(
                    patient_coordinate,
                    normal,
                )
            )

        else:

            patient_x = np.nan
            patient_y = np.nan
            patient_z = np.nan

            row_projection = np.nan
            column_projection = np.nan
            normal_projection = np.nan

        rows.append(
            {
                "study_id":
                    str(
                        point[
                            "study_id"
                        ]
                    ),

                "series_id":
                    str(
                        point[
                            "series_id"
                        ]
                    ),

                "condition":
                    str(
                        point[
                            "class_name"
                        ]
                    ),

                "class_id":
                    class_id,

                "level":
                    str(
                        point[
                            "level"
                        ]
                    ),

                "native_z":
                    native_z,

                "native_y":
                    native_y,

                "native_x":
                    native_x,

                "model_z":
                    model_z,

                "model_y":
                    model_y,

                "model_x":
                    model_x,

                "patient_x":
                    patient_x,

                "patient_y":
                    patient_y,

                "patient_z":
                    patient_z,

                "row_projection":
                    row_projection,

                "column_projection":
                    column_projection,

                "normal_projection":
                    normal_projection,
            }
        )

    return rows


# ================================================================
# ORIENTATION SUMMARY
# ================================================================

def orientation_summary(
    study_id,
    series_id,
    records,
):

    geometry = extract_geometry(
        records
    )

    base = {
        "study_id":
            study_id,

        "series_id":
            series_id,

        "num_slices":
            len(records),

        "geometry_valid":
            bool(
                geometry.get(
                    "geometry_valid",
                    False,
                )
            ),
    }

    if not geometry.get(
        "geometry_valid",
        False,
    ):

        base[
            "reason"
        ] = geometry.get(
            "reason",
            "unknown",
        )

        return (
            base,
            geometry,
        )

    row_direction = (
        geometry[
            "row_direction"
        ]
    )

    column_direction = (
        geometry[
            "column_direction"
        ]
    )

    normal = (
        geometry[
            "normal"
        ]
    )

    base.update(
        {
            "row_x":
                float(
                    row_direction[0]
                ),

            "row_y":
                float(
                    row_direction[1]
                ),

            "row_z":
                float(
                    row_direction[2]
                ),

            "column_x":
                float(
                    column_direction[0]
                ),

            "column_y":
                float(
                    column_direction[1]
                ),

            "column_z":
                float(
                    column_direction[2]
                ),

            "normal_x":
                float(
                    normal[0]
                ),

            "normal_y":
                float(
                    normal[1]
                ),

            "normal_z":
                float(
                    normal[2]
                ),

            "pixel_spacing_row":
                (
                    geometry[
                        "pixel_spacing"
                    ][0]
                    if geometry[
                        "pixel_spacing"
                    ]
                    else None
                ),

            "pixel_spacing_column":
                (
                    geometry[
                        "pixel_spacing"
                    ][1]
                    if geometry[
                        "pixel_spacing"
                    ]
                    else None
                ),

            "median_slice_spacing":
                geometry[
                    "median_slice_spacing"
                ],

            "physical_slice_extent":
                geometry[
                    "physical_slice_extent"
                ],

            "normal_alignment":
                geometry[
                    "normal_alignment"
                ],

            "normal_angle":
                geometry[
                    "normal_angle"
                ],
        }
    )

    # Acquisition orientation label based on normal.
    normal_abs = np.abs(
        normal
    )

    dominant_axis = int(
        np.argmax(
            normal_abs
        )
    )

    if dominant_axis == 0:

        estimated_orientation = (
            "Sagittal-like"
        )

    elif dominant_axis == 1:

        estimated_orientation = (
            "Coronal-like"
        )

    else:

        estimated_orientation = (
            "Axial-like"
        )

    base[
        "estimated_orientation"
    ] = estimated_orientation

    return (
        base,
        geometry,
    )


# ================================================================
# POINT DISTRIBUTION SUMMARY
# ================================================================

def calculate_distribution(
    point_df,
):

    rows = []

    for class_id in [
        LEFT_FORAMINAL,
        RIGHT_FORAMINAL,
    ]:

        subset = point_df[
            point_df[
                "class_id"
            ]
            ==
            class_id
        ]

        if subset.empty:

            continue

        rows.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    CLASS_NAMES[
                        class_id
                    ],

                "points":
                    len(subset),

                "native_x_mean":
                    float(
                        subset[
                            "native_x"
                        ].mean()
                    ),

                "native_x_median":
                    float(
                        subset[
                            "native_x"
                        ].median()
                    ),

                "native_x_std":
                    float(
                        subset[
                            "native_x"
                        ].std()
                    ),

                "native_y_mean":
                    float(
                        subset[
                            "native_y"
                        ].mean()
                    ),

                "native_y_median":
                    float(
                        subset[
                            "native_y"
                        ].median()
                    ),

                "model_x_mean":
                    float(
                        subset[
                            "model_x"
                        ].mean()
                    ),

                "model_x_median":
                    float(
                        subset[
                            "model_x"
                        ].median()
                    ),

                "model_y_mean":
                    float(
                        subset[
                            "model_y"
                        ].mean()
                    ),

                "model_y_median":
                    float(
                        subset[
                            "model_y"
                        ].median()
                    ),

                "patient_x_mean":
                    float(
                        subset[
                            "patient_x"
                        ].mean()
                    ),

                "patient_x_median":
                    float(
                        subset[
                            "patient_x"
                        ].median()
                    ),

                "patient_y_mean":
                    float(
                        subset[
                            "patient_y"
                        ].mean()
                    ),

                "patient_y_median":
                    float(
                        subset[
                            "patient_y"
                        ].median()
                    ),

                "patient_z_mean":
                    float(
                        subset[
                            "patient_z"
                        ].mean()
                    ),

                "column_projection_mean":
                    float(
                        subset[
                            "column_projection"
                        ].mean()
                    ),

                "column_projection_median":
                    float(
                        subset[
                            "column_projection"
                        ].median()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ================================================================
# WITHIN-SERIES LEFT/RIGHT SEPARATION
# ================================================================

def calculate_series_separation(
    point_df,
):

    rows = []

    grouped = point_df.groupby(
        [
            "study_id",
            "series_id",
        ]
    )

    for (
        study_id,
        series_id,
    ), group in grouped:

        left = group[
            group[
                "class_id"
            ]
            ==
            LEFT_FORAMINAL
        ]

        right = group[
            group[
                "class_id"
            ]
            ==
            RIGHT_FORAMINAL
        ]

        if (
            left.empty
            or
            right.empty
        ):

            continue

        # Difference in native x.
        native_x_delta = (
            float(
                left[
                    "native_x"
                ].mean()
            )
            -
            float(
                right[
                    "native_x"
                ].mean()
            )
        )

        model_x_delta = (
            float(
                left[
                    "model_x"
                ].mean()
            )
            -
            float(
                right[
                    "model_x"
                ].mean()
            )
        )

        patient_x_delta = (
            float(
                left[
                    "patient_x"
                ].mean()
            )
            -
            float(
                right[
                    "patient_x"
                ].mean()
            )
        )

        column_delta = (
            float(
                left[
                    "column_projection"
                ].mean()
            )
            -
            float(
                right[
                    "column_projection"
                ].mean()
            )
        )

        rows.append(
            {
                "study_id":
                    study_id,

                "series_id":
                    series_id,

                "left_points":
                    len(left),

                "right_points":
                    len(right),

                "native_x_delta_left_minus_right":
                    native_x_delta,

                "model_x_delta_left_minus_right":
                    model_x_delta,

                "patient_x_delta_left_minus_right":
                    patient_x_delta,

                "column_projection_delta_left_minus_right":
                    column_delta,

                "native_x_order":
                    (
                        "LEFT_GREATER"
                        if native_x_delta > 0
                        else
                        "LEFT_SMALLER"
                        if native_x_delta < 0
                        else
                        "EQUAL"
                    ),

                "model_x_order":
                    (
                        "LEFT_GREATER"
                        if model_x_delta > 0
                        else
                        "LEFT_SMALLER"
                        if model_x_delta < 0
                        else
                        "EQUAL"
                    ),

                "patient_x_order":
                    (
                        "LEFT_GREATER"
                        if patient_x_delta > 0
                        else
                        "LEFT_SMALLER"
                        if patient_x_delta < 0
                        else
                        "EQUAL"
                    ),

                "column_projection_order":
                    (
                        "LEFT_GREATER"
                        if column_delta > 0
                        else
                        "LEFT_SMALLER"
                        if column_delta < 0
                        else
                        "EQUAL"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ================================================================
# VISUALIZATION
# ================================================================

def save_coordinate_scatter(
    point_df,
):

    if point_df.empty:

        return

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    left = point_df[
        point_df[
            "class_id"
        ]
        ==
        LEFT_FORAMINAL
    ]

    right = point_df[
        point_df[
            "class_id"
        ]
        ==
        RIGHT_FORAMINAL
    ]

    ax.scatter(
        left["native_x"],
        left["native_y"],
        s=22,
        alpha=0.70,
        label="Left Neural Foraminal Narrowing",
    )

    ax.scatter(
        right["native_x"],
        right["native_y"],
        s=22,
        alpha=0.70,
        label="Right Neural Foraminal Narrowing",
    )

    ax.set_xlabel(
        "Native X coordinate"
    )

    ax.set_ylabel(
        "Native Y coordinate"
    )

    ax.set_title(
        "RSNA Foraminal Point Distribution"
    )

    ax.legend()

    ax.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.savefig(
        VIS_DIR
        /
        "part219_native_xy_distribution.png",
        dpi=180,
    )

    plt.close(fig)


def save_model_coordinate_scatter(
    point_df,
):

    if point_df.empty:

        return

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    left = point_df[
        point_df[
            "class_id"
        ]
        ==
        LEFT_FORAMINAL
    ]

    right = point_df[
        point_df[
            "class_id"
        ]
        ==
        RIGHT_FORAMINAL
    ]

    ax.scatter(
        left["model_x"],
        left["model_y"],
        s=22,
        alpha=0.70,
        label="Left Neural Foraminal Narrowing",
    )

    ax.scatter(
        right["model_x"],
        right["model_y"],
        s=22,
        alpha=0.70,
        label="Right Neural Foraminal Narrowing",
    )

    ax.set_xlabel(
        "Model X coordinate"
    )

    ax.set_ylabel(
        "Model Y coordinate"
    )

    ax.set_title(
        "Model-Grid Foraminal Point Distribution"
    )

    ax.legend()

    ax.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.savefig(
        VIS_DIR
        /
        "part219_model_xy_distribution.png",
        dpi=180,
    )

    plt.close(fig)


def save_patient_coordinate_scatter(
    point_df,
):

    valid = point_df.dropna(
        subset=[
            "patient_x",
            "patient_y",
        ]
    )

    if valid.empty:

        return

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    left = valid[
        valid[
            "class_id"
        ]
        ==
        LEFT_FORAMINAL
    ]

    right = valid[
        valid[
            "class_id"
        ]
        ==
        RIGHT_FORAMINAL
    ]

    ax.scatter(
        left["patient_x"],
        left["patient_y"],
        s=22,
        alpha=0.70,
        label="Left Neural Foraminal Narrowing",
    )

    ax.scatter(
        right["patient_x"],
        right["patient_y"],
        s=22,
        alpha=0.70,
        label="Right Neural Foraminal Narrowing",
    )

    ax.set_xlabel(
        "Patient X (mm)"
    )

    ax.set_ylabel(
        "Patient Y (mm)"
    )

    ax.set_title(
        "Physical Patient-Space Foraminal Distribution"
    )

    ax.legend()

    ax.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.savefig(
        VIS_DIR
        /
        "part219_patient_xy_distribution.png",
        dpi=180,
    )

    plt.close(fig)


# ================================================================
# CASE ORIENTATION VISUALIZATION
# ================================================================

def save_case_orientation_image(
    image,
    point_df,
    study_id,
    series_id,
):

    if image is None:

        return

    if image.ndim != 3:

        return

    center_z = (
        image.shape[0] // 2
    )

    image_slice = image[
        center_z
    ]

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.imshow(
        image_slice,
        cmap="gray",
    )

    case_points = point_df[
        (
            point_df[
                "study_id"
            ]
            ==
            study_id
        )
        &
        (
            point_df[
                "series_id"
            ]
            ==
            series_id
        )
    ]

    for _, point in (
        case_points.iterrows()
    ):

        # Only points close to this native slice.
        if abs(
            point["model_z"]
            -
            (
                center_z
                *
                MODEL_SHAPE[0]
                /
                max(
                    image.shape[0],
                    1,
                )
            )
        ) > 3:

            continue

        x = (
            point["model_x"]
            /
            MODEL_SHAPE[2]
            *
            image.shape[2]
        )

        y = (
            point["model_y"]
            /
            MODEL_SHAPE[1]
            *
            image.shape[1]
        )

        if (
            point["class_id"]
            ==
            LEFT_FORAMINAL
        ):

            marker = "L"

        else:

            marker = "R"

        ax.scatter(
            x,
            y,
            s=70,
            marker="x",
        )

        ax.text(
            x + 3,
            y + 3,
            marker,
            fontsize=10,
        )

    ax.set_title(
        f"Study {study_id} / Series {series_id}\n"
        "L/R foraminal annotation audit"
    )

    ax.axis(
        "off"
    )

    plt.tight_layout()

    filename = (
        f"study_{study_id}_"
        f"series_{series_id}_orientation.png"
    )

    plt.savefig(
        VIS_DIR / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 80)
    print(
        "PART 2.19"
    )
    print(
        "ANATOMICAL LEFT/RIGHT SPATIAL ORIENTATION AUDIT"
    )
    print("=" * 80)

    print()

    print(
        f"Project root:\n{ROOT}"
    )

    print(
        f"Manifest:\n{MANIFEST_PATH}"
    )

    # ------------------------------------------------------------
    # MANIFEST
    # ------------------------------------------------------------

    manifest_df = (
        load_manifest()
    )

    print()

    print(
        f"Manifest rows: "
        f"{len(manifest_df)}"
    )

    print(
        f"Annotated series: "
        f"{manifest_df[['study_id','series_id']].drop_duplicates().shape[0]}"
    )

    # ------------------------------------------------------------
    # VALIDATION SERIES
    # ------------------------------------------------------------

    (
        validation_series,
        train_studies,
        validation_studies,
    ) = get_validation_series(
        manifest_df
    )

    print(
        f"Validation studies available: "
        f"{len(validation_studies)}"
    )

    print(
        f"Validation series selected: "
        f"{len(validation_series)}"
    )

    # ------------------------------------------------------------
    # FORAMINAL POINTS
    # ------------------------------------------------------------

    foraminal_df = manifest_df[
        manifest_df[
            "class_id"
        ].isin(
            [
                LEFT_FORAMINAL,
                RIGHT_FORAMINAL,
            ]
        )
    ].copy()

    validation_points = foraminal_df[
        foraminal_df[
            "study_id"
        ].isin(
            set(
                validation_series[
                    "study_id"
                ]
            )
        )
        &
        foraminal_df[
            "series_id"
        ].isin(
            set(
                validation_series[
                    "series_id"
                ]
            )
        )
    ].copy()

    # Ensure exact pair matching.
    validation_pairs = set(
        zip(
            validation_series[
                "study_id"
            ],
            validation_series[
                "series_id"
            ],
        )
    )

    validation_points = (
        validation_points[
            validation_points.apply(
                lambda row:
                    (
                        row[
                            "study_id"
                        ],
                        row[
                            "series_id"
                        ],
                    )
                    in validation_pairs,
                axis=1,
            )
        ]
        .copy()
    )

    print()

    print(
        "Validation foraminal points:"
    )

    print(
        validation_points[
            "class_name"
        ].value_counts()
    )

    # ------------------------------------------------------------
    # PER-SERIES AUDIT
    # ------------------------------------------------------------

    orientation_rows = []

    point_rows = []

    processed = 0

    geometry_failures = 0

    for index, (_, series_row) in enumerate(
        validation_series.iterrows(),
        start=1,
    ):

        study_id = str(
            series_row[
                "study_id"
            ]
        )

        series_id = str(
            series_row[
                "series_id"
            ]
        )

        print()

        print(
            f"[{index}/{len(validation_series)}] "
            f"Study={study_id} "
            f"Series={series_id}"
        )

        case_df = validation_points[
            (
                validation_points[
                    "study_id"
                ]
                ==
                study_id
            )
            &
            (
                validation_points[
                    "series_id"
                ]
                ==
                series_id
            )
        ].copy()

        try:

            dummy_row = pd.Series(
                {
                    "study_id":
                        study_id,

                    "series_id":
                        series_id,
                }
            )

            series_dir = (
                resolve_series_dir(
                    dummy_row
                )
            )

            image_native, records, info = (
                read_dicom_series_robust(
                    series_dir
                )
            )

            image_native = np.asarray(
                image_native,
                dtype=np.float32,
            )

            summary, geometry = (
                orientation_summary(
                    study_id,
                    series_id,
                    records,
                )
            )

            orientation_rows.append(
                summary
            )

            if not geometry.get(
                "geometry_valid",
                False,
            ):

                geometry_failures += 1

                print(
                    "  Geometry INVALID"
                )

                continue

            print(
                f"  Slices: "
                f"{len(records)}"
            )

            print(
                f"  Estimated orientation: "
                f"{summary.get('estimated_orientation')}"
            )

            print(
                f"  Row direction: "
                f"{np.round(geometry['row_direction'], 5)}"
            )

            print(
                f"  Column direction: "
                f"{np.round(geometry['column_direction'], 5)}"
            )

            print(
                f"  Normal: "
                f"{np.round(geometry['normal'], 5)}"
            )

            print(
                f"  Pixel spacing: "
                f"{geometry['pixel_spacing']}"
            )

            print(
                f"  Slice spacing: "
                f"{geometry['median_slice_spacing']}"
            )

            if not case_df.empty:

                case_point_rows = (
                    build_point_audit(
                        case_df,
                        records,
                        geometry,
                    )
                )

                point_rows.extend(
                    case_point_rows
                )

                print(
                    f"  Foraminal points: "
                    f"{len(case_point_rows)}"
                )

            else:

                print(
                    "  Foraminal points: 0"
                )

            processed += 1

        except Exception as exc:

            print(
                f"  ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    # ------------------------------------------------------------
    # DATAFRAMES
    # ------------------------------------------------------------

    orientation_df = pd.DataFrame(
        orientation_rows
    )

    point_df = pd.DataFrame(
        point_rows
    )

    print()

    print(
        "=" * 80
    )

    print(
        "AUDIT DATASET SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Processed series: "
        f"{processed}"
    )

    print(
        f"Geometry failures: "
        f"{geometry_failures}"
    )

    print(
        f"Foraminal points audited: "
        f"{len(point_df)}"
    )

    # ------------------------------------------------------------
    # DISTRIBUTION
    # ------------------------------------------------------------

    distribution_df = (
        calculate_distribution(
            point_df
        )
    )

    separation_df = (
        calculate_series_separation(
            point_df
        )
    )

    # ------------------------------------------------------------
    # ORDER CONSISTENCY
    # ------------------------------------------------------------

    consistency_summary = {}

    if not separation_df.empty:

        for column in [
            "native_x_order",
            "model_x_order",
            "patient_x_order",
            "column_projection_order",
        ]:

            counts = (
                separation_df[
                    column
                ]
                .value_counts()
                .to_dict()
            )

            consistency_summary[
                column
            ] = {
                str(key):
                    int(value)
                for key, value in counts.items()
            }

    # ------------------------------------------------------------
    # SAVE TABLES
    # ------------------------------------------------------------

    orientation_df.to_csv(
        OUTPUT_DIR
        /
        "part219_series_orientation.csv",
        index=False,
    )

    point_df.to_csv(
        OUTPUT_DIR
        /
        "part219_point_distribution.csv",
        index=False,
    )

    distribution_df.to_csv(
        OUTPUT_DIR
        /
        "part219_coordinate_summary.csv",
        index=False,
    )

    separation_df.to_csv(
        OUTPUT_DIR
        /
        "part219_left_right_series_separation.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------------------

    save_coordinate_scatter(
        point_df
    )

    save_model_coordinate_scatter(
        point_df
    )

    save_patient_coordinate_scatter(
        point_df
    )

    # ------------------------------------------------------------
    # SUMMARY JSON
    # ------------------------------------------------------------

    summary = {
        "part":
            "2.19",

        "status":
            "COMPLETE",

        "validation_series":
            int(
                len(
                    validation_series
                )
            ),

        "processed_series":
            int(
                processed
            ),

        "geometry_failures":
            int(
                geometry_failures
            ),

        "foraminal_points_audited":
            int(
                len(point_df)
            ),

        "left_foraminal_points":
            int(
                (
                    point_df[
                        "class_id"
                    ]
                    ==
                    LEFT_FORAMINAL
                ).sum()
            )
            if not point_df.empty
            else 0,

        "right_foraminal_points":
            int(
                (
                    point_df[
                        "class_id"
                    ]
                    ==
                    RIGHT_FORAMINAL
                ).sum()
            )
            if not point_df.empty
            else 0,

        "coordinate_order_consistency":
            consistency_summary,

        "manual_voxel_ground_truth":
            False,

        "model_training_performed":
            False,

        "checkpoint_modified":
            False,

        "dashboard_modified":
            False,

        "scientific_limitation":
            (
                "This audit evaluates coordinate and DICOM "
                "geometry consistency. RSNA annotations are "
                "point/localization annotations rather than "
                "manual voxel-wise segmentation masks."
            ),
    }

    with open(
        OUTPUT_DIR
        /
        "part219_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------

    report_path = (
        REPORT_DIR
        /
        "part219_orientation_audit_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PART 2.19\n"
        )

        f.write(
            "ANATOMICAL LEFT/RIGHT SPATIAL ORIENTATION AUDIT\n"
        )

        f.write(
            "=" * 75
            + "\n\n"
        )

        f.write(
            f"Validation series: "
            f"{len(validation_series)}\n"
        )

        f.write(
            f"Processed series: "
            f"{processed}\n"
        )

        f.write(
            f"Geometry failures: "
            f"{geometry_failures}\n"
        )

        f.write(
            f"Foraminal points audited: "
            f"{len(point_df)}\n\n"
        )

        f.write(
            "COORDINATE SUMMARY\n"
        )

        f.write(
            "-" * 75
            + "\n"
        )

        if not distribution_df.empty:

            f.write(
                distribution_df.to_string(
                    index=False
                )
            )

        f.write(
            "\n\n"
        )

        f.write(
            "LEFT/RIGHT WITHIN-SERIES SEPARATION\n"
        )

        f.write(
            "-" * 75
            + "\n"
        )

        if not separation_df.empty:

            f.write(
                separation_df.to_string(
                    index=False
                )
            )

        f.write(
            "\n\n"
        )

        f.write(
            "ORDER CONSISTENCY\n"
        )

        f.write(
            "-" * 75
            + "\n"
        )

        for key, value in (
            consistency_summary.items()
        ):

            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            "\n\n"
        )

        f.write(
            "SCIENTIFIC LIMITATION\n"
        )

        f.write(
            "-" * 75
            + "\n"
        )

        f.write(
            "This audit does not establish voxel-wise "
            "segmentation accuracy. RSNA annotations are "
            "point/localization annotations rather than "
            "manual voxel-wise segmentation masks.\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "No model was trained or modified.\n"
        )

        f.write(
            "No checkpoint was modified.\n"
        )

        f.write(
            "The dashboard was not modified.\n"
        )

    # ------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "PART 2.19 COMPLETE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "COORDINATE SUMMARY"
    )

    print(
        distribution_df.to_string(
            index=False
        )
    )

    print()

    print(
        "LEFT/RIGHT SERIES SEPARATION"
    )

    if separation_df.empty:

        print(
            "No series contained both LFNN and RFNN points."
        )

    else:

        print(
            separation_df.to_string(
                index=False
            )
        )

    print()

    print(
        "ORDER CONSISTENCY"
    )

    for key, value in (
        consistency_summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    print()

    print(
        "No training performed."
    )

    print(
        "No checkpoint modified."
    )

    print(
        "Dashboard not modified."
    )

    print()

    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()