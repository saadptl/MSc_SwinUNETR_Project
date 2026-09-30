"""
Part 2.20A
Physical-Space Point Mapping Audit

Purpose
-------
Audit RSNA point annotations using DICOM physical-space geometry.

This part:
1. Uses the Part 2.13 point-supervision manifest.
2. Selects the same study-disjoint validation pool.
3. Reads the actual DICOM geometry for each series.
4. Converts each annotation from native pixel coordinates to
   DICOM patient physical coordinates (mm).
5. Builds a per-series canonical physical coordinate system.
6. Maps points into normalized canonical coordinates.
7. Audits anatomical left/right consistency using physical X.
8. Produces CSV + JSON + text reports.

IMPORTANT
---------
- NO training.
- NO checkpoint modification.
- NO dashboard modification.
- NO pseudo-mask generation.
- NO fabricated voxel-wise ground truth.

RSNA annotations are point/localization annotations, not manual
voxel-wise segmentation masks.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT / "src"

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part220a_physical_space_point_mapping_audit"
)

REPORT_DIR = OUTPUT_DIR / "reports"

MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part213_point_supervision_manifest"
    / "part213_point_manifest.csv"
)

TRAIN_CSV = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
    / "train.csv"
)

SERIES_DESCRIPTION_CSV = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
    / "train_series_descriptions.csv"
)

# Fallback locations in case your dataset structure differs.
DATASET_ROOT = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES_DIR = DATASET_ROOT / "train_images"


# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_SHAPE = (64, 96, 96)

# Same validation design used in previous audits.
N_VALIDATION_SERIES = 25

RANDOM_SEED = 42

# DICOM patient coordinate system:
#
# For the standard DICOM BIPED patient coordinate system:
# +X = patient's LEFT
# +Y = patient's POSTERIOR
# +Z = patient's HEAD
#
# We use physical X only as an audit axis.
# This is NOT used to fabricate a segmentation mask.
PATIENT_X_AXIS = 0


# ============================================================
# 3. IMPORT ROBUST DICOM LOADER
# ============================================================

try:
    from segmentation_rsna_part11_controlled_pilot_training_corrected import (
        read_dicom_series_robust,
        resolve_series_dir,
    )
except ImportError:
    try:
        from src.segmentation_rsna_part11_controlled_pilot_training_corrected import (
            read_dicom_series_robust,
            resolve_series_dir,
        )
    except ImportError as exc:
        raise ImportError(
            "\nCould not import Part 2.11 robust DICOM loader.\n"
            "Make sure this file exists:\n"
            "src\\segmentation_rsna_part11_controlled_pilot_training_corrected.py\n"
        ) from exc


# ============================================================
# 4. OUTPUT SETUP
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 5. HELPERS
# ============================================================

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Return a unit vector."""
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)

    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero vector.")

    return v / norm


def safe_float(value, default=np.nan):
    """Convert value to float safely."""
    try:
        return float(value)
    except Exception:
        return default


def get_dicom_value(ds, name, default=None):
    """Safely retrieve a DICOM attribute."""
    try:
        value = getattr(ds, name)
        return value
    except Exception:
        return default


def get_iop(ds):
    """Read ImageOrientationPatient."""
    value = get_dicom_value(ds, "ImageOrientationPatient")

    if value is None or len(value) != 6:
        raise ValueError(
            "Missing or invalid ImageOrientationPatient."
        )

    row_direction = normalize_vector(
        np.asarray(value[:3], dtype=np.float64)
    )

    column_direction = normalize_vector(
        np.asarray(value[3:], dtype=np.float64)
    )

    # Normal follows right-hand rule.
    normal_direction = normalize_vector(
        np.cross(row_direction, column_direction)
    )

    return row_direction, column_direction, normal_direction


def get_pixel_spacing(ds):
    """Read DICOM PixelSpacing."""
    value = get_dicom_value(ds, "PixelSpacing")

    if value is None or len(value) != 2:
        raise ValueError("Missing or invalid PixelSpacing.")

    return (
        float(value[0]),
        float(value[1]),
    )


def get_image_position(ds):
    """Read ImagePositionPatient."""
    value = get_dicom_value(ds, "ImagePositionPatient")

    if value is None or len(value) != 3:
        raise ValueError(
            "Missing or invalid ImagePositionPatient."
        )

    return np.asarray(value, dtype=np.float64)


def pixel_to_patient(
    ds,
    row: float,
    column: float,
) -> np.ndarray:
    """
    Convert a DICOM pixel coordinate to patient physical coordinates.

    DICOM convention:

        P = ImagePositionPatient
            + row * PixelSpacing[0] * row_direction
            + column * PixelSpacing[1] * column_direction
    """

    row_direction, column_direction, _ = get_iop(ds)
    row_spacing, column_spacing = get_pixel_spacing(ds)
    ipp = get_image_position(ds)

    patient_point = (
        ipp
        + row * row_spacing * row_direction
        + column * column_spacing * column_direction
    )

    return patient_point.astype(np.float64)


def physical_to_series_coordinates(
    patient_point: np.ndarray,
    origin: np.ndarray,
    row_direction: np.ndarray,
    column_direction: np.ndarray,
    normal_direction: np.ndarray,
    row_spacing: float,
    column_spacing: float,
    slice_spacing: float,
):
    """
    Project a patient-space point into a series physical coordinate
    system.

    Output coordinates are continuous voxel-like coordinates:

        series_z
        series_y
        series_x

    where:
        x -> image column direction
        y -> image row direction
        z -> slice normal
    """

    delta = patient_point - origin

    x_mm = float(np.dot(delta, column_direction))
    y_mm = float(np.dot(delta, row_direction))
    z_mm = float(np.dot(delta, normal_direction))

    x = x_mm / max(column_spacing, 1e-8)
    y = y_mm / max(row_spacing, 1e-8)
    z = z_mm / max(slice_spacing, 1e-8)

    return z, y, x


def get_slice_positions(records):
    """Return physical slice positions along the series normal."""
    if not records:
        raise ValueError("No DICOM records.")

    row_direction, column_direction, normal_direction = get_iop(
        records[0]
    )

    positions = []

    for ds in records:
        ipp = get_image_position(ds)

        projection = float(
            np.dot(ipp, normal_direction)
        )

        positions.append(projection)

    return np.asarray(positions, dtype=np.float64)


def estimate_slice_spacing(records):
    """
    Estimate slice spacing from ImagePositionPatient.

    Uses median difference of sorted physical slice positions.
    """
    projections = get_slice_positions(records)

    if len(projections) < 2:
        return 1.0

    unique_positions = np.unique(
        np.round(projections, decimals=6)
    )

    if len(unique_positions) < 2:
        return 1.0

    diffs = np.diff(
        np.sort(unique_positions)
    )

    diffs = diffs[diffs > 1e-6]

    if len(diffs) == 0:
        return 1.0

    return float(np.median(diffs))


def build_series_geometry(records, image_shape):
    """
    Build physical geometry information for one DICOM series.
    """

    if len(records) == 0:
        raise ValueError("Empty DICOM series.")

    first = records[0]

    row_direction, column_direction, normal_direction = get_iop(
        first
    )

    row_spacing, column_spacing = get_pixel_spacing(first)

    origin = get_image_position(first)

    slice_spacing = estimate_slice_spacing(records)

    depth, rows, cols = image_shape

    # Physical coordinates of all eight volume corners.
    corner_points = []

    for z in [0, depth - 1]:
        for y in [0, rows - 1]:
            for x in [0, cols - 1]:

                ds = records[
                    min(
                        max(int(round(z)), 0),
                        len(records) - 1,
                    )
                ]

                patient_point = pixel_to_patient(
                    ds,
                    float(y),
                    float(x),
                )

                corner_points.append(
                    patient_point
                )

    corner_points = np.asarray(
        corner_points,
        dtype=np.float64,
    )

    patient_min = corner_points.min(axis=0)
    patient_max = corner_points.max(axis=0)

    extent = patient_max - patient_min

    return {
        "origin": origin,
        "row_direction": row_direction,
        "column_direction": column_direction,
        "normal_direction": normal_direction,
        "row_spacing": float(row_spacing),
        "column_spacing": float(column_spacing),
        "slice_spacing": float(slice_spacing),
        "patient_min": patient_min,
        "patient_max": patient_max,
        "patient_extent": extent,
        "image_shape": tuple(int(v) for v in image_shape),
    }


def patient_to_canonical_normalized(
    patient_point: np.ndarray,
    patient_min: np.ndarray,
    patient_max: np.ndarray,
):
    """
    Map patient-space XYZ to normalized canonical XYZ.

    0 = minimum physical coordinate
    1 = maximum physical coordinate

    This is an audit representation only.
    It is NOT a segmentation target.
    """

    extent = patient_max - patient_min

    extent = np.where(
        np.abs(extent) < 1e-8,
        1.0,
        extent,
    )

    normalized = (
        patient_point - patient_min
    ) / extent

    return normalized


def canonical_to_model_grid(
    normalized_xyz: np.ndarray,
):
    """
    Map normalized canonical XYZ to model-grid coordinates.

    Canonical axes:

        X -> model width
        Y -> model height
        Z -> model depth
    """

    d, h, w = MODEL_SHAPE

    x = normalized_xyz[0] * (w - 1)
    y = normalized_xyz[1] * (h - 1)
    z = normalized_xyz[2] * (d - 1)

    return (
        float(z),
        float(y),
        float(x),
    )


# ============================================================
# 6. FIND DATASET
# ============================================================

def locate_file(filename: str):
    """
    Find a dataset file in the expected project locations.
    """

    candidates = [
        DATASET_ROOT / filename,
        ROOT / filename,
        ROOT / "data" / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def locate_images_dir():
    candidates = [
        TRAIN_IMAGES_DIR,
        DATASET_ROOT / "train_images",
        ROOT / "train_images",
        ROOT / "dataset" / "train_images",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


# ============================================================
# 7. LOAD MANIFEST
# ============================================================

def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"\nPart 2.13 manifest not found:\n"
            f"{MANIFEST_PATH}\n\n"
            "Run Part 2.13 first."
        )

    df = pd.read_csv(
        MANIFEST_PATH,
        dtype={
            "study_id": str,
            "series_id": str,
        },
    )

    required = [
        "study_id",
        "series_id",
        "condition",
        "class_id",
        "class_name",
        "level",
        "native_z",
        "native_y",
        "native_x",
        "model_z_float",
        "model_y_float",
        "model_x_float",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Part 2.13 manifest is missing columns: {missing}"
        )

    return df


# ============================================================
# 8. SELECT STUDY-DISJOINT VALIDATION SERIES
# ============================================================

def select_validation_series(manifest: pd.DataFrame):
    """
    Reproduce a study-disjoint validation selection.

    We first identify available annotated studies and select
    deterministic validation studies using the same seed.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    studies = np.array(
        sorted(
            manifest["study_id"]
            .astype(str)
            .unique()
        )
    )

    if len(studies) == 0:
        raise ValueError("No studies found in manifest.")

    # Deterministic shuffle.
    shuffled = studies.copy()
    rng.shuffle(shuffled)

    validation_studies = set(
        shuffled[
            : min(
                395,
                len(shuffled),
            )
        ]
    )

    validation_df = manifest[
        manifest["study_id"].astype(str).isin(
            validation_studies
        )
    ].copy()

    # One series per study, deterministic.
    series_table = (
        validation_df[
            [
                "study_id",
                "series_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["study_id", "series_id"]
        )
    )

    selected_rows = []

    for study_id, group in series_table.groupby(
        "study_id",
        sort=True,
    ):
        selected_rows.append(
            group.iloc[0]
        )

    selected = pd.DataFrame(
        selected_rows
    )

    if len(selected) > N_VALIDATION_SERIES:
        selected = selected.iloc[
            :N_VALIDATION_SERIES
        ].copy()

    return selected


# ============================================================
# 9. MAIN AUDIT
# ============================================================

def main():

    print("=" * 78)
    print("PART 2.20A")
    print("PHYSICAL-SPACE POINT MAPPING AUDIT")
    print("=" * 78)

    print("\nProject root:")
    print(ROOT)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nModel shape:")
    print(MODEL_SHAPE)

    print("\nLoading Part 2.13 manifest...")

    manifest = load_manifest()

    print(
        f"Manifest rows: {len(manifest)}"
    )

    print(
        f"Annotated series: "
        f"{manifest[['study_id','series_id']].drop_duplicates().shape[0]}"
    )

    selected = select_validation_series(
        manifest
    )

    print(
        f"\nValidation series selected: "
        f"{len(selected)}"
    )

    print(
        f"Validation studies: "
        f"{selected['study_id'].nunique()}"
    )

    images_dir = locate_images_dir()

    if images_dir is None:
        raise FileNotFoundError(
            "\nCould not locate train_images directory.\n"
            "Expected:\n"
            f"{TRAIN_IMAGES_DIR}"
        )

    print("\nDICOM images:")
    print(images_dir)

    all_point_rows = []
    geometry_rows = []

    failed_series = []

    total_points = 0

    # --------------------------------------------------------
    # Process each selected series
    # --------------------------------------------------------

    for series_number, (_, series_row) in enumerate(
        selected.iterrows(),
        start=1,
    ):

        study_id = str(
            series_row["study_id"]
        )

        series_id = str(
            series_row["series_id"]
        )

        print(
            f"\n[{series_number}/{len(selected)}] "
            f"Study={study_id} "
            f"Series={series_id}"
        )

        try:

            # ------------------------------------------------
            # Locate DICOM series
            # ------------------------------------------------

            series_path = (
                images_dir
                / study_id
                / series_id
            )

            if not series_path.exists():

                # Try the robust resolver.
                resolver_row = pd.Series(
                    {
                        "study_id": study_id,
                        "series_id": series_id,
                    }
                )

                try:
                    series_path = resolve_series_dir(
                        resolver_row
                    )
                except Exception:
                    pass

            if not series_path.exists():
                raise FileNotFoundError(
                    f"Series directory not found: "
                    f"{series_path}"
                )

            # ------------------------------------------------
            # Read robust DICOM series
            # ------------------------------------------------

            image, records, dicom_info = (
                read_dicom_series_robust(
                    series_path
                )
            )

            image = np.asarray(
                image,
                dtype=np.float32,
            )

            if image.ndim != 3:
                raise ValueError(
                    f"Expected 3D image, got "
                    f"{image.shape}"
                )

            image_shape = tuple(
                int(v)
                for v in image.shape
            )

            # ------------------------------------------------
            # Geometry
            # ------------------------------------------------

            geometry = build_series_geometry(
                records,
                image_shape,
            )

            origin = geometry["origin"]
            row_direction = geometry[
                "row_direction"
            ]
            column_direction = geometry[
                "column_direction"
            ]
            normal_direction = geometry[
                "normal_direction"
            ]

            row_spacing = geometry[
                "row_spacing"
            ]

            column_spacing = geometry[
                "column_spacing"
            ]

            slice_spacing = geometry[
                "slice_spacing"
            ]

            patient_min = geometry[
                "patient_min"
            ]

            patient_max = geometry[
                "patient_max"
            ]

            # ------------------------------------------------
            # Series annotations
            # ------------------------------------------------

            series_points = manifest[
                (
                    manifest["study_id"].astype(str)
                    == study_id
                )
                & (
                    manifest["series_id"].astype(str)
                    == series_id
                )
            ].copy()

            if len(series_points) == 0:
                print(
                    "  No annotation points."
                )
                continue

            print(
                f"  Image shape: {image_shape}"
            )

            print(
                f"  Annotation points: "
                f"{len(series_points)}"
            )

            print(
                "  Pixel spacing: "
                f"{row_spacing:.6f}, "
                f"{column_spacing:.6f} mm"
            )

            print(
                "  Slice spacing: "
                f"{slice_spacing:.6f} mm"
            )

            # ------------------------------------------------
            # Geometry record
            # ------------------------------------------------

            geometry_rows.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "image_shape": str(
                        image_shape
                    ),
                    "num_slices": image_shape[0],
                    "rows": image_shape[1],
                    "columns": image_shape[2],
                    "row_spacing_mm": row_spacing,
                    "column_spacing_mm": column_spacing,
                    "slice_spacing_mm": slice_spacing,
                    "origin_x": origin[0],
                    "origin_y": origin[1],
                    "origin_z": origin[2],
                    "row_dir_x": row_direction[0],
                    "row_dir_y": row_direction[1],
                    "row_dir_z": row_direction[2],
                    "column_dir_x": column_direction[0],
                    "column_dir_y": column_direction[1],
                    "column_dir_z": column_direction[2],
                    "normal_dir_x": normal_direction[0],
                    "normal_dir_y": normal_direction[1],
                    "normal_dir_z": normal_direction[2],
                    "patient_min_x": patient_min[0],
                    "patient_min_y": patient_min[1],
                    "patient_min_z": patient_min[2],
                    "patient_max_x": patient_max[0],
                    "patient_max_y": patient_max[1],
                    "patient_max_z": patient_max[2],
                    "extent_x_mm": geometry[
                        "patient_extent"
                    ][0],
                    "extent_y_mm": geometry[
                        "patient_extent"
                    ][1],
                    "extent_z_mm": geometry[
                        "patient_extent"
                    ][2],
                    "point_count": len(
                        series_points
                    ),
                }
            )

            # ------------------------------------------------
            # Point mapping
            # ------------------------------------------------

            for _, point in series_points.iterrows():

                native_z = float(
                    point["native_z"]
                )

                native_y = float(
                    point["native_y"]
                )

                native_x = float(
                    point["native_x"]
                )

                # --------------------------------------------
                # Bounds check
                # --------------------------------------------

                z_index = int(
                    np.clip(
                        round(native_z),
                        0,
                        len(records) - 1,
                    )
                )

                ds = records[z_index]

                # --------------------------------------------
                # Native pixel -> patient physical coordinates
                # --------------------------------------------

                patient_point = pixel_to_patient(
                    ds,
                    native_y,
                    native_x,
                )

                # --------------------------------------------
                # Patient physical -> canonical normalized
                # --------------------------------------------

                canonical_xyz = (
                    patient_to_canonical_normalized(
                        patient_point,
                        patient_min,
                        patient_max,
                    )
                )

                canonical_z, canonical_y, canonical_x = (
                    canonical_to_model_grid(
                        canonical_xyz
                    )
                )

                # --------------------------------------------
                # Also project into original series basis
                # --------------------------------------------

                series_z, series_y, series_x = (
                    physical_to_series_coordinates(
                        patient_point,
                        origin,
                        row_direction,
                        column_direction,
                        normal_direction,
                        row_spacing,
                        column_spacing,
                        slice_spacing,
                    )
                )

                # --------------------------------------------
                # Store
                # --------------------------------------------

                all_point_rows.append(
                    {
                        "study_id": study_id,
                        "series_id": series_id,
                        "condition": point[
                            "condition"
                        ],
                        "class_id": int(
                            point["class_id"]
                        ),
                        "class_name": point[
                            "class_name"
                        ],
                        "level": point["level"],
                        "native_z": native_z,
                        "native_y": native_y,
                        "native_x": native_x,
                        "patient_x_mm": float(
                            patient_point[0]
                        ),
                        "patient_y_mm": float(
                            patient_point[1]
                        ),
                        "patient_z_mm": float(
                            patient_point[2]
                        ),
                        "canonical_x_norm": float(
                            canonical_xyz[0]
                        ),
                        "canonical_y_norm": float(
                            canonical_xyz[1]
                        ),
                        "canonical_z_norm": float(
                            canonical_xyz[2]
                        ),
                        "canonical_model_x": canonical_x,
                        "canonical_model_y": canonical_y,
                        "canonical_model_z": canonical_z,
                        "series_x": series_x,
                        "series_y": series_y,
                        "series_z": series_z,
                        "model_x_float": float(
                            point["model_x_float"]
                        ),
                        "model_y_float": float(
                            point["model_y_float"]
                        ),
                        "model_z_float": float(
                            point["model_z_float"]
                        ),
                        "image_shape": str(
                            image_shape
                        ),
                    }
                )

                total_points += 1

        except Exception as exc:

            print(
                f"  FAILED: {type(exc).__name__}: "
                f"{exc}"
            )

            failed_series.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "error_type": type(
                        exc
                    ).__name__,
                    "error": str(exc),
                }
            )

    # ========================================================
    # 10. CREATE DATAFRAMES
    # ========================================================

    points_df = pd.DataFrame(
        all_point_rows
    )

    geometry_df = pd.DataFrame(
        geometry_rows
    )

    failed_df = pd.DataFrame(
        failed_series
    )

    # ========================================================
    # 11. LEFT / RIGHT AUDIT
    # ========================================================

    lr_rows = []

    if len(points_df) > 0:

        foraminal = points_df[
            points_df["class_name"].isin(
                [
                    "Left Neural Foraminal Narrowing",
                    "Right Neural Foraminal Narrowing",
                ]
            )
        ].copy()

        # ----------------------------------------------------
        # Overall disease-level physical X
        # ----------------------------------------------------

        for class_name, group in (
            foraminal.groupby(
                "class_name",
                sort=True,
            )
        ):

            lr_rows.append(
                {
                    "analysis": "overall_class",
                    "study_id": "",
                    "series_id": "",
                    "class_name": class_name,
                    "n_points": len(group),
                    "patient_x_mean_mm": group[
                        "patient_x_mm"
                    ].mean(),
                    "patient_x_median_mm": group[
                        "patient_x_mm"
                    ].median(),
                    "canonical_x_mean": group[
                        "canonical_x_norm"
                    ].mean(),
                    "canonical_x_median": group[
                        "canonical_x_norm"
                    ].median(),
                }
            )

        # ----------------------------------------------------
        # Within-series left/right ordering
        # ----------------------------------------------------

        for (
            study_id,
            series_id,
        ), group in foraminal.groupby(
            [
                "study_id",
                "series_id",
            ],
            sort=True,
        ):

            left = group[
                group["class_name"]
                == "Left Neural Foraminal Narrowing"
            ]

            right = group[
                group["class_name"]
                == "Right Neural Foraminal Narrowing"
            ]

            if len(left) == 0 or len(right) == 0:
                continue

            left_patient_x = float(
                left["patient_x_mm"].mean()
            )

            right_patient_x = float(
                right["patient_x_mm"].mean()
            )

            left_canonical_x = float(
                left["canonical_x_norm"].mean()
            )

            right_canonical_x = float(
                right["canonical_x_norm"].mean()
            )

            left_native_x = float(
                left["native_x"].mean()
            )

            right_native_x = float(
                right["native_x"].mean()
            )

            # In DICOM LPS, larger X means patient's left.
            physical_correct = (
                left_patient_x
                > right_patient_x
            )

            canonical_correct = (
                left_canonical_x
                > right_canonical_x
            )

            native_correct = (
                left_native_x
                > right_native_x
            )

            lr_rows.append(
                {
                    "analysis": "within_series",
                    "study_id": study_id,
                    "series_id": series_id,
                    "class_name": "LFNN_vs_RFNN",
                    "n_points": len(group),
                    "patient_x_mean_mm": np.nan,
                    "patient_x_median_mm": np.nan,
                    "canonical_x_mean": np.nan,
                    "canonical_x_median": np.nan,
                    "left_patient_x": left_patient_x,
                    "right_patient_x": right_patient_x,
                    "left_canonical_x": left_canonical_x,
                    "right_canonical_x": right_canonical_x,
                    "left_native_x": left_native_x,
                    "right_native_x": right_native_x,
                    "physical_left_greater": physical_correct,
                    "canonical_left_greater": canonical_correct,
                    "native_left_greater": native_correct,
                }
            )

    lr_df = pd.DataFrame(
        lr_rows
    )

    # ========================================================
    # 12. SAVE RESULTS
    # ========================================================

    points_path = (
        OUTPUT_DIR
        / "part220a_physical_point_manifest.csv"
    )

    geometry_path = (
        OUTPUT_DIR
        / "part220a_series_geometry.csv"
    )

    lr_path = (
        OUTPUT_DIR
        / "part220a_left_right_analysis.csv"
    )

    failed_path = (
        OUTPUT_DIR
        / "part220a_failed_series.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part220a_summary.json"
    )

    report_path = (
        REPORT_DIR
        / "part220a_report.txt"
    )

    points_df.to_csv(
        points_path,
        index=False,
    )

    geometry_df.to_csv(
        geometry_path,
        index=False,
    )

    lr_df.to_csv(
        lr_path,
        index=False,
    )

    failed_df.to_csv(
        failed_path,
        index=False,
    )

    # ========================================================
    # 13. SUMMARY STATISTICS
    # ========================================================

    within_series_df = (
        lr_df[
            lr_df["analysis"]
            == "within_series"
        ]
        if len(lr_df) > 0
        else pd.DataFrame()
    )

    if len(within_series_df) > 0:

        physical_correct_count = int(
            within_series_df[
                "physical_left_greater"
            ]
            .fillna(False)
            .sum()
        )

        canonical_correct_count = int(
            within_series_df[
                "canonical_left_greater"
            ]
            .fillna(False)
            .sum()
        )

        native_correct_count = int(
            within_series_df[
                "native_left_greater"
            ]
            .fillna(False)
            .sum()
        )

        n_lr_series = len(
            within_series_df
        )

        physical_accuracy = (
            physical_correct_count
            / n_lr_series
        )

        canonical_accuracy = (
            canonical_correct_count
            / n_lr_series
        )

        native_accuracy = (
            native_correct_count
            / n_lr_series
        )

    else:

        physical_correct_count = 0
        canonical_correct_count = 0
        native_correct_count = 0
        n_lr_series = 0
        physical_accuracy = np.nan
        canonical_accuracy = np.nan
        native_accuracy = np.nan

    summary = {
        "part": "2.20A",
        "status": "COMPLETE",
        "purpose": (
            "Physical-space mapping audit of RSNA "
            "point annotations."
        ),
        "manifest": str(
            MANIFEST_PATH
        ),
        "validation_series_selected": int(
            len(selected)
        ),
        "validation_studies_selected": int(
            selected["study_id"].nunique()
        ),
        "processed_series": int(
            len(geometry_df)
        ),
        "failed_series": int(
            len(failed_df)
        ),
        "mapped_points": int(
            len(points_df)
        ),
        "expected_points": int(
            manifest[
                manifest["study_id"].astype(str).isin(
                    selected["study_id"].astype(str)
                )
                & manifest["series_id"].astype(str).isin(
                    selected["series_id"].astype(str)
                )
            ].shape[0]
        ),
        "model_shape": list(
            MODEL_SHAPE
        ),
        "patient_coordinate_system": "DICOM LPS",
        "patient_x_interpretation": (
            "larger physical X corresponds to "
            "patient left in the standard DICOM "
            "BIPED coordinate system"
        ),
        "within_series_foraminal_pairs": int(
            n_lr_series
        ),
        "native_left_greater_count": int(
            native_correct_count
        ),
        "native_left_greater_rate": (
            float(native_accuracy)
            if not math.isnan(
                native_accuracy
            )
            else None
        ),
        "physical_left_greater_count": int(
            physical_correct_count
        ),
        "physical_left_greater_rate": (
            float(physical_accuracy)
            if not math.isnan(
                physical_accuracy
            )
            else None
        ),
        "canonical_left_greater_count": int(
            canonical_correct_count
        ),
        "canonical_left_greater_rate": (
            float(canonical_accuracy)
            if not math.isnan(
                canonical_accuracy
            )
            else None
        ),
        "manual_voxel_segmentation_ground_truth": False,
        "training_performed": False,
        "checkpoint_modified": False,
        "dashboard_modified": False,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # 14. HUMAN-READABLE REPORT
    # ========================================================

    report_lines = []

    report_lines.append(
        "PART 2.20A - PHYSICAL-SPACE POINT MAPPING AUDIT"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append("")
    report_lines.append(
        "Purpose:"
    )

    report_lines.append(
        "Audit RSNA point annotations after conversion "
        "from native DICOM pixel coordinates into "
        "patient physical coordinates."
    )

    report_lines.append("")
    report_lines.append(
        "IMPORTANT:"
    )

    report_lines.append(
        "RSNA coordinates are point/localization "
        "annotations, not manual voxel-wise "
        "segmentation masks."
    )

    report_lines.append("")
    report_lines.append(
        f"Validation series selected: {len(selected)}"
    )

    report_lines.append(
        f"Processed series: {len(geometry_df)}"
    )

    report_lines.append(
        f"Failed series: {len(failed_df)}"
    )

    report_lines.append(
        f"Mapped points: {len(points_df)}"
    )

    report_lines.append("")

    report_lines.append(
        "LEFT/RIGHT AUDIT"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        f"Within-series LFNN/RFNN pairs: "
        f"{n_lr_series}"
    )

    report_lines.append(
        f"Native X left-greater rate: "
        f"{native_accuracy:.4f}"
        if not math.isnan(native_accuracy)
        else "Native X left-greater rate: N/A"
    )

    report_lines.append(
        f"Patient physical X left-greater rate: "
        f"{physical_accuracy:.4f}"
        if not math.isnan(physical_accuracy)
        else "Patient physical X left-greater rate: N/A"
    )

    report_lines.append(
        f"Canonical X left-greater rate: "
        f"{canonical_accuracy:.4f}"
        if not math.isnan(canonical_accuracy)
        else "Canonical X left-greater rate: N/A"
    )

    report_lines.append("")

    report_lines.append(
        "Interpretation:"
    )

    report_lines.append(
        "Patient physical coordinates provide a "
        "consistent anatomical reference independent "
        "of the acquired image row/column orientation."
    )

    report_lines.append(
        "This audit does not establish voxel-wise "
        "segmentation accuracy."
    )

    report_lines.append(
        "No model was trained and no checkpoint was changed."
    )

    report_lines.append(
        "No dashboard component was changed."
    )

    if len(failed_df) > 0:

        report_lines.append("")
        report_lines.append(
            "FAILED SERIES"
        )

        report_lines.append(
            "-" * 70
        )

        for _, row in failed_df.iterrows():

            report_lines.append(
                f"{row['study_id']} / "
                f"{row['series_id']} : "
                f"{row['error_type']} : "
                f"{row['error']}"
            )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(
                report_lines
            )
        )

    # ========================================================
    # 15. TERMINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 78)
    print("PART 2.20A COMPLETE")
    print("=" * 78)

    print(
        f"Validation series: {len(selected)}"
    )

    print(
        f"Processed series: {len(geometry_df)}"
    )

    print(
        f"Failed series: {len(failed_df)}"
    )

    print(
        f"Mapped points: {len(points_df)}"
    )

    print(
        f"Within-series LFNN/RFNN pairs: "
        f"{n_lr_series}"
    )

    print(
        "\nNative X left-greater rate: "
        + (
            f"{native_accuracy:.4f}"
            if not math.isnan(
                native_accuracy
            )
            else "N/A"
        )
    )

    print(
        "Patient physical X left-greater rate: "
        + (
            f"{physical_accuracy:.4f}"
            if not math.isnan(
                physical_accuracy
            )
            else "N/A"
        )
    )

    print(
        "Canonical X left-greater rate: "
        + (
            f"{canonical_accuracy:.4f}"
            if not math.isnan(
                canonical_accuracy
            )
            else "N/A"
        )
    )

    print("\nFiles written:")

    print(
        f"  {points_path}"
    )

    print(
        f"  {geometry_path}"
    )

    print(
        f"  {lr_path}"
    )

    print(
        f"  {failed_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {report_path}"
    )

    print("\nSafety checks:")
    print("  Training performed: NO")
    print("  Checkpoint modified: NO")
    print("  Dashboard modified: NO")
    print("  Pseudo-mask generated: NO")
    print("  Manual voxel ground truth fabricated: NO")

    print("=" * 78)


# ============================================================
# 16. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()