"""
===============================================================================
PHASE 3 - PART 30
SPINAL CANAL LOCALIZATION TRADE-OFF ANALYSIS
===============================================================================

Purpose
-------
Investigate why the Part 27 hybrid image-only localization strategy improved
overall segmentation Dice while reducing spinal-canal segmentation performance.

This script compares:

    Part 11  -> Original fixed-center model
    Part 27  -> Hybrid image-only localized model
    Part 29  -> Official paired comparison

The analysis focuses specifically on the spinal canal.

IMPORTANT
---------
Ground-truth masks are used ONLY for evaluation/retention analysis.
Ground-truth masks are NOT used to calculate image-only localization centers.

No model training is performed.
No model weights are modified.

===============================================================================
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import SimpleITK as sitk
except ImportError:
    raise ImportError(
        "SimpleITK is required.\n"
        "Install it with:\n"
        "pip install SimpleITK"
    )


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "segmentation_split"
    / "test"
)

IMAGE_DIR = TEST_DIR / "images"
MASK_DIR = TEST_DIR / "masks"

BASELINE_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "test_evaluation"
    / "test_case_results.csv"
)

PART27_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part28_test_evaluation"
    / "part27_test_case_results.csv"
)

PART29_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part29_paired_comparison"
    / "baseline_vs_part27_case_comparison.csv"
)

PART26_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "image_only_localization"
    / "image_only_localization_case_analysis.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part30_spinal_canal_tradeoff"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

PATCH_SIZE = np.array([96, 96, 96], dtype=int)

# Mask labels used throughout the project
BACKGROUND_LABEL = 0
VERTEBRAE_LABEL = 1
SPINAL_CANAL_LABEL = 2
DISC_LABEL = 3

SPINAL_CANAL_NAME = "Spinal Canal"

# Percentile parameters
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

# Small threshold for image foreground detection
NONZERO_THRESHOLD = 1e-6

# Minimum number of foreground voxels for an image-only localization region
MIN_FOREGROUND_VOXELS = 100

# Amount of smoothing used for robust intensity localization
GAUSSIAN_SIGMA = 3.0

# Reproducibility
np.random.seed(42)


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_case_name(name: str) -> str:
    """
    Normalize case/file names so that CSV names and image filenames match.
    """
    name = Path(str(name)).stem
    return name.strip()


def find_case_file(directory: Path, case_name: str) -> Path | None:
    """
    Find an image/mask file for a case.

    The project normally uses .mha, but several common medical-image formats
    are supported.
    """
    case_name = normalize_case_name(case_name)

    possible_extensions = [
        ".mha",
        ".mhd",
        ".nii",
        ".nii.gz",
        ".nrrd",
    ]

    # Exact matches first
    for ext in possible_extensions:
        candidate = directory / f"{case_name}{ext}"
        if candidate.exists():
            return candidate

    # Fallback: stem matching
    for path in directory.iterdir():
        if not path.is_file():
            continue

        stem = path.name
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        else:
            stem = path.stem

        if stem == case_name:
            return path

    return None


def load_medical_image(path: Path) -> np.ndarray:
    """
    Load a medical image using SimpleITK.

    SimpleITK returns arrays as:
        Z, Y, X
    """
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    return np.asarray(array)


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def percentile_normalize(
    image: np.ndarray,
    low: float = LOW_PERCENTILE,
    high: float = HIGH_PERCENTILE,
) -> np.ndarray:
    """
    Robust percentile normalization.
    """
    image = np.asarray(image, dtype=np.float32)

    finite = np.isfinite(image)

    if not np.any(finite):
        return np.zeros_like(image, dtype=np.float32)

    values = image[finite]

    lo = np.percentile(values, low)
    hi = np.percentile(values, high)

    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)

    normalized[~finite] = 0.0

    return normalized.astype(np.float32)


def resize_center_crop_or_pad(
    array: np.ndarray,
    target_shape: tuple[int, int, int] = (96, 96, 96),
    pad_value=0,
) -> np.ndarray:
    """
    Center crop or pad an array to the requested shape.

    This reproduces the fixed-center spatial handling used by the baseline
    pipeline.

    Array order:
        Z, Y, X
    """
    array = np.asarray(array)

    target = np.asarray(target_shape, dtype=int)

    output = np.full(
        target,
        pad_value,
        dtype=array.dtype,
    )

    source_shape = np.asarray(array.shape, dtype=int)

    source_start = np.maximum(
        (source_shape - target) // 2,
        0,
    )

    source_end = np.minimum(
        source_start + target,
        source_shape,
    )

    destination_start = np.maximum(
        (target - source_shape) // 2,
        0,
    )

    destination_end = destination_start + (
        source_end - source_start
    )

    source_slices = tuple(
        slice(int(a), int(b))
        for a, b in zip(source_start, source_end)
    )

    destination_slices = tuple(
        slice(int(a), int(b))
        for a, b in zip(destination_start, destination_end)
    )

    output[destination_slices] = array[source_slices]

    return output


# =============================================================================
# IMAGE-ONLY LOCALIZATION
# =============================================================================

def fixed_center_of_shape(shape: tuple[int, int, int]) -> np.ndarray:
    """
    Fixed image center.
    """
    shape = np.asarray(shape, dtype=np.float32)

    return (shape - 1.0) / 2.0


def image_nonzero_center(image: np.ndarray) -> np.ndarray:
    """
    Center of non-zero image content.
    """
    mask = np.abs(image) > NONZERO_THRESHOLD

    if np.count_nonzero(mask) < MIN_FOREGROUND_VOXELS:
        return fixed_center_of_shape(image.shape)

    coordinates = np.argwhere(mask)

    return coordinates.mean(axis=0)


def robust_intensity_center(image: np.ndarray) -> np.ndarray:
    """
    Robust intensity-based center.

    Uses high-intensity voxels after percentile normalization.

    This is deliberately image-only:
    no ground-truth mask is involved.
    """
    image = np.asarray(image, dtype=np.float32)

    nonzero = image[image > NONZERO_THRESHOLD]

    if nonzero.size < MIN_FOREGROUND_VOXELS:
        return fixed_center_of_shape(image.shape)

    threshold = np.percentile(nonzero, 65.0)

    mask = image >= threshold

    if np.count_nonzero(mask) < MIN_FOREGROUND_VOXELS:
        return fixed_center_of_shape(image.shape)

    coordinates = np.argwhere(mask)

    return coordinates.mean(axis=0)


def intensity_weighted_center(image: np.ndarray) -> np.ndarray:
    """
    Intensity-weighted center of the image.

    No ground-truth information is used.
    """
    image = np.asarray(image, dtype=np.float64)

    positive = np.clip(image, 0.0, None)

    total = positive.sum()

    if total <= 1e-8:
        return fixed_center_of_shape(image.shape)

    grid = np.indices(image.shape, dtype=np.float64)

    center = np.array(
        [
            (grid[axis] * positive).sum() / total
            for axis in range(3)
        ],
        dtype=np.float64,
    )

    return center


def gaussian_smooth_3d(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Lightweight 3D Gaussian smoothing.

    Uses scipy if available. Falls back to the original image.
    """
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(
            image,
            sigma=sigma,
        )
    except Exception:
        return image


def hybrid_image_center(image: np.ndarray) -> np.ndarray:
    """
    Hybrid image-only localization.

    Combines:
        1. non-zero image center
        2. robust intensity center
        3. intensity-weighted center

    Ground-truth masks are NEVER used here.

    The formulation is intentionally robust for medical MRI volumes and
    avoids relying on a single intensity heuristic.
    """
    nonzero_center = image_nonzero_center(image)

    robust_center = robust_intensity_center(image)

    weighted_center = intensity_weighted_center(image)

    smoothed = gaussian_smooth_3d(image, GAUSSIAN_SIGMA)

    smoothed_center = intensity_weighted_center(
        percentile_normalize(smoothed)
    )

    centers = np.vstack(
        [
            nonzero_center,
            robust_center,
            weighted_center,
            smoothed_center,
        ]
    )

    # Median coordinate is more robust than a simple mean.
    center = np.median(
        centers,
        axis=0,
    )

    return center


# =============================================================================
# CROP CENTER AND RETENTION
# =============================================================================

def clamp_crop_center(
    center: np.ndarray,
    shape: tuple[int, int, int],
    patch_size: np.ndarray = PATCH_SIZE,
) -> np.ndarray:
    """
    Clamp a crop center so the crop remains inside the source volume as much
    as possible.
    """
    shape = np.asarray(shape, dtype=np.float32)
    patch_size = np.asarray(patch_size, dtype=np.float32)

    minimum = patch_size / 2.0

    maximum = shape - patch_size / 2.0 - 1.0

    maximum = np.maximum(maximum, minimum)

    return np.clip(
        center,
        minimum,
        maximum,
    )


def crop_bounds_from_center(
    center: np.ndarray,
    shape: tuple[int, int, int],
    patch_size: np.ndarray = PATCH_SIZE,
):
    """
    Calculate crop bounds.
    """
    shape = np.asarray(shape, dtype=int)
    patch_size = np.asarray(patch_size, dtype=int)

    center = clamp_crop_center(
        center,
        tuple(shape),
        patch_size,
    )

    start = np.floor(
        center - patch_size / 2.0
    ).astype(int)

    end = start + patch_size

    # Adjust if required
    for axis in range(3):

        if start[axis] < 0:
            end[axis] -= start[axis]
            start[axis] = 0

        if end[axis] > shape[axis]:
            shift = end[axis] - shape[axis]
            start[axis] -= shift
            end[axis] = shape[axis]

        start[axis] = max(start[axis], 0)

    return start, end


def crop_array(
    array: np.ndarray,
    center: np.ndarray,
    patch_size: np.ndarray = PATCH_SIZE,
) -> np.ndarray:
    """
    Crop or pad an array around the requested center.
    """
    start, end = crop_bounds_from_center(
        center,
        array.shape,
        patch_size,
    )

    cropped = array[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
    ]

    if cropped.shape == tuple(patch_size):
        return cropped

    return resize_center_crop_or_pad(
        cropped,
        tuple(patch_size),
        pad_value=0,
    )


def class_retention(
    mask: np.ndarray,
    center: np.ndarray,
    class_label: int,
) -> tuple[float, int, int]:
    """
    Calculate how much ground-truth class content remains inside a crop.

    Returns:
        retention
        original_voxels
        retained_voxels
    """
    class_mask = mask == class_label

    original_voxels = int(class_mask.sum())

    if original_voxels == 0:
        return np.nan, 0, 0

    cropped = crop_array(
        class_mask.astype(np.uint8),
        center,
        PATCH_SIZE,
    )

    retained_voxels = int(
        np.count_nonzero(cropped)
    )

    retention = retained_voxels / original_voxels

    return (
        float(retention),
        original_voxels,
        retained_voxels,
    )


# =============================================================================
# SPINAL CANAL DICE HELPERS
# =============================================================================

def dice_from_values(
    baseline_dice: float,
    part27_dice: float,
) -> tuple[float, float]:
    """
    Return values safely.
    """
    return (
        safe_float(baseline_dice),
        safe_float(part27_dice),
    )


# =============================================================================
# T2 SPACE DETECTION
# =============================================================================

def is_t2_space(case_name: str) -> bool:
    """
    Identify T2 SPACE cases.
    """
    return "t2_space" in case_name.lower()


def classify_dice_change(change: float) -> str:
    if not np.isfinite(change):
        return "Unknown"

    if change > 0.02:
        return "Improved"

    if change < -0.02:
        return "Degraded"

    return "Stable"


# =============================================================================
# LOAD RESULTS
# =============================================================================

def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    print_section("LOADING OFFICIAL RESULTS")

    required_files = [
        BASELINE_RESULTS,
        PART27_RESULTS,
        PART29_RESULTS,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required result file not found:\n{path}"
            )

    baseline = pd.read_csv(BASELINE_RESULTS)
    part27 = pd.read_csv(PART27_RESULTS)
    paired = pd.read_csv(PART29_RESULTS)

    print(f"Baseline cases : {len(baseline)}")
    print(f"Part 27 cases  : {len(part27)}")
    print(f"Part 29 cases  : {len(paired)}")

    return baseline, part27


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():

    start_time = time.time()

    print_header(
        "PHASE 3 - PART 30\n"
        "SPINAL CANAL LOCALIZATION TRADE-OFF ANALYSIS"
    )

    print()
    print("PROJECT ROOT")
    print(PROJECT_ROOT)

    print()
    print("TEST DIRECTORY")
    print(TEST_DIR)

    print()
    print("IMAGE DIRECTORY")
    print(IMAGE_DIR)

    print()
    print("MASK DIRECTORY")
    print(MASK_DIR)

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    print()
    print("GROUND-TRUTH USED FOR LOCALIZATION: NO")
    print("GROUND-TRUTH USED FOR EVALUATION: YES")
    print("MODEL TRAINING: NO")
    print("MODEL WEIGHTS MODIFIED: NO")

    # -------------------------------------------------------------------------
    # Validate directories
    # -------------------------------------------------------------------------

    print_section("VALIDATING DATA DIRECTORIES")

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found:\n{TEST_DIR}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    if not MASK_DIR.exists():
        raise FileNotFoundError(
            f"Mask directory not found:\n{MASK_DIR}"
        )

    # -------------------------------------------------------------------------
    # Load official results
    # -------------------------------------------------------------------------

    baseline, part27 = load_results()

    # -------------------------------------------------------------------------
    # Normalize filenames
    # -------------------------------------------------------------------------

    baseline["case"] = baseline["file"].apply(
        normalize_case_name
    )

    part27["case"] = part27["file"].apply(
        normalize_case_name
    )

    common_cases = sorted(
        set(baseline["case"])
        &
        set(part27["case"])
    )

    print_section("CASE MATCHING")

    print("Baseline cases :", len(baseline))
    print("Part 27 cases  :", len(part27))
    print("Common cases   :", len(common_cases))

    if len(common_cases) == 0:
        raise RuntimeError(
            "No common cases were found between baseline and Part 27."
        )

    baseline = baseline.set_index("case")
    part27 = part27.set_index("case")

    # -------------------------------------------------------------------------
    # Detect metric columns
    # -------------------------------------------------------------------------

    baseline_spinal_col = "Spinal Canal_dice"
    part27_spinal_col = "Spinal Canal_dice"

    baseline_overall_col = "mean_foreground_dice"
    part27_overall_col = "mean_foreground_dice"

    for column in [
        baseline_spinal_col,
        baseline_overall_col,
    ]:
        if column not in baseline.columns:
            raise KeyError(
                f"Baseline column missing: {column}"
            )

    for column in [
        part27_spinal_col,
        part27_overall_col,
    ]:
        if column not in part27.columns:
            raise KeyError(
                f"Part 27 column missing: {column}"
            )

    # -------------------------------------------------------------------------
    # Main per-case table
    # -------------------------------------------------------------------------

    print_section(
        "SPINAL CANAL TRADE-OFF ANALYSIS"
    )

    rows = []

    for index, case in enumerate(common_cases, start=1):

        if index == 1 or index % 10 == 0 or index == len(common_cases):
            print(
                f"Analyzing {index:3d} / {len(common_cases)} : {case}"
            )

        baseline_row = baseline.loc[case]
        part27_row = part27.loc[case]

        baseline_dice = safe_float(
            baseline_row[baseline_spinal_col]
        )

        part27_dice = safe_float(
            part27_row[part27_spinal_col]
        )

        baseline_overall = safe_float(
            baseline_row[baseline_overall_col]
        )

        part27_overall = safe_float(
            part27_row[part27_overall_col]
        )

        spinal_change = (
            part27_dice - baseline_dice
            if np.isfinite(baseline_dice)
            and np.isfinite(part27_dice)
            else np.nan
        )

        overall_change = (
            part27_overall - baseline_overall
            if np.isfinite(baseline_overall)
            and np.isfinite(part27_overall)
            else np.nan
        )

        # -------------------------------------------------------------
        # Load image
        # -------------------------------------------------------------

        image_path = find_case_file(
            IMAGE_DIR,
            case,
        )

        mask_path = find_case_file(
            MASK_DIR,
            case,
        )

        if image_path is None or mask_path is None:

            rows.append(
                {
                    "file": case,
                    "baseline_spinal_canal_dice": baseline_dice,
                    "part27_spinal_canal_dice": part27_dice,
                    "spinal_canal_dice_change": spinal_change,
                    "baseline_overall_dice": baseline_overall,
                    "part27_overall_dice": part27_overall,
                    "overall_dice_change": overall_change,
                    "image_found": image_path is not None,
                    "mask_found": mask_path is not None,
                    "t2_space": is_t2_space(case),
                    "analysis_status": "missing_file",
                }
            )

            continue

        try:

            image = load_medical_image(
                image_path
            )

            mask = load_medical_image(
                mask_path
            )

            image = np.asarray(
                image,
                dtype=np.float32,
            )

            mask = np.asarray(
                mask
            )

            if image.shape != mask.shape:
                rows.append(
                    {
                        "file": case,
                        "baseline_spinal_canal_dice": baseline_dice,
                        "part27_spinal_canal_dice": part27_dice,
                        "spinal_canal_dice_change": spinal_change,
                        "baseline_overall_dice": baseline_overall,
                        "part27_overall_dice": part27_overall,
                        "overall_dice_change": overall_change,
                        "image_found": True,
                        "mask_found": True,
                        "t2_space": is_t2_space(case),
                        "analysis_status": "shape_mismatch",
                        "original_z": image.shape[0],
                        "original_y": image.shape[1],
                        "original_x": image.shape[2],
                    }
                )

                continue

            # -------------------------------------------------------------
            # Normalize image
            # -------------------------------------------------------------

            normalized_image = percentile_normalize(
                image
            )

            # -------------------------------------------------------------
            # Fixed center
            # -------------------------------------------------------------

            fixed_center = fixed_center_of_shape(
                image.shape
            )

            fixed_center = clamp_crop_center(
                fixed_center,
                image.shape,
                PATCH_SIZE,
            )

            # -------------------------------------------------------------
            # Hybrid image-only center
            # -------------------------------------------------------------

            hybrid_center = hybrid_image_center(
                normalized_image
            )

            hybrid_center = clamp_crop_center(
                hybrid_center,
                image.shape,
                PATCH_SIZE,
            )

            # -------------------------------------------------------------
            # Localization displacement
            # -------------------------------------------------------------

            center_shift = float(
                np.linalg.norm(
                    hybrid_center - fixed_center
                )
            )

            axis_shift_z = float(
                hybrid_center[0] - fixed_center[0]
            )

            axis_shift_y = float(
                hybrid_center[1] - fixed_center[1]
            )

            axis_shift_x = float(
                hybrid_center[2] - fixed_center[2]
            )

            # -------------------------------------------------------------
            # Spinal canal retention
            # -------------------------------------------------------------

            (
                fixed_canal_retention,
                original_canal_voxels,
                fixed_canal_voxels,
            ) = class_retention(
                mask,
                fixed_center,
                SPINAL_CANAL_LABEL,
            )

            (
                hybrid_canal_retention,
                _,
                hybrid_canal_voxels,
            ) = class_retention(
                mask,
                hybrid_center,
                SPINAL_CANAL_LABEL,
            )

            # -------------------------------------------------------------
            # Other anatomical retention
            # -------------------------------------------------------------

            (
                fixed_vertebra_retention,
                original_vertebra_voxels,
                fixed_vertebra_voxels,
            ) = class_retention(
                mask,
                fixed_center,
                VERTEBRAE_LABEL,
            )

            (
                hybrid_vertebra_retention,
                _,
                hybrid_vertebra_voxels,
            ) = class_retention(
                mask,
                hybrid_center,
                VERTEBRAE_LABEL,
            )

            (
                fixed_disc_retention,
                original_disc_voxels,
                fixed_disc_voxels,
            ) = class_retention(
                mask,
                fixed_center,
                DISC_LABEL,
            )

            (
                hybrid_disc_retention,
                _,
                hybrid_disc_voxels,
            ) = class_retention(
                mask,
                hybrid_center,
                DISC_LABEL,
            )

            # -------------------------------------------------------------
            # Combined retention
            # -------------------------------------------------------------

            retention_values = [
                hybrid_vertebra_retention,
                hybrid_canal_retention,
                hybrid_disc_retention,
            ]

            valid_retention = [
                value
                for value in retention_values
                if np.isfinite(value)
            ]

            if valid_retention:
                hybrid_minimum_retention = float(
                    min(valid_retention)
                )

                hybrid_mean_retention = float(
                    np.mean(valid_retention)
                )
            else:
                hybrid_minimum_retention = np.nan
                hybrid_mean_retention = np.nan

            fixed_retention_values = [
                fixed_vertebra_retention,
                fixed_canal_retention,
                fixed_disc_retention,
            ]

            valid_fixed = [
                value
                for value in fixed_retention_values
                if np.isfinite(value)
            ]

            if valid_fixed:
                fixed_minimum_retention = float(
                    min(valid_fixed)
                )

                fixed_mean_retention = float(
                    np.mean(valid_fixed)
                )
            else:
                fixed_minimum_retention = np.nan
                fixed_mean_retention = np.nan

            # -------------------------------------------------------------
            # Determine trade-off category
            # -------------------------------------------------------------

            if (
                np.isfinite(spinal_change)
                and spinal_change < -0.05
            ):
                tradeoff_category = (
                    "Spinal canal degradation"
                )

            elif (
                np.isfinite(spinal_change)
                and spinal_change > 0.05
            ):
                tradeoff_category = (
                    "Spinal canal improvement"
                )

            else:
                tradeoff_category = (
                    "Spinal canal stable"
                )

            rows.append(
                {
                    "file": case,

                    "t2_space": is_t2_space(case),

                    "original_z": image.shape[0],
                    "original_y": image.shape[1],
                    "original_x": image.shape[2],

                    "baseline_overall_dice": baseline_overall,
                    "part27_overall_dice": part27_overall,
                    "overall_dice_change": overall_change,

                    "baseline_spinal_canal_dice": baseline_dice,
                    "part27_spinal_canal_dice": part27_dice,
                    "spinal_canal_dice_change": spinal_change,

                    "fixed_center_z": fixed_center[0],
                    "fixed_center_y": fixed_center[1],
                    "fixed_center_x": fixed_center[2],

                    "hybrid_center_z": hybrid_center[0],
                    "hybrid_center_y": hybrid_center[1],
                    "hybrid_center_x": hybrid_center[2],

                    "center_shift_voxels": center_shift,

                    "axis_shift_z": axis_shift_z,
                    "axis_shift_y": axis_shift_y,
                    "axis_shift_x": axis_shift_x,

                    "original_canal_voxels": original_canal_voxels,
                    "fixed_canal_voxels": fixed_canal_voxels,
                    "hybrid_canal_voxels": hybrid_canal_voxels,

                    "fixed_canal_retention": fixed_canal_retention,
                    "hybrid_canal_retention": hybrid_canal_retention,

                    "canal_retention_change": (
                        hybrid_canal_retention
                        - fixed_canal_retention
                        if np.isfinite(
                            hybrid_canal_retention
                        )
                        and np.isfinite(
                            fixed_canal_retention
                        )
                        else np.nan
                    ),

                    "original_vertebra_voxels": (
                        original_vertebra_voxels
                    ),

                    "fixed_vertebra_voxels": (
                        fixed_vertebra_voxels
                    ),

                    "hybrid_vertebra_voxels": (
                        hybrid_vertebra_voxels
                    ),

                    "fixed_vertebra_retention": (
                        fixed_vertebra_retention
                    ),

                    "hybrid_vertebra_retention": (
                        hybrid_vertebra_retention
                    ),

                    "original_disc_voxels": (
                        original_disc_voxels
                    ),

                    "fixed_disc_voxels": (
                        fixed_disc_voxels
                    ),

                    "hybrid_disc_voxels": (
                        hybrid_disc_voxels
                    ),

                    "fixed_disc_retention": (
                        fixed_disc_retention
                    ),

                    "hybrid_disc_retention": (
                        hybrid_disc_retention
                    ),

                    "fixed_minimum_class_retention": (
                        fixed_minimum_retention
                    ),

                    "hybrid_minimum_class_retention": (
                        hybrid_minimum_retention
                    ),

                    "fixed_mean_class_retention": (
                        fixed_mean_retention
                    ),

                    "hybrid_mean_class_retention": (
                        hybrid_mean_retention
                    ),

                    "tradeoff_category": (
                        tradeoff_category
                    ),

                    "analysis_status": "success",
                }
            )

        except Exception as exc:

            rows.append(
                {
                    "file": case,
                    "baseline_spinal_canal_dice": baseline_dice,
                    "part27_spinal_canal_dice": part27_dice,
                    "spinal_canal_dice_change": spinal_change,
                    "baseline_overall_dice": baseline_overall,
                    "part27_overall_dice": part27_overall,
                    "overall_dice_change": overall_change,
                    "t2_space": is_t2_space(case),
                    "analysis_status": (
                        f"error: {type(exc).__name__}: {exc}"
                    ),
                }
            )

    results = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Convert numeric columns
    # -------------------------------------------------------------------------

    numeric_columns = [
        column
        for column in results.columns
        if column not in [
            "file",
            "tradeoff_category",
            "analysis_status",
        ]
    ]

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Save complete analysis
    # -------------------------------------------------------------------------

    print_section(
        "SAVING CASE-LEVEL ANALYSIS"
    )

    case_output = (
        OUTPUT_DIR
        / "spinal_canal_tradeoff_case_analysis.csv"
    )

    results.to_csv(
        case_output,
        index=False,
    )

    print(f"Saved: {case_output}")

    # -------------------------------------------------------------------------
    # Valid successful cases
    # -------------------------------------------------------------------------

    valid = results[
        results["analysis_status"] == "success"
    ].copy()

    print()
    print(
        f"Successful spatial analyses: {len(valid)}"
    )

    # -------------------------------------------------------------------------
    # Overall summary
    # -------------------------------------------------------------------------

    print_section(
        "SPINAL CANAL PERFORMANCE COMPARISON"
    )

    baseline_canal = valid[
        "baseline_spinal_canal_dice"
    ]

    part27_canal = valid[
        "part27_spinal_canal_dice"
    ]

    baseline_mean = float(
        baseline_canal.mean()
    )

    part27_mean = float(
        part27_canal.mean()
    )

    mean_change = float(
        (part27_canal - baseline_canal).mean()
    )

    baseline_median = float(
        baseline_canal.median()
    )

    part27_median = float(
        part27_canal.median()
    )

    print(
        f"Baseline Spinal Canal Dice : {baseline_mean:.6f}"
    )

    print(
        f"Part 27 Spinal Canal Dice  : {part27_mean:.6f}"
    )

    print(
        f"Mean Dice change            : {mean_change:+.6f}"
    )

    print(
        f"Baseline median Dice        : {baseline_median:.6f}"
    )

    print(
        f"Part 27 median Dice         : {part27_median:.6f}"
    )

    # -------------------------------------------------------------------------
    # Improved / degraded cases
    # -------------------------------------------------------------------------

    print_section(
        "CASE-WISE SPINAL CANAL CHANGE"
    )

    improved = valid[
        valid["spinal_canal_dice_change"] > 0.02
    ]

    stable = valid[
        valid["spinal_canal_dice_change"].abs() <= 0.02
    ]

    degraded = valid[
        valid["spinal_canal_dice_change"] < -0.02
    ]

    print(
        f"Improved cases : {len(improved)} "
        f"({len(improved) / len(valid) * 100:.2f}%)"
    )

    print(
        f"Stable cases   : {len(stable)} "
        f"({len(stable) / len(valid) * 100:.2f}%)"
    )

    print(
        f"Degraded cases: {len(degraded)} "
        f"({len(degraded) / len(valid) * 100:.2f}%)"
    )

    # -------------------------------------------------------------------------
    # Worst spinal canal degradations
    # -------------------------------------------------------------------------

    print_section(
        "TOP SPINAL CANAL DEGRADATIONS"
    )

    worst = (
        valid
        .sort_values(
            "spinal_canal_dice_change",
            ascending=True,
        )
        .head(15)
    )

    display_columns = [
        "file",
        "baseline_spinal_canal_dice",
        "part27_spinal_canal_dice",
        "spinal_canal_dice_change",
        "fixed_canal_retention",
        "hybrid_canal_retention",
        "canal_retention_change",
        "center_shift_voxels",
        "t2_space",
    ]

    print(
        worst[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    worst.to_csv(
        OUTPUT_DIR
        / "spinal_canal_degradation_cases.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Best spinal canal improvements
    # -------------------------------------------------------------------------

    print_section(
        "TOP SPINAL CANAL IMPROVEMENTS"
    )

    best = (
        valid
        .sort_values(
            "spinal_canal_dice_change",
            ascending=False,
        )
        .head(15)
    )

    print(
        best[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    best.to_csv(
        OUTPUT_DIR
        / "spinal_canal_improvement_cases.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Retention comparison
    # -------------------------------------------------------------------------

    print_section(
        "SPINAL CANAL CROP RETENTION"
    )

    fixed_retention_mean = float(
        valid["fixed_canal_retention"].mean()
    )

    hybrid_retention_mean = float(
        valid["hybrid_canal_retention"].mean()
    )

    retention_change_mean = (
        hybrid_retention_mean
        - fixed_retention_mean
    )

    print(
        f"Fixed-center canal retention  : "
        f"{fixed_retention_mean:.6f}"
    )

    print(
        f"Hybrid-center canal retention : "
        f"{hybrid_retention_mean:.6f}"
    )

    print(
        f"Retention change               : "
        f"{retention_change_mean:+.6f}"
    )

    # -------------------------------------------------------------------------
    # Localization shift
    # -------------------------------------------------------------------------

    print_section(
        "LOCALIZATION SHIFT"
    )

    print(
        f"Mean center shift   : "
        f"{valid['center_shift_voxels'].mean():.4f} voxels"
    )

    print(
        f"Median center shift : "
        f"{valid['center_shift_voxels'].median():.4f} voxels"
    )

    print(
        f"Maximum center shift: "
        f"{valid['center_shift_voxels'].max():.4f} voxels"
    )

    # -------------------------------------------------------------------------
    # Correlation analysis
    # -------------------------------------------------------------------------

    print_section(
        "CORRELATION ANALYSIS"
    )

    correlations = {}

    correlation_pairs = [
        (
            "center_shift_vs_canal_dice_change",
            "center_shift_voxels",
            "spinal_canal_dice_change",
        ),
        (
            "canal_retention_vs_canal_dice_change",
            "hybrid_canal_retention",
            "spinal_canal_dice_change",
        ),
        (
            "canal_retention_change_vs_canal_dice_change",
            "canal_retention_change",
            "spinal_canal_dice_change",
        ),
        (
            "overall_change_vs_canal_change",
            "overall_dice_change",
            "spinal_canal_dice_change",
        ),
    ]

    for name, x_column, y_column in correlation_pairs:

        pair = valid[
            [
                x_column,
                y_column,
            ]
        ].dropna()

        if len(pair) >= 3:

            correlation = pair[
                x_column
            ].corr(
                pair[y_column]
            )

            correlations[name] = float(
                correlation
            )

            print(
                f"{name:45s}: "
                f"{correlation:+.6f}"
            )

        else:

            correlations[name] = np.nan

            print(
                f"{name:45s}: insufficient data"
            )

    # -------------------------------------------------------------------------
    # T2 SPACE analysis
    # -------------------------------------------------------------------------

    print_section(
        "T2 SPACE VS NON-T2 SPACE ANALYSIS"
    )

    t2_space = valid[
        valid["t2_space"] == True
    ].copy()

    non_t2 = valid[
        valid["t2_space"] == False
    ].copy()

    def print_group_stats(
        name: str,
        group: pd.DataFrame,
    ):

        print()
        print(name)
        print("-" * len(name))

        print(
            f"Cases                    : {len(group)}"
        )

        if len(group) == 0:
            return

        print(
            f"Baseline canal Dice      : "
            f"{group['baseline_spinal_canal_dice'].mean():.6f}"
        )

        print(
            f"Part 27 canal Dice       : "
            f"{group['part27_spinal_canal_dice'].mean():.6f}"
        )

        print(
            f"Mean canal Dice change   : "
            f"{group['spinal_canal_dice_change'].mean():+.6f}"
        )

        print(
            f"Mean canal retention     : "
            f"{group['hybrid_canal_retention'].mean():.6f}"
        )

        print(
            f"Mean center shift        : "
            f"{group['center_shift_voxels'].mean():.4f}"
        )

    print_group_stats(
        "T2 SPACE",
        t2_space,
    )

    print_group_stats(
        "NON-T2 SPACE",
        non_t2,
    )

    # -------------------------------------------------------------------------
    # Retention threshold analysis
    # -------------------------------------------------------------------------

    print_section(
        "CANAL RETENTION THRESHOLD ANALYSIS"
    )

    thresholds = [
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.50,
    ]

    threshold_rows = []

    for threshold in thresholds:

        subset = valid[
            valid["hybrid_canal_retention"]
            < threshold
        ]

        threshold_rows.append(
            {
                "retention_threshold": threshold,
                "cases_below_threshold": len(subset),
                "percentage_cases": (
                    len(subset)
                    / len(valid)
                    * 100.0
                ),
                "mean_spinal_canal_dice": (
                    subset[
                        "part27_spinal_canal_dice"
                    ].mean()
                    if len(subset)
                    else np.nan
                ),
                "mean_dice_change": (
                    subset[
                        "spinal_canal_dice_change"
                    ].mean()
                    if len(subset)
                    else np.nan
                ),
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    print(
        threshold_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    threshold_df.to_csv(
        OUTPUT_DIR
        / "spinal_canal_retention_threshold_analysis.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Trade-off categories
    # -------------------------------------------------------------------------

    print_section(
        "TRADE-OFF CATEGORY DISTRIBUTION"
    )

    category_counts = (
        valid[
            "tradeoff_category"
        ]
        .value_counts()
        .rename_axis("category")
        .reset_index(
            name="cases"
        )
    )

    category_counts[
        "percentage"
    ] = (
        category_counts["cases"]
        / len(valid)
        * 100.0
    )

    print(
        category_counts.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    category_counts.to_csv(
        OUTPUT_DIR
        / "spinal_canal_tradeoff_categories.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------

    summary = {
        "part": 30,
        "title": (
            "Spinal Canal Localization Trade-off Analysis"
        ),

        "test_cases": int(len(valid)),

        "baseline_spinal_canal_dice": baseline_mean,
        "part27_spinal_canal_dice": part27_mean,
        "mean_spinal_canal_dice_change": mean_change,

        "baseline_spinal_canal_median": baseline_median,
        "part27_spinal_canal_median": part27_median,

        "improved_cases": int(len(improved)),
        "stable_cases": int(len(stable)),
        "degraded_cases": int(len(degraded)),

        "fixed_canal_retention_mean": fixed_retention_mean,
        "hybrid_canal_retention_mean": hybrid_retention_mean,
        "mean_canal_retention_change": (
            retention_change_mean
        ),

        "mean_center_shift_voxels": float(
            valid[
                "center_shift_voxels"
            ].mean()
        ),

        "median_center_shift_voxels": float(
            valid[
                "center_shift_voxels"
            ].median()
        ),

        "maximum_center_shift_voxels": float(
            valid[
                "center_shift_voxels"
            ].max()
        ),

        "correlations": correlations,

        "t2_space_cases": int(
            len(t2_space)
        ),

        "non_t2_space_cases": int(
            len(non_t2)
        ),

        "t2_space_mean_canal_change": (
            float(
                t2_space[
                    "spinal_canal_dice_change"
                ].mean()
            )
            if len(t2_space)
            else None
        ),

        "non_t2_space_mean_canal_change": (
            float(
                non_t2[
                    "spinal_canal_dice_change"
                ].mean()
            )
            if len(non_t2)
            else None
        ),

        "ground_truth_used_for_localization": False,
        "ground_truth_used_for_evaluation": True,
        "training_performed": False,
        "model_weights_modified": False,

        "interpretation": (
            "Part 30 investigates whether the reduction in spinal-canal "
            "Dice observed in Part 29 is associated with image-only "
            "localization and crop retention."
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "phase3_part30_spinal_canal_tradeoff_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print()
    print(
        f"Saved: {summary_path}"
    )

    # =========================================================================
    # CHARTS
    # =========================================================================

    print_section(
        "CREATING PART 30 CHARTS"
    )

    # -------------------------------------------------------------------------
    # Chart 1: baseline vs Part 27 spinal canal Dice
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        [
            valid[
                "baseline_spinal_canal_dice"
            ].dropna(),

            valid[
                "part27_spinal_canal_dice"
            ].dropna(),
        ],
        labels=[
            "Baseline Fixed-Center",
            "Part 27 Hybrid",
        ],
    )

    plt.ylabel(
        "Spinal Canal Dice"
    )

    plt.title(
        "Baseline vs Part 27 Spinal Canal Dice"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    chart1 = (
        OUTPUT_DIR
        / "baseline_vs_part27_spinal_canal_dice.png"
    )

    plt.savefig(
        chart1,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {chart1}")

    # -------------------------------------------------------------------------
    # Chart 2: paired spinal canal change
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    sorted_change = valid.sort_values(
        "spinal_canal_dice_change"
    )

    plt.bar(
        range(len(sorted_change)),
        sorted_change[
            "spinal_canal_dice_change"
        ],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Test Cases"
    )

    plt.ylabel(
        "Spinal Canal Dice Change"
    )

    plt.title(
        "Case-wise Spinal Canal Dice Change"
    )

    plt.tight_layout()

    chart2 = (
        OUTPUT_DIR
        / "casewise_spinal_canal_dice_change.png"
    )

    plt.savefig(
        chart2,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {chart2}")

    # -------------------------------------------------------------------------
    # Chart 3: retention vs Dice change
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        valid[
            "hybrid_canal_retention"
        ],
        valid[
            "spinal_canal_dice_change"
        ],
        alpha=0.75,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Hybrid Crop Spinal Canal Retention"
    )

    plt.ylabel(
        "Spinal Canal Dice Change"
    )

    plt.title(
        "Spinal Canal Retention vs Dice Change"
    )

    plt.grid(
        alpha=0.3,
    )

    plt.tight_layout()

    chart3 = (
        OUTPUT_DIR
        / "spinal_canal_retention_vs_dice_change.png"
    )

    plt.savefig(
        chart3,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {chart3}")

    # -------------------------------------------------------------------------
    # Chart 4: localization shift vs Dice change
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        valid[
            "center_shift_voxels"
        ],
        valid[
            "spinal_canal_dice_change"
        ],
        alpha=0.75,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Hybrid Center Shift (voxels)"
    )

    plt.ylabel(
        "Spinal Canal Dice Change"
    )

    plt.title(
        "Localization Shift vs Spinal Canal Dice Change"
    )

    plt.grid(
        alpha=0.3,
    )

    plt.tight_layout()

    chart4 = (
        OUTPUT_DIR
        / "localization_shift_vs_spinal_canal_change.png"
    )

    plt.savefig(
        chart4,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {chart4}")

    # -------------------------------------------------------------------------
    # Chart 5: fixed vs hybrid canal retention
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        [
            valid[
                "fixed_canal_retention"
            ].dropna(),

            valid[
                "hybrid_canal_retention"
            ].dropna(),
        ],
        labels=[
            "Fixed Center",
            "Hybrid Image-Only",
        ],
    )

    plt.ylabel(
        "Spinal Canal Ground-Truth Retention"
    )

    plt.title(
        "Fixed vs Hybrid Crop Spinal Canal Retention"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    chart5 = (
        OUTPUT_DIR
        / "fixed_vs_hybrid_spinal_canal_retention.png"
    )

    plt.savefig(
        chart5,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {chart5}")

    # -------------------------------------------------------------------------
    # Chart 6: T2 SPACE vs non-T2 SPACE
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    group_values = []

    group_labels = []

    if len(t2_space) > 0:
        group_values.append(
            t2_space[
                "spinal_canal_dice_change"
            ].dropna()
        )

        group_labels.append(
            "T2 SPACE"
        )

    if len(non_t2) > 0:
        group_values.append(
            non_t2[
                "spinal_canal_dice_change"
            ].dropna()
        )

        group_labels.append(
            "Non-T2 SPACE"
        )

    if group_values:

        plt.boxplot(
            group_values,
            labels=group_labels,
        )

        plt.axhline(
            0,
            linewidth=1,
        )

        plt.ylabel(
            "Spinal Canal Dice Change"
        )

        plt.title(
            "Spinal Canal Change by MRI Sequence"
        )

        plt.grid(
            axis="y",
            alpha=0.3,
        )

        plt.tight_layout()

        chart6 = (
            OUTPUT_DIR
            / "t2_space_vs_non_t2_spinal_canal_change.png"
        )

        plt.savefig(
            chart6,
            dpi=200,
        )

        plt.close()

        print(f"Saved: {chart6}")

    # =========================================================================
    # REPORT
    # =========================================================================

    print_section(
        "CREATING PART 30 REPORT"
    )

    report_path = (
        OUTPUT_DIR
        / "phase3_part30_spinal_canal_tradeoff_report.txt"
    )

    report_lines = []

    report_lines.append(
        "PHASE 3 - PART 30\n"
    )

    report_lines.append(
        "SPINAL CANAL LOCALIZATION TRADE-OFF ANALYSIS\n"
    )

    report_lines.append(
        "=" * 78
    )

    report_lines.append(
        "\nOBJECTIVE\n"
    )

    report_lines.append(
        "Part 30 investigates the reduction in spinal-canal segmentation "
        "performance observed after Part 27 image-only hybrid localization.\n"
    )

    report_lines.append(
        "\nEXPERIMENTAL CONDITIONS\n"
    )

    report_lines.append(
        "Ground-truth used for localization: NO\n"
    )

    report_lines.append(
        "Ground-truth used for evaluation: YES\n"
    )

    report_lines.append(
        "Training performed: NO\n"
    )

    report_lines.append(
        "Model weights modified: NO\n"
    )

    report_lines.append(
        "\nDATASET\n"
    )

    report_lines.append(
        f"Common paired test cases: {len(valid)}\n"
    )

    report_lines.append(
        "\nSPINAL CANAL PERFORMANCE\n"
    )

    report_lines.append(
        f"Baseline Dice: {baseline_mean:.6f}\n"
    )

    report_lines.append(
        f"Part 27 Dice: {part27_mean:.6f}\n"
    )

    report_lines.append(
        f"Mean Dice change: {mean_change:+.6f}\n"
    )

    report_lines.append(
        f"Baseline median Dice: {baseline_median:.6f}\n"
    )

    report_lines.append(
        f"Part 27 median Dice: {part27_median:.6f}\n"
    )

    report_lines.append(
        "\nCASE-WISE CHANGE\n"
    )

    report_lines.append(
        f"Improved: {len(improved)}\n"
    )

    report_lines.append(
        f"Stable: {len(stable)}\n"
    )

    report_lines.append(
        f"Degraded: {len(degraded)}\n"
    )

    report_lines.append(
        "\nCROP RETENTION\n"
    )

    report_lines.append(
        f"Fixed-center canal retention: "
        f"{fixed_retention_mean:.6f}\n"
    )

    report_lines.append(
        f"Hybrid canal retention: "
        f"{hybrid_retention_mean:.6f}\n"
    )

    report_lines.append(
        f"Retention change: "
        f"{retention_change_mean:+.6f}\n"
    )

    report_lines.append(
        "\nLOCALIZATION SHIFT\n"
    )

    report_lines.append(
        f"Mean center shift: "
        f"{valid['center_shift_voxels'].mean():.4f} voxels\n"
    )

    report_lines.append(
        f"Median center shift: "
        f"{valid['center_shift_voxels'].median():.4f} voxels\n"
    )

    report_lines.append(
        f"Maximum center shift: "
        f"{valid['center_shift_voxels'].max():.4f} voxels\n"
    )

    report_lines.append(
        "\nCORRELATIONS\n"
    )

    for name, value in correlations.items():

        if np.isfinite(value):

            report_lines.append(
                f"{name}: {value:+.6f}\n"
            )

        else:

            report_lines.append(
                f"{name}: unavailable\n"
            )

    report_lines.append(
        "\nT2 SPACE ANALYSIS\n"
    )

    if len(t2_space):

        report_lines.append(
            f"T2 SPACE cases: {len(t2_space)}\n"
        )

        report_lines.append(
            "T2 SPACE mean spinal canal Dice change: "
            f"{t2_space['spinal_canal_dice_change'].mean():+.6f}\n"
        )

    else:

        report_lines.append(
            "No T2 SPACE cases detected.\n"
        )

    report_lines.append(
        "\nNON-T2 SPACE ANALYSIS\n"
    )

    if len(non_t2):

        report_lines.append(
            f"Non-T2 SPACE cases: {len(non_t2)}\n"
        )

        report_lines.append(
            "Non-T2 SPACE mean spinal canal Dice change: "
            f"{non_t2['spinal_canal_dice_change'].mean():+.6f}\n"
        )

    else:

        report_lines.append(
            "No non-T2 SPACE cases detected.\n"
        )

    report_lines.append(
        "\nINTERPRETATION\n"
    )

    if mean_change < 0:

        report_lines.append(
            "Part 27 produced lower mean spinal-canal Dice than the "
            "baseline model. This confirms the spinal-canal trade-off "
            "identified in Part 29.\n"
        )

    else:

        report_lines.append(
            "Part 27 did not reduce mean spinal-canal Dice in this "
            "spatial analysis. The Part 29 reduction should therefore "
            "be investigated through model prediction-level analysis.\n"
        )

    if (
        hybrid_retention_mean
        < fixed_retention_mean
    ):

        report_lines.append(
            "The hybrid image-only crop retains less spinal-canal "
            "ground-truth content on average than the fixed-center crop. "
            "This supports a localization-related explanation for at "
            "least part of the observed spinal-canal degradation.\n"
        )

    else:

        report_lines.append(
            "The hybrid crop does not reduce mean spinal-canal "
            "ground-truth retention relative to the fixed-center crop. "
            "Therefore, crop localization alone may not explain the "
            "observed spinal-canal degradation.\n"
        )

    if (
        correlations[
            "canal_retention_vs_canal_dice_change"
        ]
        is not None
        and np.isfinite(
            correlations[
                "canal_retention_vs_canal_dice_change"
            ]
        )
    ):

        correlation = correlations[
            "canal_retention_vs_canal_dice_change"
        ]

        if abs(correlation) >= 0.5:

            report_lines.append(
                f"The correlation between hybrid spinal-canal retention "
                f"and spinal-canal Dice change is {correlation:+.4f}, "
                "indicating a substantial relationship.\n"
            )

        else:

            report_lines.append(
                f"The correlation between hybrid spinal-canal retention "
                f"and spinal-canal Dice change is {correlation:+.4f}, "
                "which is relatively weak/moderate.\n"
            )

    report_lines.append(
        "\nRECOMMENDATION FOR NEXT EXPERIMENT\n"
    )

    report_lines.append(
        "Before performing additional model training, investigate a "
        "canal-aware image-only localization strategy. The strategy "
        "should preserve the overall gains from hybrid localization "
        "while improving spinal-canal retention without using "
        "ground-truth information during inference.\n"
    )

    report_lines.append(
        "\nPART 30 CONCLUSION\n"
    )

    report_lines.append(
        "Part 30 provides a quantitative assessment of the localization "
        "trade-off responsible for the spinal-canal performance change "
        "between the baseline and Part 27 models.\n"
    )

    report_lines.append(
        "\n" + "=" * 78 + "\n"
    )

    report_lines.append(
        "PHASE 3 - PART 30 COMPLETE\n"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.writelines(
            report_lines
        )

    print(
        f"Saved: {report_path}"
    )

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    elapsed_minutes = (
        time.time() - start_time
    ) / 60.0

    print_section(
        "PART 30 COMPLETE"
    )

    print(
        f"Cases analyzed              : {len(valid)}"
    )

    print(
        f"Baseline Canal Dice         : "
        f"{baseline_mean:.6f}"
    )

    print(
        f"Part 27 Canal Dice          : "
        f"{part27_mean:.6f}"
    )

    print(
        f"Canal Dice change           : "
        f"{mean_change:+.6f}"
    )

    print(
        f"Improved cases              : "
        f"{len(improved)}"
    )

    print(
        f"Stable cases                : "
        f"{len(stable)}"
    )

    print(
        f"Degraded cases              : "
        f"{len(degraded)}"
    )

    print(
        f"Mean center shift           : "
        f"{valid['center_shift_voxels'].mean():.4f} voxels"
    )

    print(
        f"Fixed canal retention       : "
        f"{fixed_retention_mean:.6f}"
    )

    print(
        f"Hybrid canal retention     : "
        f"{hybrid_retention_mean:.6f}"
    )

    print(
        f"Analysis time               : "
        f"{elapsed_minutes:.2f} minutes"
    )

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    print()
    print("=" * 78)
    print("PHASE 3 - PART 30 COMPLETE")
    print("=" * 78)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()