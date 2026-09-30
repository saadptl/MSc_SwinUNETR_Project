"""
==============================================================================
PHASE 3 - PART 32
SPINAL-CANAL-AWARE IMAGE-ONLY LOCALIZATION ANALYSIS
==============================================================================

Purpose
-------
Evaluate image-only localization strategies for the 3D spinal segmentation
problem, with special emphasis on preserving the spinal canal.

IMPORTANT
---------
Ground-truth masks are NEVER used to select the crop/localization.

Ground-truth masks are used ONLY AFTER localization to calculate retention
metrics for scientific evaluation.

No model training is performed.
No model weights are modified.

Strategies
----------
1. fixed_center
2. hybrid_image_center
3. intensity_weighted_center
4. canal_axis_image_center
5. canal_preserving_hybrid
6. multi_candidate_canal_center

Outputs
-------
outputs/segmentation/part32_spinal_canal_aware_localization/

    part32_case_analysis.csv
    part32_strategy_summary.csv
    part32_sequence_summary.csv
    part32_high_risk_cases.csv

    part32_overall_strategy_comparison.png
    part32_canal_retention_comparison.png
    part32_sequence_comparison.png
    part32_localization_error.png
    part32_canal_failure_cases.png

    phase3_part32_summary.json
    phase3_part32_report.txt
==============================================================================
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleITK as sitk


# =============================================================================
# CONFIGURATION
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part32_spinal_canal_aware_localization"
)

PART31_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part31_spinal_canal_aware_localization"
)

PART25_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "crop_strategy_comparison"
)

PATCH_SIZE = (96, 96, 96)

# Label mapping used throughout the project.
BACKGROUND = 0
VERTEBRAE = 1
SPINAL_CANAL = 2
INTERVERTEBRAL_DISC = 3

CLASS_NAMES = {
    VERTEBRAE: "Vertebrae",
    SPINAL_CANAL: "Spinal Canal",
    INTERVERTEBRAL_DISC: "Intervertebral Disc",
}

STRATEGIES = [
    "fixed_center",
    "hybrid_image_center",
    "intensity_weighted_center",
    "canal_axis_image_center",
    "canal_preserving_hybrid",
    "multi_candidate_canal_center",
]


# =============================================================================
# PRINTING
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
# FILE UTILITIES
# =============================================================================

def get_mha_files(directory: Path) -> Dict[str, Path]:
    """
    Return .mha files indexed by stem.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob("*.mha"))

    return {f.stem: f for f in files}


def infer_sequence(case_name: str) -> str:
    """
    Infer sequence from filename.
    """
    name = case_name.upper()

    if "T2_SPACE" in name:
        return "T2 SPACE"

    if "_T2" in name:
        return "T2"

    if "_T1" in name:
        return "T1"

    return "Unknown"


# =============================================================================
# IMAGE LOADING
# =============================================================================

def read_mha(path: Path) -> np.ndarray:
    """
    Read MHA and return NumPy array in Z,Y,X order.

    SimpleITK GetArrayFromImage returns:
        [z, y, x]
    """
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    return np.asarray(array)


# =============================================================================
# NORMALIZATION
# =============================================================================

def percentile_normalize(
    volume: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Robust percentile normalization.

    Returns values approximately in [0,1].
    """
    volume = np.asarray(volume, dtype=np.float32)

    finite = volume[np.isfinite(volume)]

    if finite.size == 0:
        return np.zeros_like(volume, dtype=np.float32)

    lo = np.percentile(finite, low_percentile)
    hi = np.percentile(finite, high_percentile)

    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)

    volume = np.clip(volume, lo, hi)

    volume = (volume - lo) / (hi - lo)

    return volume.astype(np.float32)


# =============================================================================
# IMAGE-ONLY ANATOMICAL SIGNAL
# =============================================================================

def calculate_nonzero_mask(
    image: np.ndarray,
    threshold_percentile: float = 5.0,
) -> np.ndarray:
    """
    Estimate foreground from image intensity only.

    No ground truth is used.
    """
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return np.zeros_like(image, dtype=bool)

    threshold = np.percentile(finite, threshold_percentile)

    return image > threshold


def calculate_body_center(
    image: np.ndarray,
) -> np.ndarray:
    """
    Estimate image foreground/body center using intensity only.

    Returns center in Z,Y,X coordinates.
    """
    mask = calculate_nonzero_mask(image)

    coords = np.argwhere(mask)

    if coords.size == 0:
        return np.asarray(
            [
                (image.shape[0] - 1) / 2.0,
                (image.shape[1] - 1) / 2.0,
                (image.shape[2] - 1) / 2.0,
            ],
            dtype=np.float64,
        )

    return coords.mean(axis=0)


def calculate_intensity_weighted_center(
    image: np.ndarray,
) -> np.ndarray:
    """
    Calculate intensity-weighted image center.

    No ground truth is used.
    """
    volume = np.asarray(image, dtype=np.float64)

    volume = np.nan_to_num(
        volume,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    volume = np.clip(volume, 0.0, None)

    # Suppress weak background intensity.
    threshold = np.percentile(volume, 30.0)

    weights = np.maximum(volume - threshold, 0.0)

    total = weights.sum()

    if total <= 0:
        return calculate_body_center(image)

    z_grid = np.arange(image.shape[0], dtype=np.float64)
    y_grid = np.arange(image.shape[1], dtype=np.float64)
    x_grid = np.arange(image.shape[2], dtype=np.float64)

    z_weight = weights.sum(axis=(1, 2))
    y_weight = weights.sum(axis=(0, 2))
    x_weight = weights.sum(axis=(0, 1))

    z = np.sum(z_grid * z_weight) / max(z_weight.sum(), 1e-12)
    y = np.sum(y_grid * y_weight) / max(y_weight.sum(), 1e-12)
    x = np.sum(x_grid * x_weight) / max(x_weight.sum(), 1e-12)

    return np.asarray([z, y, x], dtype=np.float64)


def calculate_hybrid_image_center(
    image: np.ndarray,
) -> np.ndarray:
    """
    Hybrid image-only localization.

    Combines:
        - foreground/body center
        - intensity-weighted center

    This is intentionally similar in spirit to the successful Part 26/31
    hybrid strategy.
    """
    body_center = calculate_body_center(image)
    intensity_center = calculate_intensity_weighted_center(image)

    center = (
        0.60 * body_center
        + 0.40 * intensity_center
    )

    return center


# =============================================================================
# SPINAL-CANAL-ORIENTED IMAGE-ONLY SIGNAL
# =============================================================================

def calculate_axis_energy(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate image intensity energy along each spatial axis.

    Returns:
        z_energy
        y_energy
        x_energy

    This uses image intensity only.
    """
    volume = np.asarray(image, dtype=np.float64)

    volume = np.nan_to_num(
        volume,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Remove low-level background.
    threshold = np.percentile(volume, 25.0)

    signal = np.maximum(volume - threshold, 0.0)

    z_energy = signal.mean(axis=(1, 2))
    y_energy = signal.mean(axis=(0, 2))
    x_energy = signal.mean(axis=(0, 1))

    return z_energy, y_energy, x_energy


def smooth_1d(values: np.ndarray, window: int = 9) -> np.ndarray:
    """
    Simple moving-average smoothing.
    """
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return values

    window = max(1, int(window))

    if window == 1:
        return values.copy()

    if window > values.size:
        window = values.size

    kernel = np.ones(window, dtype=np.float64) / float(window)

    padded = np.pad(
        values,
        (window // 2, window // 2),
        mode="edge",
    )

    result = np.convolve(
        padded,
        kernel,
        mode="valid",
    )

    return result[: values.size]


def calculate_central_axis_center(
    image: np.ndarray,
) -> np.ndarray:
    """
    Estimate the central anatomical axis from image information.

    The spinal column is expected to occupy the central body region.
    This method searches for a high-signal central region while avoiding
    extreme peripheral positions.

    No ground truth is used.
    """
    hybrid = calculate_hybrid_image_center(image)

    z_energy, y_energy, x_energy = calculate_axis_energy(image)

    y_energy = smooth_1d(y_energy, 11)
    x_energy = smooth_1d(x_energy, 11)

    y_center = hybrid[1]
    x_center = hybrid[2]

    # Restrict search to central body region.
    y_low = int(image.shape[1] * 0.25)
    y_high = int(image.shape[1] * 0.75)

    x_low = int(image.shape[2] * 0.25)
    x_high = int(image.shape[2] * 0.75)

    y_range = np.arange(y_low, max(y_low + 1, y_high))
    x_range = np.arange(x_low, max(x_low + 1, x_high))

    if len(y_range) > 0:
        y_peak = y_range[np.argmax(y_energy[y_range])]
    else:
        y_peak = int(round(y_center))

    if len(x_range) > 0:
        x_peak = x_range[np.argmax(x_energy[x_range])]
    else:
        x_peak = int(round(x_center))

    # Avoid excessive movement from the hybrid estimate.
    y_peak = (
        0.65 * y_center
        + 0.35 * float(y_peak)
    )

    x_peak = (
        0.65 * x_center
        + 0.35 * float(x_peak)
    )

    return np.asarray(
        [
            hybrid[0],
            y_peak,
            x_peak,
        ],
        dtype=np.float64,
    )


def calculate_canal_preserving_center(
    image: np.ndarray,
) -> np.ndarray:
    """
    Canal-preserving image-only center.

    The strategy gives stronger importance to the estimated central
    spinal-axis location while retaining the robust hybrid estimate.
    """
    hybrid = calculate_hybrid_image_center(image)
    axis_center = calculate_central_axis_center(image)

    center = (
        0.40 * hybrid
        + 0.60 * axis_center
    )

    return center


# =============================================================================
# MULTI-CANDIDATE IMAGE-ONLY LOCALIZATION
# =============================================================================

def generate_candidate_centers(
    image: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    """
    Generate multiple candidate centers from image-only signals.
    """
    centers = []

    fixed = np.asarray(
        [
            (image.shape[0] - 1) / 2.0,
            (image.shape[1] - 1) / 2.0,
            (image.shape[2] - 1) / 2.0,
        ],
        dtype=np.float64,
    )

    hybrid = calculate_hybrid_image_center(image)
    intensity = calculate_intensity_weighted_center(image)
    axis = calculate_central_axis_center(image)
    canal = calculate_canal_preserving_center(image)

    centers.append(("fixed_center", fixed))
    centers.append(("hybrid_image_center", hybrid))
    centers.append(("intensity_weighted_center", intensity))
    centers.append(("canal_axis_image_center", axis))
    centers.append(("canal_preserving_hybrid", canal))

    return centers


def select_multi_candidate_center(
    image: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """
    Select a candidate using image-only anatomical plausibility.

    This is NOT allowed to inspect the ground-truth mask.

    The scoring function favors:
        - central body position
        - moderate intensity concentration
        - proximity to hybrid/body center
        - proximity to central anatomical axis
    """
    candidates = generate_candidate_centers(image)

    body = calculate_body_center(image)
    hybrid = calculate_hybrid_image_center(image)
    axis = calculate_central_axis_center(image)

    shape = np.asarray(image.shape, dtype=np.float64)

    center_of_volume = (shape - 1.0) / 2.0

    def normalized_distance(a, b):
        distance = np.linalg.norm(
            (np.asarray(a) - np.asarray(b))
            / np.maximum(shape, 1.0)
        )

        return float(distance)

    scored = []

    for name, center in candidates:
        central_penalty = normalized_distance(
            center,
            center_of_volume,
        )

        body_penalty = normalized_distance(
            center,
            body,
        )

        hybrid_penalty = normalized_distance(
            center,
            hybrid,
        )

        axis_penalty = normalized_distance(
            center,
            axis,
        )

        score = (
            0.15 * central_penalty
            + 0.20 * body_penalty
            + 0.25 * hybrid_penalty
            + 0.40 * axis_penalty
        )

        scored.append(
            (
                score,
                name,
                center,
            )
        )

    scored.sort(key=lambda x: x[0])

    _, selected_name, selected_center = scored[0]

    return selected_center, selected_name


# =============================================================================
# CROP UTILITIES
# =============================================================================

def crop_bounds(
    shape: Tuple[int, int, int],
    center: np.ndarray,
    patch_size: Tuple[int, int, int] = PATCH_SIZE,
) -> Tuple[List[int], List[int]]:
    """
    Calculate crop bounds.
    """
    starts = []
    ends = []

    for dim, c, patch in zip(shape, center, patch_size):
        half = patch / 2.0

        start = int(round(float(c) - half))
        end = start + patch

        starts.append(start)
        ends.append(end)

    return starts, ends


def crop_with_padding(
    volume: np.ndarray,
    center: np.ndarray,
    patch_size: Tuple[int, int, int] = PATCH_SIZE,
    pad_value: float = 0,
) -> np.ndarray:
    """
    Crop around center and zero-pad outside image boundaries.
    """
    starts, ends = crop_bounds(
        volume.shape,
        center,
        patch_size,
    )

    output = np.full(
        patch_size,
        pad_value,
        dtype=volume.dtype,
    )

    source_slices = []
    destination_slices = []

    for dim, start, end in zip(
        volume.shape,
        starts,
        ends,
    ):
        src_start = max(0, start)
        src_end = min(dim, end)

        dst_start = max(0, -start)
        dst_end = dst_start + max(0, src_end - src_start)

        source_slices.append(
            slice(src_start, src_end)
        )

        destination_slices.append(
            slice(dst_start, dst_end)
        )

    if all(
        s.stop > s.start
        for s in source_slices
    ):
        output[
            tuple(destination_slices)
        ] = volume[
            tuple(source_slices)
        ]

    return output


# =============================================================================
# GROUND-TRUTH EVALUATION
# =============================================================================

def calculate_class_voxels(
    mask: np.ndarray,
    label: int,
) -> int:
    return int(np.sum(mask == label))


def calculate_retention(
    original_mask: np.ndarray,
    cropped_mask: np.ndarray,
    label: int,
) -> float:
    """
    Fraction of ground-truth voxels retained inside crop.
    """
    original_count = calculate_class_voxels(
        original_mask,
        label,
    )

    if original_count <= 0:
        return 0.0

    cropped_count = calculate_class_voxels(
        cropped_mask,
        label,
    )

    return float(
        cropped_count / original_count
    )


def calculate_foreground_retention(
    original_mask: np.ndarray,
    cropped_mask: np.ndarray,
) -> float:
    original_fg = original_mask != BACKGROUND
    cropped_fg = cropped_mask != BACKGROUND

    total = int(original_fg.sum())

    if total == 0:
        return 0.0

    return float(cropped_fg.sum() / total)


def calculate_combined_retention(
    vertebra_retention: float,
    canal_retention: float,
    disc_retention: float,
) -> float:
    return float(
        (
            vertebra_retention
            + canal_retention
            + disc_retention
        )
        / 3.0
    )


def calculate_minimum_retention(
    vertebra_retention: float,
    canal_retention: float,
    disc_retention: float,
) -> float:
    return float(
        min(
            vertebra_retention,
            canal_retention,
            disc_retention,
        )
    )


# =============================================================================
# ORACLE CENTER FOR EVALUATION ONLY
# =============================================================================

def calculate_ground_truth_center(
    mask: np.ndarray,
    label: int = SPINAL_CANAL,
) -> np.ndarray:
    """
    Ground-truth center used ONLY for evaluation.

    This center is NOT used by any image-only strategy.
    """
    coords = np.argwhere(mask == label)

    if coords.size == 0:
        shape = np.asarray(mask.shape, dtype=np.float64)

        return (shape - 1.0) / 2.0

    return coords.mean(axis=0)


def calculate_localization_error(
    predicted_center: np.ndarray,
    target_center: np.ndarray,
    shape: Tuple[int, int, int],
) -> float:
    """
    Euclidean localization error in voxels.
    """
    # Direct voxel distance.
    distance = np.linalg.norm(
        np.asarray(predicted_center, dtype=np.float64)
        - np.asarray(target_center, dtype=np.float64)
    )

    return float(distance)


# =============================================================================
# STRATEGY CENTER CALCULATION
# =============================================================================

def get_strategy_center(
    image: np.ndarray,
    strategy: str,
) -> Tuple[np.ndarray, str]:
    """
    Get image-only center for a strategy.
    """
    shape = np.asarray(image.shape, dtype=np.float64)

    fixed = (shape - 1.0) / 2.0

    if strategy == "fixed_center":
        return fixed, strategy

    if strategy == "hybrid_image_center":
        return (
            calculate_hybrid_image_center(image),
            strategy,
        )

    if strategy == "intensity_weighted_center":
        return (
            calculate_intensity_weighted_center(image),
            strategy,
        )

    if strategy == "canal_axis_image_center":
        return (
            calculate_central_axis_center(image),
            strategy,
        )

    if strategy == "canal_preserving_hybrid":
        return (
            calculate_canal_preserving_center(image),
            strategy,
        )

    if strategy == "multi_candidate_canal_center":
        center, selected = select_multi_candidate_center(image)

        return center, selected

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# =============================================================================
# PER-CASE ANALYSIS
# =============================================================================

def analyze_case(
    case_name: str,
    image_path: Path,
    mask_path: Path,
) -> List[Dict]:
    """
    Analyze one case using all image-only localization strategies.
    """
    image_original = read_mha(image_path)
    mask_original = read_mha(mask_path)

    if image_original.shape != mask_original.shape:
        raise ValueError(
            f"Shape mismatch for {case_name}: "
            f"image={image_original.shape}, "
            f"mask={mask_original.shape}"
        )

    image = percentile_normalize(
        image_original,
        1.0,
        99.0,
    )

    results = []

    canal_gt_center = calculate_ground_truth_center(
        mask_original,
        SPINAL_CANAL,
    )

    sequence = infer_sequence(case_name)

    for strategy in STRATEGIES:

        center, selected_substrategy = get_strategy_center(
            image,
            strategy,
        )

        cropped_image = crop_with_padding(
            image,
            center,
            PATCH_SIZE,
            pad_value=0.0,
        )

        cropped_mask = crop_with_padding(
            mask_original,
            center,
            PATCH_SIZE,
            pad_value=0,
        )

        vertebra_retention = calculate_retention(
            mask_original,
            cropped_mask,
            VERTEBRAE,
        )

        canal_retention = calculate_retention(
            mask_original,
            cropped_mask,
            SPINAL_CANAL,
        )

        disc_retention = calculate_retention(
            mask_original,
            cropped_mask,
            INTERVERTEBRAL_DISC,
        )

        foreground_retention = calculate_foreground_retention(
            mask_original,
            cropped_mask,
        )

        combined_retention = calculate_combined_retention(
            vertebra_retention,
            canal_retention,
            disc_retention,
        )

        minimum_retention = calculate_minimum_retention(
            vertebra_retention,
            canal_retention,
            disc_retention,
        )

        localization_error = calculate_localization_error(
            center,
            canal_gt_center,
            image_original.shape,
        )

        result = {
            "file": case_name,
            "sequence": sequence,
            "strategy": strategy,

            "selected_substrategy": selected_substrategy,

            "original_z": int(image_original.shape[0]),
            "original_y": int(image_original.shape[1]),
            "original_x": int(image_original.shape[2]),

            "center_z": float(center[0]),
            "center_y": float(center[1]),
            "center_x": float(center[2]),

            "canal_gt_center_z": float(canal_gt_center[0]),
            "canal_gt_center_y": float(canal_gt_center[1]),
            "canal_gt_center_x": float(canal_gt_center[2]),

            "localization_error_voxels": localization_error,

            "foreground_retention": foreground_retention,

            "vertebrae_retention": vertebra_retention,
            "spinal_canal_retention": canal_retention,
            "disc_retention": disc_retention,

            "combined_retention": combined_retention,
            "minimum_class_retention": minimum_retention,

            "canal_below_0.02": int(canal_retention < 0.02),
            "canal_below_0.10": int(canal_retention < 0.10),
            "canal_below_0.20": int(canal_retention < 0.20),

            "combined_above_0.20": int(
                combined_retention >= 0.20
            ),
        }

        results.append(result)

    return results


# =============================================================================
# STRATEGY SUMMARY
# =============================================================================

def create_strategy_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create overall strategy summary.
    """
    rows = []

    for strategy in STRATEGIES:

        subset = dataframe[
            dataframe["strategy"] == strategy
        ]

        rows.append(
            {
                "strategy": strategy,
                "cases": int(len(subset)),

                "mean_localization_error_voxels":
                    float(
                        subset[
                            "localization_error_voxels"
                        ].mean()
                    ),

                "median_localization_error_voxels":
                    float(
                        subset[
                            "localization_error_voxels"
                        ].median()
                    ),

                "mean_foreground_retention":
                    float(
                        subset[
                            "foreground_retention"
                        ].mean()
                    ),

                "mean_vertebrae_retention":
                    float(
                        subset[
                            "vertebrae_retention"
                        ].mean()
                    ),

                "mean_spinal_canal_retention":
                    float(
                        subset[
                            "spinal_canal_retention"
                        ].mean()
                    ),

                "mean_disc_retention":
                    float(
                        subset[
                            "disc_retention"
                        ].mean()
                    ),

                "mean_combined_retention":
                    float(
                        subset[
                            "combined_retention"
                        ].mean()
                    ),

                "median_combined_retention":
                    float(
                        subset[
                            "combined_retention"
                        ].median()
                    ),

                "mean_minimum_class_retention":
                    float(
                        subset[
                            "minimum_class_retention"
                        ].mean()
                    ),

                "median_minimum_class_retention":
                    float(
                        subset[
                            "minimum_class_retention"
                        ].median()
                    ),

                "canal_below_0.02":
                    int(
                        subset[
                            "canal_below_0.02"
                        ].sum()
                    ),

                "canal_below_0.10":
                    int(
                        subset[
                            "canal_below_0.10"
                        ].sum()
                    ),

                "canal_below_0.20":
                    int(
                        subset[
                            "canal_below_0.20"
                        ].sum()
                    ),

                "combined_above_0.20":
                    int(
                        subset[
                            "combined_above_0.20"
                        ].sum()
                    ),
            }
        )

    summary = pd.DataFrame(rows)

    # Main ranking:
    # combined retention first,
    # then balanced anatomical retention.
    summary = summary.sort_values(
        [
            "mean_combined_retention",
            "mean_minimum_class_retention",
        ],
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        np.arange(1, len(summary) + 1),
    )

    return summary


# =============================================================================
# SEQUENCE SUMMARY
# =============================================================================

def create_sequence_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize strategies separately for T1, T2 and T2 SPACE.
    """
    rows = []

    sequences = [
        "T1",
        "T2",
        "T2 SPACE",
    ]

    for sequence in sequences:

        subset_sequence = dataframe[
            dataframe["sequence"] == sequence
        ]

        for strategy in STRATEGIES:

            subset = subset_sequence[
                subset_sequence["strategy"] == strategy
            ]

            if len(subset) == 0:
                continue

            rows.append(
                {
                    "sequence": sequence,
                    "strategy": strategy,
                    "cases": int(len(subset)),

                    "mean_localization_error_voxels":
                        float(
                            subset[
                                "localization_error_voxels"
                            ].mean()
                        ),

                    "mean_foreground_retention":
                        float(
                            subset[
                                "foreground_retention"
                            ].mean()
                        ),

                    "mean_vertebrae_retention":
                        float(
                            subset[
                                "vertebrae_retention"
                            ].mean()
                        ),

                    "mean_spinal_canal_retention":
                        float(
                            subset[
                                "spinal_canal_retention"
                            ].mean()
                        ),

                    "mean_disc_retention":
                        float(
                            subset[
                                "disc_retention"
                            ].mean()
                        ),

                    "mean_combined_retention":
                        float(
                            subset[
                                "combined_retention"
                            ].mean()
                        ),

                    "mean_minimum_class_retention":
                        float(
                            subset[
                                "minimum_class_retention"
                            ].mean()
                        ),

                    "canal_below_0.02":
                        int(
                            subset[
                                "canal_below_0.02"
                            ].sum()
                        ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# HIGH-RISK CASES
# =============================================================================

def create_high_risk_cases(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify cases with poor spinal canal localization.

    A case is considered high-risk if the best available image-only strategy
    still produces poor canal retention.
    """
    rows = []

    for case_name in sorted(
        dataframe["file"].unique()
    ):

        subset = dataframe[
            dataframe["file"] == case_name
        ]

        best_idx = subset[
            "spinal_canal_retention"
        ].idxmax()

        best = subset.loc[best_idx]

        worst_idx = subset[
            "spinal_canal_retention"
        ].idxmin()

        worst = subset.loc[worst_idx]

        rows.append(
            {
                "file": case_name,
                "sequence": best["sequence"],

                "best_strategy":
                    best["strategy"],

                "best_canal_retention":
                    float(
                        best[
                            "spinal_canal_retention"
                        ]
                    ),

                "best_combined_retention":
                    float(
                        best[
                            "combined_retention"
                        ]
                    ),

                "best_localization_error_voxels":
                    float(
                        best[
                            "localization_error_voxels"
                        ]
                    ),

                "worst_strategy":
                    worst["strategy"],

                "worst_canal_retention":
                    float(
                        worst[
                            "spinal_canal_retention"
                        ]
                    ),

                "worst_combined_retention":
                    float(
                        worst[
                            "combined_retention"
                        ]
                    ),

                "high_risk":
                    int(
                        best[
                            "spinal_canal_retention"
                        ] < 0.20
                    ),
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "high_risk",
            "best_canal_retention",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    return result


# =============================================================================
# CHARTS
# =============================================================================

def save_bar_chart(
    x_labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    rotation: int = 25,
) -> None:

    plt.figure(figsize=(12, 7))

    positions = np.arange(len(x_labels))

    plt.bar(
        positions,
        values,
    )

    plt.xticks(
        positions,
        x_labels,
        rotation=rotation,
        ha="right",
    )

    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def create_overall_strategy_chart(
    summary: pd.DataFrame,
) -> None:

    save_bar_chart(
        summary["strategy"].tolist(),
        summary["mean_combined_retention"].tolist(),
        "Part 32 - Overall Image-Only Strategy Comparison",
        "Mean Combined Anatomical Retention",
        OUTPUT_DIR
        / "part32_overall_strategy_comparison.png",
    )


def create_canal_retention_chart(
    summary: pd.DataFrame,
) -> None:

    save_bar_chart(
        summary["strategy"].tolist(),
        summary["mean_spinal_canal_retention"].tolist(),
        "Part 32 - Spinal Canal Retention Comparison",
        "Mean Spinal Canal Retention",
        OUTPUT_DIR
        / "part32_canal_retention_comparison.png",
    )


def create_sequence_chart(
    sequence_summary: pd.DataFrame,
) -> None:

    sequences = [
        "T1",
        "T2",
        "T2 SPACE",
    ]

    plt.figure(figsize=(13, 7))

    for strategy in STRATEGIES:

        subset = sequence_summary[
            sequence_summary["strategy"] == strategy
        ]

        values = []

        for sequence in sequences:
            row = subset[
                subset["sequence"] == sequence
            ]

            if len(row) == 0:
                values.append(np.nan)
            else:
                values.append(
                    float(
                        row.iloc[0][
                            "mean_spinal_canal_retention"
                        ]
                    )
                )

        plt.plot(
            sequences,
            values,
            marker="o",
            label=strategy,
        )

    plt.ylabel(
        "Mean Spinal Canal Retention"
    )

    plt.xlabel("MRI Sequence")

    plt.title(
        "Part 32 - Sequence-wise Spinal Canal Retention"
    )

    plt.legend(
        fontsize=8,
        loc="best",
    )

    plt.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "part32_sequence_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def create_localization_error_chart(
    summary: pd.DataFrame,
) -> None:

    save_bar_chart(
        summary["strategy"].tolist(),
        summary[
            "mean_localization_error_voxels"
        ].tolist(),
        "Part 32 - Image-Only Canal Localization Error",
        "Mean Localization Error (voxels)",
        OUTPUT_DIR
        / "part32_localization_error.png",
    )


def create_canal_failure_chart(
    summary: pd.DataFrame,
) -> None:

    save_bar_chart(
        summary["strategy"].tolist(),
        summary[
            "canal_below_0.20"
        ].tolist(),
        "Part 32 - Spinal Canal Low-Retention Cases",
        "Cases with Canal Retention < 0.20",
        OUTPUT_DIR
        / "part32_canal_failure_cases.png",
    )


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def clean_for_json(value):
    """
    Convert NumPy/Pandas values to JSON-safe values.
    """
    if isinstance(
        value,
        (
            np.integer,
            np.int32,
            np.int64,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float32,
            np.float64,
        ),
    ):
        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(value, Path):
        return str(value)

    return value


# =============================================================================
# REPORT
# =============================================================================

def create_report(
    case_df: pd.DataFrame,
    summary: pd.DataFrame,
    sequence_summary: pd.DataFrame,
    high_risk: pd.DataFrame,
    execution_minutes: float,
) -> Dict:

    best_combined = summary.iloc[0]

    canal_ranked = summary.sort_values(
        "mean_spinal_canal_retention",
        ascending=False,
    ).reset_index(drop=True)

    best_canal = canal_ranked.iloc[0]

    balanced_ranked = summary.sort_values(
        "mean_minimum_class_retention",
        ascending=False,
    ).reset_index(drop=True)

    best_balanced = balanced_ranked.iloc[0]

    fixed_row = summary[
        summary["strategy"] == "fixed_center"
    ].iloc[0]

    hybrid_row = summary[
        summary["strategy"] == "hybrid_image_center"
    ].iloc[0]

    multi_row = summary[
        summary["strategy"]
        == "multi_candidate_canal_center"
    ].iloc[0]

    report = {
        "phase": "Phase 3 - Part 32",
        "title": "Spinal-Canal-Aware Image-Only Localization Analysis",

        "cases_analyzed": int(
            case_df["file"].nunique()
        ),

        "strategies_evaluated": STRATEGIES,

        "best_combined_strategy":
            str(best_combined["strategy"]),

        "best_combined_retention":
            float(
                best_combined[
                    "mean_combined_retention"
                ]
            ),

        "best_canal_strategy":
            str(best_canal["strategy"]),

        "best_canal_retention":
            float(
                best_canal[
                    "mean_spinal_canal_retention"
                ]
            ),

        "best_balanced_strategy":
            str(best_balanced["strategy"]),

        "best_balanced_retention":
            float(
                best_balanced[
                    "mean_minimum_class_retention"
                ]
            ),

        "fixed_center_combined_retention":
            float(
                fixed_row[
                    "mean_combined_retention"
                ]
            ),

        "hybrid_image_center_combined_retention":
            float(
                hybrid_row[
                    "mean_combined_retention"
                ]
            ),

        "multi_candidate_combined_retention":
            float(
                multi_row[
                    "mean_combined_retention"
                ]
            ),

        "hybrid_vs_fixed_combined_improvement":
            float(
                hybrid_row[
                    "mean_combined_retention"
                ]
                - fixed_row[
                    "mean_combined_retention"
                ]
            ),

        "best_vs_fixed_combined_improvement":
            float(
                best_combined[
                    "mean_combined_retention"
                ]
                - fixed_row[
                    "mean_combined_retention"
                ]
            ),

        "best_vs_fixed_canal_improvement":
            float(
                best_canal[
                    "mean_spinal_canal_retention"
                ]
                - fixed_row[
                    "mean_spinal_canal_retention"
                ]
            ),

        "high_risk_cases": int(
            high_risk["high_risk"].sum()
        ),

        "ground_truth_used_for_localization": False,

        "ground_truth_used_for_evaluation": True,

        "training_performed": False,

        "model_weights_modified": False,

        "execution_minutes":
            float(execution_minutes),
    }

    return report


def save_text_report(
    report: Dict,
    summary: pd.DataFrame,
    sequence_summary: pd.DataFrame,
    high_risk: pd.DataFrame,
) -> None:

    output_file = (
        OUTPUT_DIR
        / "phase3_part32_report.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 3 - PART 32\n"
        )

        f.write(
            "SPINAL-CANAL-AWARE IMAGE-ONLY "
            "LOCALIZATION ANALYSIS\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            "PURPOSE\n"
        )

        f.write(
            "Evaluate image-only localization strategies "
            "with emphasis on spinal canal preservation.\n\n"
        )

        f.write(
            "GROUND-TRUTH LOCALIZATION USED: NO\n"
        )

        f.write(
            "GROUND-TRUTH EVALUATION USED: YES\n"
        )

        f.write(
            "TRAINING PERFORMED: NO\n"
        )

        f.write(
            "MODEL WEIGHTS MODIFIED: NO\n\n"
        )

        f.write(
            "OVERALL RESULTS\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            f"Cases analyzed: "
            f"{report['cases_analyzed']}\n"
        )

        f.write(
            f"Best combined strategy: "
            f"{report['best_combined_strategy']}\n"
        )

        f.write(
            f"Best combined retention: "
            f"{report['best_combined_retention']:.6f}\n"
        )

        f.write(
            f"Best canal strategy: "
            f"{report['best_canal_strategy']}\n"
        )

        f.write(
            f"Best canal retention: "
            f"{report['best_canal_retention']:.6f}\n"
        )

        f.write(
            f"Best balanced strategy: "
            f"{report['best_balanced_strategy']}\n"
        )

        f.write(
            f"Best balanced retention: "
            f"{report['best_balanced_retention']:.6f}\n"
        )

        f.write(
            f"Fixed-center combined retention: "
            f"{report['fixed_center_combined_retention']:.6f}\n"
        )

        f.write(
            f"Best-vs-fixed combined improvement: "
            f"{report['best_vs_fixed_combined_improvement']:+.6f}\n"
        )

        f.write(
            f"Best-vs-fixed canal improvement: "
            f"{report['best_vs_fixed_canal_improvement']:+.6f}\n"
        )

        f.write(
            f"High-risk cases: "
            f"{report['high_risk_cases']}\n\n"
        )

        f.write(
            "STRATEGY SUMMARY\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "SEQUENCE SUMMARY\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            sequence_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "HIGH-RISK CASES\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            high_risk[
                high_risk["high_risk"] == 1
            ].to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "INTERPRETATION\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "Part 32 evaluates whether image-only localization "
            "can preserve the spinal canal more reliably than "
            "fixed-center localization.\n\n"
        )

        f.write(
            "The results should be interpreted as a localization "
            "experiment rather than a segmentation performance "
            "experiment. Ground-truth masks are used only to "
            "measure how much anatomy is retained by each crop.\n\n"
        )

        f.write(
            "The best strategy should be considered for subsequent "
            "model training only if it improves spinal-canal "
            "retention while maintaining adequate vertebrae and "
            "disc retention.\n\n"
        )

        f.write(
            f"Execution time: "
            f"{report['execution_minutes']:.2f} minutes\n"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    start_time = time.time()

    print_header(
        "PHASE 3 - PART 32\n"
        "SPINAL-CANAL-AWARE IMAGE-ONLY LOCALIZATION ANALYSIS"
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # DATASET DISCOVERY
    # -------------------------------------------------------------------------

    print_section(
        "LOADING TEST DATASET"
    )

    image_files = get_mha_files(
        IMAGE_DIR
    )

    mask_files = get_mha_files(
        MASK_DIR
    )

    common_cases = sorted(
        set(image_files)
        & set(mask_files)
    )

    image_only = sorted(
        set(image_files)
        - set(mask_files)
    )

    mask_only = sorted(
        set(mask_files)
        - set(image_files)
    )

    print(
        f"Image files : {len(image_files)}"
    )

    print(
        f"Mask files  : {len(mask_files)}"
    )

    print(
        f"Matching    : {len(common_cases)}"
    )

    print(
        f"Image-only  : {len(image_only)}"
    )

    print(
        f"Mask-only   : {len(mask_only)}"
    )

    if len(common_cases) == 0:
        raise RuntimeError(
            "No matching image/mask cases found."
        )

    # -------------------------------------------------------------------------
    # STRATEGIES
    # -------------------------------------------------------------------------

    print_section(
        "IMAGE-ONLY LOCALIZATION STRATEGIES"
    )

    for strategy in STRATEGIES:
        print(
            f"✓ {strategy}"
        )

    print()
    print(
        "Ground-truth used for localization: NO"
    )

    print(
        "Ground-truth used for evaluation: YES"
    )

    print(
        f"Patch size: {PATCH_SIZE}"
    )

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------

    print_section(
        "RUNNING SPINAL-CANAL-AWARE LOCALIZATION"
    )

    all_results = []

    total_cases = len(common_cases)

    for index, case_name in enumerate(
        common_cases,
        start=1,
    ):

        print(
            f"Analyzing {index:2d} / "
            f"{total_cases} : {case_name}"
        )

        try:

            case_results = analyze_case(
                case_name,
                image_files[case_name],
                mask_files[case_name],
            )

            all_results.extend(
                case_results
            )

        except Exception as exc:

            print(
                f"WARNING: Failed case "
                f"{case_name}: {exc}"
            )

    if not all_results:
        raise RuntimeError(
            "No cases were successfully analyzed."
        )

    case_df = pd.DataFrame(
        all_results
    )

    # -------------------------------------------------------------------------
    # SUMMARIES
    # -------------------------------------------------------------------------

    print_section(
        "CREATING STRATEGY SUMMARY"
    )

    strategy_summary = create_strategy_summary(
        case_df
    )

    print(
        strategy_summary.to_string(
            index=False
        )
    )

    sequence_summary = create_sequence_summary(
        case_df
    )

    high_risk = create_high_risk_cases(
        case_df
    )

    # -------------------------------------------------------------------------
    # BEST STRATEGIES
    # -------------------------------------------------------------------------

    best_combined = strategy_summary.iloc[0]

    best_canal = (
        strategy_summary
        .sort_values(
            "mean_spinal_canal_retention",
            ascending=False,
        )
        .iloc[0]
    )

    best_balanced = (
        strategy_summary
        .sort_values(
            "mean_minimum_class_retention",
            ascending=False,
        )
        .iloc[0]
    )

    fixed = strategy_summary[
        strategy_summary["strategy"]
        == "fixed_center"
    ].iloc[0]

    hybrid = strategy_summary[
        strategy_summary["strategy"]
        == "hybrid_image_center"
    ].iloc[0]

    print_section(
        "BEST STRATEGIES"
    )

    print(
        "Best combined-retention strategy:"
    )

    print(
        f"  {best_combined['strategy']}"
    )

    print(
        f"  Combined retention: "
        f"{best_combined['mean_combined_retention']:.6f}"
    )

    print()

    print(
        "Best spinal-canal-retention strategy:"
    )

    print(
        f"  {best_canal['strategy']}"
    )

    print(
        f"  Canal retention: "
        f"{best_canal['mean_spinal_canal_retention']:.6f}"
    )

    print()

    print(
        "Best balanced anatomical strategy:"
    )

    print(
        f"  {best_balanced['strategy']}"
    )

    print(
        f"  Minimum-class retention: "
        f"{best_balanced['mean_minimum_class_retention']:.6f}"
    )

    print()

    print(
        "HYBRID vs FIXED CENTER"
    )

    print(
        f"Fixed combined retention : "
        f"{fixed['mean_combined_retention']:.6f}"
    )

    print(
        f"Hybrid combined retention: "
        f"{hybrid['mean_combined_retention']:.6f}"
    )

    print(
        f"Improvement              : "
        f"{hybrid['mean_combined_retention'] - fixed['mean_combined_retention']:+.6f}"
    )

    # -------------------------------------------------------------------------
    # CANAL ANALYSIS
    # -------------------------------------------------------------------------

    print_section(
        "SPINAL CANAL RETENTION ANALYSIS"
    )

    canal_table = strategy_summary[
        [
            "rank",
            "strategy",
            "mean_spinal_canal_retention",
            "canal_below_0.02",
            "canal_below_0.10",
            "canal_below_0.20",
        ]
    ].copy()

    print(
        canal_table.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # SEQUENCE ANALYSIS
    # -------------------------------------------------------------------------

    print_section(
        "SEQUENCE-WISE ANALYSIS"
    )

    print(
        sequence_summary.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # HIGH-RISK CASES
    # -------------------------------------------------------------------------

    print_section(
        "HIGH-RISK SPINAL CANAL CASES"
    )

    high_risk_cases = high_risk[
        high_risk["high_risk"] == 1
    ]

    print(
        f"High-risk cases: "
        f"{len(high_risk_cases)}"
    )

    if len(high_risk_cases) > 0:
        print()

        print(
            high_risk_cases[
                [
                    "file",
                    "sequence",
                    "best_strategy",
                    "best_canal_retention",
                    "best_combined_retention",
                    "best_localization_error_voxels",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No cases had best image-only canal retention "
            "below 0.20."
        )

    # -------------------------------------------------------------------------
    # SAVE CSV
    # -------------------------------------------------------------------------

    print_section(
        "SAVING ANALYSIS TABLES"
    )

    case_csv = (
        OUTPUT_DIR
        / "part32_case_analysis.csv"
    )

    summary_csv = (
        OUTPUT_DIR
        / "part32_strategy_summary.csv"
    )

    sequence_csv = (
        OUTPUT_DIR
        / "part32_sequence_summary.csv"
    )

    high_risk_csv = (
        OUTPUT_DIR
        / "part32_high_risk_cases.csv"
    )

    case_df.to_csv(
        case_csv,
        index=False,
    )

    strategy_summary.to_csv(
        summary_csv,
        index=False,
    )

    sequence_summary.to_csv(
        sequence_csv,
        index=False,
    )

    high_risk.to_csv(
        high_risk_csv,
        index=False,
    )

    print(
        f"Saved: {case_csv}"
    )

    print(
        f"Saved: {summary_csv}"
    )

    print(
        f"Saved: {sequence_csv}"
    )

    print(
        f"Saved: {high_risk_csv}"
    )

    # -------------------------------------------------------------------------
    # CHARTS
    # -------------------------------------------------------------------------

    print_section(
        "CREATING CHARTS"
    )

    create_overall_strategy_chart(
        strategy_summary
    )

    print(
        "Saved: "
        f"{OUTPUT_DIR / 'part32_overall_strategy_comparison.png'}"
    )

    create_canal_retention_chart(
        strategy_summary
    )

    print(
        "Saved: "
        f"{OUTPUT_DIR / 'part32_canal_retention_comparison.png'}"
    )

    create_sequence_chart(
        sequence_summary
    )

    print(
        "Saved: "
        f"{OUTPUT_DIR / 'part32_sequence_comparison.png'}"
    )

    create_localization_error_chart(
        strategy_summary
    )

    print(
        "Saved: "
        f"{OUTPUT_DIR / 'part32_localization_error.png'}"
    )

    create_canal_failure_chart(
        strategy_summary
    )

    print(
        "Saved: "
        f"{OUTPUT_DIR / 'part32_canal_failure_cases.png'}"
    )

    # -------------------------------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------------------------------

    execution_minutes = (
        time.time() - start_time
    ) / 60.0

    report = create_report(
        case_df,
        strategy_summary,
        sequence_summary,
        high_risk,
        execution_minutes,
    )

    json_output = (
        OUTPUT_DIR
        / "phase3_part32_summary.json"
    )

    with open(
        json_output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            default=clean_for_json,
        )

    print(
        f"Saved: {json_output}"
    )

    save_text_report(
        report,
        strategy_summary,
        sequence_summary,
        high_risk,
    )

    report_output = (
        OUTPUT_DIR
        / "phase3_part32_report.txt"
    )

    print(
        f"Saved: {report_output}"
    )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    print_section(
        "PART 32 COMPLETE"
    )

    print(
        f"Cases analyzed              : "
        f"{report['cases_analyzed']}"
    )

    print(
        f"Best combined strategy     : "
        f"{report['best_combined_strategy']}"
    )

    print(
        f"Best combined retention    : "
        f"{report['best_combined_retention']:.6f}"
    )

    print(
        f"Best canal strategy        : "
        f"{report['best_canal_strategy']}"
    )

    print(
        f"Best canal retention       : "
        f"{report['best_canal_retention']:.6f}"
    )

    print(
        f"Best balanced strategy     : "
        f"{report['best_balanced_strategy']}"
    )

    print(
        f"Best balanced retention    : "
        f"{report['best_balanced_retention']:.6f}"
    )

    print(
        f"Fixed-center retention     : "
        f"{report['fixed_center_combined_retention']:.6f}"
    )

    print(
        f"Best vs fixed improvement  : "
        f"{report['best_vs_fixed_combined_improvement']:+.6f}"
    )

    print(
        f"High-risk cases            : "
        f"{report['high_risk_cases']}"
    )

    print(
        f"Execution time             : "
        f"{report['execution_minutes']:.2f} minutes"
    )

    print()

    print(
        "Ground-truth used for localization: NO"
    )

    print(
        "Ground-truth used for evaluation: YES"
    )

    print(
        "Training performed: NO"
    )

    print(
        "Model weights modified: NO"
    )

    print_section(
        "OUTPUT DIRECTORY"
    )

    print(
        OUTPUT_DIR
    )

    print_section(
        "PHASE 3 - PART 32 COMPLETE"
    )


if __name__ == "__main__":
    main()