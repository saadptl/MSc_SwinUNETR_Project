"""
===============================================================================
PHASE 3 - PART 31
SEQUENCE-AWARE IMAGE-ONLY LOCALIZATION ANALYSIS
===============================================================================

Purpose
-------
Investigate whether image-only localization behaves differently across MRI
sequence types and identify sequence-specific localization strategies.

IMPORTANT
---------
Ground-truth masks are NEVER used to determine the crop/localization center.

Ground-truth masks are used ONLY after localization to evaluate anatomical
retention.

No model training is performed.
No model weights are modified.

Strategies
----------
1. fixed_center
2. image_nonzero_center
3. robust_intensity_center
4. intensity_weighted_center
5. hybrid_image_center

Sequence groups
---------------
1. T1
2. T2
3. T2 SPACE

Outputs
-------
outputs/segmentation/part31_sequence_aware_localization/

===============================================================================
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError(
        "SimpleITK is required. Install with:\n"
        "pip install SimpleITK"
    ) from exc


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
    / "part31_sequence_aware_localization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (96, 96, 96)

BACKGROUND_LABEL = 0
VERTEBRA_LABEL = 1
CANAL_LABEL = 2
DISC_LABEL = 3

CLASS_NAMES = {
    1: "Vertebrae",
    2: "Spinal Canal",
    3: "Intervertebral Disc",
}

EPS = 1e-8


# =============================================================================
# PRINT HELPERS
# =============================================================================

def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_volume(path: Path) -> np.ndarray:
    """
    Load MHA/MHD/NIfTI volume using SimpleITK.

    Returned shape:
        Z, Y, X
    """
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    array = np.asarray(array)

    if array.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, got shape {array.shape} for {path.name}"
        )

    return array


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_percentile(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Robust intensity normalization.

    No ground-truth information is used.
    """

    image = np.asarray(image, dtype=np.float32)

    finite = np.isfinite(image)

    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    values = image[finite]

    low = np.percentile(values, low_percentile)
    high = np.percentile(values, high_percentile)

    if high <= low + EPS:
        minimum = values.min()
        maximum = values.max()

        if maximum <= minimum + EPS:
            return np.zeros_like(image, dtype=np.float32)

        result = (image - minimum) / (maximum - minimum + EPS)
        result[~finite] = 0.0
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    result = (image - low) / (high - low + EPS)
    result[~finite] = 0.0

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# =============================================================================
# SEQUENCE CLASSIFICATION
# =============================================================================

def classify_sequence(filename: str) -> str:
    """
    Determine sequence group from filename.

    T2 SPACE must be checked first because it also contains "t2".
    """

    name = filename.lower()

    if "t2_space" in name or "t2-space" in name or "t2 space" in name:
        return "T2_SPACE"

    if "_t2" in name or name.endswith("t2") or "t2." in name:
        return "T2"

    if "_t1" in name or name.endswith("t1") or "t1." in name:
        return "T1"

    return "UNKNOWN"


# =============================================================================
# CENTER / CROP UTILITIES
# =============================================================================

def clamp_center(
    center: Tuple[float, float, float],
    shape: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    Clamp center coordinates to valid image coordinates.
    """

    return tuple(
        int(np.clip(round(float(c)), 0, max(0, int(s) - 1)))
        for c, s in zip(center, shape)
    )


def fixed_center(
    image: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Geometric image center.
    """

    return tuple(
        int(s // 2)
        for s in image.shape
    )


def image_nonzero_center(
    image: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Center of non-zero image region.
    """

    mask = image > 0

    if not np.any(mask):
        return fixed_center(image)

    coords = np.argwhere(mask)

    center = coords.mean(axis=0)

    return clamp_center(center, image.shape)


def robust_intensity_center(
    image: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Robust intensity-based center.

    Uses voxels above the 75th percentile.
    """

    positive = image[image > 0]

    if positive.size == 0:
        return fixed_center(image)

    threshold = np.percentile(positive, 75)

    mask = image >= threshold

    if mask.sum() < 10:
        return fixed_center(image)

    coords = np.argwhere(mask)

    center = coords.mean(axis=0)

    return clamp_center(center, image.shape)


def intensity_weighted_center(
    image: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Intensity-weighted center.

    This is completely image-only.
    """

    values = np.asarray(image, dtype=np.float64)

    values = np.maximum(values, 0)

    total = values.sum()

    if total <= EPS:
        return fixed_center(image)

    z_axis = np.arange(values.shape[0], dtype=np.float64)
    y_axis = np.arange(values.shape[1], dtype=np.float64)
    x_axis = np.arange(values.shape[2], dtype=np.float64)

    wz = values.sum(axis=(1, 2))
    wy = values.sum(axis=(0, 2))
    wx = values.sum(axis=(0, 1))

    cz = float((z_axis * wz).sum() / (wz.sum() + EPS))
    cy = float((y_axis * wy).sum() / (wy.sum() + EPS))
    cx = float((x_axis * wx).sum() / (wx.sum() + EPS))

    return clamp_center(
        (cz, cy, cx),
        image.shape,
    )


def hybrid_image_center(
    image: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Hybrid image-only center.

    Combines:
        - non-zero foreground extent
        - robust intensity center
        - intensity-weighted center

    No mask information is used.
    """

    c1 = np.asarray(image_nonzero_center(image), dtype=np.float64)
    c2 = np.asarray(robust_intensity_center(image), dtype=np.float64)
    c3 = np.asarray(intensity_weighted_center(image), dtype=np.float64)

    center = (
        0.35 * c1
        + 0.35 * c2
        + 0.30 * c3
    )

    return clamp_center(
        tuple(center),
        image.shape,
    )


STRATEGIES = {
    "fixed_center": fixed_center,
    "image_nonzero_center": image_nonzero_center,
    "robust_intensity_center": robust_intensity_center,
    "intensity_weighted_center": intensity_weighted_center,
    "hybrid_image_center": hybrid_image_center,
}


# =============================================================================
# CROP / PAD
# =============================================================================

def crop_or_pad(
    volume: np.ndarray,
    center: Tuple[int, int, int],
    target_shape: Tuple[int, int, int] = PATCH_SIZE,
    pad_value: float = 0,
) -> np.ndarray:
    """
    Center crop or pad volume to target shape.

    Returns exactly target_shape.
    """

    output = np.full(
        target_shape,
        pad_value,
        dtype=volume.dtype,
    )

    source_slices = []
    target_slices = []

    for axis in range(3):

        source_size = volume.shape[axis]
        target_size = target_shape[axis]
        center_axis = int(center[axis])

        start = center_axis - target_size // 2
        end = start + target_size

        source_start = max(0, start)
        source_end = min(source_size, end)

        target_start = max(0, -start)
        target_end = target_start + (source_end - source_start)

        source_slices.append(
            slice(source_start, source_end)
        )

        target_slices.append(
            slice(target_start, target_end)
        )

    output[tuple(target_slices)] = volume[
        tuple(source_slices)
    ]

    return output


# =============================================================================
# ANATOMICAL RETENTION
# =============================================================================

def class_retention(
    mask: np.ndarray,
    center: Tuple[int, int, int],
    label: int,
) -> float:
    """
    Calculate fraction of class voxels retained inside the crop.
    """

    binary = mask == label

    total = int(binary.sum())

    if total == 0:
        return 0.0

    cropped = crop_or_pad(
        binary.astype(np.uint8),
        center,
        PATCH_SIZE,
        0,
    )

    retained = int(cropped.sum())

    return retained / float(total)


def calculate_metrics(
    mask: np.ndarray,
    center: Tuple[int, int, int],
) -> Dict[str, float]:

    vertebra = class_retention(
        mask,
        center,
        VERTEBRA_LABEL,
    )

    canal = class_retention(
        mask,
        center,
        CANAL_LABEL,
    )

    disc = class_retention(
        mask,
        center,
        DISC_LABEL,
    )

    foreground = mask > 0

    total_foreground = int(foreground.sum())

    if total_foreground > 0:
        foreground_crop = crop_or_pad(
            foreground.astype(np.uint8),
            center,
            PATCH_SIZE,
            0,
        )

        foreground_retention = (
            int(foreground_crop.sum())
            / float(total_foreground)
        )
    else:
        foreground_retention = 0.0

    combined = (
        vertebra
        + canal
        + disc
    ) / 3.0

    minimum = min(
        vertebra,
        canal,
        disc,
    )

    return {
        "foreground_retention": foreground_retention,
        "vertebrae_retention": vertebra,
        "spinal_canal_retention": canal,
        "disc_retention": disc,
        "combined_retention": combined,
        "minimum_class_retention": minimum,
    }


# =============================================================================
# CENTER DISTANCE
# =============================================================================

def euclidean_center_distance(
    a: Tuple[int, int, int],
    b: Tuple[int, int, int],
) -> float:

    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float64)
            - np.asarray(b, dtype=np.float64)
        )
    )


# =============================================================================
# SEQUENCE-AWARE DECISION
# =============================================================================

def determine_sequence_best_strategy(
    sequence_df: pd.DataFrame,
) -> str:
    """
    Select the best strategy based on mean combined anatomical retention.

    This decision is made only after evaluation for research analysis.
    It is NOT used to produce the original localization.

    This allows us to determine whether sequence-specific localization would
    be useful for a future experiment.
    """

    summary = (
        sequence_df
        .groupby("strategy")["combined_retention"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    if summary.empty:
        return "fixed_center"

    return str(summary.index[0])


# =============================================================================
# CASE ANALYSIS
# =============================================================================

def analyze_case(
    image_path: Path,
    mask_path: Path,
) -> List[Dict]:

    image = load_volume(image_path)
    mask = load_volume(mask_path)

    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch for {image_path.name}: "
            f"{image.shape} vs {mask.shape}"
        )

    image = normalize_percentile(image)

    sequence = classify_sequence(
        image_path.name
    )

    fixed = fixed_center(image)

    rows = []

    for strategy_name, strategy_function in STRATEGIES.items():

        center = strategy_function(image)

        metrics = calculate_metrics(
            mask,
            center,
        )

        rows.append(
            {
                "file": image_path.stem,
                "sequence": sequence,
                "strategy": strategy_name,

                "image_z": image.shape[0],
                "image_y": image.shape[1],
                "image_x": image.shape[2],

                "center_z": center[0],
                "center_y": center[1],
                "center_x": center[2],

                "fixed_center_z": fixed[0],
                "fixed_center_y": fixed[1],
                "fixed_center_x": fixed[2],

                "center_shift_voxels":
                    euclidean_center_distance(
                        fixed,
                        center,
                    ),

                **metrics,
            }
        )

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():

    start_time = time.time()

    banner(
        "PHASE 3 - PART 31\n"
        "SEQUENCE-AWARE IMAGE-ONLY LOCALIZATION ANALYSIS"
    )

    print("\nPROJECT ROOT")
    print(PROJECT_ROOT)

    print("\nTEST DIRECTORY")
    print(TEST_DIR)

    print("\nIMAGE DIRECTORY")
    print(IMAGE_DIR)

    print("\nMASK DIRECTORY")
    print(MASK_DIR)

    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    print("\n" + "=" * 78)
    print("LOCALIZATION POLICY")
    print("=" * 78)

    print("Ground-truth used for localization : NO")
    print("Ground-truth used for evaluation  : YES")
    print("Model training                    : NO")
    print("Model weights modified            : NO")

    # -------------------------------------------------------------------------
    # Validate directories
    # -------------------------------------------------------------------------

    banner("VALIDATING TEST DATA")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n{IMAGE_DIR}"
        )

    if not MASK_DIR.exists():
        raise FileNotFoundError(
            f"Mask directory does not exist:\n{MASK_DIR}"
        )

    image_files = sorted(
        list(IMAGE_DIR.glob("*.mha"))
        + list(IMAGE_DIR.glob("*.mhd"))
        + list(IMAGE_DIR.glob("*.nii"))
        + list(IMAGE_DIR.glob("*.nii.gz"))
    )

    mask_files = sorted(
        list(MASK_DIR.glob("*.mha"))
        + list(MASK_DIR.glob("*.mhd"))
        + list(MASK_DIR.glob("*.nii"))
        + list(MASK_DIR.glob("*.nii.gz"))
    )

    mask_lookup = {
        p.stem: p
        for p in mask_files
    }

    pairs = []

    for image_path in image_files:

        mask_path = mask_lookup.get(
            image_path.stem
        )

        if mask_path is not None:
            pairs.append(
                (image_path, mask_path)
            )

    print("\nImage files :", len(image_files))
    print("Mask files  :", len(mask_files))
    print("Matched     :", len(pairs))

    if not pairs:
        raise RuntimeError(
            "No image/mask pairs found."
        )

    # -------------------------------------------------------------------------
    # Analyze all cases
    # -------------------------------------------------------------------------

    banner(
        "RUNNING SEQUENCE-AWARE LOCALIZATION ANALYSIS"
    )

    all_rows = []

    total = len(pairs)

    for index, (image_path, mask_path) in enumerate(
        pairs,
        start=1,
    ):

        print(
            f"Analyzing {index:3d} / {total} : "
            f"{image_path.stem}"
        )

        try:

            rows = analyze_case(
                image_path,
                mask_path,
            )

            all_rows.extend(rows)

        except Exception as exc:

            print(
                f"WARNING: failed {image_path.name}: {exc}"
            )

    df = pd.DataFrame(
        all_rows
    )

    if df.empty:
        raise RuntimeError(
            "No successful analyses were produced."
        )

    # -------------------------------------------------------------------------
    # Save case analysis
    # -------------------------------------------------------------------------

    banner(
        "SAVING CASE-LEVEL ANALYSIS"
    )

    case_csv = (
        OUTPUT_DIR
        / "part31_sequence_aware_case_analysis.csv"
    )

    df.to_csv(
        case_csv,
        index=False,
    )

    print(f"Saved: {case_csv}")

    # -------------------------------------------------------------------------
    # Sequence distribution
    # -------------------------------------------------------------------------

    banner(
        "SEQUENCE DISTRIBUTION"
    )

    sequence_counts = (
        df[
            [
                "file",
                "sequence",
            ]
        ]
        .drop_duplicates()
        ["sequence"]
        .value_counts()
    )

    for sequence, count in sequence_counts.items():

        print(
            f"{sequence:12s}: "
            f"{int(count):2d} cases"
        )

    # -------------------------------------------------------------------------
    # Overall strategy summary
    # -------------------------------------------------------------------------

    banner(
        "OVERALL LOCALIZATION STRATEGY PERFORMANCE"
    )

    overall_summary = (
        df.groupby("strategy")
        .agg(
            cases=("file", "nunique"),

            mean_foreground_retention=(
                "foreground_retention",
                "mean",
            ),

            mean_vertebrae_retention=(
                "vertebrae_retention",
                "mean",
            ),

            mean_spinal_canal_retention=(
                "spinal_canal_retention",
                "mean",
            ),

            mean_disc_retention=(
                "disc_retention",
                "mean",
            ),

            mean_combined_retention=(
                "combined_retention",
                "mean",
            ),

            mean_minimum_class_retention=(
                "minimum_class_retention",
                "mean",
            ),

            mean_center_shift=(
                "center_shift_voxels",
                "mean",
            ),
        )
        .reset_index()
    )

    overall_summary = (
        overall_summary
        .sort_values(
            "mean_combined_retention",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    overall_summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(overall_summary) + 1,
        ),
    )

    print(
        overall_summary.to_string(
            index=False
        )
    )

    overall_csv = (
        OUTPUT_DIR
        / "part31_overall_strategy_summary.csv"
    )

    overall_summary.to_csv(
        overall_csv,
        index=False,
    )

    print(
        f"\nSaved: {overall_csv}"
    )

    # -------------------------------------------------------------------------
    # Sequence-specific strategy summary
    # -------------------------------------------------------------------------

    banner(
        "SEQUENCE-SPECIFIC STRATEGY PERFORMANCE"
    )

    sequence_summary_rows = []

    for sequence in sorted(
        df["sequence"].unique()
    ):

        sequence_df = df[
            df["sequence"] == sequence
        ]

        grouped = (
            sequence_df
            .groupby("strategy")
            .agg(
                cases=("file", "nunique"),

                mean_foreground_retention=(
                    "foreground_retention",
                    "mean",
                ),

                mean_vertebrae_retention=(
                    "vertebrae_retention",
                    "mean",
                ),

                mean_spinal_canal_retention=(
                    "spinal_canal_retention",
                    "mean",
                ),

                mean_disc_retention=(
                    "disc_retention",
                    "mean",
                ),

                mean_combined_retention=(
                    "combined_retention",
                    "mean",
                ),

                mean_minimum_class_retention=(
                    "minimum_class_retention",
                    "mean",
                ),

                mean_center_shift=(
                    "center_shift_voxels",
                    "mean",
                ),
            )
            .reset_index()
        )

        grouped = grouped.sort_values(
            "mean_combined_retention",
            ascending=False,
        )

        best_strategy = (
            grouped.iloc[0]["strategy"]
            if len(grouped)
            else "N/A"
        )

        for rank, (_, row) in enumerate(
            grouped.iterrows(),
            start=1,
        ):

            sequence_summary_rows.append(
                {
                    "sequence": sequence,
                    "rank": rank,
                    "strategy": row["strategy"],
                    "cases": int(row["cases"]),

                    "mean_foreground_retention":
                        row[
                            "mean_foreground_retention"
                        ],

                    "mean_vertebrae_retention":
                        row[
                            "mean_vertebrae_retention"
                        ],

                    "mean_spinal_canal_retention":
                        row[
                            "mean_spinal_canal_retention"
                        ],

                    "mean_disc_retention":
                        row[
                            "mean_disc_retention"
                        ],

                    "mean_combined_retention":
                        row[
                            "mean_combined_retention"
                        ],

                    "mean_minimum_class_retention":
                        row[
                            "mean_minimum_class_retention"
                        ],

                    "mean_center_shift":
                        row[
                            "mean_center_shift"
                        ],

                    "sequence_best_strategy":
                        best_strategy,
                }
            )

        print(
            f"\n{sequence}"
        )

        print(
            grouped[
                [
                    "strategy",
                    "mean_combined_retention",
                    "mean_spinal_canal_retention",
                    "mean_minimum_class_retention",
                    "mean_center_shift",
                ]
            ].to_string(
                index=False
            )
        )

    sequence_summary = pd.DataFrame(
        sequence_summary_rows
    )

    sequence_csv = (
        OUTPUT_DIR
        / "part31_sequence_strategy_summary.csv"
    )

    sequence_summary.to_csv(
        sequence_csv,
        index=False,
    )

    print(
        f"\nSaved: {sequence_csv}"
    )

    # -------------------------------------------------------------------------
    # Best strategy per sequence
    # -------------------------------------------------------------------------

    banner(
        "BEST IMAGE-ONLY STRATEGY BY SEQUENCE"
    )

    best_sequence_rows = []

    for sequence in sorted(
        df["sequence"].unique()
    ):

        sequence_df = df[
            df["sequence"] == sequence
        ]

        grouped = (
            sequence_df
            .groupby("strategy")
            .agg(
                combined=(
                    "combined_retention",
                    "mean",
                ),
                canal=(
                    "spinal_canal_retention",
                    "mean",
                ),
                minimum=(
                    "minimum_class_retention",
                    "mean",
                ),
            )
        )

        if grouped.empty:
            continue

        best = grouped.sort_values(
            "combined",
            ascending=False,
        ).iloc[0]

        best_strategy = (
            grouped["combined"]
            .idxmax()
        )

        print(
            f"{sequence:12s} -> "
            f"{best_strategy:28s} "
            f"Combined: {best['combined']:.6f} "
            f"Canal: {best['canal']:.6f} "
            f"Minimum: {best['minimum']:.6f}"
        )

        best_sequence_rows.append(
            {
                "sequence": sequence,
                "best_strategy": best_strategy,
                "combined_retention":
                    float(best["combined"]),
                "spinal_canal_retention":
                    float(best["canal"]),
                "minimum_class_retention":
                    float(best["minimum"]),
            }
        )

    best_sequence_df = pd.DataFrame(
        best_sequence_rows
    )

    best_sequence_csv = (
        OUTPUT_DIR
        / "part31_best_strategy_by_sequence.csv"
    )

    best_sequence_df.to_csv(
        best_sequence_csv,
        index=False,
    )

    print(
        f"\nSaved: {best_sequence_csv}"
    )

    # -------------------------------------------------------------------------
    # Hybrid vs fixed sequence comparison
    # -------------------------------------------------------------------------

    banner(
        "HYBRID VS FIXED-CENTER BY SEQUENCE"
    )

    pivot = (
        df[
            df["strategy"].isin(
                [
                    "fixed_center",
                    "hybrid_image_center",
                ]
            )
        ]
        .pivot_table(
            index="sequence",
            columns="strategy",
            values=[
                "combined_retention",
                "spinal_canal_retention",
                "minimum_class_retention",
            ],
            aggfunc="mean",
        )
    )

    hybrid_comparison_rows = []

    for sequence in sorted(
        df["sequence"].unique()
    ):

        seq_df = df[
            df["sequence"] == sequence
        ]

        fixed_df = seq_df[
            seq_df["strategy"]
            == "fixed_center"
        ]

        hybrid_df = seq_df[
            seq_df["strategy"]
            == "hybrid_image_center"
        ]

        if fixed_df.empty or hybrid_df.empty:
            continue

        fixed_combined = float(
            fixed_df[
                "combined_retention"
            ].mean()
        )

        hybrid_combined = float(
            hybrid_df[
                "combined_retention"
            ].mean()
        )

        fixed_canal = float(
            fixed_df[
                "spinal_canal_retention"
            ].mean()
        )

        hybrid_canal = float(
            hybrid_df[
                "spinal_canal_retention"
            ].mean()
        )

        fixed_minimum = float(
            fixed_df[
                "minimum_class_retention"
            ].mean()
        )

        hybrid_minimum = float(
            hybrid_df[
                "minimum_class_retention"
            ].mean()
        )

        print(
            f"\n{sequence}"
        )

        print(
            f"Combined retention:"
            f" {fixed_combined:.6f}"
            f" -> {hybrid_combined:.6f}"
            f" "
            f"({hybrid_combined - fixed_combined:+.6f})"
        )

        print(
            f"Canal retention:"
            f" {fixed_canal:.6f}"
            f" -> {hybrid_canal:.6f}"
            f" "
            f"({hybrid_canal - fixed_canal:+.6f})"
        )

        print(
            f"Minimum retention:"
            f" {fixed_minimum:.6f}"
            f" -> {hybrid_minimum:.6f}"
            f" "
            f"({hybrid_minimum - fixed_minimum:+.6f})"
        )

        hybrid_comparison_rows.append(
            {
                "sequence": sequence,

                "fixed_combined_retention":
                    fixed_combined,

                "hybrid_combined_retention":
                    hybrid_combined,

                "combined_change":
                    hybrid_combined
                    - fixed_combined,

                "fixed_spinal_canal_retention":
                    fixed_canal,

                "hybrid_spinal_canal_retention":
                    hybrid_canal,

                "canal_change":
                    hybrid_canal
                    - fixed_canal,

                "fixed_minimum_retention":
                    fixed_minimum,

                "hybrid_minimum_retention":
                    hybrid_minimum,

                "minimum_change":
                    hybrid_minimum
                    - fixed_minimum,
            }
        )

    hybrid_comparison = pd.DataFrame(
        hybrid_comparison_rows
    )

    hybrid_csv = (
        OUTPUT_DIR
        / "part31_hybrid_vs_fixed_by_sequence.csv"
    )

    hybrid_comparison.to_csv(
        hybrid_csv,
        index=False,
    )

    print(
        f"\nSaved: {hybrid_csv}"
    )

    # -------------------------------------------------------------------------
    # Identify high-risk sequence cases
    # -------------------------------------------------------------------------

    banner(
        "SEQUENCE-AWARE HIGH-RISK CASE ANALYSIS"
    )

    fixed_df = df[
        df["strategy"] == "fixed_center"
    ].copy()

    hybrid_df = df[
        df["strategy"] == "hybrid_image_center"
    ].copy()

    merged = fixed_df.merge(
        hybrid_df,
        on=[
            "file",
            "sequence",
        ],
        suffixes=(
            "_fixed",
            "_hybrid",
        ),
    )

    merged[
        "canal_retention_change"
    ] = (
        merged[
            "spinal_canal_retention_hybrid"
        ]
        - merged[
            "spinal_canal_retention_fixed"
        ]
    )

    merged[
        "combined_retention_change"
    ] = (
        merged[
            "combined_retention_hybrid"
        ]
        - merged[
            "combined_retention_fixed"
        ]
    )

    merged[
        "minimum_retention_change"
    ] = (
        merged[
            "minimum_class_retention_hybrid"
        ]
        - merged[
            "minimum_class_retention_fixed"
        ]
    )

    merged[
        "center_shift_voxels"
    ] = merged[
        "center_shift_voxels_hybrid"
    ]

    risk_df = merged.sort_values(
        [
            "sequence",
            "canal_retention_change",
        ],
        ascending=[
            True,
            True,
        ],
    )

    risk_csv = (
        OUTPUT_DIR
        / "part31_sequence_aware_risk_cases.csv"
    )

    risk_df.to_csv(
        risk_csv,
        index=False,
    )

    print(
        f"Saved: {risk_csv}"
    )

    # -------------------------------------------------------------------------
    # Worst hybrid canal retention cases
    # -------------------------------------------------------------------------

    subsection(
        "LOW HYBRID SPINAL-CANAL RETENTION"
    )

    worst_hybrid = (
        hybrid_df
        .sort_values(
            "spinal_canal_retention"
        )
        .head(15)
    )

    print(
        worst_hybrid[
            [
                "file",
                "sequence",
                "spinal_canal_retention",
                "combined_retention",
                "center_shift_voxels",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Sequence statistics
    # -------------------------------------------------------------------------

    banner(
        "SEQUENCE-LEVEL STATISTICS"
    )

    sequence_stats_rows = []

    for sequence in sorted(
        df["sequence"].unique()
    ):

        seq = df[
            (df["sequence"] == sequence)
            & (
                df["strategy"]
                == "hybrid_image_center"
            )
        ]

        if seq.empty:
            continue

        sequence_stats_rows.append(
            {
                "sequence": sequence,
                "cases": int(
                    seq["file"].nunique()
                ),
                "mean_combined_retention":
                    float(
                        seq[
                            "combined_retention"
                        ].mean()
                    ),
                "median_combined_retention":
                    float(
                        seq[
                            "combined_retention"
                        ].median()
                    ),
                "mean_canal_retention":
                    float(
                        seq[
                            "spinal_canal_retention"
                        ].mean()
                    ),
                "median_canal_retention":
                    float(
                        seq[
                            "spinal_canal_retention"
                        ].median()
                    ),
                "mean_center_shift":
                    float(
                        seq[
                            "center_shift_voxels"
                        ].mean()
                    ),
            }
        )

    sequence_stats = pd.DataFrame(
        sequence_stats_rows
    )

    print(
        sequence_stats.to_string(
            index=False
        )
    )

    sequence_stats_csv = (
        OUTPUT_DIR
        / "part31_sequence_statistics.csv"
    )

    sequence_stats.to_csv(
        sequence_stats_csv,
        index=False,
    )

    print(
        f"\nSaved: {sequence_stats_csv}"
    )

    # -------------------------------------------------------------------------
    # Charts
    # -------------------------------------------------------------------------

    banner(
        "CREATING PART 31 CHARTS"
    )

    # 1. Overall strategy comparison

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        overall_summary["strategy"],
        overall_summary[
            "mean_combined_retention"
        ],
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.ylabel(
        "Mean Combined Anatomical Retention"
    )

    plt.title(
        "Part 31 - Overall Image-Only Localization"
    )

    plt.tight_layout()

    chart1 = (
        OUTPUT_DIR
        / "part31_overall_strategy_comparison.png"
    )

    plt.savefig(
        chart1,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {chart1}"
    )

    # 2. Sequence × strategy combined retention

    pivot_combined = (
        sequence_summary
        .pivot(
            index="sequence",
            columns="strategy",
            values="mean_combined_retention",
        )
    )

    ax = pivot_combined.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_ylabel(
        "Mean Combined Anatomical Retention"
    )

    ax.set_xlabel(
        "MRI Sequence"
    )

    ax.set_title(
        "Sequence-Specific Localization Performance"
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Strategy",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()

    chart2 = (
        OUTPUT_DIR
        / "part31_sequence_strategy_comparison.png"
    )

    plt.savefig(
        chart2,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {chart2}"
    )

    # 3. Sequence × strategy canal retention

    pivot_canal = (
        sequence_summary
        .pivot(
            index="sequence",
            columns="strategy",
            values="mean_spinal_canal_retention",
        )
    )

    ax = pivot_canal.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_ylabel(
        "Mean Spinal Canal Retention"
    )

    ax.set_xlabel(
        "MRI Sequence"
    )

    ax.set_title(
        "Sequence-Specific Spinal Canal Retention"
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Strategy",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()

    chart3 = (
        OUTPUT_DIR
        / "part31_sequence_canal_retention.png"
    )

    plt.savefig(
        chart3,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {chart3}"
    )

    # 4. Hybrid vs fixed canal retention

    if not hybrid_comparison.empty:

        x = np.arange(
            len(
                hybrid_comparison
            )
        )

        width = 0.35

        plt.figure(
            figsize=(10, 6)
        )

        plt.bar(
            x - width / 2,
            hybrid_comparison[
                "fixed_spinal_canal_retention"
            ],
            width,
            label="Fixed Center",
        )

        plt.bar(
            x + width / 2,
            hybrid_comparison[
                "hybrid_spinal_canal_retention"
            ],
            width,
            label="Hybrid Image-Only",
        )

        plt.xticks(
            x,
            hybrid_comparison[
                "sequence"
            ],
        )

        plt.ylabel(
            "Spinal Canal Retention"
        )

        plt.title(
            "Fixed vs Hybrid Canal Retention by Sequence"
        )

        plt.legend()

        plt.tight_layout()

        chart4 = (
            OUTPUT_DIR
            / "part31_fixed_vs_hybrid_canal_by_sequence.png"
        )

        plt.savefig(
            chart4,
            dpi=200,
        )

        plt.close()

        print(
            f"Saved: {chart4}"
        )

    # 5. Center shift

    plt.figure(
        figsize=(10, 6)
    )

    sequence_shift = (
        hybrid_df
        .groupby("sequence")
        ["center_shift_voxels"]
        .mean()
        .reindex(
            sorted(
                hybrid_df["sequence"].unique()
            )
        )
    )

    plt.bar(
        sequence_shift.index,
        sequence_shift.values,
    )

    plt.ylabel(
        "Mean Center Shift (voxels)"
    )

    plt.xlabel(
        "MRI Sequence"
    )

    plt.title(
        "Hybrid Image-Only Localization Shift by Sequence"
    )

    plt.tight_layout()

    chart5 = (
        OUTPUT_DIR
        / "part31_sequence_center_shift.png"
    )

    plt.savefig(
        chart5,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {chart5}"
    )

    # 6. Canal retention vs center shift

    plt.figure(
        figsize=(9, 6)
    )

    for sequence in sorted(
        hybrid_df["sequence"].unique()
    ):

        subset = hybrid_df[
            hybrid_df["sequence"]
            == sequence
        ]

        plt.scatter(
            subset["center_shift_voxels"],
            subset["spinal_canal_retention"],
            label=sequence,
        )

    plt.xlabel(
        "Center Shift from Fixed Center (voxels)"
    )

    plt.ylabel(
        "Spinal Canal Retention"
    )

    plt.title(
        "Localization Shift vs Spinal Canal Retention"
    )

    plt.legend()

    plt.tight_layout()

    chart6 = (
        OUTPUT_DIR
        / "part31_center_shift_vs_canal_retention.png"
    )

    plt.savefig(
        chart6,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {chart6}"
    )

    # -------------------------------------------------------------------------
    # Final recommendation
    # -------------------------------------------------------------------------

    banner(
        "PART 31 LOCALIZATION RECOMMENDATION"
    )

    overall_best = str(
        overall_summary.iloc[0]["strategy"]
    )

    sequence_best_map = {}

    for _, row in best_sequence_df.iterrows():

        sequence_best_map[
            row["sequence"]
        ] = row["best_strategy"]

    print(
        f"\nOverall best image-only strategy:"
        f" {overall_best}"
    )

    for sequence, strategy in (
        sequence_best_map.items()
    ):

        print(
            f"{sequence:12s}: {strategy}"
        )

    # -------------------------------------------------------------------------
    # Build JSON summary
    # -------------------------------------------------------------------------

    summary = {
        "phase": "Phase 3 - Part 31",
        "title": (
            "Sequence-Aware Image-Only "
            "Localization Analysis"
        ),

        "cases_analyzed": int(
            df["file"].nunique()
        ),

        "strategies": list(
            STRATEGIES.keys()
        ),

        "ground_truth_used_for_localization":
            False,

        "ground_truth_used_for_evaluation":
            True,

        "training_performed": False,

        "model_weights_modified": False,

        "patch_size": list(
            PATCH_SIZE
        ),

        "sequence_counts": {
            str(k): int(v)
            for k, v in sequence_counts.items()
        },

        "overall_best_strategy":
            overall_best,

        "best_strategy_by_sequence":
            sequence_best_map,

        "overall_strategy_summary":
            overall_summary.to_dict(
                orient="records"
            ),

        "sequence_strategy_summary":
            sequence_summary.to_dict(
                orient="records"
            ),

        "hybrid_vs_fixed_by_sequence":
            hybrid_comparison.to_dict(
                orient="records"
            ),

        "runtime_minutes":
            (time.time() - start_time)
            / 60.0,
    }

    json_path = (
        OUTPUT_DIR
        / "phase3_part31_sequence_aware_localization_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            default=float,
        )

    print(
        f"\nSaved: {json_path}"
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    banner(
        "CREATING PART 31 REPORT"
    )

    report_path = (
        OUTPUT_DIR
        / "phase3_part31_sequence_aware_localization_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 3 - PART 31\n"
        )

        f.write(
            "SEQUENCE-AWARE IMAGE-ONLY "
            "LOCALIZATION ANALYSIS\n"
        )

        f.write(
            "=" * 78 + "\n\n"
        )

        f.write(
            "OBJECTIVE\n"
        )

        f.write(
            "Determine whether image-only localization "
            "behaves differently across MRI sequence "
            "types and identify sequence-specific "
            "localization strategies.\n\n"
        )

        f.write(
            "LOCALIZATION POLICY\n"
        )

        f.write(
            "Ground-truth used for localization: NO\n"
        )

        f.write(
            "Ground-truth used for evaluation: YES\n"
        )

        f.write(
            "Training performed: NO\n"
        )

        f.write(
            "Model weights modified: NO\n\n"
        )

        f.write(
            "TEST CASES\n"
        )

        f.write(
            f"{df['file'].nunique()}\n\n"
        )

        f.write(
            "SEQUENCE DISTRIBUTION\n"
        )

        for sequence, count in (
            sequence_counts.items()
        ):

            f.write(
                f"{sequence}: {int(count)} cases\n"
            )

        f.write(
            "\nOVERALL BEST STRATEGY\n"
        )

        f.write(
            f"{overall_best}\n\n"
        )

        f.write(
            "BEST STRATEGY BY SEQUENCE\n"
        )

        for sequence, strategy in (
            sequence_best_map.items()
        ):

            f.write(
                f"{sequence}: {strategy}\n"
            )

        f.write(
            "\nOVERALL STRATEGY SUMMARY\n"
        )

        f.write(
            overall_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\nSEQUENCE STRATEGY SUMMARY\n"
        )

        f.write(
            sequence_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\nHYBRID VS FIXED BY SEQUENCE\n"
        )

        f.write(
            hybrid_comparison.to_string(
                index=False
            )
        )

        f.write(
            "\n\nINTERPRETATION\n"
        )

        f.write(
            "Part 31 evaluates image-only localization "
            "without using ground-truth masks to determine "
            "the crop. The purpose is to determine whether "
            "a sequence-aware localization policy may be "
            "useful for future model development.\n\n"
        )

        f.write(
            "The results should be interpreted as "
            "localization-retention evidence rather than "
            "segmentation performance. A strategy with "
            "higher anatomical retention does not "
            "automatically guarantee higher model Dice.\n"
        )

        f.write(
            "\nRUNTIME\n"
        )

        f.write(
            f"{(time.time() - start_time) / 60.0:.2f} minutes\n"
        )

    print(
        f"Saved: {report_path}"
    )

    # -------------------------------------------------------------------------
    # Completion
    # -------------------------------------------------------------------------

    banner(
        "PART 31 COMPLETE"
    )

    print(
        f"Cases analyzed       : "
        f"{df['file'].nunique()}"
    )

    print(
        f"Overall best strategy: "
        f"{overall_best}"
    )

    print(
        f"Runtime               : "
        f"{(time.time() - start_time) / 60.0:.2f} minutes"
    )

    print(
        "\nOUTPUT DIRECTORY"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "PHASE 3 - PART 31 COMPLETE"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()