"""
PHASE 3 - PART 26
IMAGE-ONLY ANATOMY LOCALIZATION ANALYSIS

Purpose
-------
Develop and compare realistic inference-time localization strategies
without using ground-truth segmentation masks to determine the crop.

The following strategies are evaluated:

1. fixed_center
2. image_nonzero_center
3. robust_intensity_center
4. intensity_weighted_center
5. hybrid_image_center

Ground-truth masks are used ONLY AFTER the crop is generated to
calculate anatomical retention.

IMPORTANT
---------
This is NOT a training experiment.

No model is loaded.
No model weights are modified.
No ground-truth mask is used to choose the crop.

The ground-truth mask is used only for evaluation of the
image-only crop strategies.

This establishes whether an inference-safe localization method
can approximate the promising crop strategy identified in Part 25.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
)

PART25_DIR = (
    OUTPUT_ROOT
    / "crop_strategy_comparison"
)

PART24_DIR = (
    OUTPUT_ROOT
    / "adaptive_crop_analysis"
)

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
    OUTPUT_ROOT
    / "image_only_localization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# INPUT FILES
# ============================================================================

PART25_CASE_FILE = (
    PART25_DIR
    / "crop_strategy_case_analysis.csv"
)

PART24_CASE_FILE = (
    PART24_DIR
    / "adaptive_crop_case_analysis.csv"
)


# ============================================================================
# TARGET MODEL INPUT SIZE
# ============================================================================

TARGET_SHAPE = np.array(
    [96, 96, 96],
    dtype=int
)


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

CLASS_IDS = {
    "vertebrae": 1,
    "spinal_canal": 2,
    "intervertebral_disc": 3,
}


# ============================================================================
# PRINT HELPER
# ============================================================================

def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================================
# LOAD MHA
# ============================================================================

def load_mha(path):
    image = sitk.ReadImage(
        str(path)
    )

    return sitk.GetArrayFromImage(
        image
    )


# ============================================================================
# NORMALIZE IMAGE
# ============================================================================

def robust_normalize(image):
    """
    Robust percentile normalization.

    1st percentile -> 0
    99th percentile -> 1
    """

    image = image.astype(
        np.float32
    )

    finite_values = image[
        np.isfinite(image)
    ]

    if finite_values.size == 0:
        return np.zeros_like(
            image,
            dtype=np.float32
        )

    low = np.percentile(
        finite_values,
        1
    )

    high = np.percentile(
        finite_values,
        99
    )

    if high <= low:

        return np.zeros_like(
            image,
            dtype=np.float32
        )

    normalized = (
        image - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return normalized.astype(
        np.float32
    )


# ============================================================================
# CENTER CROP / PAD
# ============================================================================

def crop_using_center(
    array,
    center,
    target_shape=TARGET_SHAPE
):
    """
    Crop or pad a 3-D volume around center.

    Array:
        Z, Y, X

    Center:
        Z, Y, X
    """

    target_shape = np.asarray(
        target_shape,
        dtype=int
    )

    center = np.asarray(
        center,
        dtype=float
    )

    center = np.round(
        center
    ).astype(int)

    output = np.zeros(
        target_shape,
        dtype=array.dtype
    )

    input_shape = np.asarray(
        array.shape,
        dtype=int
    )

    start = (
        center
        - target_shape // 2
    )

    end = (
        start
        + target_shape
    )

    src_start = np.maximum(
        start,
        0
    )

    src_end = np.minimum(
        end,
        input_shape
    )

    dst_start = np.maximum(
        -start,
        0
    )

    dst_end = (
        dst_start
        + (
            src_end
            - src_start
        )
    )

    output[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = array[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]

    return output


# ============================================================================
# IMAGE NONZERO CENTER
# ============================================================================

def nonzero_image_center(
    image
):
    """
    Estimate body/anatomical center from non-zero voxels.

    No ground-truth information is used.
    """

    finite = np.isfinite(
        image
    )

    nonzero = (
        finite
        & (np.abs(image) > 1e-8)
    )

    coordinates = np.argwhere(
        nonzero
    )

    if len(coordinates) == 0:

        return (
            np.asarray(
                image.shape,
                dtype=float
            )
            / 2.0
        )

    minimum = coordinates.min(
        axis=0
    )

    maximum = coordinates.max(
        axis=0
    )

    return (
        minimum
        + maximum
    ) / 2.0


# ============================================================================
# ROBUST INTENSITY FOREGROUND CENTER
# ============================================================================

def robust_intensity_center(
    image
):
    """
    Estimate anatomical foreground using a robust
    intensity threshold.

    The threshold is based on the 10th percentile
    of non-zero normalized intensities.

    This is intentionally image-only.
    """

    normalized = robust_normalize(
        image
    )

    positive = normalized[
        normalized > 0
    ]

    if positive.size == 0:

        return (
            np.asarray(
                image.shape,
                dtype=float
            )
            / 2.0
        )

    threshold = np.percentile(
        positive,
        10
    )

    foreground = (
        normalized >= threshold
    )

    coordinates = np.argwhere(
        foreground
    )

    if len(coordinates) == 0:

        return (
            np.asarray(
                image.shape,
                dtype=float
            )
            / 2.0
        )

    minimum = coordinates.min(
        axis=0
    )

    maximum = coordinates.max(
        axis=0
    )

    return (
        minimum
        + maximum
    ) / 2.0


# ============================================================================
# INTENSITY WEIGHTED CENTER
# ============================================================================

def intensity_weighted_center(
    image
):
    """
    Estimate center using intensity-weighted coordinates.

    Very low intensities are suppressed.

    No segmentation mask is used.
    """

    normalized = robust_normalize(
        image
    )

    weights = np.maximum(
        normalized,
        0
    )

    total_weight = weights.sum()

    if total_weight <= 0:

        return (
            np.asarray(
                image.shape,
                dtype=float
            )
            / 2.0
        )

    coordinates = np.indices(
        image.shape,
        dtype=np.float32
    )

    center = np.array([
        np.sum(
            coordinates[0]
            * weights
        ) / total_weight,

        np.sum(
            coordinates[1]
            * weights
        ) / total_weight,

        np.sum(
            coordinates[2]
            * weights
        ) / total_weight,
    ])

    return center


# ============================================================================
# HYBRID IMAGE CENTER
# ============================================================================

def hybrid_image_center(
    image
):
    """
    Hybrid image-only center.

    Combines:
        - non-zero anatomical center
        - robust intensity center
        - intensity weighted center

    Equal weighting is used.

    No ground-truth information is used.
    """

    center_1 = nonzero_image_center(
        image
    )

    center_2 = robust_intensity_center(
        image
    )

    center_3 = intensity_weighted_center(
        image
    )

    centers = np.vstack([
        center_1,
        center_2,
        center_3,
    ])

    return np.mean(
        centers,
        axis=0
    )


# ============================================================================
# RETENTION
# ============================================================================

def calculate_retention(
    original_mask,
    cropped_mask,
    class_id
):
    original_count = np.sum(
        original_mask == class_id
    )

    cropped_count = np.sum(
        cropped_mask == class_id
    )

    if original_count == 0:
        return np.nan

    return float(
        cropped_count
        / original_count
    )


def calculate_retention_set(
    original_mask,
    cropped_mask
):
    result = {}

    for name, class_id in CLASS_IDS.items():

        result[name] = calculate_retention(
            original_mask,
            cropped_mask,
            class_id
        )

    original_foreground = np.sum(
        original_mask > 0
    )

    cropped_foreground = np.sum(
        cropped_mask > 0
    )

    if original_foreground > 0:

        result["foreground"] = float(
            cropped_foreground
            / original_foreground
        )

    else:

        result["foreground"] = np.nan

    result[
        "combined"
    ] = np.nanmean([
        result["vertebrae"],
        result["spinal_canal"],
        result["intervertebral_disc"],
    ])

    result[
        "minimum_class"
    ] = np.nanmin([
        result["vertebrae"],
        result["spinal_canal"],
        result["intervertebral_disc"],
    ])

    return result


# ============================================================================
# CENTER DISTANCE
# ============================================================================

def center_distance(
    predicted_center,
    reference_center
):
    """
    Euclidean distance in voxels.
    """

    return float(
        np.linalg.norm(
            np.asarray(
                predicted_center
            )
            -
            np.asarray(
                reference_center
            )
        )
    )


# ============================================================================
# LOAD CASES
# ============================================================================

print_header(
    "PHASE 3 - PART 26"
)

print(
    "IMAGE-ONLY ANATOMY LOCALIZATION ANALYSIS"
)

print("=" * 78)

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


print_header(
    "LOADING PART 25 RESULTS"
)

if PART25_CASE_FILE.exists():

    part25 = pd.read_csv(
        PART25_CASE_FILE
    )

    cases = (
        part25["file"]
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"Cases from Part 25: "
        f"{len(cases)}"
    )

else:

    raise FileNotFoundError(
        f"Part 25 case file not found:\n"
        f"{PART25_CASE_FILE}"
    )


# ============================================================================
# MODEL DICE
# ============================================================================

dice_lookup = {}

if "model_dice" in part25.columns:

    dice_data = (
        part25[
            [
                "file",
                "model_dice"
            ]
        ]
        .drop_duplicates(
            subset=["file"]
        )
    )

    dice_lookup = dict(
        zip(
            dice_data["file"].astype(str),
            pd.to_numeric(
                dice_data["model_dice"],
                errors="coerce"
            )
        )
    )


# ============================================================================
# STRATEGIES
# ============================================================================

strategies = [
    "fixed_center",
    "image_nonzero_center",
    "robust_intensity_center",
    "intensity_weighted_center",
    "hybrid_image_center",
]


print_header(
    "IMAGE-ONLY LOCALIZATION STRATEGIES"
)

for strategy in strategies:
    print(
        f"✓ {strategy}"
    )


# ============================================================================
# ANALYSIS
# ============================================================================

print_header(
    "RUNNING IMAGE-ONLY LOCALIZATION"
)

rows = []


for index, case in enumerate(
    cases,
    start=1
):

    print(
        f"Analyzing {index:2d} / "
        f"{len(cases)} : {case}"
    )

    image_path = (
        IMAGE_DIR
        / f"{case}.mha"
    )

    mask_path = (
        MASK_DIR
        / f"{case}.mha"
    )

    if not image_path.exists():

        print(
            "  WARNING: image missing"
        )

        continue

    if not mask_path.exists():

        print(
            "  WARNING: mask missing"
        )

        continue


    try:

        image = load_mha(
            image_path
        )

        mask = load_mha(
            mask_path
        )

    except Exception as exc:

        print(
            f"  ERROR: {exc}"
        )

        continue


    image_shape = np.asarray(
        image.shape,
        dtype=float
    )


    # ------------------------------------------------------------------------
    # IMAGE-ONLY CENTERS
    # ------------------------------------------------------------------------

    fixed_center = (
        image_shape
        / 2.0
    )

    nonzero_center = (
        nonzero_image_center(
            image
        )
    )

    robust_center = (
        robust_intensity_center(
            image
        )
    )

    weighted_center = (
        intensity_weighted_center(
            image
        )
    )

    hybrid_center = (
        hybrid_image_center(
            image
        )
    )


    centers = {

        "fixed_center":
            fixed_center,

        "image_nonzero_center":
            nonzero_center,

        "robust_intensity_center":
            robust_center,

        "intensity_weighted_center":
            weighted_center,

        "hybrid_image_center":
            hybrid_center,
    }


    # ------------------------------------------------------------------------
    # GROUND-TRUTH REFERENCE CENTER
    # ------------------------------------------------------------------------
    #
    # IMPORTANT:
    # This center is NOT used to create the crop.
    #
    # It is calculated only after the image-only centers have been
    # generated, so that localization error can be measured.
    # ------------------------------------------------------------------------

    foreground_coordinates = np.argwhere(
        mask > 0
    )

    if len(foreground_coordinates) > 0:

        reference_min = (
            foreground_coordinates.min(
                axis=0
            )
        )

        reference_max = (
            foreground_coordinates.max(
                axis=0
            )
        )

        reference_center = (
            reference_min
            + reference_max
        ) / 2.0

    else:

        reference_center = (
            image_shape
            / 2.0
        )


    # ------------------------------------------------------------------------
    # PROCESS STRATEGIES
    # ------------------------------------------------------------------------

    for strategy, center in centers.items():

        cropped_mask = crop_using_center(
            mask,
            center
        )

        retention = calculate_retention_set(
            mask,
            cropped_mask
        )

        localization_error = (
            center_distance(
                center,
                reference_center
            )
        )


        rows.append({

            "file":
                case,

            "strategy":
                strategy,

            "model_dice":
                dice_lookup.get(
                    case,
                    np.nan
                ),

            "center_z":
                float(center[0]),

            "center_y":
                float(center[1]),

            "center_x":
                float(center[2]),

            "reference_center_z":
                float(reference_center[0]),

            "reference_center_y":
                float(reference_center[1]),

            "reference_center_x":
                float(reference_center[2]),

            "localization_error_voxels":
                localization_error,

            "foreground_retention":
                retention["foreground"],

            "vertebrae_retention":
                retention["vertebrae"],

            "spinal_canal_retention":
                retention["spinal_canal"],

            "intervertebral_disc_retention":
                retention[
                    "intervertebral_disc"
                ],

            "combined_anatomical_retention":
                retention["combined"],

            "minimum_class_retention":
                retention["minimum_class"],
        })


# ============================================================================
# DATAFRAME
# ============================================================================

results = pd.DataFrame(
    rows
)

if len(results) == 0:

    raise RuntimeError(
        "No image-only localization results "
        "were generated."
    )


numeric_columns = [
    column
    for column in results.columns
    if column not in [
        "file",
        "strategy",
    ]
]

for column in numeric_columns:

    results[column] = pd.to_numeric(
        results[column],
        errors="coerce"
    )


# ============================================================================
# SUMMARY
# ============================================================================

print_header(
    "IMAGE-ONLY STRATEGY SUMMARY"
)

summary_rows = []


for strategy in strategies:

    subset = results[
        results["strategy"]
        == strategy
    ]

    if len(subset) == 0:
        continue

    summary_rows.append({

        "strategy":
            strategy,

        "cases":
            len(subset),

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
                    "intervertebral_disc_retention"
                ].mean()
            ),

        "mean_combined_retention":
            float(
                subset[
                    "combined_anatomical_retention"
                ].mean()
            ),

        "mean_minimum_class_retention":
            float(
                subset[
                    "minimum_class_retention"
                ].mean()
            ),

        "canal_retention_below_0.02":
            int(
                (
                    subset[
                        "spinal_canal_retention"
                    ]
                    < 0.02
                ).sum()
            ),
    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================================
# RANKINGS
# ============================================================================

summary = summary.sort_values(
    [
        "mean_combined_retention",
        "mean_minimum_class_retention",
    ],
    ascending=False
).reset_index(
    drop=True
)

summary.insert(
    0,
    "rank",
    np.arange(
        1,
        len(summary) + 1
    )
)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================================
# BEST IMAGE-ONLY STRATEGY
# ============================================================================

print_header(
    "BEST IMAGE-ONLY LOCALIZATION STRATEGY"
)

best_strategy = (
    summary.iloc[0]["strategy"]
)

best_combined = float(
    summary.iloc[0][
        "mean_combined_retention"
    ]
)

best_balanced = float(
    summary.iloc[0][
        "mean_minimum_class_retention"
    ]
)

print(
    f"Best image-only strategy:"
)

print(
    f"  {best_strategy}"
)

print(
    f"Combined anatomical retention:"
    f" {best_combined:.6f}"
)

print(
    f"Balanced minimum-class retention:"
    f" {best_balanced:.6f}"
)


# ============================================================================
# FIXED BASELINE
# ============================================================================

fixed = summary[
    summary["strategy"]
    == "fixed_center"
]

if len(fixed) > 0:

    fixed_combined = float(
        fixed.iloc[0][
            "mean_combined_retention"
        ]
    )

    fixed_balanced = float(
        fixed.iloc[0][
            "mean_minimum_class_retention"
        ]
    )

    print()

    print(
        f"Fixed-center combined retention:"
        f" {fixed_combined:.6f}"
    )

    print(
        f"Improvement:"
        f" {best_combined - fixed_combined:+.6f}"
    )

    print()

    print(
        f"Fixed-center balanced retention:"
        f" {fixed_balanced:.6f}"
    )

    print(
        f"Improvement:"
        f" {best_balanced - fixed_balanced:+.6f}"
    )


# ============================================================================
# CANAL ANALYSIS
# ============================================================================

print_header(
    "SPINAL CANAL RETENTION"
)

canal_table = summary[
    [
        "rank",
        "strategy",
        "mean_spinal_canal_retention",
        "canal_retention_below_0.02",
    ]
]

print(
    canal_table.to_string(
        index=False
    )
)


# ============================================================================
# COMPARISON WITH ORACLE DISC CENTER
# ============================================================================

print_header(
    "COMPARISON WITH PART 25 ORACLE STRATEGY"
)

if PART25_CASE_FILE.exists():

    part25 = pd.read_csv(
        PART25_CASE_FILE
    )

    oracle_disc = part25[
        part25["strategy"]
        == "disc_centered"
    ]

    if len(oracle_disc) > 0:

        oracle_mean = float(
            oracle_disc[
                "combined_anatomical_retention"
            ].mean()
        )

        print(
            "Part 25 disc-centered "
            "oracle retention:"
        )

        print(
            f"  {oracle_mean:.6f}"
        )

        print()

        print(
            "Best image-only "
            "retention:"
        )

        print(
            f"  {best_combined:.6f}"
        )

        print()

        print(
            "Gap to oracle:"
        )

        print(
            f"  {oracle_mean - best_combined:+.6f}"
        )

    else:

        oracle_mean = np.nan

else:

    oracle_mean = np.nan


# ============================================================================
# PER-CASE BEST IMAGE-ONLY STRATEGY
# ============================================================================

print_header(
    "PER-CASE IMAGE-ONLY BEST STRATEGY"
)

per_case_rows = []

for case in cases:

    subset = results[
        results["file"]
        == case
    ].copy()

    if len(subset) == 0:
        continue

    subset = subset.sort_values(
        "combined_anatomical_retention",
        ascending=False
    )

    best_row = subset.iloc[0]

    per_case_rows.append({

        "file":
            case,

        "model_dice":
            best_row["model_dice"],

        "best_image_only_strategy":
            best_row["strategy"],

        "combined_retention":
            best_row[
                "combined_anatomical_retention"
            ],

        "minimum_class_retention":
            best_row[
                "minimum_class_retention"
            ],

        "canal_retention":
            best_row[
                "spinal_canal_retention"
            ],

        "localization_error_voxels":
            best_row[
                "localization_error_voxels"
            ],
    })


per_case = pd.DataFrame(
    per_case_rows
)

print(
    per_case.to_string(
        index=False
    )
)


# ============================================================================
# SAVE TABLES
# ============================================================================

print_header(
    "SAVING ANALYSIS TABLES"
)

case_path = (
    OUTPUT_DIR
    / "image_only_localization_case_analysis.csv"
)

results.to_csv(
    case_path,
    index=False
)

print(
    f"Saved: {case_path}"
)


summary_path = (
    OUTPUT_DIR
    / "image_only_localization_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    f"Saved: {summary_path}"
)


per_case_path = (
    OUTPUT_DIR
    / "image_only_per_case_best_strategy.csv"
)

per_case.to_csv(
    per_case_path,
    index=False
)

print(
    f"Saved: {per_case_path}"
)


# ============================================================================
# CHART 1 — COMBINED RETENTION
# ============================================================================

print_header(
    "CREATING CHARTS"
)

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    summary["strategy"],
    summary[
        "mean_combined_retention"
    ]
)

plt.xlabel(
    "Image-Only Localization Strategy"
)

plt.ylabel(
    "Mean Combined Anatomical Retention"
)

plt.title(
    "Image-Only Localization Strategy Comparison"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "image_only_combined_retention.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    f"Saved: {path}"
)


# ============================================================================
# CHART 2 — CLASS RETENTION
# ============================================================================

plt.figure(
    figsize=(11, 6)
)

x = np.arange(
    len(summary)
)

width = 0.25

plt.bar(
    x - width,
    summary[
        "mean_vertebrae_retention"
    ],
    width,
    label="Vertebrae"
)

plt.bar(
    x,
    summary[
        "mean_spinal_canal_retention"
    ],
    width,
    label="Spinal Canal"
)

plt.bar(
    x + width,
    summary[
        "mean_disc_retention"
    ],
    width,
    label="Intervertebral Disc"
)

plt.xlabel(
    "Image-Only Localization Strategy"
)

plt.ylabel(
    "Mean Retention"
)

plt.title(
    "Anatomical Retention by Image-Only Strategy"
)

plt.xticks(
    x,
    summary["strategy"],
    rotation=35,
    ha="right"
)

plt.legend()

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "image_only_class_retention.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    f"Saved: {path}"
)


# ============================================================================
# CHART 3 — LOCALIZATION ERROR
# ============================================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    summary["strategy"],
    summary[
        "mean_localization_error_voxels"
    ]
)

plt.xlabel(
    "Image-Only Localization Strategy"
)

plt.ylabel(
    "Mean Localization Error (voxels)"
)

plt.title(
    "Image-Only Localization Error"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "image_only_localization_error.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    f"Saved: {path}"
)


# ============================================================================
# CHART 4 — BALANCED RETENTION
# ============================================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    summary["strategy"],
    summary[
        "mean_minimum_class_retention"
    ]
)

plt.xlabel(
    "Image-Only Localization Strategy"
)

plt.ylabel(
    "Mean Minimum-Class Retention"
)

plt.title(
    "Balanced Anatomical Retention"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "image_only_balanced_retention.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    f"Saved: {path}"
)


# ============================================================================
# CHART 5 — CANAL FAILURES
# ============================================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    summary["strategy"],
    summary[
        "canal_retention_below_0.02"
    ]
)

plt.xlabel(
    "Image-Only Localization Strategy"
)

plt.ylabel(
    "Number of Cases"
)

plt.title(
    "Low Spinal Canal Retention Cases"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "image_only_canal_failure_count.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    f"Saved: {path}"
)


# ============================================================================
# JSON SUMMARY
# ============================================================================

print_header(
    "CREATING FINAL SUMMARY"
)

json_summary = {

    "part":
        "Phase 3 - Part 26",

    "purpose":
        "Develop realistic image-only anatomy "
        "localization strategies.",

    "cases_analyzed":
        int(
            results["file"].nunique()
        ),

    "strategies":
        strategies,

    "best_image_only_strategy":
        str(
            best_strategy
        ),

    "best_combined_retention":
        best_combined,

    "best_balanced_retention":
        best_balanced,

    "fixed_center_combined_retention":
        (
            fixed_combined
            if len(fixed) > 0
            else None
        ),

    "fixed_center_balanced_retention":
        (
            fixed_balanced
            if len(fixed) > 0
            else None
        ),

    "oracle_disc_center_retention":
        (
            None
            if pd.isna(oracle_mean)
            else oracle_mean
        ),

    "strategy_summary":
        summary.to_dict(
            orient="records"
        ),

    "per_case_best":
        per_case.to_dict(
            orient="records"
        ),

    "ground_truth_used_for_crop":
        False,

    "ground_truth_used_for_evaluation":
        True,

    "training_performed":
        False,

    "model_weights_modified":
        False,

    "important_limitation":
        "Ground-truth masks are used only after "
        "image-only localization to calculate "
        "retention and localization error. "
        "They are not used to determine crop centers.",
}


json_path = (
    OUTPUT_DIR
    / "phase3_part26_image_only_localization_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        json_summary,
        file,
        indent=4
    )

print(
    f"Saved: {json_path}"
)


# ============================================================================
# REPORT
# ============================================================================

print_header(
    "CREATING REPORT"
)

report_path = (
    OUTPUT_DIR
    / "phase3_part26_image_only_localization_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "=" * 78 + "\n"
    )

    file.write(
        "PHASE 3 - PART 26\n"
    )

    file.write(
        "IMAGE-ONLY ANATOMY LOCALIZATION ANALYSIS\n"
    )

    file.write(
        "=" * 78 + "\n\n"
    )

    file.write(
        "PURPOSE\n"
    )

    file.write(
        "Evaluate whether realistic image-only localization "
        "can approximate the anatomy-aware cropping "
        "identified in Parts 24-25.\n\n"
    )

    file.write(
        "IMAGE-ONLY STRATEGIES\n"
    )

    for strategy in strategies:

        file.write(
            f"- {strategy}\n"
        )

    file.write("\n")

    file.write(
        f"Cases analyzed: "
        f"{results['file'].nunique()}\n\n"
    )

    file.write(
        "BEST IMAGE-ONLY STRATEGY\n"
    )

    file.write(
        f"{best_strategy}\n"
    )

    file.write(
        f"Combined anatomical retention: "
        f"{best_combined:.6f}\n"
    )

    file.write(
        f"Balanced minimum-class retention: "
        f"{best_balanced:.6f}\n\n"
    )

    file.write(
        "STRATEGY SUMMARY\n"
    )

    file.write(
        summary.to_string(
            index=False
        )
    )

    file.write("\n\n")

    file.write(
        "PER-CASE BEST STRATEGIES\n"
    )

    file.write(
        per_case.to_string(
            index=False
        )
    )

    file.write("\n\n")

    file.write(
        "METHODOLOGICAL NOTE\n"
    )

    file.write(
        "Ground-truth masks were NOT used to select "
        "the crop centers. They were used only after "
        "crop generation to calculate anatomical "
        "retention and localization error.\n\n"
    )

    file.write(
        "NO TRAINING WAS PERFORMED.\n"
    )

    file.write(
        "NO MODEL WEIGHTS WERE MODIFIED.\n"
    )

print(
    f"Saved: {report_path}"
)


# ============================================================================
# FINAL
# ============================================================================

print_header(
    "PART 26 COMPLETE"
)

print(
    f"Cases analyzed : "
    f"{results['file'].nunique()}"
)

print()

print(
    "Best image-only strategy:"
)

print(
    f"  {best_strategy}"
)

print(
    f"Combined retention:"
    f" {best_combined:.6f}"
)

print(
    f"Balanced retention:"
    f" {best_balanced:.6f}"
)

print()

print(
    "Ground-truth used for crop: NO"
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

print()
print("=" * 78)
print("OUTPUT DIRECTORY")
print("=" * 78)
print(OUTPUT_DIR)

print()
print("=" * 78)
print("PHASE 3 - PART 26 COMPLETE")
print("=" * 78)