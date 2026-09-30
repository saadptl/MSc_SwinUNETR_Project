"""
PHASE 3 - PART 24
ADAPTIVE CROP FEASIBILITY & FAILURE-RECOVERY ANALYSIS

Purpose
-------
Evaluate whether an anatomy-aware crop could improve anatomical retention
for the T2 SPACE cases analyzed in Parts 19-23.

IMPORTANT:
    This is an analysis-only experiment.
    No model retraining is performed.
    The existing best_model.pth is not modified.

The analysis compares:

    Current fixed center crop
        vs
    Anatomy-aware bounding-box crop

using the existing ground-truth masks.

This establishes whether preprocessing/localization is a plausible
source of the observed T2 SPACE segmentation failures before retraining.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleITK as sitk

warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "segmentation"

PART22_DIR = OUTPUT_ROOT / "t2_space_anatomy_alignment"
PART23_DIR = OUTPUT_ROOT / "crop_retention_analysis"
PART19_DIR = OUTPUT_ROOT / "t2_space_visual_inspection"

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
    / "adaptive_crop_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# INPUT FILES
# ============================================================================

PART22_CASES = (
    PART22_DIR
    / "t2_space_anatomy_alignment_case_analysis.csv"
)

PART23_MASTER = (
    PART23_DIR
    / "crop_retention_master_analysis.csv"
)

PART19_RESULTS = (
    PART19_DIR
    / "t2_space_visual_inspection_summary.csv"
)


# ============================================================================
# MODEL INPUT SIZE
# ============================================================================

TARGET_SHAPE = np.array(
    [96, 96, 96],
    dtype=int
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_mha(path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return array


def center_crop_or_pad(
    array,
    target_shape=TARGET_SHAPE
):
    """
    Center crop/pad a 3-D volume.

    Input:
        Z, Y, X

    Output:
        96, 96, 96
    """

    target_shape = np.asarray(
        target_shape,
        dtype=int
    )

    output = np.zeros(
        target_shape,
        dtype=array.dtype
    )

    input_shape = np.asarray(
        array.shape,
        dtype=int
    )

    src_start = np.maximum(
        (input_shape - target_shape) // 2,
        0
    )

    src_end = np.minimum(
        src_start + target_shape,
        input_shape
    )

    dst_start = np.maximum(
        (target_shape - input_shape) // 2,
        0
    )

    dst_end = dst_start + (
        src_end - src_start
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


def crop_using_center(
    array,
    center,
    target_shape=TARGET_SHAPE
):
    """
    Crop/pad around a supplied center.

    center:
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


def get_class_mask(
    mask,
    class_id
):
    return mask == class_id


def calculate_retention(
    original_mask,
    cropped_mask,
    class_id
):
    """
    Fraction of original class voxels retained in crop.
    """

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


def calculate_centroid(
    binary_mask
):
    """
    Returns centroid in Z,Y,X coordinates.
    """

    coordinates = np.argwhere(
        binary_mask
    )

    if len(coordinates) == 0:
        return None

    return coordinates.mean(
        axis=0
    )


def get_foreground_centroid(
    mask
):
    coordinates = np.argwhere(
        mask > 0
    )

    if len(coordinates) == 0:
        return None

    return coordinates.mean(
        axis=0
    )


def bbox_from_mask(
    mask,
    margin=0
):
    """
    Bounding box around all foreground anatomy.

    Returns:
        min_coord, max_coord
    """

    coordinates = np.argwhere(
        mask > 0
    )

    if len(coordinates) == 0:
        return None

    minimum = coordinates.min(
        axis=0
    )

    maximum = coordinates.max(
        axis=0
    )

    minimum = np.maximum(
        minimum - margin,
        0
    )

    maximum = np.minimum(
        maximum + margin,
        np.asarray(mask.shape) - 1
    )

    return minimum, maximum


def bbox_center(
    mask,
    margin=0
):
    bbox = bbox_from_mask(
        mask,
        margin=margin
    )

    if bbox is None:
        return None

    minimum, maximum = bbox

    return (
        minimum
        + maximum
    ) / 2.0


def retention_for_crop(
    original_mask,
    cropped_mask
):
    results = {}

    for class_id, class_name in [
        (1, "vertebrae"),
        (2, "spinal_canal"),
        (3, "intervertebral_disc"),
    ]:

        results[class_name] = calculate_retention(
            original_mask,
            cropped_mask,
            class_id
        )

    foreground_original = np.sum(
        original_mask > 0
    )

    foreground_crop = np.sum(
        cropped_mask > 0
    )

    if foreground_original > 0:
        results["foreground"] = float(
            foreground_crop
            / foreground_original
        )
    else:
        results["foreground"] = np.nan

    return results


def safe_mean(values):
    values = pd.Series(
        values,
        dtype="float64"
    )

    return float(
        values.mean()
    )


def safe_corr(x, y):
    data = pd.DataFrame({
        "x": x,
        "y": y
    }).dropna()

    if len(data) < 3:
        return np.nan

    if data["x"].nunique() < 2:
        return np.nan

    if data["y"].nunique() < 2:
        return np.nan

    return float(
        data["x"].corr(
            data["y"]
        )
    )


# ============================================================================
# START
# ============================================================================

print_header(
    "PHASE 3 - PART 24"
)

print(
    "ADAPTIVE CROP FEASIBILITY & "
    "FAILURE-RECOVERY ANALYSIS"
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


# ============================================================================
# LOAD PART 22
# ============================================================================

print_header(
    "LOADING PART 22 RESULTS"
)

if not PART22_CASES.exists():
    raise FileNotFoundError(
        f"Part 22 results not found:\n"
        f"{PART22_CASES}"
    )

part22 = pd.read_csv(
    PART22_CASES
)

print(
    f"Part 22 cases: {len(part22)}"
)


# ============================================================================
# LOAD PART 23
# ============================================================================

print_header(
    "LOADING PART 23 RESULTS"
)

if PART23_MASTER.exists():

    part23 = pd.read_csv(
        PART23_MASTER
    )

    print(
        f"Part 23 cases: {len(part23)}"
    )

else:

    part23 = None

    print(
        "Part 23 master analysis not found."
    )


# ============================================================================
# DETERMINE CASE LIST
# ============================================================================

if "file" not in part22.columns:
    raise RuntimeError(
        "Part 22 CSV does not contain "
        "the 'file' column."
    )

cases = part22[
    "file"
].astype(str).tolist()

print()
print(
    f"Cases selected for analysis: {len(cases)}"
)


# ============================================================================
# MODEL PERFORMANCE
# ============================================================================

if "model_dice" in part22.columns:

    dice_lookup = dict(
        zip(
            part22["file"].astype(str),
            pd.to_numeric(
                part22["model_dice"],
                errors="coerce"
            )
        )
    )

else:

    dice_lookup = {}


# ============================================================================
# ANALYSIS
# ============================================================================

print_header(
    "ADAPTIVE CROP ANALYSIS"
)

print(
    "Current crop : fixed center 96 x 96 x 96"
)

print(
    "Adaptive crop: crop centered on ground-truth "
    "foreground anatomy"
)

print()
print(
    "IMPORTANT: The adaptive crop uses ground-truth "
    "masks only to estimate the maximum achievable "
    "preprocessing benefit."
)

print(
    "It is NOT being used as a deployable inference "
    "method and does NOT change model weights."
)


results = []


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
            f"  ERROR loading case: "
            f"{exc}"
        )

        continue


    # ------------------------------------------------------------------------
    # CURRENT CENTER CROP
    # ------------------------------------------------------------------------

    current_crop = center_crop_or_pad(
        mask
    )

    current_retention = (
        retention_for_crop(
            mask,
            current_crop
        )
    )


    # ------------------------------------------------------------------------
    # ANATOMY-AWARE CROP
    # ------------------------------------------------------------------------

    anatomy_center = (
        get_foreground_centroid(
            mask
        )
    )

    if anatomy_center is None:

        print(
            "  WARNING: no foreground "
            "anatomy found"
        )

        continue


    adaptive_crop = crop_using_center(
        mask,
        anatomy_center
    )

    adaptive_retention = (
        retention_for_crop(
            mask,
            adaptive_crop
        )
    )


    # ------------------------------------------------------------------------
    # CLASS-SPECIFIC CENTERS
    # ------------------------------------------------------------------------

    vertebrae_center = bbox_center(
        get_class_mask(
            mask,
            1
        )
    )

    canal_center = bbox_center(
        get_class_mask(
            mask,
            2
        )
    )

    disc_center = bbox_center(
        get_class_mask(
            mask,
            3
        )
    )


    # ------------------------------------------------------------------------
    # CLASS-SPECIFIC RETENTION
    # ------------------------------------------------------------------------

    if vertebrae_center is not None:

        vertebrae_crop = crop_using_center(
            mask,
            vertebrae_center
        )

        vertebrae_retention = (
            retention_for_crop(
                mask,
                vertebrae_crop
            )
        )

    else:

        vertebrae_retention = {
            "foreground": np.nan,
            "vertebrae": np.nan,
            "spinal_canal": np.nan,
            "intervertebral_disc": np.nan,
        }


    if canal_center is not None:

        canal_crop = crop_using_center(
            mask,
            canal_center
        )

        canal_retention = (
            retention_for_crop(
                mask,
                canal_crop
            )
        )

    else:

        canal_retention = {
            "foreground": np.nan,
            "vertebrae": np.nan,
            "spinal_canal": np.nan,
            "intervertebral_disc": np.nan,
        }


    if disc_center is not None:

        disc_crop = crop_using_center(
            mask,
            disc_center
        )

        disc_retention = (
            retention_for_crop(
                mask,
                disc_crop
            )
        )

    else:

        disc_retention = {
            "foreground": np.nan,
            "vertebrae": np.nan,
            "spinal_canal": np.nan,
            "intervertebral_disc": np.nan,
        }


    # ------------------------------------------------------------------------
    # IMPROVEMENT CALCULATIONS
    # ------------------------------------------------------------------------

    row = {
        "file": case,
        "model_dice": dice_lookup.get(
            case,
            np.nan
        ),

        "original_z": image.shape[0],
        "original_y": image.shape[1],
        "original_x": image.shape[2],

        # Current center crop
        "current_foreground_retention":
            current_retention["foreground"],

        "current_vertebrae_retention":
            current_retention["vertebrae"],

        "current_canal_retention":
            current_retention["spinal_canal"],

        "current_disc_retention":
            current_retention[
                "intervertebral_disc"
            ],

        # Foreground adaptive crop
        "adaptive_foreground_retention":
            adaptive_retention["foreground"],

        "adaptive_vertebrae_retention":
            adaptive_retention["vertebrae"],

        "adaptive_canal_retention":
            adaptive_retention["spinal_canal"],

        "adaptive_disc_retention":
            adaptive_retention[
                "intervertebral_disc"
            ],

        # Vertebra-centered crop
        "vertebrae_centered_foreground_retention":
            vertebrae_retention["foreground"],

        "vertebrae_centered_vertebrae_retention":
            vertebrae_retention["vertebrae"],

        "vertebrae_centered_canal_retention":
            vertebrae_retention["spinal_canal"],

        "vertebrae_centered_disc_retention":
            vertebrae_retention[
                "intervertebral_disc"
            ],

        # Canal-centered crop
        "canal_centered_foreground_retention":
            canal_retention["foreground"],

        "canal_centered_vertebrae_retention":
            canal_retention["vertebrae"],

        "canal_centered_canal_retention":
            canal_retention["spinal_canal"],

        "canal_centered_disc_retention":
            canal_retention[
                "intervertebral_disc"
            ],

        # Disc-centered crop
        "disc_centered_foreground_retention":
            disc_retention["foreground"],

        "disc_centered_vertebrae_retention":
            disc_retention["vertebrae"],

        "disc_centered_canal_retention":
            disc_retention["spinal_canal"],

        "disc_centered_disc_retention":
            disc_retention[
                "intervertebral_disc"
            ],
    }


    # ------------------------------------------------------------------------
    # RETENTION IMPROVEMENTS
    # ------------------------------------------------------------------------

    row[
        "foreground_retention_gain"
    ] = (
        row["adaptive_foreground_retention"]
        - row["current_foreground_retention"]
    )

    row[
        "vertebrae_retention_gain"
    ] = (
        row["adaptive_vertebrae_retention"]
        - row["current_vertebrae_retention"]
    )

    row[
        "canal_retention_gain"
    ] = (
        row["adaptive_canal_retention"]
        - row["current_canal_retention"]
    )

    row[
        "disc_retention_gain"
    ] = (
        row["adaptive_disc_retention"]
        - row["current_disc_retention"]
    )


    # Average anatomical improvement
    adaptive_values = [
        row["adaptive_foreground_retention"],
        row["adaptive_vertebrae_retention"],
        row["adaptive_canal_retention"],
        row["adaptive_disc_retention"],
    ]

    current_values = [
        row["current_foreground_retention"],
        row["current_vertebrae_retention"],
        row["current_canal_retention"],
        row["current_disc_retention"],
    ]

    row[
        "current_combined_retention"
    ] = np.nanmean(
        current_values
    )

    row[
        "adaptive_combined_retention"
    ] = np.nanmean(
        adaptive_values
    )

    row[
        "combined_retention_gain"
    ] = (
        row["adaptive_combined_retention"]
        - row["current_combined_retention"]
    )


    # ------------------------------------------------------------------------
    # SUCCESS INDICATOR
    # ------------------------------------------------------------------------

    row[
        "current_low_canal_retention"
    ] = int(
        row["current_canal_retention"]
        < 0.02
    )

    row[
        "adaptive_low_canal_retention"
    ] = int(
        row["adaptive_canal_retention"]
        < 0.02
    )


    results.append(
        row
    )


# ============================================================================
# DATAFRAME
# ============================================================================

analysis = pd.DataFrame(
    results
)

if len(analysis) == 0:
    raise RuntimeError(
        "No cases could be analyzed."
    )


# ============================================================================
# NUMERIC CLEANUP
# ============================================================================

numeric_columns = [
    c for c in analysis.columns
    if c != "file"
]

for col in numeric_columns:

    analysis[col] = pd.to_numeric(
        analysis[col],
        errors="coerce"
    )


# ============================================================================
# SUMMARY
# ============================================================================

print_header(
    "ADAPTIVE CROP FEASIBILITY SUMMARY"
)

print(
    f"Cases analyzed: {len(analysis)}"
)

print()

print(
    "Current center-crop mean retention:"
)

print(
    f"  Foreground : "
    f"{analysis['current_foreground_retention'].mean():.6f}"
)

print(
    f"  Vertebrae  : "
    f"{analysis['current_vertebrae_retention'].mean():.6f}"
)

print(
    f"  Canal      : "
    f"{analysis['current_canal_retention'].mean():.6f}"
)

print(
    f"  Disc       : "
    f"{analysis['current_disc_retention'].mean():.6f}"
)


print()

print(
    "Anatomy-aware crop mean retention:"
)

print(
    f"  Foreground : "
    f"{analysis['adaptive_foreground_retention'].mean():.6f}"
)

print(
    f"  Vertebrae  : "
    f"{analysis['adaptive_vertebrae_retention'].mean():.6f}"
)

print(
    f"  Canal      : "
    f"{analysis['adaptive_canal_retention'].mean():.6f}"
)

print(
    f"  Disc       : "
    f"{analysis['adaptive_disc_retention'].mean():.6f}"
)


# ============================================================================
# RETENTION GAINS
# ============================================================================

print_header(
    "RETENTION IMPROVEMENT"
)

gain_columns = [
    "foreground_retention_gain",
    "vertebrae_retention_gain",
    "canal_retention_gain",
    "disc_retention_gain",
    "combined_retention_gain",
]

for col in gain_columns:

    print(
        f"{col:35s}: "
        f"{analysis[col].mean():+.6f}"
    )


# ============================================================================
# CASE-WISE TABLE
# ============================================================================

print_header(
    "CASE-WISE ADAPTIVE CROP RESULTS"
)

display_columns = [
    "file",
    "model_dice",
    "current_foreground_retention",
    "adaptive_foreground_retention",
    "current_vertebrae_retention",
    "adaptive_vertebrae_retention",
    "current_canal_retention",
    "adaptive_canal_retention",
    "current_disc_retention",
    "adaptive_disc_retention",
    "combined_retention_gain",
]

display_columns = [
    c for c in display_columns
    if c in analysis.columns
]

display_df = analysis[
    display_columns
].sort_values(
    "model_dice"
)

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================================
# CANAL RECOVERY
# ============================================================================

print_header(
    "SPINAL CANAL RETENTION RECOVERY"
)

current_low = analysis[
    analysis["current_canal_retention"]
    < 0.02
]

adaptive_low = analysis[
    analysis["adaptive_canal_retention"]
    < 0.02
]

print(
    f"Current crop cases with canal retention < 0.02:"
    f" {len(current_low)}"
)

print(
    f"Adaptive crop cases with canal retention < 0.02:"
    f" {len(adaptive_low)}"
)

print()

if len(current_low) > 0:

    print(
        "Cases with low current canal retention:"
    )

    for case in current_low["file"]:
        print(
            f"  - {case}"
        )

print()

if len(adaptive_low) > 0:

    print(
        "Cases still below adaptive canal retention threshold:"
    )

    for case in adaptive_low["file"]:
        print(
            f"  - {case}"
        )

else:

    print(
        "✓ All analyzed cases exceed the "
        "0.02 canal-retention threshold "
        "under the anatomy-aware crop."
    )


# ============================================================================
# FAILURE RECOVERY POTENTIAL
# ============================================================================

print_header(
    "FAILURE-RECOVERY POTENTIAL"
)

analysis[
    "potentially_recoverable"
] = (
    (
        analysis["model_dice"] < 0.70
    )
    &
    (
        analysis["combined_retention_gain"]
        > 0
    )
)

potential_cases = analysis[
    analysis["potentially_recoverable"]
]

print(
    f"Current failure cases: "
    f"{int((analysis['model_dice'] < 0.70).sum())}"
)

print(
    f"Cases with positive retention gain: "
    f"{int((analysis['combined_retention_gain'] > 0).sum())}"
)

print(
    f"Failure cases with positive retention gain: "
    f"{len(potential_cases)}"
)

if len(potential_cases) > 0:

    print()

    print(
        "Potentially recoverable failure cases:"
    )

    for _, row in potential_cases.iterrows():

        print(
            f"  {row['file']:20s} "
            f"Dice={row['model_dice']:.6f} "
            f"Gain={row['combined_retention_gain']:+.6f}"
        )


# ============================================================================
# CORRELATION OF RETENTION GAIN WITH DICE
# ============================================================================

print_header(
    "RETENTION GAIN VS EXISTING DICE"
)

gain_correlations = []

for col in gain_columns:

    r = safe_corr(
        analysis[col],
        analysis["model_dice"]
    )

    gain_correlations.append({
        "feature": col,
        "correlation_with_existing_dice": r,
    })

    print(
        f"{col:35s}: "
        f"{r:.6f}"
    )

gain_corr_df = pd.DataFrame(
    gain_correlations
)


# ============================================================================
# SAVE MASTER CSV
# ============================================================================

print_header(
    "SAVING ANALYSIS TABLES"
)

master_path = (
    OUTPUT_DIR
    / "adaptive_crop_case_analysis.csv"
)

analysis.to_csv(
    master_path,
    index=False
)

print(
    f"Saved: {master_path}"
)


gain_path = (
    OUTPUT_DIR
    / "adaptive_crop_gain_analysis.csv"
)

gain_corr_df.to_csv(
    gain_path,
    index=False
)

print(
    f"Saved: {gain_path}"
)


# ============================================================================
# CHART 1 — CURRENT VS ADAPTIVE RETENTION
# ============================================================================

print_header(
    "CREATING ADAPTIVE CROP CHARTS"
)

features = [
    (
        "foreground",
        "Foreground Retention"
    ),
    (
        "vertebrae",
        "Vertebrae Retention"
    ),
    (
        "canal",
        "Spinal Canal Retention"
    ),
    (
        "disc",
        "Intervertebral Disc Retention"
    ),
]

for feature, label in features:

    current_col = (
        f"current_{feature}_retention"
    )

    adaptive_col = (
        f"adaptive_{feature}_retention"
    )

    plt.figure(
        figsize=(8, 6)
    )

    x = np.arange(
        len(analysis)
    )

    width = 0.36

    plt.bar(
        x - width / 2,
        analysis[current_col],
        width,
        label="Current Center Crop"
    )

    plt.bar(
        x + width / 2,
        analysis[adaptive_col],
        width,
        label="Anatomy-Aware Crop"
    )

    plt.xlabel(
        "T2 SPACE Case"
    )

    plt.ylabel(
        label
    )

    plt.title(
        f"{label}: Current vs Anatomy-Aware Crop"
    )

    plt.xticks(
        x,
        analysis["file"],
        rotation=45,
        ha="right"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / f"{feature}_current_vs_adaptive.png"
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
# CHART 2 — RETENTION GAIN
# ============================================================================

plt.figure(
    figsize=(10, 6)
)

x = np.arange(
    len(analysis)
)

plt.bar(
    x,
    analysis[
        "combined_retention_gain"
    ]
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "T2 SPACE Case"
)

plt.ylabel(
    "Combined Retention Gain"
)

plt.title(
    "Potential Anatomical Retention Gain from Adaptive Cropping"
)

plt.xticks(
    x,
    analysis["file"],
    rotation=45,
    ha="right"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "combined_retention_gain.png"
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
# CHART 3 — CURRENT VS ADAPTIVE CANAL RETENTION
# ============================================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    analysis[
        "current_canal_retention"
    ],
    analysis[
        "adaptive_canal_retention"
    ],
    s=80
)

maximum = max(
    analysis[
        "current_canal_retention"
    ].max(),

    analysis[
        "adaptive_canal_retention"
    ].max()
)

plt.plot(
    [0, maximum],
    [0, maximum],
    linestyle="--"
)

plt.xlabel(
    "Current Center-Crop Canal Retention"
)

plt.ylabel(
    "Adaptive-Crop Canal Retention"
)

plt.title(
    "Spinal Canal Retention: Current vs Adaptive Crop"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "canal_retention_current_vs_adaptive_scatter.png"
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
# CHART 4 — COMBINED RETENTION
# ============================================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    analysis[
        "current_combined_retention"
    ],
    analysis[
        "adaptive_combined_retention"
    ],
    s=80
)

maximum = max(
    analysis[
        "current_combined_retention"
    ].max(),

    analysis[
        "adaptive_combined_retention"
    ].max()
)

plt.plot(
    [0, maximum],
    [0, maximum],
    linestyle="--"
)

plt.xlabel(
    "Current Combined Retention"
)

plt.ylabel(
    "Adaptive Combined Retention"
)

plt.title(
    "Combined Anatomical Retention: Current vs Adaptive Crop"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "combined_retention_current_vs_adaptive.png"
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

summary = {

    "part":
        "Phase 3 - Part 24",

    "purpose":
        "Evaluate feasibility of anatomy-aware cropping "
        "before retraining.",

    "cases_analyzed":
        int(len(analysis)),

    "current_center_crop":
        {
            "mean_foreground_retention":
                float(
                    analysis[
                        "current_foreground_retention"
                    ].mean()
                ),

            "mean_vertebrae_retention":
                float(
                    analysis[
                        "current_vertebrae_retention"
                    ].mean()
                ),

            "mean_canal_retention":
                float(
                    analysis[
                        "current_canal_retention"
                    ].mean()
                ),

            "mean_disc_retention":
                float(
                    analysis[
                        "current_disc_retention"
                    ].mean()
                ),

            "mean_combined_retention":
                float(
                    analysis[
                        "current_combined_retention"
                    ].mean()
                ),
        },

    "adaptive_crop":
        {
            "mean_foreground_retention":
                float(
                    analysis[
                        "adaptive_foreground_retention"
                    ].mean()
                ),

            "mean_vertebrae_retention":
                float(
                    analysis[
                        "adaptive_vertebrae_retention"
                    ].mean()
                ),

            "mean_canal_retention":
                float(
                    analysis[
                        "adaptive_canal_retention"
                    ].mean()
                ),

            "mean_disc_retention":
                float(
                    analysis[
                        "adaptive_disc_retention"
                    ].mean()
                ),

            "mean_combined_retention":
                float(
                    analysis[
                        "adaptive_combined_retention"
                    ].mean()
                ),
        },

    "mean_retention_gains":
        {
            col:
                float(
                    analysis[col].mean()
                )
            for col in gain_columns
        },

    "current_cases_below_canal_threshold":
        int(len(current_low)),

    "adaptive_cases_below_canal_threshold":
        int(len(adaptive_low)),

    "potentially_recoverable_failure_cases":
        potential_cases[
            "file"
        ].tolist(),

    "gain_correlations":
        {
            row["feature"]:
                (
                    None
                    if pd.isna(
                        row[
                            "correlation_with_existing_dice"
                        ]
                    )
                    else float(
                        row[
                            "correlation_with_existing_dice"
                        ]
                    )
                )
            for _, row
            in gain_corr_df.iterrows()
        },

    "important_note":
        "Adaptive crop results are an upper-bound feasibility "
        "analysis because ground-truth masks were used to "
        "determine anatomy location. This is not a deployable "
        "inference pipeline.",
}


json_path = (
    OUTPUT_DIR
    / "phase3_part24_adaptive_crop_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print(
    f"Saved: {json_path}"
)


# ============================================================================
# TEXT REPORT
# ============================================================================

report_path = (
    OUTPUT_DIR
    / "phase3_part24_adaptive_crop_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 78 + "\n"
    )

    f.write(
        "PHASE 3 - PART 24\n"
    )

    f.write(
        "ADAPTIVE CROP FEASIBILITY & "
        "FAILURE-RECOVERY ANALYSIS\n"
    )

    f.write(
        "=" * 78 + "\n\n"
    )

    f.write(
        "PURPOSE\n"
    )

    f.write(
        "Evaluate whether an anatomy-aware crop could "
        "increase anatomical retention in T2 SPACE "
        "failure cases before model retraining.\n\n"
    )

    f.write(
        "IMPORTANT LIMITATION\n"
    )

    f.write(
        "The adaptive crop uses ground-truth segmentation "
        "masks to determine the anatomy center. Therefore "
        "the results represent feasibility / upper-bound "
        "evidence rather than deployable inference "
        "performance.\n\n"
    )

    f.write(
        f"Cases analyzed: {len(analysis)}\n\n"
    )

    f.write(
        "CURRENT CENTER CROP\n"
    )

    f.write(
        f"Foreground retention: "
        f"{analysis['current_foreground_retention'].mean():.6f}\n"
    )

    f.write(
        f"Vertebrae retention: "
        f"{analysis['current_vertebrae_retention'].mean():.6f}\n"
    )

    f.write(
        f"Canal retention: "
        f"{analysis['current_canal_retention'].mean():.6f}\n"
    )

    f.write(
        f"Disc retention: "
        f"{analysis['current_disc_retention'].mean():.6f}\n"
    )

    f.write(
        f"Combined retention: "
        f"{analysis['current_combined_retention'].mean():.6f}\n\n"
    )

    f.write(
        "ADAPTIVE CROP\n"
    )

    f.write(
        f"Foreground retention: "
        f"{analysis['adaptive_foreground_retention'].mean():.6f}\n"
    )

    f.write(
        f"Vertebrae retention: "
        f"{analysis['adaptive_vertebrae_retention'].mean():.6f}\n"
    )

    f.write(
        f"Canal retention: "
        f"{analysis['adaptive_canal_retention'].mean():.6f}\n"
    )

    f.write(
        f"Disc retention: "
        f"{analysis['adaptive_disc_retention'].mean():.6f}\n"
    )

    f.write(
        f"Combined retention: "
        f"{analysis['adaptive_combined_retention'].mean():.6f}\n\n"
    )

    f.write(
        "RETENTION GAINS\n"
    )

    for col in gain_columns:

        f.write(
            f"{col}: "
            f"{analysis[col].mean():+.6f}\n"
        )

    f.write("\n")

    f.write(
        "CANAL RETENTION THRESHOLD\n"
    )

    f.write(
        f"Current cases below 0.02: "
        f"{len(current_low)}\n"
    )

    f.write(
        f"Adaptive cases below 0.02: "
        f"{len(adaptive_low)}\n\n"
    )

    f.write(
        "POTENTIALLY RECOVERABLE FAILURE CASES\n"
    )

    for case in potential_cases[
        "file"
    ].tolist():

        f.write(
            f"- {case}\n"
        )

    f.write("\n")

    f.write(
        "CONCLUSION\n"
    )

    f.write(
        "This analysis should be interpreted as a "
        "preprocessing feasibility study. Positive "
        "retention gains indicate that anatomy-aware "
        "localization may be worth testing in a future "
        "training experiment. They do not prove that "
        "retraining with the adaptive crop will improve "
        "test Dice.\n"
    )

print(
    f"Saved: {report_path}"
)


# ============================================================================
# FINAL
# ============================================================================

print_header(
    "PART 24 COMPLETE"
)

print(
    f"Cases analyzed : {len(analysis)}"
)

print(
    f"Current canal-retention failures "
    f"(<0.02): {len(current_low)}"
)

print(
    f"Adaptive canal-retention failures "
    f"(<0.02): {len(adaptive_low)}"
)

print(
    f"Potentially recoverable cases: "
    f"{len(potential_cases)}"
)

print()
print(
    "Current combined retention : "
    f"{analysis['current_combined_retention'].mean():.6f}"
)

print(
    "Adaptive combined retention: "
    f"{analysis['adaptive_combined_retention'].mean():.6f}"
)

print()
print("=" * 78)
print("OUTPUT DIRECTORY")
print("=" * 78)
print(OUTPUT_DIR)

print()
print("=" * 78)
print("PHASE 3 - PART 24 COMPLETE")
print("=" * 78)