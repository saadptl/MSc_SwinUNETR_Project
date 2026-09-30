"""
PHASE 3 - PART 17
SPINAL CANAL FAILURE PATTERN & SEQUENCE ANALYSIS

Purpose:
    Analyze the 15 genuine Spinal Canal failure cases identified
    in Phase 3 Part 16.

Analysis:
    1. MRI sequence/type analysis
    2. Failure severity distribution
    3. Ground-truth spinal canal size analysis
    4. Precision / recall analysis
    5. False-positive / false-negative behavior
    6. Ground-truth size vs Dice relationship
    7. Sequence-wise Dice comparison
    8. Successful vs failure case comparison
    9. Identification of representative failure cases
   10. Research-oriented interpretation

Classes:
    0 - Background
    1 - Vertebrae
    2 - Spinal Canal
    3 - Intervertebral Disc

Important:
    This script does NOT retrain the model.
    It does NOT modify checkpoints.
    It only analyzes existing evaluation results.
"""

from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

start_time = time.time()

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PART11_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "test_evaluation"
    / "test_case_results.csv"
)

PART14_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "spinal_canal_analysis"
    / "spinal_canal_case_analysis.csv"
)

PART16_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "class_presence_audit"
    / "ground_truth_class_presence.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "spinal_canal_failure_pattern"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

FAILURE_THRESHOLD = 0.70
SEVERE_FAILURE_THRESHOLD = 0.50


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("PHASE 3 - PART 17")
print("SPINAL CANAL FAILURE PATTERN & SEQUENCE ANALYSIS")
print("=" * 78)

print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nOUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================
# VALIDATE INPUTS
# ============================================================

required_files = [
    PART11_RESULTS,
    PART14_RESULTS,
    PART16_RESULTS,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 78)
print("LOADING PART 11 / PART 14 / PART 16 RESULTS")
print("=" * 78)

part11_df = pd.read_csv(
    PART11_RESULTS
)

part14_df = pd.read_csv(
    PART14_RESULTS
)

part16_df = pd.read_csv(
    PART16_RESULTS
)

print(
    f"Part 11 cases: {len(part11_df)}"
)

print(
    f"Part 14 cases: {len(part14_df)}"
)

print(
    f"Part 16 cases: {len(part16_df)}"
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "mean_foreground_dice",
    "mean_foreground_iou",

    "Vertebrae_dice",
    "Vertebrae_iou",
    "Vertebrae_precision",
    "Vertebrae_recall",

    "Spinal Canal_dice",
    "Spinal Canal_iou",
    "Spinal Canal_precision",
    "Spinal Canal_recall",

    "Intervertebral Disc_dice",
    "Intervertebral Disc_iou",
    "Intervertebral Disc_precision",
    "Intervertebral Disc_recall",

    "spinal_canal_voxels",
    "spinal_canal_centroid_distance",
]


for df in [
    part11_df,
    part14_df,
    part16_df,
]:

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )


# ============================================================
# MERGE RESULTS
# ============================================================

print("\n" + "=" * 78)
print("MERGING ANALYSIS RESULTS")
print("=" * 78)

df = pd.merge(
    part11_df,
    part16_df,
    on="file",
    how="left",
    suffixes=("", "_audit"),
)

# Add Part 14 information where available
part14_keep = [
    column
    for column in [
        "file",
        "spinal_canal_dice",
        "spinal_canal_iou",
        "spinal_canal_precision",
        "spinal_canal_recall",
        "centroid_distance",
        "tp_voxels",
        "fp_voxels",
        "fn_voxels",
        "failure_category",
    ]
    if column in part14_df.columns
]

if len(part14_keep) > 1:

    df = pd.merge(
        df,
        part14_df[part14_keep],
        on="file",
        how="left",
        suffixes=("", "_part14"),
    )


print(
    f"Merged cases: {len(df)}"
)


# ============================================================
# IDENTIFY SPINAL CANAL METRIC COLUMNS
# ============================================================

# Part 11 is the official source of the final test Dice.
dice_column = "Spinal Canal_dice"

precision_column = "Spinal Canal_precision"

recall_column = "Spinal Canal_recall"

iou_column = "Spinal Canal_iou"


for column in [
    dice_column,
    precision_column,
    recall_column,
    iou_column,
]:

    if column not in df.columns:

        raise KeyError(
            f"Required column missing: {column}"
        )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# EXTRACT MRI SEQUENCE
# ============================================================

print("\n" + "=" * 78)
print("IDENTIFYING MRI SEQUENCES")
print("=" * 78)


def identify_sequence(filename):
    """
    Identify the sequence from the case filename.

    Expected examples:
        161_t1
        166_t2
        166_t2_SPACE
        69_t2_SPACE

    Returns:
        T1
        T2
        T2_SPACE
        UNKNOWN
    """

    name = str(filename).upper()

    if "SPACE" in name:

        return "T2_SPACE"

    if re.search(
        r"_T1$",
        name,
    ):

        return "T1"

    if re.search(
        r"_T2$",
        name,
    ):

        return "T2"

    # fallback
    if "_T1_" in name:

        return "T1"

    if "_T2_" in name:

        return "T2"

    return "UNKNOWN"


df["sequence"] = df[
    "file"
].apply(
    identify_sequence
)


print(
    "\nSequence distribution:"
)

print(
    df["sequence"]
    .value_counts()
    .to_string()
)


# ============================================================
# CREATE PATIENT / STUDY ID
# ============================================================

def extract_patient_id(filename):

    name = str(filename)

    match = re.match(
        r"^([^_]+)",
        name,
    )

    if match:

        return match.group(1)

    return name


df["patient_id"] = df[
    "file"
].apply(
    extract_patient_id
)


# ============================================================
# FAILURE LABELS
# ============================================================

df["failure_status"] = np.where(
    df[dice_column] < FAILURE_THRESHOLD,
    "Failure",
    "Acceptable",
)

df["severe_failure"] = np.where(
    df[dice_column] < SEVERE_FAILURE_THRESHOLD,
    "Severe Failure",
    "Not Severe",
)


def classify_failure(row):

    dice = float(
        row[dice_column]
    )

    precision = float(
        row[precision_column]
    )

    recall = float(
        row[recall_column]
    )

    if dice >= FAILURE_THRESHOLD:

        return "Acceptable"

    if dice < SEVERE_FAILURE_THRESHOLD:

        if recall < 0.10:

            return "Major Under-segmentation"

        if precision < 0.50 and recall >= 0.50:

            return "Mixed / Localization Error"

        return "Severe Segmentation Failure"

    # 0.50 <= Dice < 0.70

    if recall < precision:

        return "Under-segmentation"

    if precision < recall:

        return "Over-segmentation"

    return "Moderate Segmentation Error"


df["failure_pattern"] = df.apply(
    classify_failure,
    axis=1,
)


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n" + "=" * 78)
print("OVERALL SPINAL CANAL PERFORMANCE")
print("=" * 78)

dice_values = df[
    dice_column
].dropna()

print(
    f"Test cases: {len(dice_values)}"
)

print(
    f"Mean Dice: {dice_values.mean():.6f}"
)

print(
    f"Median Dice: {dice_values.median():.6f}"
)

print(
    f"Std Dice: {dice_values.std():.6f}"
)

print(
    f"Minimum Dice: {dice_values.min():.6f}"
)

print(
    f"Maximum Dice: {dice_values.max():.6f}"
)


# ============================================================
# FAILURE COUNTS
# ============================================================

print("\n" + "=" * 78)
print("FAILURE COUNTS")
print("=" * 78)

failure_count = int(
    (
        df[dice_column]
        < FAILURE_THRESHOLD
    ).sum()
)

severe_failure_count = int(
    (
        df[dice_column]
        < SEVERE_FAILURE_THRESHOLD
    ).sum()
)

acceptable_count = int(
    (
        df[dice_column]
        >= FAILURE_THRESHOLD
    ).sum()
)

print(
    f"Acceptable (Dice >= 0.70): "
    f"{acceptable_count} "
    f"({acceptable_count / len(df) * 100:.2f}%)"
)

print(
    f"Failure (Dice < 0.70): "
    f"{failure_count} "
    f"({failure_count / len(df) * 100:.2f}%)"
)

print(
    f"Severe (Dice < 0.50): "
    f"{severe_failure_count} "
    f"({severe_failure_count / len(df) * 100:.2f}%)"
)


# ============================================================
# FAILURE PATTERN DISTRIBUTION
# ============================================================

print("\n" + "=" * 78)
print("FAILURE PATTERN DISTRIBUTION")
print("=" * 78)

pattern_counts = (
    df[
        df["failure_status"]
        == "Failure"
    ]["failure_pattern"]
    .value_counts()
)

if len(pattern_counts) == 0:

    print(
        "No failure cases detected."
    )

else:

    for pattern, count in (
        pattern_counts.items()
    ):

        print(
            f"{pattern:<35} "
            f"{count:3d} "
            f"({count / failure_count * 100:.2f}%)"
        )


# ============================================================
# SEQUENCE-WISE PERFORMANCE
# ============================================================

print("\n" + "=" * 78)
print("SEQUENCE-WISE SPINAL CANAL PERFORMANCE")
print("=" * 78)

sequence_records = []

for sequence, group in (
    df.groupby("sequence")
):

    dice = group[
        dice_column
    ]

    failures = (
        dice < FAILURE_THRESHOLD
    ).sum()

    severe = (
        dice < SEVERE_FAILURE_THRESHOLD
    ).sum()

    record = {
        "sequence": sequence,
        "cases": int(len(group)),
        "mean_dice": float(
            dice.mean()
        ),
        "median_dice": float(
            dice.median()
        ),
        "std_dice": float(
            dice.std()
        )
        if len(dice) > 1
        else 0.0,
        "minimum_dice": float(
            dice.min()
        ),
        "maximum_dice": float(
            dice.max()
        ),
        "failures_below_0_70": int(
            failures
        ),
        "failure_percentage": float(
            failures
            / len(group)
            * 100.0
        ),
        "severe_failures_below_0_50": int(
            severe
        ),
    }

    sequence_records.append(
        record
    )

    print(
        f"\n{sequence}"
    )

    print(
        f"  Cases: {len(group)}"
    )

    print(
        f"  Mean Dice: {dice.mean():.6f}"
    )

    print(
        f"  Median Dice: {dice.median():.6f}"
    )

    print(
        f"  Failure < 0.70: "
        f"{failures} "
        f"({failures / len(group) * 100:.2f}%)"
    )

    print(
        f"  Severe < 0.50: "
        f"{severe}"
    )


sequence_df = pd.DataFrame(
    sequence_records
)


# ============================================================
# SEQUENCE FAILURE CASES
# ============================================================

print("\n" + "=" * 78)
print("FAILURES BY MRI SEQUENCE")
print("=" * 78)

sequence_failure_table = (
    df[
        df["failure_status"]
        == "Failure"
    ]
    .groupby("sequence")
    .agg(
        failure_cases=(
            "file",
            "count",
        ),
        mean_failure_dice=(
            dice_column,
            "mean",
        ),
    )
    .reset_index()
)

if len(
    sequence_failure_table
) > 0:

    print(
        sequence_failure_table.to_string(
            index=False
        )
    )


# ============================================================
# GROUND-TRUTH SPINAL CANAL SIZE
# ============================================================

print("\n" + "=" * 78)
print("SPINAL CANAL GROUND-TRUTH SIZE ANALYSIS")
print("=" * 78)

voxel_column = "spinal_canal_voxels"

if voxel_column not in df.columns:

    raise KeyError(
        "spinal_canal_voxels "
        "column not found."
    )

df[voxel_column] = pd.to_numeric(
    df[voxel_column],
    errors="coerce",
)


print(
    f"Minimum voxels: "
    f"{df[voxel_column].min():,.0f}"
)

print(
    f"Median voxels: "
    f"{df[voxel_column].median():,.0f}"
)

print(
    f"Mean voxels: "
    f"{df[voxel_column].mean():,.0f}"
)

print(
    f"Maximum voxels: "
    f"{df[voxel_column].max():,.0f}"
)


# ============================================================
# SIZE GROUPS
# ============================================================

df["canal_size_group"] = pd.qcut(
    df[voxel_column],
    q=3,
    labels=[
        "Small",
        "Medium",
        "Large",
    ],
    duplicates="drop",
)


print("\n" + "=" * 78)
print("SPINAL CANAL SIZE GROUP ANALYSIS")
print("=" * 78)

size_records = []

for size_group, group in (
    df.groupby(
        "canal_size_group",
        observed=False,
    )
):

    if len(group) == 0:

        continue

    dice = group[
        dice_column
    ]

    failures = (
        dice < FAILURE_THRESHOLD
    ).sum()

    record = {
        "size_group": str(
            size_group
        ),
        "cases": int(
            len(group)
        ),
        "mean_ground_truth_voxels":
            float(
                group[
                    voxel_column
                ].mean()
            ),
        "mean_dice":
            float(
                dice.mean()
            ),
        "median_dice":
            float(
                dice.median()
            ),
        "failure_cases":
            int(
                failures
            ),
        "failure_percentage":
            float(
                failures
                / len(group)
                * 100.0
            ),
    }

    size_records.append(
        record
    )

    print(
        f"\n{size_group}"
    )

    print(
        f"  Cases: {len(group)}"
    )

    print(
        f"  Mean canal voxels: "
        f"{group[voxel_column].mean():,.0f}"
    )

    print(
        f"  Mean Dice: "
        f"{dice.mean():.6f}"
    )

    print(
        f"  Failure < 0.70: "
        f"{failures} "
        f"({failures / len(group) * 100:.2f}%)"
    )


size_df = pd.DataFrame(
    size_records
)


# ============================================================
# DICE CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("CORRELATION ANALYSIS")
print("=" * 78)

correlation_records = []

for feature, label in [
    (
        voxel_column,
        "Ground-truth spinal canal size",
    ),
    (
        precision_column,
        "Precision",
    ),
    (
        recall_column,
        "Recall",
    ),
    (
        iou_column,
        "IoU",
    ),
]:

    valid = df[
        [
            feature,
            dice_column,
        ]
    ].dropna()

    if len(valid) < 2:

        correlation = np.nan

    else:

        correlation = (
            valid[
                feature
            ].corr(
                valid[
                    dice_column
                ]
            )
        )

    correlation_records.append(
        {
            "feature": label,
            "column": feature,
            "pearson_correlation_with_dice":
                float(correlation)
                if not pd.isna(
                    correlation
                )
                else None,
        }
    )

    print(
        f"{label:<40}: "
        f"{correlation:.6f}"
        if not pd.isna(correlation)
        else
        f"{label:<40}: NaN"
    )


correlation_df = pd.DataFrame(
    correlation_records
)


# ============================================================
# BEST / WORST CASES
# ============================================================

print("\n" + "=" * 78)
print("WORST SPINAL CANAL CASES")
print("=" * 78)

worst_cases = (
    df[
        [
            "file",
            "sequence",
            dice_column,
            iou_column,
            precision_column,
            recall_column,
            voxel_column,
            "failure_pattern",
        ]
    ]
    .sort_values(
        dice_column
    )
    .head(15)
)

print(
    worst_cases.to_string(
        index=False
    )
)


print("\n" + "=" * 78)
print("BEST SPINAL CANAL CASES")
print("=" * 78)

best_cases = (
    df[
        [
            "file",
            "sequence",
            dice_column,
            iou_column,
            precision_column,
            recall_column,
            voxel_column,
            "failure_pattern",
        ]
    ]
    .sort_values(
        dice_column,
        ascending=False,
    )
    .head(10)
)

print(
    best_cases.to_string(
        index=False
    )
)


# ============================================================
# TRUE FAILURE TABLE
# ============================================================

failure_df = (
    df[
        df["failure_status"]
        == "Failure"
    ]
    .sort_values(
        dice_column
    )
    .copy()
)

print("\n" + "=" * 78)
print("TRUE FAILURE CASE SUMMARY")
print("=" * 78)

print(
    f"Total genuine failure cases: "
    f"{len(failure_df)}"
)

if len(failure_df) > 0:

    print(
        failure_df[
            [
                "file",
                "sequence",
                dice_column,
                iou_column,
                precision_column,
                recall_column,
                voxel_column,
                "failure_pattern",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SUCCESS VS FAILURE COMPARISON
# ============================================================

print("\n" + "=" * 78)
print("SUCCESS VS FAILURE COMPARISON")
print("=" * 78)

comparison_records = []

for group_name, group in df.groupby(
    "failure_status"
):

    comparison_records.append(
        {
            "group": group_name,
            "cases": len(group),
            "mean_dice":
                group[
                    dice_column
                ].mean(),
            "median_dice":
                group[
                    dice_column
                ].median(),
            "mean_precision":
                group[
                    precision_column
                ].mean(),
            "mean_recall":
                group[
                    recall_column
                ].mean(),
            "mean_iou":
                group[
                    iou_column
                ].mean(),
            "mean_ground_truth_voxels":
                group[
                    voxel_column
                ].mean(),
            "median_ground_truth_voxels":
                group[
                    voxel_column
                ].median(),
        }
    )


comparison_df = pd.DataFrame(
    comparison_records
)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# SEQUENCE DISTRIBUTION OF FAILURES
# ============================================================

failure_sequence_counts = (
    failure_df[
        "sequence"
    ]
    .value_counts()
    .rename_axis(
        "sequence"
    )
    .reset_index(
        name="failure_cases"
    )
)

if len(
    failure_sequence_counts
) > 0:

    failure_sequence_counts[
        "failure_percentage"
    ] = (
        failure_sequence_counts[
            "failure_cases"
        ]
        / len(failure_df)
        * 100.0
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

print("\n" + "=" * 78)
print("SAVING ANALYSIS TABLES")
print("=" * 78)

main_analysis_csv = (
    OUTPUT_DIR
    / "spinal_canal_failure_pattern_analysis.csv"
)

failures_csv = (
    OUTPUT_DIR
    / "spinal_canal_true_failure_cases.csv"
)

sequence_csv = (
    OUTPUT_DIR
    / "spinal_canal_sequence_analysis.csv"
)

size_csv = (
    OUTPUT_DIR
    / "spinal_canal_size_group_analysis.csv"
)

correlation_csv = (
    OUTPUT_DIR
    / "spinal_canal_correlation_analysis.csv"
)

comparison_csv = (
    OUTPUT_DIR
    / "spinal_canal_success_vs_failure.csv"
)

failure_sequence_csv = (
    OUTPUT_DIR
    / "spinal_canal_failure_by_sequence.csv"
)


df.to_csv(
    main_analysis_csv,
    index=False,
)

failure_df.to_csv(
    failures_csv,
    index=False,
)

sequence_df.to_csv(
    sequence_csv,
    index=False,
)

size_df.to_csv(
    size_csv,
    index=False,
)

correlation_df.to_csv(
    correlation_csv,
    index=False,
)

comparison_df.to_csv(
    comparison_csv,
    index=False,
)

failure_sequence_counts.to_csv(
    failure_sequence_csv,
    index=False,
)


print(
    f"Main analysis:\n{main_analysis_csv}"
)

print(
    f"True failures:\n{failures_csv}"
)

print(
    f"Sequence analysis:\n{sequence_csv}"
)

print(
    f"Size analysis:\n{size_csv}"
)

print(
    f"Correlation analysis:\n{correlation_csv}"
)

print(
    f"Success vs failure:\n{comparison_csv}"
)

print(
    f"Failure by sequence:\n{failure_sequence_csv}"
)


# ============================================================
# CHART 1
# SEQUENCE-WISE DICE
# ============================================================

print("\n" + "=" * 78)
print("CREATING VISUAL ANALYSIS CHARTS")
print("=" * 78)

sequence_order = [
    sequence
    for sequence in [
        "T1",
        "T2",
        "T2_SPACE",
        "UNKNOWN",
    ]
    if sequence in df[
        "sequence"
    ].unique()
]

sequence_values = []

for sequence in sequence_order:

    sequence_values.append(
        df[
            df["sequence"]
            == sequence
        ][dice_column].mean()
    )


plt.figure(
    figsize=(9, 6)
)

plt.bar(
    sequence_order,
    sequence_values,
)

plt.ylim(
    0,
    1.0,
)

plt.ylabel(
    "Mean Spinal Canal Dice"
)

plt.xlabel(
    "MRI Sequence"
)

plt.title(
    "Spinal Canal Dice by MRI Sequence"
)

plt.tight_layout()

sequence_dice_chart = (
    OUTPUT_DIR
    / "spinal_canal_dice_by_sequence.png"
)

plt.savefig(
    sequence_dice_chart,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {sequence_dice_chart}"
)


# ============================================================
# CHART 2
# FAILURE COUNT BY SEQUENCE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

failure_counts_by_sequence = []

for sequence in sequence_order:

    count = int(
        (
            failure_df[
                "sequence"
            ]
            == sequence
        ).sum()
    )

    failure_counts_by_sequence.append(
        count
    )

plt.bar(
    sequence_order,
    failure_counts_by_sequence,
)

plt.ylabel(
    "Failure Cases (Dice < 0.70)"
)

plt.xlabel(
    "MRI Sequence"
)

plt.title(
    "Spinal Canal Failures by MRI Sequence"
)

plt.tight_layout()

sequence_failure_chart = (
    OUTPUT_DIR
    / "spinal_canal_failures_by_sequence.png"
)

plt.savefig(
    sequence_failure_chart,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {sequence_failure_chart}"
)


# ============================================================
# CHART 3
# DICE VS GROUND-TRUTH SIZE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    df[
        voxel_column
    ],
    df[
        dice_column
    ],
)

plt.axhline(
    FAILURE_THRESHOLD,
    linestyle="--",
)

plt.xlabel(
    "Ground-Truth Spinal Canal Voxels"
)

plt.ylabel(
    "Spinal Canal Dice"
)

plt.title(
    "Spinal Canal Size vs Dice"
)

plt.tight_layout()

size_dice_chart = (
    OUTPUT_DIR
    / "spinal_canal_size_vs_dice.png"
)

plt.savefig(
    size_dice_chart,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {size_dice_chart}"
)


# ============================================================
# CHART 4
# PRECISION VS RECALL
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    df[
        precision_column
    ],
    df[
        recall_column
    ],
)

plt.xlabel(
    "Precision"
)

plt.ylabel(
    "Recall"
)

plt.title(
    "Spinal Canal Precision vs Recall"
)

plt.xlim(
    0,
    1.05,
)

plt.ylim(
    0,
    1.05,
)

plt.tight_layout()

precision_recall_chart = (
    OUTPUT_DIR
    / "spinal_canal_precision_vs_recall.png"
)

plt.savefig(
    precision_recall_chart,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {precision_recall_chart}"
)


# ============================================================
# CHART 5
# FAILURE DICE DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    df[
        dice_column
    ],
    bins=12,
)

plt.axvline(
    FAILURE_THRESHOLD,
    linestyle="--",
)

plt.xlabel(
    "Spinal Canal Dice"
)

plt.ylabel(
    "Number of Cases"
)

plt.title(
    "Spinal Canal Dice Distribution"
)

plt.tight_layout()

dice_distribution_chart = (
    OUTPUT_DIR
    / "spinal_canal_failure_dice_distribution.png"
)

plt.savefig(
    dice_distribution_chart,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {dice_distribution_chart}"
)


# ============================================================
# CHART 6
# SUCCESS VS FAILURE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

groups = [
    "Acceptable",
    "Failure",
]

group_means = []

for group_name in groups:

    subset = df[
        df["failure_status"]
        == group_name
    ]

    if len(subset) > 0:

        group_means.append(
            subset[
                voxel_column
            ].mean()
        )

    else:

        group_means.append(
            0
        )


plt.bar(
    groups,
    group_means,
)

plt.ylabel(
    "Mean Ground-Truth Spinal Canal Voxels"
)

plt.xlabel(
    "Performance Group"
)

plt.title(
    "Spinal Canal Size: Successful vs Failure Cases"
)

plt.tight_layout()

success_failure_size_chart = (
    OUTPUT_DIR
    / "success_vs_failure_spinal_canal_size.png"
)

plt.savefig(
    success_failure_size_chart,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {success_failure_size_chart}"
)


# ============================================================
# JSON SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("CREATING JSON SUMMARY")
print("=" * 78)

best_case = (
    df.sort_values(
        dice_column,
        ascending=False,
    )
    .iloc[0]
)

worst_case = (
    df.sort_values(
        dice_column,
        ascending=True,
    )
    .iloc[0]
)


# Find sequence with lowest mean Dice
if len(sequence_df) > 0:

    weakest_sequence_row = (
        sequence_df.sort_values(
            "mean_dice"
        )
        .iloc[0]
    )

    strongest_sequence_row = (
        sequence_df.sort_values(
            "mean_dice",
            ascending=False,
        )
        .iloc[0]
    )

else:

    weakest_sequence_row = None
    strongest_sequence_row = None


json_summary = {
    "phase": "Phase 3 - Part 17",

    "test_cases": int(
        len(df)
    ),

    "failure_threshold": FAILURE_THRESHOLD,

    "severe_failure_threshold":
        SEVERE_FAILURE_THRESHOLD,

    "overall": {
        "mean_dice":
            float(
                dice_values.mean()
            ),
        "median_dice":
            float(
                dice_values.median()
            ),
        "std_dice":
            float(
                dice_values.std()
            ),
        "minimum_dice":
            float(
                dice_values.min()
            ),
        "maximum_dice":
            float(
                dice_values.max()
            ),
    },

    "failure_counts": {
        "acceptable_cases":
            acceptable_count,
        "failure_cases":
            failure_count,
        "severe_failure_cases":
            severe_failure_count,
        "failure_percentage":
            float(
                failure_count
                / len(df)
                * 100.0
            ),
        "severe_failure_percentage":
            float(
                severe_failure_count
                / len(df)
                * 100.0
            ),
    },

    "worst_case": {
        "file":
            str(
                worst_case["file"]
            ),
        "sequence":
            str(
                worst_case["sequence"]
            ),
        "dice":
            float(
                worst_case[dice_column]
            ),
        "precision":
            float(
                worst_case[
                    precision_column
                ]
            ),
        "recall":
            float(
                worst_case[
                    recall_column
                ]
            ),
    },

    "best_case": {
        "file":
            str(
                best_case["file"]
            ),
        "sequence":
            str(
                best_case["sequence"]
            ),
        "dice":
            float(
                best_case[dice_column]
            ),
    },

    "correlations":
        correlation_df.to_dict(
            orient="records"
        ),

    "sequence_analysis":
        sequence_df.to_dict(
            orient="records"
        ),

    "size_analysis":
        size_df.to_dict(
            orient="records"
        ),

    "failure_patterns":
        pattern_counts.to_dict(),

    "files": {
        "main_analysis":
            str(
                main_analysis_csv
            ),
        "true_failures":
            str(
                failures_csv
            ),
        "sequence_analysis":
            str(
                sequence_csv
            ),
        "size_analysis":
            str(
                size_csv
            ),
        "correlation_analysis":
            str(
                correlation_csv
            ),
        "comparison":
            str(
                comparison_csv
            ),
        "failure_by_sequence":
            str(
                failure_sequence_csv
            ),
        "sequence_dice_chart":
            str(
                sequence_dice_chart
            ),
        "sequence_failure_chart":
            str(
                sequence_failure_chart
            ),
        "size_dice_chart":
            str(
                size_dice_chart
            ),
        "precision_recall_chart":
            str(
                precision_recall_chart
            ),
        "dice_distribution_chart":
            str(
                dice_distribution_chart
            ),
        "success_failure_size_chart":
            str(
                success_failure_size_chart
            ),
    },
}

json_path = (
    OUTPUT_DIR
    / "spinal_canal_failure_pattern_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_summary,
        f,
        indent=4,
        default=str,
    )

print(
    f"Saved: {json_path}"
)


# ============================================================
# RESEARCH REPORT
# ============================================================

print("\n" + "=" * 78)
print("CREATING RESEARCH REPORT")
print("=" * 78)

report_path = (
    OUTPUT_DIR
    / "phase3_part17_spinal_canal_failure_pattern_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "PHASE 3 - PART 17\n"
    )

    f.write(
        "SPINAL CANAL FAILURE PATTERN & SEQUENCE ANALYSIS\n"
    )

    f.write(
        "=" * 72
        + "\n\n"
    )

    f.write(
        "OBJECTIVE\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        "To investigate the characteristics of Spinal Canal "
        "segmentation failures identified during test-set "
        "evaluation, including MRI sequence, anatomical "
        "target size, precision, recall and failure pattern.\n\n"
    )

    f.write(
        "OVERALL PERFORMANCE\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        f"Test cases: {len(df)}\n"
    )

    f.write(
        f"Mean Dice: {dice_values.mean():.6f}\n"
    )

    f.write(
        f"Median Dice: {dice_values.median():.6f}\n"
    )

    f.write(
        f"Standard deviation: {dice_values.std():.6f}\n"
    )

    f.write(
        f"Minimum Dice: {dice_values.min():.6f}\n"
    )

    f.write(
        f"Maximum Dice: {dice_values.max():.6f}\n\n"
    )

    f.write(
        "FAILURE SUMMARY\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        f"Acceptable cases (Dice >= 0.70): "
        f"{acceptable_count}\n"
    )

    f.write(
        f"Failure cases (Dice < 0.70): "
        f"{failure_count}\n"
    )

    f.write(
        f"Severe failures (Dice < 0.50): "
        f"{severe_failure_count}\n\n"
    )

    f.write(
        "SEQUENCE ANALYSIS\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    for _, row in (
        sequence_df.iterrows()
    ):

        f.write(
            f"{row['sequence']}: "
            f"cases={row['cases']}, "
            f"mean Dice={row['mean_dice']:.6f}, "
            f"median Dice={row['median_dice']:.6f}, "
            f"failures={row['failures_below_0_70']} "
            f"({row['failure_percentage']:.2f}%)\n"
        )

    if weakest_sequence_row is not None:

        f.write(
            "\nLowest mean Dice sequence: "
            f"{weakest_sequence_row['sequence']} "
            f"({weakest_sequence_row['mean_dice']:.6f})\n"
        )

        f.write(
            "Highest mean Dice sequence: "
            f"{strongest_sequence_row['sequence']} "
            f"({strongest_sequence_row['mean_dice']:.6f})\n"
        )

    f.write(
        "\nFAILURE PATTERNS\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    if len(
        pattern_counts
    ) > 0:

        for pattern, count in (
            pattern_counts.items()
        ):

            f.write(
                f"{pattern}: "
                f"{count} cases\n"
            )

    else:

        f.write(
            "No failure patterns detected.\n"
        )

    f.write(
        "\nCORRELATION ANALYSIS\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    for _, row in (
        correlation_df.iterrows()
    ):

        correlation = row[
            "pearson_correlation_with_dice"
        ]

        if pd.isna(
            correlation
        ):

            correlation_text = "NaN"

        else:

            correlation_text = (
                f"{correlation:.6f}"
            )

        f.write(
            f"{row['feature']}: "
            f"{correlation_text}\n"
        )

    f.write(
        "\nWORST CASE\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        f"Case: {worst_case['file']}\n"
    )

    f.write(
        f"Sequence: {worst_case['sequence']}\n"
    )

    f.write(
        f"Dice: {worst_case[dice_column]:.6f}\n"
    )

    f.write(
        f"Precision: "
        f"{worst_case[precision_column]:.6f}\n"
    )

    f.write(
        f"Recall: "
        f"{worst_case[recall_column]:.6f}\n"
    )

    f.write(
        f"Ground-truth voxels: "
        f"{worst_case[voxel_column]:,.0f}\n"
    )

    f.write(
        "\nBEST CASE\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        f"Case: {best_case['file']}\n"
    )

    f.write(
        f"Sequence: {best_case['sequence']}\n"
    )

    f.write(
        f"Dice: {best_case[dice_column]:.6f}\n"
    )

    f.write(
        "\nRESEARCH INTERPRETATION\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        "The analysis is intended to identify patterns "
        "associated with poor Spinal Canal segmentation. "
        "Sequence-level differences should be interpreted "
        "descriptively because the number of cases in each "
        "sequence group may be limited.\n"
    )

    f.write(
        "Correlation values indicate association rather "
        "than causation. They should not be interpreted as "
        "evidence that a particular factor directly causes "
        "segmentation failure.\n"
    )

    f.write(
        "The results should be used to guide subsequent "
        "model-improvement experiments rather than "
        "selecting hyperparameters solely from the test set.\n"
    )


# ============================================================
# COMPLETION
# ============================================================

elapsed = (
    time.time()
    - start_time
)

print("\n" + "=" * 78)
print("PART 17 COMPLETE")
print("=" * 78)

print(
    f"Test cases analyzed: "
    f"{len(df)}"
)

print(
    f"Genuine failures: "
    f"{failure_count}"
)

print(
    f"Severe failures: "
    f"{severe_failure_count}"
)

if weakest_sequence_row is not None:

    print(
        f"Weakest sequence by mean Dice: "
        f"{weakest_sequence_row['sequence']} "
        f"({weakest_sequence_row['mean_dice']:.6f})"
    )

if len(
    correlation_df
) > 0:

    size_correlation = (
        correlation_df[
            correlation_df[
                "column"
            ]
            == voxel_column
        ]
    )

    if len(
        size_correlation
    ) > 0:

        value = size_correlation.iloc[0][
            "pearson_correlation_with_dice"
        ]

        if not pd.isna(value):

            print(
                f"Canal size vs Dice correlation: "
                f"{value:.6f}"
            )

print(
    f"Execution time: "
    f"{elapsed / 60:.2f} minutes"
)

print("\n" + "=" * 78)
print("OUTPUT FILES")
print("=" * 78)

print(
    f"Main analysis:\n"
    f"{main_analysis_csv}"
)

print(
    f"True failures:\n"
    f"{failures_csv}"
)

print(
    f"Sequence analysis:\n"
    f"{sequence_csv}"
)

print(
    f"Size analysis:\n"
    f"{size_csv}"
)

print(
    f"Correlation analysis:\n"
    f"{correlation_csv}"
)

print(
    f"Success vs failure:\n"
    f"{comparison_csv}"
)

print(
    f"Failure by sequence:\n"
    f"{failure_sequence_csv}"
)

print(
    f"JSON summary:\n"
    f"{json_path}"
)

print(
    f"Research report:\n"
    f"{report_path}"
)

print("\n" + "=" * 78)
print("PHASE 3 - PART 17 COMPLETE")
print("=" * 78)