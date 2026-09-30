"""
==============================================================================
PHASE 3 - PART 29
PAIRED BASELINE VS PART 27 TEST COMPARISON
==============================================================================

Purpose
-------
Compare the original baseline Swin-UNETR against the Part 27
hybrid image-only localization model using the SAME 71 test cases.

Baseline:
    Part 11
    Fixed-center localization

Part 27:
    Part 28 test evaluation
    Hybrid image-only localization

This analysis does NOT:
    - train a model
    - modify model weights
    - use ground truth for localization
    - change either experiment

It performs a paired statistical/per-case comparison.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part29_paired_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# CONSTANTS
# =============================================================================

BASELINE_NAME = "Baseline Fixed-Center"
PART27_NAME = "Part 27 Hybrid Image-Only"

CLASS_NAMES = [
    "Vertebrae",
    "Spinal Canal",
    "Intervertebral Disc",
]


# =============================================================================
# HEADER
# =============================================================================

print("=" * 78)
print("PHASE 3 - PART 29")
print("PAIRED BASELINE VS PART 27 TEST COMPARISON")
print("=" * 78)

print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nBASELINE RESULTS")
print(BASELINE_RESULTS)

print("\nPART 27 RESULTS")
print(PART27_RESULTS)

print("\nOUTPUT DIRECTORY")
print(OUTPUT_DIR)


# =============================================================================
# CHECK FILES
# =============================================================================

if not BASELINE_RESULTS.exists():

    raise FileNotFoundError(
        f"\nBaseline Part 11 results not found:\n"
        f"{BASELINE_RESULTS}"
    )

if not PART27_RESULTS.exists():

    raise FileNotFoundError(
        f"\nPart 27 Part 28 results not found:\n"
        f"{PART27_RESULTS}"
    )


# =============================================================================
# LOAD RESULTS
# =============================================================================

print("\n" + "=" * 78)
print("LOADING OFFICIAL TEST RESULTS")
print("=" * 78)

baseline_df = pd.read_csv(
    BASELINE_RESULTS
)

part27_df = pd.read_csv(
    PART27_RESULTS
)

print(
    "Baseline cases:",
    len(baseline_df)
)

print(
    "Part 27 cases:",
    len(part27_df)
)


# =============================================================================
# VALIDATE REQUIRED COLUMNS
# =============================================================================

required_columns = [
    "file",
    "mean_foreground_dice",
    "mean_foreground_iou",
]

for class_name in CLASS_NAMES:

    required_columns.extend(
        [
            f"{class_name}_dice",
            f"{class_name}_iou",
            f"{class_name}_precision",
            f"{class_name}_recall",
        ]
    )


for column in required_columns:

    if column not in baseline_df.columns:

        raise ValueError(
            f"Missing baseline column: {column}"
        )

    if column not in part27_df.columns:

        raise ValueError(
            f"Missing Part 27 column: {column}"
        )


# =============================================================================
# CHECK DUPLICATES
# =============================================================================

if baseline_df["file"].duplicated().any():

    raise ValueError(
        "Duplicate case names found in baseline results."
    )

if part27_df["file"].duplicated().any():

    raise ValueError(
        "Duplicate case names found in Part 27 results."
    )


# =============================================================================
# SORT CASES
# =============================================================================

baseline_df = baseline_df.sort_values(
    "file"
).reset_index(
    drop=True
)

part27_df = part27_df.sort_values(
    "file"
).reset_index(
    drop=True
)


# =============================================================================
# MATCH CASES
# =============================================================================

baseline_cases = set(
    baseline_df["file"]
)

part27_cases = set(
    part27_df["file"]
)

common_cases = sorted(
    baseline_cases
    &
    part27_cases
)

baseline_only = sorted(
    baseline_cases
    -
    part27_cases
)

part27_only = sorted(
    part27_cases
    -
    baseline_cases
)


print("\n" + "=" * 78)
print("CASE MATCHING")
print("=" * 78)

print(
    "Baseline cases:",
    len(baseline_cases)
)

print(
    "Part 27 cases:",
    len(part27_cases)
)

print(
    "Common cases:",
    len(common_cases)
)

print(
    "Baseline-only cases:",
    len(baseline_only)
)

print(
    "Part 27-only cases:",
    len(part27_only)
)

if len(common_cases) != 71:

    raise RuntimeError(
        "Expected exactly 71 paired test cases."
    )


# =============================================================================
# MERGE
# =============================================================================

baseline_common = (
    baseline_df[
        baseline_df["file"].isin(
            common_cases
        )
    ]
    .copy()
)

part27_common = (
    part27_df[
        part27_df["file"].isin(
            common_cases
        )
    ]
    .copy()
)


baseline_common = baseline_common.sort_values(
    "file"
).reset_index(
    drop=True
)

part27_common = part27_common.sort_values(
    "file"
).reset_index(
    drop=True
)


if not np.array_equal(
    baseline_common["file"].values,
    part27_common["file"].values
):

    raise RuntimeError(
        "Case ordering mismatch after sorting."
    )


# =============================================================================
# CREATE PAIRED DATAFRAME
# =============================================================================

paired_df = pd.DataFrame()

paired_df["file"] = common_cases

# -------------------------------------------------------------------------
# Overall
# -------------------------------------------------------------------------

paired_df["baseline_dice"] = (
    baseline_common[
        "mean_foreground_dice"
    ].values
)

paired_df["part27_dice"] = (
    part27_common[
        "mean_foreground_dice"
    ].values
)

paired_df["dice_change"] = (
    paired_df["part27_dice"]
    -
    paired_df["baseline_dice"]
)

paired_df["baseline_iou"] = (
    baseline_common[
        "mean_foreground_iou"
    ].values
)

paired_df["part27_iou"] = (
    part27_common[
        "mean_foreground_iou"
    ].values
)

paired_df["iou_change"] = (
    paired_df["part27_iou"]
    -
    paired_df["baseline_iou"]
)


# -------------------------------------------------------------------------
# Per-class metrics
# -------------------------------------------------------------------------

for class_name in CLASS_NAMES:

    dice_column = (
        f"{class_name}_dice"
    )

    iou_column = (
        f"{class_name}_iou"
    )

    precision_column = (
        f"{class_name}_precision"
    )

    recall_column = (
        f"{class_name}_recall"
    )

    paired_df[
        f"baseline_{class_name}_dice"
    ] = baseline_common[
        dice_column
    ].values

    paired_df[
        f"part27_{class_name}_dice"
    ] = part27_common[
        dice_column
    ].values

    paired_df[
        f"{class_name}_dice_change"
    ] = (
        paired_df[
            f"part27_{class_name}_dice"
        ]
        -
        paired_df[
            f"baseline_{class_name}_dice"
        ]
    )

    paired_df[
        f"baseline_{class_name}_iou"
    ] = baseline_common[
        iou_column
    ].values

    paired_df[
        f"part27_{class_name}_iou"
    ] = part27_common[
        iou_column
    ].values

    paired_df[
        f"{class_name}_iou_change"
    ] = (
        paired_df[
            f"part27_{class_name}_iou"
        ]
        -
        paired_df[
            f"baseline_{class_name}_iou"
        ]
    )

    paired_df[
        f"baseline_{class_name}_precision"
    ] = baseline_common[
        precision_column
    ].values

    paired_df[
        f"part27_{class_name}_precision"
    ] = part27_common[
        precision_column
    ].values

    paired_df[
        f"{class_name}_precision_change"
    ] = (
        paired_df[
            f"part27_{class_name}_precision"
        ]
        -
        paired_df[
            f"baseline_{class_name}_precision"
        ]
    )

    paired_df[
        f"baseline_{class_name}_recall"
    ] = baseline_common[
        recall_column
    ].values

    paired_df[
        f"part27_{class_name}_recall"
    ] = part27_common[
        recall_column
    ].values

    paired_df[
        f"{class_name}_recall_change"
    ] = (
        paired_df[
            f"part27_{class_name}_recall"
        ]
        -
        paired_df[
            f"baseline_{class_name}_recall"
        ]
    )


# =============================================================================
# OVERALL SUMMARY
# =============================================================================

print("\n" + "=" * 78)
print("OVERALL PAIRED COMPARISON")
print("=" * 78)

baseline_mean_dice = (
    paired_df[
        "baseline_dice"
    ].mean()
)

part27_mean_dice = (
    paired_df[
        "part27_dice"
    ].mean()
)

mean_dice_change = (
    paired_df[
        "dice_change"
    ].mean()
)

baseline_median_dice = (
    paired_df[
        "baseline_dice"
    ].median()
)

part27_median_dice = (
    paired_df[
        "part27_dice"
    ].median()
)

median_dice_change = (
    paired_df[
        "dice_change"
    ].median()
)

baseline_mean_iou = (
    paired_df[
        "baseline_iou"
    ].mean()
)

part27_mean_iou = (
    paired_df[
        "part27_iou"
    ].mean()
)

mean_iou_change = (
    paired_df[
        "iou_change"
    ].mean()
)


print(
    f"{BASELINE_NAME:30s}: "
    f"{baseline_mean_dice:.6f}"
)

print(
    f"{PART27_NAME:30s}: "
    f"{part27_mean_dice:.6f}"
)

print(
    f"{'Mean Dice change':30s}: "
    f"{mean_dice_change:+.6f}"
)

print()

print(
    f"{'Baseline median Dice':30s}: "
    f"{baseline_median_dice:.6f}"
)

print(
    f"{'Part 27 median Dice':30s}: "
    f"{part27_median_dice:.6f}"
)

print(
    f"{'Median Dice change':30s}: "
    f"{median_dice_change:+.6f}"
)

print()

print(
    f"{'Baseline mean IoU':30s}: "
    f"{baseline_mean_iou:.6f}"
)

print(
    f"{'Part 27 mean IoU':30s}: "
    f"{part27_mean_iou:.6f}"
)

print(
    f"{'Mean IoU change':30s}: "
    f"{mean_iou_change:+.6f}"
)


# =============================================================================
# CASE-WISE IMPROVEMENT
# =============================================================================

print("\n" + "=" * 78)
print("CASE-WISE DICE CHANGE")
print("=" * 78)

improved_cases = (
    paired_df[
        paired_df["dice_change"] > 0
    ]
)

unchanged_cases = (
    paired_df[
        np.isclose(
            paired_df["dice_change"],
            0,
            atol=1e-12
        )
    ]
)

worsened_cases = (
    paired_df[
        paired_df["dice_change"] < 0
    ]
)

print(
    "Improved cases :",
    len(improved_cases),
    f"({len(improved_cases) / len(paired_df) * 100:.2f}%)"
)

print(
    "Unchanged cases:",
    len(unchanged_cases),
    f"({len(unchanged_cases) / len(paired_df) * 100:.2f}%)"
)

print(
    "Worsened cases :",
    len(worsened_cases),
    f"({len(worsened_cases) / len(paired_df) * 100:.2f}%)"
)


# =============================================================================
# LARGE IMPROVEMENTS
# =============================================================================

print("\n" + "=" * 78)
print("TOP 10 IMPROVEMENTS")
print("=" * 78)

top_improvements = (
    paired_df
    .sort_values(
        "dice_change",
        ascending=False
    )
    .head(10)
)

print(
    top_improvements[
        [
            "file",
            "baseline_dice",
            "part27_dice",
            "dice_change",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# LARGE DEGRADATIONS
# =============================================================================

print("\n" + "=" * 78)
print("TOP 10 DEGRADATIONS")
print("=" * 78)

top_degradations = (
    paired_df
    .sort_values(
        "dice_change",
        ascending=True
    )
    .head(10)
)

print(
    top_degradations[
        [
            "file",
            "baseline_dice",
            "part27_dice",
            "dice_change",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# PER-CLASS COMPARISON
# =============================================================================

print("\n" + "=" * 78)
print("PER-CLASS PAIRED DICE COMPARISON")
print("=" * 78)

class_summary = {}

for class_name in CLASS_NAMES:

    baseline_column = (
        f"baseline_{class_name}_dice"
    )

    part27_column = (
        f"part27_{class_name}_dice"
    )

    change_column = (
        f"{class_name}_dice_change"
    )

    baseline_mean = (
        paired_df[
            baseline_column
        ].mean()
    )

    part27_mean = (
        paired_df[
            part27_column
        ].mean()
    )

    mean_change = (
        paired_df[
            change_column
        ].mean()
    )

    improved = (
        paired_df[
            change_column
        ] > 0
    ).sum()

    unchanged = (
        np.isclose(
            paired_df[
                change_column
            ],
            0,
            atol=1e-12
        )
    ).sum()

    worsened = (
        paired_df[
            change_column
        ] < 0
    ).sum()

    class_summary[
        class_name
    ] = {
        "baseline_mean_dice":
            float(baseline_mean),
        "part27_mean_dice":
            float(part27_mean),
        "mean_dice_change":
            float(mean_change),
        "improved_cases":
            int(improved),
        "unchanged_cases":
            int(unchanged),
        "worsened_cases":
            int(worsened),
    }

    print(
        f"\n{class_name}"
    )

    print(
        f"  Baseline Dice : "
        f"{baseline_mean:.6f}"
    )

    print(
        f"  Part 27 Dice  : "
        f"{part27_mean:.6f}"
    )

    print(
        f"  Change        : "
        f"{mean_change:+.6f}"
    )

    print(
        f"  Improved      : {improved}"
    )

    print(
        f"  Unchanged     : {unchanged}"
    )

    print(
        f"  Worsened      : {worsened}"
    )


# =============================================================================
# PER-CLASS IOU
# =============================================================================

print("\n" + "=" * 78)
print("PER-CLASS IOU COMPARISON")
print("=" * 78)

iou_summary = {}

for class_name in CLASS_NAMES:

    baseline_column = (
        f"baseline_{class_name}_iou"
    )

    part27_column = (
        f"part27_{class_name}_iou"
    )

    change_column = (
        f"{class_name}_iou_change"
    )

    baseline_mean = (
        paired_df[
            baseline_column
        ].mean()
    )

    part27_mean = (
        paired_df[
            part27_column
        ].mean()
    )

    mean_change = (
        paired_df[
            change_column
        ].mean()
    )

    iou_summary[
        class_name
    ] = {
        "baseline_mean_iou":
            float(baseline_mean),
        "part27_mean_iou":
            float(part27_mean),
        "mean_iou_change":
            float(mean_change),
    }

    print(
        f"{class_name:25s}"
        f"Baseline: {baseline_mean:.6f}  "
        f"Part27: {part27_mean:.6f}  "
        f"Change: {mean_change:+.6f}"
    )


# =============================================================================
# PER-CLASS PRECISION / RECALL
# =============================================================================

print("\n" + "=" * 78)
print("PER-CLASS PRECISION / RECALL COMPARISON")
print("=" * 78)

precision_recall_summary = {}

for class_name in CLASS_NAMES:

    baseline_precision = (
        paired_df[
            f"baseline_{class_name}_precision"
        ].mean()
    )

    part27_precision = (
        paired_df[
            f"part27_{class_name}_precision"
        ].mean()
    )

    baseline_recall = (
        paired_df[
            f"baseline_{class_name}_recall"
        ].mean()
    )

    part27_recall = (
        paired_df[
            f"part27_{class_name}_recall"
        ].mean()
    )

    precision_recall_summary[
        class_name
    ] = {
        "baseline_precision":
            float(baseline_precision),
        "part27_precision":
            float(part27_precision),
        "precision_change":
            float(
                part27_precision
                -
                baseline_precision
            ),
        "baseline_recall":
            float(baseline_recall),
        "part27_recall":
            float(part27_recall),
        "recall_change":
            float(
                part27_recall
                -
                baseline_recall
            ),
    }

    print(
        f"\n{class_name}"
    )

    print(
        f"  Precision: "
        f"{baseline_precision:.6f}"
        f" → "
        f"{part27_precision:.6f}"
        f" "
        f"({part27_precision - baseline_precision:+.6f})"
    )

    print(
        f"  Recall   : "
        f"{baseline_recall:.6f}"
        f" → "
        f"{part27_recall:.6f}"
        f" "
        f"({part27_recall - baseline_recall:+.6f})"
    )


# =============================================================================
# DICE THRESHOLD ANALYSIS
# =============================================================================

print("\n" + "=" * 78)
print("DICE THRESHOLD COMPARISON")
print("=" * 78)

threshold_summary = {}

for threshold in [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]:

    baseline_count = (
        paired_df[
            "baseline_dice"
        ] >= threshold
    ).sum()

    part27_count = (
        paired_df[
            "part27_dice"
        ] >= threshold
    ).sum()

    threshold_summary[
        str(threshold)
    ] = {
        "baseline_cases":
            int(baseline_count),
        "part27_cases":
            int(part27_count),
        "change":
            int(
                part27_count
                -
                baseline_count
            ),
    }

    print(
        f"Dice >= {threshold:.2f}: "
        f"Baseline={baseline_count:2d}  "
        f"Part27={part27_count:2d}  "
        f"Change={part27_count - baseline_count:+d}"
    )


# =============================================================================
# LOW PERFORMANCE ANALYSIS
# =============================================================================

print("\n" + "=" * 78)
print("LOW-PERFORMANCE CASE COMPARISON")
print("=" * 78)

for threshold in [
    0.60,
    0.70,
    0.80,
]:

    baseline_count = (
        paired_df[
            "baseline_dice"
        ] < threshold
    ).sum()

    part27_count = (
        paired_df[
            "part27_dice"
        ] < threshold
    ).sum()

    print(
        f"Dice < {threshold:.2f}: "
        f"Baseline={baseline_count:2d}  "
        f"Part27={part27_count:2d}  "
        f"Change={part27_count - baseline_count:+d}"
    )


# =============================================================================
# CORRELATION
# =============================================================================

print("\n" + "=" * 78)
print("BASELINE VS PART 27 CORRELATION")
print("=" * 78)

dice_correlation = (
    paired_df[
        "baseline_dice"
    ].corr(
        paired_df[
            "part27_dice"
        ]
    )
)

print(
    "Case-wise Dice correlation:",
    f"{dice_correlation:.6f}"
)


# =============================================================================
# STATISTICAL TEST
# =============================================================================

print("\n" + "=" * 78)
print("PAIRED STATISTICAL TEST")
print("=" * 78)

try:

    from scipy.stats import wilcoxon

    statistic, p_value = wilcoxon(
        paired_df[
            "baseline_dice"
        ],
        paired_df[
            "part27_dice"
        ],
        alternative="two-sided"
    )

    print(
        "Wilcoxon signed-rank statistic:",
        f"{statistic:.6f}"
    )

    print(
        "p-value:",
        f"{p_value:.10f}"
    )

    if p_value < 0.05:

        significance = (
            "Statistically significant "
            "difference at alpha=0.05."
        )

    else:

        significance = (
            "Not statistically significant "
            "at alpha=0.05."
        )

    print(
        significance
    )

except Exception as exc:

    statistic = None
    p_value = None

    significance = (
        f"Statistical test unavailable: {exc}"
    )

    print(
        significance
    )


# =============================================================================
# EFFECT SIZE
# =============================================================================

differences = (
    paired_df[
        "dice_change"
    ].values
)

difference_std = (
    differences.std(
        ddof=1
    )
)

if difference_std > 0:

    cohens_d = (
        differences.mean()
        /
        difference_std
    )

else:

    cohens_d = 0.0


print("\n" + "=" * 78)
print("EFFECT SIZE")
print("=" * 78)

print(
    "Mean paired Dice difference:",
    f"{differences.mean():+.6f}"
)

print(
    "Std paired Dice difference:",
    f"{difference_std:.6f}"
)

print(
    "Paired Cohen's d:",
    f"{cohens_d:.6f}"
)


# =============================================================================
# SAVE PAIRED CASE TABLE
# =============================================================================

paired_path = (
    OUTPUT_DIR
    /
    "baseline_vs_part27_case_comparison.csv"
)

paired_df.to_csv(
    paired_path,
    index=False
)

print("\n" + "=" * 78)
print("SAVING ANALYSIS TABLES")
print("=" * 78)

print(
    "Saved:",
    paired_path
)


# =============================================================================
# SAVE CLASS SUMMARY
# =============================================================================

class_rows = []

for class_name, values in class_summary.items():

    class_rows.append(
        {
            "class": class_name,
            **values,
        }
    )

class_summary_df = pd.DataFrame(
    class_rows
)

class_summary_path = (
    OUTPUT_DIR
    /
    "baseline_vs_part27_class_summary.csv"
)

class_summary_df.to_csv(
    class_summary_path,
    index=False
)

print(
    "Saved:",
    class_summary_path
)


# =============================================================================
# SAVE THRESHOLD SUMMARY
# =============================================================================

threshold_rows = []

for threshold, values in threshold_summary.items():

    threshold_rows.append(
        {
            "threshold":
                float(threshold),
            **values,
        }
    )

threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_path = (
    OUTPUT_DIR
    /
    "baseline_vs_part27_threshold_summary.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False
)

print(
    "Saved:",
    threshold_path
)


# =============================================================================
# CHART 1
# =============================================================================

print("\n" + "=" * 78)
print("CREATING CHARTS")
print("=" * 78)

plt.figure(
    figsize=(8, 6)
)

plt.boxplot(
    [
        paired_df[
            "baseline_dice"
        ],
        paired_df[
            "part27_dice"
        ],
    ],
    tick_labels=[
        "Baseline",
        "Part 27",
    ]
)

plt.ylabel(
    "Mean Foreground Dice"
)

plt.title(
    "Baseline vs Part 27 Dice Distribution"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

chart_path = (
    OUTPUT_DIR
    /
    "baseline_vs_part27_dice_distribution.png"
)

plt.savefig(
    chart_path,
    dpi=200
)

plt.close()

print(
    "Saved:",
    chart_path
)


# =============================================================================
# CHART 2
# =============================================================================

plt.figure(
    figsize=(9, 6)
)

x = np.arange(
    len(CLASS_NAMES)
)

width = 0.35

baseline_values = [
    class_summary[
        name
    ]["baseline_mean_dice"]
    for name in CLASS_NAMES
]

part27_values = [
    class_summary[
        name
    ]["part27_mean_dice"]
    for name in CLASS_NAMES
]

plt.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline"
)

plt.bar(
    x + width / 2,
    part27_values,
    width,
    label="Part 27"
)

plt.xticks(
    x,
    CLASS_NAMES,
    rotation=15
)

plt.ylabel(
    "Dice"
)

plt.title(
    "Per-Class Dice: Baseline vs Part 27"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

chart_path = (
    OUTPUT_DIR
    /
    "paired_class_comparison.png"
)

plt.savefig(
    chart_path,
    dpi=200
)

plt.close()

print(
    "Saved:",
    chart_path
)


# =============================================================================
# CHART 3
# =============================================================================

sorted_difference_df = (
    paired_df
    .sort_values(
        "dice_change"
    )
)

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    np.arange(
        len(sorted_difference_df)
    ),
    sorted_difference_df[
        "dice_change"
    ]
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Test Case (sorted by Dice change)"
)

plt.ylabel(
    "Part 27 Dice - Baseline Dice"
)

plt.title(
    "Case-wise Dice Improvement"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

chart_path = (
    OUTPUT_DIR
    /
    "casewise_dice_difference.png"
)

plt.savefig(
    chart_path,
    dpi=200
)

plt.close()

print(
    "Saved:",
    chart_path
)


# =============================================================================
# CHART 4
# =============================================================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    paired_df[
        "baseline_dice"
    ],
    paired_df[
        "part27_dice"
    ],
    alpha=0.75
)

minimum = min(
    paired_df["baseline_dice"].min(),
    paired_df["part27_dice"].min()
)

maximum = max(
    paired_df["baseline_dice"].max(),
    paired_df["part27_dice"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Baseline Dice"
)

plt.ylabel(
    "Part 27 Dice"
)

plt.title(
    "Baseline vs Part 27 Case-wise Dice"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

chart_path = (
    OUTPUT_DIR
    /
    "baseline_vs_part27_scatter.png"
)

plt.savefig(
    chart_path,
    dpi=200
)

plt.close()

print(
    "Saved:",
    chart_path
)


# =============================================================================
# CHART 5 — THRESHOLD COMPARISON
# =============================================================================

plt.figure(
    figsize=(9, 6)
)

threshold_values = np.array(
    [
        float(x)
        for x in threshold_summary.keys()
    ]
)

baseline_counts = np.array(
    [
        threshold_summary[
            str(x)
        ]["baseline_cases"]
        for x in threshold_values
    ]
)

part27_counts = np.array(
    [
        threshold_summary[
            str(x)
        ]["part27_cases"]
        for x in threshold_values
    ]
)

plt.plot(
    threshold_values,
    baseline_counts,
    marker="o",
    label="Baseline"
)

plt.plot(
    threshold_values,
    part27_counts,
    marker="o",
    label="Part 27"
)

plt.xlabel(
    "Dice Threshold"
)

plt.ylabel(
    "Number of Test Cases"
)

plt.title(
    "Cases Meeting Dice Threshold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

chart_path = (
    OUTPUT_DIR
    /
    "dice_threshold_comparison.png"
)

plt.savefig(
    chart_path,
    dpi=200
)

plt.close()

print(
    "Saved:",
    chart_path
)


# =============================================================================
# FINAL JSON SUMMARY
# =============================================================================

summary = {
    "phase": "Phase 3 - Part 29",
    "experiment":
        "Paired Baseline vs Part 27 Test Comparison",

    "baseline":
        BASELINE_NAME,

    "part27":
        PART27_NAME,

    "test_cases":
        int(len(paired_df)),

    "overall": {
        "baseline_mean_dice":
            float(baseline_mean_dice),

        "part27_mean_dice":
            float(part27_mean_dice),

        "mean_dice_change":
            float(mean_dice_change),

        "baseline_median_dice":
            float(baseline_median_dice),

        "part27_median_dice":
            float(part27_median_dice),

        "median_dice_change":
            float(median_dice_change),

        "baseline_mean_iou":
            float(baseline_mean_iou),

        "part27_mean_iou":
            float(part27_mean_iou),

        "mean_iou_change":
            float(mean_iou_change),
    },

    "case_outcomes": {
        "improved":
            int(len(improved_cases)),

        "unchanged":
            int(len(unchanged_cases)),

        "worsened":
            int(len(worsened_cases)),
    },

    "class_dice":
        class_summary,

    "class_iou":
        iou_summary,

    "precision_recall":
        precision_recall_summary,

    "thresholds":
        threshold_summary,

    "casewise_dice_correlation":
        float(dice_correlation),

    "wilcoxon": {
        "statistic":
            None
            if statistic is None
            else float(statistic),

        "p_value":
            None
            if p_value is None
            else float(p_value),

        "interpretation":
            significance,
    },

    "effect_size": {
        "paired_cohens_d":
            float(cohens_d),
    },

    "ground_truth_used_for_localization":
        False,

    "training_performed":
        False,

    "model_weights_modified":
        False,
}


json_path = (
    OUTPUT_DIR
    /
    "phase3_part29_paired_comparison_summary.json"
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
    "Saved:",
    json_path
)


# =============================================================================
# TEXT REPORT
# =============================================================================

report_path = (
    OUTPUT_DIR
    /
    "phase3_part29_paired_comparison_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PHASE 3 - PART 29\n"
    )

    f.write(
        "PAIRED BASELINE VS PART 27 "
        "TEST COMPARISON\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Number of paired test cases: "
        f"{len(paired_df)}\n\n"
    )

    f.write(
        "OVERALL RESULTS\n"
    )

    f.write(
        f"Baseline mean Dice: "
        f"{baseline_mean_dice:.6f}\n"
    )

    f.write(
        f"Part 27 mean Dice: "
        f"{part27_mean_dice:.6f}\n"
    )

    f.write(
        f"Mean Dice change: "
        f"{mean_dice_change:+.6f}\n"
    )

    f.write(
        f"Baseline median Dice: "
        f"{baseline_median_dice:.6f}\n"
    )

    f.write(
        f"Part 27 median Dice: "
        f"{part27_median_dice:.6f}\n"
    )

    f.write(
        f"Median Dice change: "
        f"{median_dice_change:+.6f}\n\n"
    )

    f.write(
        "CASE-WISE OUTCOMES\n"
    )

    f.write(
        f"Improved: "
        f"{len(improved_cases)} "
        f"({len(improved_cases) / len(paired_df) * 100:.2f}%)\n"
    )

    f.write(
        f"Unchanged: "
        f"{len(unchanged_cases)} "
        f"({len(unchanged_cases) / len(paired_df) * 100:.2f}%)\n"
    )

    f.write(
        f"Worsened: "
        f"{len(worsened_cases)} "
        f"({len(worsened_cases) / len(paired_df) * 100:.2f}%)\n\n"
    )

    f.write(
        "PER-CLASS DICE\n"
    )

    for class_name, values in class_summary.items():

        f.write(
            f"{class_name}: "
            f"Baseline={values['baseline_mean_dice']:.6f}, "
            f"Part27={values['part27_mean_dice']:.6f}, "
            f"Change={values['mean_dice_change']:+.6f}\n"
        )

    f.write(
        "\nDICE THRESHOLDS\n"
    )

    for threshold, values in threshold_summary.items():

        f.write(
            f"Dice >= {threshold}: "
            f"Baseline={values['baseline_cases']}, "
            f"Part27={values['part27_cases']}, "
            f"Change={values['change']:+d}\n"
        )

    f.write(
        "\nSTATISTICAL ANALYSIS\n"
    )

    f.write(
        f"Wilcoxon statistic: "
        f"{statistic}\n"
    )

    f.write(
        f"Wilcoxon p-value: "
        f"{p_value}\n"
    )

    f.write(
        f"Interpretation: "
        f"{significance}\n"
    )

    f.write(
        f"Paired Cohen's d: "
        f"{cohens_d:.6f}\n"
    )

    f.write(
        "\nMETHODOLOGY\n"
    )

    f.write(
        "Baseline: fixed-center localization.\n"
    )

    f.write(
        "Part 27: hybrid image-only localization.\n"
    )

    f.write(
        "Ground truth was not used for localization.\n"
    )

    f.write(
        "Both models were evaluated on the same 71 test cases.\n"
    )

    f.write(
        "No training was performed in Part 29.\n"
    )


print(
    "Saved:",
    report_path
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n" + "=" * 78)
print("PART 29 COMPLETE")
print("=" * 78)

print(
    "Paired test cases:",
    len(paired_df)
)

print(
    "Baseline Mean Dice:",
    f"{baseline_mean_dice:.6f}"
)

print(
    "Part 27 Mean Dice:",
    f"{part27_mean_dice:.6f}"
)

print(
    "Mean Dice change:",
    f"{mean_dice_change:+.6f}"
)

print()

print(
    "Improved cases:",
    len(improved_cases)
)

print(
    "Unchanged cases:",
    len(unchanged_cases)
)

print(
    "Worsened cases:",
    len(worsened_cases)
)

print()

print(
    "Wilcoxon p-value:",
    p_value
)

print(
    "Paired Cohen's d:",
    f"{cohens_d:.6f}"
)

print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print("=" * 78)
print("PHASE 3 - PART 29 COMPLETE")
print("=" * 78)