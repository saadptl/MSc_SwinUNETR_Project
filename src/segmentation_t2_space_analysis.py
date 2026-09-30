"""
PHASE 3 - PART 18
T2 SPACE FAILURE VISUALIZATION & COMPARATIVE ANALYSIS

Purpose
-------
Detailed investigation of T2 SPACE spinal-canal segmentation
performance using the official results generated in Parts 11, 14,
16 and 17.

No model training is performed.
No model weights are modified.

Outputs
-------
- T2 SPACE case analysis
- Best/worst T2 SPACE cases
- Success/failure comparison
- Sequence comparison
- Failure-pattern analysis
- Canal-size analysis
- Correlation analysis
- Research charts
- JSON summary
- Research report
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# START TIME
# ============================================================

start_time = time.time()


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "t2_space_analysis"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


PART11_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "test_evaluation"
    / "test_case_results.csv"
)

PART14_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "spinal_canal_analysis"
    / "spinal_canal_case_analysis.csv"
)

PART16_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "class_presence_audit"
    / "class_presence_audit.csv"
)

PART17_MAIN_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "spinal_canal_failure_pattern"
    / "spinal_canal_failure_pattern_analysis.csv"
)

PART17_SEQUENCE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "spinal_canal_failure_pattern"
    / "spinal_canal_sequence_analysis.csv"
)

PART17_SIZE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "spinal_canal_failure_pattern"
    / "spinal_canal_size_group_analysis.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def save_dataframe(df, filename):
    path = OUTPUT_ROOT / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# ============================================================
# HEADER
# ============================================================

section("PHASE 3 - PART 18")
print("T2 SPACE FAILURE VISUALIZATION & COMPARATIVE ANALYSIS")
print("=" * 78)

print()
print("PROJECT ROOT")
print(PROJECT_ROOT)

print()
print("OUTPUT DIRECTORY")
print(OUTPUT_ROOT)


# ============================================================
# LOAD PART 11
# ============================================================

section("LOADING PART 11 RESULTS")

if not PART11_FILE.exists():
    raise FileNotFoundError(
        f"Part 11 result not found:\n{PART11_FILE}"
    )

part11 = pd.read_csv(PART11_FILE)

print(f"Part 11 cases: {len(part11)}")
print(f"Part 11 columns: {len(part11.columns)}")


# ============================================================
# LOAD PART 14
# ============================================================

section("LOADING PART 14 RESULTS")

if not PART14_FILE.exists():
    raise FileNotFoundError(
        f"Part 14 result not found:\n{PART14_FILE}"
    )

part14 = pd.read_csv(PART14_FILE)

print(f"Part 14 cases: {len(part14)}")
print(f"Part 14 columns: {len(part14.columns)}")


# ============================================================
# LOAD PART 17
# ============================================================

section("LOADING PART 17 RESULTS")

if not PART17_MAIN_FILE.exists():
    raise FileNotFoundError(
        f"Part 17 result not found:\n{PART17_MAIN_FILE}"
    )

part17 = pd.read_csv(PART17_MAIN_FILE)

print(f"Part 17 cases: {len(part17)}")
print(f"Part 17 columns: {len(part17.columns)}")


# ============================================================
# MERGE DATA
# ============================================================

section("MERGING OFFICIAL ANALYSIS RESULTS")

# Part 17 already contains most information required.
df = part17.copy()

print(f"Cases available: {len(df)}")


# ============================================================
# IDENTIFY T2 SPACE
# ============================================================

section("IDENTIFYING T2 SPACE CASES")

if "sequence" not in df.columns:
    raise ValueError(
        "The Part 17 CSV does not contain the 'sequence' column."
    )

t2_space = df[
    df["sequence"].astype(str).str.upper() == "T2_SPACE"
].copy()

t2_space = t2_space.reset_index(drop=True)

print(f"T2 SPACE cases: {len(t2_space)}")

if len(t2_space) == 0:
    raise RuntimeError("No T2 SPACE cases were detected.")


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "file",
    "sequence",
    "Spinal Canal_dice",
    "Spinal Canal_iou",
    "Spinal Canal_precision",
    "Spinal Canal_recall",
    "spinal_canal_voxels",
    "failure_pattern",
]

missing = [
    col for col in required_columns
    if col not in t2_space.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# BASIC T2 SPACE PERFORMANCE
# ============================================================

section("T2 SPACE OVERALL PERFORMANCE")

dice = t2_space["Spinal Canal_dice"].astype(float)

print(f"Cases              : {len(t2_space)}")
print(f"Mean Dice          : {dice.mean():.6f}")
print(f"Median Dice        : {dice.median():.6f}")
print(f"Std Dice           : {dice.std():.6f}")
print(f"Minimum Dice       : {dice.min():.6f}")
print(f"Maximum Dice       : {dice.max():.6f}")

failure_count = int((dice < 0.70).sum())
severe_count = int((dice < 0.50).sum())

print(
    f"Failure < 0.70     : "
    f"{failure_count} ({failure_count / len(t2_space) * 100:.2f}%)"
)

print(
    f"Severe < 0.50      : "
    f"{severe_count} ({severe_count / len(t2_space) * 100:.2f}%)"
)


# ============================================================
# SORTED CASE TABLE
# ============================================================

section("T2 SPACE CASE PERFORMANCE")

case_columns = [
    "file",
    "sequence",
    "Spinal Canal_dice",
    "Spinal Canal_iou",
    "Spinal Canal_precision",
    "Spinal Canal_recall",
    "spinal_canal_voxels",
    "failure_pattern",
]

t2_cases = (
    t2_space[case_columns]
    .sort_values("Spinal Canal_dice")
    .reset_index(drop=True)
)

print(t2_cases.to_string(index=False))

save_dataframe(
    t2_cases,
    "t2_space_case_analysis.csv"
)


# ============================================================
# WORST CASES
# ============================================================

section("WORST T2 SPACE CASES")

worst = (
    t2_space
    .sort_values("Spinal Canal_dice")
    .head(5)
    [case_columns]
)

print(worst.to_string(index=False))

save_dataframe(
    worst,
    "t2_space_worst_cases.csv"
)


# ============================================================
# BEST CASES
# ============================================================

section("BEST T2 SPACE CASES")

best = (
    t2_space
    .sort_values("Spinal Canal_dice", ascending=False)
    .head(5)
    [case_columns]
)

print(best.to_string(index=False))

save_dataframe(
    best,
    "t2_space_best_cases.csv"
)


# ============================================================
# SUCCESS VS FAILURE
# ============================================================

section("T2 SPACE SUCCESS VS FAILURE")

t2_space["performance_group"] = np.where(
    t2_space["Spinal Canal_dice"] >= 0.70,
    "Acceptable",
    "Failure"
)

group_rows = []

for group_name, group in t2_space.groupby("performance_group"):

    group_rows.append({
        "group": group_name,
        "cases": len(group),
        "mean_dice": group["Spinal Canal_dice"].mean(),
        "median_dice": group["Spinal Canal_dice"].median(),
        "mean_precision": group["Spinal Canal_precision"].mean(),
        "mean_recall": group["Spinal Canal_recall"].mean(),
        "mean_iou": group["Spinal Canal_iou"].mean(),
        "mean_ground_truth_voxels":
            group["spinal_canal_voxels"].mean(),
        "median_ground_truth_voxels":
            group["spinal_canal_voxels"].median(),
    })

t2_groups = pd.DataFrame(group_rows)

print(t2_groups.to_string(index=False))

save_dataframe(
    t2_groups,
    "t2_space_success_vs_failure.csv"
)


# ============================================================
# FAILURE PATTERN
# ============================================================

section("T2 SPACE FAILURE PATTERNS")

pattern_counts = (
    t2_space["failure_pattern"]
    .value_counts()
    .reset_index()
)

pattern_counts.columns = [
    "failure_pattern",
    "cases"
]

pattern_counts["percentage"] = (
    pattern_counts["cases"]
    / len(t2_space)
    * 100
)

print(pattern_counts.to_string(index=False))

save_dataframe(
    pattern_counts,
    "t2_space_failure_patterns.csv"
)


# ============================================================
# SIZE ANALYSIS
# ============================================================

section("T2 SPACE CANAL SIZE ANALYSIS")

# Use quantiles within the T2 SPACE subset.
q1 = t2_space["spinal_canal_voxels"].quantile(1 / 3)
q2 = t2_space["spinal_canal_voxels"].quantile(2 / 3)

def size_group(value):
    if value <= q1:
        return "Small"
    elif value <= q2:
        return "Medium"
    else:
        return "Large"


t2_space["size_group"] = (
    t2_space["spinal_canal_voxels"]
    .apply(size_group)
)

size_rows = []

for group_name, group in (
    t2_space.groupby("size_group", sort=False)
):

    size_rows.append({
        "size_group": group_name,
        "cases": len(group),
        "mean_canal_voxels":
            group["spinal_canal_voxels"].mean(),
        "mean_dice":
            group["Spinal Canal_dice"].mean(),
        "median_dice":
            group["Spinal Canal_dice"].median(),
        "failure_cases":
            int((group["Spinal Canal_dice"] < 0.70).sum()),
        "failure_percentage":
            float(
                (group["Spinal Canal_dice"] < 0.70).mean()
                * 100
            ),
    })

size_analysis = pd.DataFrame(size_rows)

print(size_analysis.to_string(index=False))

save_dataframe(
    size_analysis,
    "t2_space_size_analysis.csv"
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

section("T2 SPACE CORRELATION ANALYSIS")

correlations = []

numeric_targets = [
    "spinal_canal_voxels",
    "Spinal Canal_precision",
    "Spinal Canal_recall",
    "Spinal Canal_iou",
]

for column in numeric_targets:

    value = t2_space[
        "Spinal Canal_dice"
    ].corr(
        t2_space[column]
    )

    correlations.append({
        "variable": column,
        "correlation_with_dice": value
    })

correlation_df = pd.DataFrame(correlations)

print(correlation_df.to_string(index=False))

save_dataframe(
    correlation_df,
    "t2_space_correlations.csv"
)


# ============================================================
# COMPARISON WITH OTHER SEQUENCES
# ============================================================

section("SEQUENCE COMPARISON")

sequence_rows = []

for sequence_name, group in df.groupby("sequence"):

    sequence_rows.append({
        "sequence": sequence_name,
        "cases": len(group),
        "mean_dice": group["Spinal Canal_dice"].mean(),
        "median_dice": group["Spinal Canal_dice"].median(),
        "failure_cases": int(
            (group["Spinal Canal_dice"] < 0.70).sum()
        ),
        "failure_percentage":
            float(
                (group["Spinal Canal_dice"] < 0.70).mean()
                * 100
            ),
        "severe_cases": int(
            (group["Spinal Canal_dice"] < 0.50).sum()
        ),
    })

sequence_comparison = (
    pd.DataFrame(sequence_rows)
    .sort_values("mean_dice", ascending=False)
)

print(sequence_comparison.to_string(index=False))

save_dataframe(
    sequence_comparison,
    "sequence_comparison.csv"
)


# ============================================================
# T2 SPACE GAP
# ============================================================

section("T2 SPACE PERFORMANCE GAP")

other_sequences = df[
    df["sequence"].astype(str).str.upper()
    != "T2_SPACE"
]

other_mean = (
    other_sequences["Spinal Canal_dice"]
    .mean()
)

t2_space_mean = dice.mean()

performance_gap = other_mean - t2_space_mean

print(f"T2 SPACE mean Dice : {t2_space_mean:.6f}")
print(f"T1/T2 mean Dice    : {other_mean:.6f}")
print(f"Performance gap    : {performance_gap:.6f}")


# ============================================================
# CHART 1
# ============================================================

section("CREATING T2 SPACE DICE DISTRIBUTION")

plt.figure(figsize=(9, 6))

plt.hist(
    dice,
    bins=8,
    edgecolor="black"
)

plt.axvline(
    0.70,
    linestyle="--",
    linewidth=2,
    label="Failure threshold = 0.70"
)

plt.xlabel("Spinal Canal Dice")
plt.ylabel("Number of Cases")
plt.title("T2 SPACE Spinal Canal Dice Distribution")
plt.legend()
plt.grid(alpha=0.25)

path = OUTPUT_ROOT / "t2_space_dice_distribution.png"
plt.tight_layout()
plt.savefig(path, dpi=300)
plt.close()

print(f"Saved: {path}")


# ============================================================
# CHART 2
# ============================================================

section("CREATING SEQUENCE COMPARISON CHART")

plt.figure(figsize=(9, 6))

plt.bar(
    sequence_comparison["sequence"],
    sequence_comparison["mean_dice"]
)

plt.axhline(
    0.70,
    linestyle="--",
    linewidth=2,
    label="Failure threshold = 0.70"
)

plt.xlabel("MRI Sequence")
plt.ylabel("Mean Spinal Canal Dice")
plt.title("Spinal Canal Performance by MRI Sequence")
plt.legend()
plt.grid(axis="y", alpha=0.25)

path = OUTPUT_ROOT / "spinal_canal_sequence_comparison.png"
plt.tight_layout()
plt.savefig(path, dpi=300)
plt.close()

print(f"Saved: {path}")


# ============================================================
# CHART 3
# ============================================================

section("CREATING T2 SPACE CASE PERFORMANCE CHART")

plot_cases = (
    t2_space
    .sort_values("Spinal Canal_dice")
)

plt.figure(figsize=(11, 6))

plt.bar(
    plot_cases["file"],
    plot_cases["Spinal Canal_dice"]
)

plt.axhline(
    0.70,
    linestyle="--",
    linewidth=2,
    label="Acceptable threshold = 0.70"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.xlabel("T2 SPACE Case")
plt.ylabel("Spinal Canal Dice")
plt.title("T2 SPACE Case-wise Spinal Canal Performance")
plt.legend()
plt.grid(axis="y", alpha=0.25)

path = OUTPUT_ROOT / "t2_space_case_performance.png"
plt.tight_layout()
plt.savefig(path, dpi=300)
plt.close()

print(f"Saved: {path}")


# ============================================================
# CHART 4
# ============================================================

section("CREATING SIZE VS DICE CHART")

plt.figure(figsize=(9, 6))

plt.scatter(
    t2_space["spinal_canal_voxels"],
    t2_space["Spinal Canal_dice"],
    s=70
)

plt.axhline(
    0.70,
    linestyle="--",
    linewidth=2,
    label="Dice = 0.70"
)

plt.xlabel("Ground-truth Spinal Canal Voxels")
plt.ylabel("Spinal Canal Dice")
plt.title("T2 SPACE Canal Size vs Segmentation Dice")
plt.legend()
plt.grid(alpha=0.25)

path = OUTPUT_ROOT / "t2_space_size_vs_dice.png"
plt.tight_layout()
plt.savefig(path, dpi=300)
plt.close()

print(f"Saved: {path}")


# ============================================================
# CHART 5
# ============================================================

section("CREATING PRECISION-RECALL CHART")

plt.figure(figsize=(9, 6))

plt.scatter(
    t2_space["Spinal Canal_recall"],
    t2_space["Spinal Canal_precision"],
    s=80
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("T2 SPACE Spinal Canal Precision vs Recall")
plt.grid(alpha=0.25)

path = OUTPUT_ROOT / "t2_space_precision_vs_recall.png"
plt.tight_layout()
plt.savefig(path, dpi=300)
plt.close()

print(f"Saved: {path}")


# ============================================================
# CHART 6
# ============================================================

section("CREATING FAILURE PATTERN CHART")

plt.figure(figsize=(10, 6))

plt.bar(
    pattern_counts["failure_pattern"],
    pattern_counts["cases"]
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.xlabel("Failure Pattern")
plt.ylabel("Number of Cases")
plt.title("T2 SPACE Failure Pattern Distribution")
plt.grid(axis="y", alpha=0.25)

path = OUTPUT_ROOT / "t2_space_failure_patterns.png"
plt.tight_layout()
plt.savefig(path, dpi=300)
plt.close()

print(f"Saved: {path}")


# ============================================================
# CHART 7
# ============================================================

section("CREATING SUCCESS VS FAILURE CHART")

plt.figure(figsize=(8, 6))

plt.bar(
    t2_groups["group"],
    t2_groups["mean_dice"]
)

plt.axhline(
    0.70,
    linestyle="--",
    linewidth=2,
    label="Dice = 0.70"
)

plt.xlabel("Performance Group")
plt.ylabel("Mean Spinal Canal Dice")
plt.title("T2 SPACE: Successful vs Failed Cases")
plt.legend()
plt.grid(axis="y", alpha=0.25)

path = OUTPUT_ROOT / "t2_space_success_vs_failure.png"
plt.tight_layout()
plt.savefig(path, dpi=300)
plt.close()

print(f"Saved: {path}")


# ============================================================
# JSON SUMMARY
# ============================================================

section("CREATING JSON SUMMARY")

summary = {
    "phase": "Phase 3 - Part 18",
    "analysis": "T2 SPACE Failure Visualization and Comparative Analysis",

    "t2_space_cases": int(len(t2_space)),

    "mean_dice": float(dice.mean()),
    "median_dice": float(dice.median()),
    "std_dice": float(dice.std()),
    "minimum_dice": float(dice.min()),
    "maximum_dice": float(dice.max()),

    "failure_threshold": 0.70,
    "failure_cases": failure_count,
    "failure_percentage":
        float(failure_count / len(t2_space) * 100),

    "severe_threshold": 0.50,
    "severe_cases": severe_count,
    "severe_percentage":
        float(severe_count / len(t2_space) * 100),

    "other_sequences_mean_dice":
        float(other_mean),

    "performance_gap":
        float(performance_gap),

    "worst_case":
        str(
            t2_space
            .sort_values("Spinal Canal_dice")
            .iloc[0]["file"]
        ),

    "worst_case_dice":
        float(
            t2_space
            .sort_values("Spinal Canal_dice")
            .iloc[0]["Spinal Canal_dice"]
        ),

    "best_case":
        str(
            t2_space
            .sort_values(
                "Spinal Canal_dice",
                ascending=False
            )
            .iloc[0]["file"]
        ),

    "best_case_dice":
        float(
            t2_space
            .sort_values(
                "Spinal Canal_dice",
                ascending=False
            )
            .iloc[0]["Spinal Canal_dice"]
        ),

    "size_correlation":
        float(
            t2_space[
                "Spinal Canal_dice"
            ].corr(
                t2_space["spinal_canal_voxels"]
            )
        ),

    "performance_interpretation":
        "T2 SPACE shows substantially lower spinal canal "
        "segmentation performance than T1 and conventional T2 "
        "in the evaluated test set."
}

json_path = (
    OUTPUT_ROOT
    / "t2_space_analysis_summary.json"
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

print(f"Saved: {json_path}")


# ============================================================
# RESEARCH REPORT
# ============================================================

section("CREATING RESEARCH REPORT")

report_path = (
    OUTPUT_ROOT
    / "phase3_part18_t2_space_analysis_report.txt"
)

worst_case = (
    t2_space
    .sort_values("Spinal Canal_dice")
    .iloc[0]
)

best_case = (
    t2_space
    .sort_values(
        "Spinal Canal_dice",
        ascending=False
    )
    .iloc[0]
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 78 + "\n")
    f.write("PHASE 3 - PART 18\n")
    f.write("T2 SPACE FAILURE VISUALIZATION & COMPARATIVE ANALYSIS\n")
    f.write("=" * 78 + "\n\n")

    f.write("1. OBJECTIVE\n")
    f.write("-" * 78 + "\n")

    f.write(
        "This analysis investigates the poor spinal-canal "
        "segmentation performance observed for T2 SPACE MRI "
        "sequences in the test dataset.\n\n"
    )

    f.write("2. T2 SPACE PERFORMANCE\n")
    f.write("-" * 78 + "\n")

    f.write(
        f"Cases: {len(t2_space)}\n"
        f"Mean Dice: {dice.mean():.6f}\n"
        f"Median Dice: {dice.median():.6f}\n"
        f"Standard deviation: {dice.std():.6f}\n"
        f"Minimum Dice: {dice.min():.6f}\n"
        f"Maximum Dice: {dice.max():.6f}\n"
        f"Failure cases (<0.70): {failure_count}\n"
        f"Severe cases (<0.50): {severe_count}\n\n"
    )

    f.write("3. COMPARISON WITH OTHER SEQUENCES\n")
    f.write("-" * 78 + "\n")

    f.write(
        f"T2 SPACE mean Dice: {t2_space_mean:.6f}\n"
        f"T1/T2 mean Dice: {other_mean:.6f}\n"
        f"Performance gap: {performance_gap:.6f}\n\n"
    )

    f.write("4. WORST T2 SPACE CASE\n")
    f.write("-" * 78 + "\n")

    f.write(
        f"Case: {worst_case['file']}\n"
        f"Dice: {worst_case['Spinal Canal_dice']:.6f}\n"
        f"IoU: {worst_case['Spinal Canal_iou']:.6f}\n"
        f"Precision: {worst_case['Spinal Canal_precision']:.6f}\n"
        f"Recall: {worst_case['Spinal Canal_recall']:.6f}\n"
        f"Ground-truth voxels: "
        f"{int(worst_case['spinal_canal_voxels'])}\n"
        f"Failure pattern: {worst_case['failure_pattern']}\n\n"
    )

    f.write("5. BEST T2 SPACE CASE\n")
    f.write("-" * 78 + "\n")

    f.write(
        f"Case: {best_case['file']}\n"
        f"Dice: {best_case['Spinal Canal_dice']:.6f}\n"
        f"IoU: {best_case['Spinal Canal_iou']:.6f}\n"
        f"Precision: {best_case['Spinal Canal_precision']:.6f}\n"
        f"Recall: {best_case['Spinal Canal_recall']:.6f}\n\n"
    )

    f.write("6. FAILURE PATTERNS\n")
    f.write("-" * 78 + "\n")

    for _, row in pattern_counts.iterrows():

        f.write(
            f"{row['failure_pattern']}: "
            f"{int(row['cases'])} cases "
            f"({row['percentage']:.2f}%)\n"
        )

    f.write("\n")

    f.write("7. SIZE ASSOCIATION\n")
    f.write("-" * 78 + "\n")

    size_corr = (
        t2_space[
            "Spinal Canal_dice"
        ].corr(
            t2_space["spinal_canal_voxels"]
        )
    )

    f.write(
        f"Correlation between ground-truth canal size "
        f"and Dice: {size_corr:.6f}\n\n"
    )

    f.write("8. RESEARCH INTERPRETATION\n")
    f.write("-" * 78 + "\n")

    f.write(
        "The T2 SPACE subset demonstrates substantially lower "
        "spinal-canal segmentation performance compared with T1 "
        "and conventional T2 sequences. The result suggests that "
        "MRI sequence characteristics may influence the ability "
        "of the current Swin-UNETR model to localize and segment "
        "the spinal canal.\n\n"
    )

    f.write(
        "The analysis does not establish causality. The observed "
        "sequence-related difference should therefore be treated "
        "as an empirical finding of this test dataset and should "
        "be investigated further through qualitative examination "
        "and, if required, future model improvements.\n\n"
    )

    f.write("9. RECOMMENDED NEXT STEP\n")
    f.write("-" * 78 + "\n")

    f.write(
        "Perform detailed qualitative visualization of representative "
        "T2 SPACE failures and successful cases, comparing MRI "
        "appearance, ground-truth segmentation and predicted "
        "segmentation. This should be completed before changing "
        "the training strategy.\n\n"
    )

    f.write("=" * 78 + "\n")
    f.write("END OF PART 18 REPORT\n")
    f.write("=" * 78 + "\n")

print(f"Saved: {report_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

elapsed = time.time() - start_time

section("PART 18 COMPLETE")

print(f"T2 SPACE cases analyzed : {len(t2_space)}")
print(f"Mean T2 SPACE Dice     : {dice.mean():.6f}")
print(f"Failure cases          : {failure_count}")
print(f"Severe cases           : {severe_count}")
print(f"Worst case             : {worst_case['file']}")
print(
    f"Worst Dice             : "
    f"{worst_case['Spinal Canal_dice']:.6f}"
)
print(f"Execution time         : {elapsed / 60:.2f} minutes")

print()
print("=" * 78)
print("OUTPUT DIRECTORY")
print("=" * 78)
print(OUTPUT_ROOT)

print()
print("=" * 78)
print("PHASE 3 - PART 18 COMPLETE")
print("=" * 78)