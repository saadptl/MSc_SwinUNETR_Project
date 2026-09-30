"""
PHASE 3 - PART 23
CROP RETENTION VS SEGMENTATION PERFORMANCE ANALYSIS

Purpose
-------
Quantitatively analyze whether anatomical retention inside the
96 x 96 x 96 model crop is associated with segmentation performance.

Uses:
    Part 22 anatomy alignment results
    Part 19 T2 SPACE visual inspection results
    Part 11 official test metrics

No model retraining is performed.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "segmentation"

PART22_DIR = OUTPUT_ROOT / "t2_space_anatomy_alignment"
PART19_DIR = OUTPUT_ROOT / "t2_space_visual_inspection"
PART11_DIR = OUTPUT_ROOT / "test_evaluation"

OUTPUT_DIR = OUTPUT_ROOT / "crop_retention_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FILES
# ============================================================================

PART22_CASES = PART22_DIR / "t2_space_anatomy_alignment_case_analysis.csv"
PART22_GROUP = PART22_DIR / "t2_space_anatomy_alignment_group_summary.csv"
PART22_CORR = PART22_DIR / "t2_space_anatomy_alignment_correlations.csv"

PART19_RESULTS = PART19_DIR / "t2_space_visual_inspection_summary.csv"

PART11_RESULTS = PART11_DIR / "test_case_results.csv"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def classify_retention(value):
    """
    Retention categories used for exploratory analysis.

    These are deliberately descriptive rather than medical thresholds.
    """
    if pd.isna(value):
        return "Unknown"

    if value < 0.05:
        return "Very Low"

    if value < 0.10:
        return "Low"

    if value < 0.15:
        return "Moderate"

    return "Higher"


def classify_performance(dice):
    if pd.isna(dice):
        return "Unknown"

    if dice < 0.50:
        return "Severe Failure"

    if dice < 0.70:
        return "Low Performance"

    if dice < 0.80:
        return "Moderate Performance"

    return "Good Performance"


def pearson(x, y):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(valid) < 3:
        return np.nan

    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan

    return float(valid["x"].corr(valid["y"], method="pearson"))


# ============================================================================
# START
# ============================================================================

print_header("PHASE 3 - PART 23")
print("CROP RETENTION VS SEGMENTATION PERFORMANCE ANALYSIS")
print("=" * 78)

print()
print("PROJECT ROOT")
print(PROJECT_ROOT)

print()
print("PART 22 DIRECTORY")
print(PART22_DIR)

print()
print("PART 19 DIRECTORY")
print(PART19_DIR)

print()
print("PART 11 DIRECTORY")
print(PART11_DIR)

print()
print("OUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================================
# LOAD PART 22
# ============================================================================

print_header("LOADING PART 22 ANATOMY ALIGNMENT RESULTS")

if not PART22_CASES.exists():
    raise FileNotFoundError(
        f"Part 22 case analysis not found:\n{PART22_CASES}"
    )

df = pd.read_csv(PART22_CASES)

print(f"Part 22 rows: {len(df)}")
print(f"Part 22 columns: {len(df.columns)}")


# ============================================================================
# LOAD PART 19 IF AVAILABLE
# ============================================================================

print_header("LOADING PART 19 VISUAL INSPECTION RESULTS")

if PART19_RESULTS.exists():
    df19 = pd.read_csv(PART19_RESULTS)

    print(f"Part 19 rows: {len(df19)}")
    print("Part 19 results loaded.")
else:
    df19 = None
    print("Part 19 summary not found.")
    print("Continuing using Part 22 results.")


# ============================================================================
# LOAD PART 11
# ============================================================================

print_header("LOADING PART 11 OFFICIAL TEST RESULTS")

if PART11_RESULTS.exists():
    df11 = pd.read_csv(PART11_RESULTS)

    print(f"Part 11 rows: {len(df11)}")

    if "file" in df11.columns:
        keep_cols = [
            "file",
            "mean_foreground_dice",
            "mean_foreground_iou",
        ]

        keep_cols = [
            c for c in keep_cols
            if c in df11.columns
        ]

        df11_small = df11[keep_cols].copy()

        df = df.merge(
            df11_small,
            on="file",
            how="left",
            suffixes=("", "_part11"),
        )

else:
    print("Part 11 results not found.")


# ============================================================================
# NUMERIC CONVERSION
# ============================================================================

numeric_columns = [
    "model_dice",
    "foreground_retention",
    "vertebrae_retention",
    "canal_retention",
    "disc_retention",
    "foreground_alignment_distance",
    "vertebrae_alignment_distance",
    "canal_alignment_distance",
    "disc_alignment_distance",
    "overall_alignment_score",
    "intensity_mean",
    "intensity_std",
]

df = safe_numeric(df, numeric_columns)


# ============================================================================
# IDENTIFY PERFORMANCE COLUMN
# ============================================================================

if "model_dice" not in df.columns:
    raise RuntimeError(
        "The Part 22 CSV does not contain 'model_dice'."
    )

df["performance_category"] = df["model_dice"].apply(
    classify_performance
)


# ============================================================================
# RETENTION CATEGORIES
# ============================================================================

df["foreground_retention_category"] = df[
    "foreground_retention"
].apply(classify_retention)

df["vertebrae_retention_category"] = df[
    "vertebrae_retention"
].apply(classify_retention)

df["canal_retention_category"] = df[
    "canal_retention"
].apply(classify_retention)

df["disc_retention_category"] = df[
    "disc_retention"
].apply(classify_retention)


# ============================================================================
# PRINT BASIC RESULTS
# ============================================================================

print_header("DATASET SUMMARY")

print(f"Cases analyzed: {len(df)}")
print(f"Mean Dice: {df['model_dice'].mean():.6f}")
print(f"Median Dice: {df['model_dice'].median():.6f}")
print(f"Minimum Dice: {df['model_dice'].min():.6f}")
print(f"Maximum Dice: {df['model_dice'].max():.6f}")


# ============================================================================
# RETENTION SUMMARY
# ============================================================================

print_header("ANATOMICAL RETENTION SUMMARY")

retention_columns = [
    "foreground_retention",
    "vertebrae_retention",
    "canal_retention",
    "disc_retention",
]

retention_summary = []

for col in retention_columns:

    values = pd.to_numeric(df[col], errors="coerce")

    retention_summary.append({
        "feature": col,
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "minimum": values.min(),
        "maximum": values.max(),
    })

    print()
    print(col)
    print(f"  Mean   : {values.mean():.6f}")
    print(f"  Median : {values.median():.6f}")
    print(f"  Std    : {values.std():.6f}")
    print(f"  Min    : {values.min():.6f}")
    print(f"  Max    : {values.max():.6f}")

retention_summary_df = pd.DataFrame(retention_summary)

retention_summary_df.to_csv(
    OUTPUT_DIR / "retention_summary.csv",
    index=False,
)


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print_header("RETENTION CORRELATION WITH MODEL DICE")

correlation_rows = []

for col in retention_columns:

    r = pearson(df[col], df["model_dice"])

    correlation_rows.append({
        "feature": col,
        "pearson_correlation_with_dice": r,
        "valid_cases": int(
            pd.DataFrame({
                "x": df[col],
                "y": df["model_dice"]
            }).dropna().shape[0]
        ),
    })

    print(
        f"{col:30s}: "
        f"{r:.6f}"
    )

correlation_df = pd.DataFrame(correlation_rows)

correlation_df = correlation_df.sort_values(
    "pearson_correlation_with_dice",
    ascending=False,
)

correlation_df.to_csv(
    OUTPUT_DIR / "retention_dice_correlations.csv",
    index=False,
)


# ============================================================================
# FAILURE VS SUCCESS GROUPS
# ============================================================================

print_header("FAILURE VS SUCCESSFUL RETENTION COMPARISON")

df["success_group"] = np.where(
    df["model_dice"] >= 0.70,
    "Successful",
    "Failure",
)

group_rows = []

for group_name, group in df.groupby("success_group"):

    row = {
        "group": group_name,
        "cases": len(group),
        "mean_dice": group["model_dice"].mean(),
    }

    for col in retention_columns:
        row[f"{col}_mean"] = group[col].mean()
        row[f"{col}_median"] = group[col].median()

    group_rows.append(row)

    print()
    print(group_name)
    print(f"  Cases: {len(group)}")
    print(f"  Mean Dice: {group['model_dice'].mean():.6f}")

    for col in retention_columns:
        print(
            f"  {col:30s}: "
            f"{group[col].mean():.6f}"
        )

group_summary_df = pd.DataFrame(group_rows)

group_summary_df.to_csv(
    OUTPUT_DIR / "failure_vs_success_retention.csv",
    index=False,
)


# ============================================================================
# RETENTION THRESHOLD ANALYSIS
# ============================================================================

print_header("RETENTION THRESHOLD ANALYSIS")

thresholds = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
]

threshold_rows = []

for col in retention_columns:

    for threshold in thresholds:

        low = df[df[col] < threshold]
        high = df[df[col] >= threshold]

        low_failure_rate = (
            float((low["model_dice"] < 0.70).mean())
            if len(low) > 0 else np.nan
        )

        high_failure_rate = (
            float((high["model_dice"] < 0.70).mean())
            if len(high) > 0 else np.nan
        )

        threshold_rows.append({
            "feature": col,
            "threshold": threshold,
            "low_retention_cases": len(low),
            "high_retention_cases": len(high),
            "low_retention_mean_dice": low["model_dice"].mean(),
            "high_retention_mean_dice": high["model_dice"].mean(),
            "low_retention_failure_rate": low_failure_rate,
            "high_retention_failure_rate": high_failure_rate,
        })

        print()
        print(
            f"{col} < {threshold:.2f}"
        )
        print(
            f"  Low-retention cases : {len(low)}"
        )
        print(
            f"  High-retention cases: {len(high)}"
        )
        print(
            f"  Low mean Dice       : "
            f"{low['model_dice'].mean():.6f}"
        )
        print(
            f"  High mean Dice      : "
            f"{high['model_dice'].mean():.6f}"
        )

threshold_df = pd.DataFrame(threshold_rows)

threshold_df.to_csv(
    OUTPUT_DIR / "retention_threshold_analysis.csv",
    index=False,
)


# ============================================================================
# LOWEST RETENTION CASES
# ============================================================================

print_header("LOWEST RETENTION CASES")

for col in retention_columns:

    print()
    print(col)

    temp = df[
        [
            "file",
            "model_dice",
            col,
        ]
    ].sort_values(col).head(5)

    print(temp.to_string(index=False))


# ============================================================================
# COMBINED RETENTION SCORE
# ============================================================================

print_header("COMBINED ANATOMICAL RETENTION SCORE")

available_retention = [
    c for c in retention_columns
    if c in df.columns
]

df["combined_retention_score"] = df[
    available_retention
].mean(axis=1)

combined_corr = pearson(
    df["combined_retention_score"],
    df["model_dice"],
)

print(
    f"Combined retention score correlation with Dice: "
    f"{combined_corr:.6f}"
)

combined_summary = pd.DataFrame([
    {
        "metric": "combined_retention_score",
        "mean": df["combined_retention_score"].mean(),
        "median": df["combined_retention_score"].median(),
        "std": df["combined_retention_score"].std(),
        "minimum": df["combined_retention_score"].min(),
        "maximum": df["combined_retention_score"].max(),
        "dice_correlation": combined_corr,
    }
])

combined_summary.to_csv(
    OUTPUT_DIR / "combined_retention_summary.csv",
    index=False,
)


# ============================================================================
# IDENTIFY BEST RETENTION PREDICTOR
# ============================================================================

print_header("BEST RETENTION PREDICTOR")

best_row = correlation_df.iloc[0]

print(
    f"Best predictor : {best_row['feature']}"
)

print(
    f"Correlation    : "
    f"{best_row['pearson_correlation_with_dice']:.6f}"
)

# ============================================================================
# CREATE MASTER ANALYSIS TABLE
# ============================================================================

master_columns = [
    "file",
    "model_dice",
    "performance_category",
    "success_group",
    "foreground_retention",
    "vertebrae_retention",
    "canal_retention",
    "disc_retention",
    "combined_retention_score",
    "foreground_alignment_distance",
    "vertebrae_alignment_distance",
    "canal_alignment_distance",
    "disc_alignment_distance",
    "overall_alignment_score",
]

master_columns = [
    c for c in master_columns
    if c in df.columns
]

master_df = df[master_columns].copy()

master_df = master_df.sort_values(
    "model_dice",
    ascending=True,
)

master_df.to_csv(
    OUTPUT_DIR / "crop_retention_master_analysis.csv",
    index=False,
)


# ============================================================================
# CHART 1 — FOREGROUND RETENTION VS DICE
# ============================================================================

print_header("CREATING RETENTION ANALYSIS CHARTS")

plt.figure(figsize=(8, 6))

plt.scatter(
    df["foreground_retention"],
    df["model_dice"],
    s=70,
)

plt.xlabel("Foreground Retention")
plt.ylabel("Model Dice")
plt.title("Foreground Retention vs Segmentation Dice")
plt.grid(alpha=0.25)

plt.tight_layout()

path = OUTPUT_DIR / "foreground_retention_vs_dice.png"
plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path}")


# ============================================================================
# CHART 2 — VERTEBRAE RETENTION VS DICE
# ============================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["vertebrae_retention"],
    df["model_dice"],
    s=70,
)

plt.xlabel("Vertebrae Retention")
plt.ylabel("Model Dice")
plt.title("Vertebrae Retention vs Segmentation Dice")
plt.grid(alpha=0.25)

plt.tight_layout()

path = OUTPUT_DIR / "vertebrae_retention_vs_dice.png"
plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path}")


# ============================================================================
# CHART 3 — CANAL RETENTION VS DICE
# ============================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["canal_retention"],
    df["model_dice"],
    s=70,
)

plt.xlabel("Spinal Canal Retention")
plt.ylabel("Model Dice")
plt.title("Spinal Canal Retention vs Segmentation Dice")
plt.grid(alpha=0.25)

plt.tight_layout()

path = OUTPUT_DIR / "canal_retention_vs_dice.png"
plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path}")


# ============================================================================
# CHART 4 — DISC RETENTION VS DICE
# ============================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["disc_retention"],
    df["model_dice"],
    s=70,
)

plt.xlabel("Disc Retention")
plt.ylabel("Model Dice")
plt.title("Intervertebral Disc Retention vs Segmentation Dice")
plt.grid(alpha=0.25)

plt.tight_layout()

path = OUTPUT_DIR / "disc_retention_vs_dice.png"
plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path}")


# ============================================================================
# CHART 5 — COMBINED RETENTION VS DICE
# ============================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["combined_retention_score"],
    df["model_dice"],
    s=80,
)

plt.xlabel("Combined Anatomical Retention Score")
plt.ylabel("Model Dice")
plt.title("Combined Anatomical Retention vs Segmentation Dice")
plt.grid(alpha=0.25)

plt.tight_layout()

path = OUTPUT_DIR / "combined_retention_vs_dice.png"
plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path}")


# ============================================================================
# CHART 6 — FAILURE VS SUCCESS
# ============================================================================

plt.figure(figsize=(8, 6))

groups = [
    df[df["success_group"] == "Failure"]["combined_retention_score"],
    df[df["success_group"] == "Successful"]["combined_retention_score"],
]

plt.boxplot(
    groups,
    tick_labels=["Failure", "Successful"],
)

plt.ylabel("Combined Anatomical Retention")
plt.title("Anatomical Retention: Failure vs Successful Cases")
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()

path = OUTPUT_DIR / "failure_vs_success_retention_boxplot.png"
plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path}")


# ============================================================================
# CHART 7 — ALL RETENTION FEATURES
# ============================================================================

plot_data = df[
    [
        "foreground_retention",
        "vertebrae_retention",
        "canal_retention",
        "disc_retention",
    ]
].copy()

plt.figure(figsize=(10, 6))

plt.boxplot(
    [
        plot_data["foreground_retention"].dropna(),
        plot_data["vertebrae_retention"].dropna(),
        plot_data["canal_retention"].dropna(),
        plot_data["disc_retention"].dropna(),
    ],
    tick_labels=[
        "Foreground",
        "Vertebrae",
        "Canal",
        "Disc",
    ],
)

plt.ylabel("Retention")
plt.title("Anatomical Retention Distribution")
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()

path = OUTPUT_DIR / "anatomical_retention_distribution.png"
plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path}")


# ============================================================================
# FINAL JSON SUMMARY
# ============================================================================

print_header("CREATING FINAL SUMMARY")

failure_cases = df[
    df["model_dice"] < 0.70
]["file"].tolist()

successful_cases = df[
    df["model_dice"] >= 0.70
]["file"].tolist()

summary = {
    "part": "Phase 3 - Part 23",
    "cases_analyzed": int(len(df)),
    "failure_cases": int(len(failure_cases)),
    "successful_cases": int(len(successful_cases)),
    "mean_dice": float(df["model_dice"].mean()),
    "median_dice": float(df["model_dice"].median()),
    "minimum_dice": float(df["model_dice"].min()),
    "maximum_dice": float(df["model_dice"].max()),
    "retention_dice_correlations": {
        row["feature"]: (
            None
            if pd.isna(row["pearson_correlation_with_dice"])
            else float(row["pearson_correlation_with_dice"])
        )
        for _, row in correlation_df.iterrows()
    },
    "combined_retention_correlation": (
        None
        if pd.isna(combined_corr)
        else float(combined_corr)
    ),
    "best_retention_predictor": str(
        best_row["feature"]
    ),
    "best_retention_predictor_correlation": float(
        best_row["pearson_correlation_with_dice"]
    ),
    "failure_case_names": failure_cases,
    "successful_case_names": successful_cases,
}


summary_path = (
    OUTPUT_DIR /
    "phase3_part23_crop_retention_summary.json"
)

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print(f"Saved: {summary_path}")


# ============================================================================
# TEXT REPORT
# ============================================================================

report_path = (
    OUTPUT_DIR /
    "phase3_part23_crop_retention_report.txt"
)

with open(report_path, "w", encoding="utf-8") as f:

    f.write("=" * 78 + "\n")
    f.write("PHASE 3 - PART 23\n")
    f.write("CROP RETENTION VS SEGMENTATION PERFORMANCE ANALYSIS\n")
    f.write("=" * 78 + "\n\n")

    f.write("PURPOSE\n")
    f.write(
        "Analyze whether anatomical retention within the fixed "
        "96 x 96 x 96 model crop is associated with segmentation "
        "performance in T2 SPACE cases.\n\n"
    )

    f.write("DATASET\n")
    f.write(f"Cases analyzed: {len(df)}\n")
    f.write(f"Failure cases (Dice < 0.70): {len(failure_cases)}\n")
    f.write(f"Successful cases (Dice >= 0.70): {len(successful_cases)}\n\n")

    f.write("MODEL PERFORMANCE\n")
    f.write(f"Mean Dice: {df['model_dice'].mean():.6f}\n")
    f.write(f"Median Dice: {df['model_dice'].median():.6f}\n")
    f.write(f"Minimum Dice: {df['model_dice'].min():.6f}\n")
    f.write(f"Maximum Dice: {df['model_dice'].max():.6f}\n\n")

    f.write("RETENTION CORRELATIONS\n")

    for _, row in correlation_df.iterrows():
        f.write(
            f"{row['feature']}: "
            f"{row['pearson_correlation_with_dice']:.6f}\n"
        )

    f.write("\n")
    f.write(
        f"Combined retention correlation: "
        f"{combined_corr:.6f}\n"
    )

    f.write("\nBEST RETENTION PREDICTOR\n")
    f.write(
        f"{best_row['feature']}: "
        f"{best_row['pearson_correlation_with_dice']:.6f}\n"
    )

    f.write("\nFAILURE CASES\n")
    for case in failure_cases:
        f.write(f"- {case}\n")

    f.write("\nSUCCESSFUL CASES\n")
    for case in successful_cases:
        f.write(f"- {case}\n")

    f.write("\nINTERPRETATION\n")
    f.write(
        "The analysis is exploratory and is based on the available "
        "T2 SPACE cases. Correlation indicates association and does "
        "not by itself establish causation. The results should be "
        "interpreted together with the visual and error analyses "
        "from Parts 14, 15, 19, 21 and 22.\n"
    )

print(f"Saved: {report_path}")


# ============================================================================
# FINAL OUTPUT
# ============================================================================

print_header("PART 23 COMPLETE")

print(f"Cases analyzed      : {len(df)}")
print(f"Failure cases       : {len(failure_cases)}")
print(f"Successful cases    : {len(successful_cases)}")
print(f"Mean Dice           : {df['model_dice'].mean():.6f}")
print(
    f"Best retention predictor: "
    f"{best_row['feature']}"
)
print(
    f"Best correlation    : "
    f"{best_row['pearson_correlation_with_dice']:.6f}"
)

print()
print("=" * 78)
print("OUTPUT DIRECTORY")
print("=" * 78)
print(OUTPUT_DIR)

print()
print("=" * 78)
print("PHASE 3 - PART 23 COMPLETE")
print("=" * 78)