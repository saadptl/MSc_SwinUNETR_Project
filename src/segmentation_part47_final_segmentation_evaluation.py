"""
Part 4.7 — Final Segmentation Evaluation and Error Analysis

READ-ONLY FINAL ANALYSIS

This script does NOT:
- train a model
- load the MRI images
- run new inference
- modify the checkpoint
- modify the Part 2.20B geometry pipeline
- create a new test split
- fabricate voxel masks

It analyzes the already completed Part 3.3 untouched-test results.

Source files:
    outputs/segmentation/rsna_part33_untouched_test_evaluation/
        part33_test_point_records.csv
        part33_untouched_test_summary.json

Outputs:
    outputs/segmentation/part47_final_segmentation_evaluation/
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part33_untouched_test_evaluation"
)

POINT_RECORDS = SOURCE_DIR / "part33_test_point_records.csv"
SUMMARY_JSON = SOURCE_DIR / "part33_untouched_test_summary.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part47_final_segmentation_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DISEASE_ORDER = [
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]

LEVEL_ORDER = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]


# ============================================================
# HELPERS
# ============================================================

def save_dataframe(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def save_figure(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def safe_mean(series):
    if len(series) == 0:
        return np.nan
    return float(series.mean())


def safe_median(series):
    if len(series) == 0:
        return np.nan
    return float(series.median())


def safe_std(series):
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1))


def pct(value):
    return float(value) * 100.0


# ============================================================
# LOAD OFFICIAL PART 3.3 RESULTS
# ============================================================

print("=" * 80)
print("PART 4.7 — FINAL SEGMENTATION EVALUATION")
print("=" * 80)

print("\nREAD-ONLY ANALYSIS")
print("No model inference will be performed.")
print("No checkpoint will be modified.")
print("No test split will be changed.")
print()

if not POINT_RECORDS.exists():
    raise FileNotFoundError(
        f"Point-record file not found:\n{POINT_RECORDS}"
    )

if not SUMMARY_JSON.exists():
    raise FileNotFoundError(
        f"Summary JSON not found:\n{SUMMARY_JSON}"
    )


df = pd.read_csv(POINT_RECORDS)

with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
    official_summary = json.load(f)


print(f"Source point records: {POINT_RECORDS}")
print(f"Source summary:       {SUMMARY_JSON}")

print(f"\nPoint-record shape: {df.shape}")

required_columns = [
    "study_id",
    "series_id",
    "level",
    "class_id",
    "class_name",
    "true_probability",
    "predicted_class",
    "correct",
    "hit_050",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Required columns missing from point records: {missing}"
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "-" * 80)
print("SOURCE VALIDATION")
print("-" * 80)

expected_points = int(official_summary["test_points"])
expected_cases = int(official_summary["test_cases"])

actual_points = len(df)

# The Part 3.3 point-record CSV contains study/series identifiers,
# but study_id alone is NOT guaranteed to uniquely identify every
# untouched test case. Therefore, the official test-case count from
# Part 3.3 is retained as the authoritative cohort count.
actual_studies = df["study_id"].nunique()
actual_series = df["series_id"].nunique()

print(f"Official test cases     : {expected_cases}")
print(f"Unique study IDs        : {actual_studies}")
print(f"Unique series IDs       : {actual_series}")

print(f"Expected test points    : {expected_points}")
print(f"Observed test points    : {actual_points}")

if actual_points != expected_points:
    raise ValueError(
        f"Point-count mismatch: expected {expected_points}, "
        f"observed {actual_points}"
    )

print("✓ Official Part 3.3 test-point cohort confirmed.")
print(
    "✓ Official Part 3.3 test-case count retained "
    "from the authoritative summary JSON."
)

print("✓ Test cohort integrity confirmed.")


# Check class names
unexpected_diseases = sorted(
    set(df["class_name"].unique()) - set(DISEASE_ORDER)
)

if unexpected_diseases:
    raise ValueError(
        f"Unexpected disease labels found: {unexpected_diseases}"
    )

# Check levels
unexpected_levels = sorted(
    set(df["level"].unique()) - set(LEVEL_ORDER)
)

if unexpected_levels:
    raise ValueError(
        f"Unexpected levels found: {unexpected_levels}"
    )

# Check probability range
if (
    df["true_probability"].min() < 0
    or df["true_probability"].max() > 1
):
    raise ValueError("true_probability contains values outside [0,1].")

print("✓ Disease labels confirmed.")
print("✓ Level labels confirmed.")
print("✓ Probability range confirmed.")


# ============================================================
# OVERALL SUMMARY
# ============================================================

overall = {
    "test_cases": expected_cases,
    "test_points": expected_points,
    "observed_cases": actual_studies,
    "observed_points": actual_points,
    "official_overall": official_summary["test_overall"],
    "official_macro_disease": official_summary["test_macro"],
    "official_mean_true_probability": official_summary[
        "test_mean_probability"
    ],
    "official_hit_050": official_summary["test_hit_050"],
    "official_foreground_ratio": official_summary[
        "test_foreground_ratio"
    ],
    "recomputed_mean_true_probability": float(
        df["true_probability"].mean()
    ),
    "recomputed_accuracy": float(
        df["correct"].mean()
    ),
    "recomputed_hit_050": float(
        df["hit_050"].mean()
    ),
    "training_performed": official_summary["training_performed"],
    "checkpoint_modified": official_summary["checkpoint_modified"],
    "dashboard_modified": official_summary["dashboard_modified"],
    "manual_voxel_masks_fabricated": official_summary[
        "manual_voxel_masks_fabricated"
    ],
}

overall_df = pd.DataFrame([overall])

save_dataframe(
    overall_df,
    "overall_summary.csv"
)


# ============================================================
# DISEASE-WISE METRICS
# ============================================================

disease_rows = []

for disease in DISEASE_ORDER:

    sub = df[df["class_name"] == disease].copy()

    disease_rows.append({
        "class_id": int(sub["class_id"].iloc[0]),
        "class_name": disease,
        "count": len(sub),
        "accuracy": float(sub["correct"].mean()),
        "accuracy_percent": pct(sub["correct"].mean()),
        "mean_true_probability": safe_mean(
            sub["true_probability"]
        ),
        "median_true_probability": safe_median(
            sub["true_probability"]
        ),
        "std_true_probability": safe_std(
            sub["true_probability"]
        ),
        "hit_050": float(sub["hit_050"].mean()),
        "hit_050_percent": pct(sub["hit_050"].mean()),
        "correct_count": int(sub["correct"].sum()),
        "incorrect_count": int(
            (1 - sub["correct"]).sum()
        ),
    })


disease_df = pd.DataFrame(disease_rows)

save_dataframe(
    disease_df,
    "disease_wise_metrics.csv"
)


# ============================================================
# LEVEL-WISE METRICS
# ============================================================

level_rows = []

for level in LEVEL_ORDER:

    sub = df[df["level"] == level].copy()

    level_rows.append({
        "level": level,
        "count": len(sub),
        "accuracy": float(sub["correct"].mean()),
        "accuracy_percent": pct(sub["correct"].mean()),
        "mean_true_probability": safe_mean(
            sub["true_probability"]
        ),
        "median_true_probability": safe_median(
            sub["true_probability"]
        ),
        "hit_050": float(sub["hit_050"].mean()),
        "hit_050_percent": pct(sub["hit_050"].mean()),
        "correct_count": int(sub["correct"].sum()),
        "incorrect_count": int(
            (1 - sub["correct"]).sum()
        ),
    })


level_df = pd.DataFrame(level_rows)

save_dataframe(
    level_df,
    "level_wise_metrics.csv"
)


# ============================================================
# DISEASE × LEVEL METRICS
# ============================================================

disease_level_rows = []

for disease in DISEASE_ORDER:

    for level in LEVEL_ORDER:

        sub = df[
            (df["class_name"] == disease)
            & (df["level"] == level)
        ].copy()

        if len(sub) == 0:
            continue

        disease_level_rows.append({
            "class_name": disease,
            "level": level,
            "count": len(sub),
            "accuracy": float(sub["correct"].mean()),
            "accuracy_percent": pct(
                sub["correct"].mean()
            ),
            "mean_true_probability": safe_mean(
                sub["true_probability"]
            ),
            "hit_050": float(
                sub["hit_050"].mean()
            ),
            "hit_050_percent": pct(
                sub["hit_050"].mean()
            ),
        })


disease_level_df = pd.DataFrame(
    disease_level_rows
)

save_dataframe(
    disease_level_df,
    "disease_level_metrics.csv"
)


# ============================================================
# ERROR RECORDS
# ============================================================

errors = df[df["correct"] == 0].copy()

errors["error_type"] = np.where(
    errors["predicted_class"] == 0,
    "Predicted background",
    "Predicted another class",
)

errors["confidence_band"] = pd.cut(
    errors["true_probability"],
    bins=[-0.001, 0.25, 0.50, 0.75, 1.0],
    labels=[
        "<0.25",
        "0.25-0.50",
        "0.50-0.75",
        "0.75-1.00",
    ],
)

save_dataframe(
    errors,
    "error_records.csv"
)


# ============================================================
# HIGH-CONFIDENCE ERRORS
# ============================================================

high_confidence_errors = df[
    (df["correct"] == 0)
    & (df["true_probability"] >= 0.50)
].copy()

high_confidence_errors["error_type"] = np.where(
    high_confidence_errors["predicted_class"] == 0,
    "Predicted background",
    "Predicted another class",
)

high_confidence_errors = (
    high_confidence_errors
    .sort_values(
        "true_probability",
        ascending=False
    )
)

save_dataframe(
    high_confidence_errors,
    "high_confidence_errors.csv"
)


# ============================================================
# PROBABILITY BINS
# ============================================================

bins = [
    -0.001,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

labels = [
    "0.00-0.10",
    "0.10-0.20",
    "0.20-0.30",
    "0.30-0.40",
    "0.40-0.50",
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-0.90",
    "0.90-1.00",
]

df["probability_bin"] = pd.cut(
    df["true_probability"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)

probability_df = (
    df.groupby(
        "probability_bin",
        observed=False
    )
    .agg(
        count=("correct", "size"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
        hit_050=("hit_050", "mean"),
    )
    .reset_index()
)

probability_df["accuracy_percent"] = (
    probability_df["accuracy"] * 100
)

probability_df["hit_050_percent"] = (
    probability_df["hit_050"] * 100
)

save_dataframe(
    probability_df,
    "probability_bins.csv"
)


# ============================================================
# REPRESENTATIVE TEST RECORD GROUPS
# ============================================================

# The official Part 3.3 test-case count is authoritative.
# The point-record CSV does not expose a unique test-case ID,
# therefore representative records are selected at the
# study/series level without calling them individual cases.

series_summary = (
    df.groupby(["study_id", "series_id"])
    .agg(
        points=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_probability=(
            "true_probability",
            "mean"
        ),
        hit_050=("hit_050", "mean"),
        errors=(
            "correct",
            lambda x: int((x == 0).sum())
        ),
    )
    .reset_index()
)

series_summary["accuracy_percent"] = (
    series_summary["accuracy"] * 100
)

series_summary["hit_050_percent"] = (
    series_summary["hit_050"] * 100
)

# Higher-performing study/series groups
best_series = (
    series_summary
    .sort_values(
        ["accuracy", "mean_probability"],
        ascending=[False, False]
    )
    .head(5)
    .copy()
)

best_series["group"] = "Higher-performing study/series"


# Lower-performing study/series groups
difficult_series = (
    series_summary
    .sort_values(
        ["accuracy", "mean_probability"],
        ascending=[True, True]
    )
    .head(5)
    .copy()
)

difficult_series["group"] = "Lower-performing study/series"


representative_cases = pd.concat(
    [
        best_series,
        difficult_series,
    ],
    ignore_index=True,
)

save_dataframe(
    representative_cases,
    "representative_study_series.csv"
)


# ============================================================
# PREDICTION CONFUSION MATRIX
# ============================================================

class_names = [
    "Background",
] + DISEASE_ORDER

class_ids = [0, 1, 2, 3, 4, 5]

confusion = pd.crosstab(
    df["class_id"],
    df["predicted_class"],
    rownames=["true_class_id"],
    colnames=["predicted_class_id"],
    dropna=False,
)

confusion = confusion.reindex(
    index=class_ids,
    columns=class_ids,
    fill_value=0,
)

confusion.index = class_names
confusion.columns = class_names

save_dataframe(
    confusion.reset_index().rename(
        columns={"index": "true_class"}
    ),
    "prediction_confusion_matrix.csv"
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL OVERALL RESULTS")
print("=" * 80)

print(f"Test cases             : {expected_cases}")
print(f"Test points            : {expected_points}")
print(
    f"Overall accuracy       : "
    f"{official_summary['test_overall']:.6f}"
)
print(
    f"Macro disease accuracy : "
    f"{official_summary['test_macro']:.6f}"
)
print(
    f"Mean true probability  : "
    f"{official_summary['test_mean_probability']:.6f}"
)
print(
    f"Hit @ 0.50             : "
    f"{official_summary['test_hit_050']:.6f}"
)
print(
    f"Foreground ratio       : "
    f"{official_summary['test_foreground_ratio']:.6f}"
)

print("\n" + "=" * 80)
print("DISEASE-WISE RESULTS")
print("=" * 80)

print(
    disease_df[
        [
            "class_name",
            "count",
            "accuracy",
            "mean_true_probability",
            "hit_050",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("LEVEL-WISE RESULTS")
print("=" * 80)

print(
    level_df[
        [
            "level",
            "count",
            "accuracy",
            "mean_true_probability",
            "hit_050",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("ERROR ANALYSIS")
print("=" * 80)

print(f"Total incorrect points : {len(errors)}")
print(
    f"Error rate             : "
    f"{len(errors) / len(df):.6f}"
)
print(
    f"High-confidence errors : "
    f"{len(high_confidence_errors)}"
)

print("\nError distribution by disease:")

error_disease = (
    errors["class_name"]
    .value_counts()
    .reindex(DISEASE_ORDER, fill_value=0)
)

print(error_disease.to_string())


# ============================================================
# FIGURE 1 — DISEASE-WISE ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    disease_df["class_name"],
    disease_df["accuracy_percent"],
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Disease")
plt.title("Part 3.3 Untouched Test — Disease-wise Accuracy")
plt.ylim(0, 100)
plt.xticks(rotation=25, ha="right")

save_figure("disease_wise_accuracy.png")


# ============================================================
# FIGURE 2 — DISEASE-WISE HIT@0.50
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    disease_df["class_name"],
    disease_df["hit_050_percent"],
)

plt.ylabel("Hit @ 0.50 (%)")
plt.xlabel("Disease")
plt.title("Part 3.3 Untouched Test — Disease-wise Hit Rate")
plt.ylim(0, 100)
plt.xticks(rotation=25, ha="right")

save_figure("disease_wise_hit050.png")


# ============================================================
# FIGURE 3 — DISEASE-WISE TRUE PROBABILITY
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    disease_df["class_name"],
    disease_df["mean_true_probability"],
)

plt.ylabel("Mean True-class Probability")
plt.xlabel("Disease")
plt.title(
    "Part 3.3 Untouched Test — Mean True-class Probability"
)
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")

save_figure("disease_wise_probability.png")


# ============================================================
# FIGURE 4 — LEVEL-WISE ACCURACY
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    level_df["level"],
    level_df["accuracy_percent"],
    marker="o",
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Spinal Level")
plt.title("Part 3.3 Untouched Test — Level-wise Accuracy")
plt.ylim(0, 100)

save_figure("level_wise_accuracy.png")


# ============================================================
# FIGURE 5 — LEVEL-WISE HIT@0.50
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    level_df["level"],
    level_df["hit_050_percent"],
    marker="o",
)

plt.ylabel("Hit @ 0.50 (%)")
plt.xlabel("Spinal Level")
plt.title("Part 3.3 Untouched Test — Level-wise Hit Rate")
plt.ylim(0, 100)

save_figure("level_wise_hit050.png")


# ============================================================
# FIGURE 6 — DISEASE × LEVEL ACCURACY HEATMAP
# ============================================================

heatmap = (
    disease_level_df
    .pivot(
        index="class_name",
        columns="level",
        values="accuracy_percent",
    )
    .reindex(index=DISEASE_ORDER, columns=LEVEL_ORDER)
)

plt.figure(figsize=(10, 6))

plt.imshow(
    heatmap.values,
    aspect="auto",
)

plt.colorbar(
    label="Accuracy (%)"
)

plt.xticks(
    range(len(LEVEL_ORDER)),
    LEVEL_ORDER,
)

plt.yticks(
    range(len(DISEASE_ORDER)),
    DISEASE_ORDER,
)

plt.xlabel("Spinal Level")
plt.ylabel("Disease")
plt.title("Disease × Level Accuracy")

for i in range(len(DISEASE_ORDER)):
    for j in range(len(LEVEL_ORDER)):
        value = heatmap.iloc[i, j]

        if not pd.isna(value):
            plt.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
            )

save_figure(
    "disease_level_accuracy_heatmap.png"
)


# ============================================================
# FIGURE 7 — PROBABILITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["true_probability"],
    bins=20,
)

plt.axvline(
    0.50,
    linestyle="--",
    label="0.50 threshold",
)

plt.xlabel("True-class Probability")
plt.ylabel("Number of Points")
plt.title(
    "Part 3.3 Untouched Test — True-class Probability Distribution"
)

plt.legend()

save_figure(
    "probability_distribution.png"
)


# ============================================================
# FIGURE 8 — ERROR DISTRIBUTION
# ============================================================

error_counts = (
    errors["class_name"]
    .value_counts()
    .reindex(DISEASE_ORDER, fill_value=0)
)

plt.figure(figsize=(10, 6))

plt.bar(
    error_counts.index,
    error_counts.values,
)

plt.ylabel("Number of Errors")
plt.xlabel("Disease")
plt.title("Part 3.3 Untouched Test — Error Distribution")
plt.xticks(rotation=25, ha="right")

save_figure(
    "error_distribution.png"
)


# ============================================================
# FIGURE 9 — CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    confusion.values,
    aspect="auto",
)

plt.colorbar(
    label="Number of predictions"
)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45,
    ha="right",
)

plt.yticks(
    range(len(class_names)),
    class_names,
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Part 3.3 Untouched Test — Prediction Confusion Matrix")

for i in range(len(class_names)):
    for j in range(len(class_names)):

        value = confusion.iloc[i, j]

        plt.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
        )

save_figure(
    "confusion_matrix.png"
)


# ============================================================
# FINAL JSON
# ============================================================

final_json = {
    "experiment": "Part 4.7",
    "evaluation_type": "final_read_only_analysis",
    "source_evaluation": "Part 3.3 untouched test evaluation",
    "source_point_records": str(POINT_RECORDS),
    "source_summary": str(SUMMARY_JSON),

    "test_cases": expected_cases,
    "test_points": expected_points,

    "overall": {
        "accuracy": official_summary["test_overall"],
        "macro_disease_accuracy": official_summary[
            "test_macro"
        ],
        "mean_true_probability": official_summary[
            "test_mean_probability"
        ],
        "hit_050": official_summary["test_hit_050"],
        "foreground_ratio": official_summary[
            "test_foreground_ratio"
        ],
    },

    "disease_wise": disease_df.to_dict(
        orient="records"
    ),

    "level_wise": level_df.to_dict(
        orient="records"
    ),

    "error_analysis": {
        "total_errors": int(len(errors)),
        "error_rate": float(
            len(errors) / len(df)
        ),
        "high_confidence_errors": int(
            len(high_confidence_errors)
        ),
    },

    "integrity": {
        "training_performed": official_summary[
            "training_performed"
        ],
        "checkpoint_modified": official_summary[
            "checkpoint_modified"
        ],
        "dashboard_modified": official_summary[
            "dashboard_modified"
        ],
        "manual_voxel_masks_fabricated": official_summary[
            "manual_voxel_masks_fabricated"
        ],
    },
}

with open(
    OUTPUT_DIR / "part47_final_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        final_json,
        f,
        indent=2,
    )

print(
    f"Saved: "
    f"{OUTPUT_DIR / 'part47_final_summary.json'}"
)


# ============================================================
# FINAL REPORT
# ============================================================

report_lines = []

report_lines.append(
    "PART 4.7 — FINAL SEGMENTATION EVALUATION"
)

report_lines.append("=" * 80)
report_lines.append("")

report_lines.append(
    "Evaluation basis:"
)
report_lines.append(
    "Existing Part 3.3 untouched-test evaluation results."
)
report_lines.append(
    "No new inference was performed."
)
report_lines.append(
    "No new test split was created."
)
report_lines.append(
    "No checkpoint was modified."
)
report_lines.append("")

report_lines.append(
    f"Test cases: {expected_cases}"
)
report_lines.append(
    f"Test points: {expected_points}"
)
report_lines.append(
    f"Overall accuracy: "
    f"{official_summary['test_overall']:.6f}"
)
report_lines.append(
    f"Macro disease accuracy: "
    f"{official_summary['test_macro']:.6f}"
)
report_lines.append(
    f"Mean true probability: "
    f"{official_summary['test_mean_probability']:.6f}"
)
report_lines.append(
    f"Hit @ 0.50: "
    f"{official_summary['test_hit_050']:.6f}"
)
report_lines.append(
    f"Foreground ratio: "
    f"{official_summary['test_foreground_ratio']:.6f}"
)
report_lines.append("")

report_lines.append(
    "DISEASE-WISE RESULTS"
)
report_lines.append("-" * 80)

for _, row in disease_df.iterrows():

    report_lines.append(
        f"{row['class_name']}: "
        f"n={int(row['count'])}, "
        f"accuracy={row['accuracy']:.6f}, "
        f"mean_probability="
        f"{row['mean_true_probability']:.6f}, "
        f"hit@0.50={row['hit_050']:.6f}"
    )

report_lines.append("")
report_lines.append(
    "LEVEL-WISE RESULTS"
)
report_lines.append("-" * 80)

for _, row in level_df.iterrows():

    report_lines.append(
        f"{row['level']}: "
        f"n={int(row['count'])}, "
        f"accuracy={row['accuracy']:.6f}, "
        f"mean_probability="
        f"{row['mean_true_probability']:.6f}, "
        f"hit@0.50={row['hit_050']:.6f}"
    )

report_lines.append("")
report_lines.append(
    "ERROR ANALYSIS"
)
report_lines.append("-" * 80)

report_lines.append(
    f"Total incorrect points: {len(errors)}"
)

report_lines.append(
    f"Error rate: {len(errors) / len(df):.6f}"
)

report_lines.append(
    f"High-confidence errors "
    f"(true probability >= 0.50): "
    f"{len(high_confidence_errors)}"
)

report_lines.append("")
report_lines.append(
    "Disease-wise error counts:"
)

for disease, count in error_disease.items():

    report_lines.append(
        f"  {disease}: {int(count)}"
    )

report_lines.append("")
report_lines.append(
    "INTEGRITY FLAGS"
)
report_lines.append("-" * 80)

report_lines.append(
    f"Training performed: "
    f"{official_summary['training_performed']}"
)

report_lines.append(
    f"Checkpoint modified: "
    f"{official_summary['checkpoint_modified']}"
)

report_lines.append(
    f"Dashboard modified: "
    f"{official_summary['dashboard_modified']}"
)

report_lines.append(
    f"Manual voxel masks fabricated: "
    f"{official_summary['manual_voxel_masks_fabricated']}"
)

with open(
    OUTPUT_DIR / "part47_final_report.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(report_lines))

print(
    f"Saved: "
    f"{OUTPUT_DIR / 'part47_final_report.txt'}"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("PART 4.7 COMPLETE")
print("=" * 80)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nNo training performed.")
print("No checkpoint modified.")
print("No geometry modified.")
print("No new test split created.")
print("No voxel ground truth fabricated.")