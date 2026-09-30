"""
Part 4.8 — Final Error Analysis & Representative Test Cases

READ-ONLY ANALYSIS

Source:
    outputs/segmentation/part47_final_segmentation_evaluation/

No:
    - model inference
    - training
    - checkpoint modification
    - geometry modification
    - test split modification
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part47_final_segmentation_evaluation"
)

ERROR_FILE = SOURCE_DIR / "error_records.csv"
SUMMARY_FILE = SOURCE_DIR / "part47_final_summary.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part48_final_error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LABELS
# ============================================================

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

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

def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_plot(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("PART 4.8 — FINAL ERROR ANALYSIS")
print("=" * 80)

print("\nREAD-ONLY ANALYSIS")
print("No model inference will be performed.")
print("No training will be performed.")
print("No checkpoint will be modified.")
print("No geometry will be modified.")
print("No test split will be modified.")

if not ERROR_FILE.exists():
    raise FileNotFoundError(ERROR_FILE)

if not SUMMARY_FILE.exists():
    raise FileNotFoundError(SUMMARY_FILE)

errors = pd.read_csv(ERROR_FILE)

with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
    summary = json.load(f)

print(f"\nSource: {ERROR_FILE}")
print(f"Error records: {len(errors)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

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
    "error_type",
    "confidence_band",
]

missing = [
    c for c in required_columns
    if c not in errors.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

if not (errors["correct"] == 0).all():
    raise ValueError(
        "error_records.csv contains non-error records."
    )

print("✓ Error-record integrity confirmed.")


# ============================================================
# ADD PREDICTED CLASS NAME
# ============================================================

errors["predicted_class_name"] = (
    errors["predicted_class"]
    .map(CLASS_NAMES)
    .fillna("Unknown")
)


# ============================================================
# 1. OVERALL ERROR SUMMARY
# ============================================================

total_errors = len(errors)

total_points = int(summary["test_points"])

error_rate = total_errors / total_points

overall_error = pd.DataFrame([{
    "test_points": total_points,
    "total_errors": total_errors,
    "error_rate": error_rate,
    "error_rate_percent": error_rate * 100,
}])

save_csv(
    overall_error,
    "overall_error_summary.csv"
)


# ============================================================
# 2. ERRORS BY TRUE DISEASE
# ============================================================

disease_error = (
    errors["class_name"]
    .value_counts()
    .reindex(DISEASE_ORDER, fill_value=0)
    .reset_index()
)

disease_error.columns = [
    "class_name",
    "error_count",
]

disease_counts = (
    pd.read_csv(
        PROJECT_ROOT
        / "outputs"
        / "segmentation"
        / "part47_final_segmentation_evaluation"
        / "disease_wise_metrics.csv"
    )
)

disease_error = disease_error.merge(
    disease_counts[
        [
            "class_name",
            "count",
            "accuracy",
        ]
    ],
    on="class_name",
    how="left",
)

disease_error["error_rate_within_disease"] = (
    disease_error["error_count"]
    / disease_error["count"]
)

disease_error["error_share_percent"] = (
    disease_error["error_count"]
    / total_errors
    * 100
)

save_csv(
    disease_error,
    "disease_error_analysis.csv"
)


# ============================================================
# 3. ERRORS BY LEVEL
# ============================================================

level_error = (
    errors["level"]
    .value_counts()
    .reindex(LEVEL_ORDER, fill_value=0)
    .reset_index()
)

level_error.columns = [
    "level",
    "error_count",
]

level_counts = pd.read_csv(
    SOURCE_DIR / "level_wise_metrics.csv"
)

level_error = level_error.merge(
    level_counts[
        [
            "level",
            "count",
            "accuracy",
        ]
    ],
    on="level",
    how="left",
)

level_error["error_rate_within_level"] = (
    level_error["error_count"]
    / level_error["count"]
)

level_error["error_share_percent"] = (
    level_error["error_count"]
    / total_errors
    * 100
)

save_csv(
    level_error,
    "level_error_analysis.csv"
)


# ============================================================
# 4. TRUE → PREDICTED ERROR TRANSITIONS
# ============================================================

transition = (
    errors
    .groupby(
        [
            "class_name",
            "predicted_class",
            "predicted_class_name",
        ]
    )
    .size()
    .reset_index(
        name="error_count"
    )
    .sort_values(
        "error_count",
        ascending=False,
    )
)

transition["error_share_percent"] = (
    transition["error_count"]
    / total_errors
    * 100
)

save_csv(
    transition,
    "true_to_predicted_error_transitions.csv"
)


# ============================================================
# 5. DISEASE × PREDICTED CLASS MATRIX
# ============================================================

transition_matrix = pd.crosstab(
    errors["class_name"],
    errors["predicted_class_name"],
)

transition_matrix = transition_matrix.reindex(
    index=DISEASE_ORDER,
    fill_value=0,
)

transition_matrix = transition_matrix.reindex(
    columns=[
        "Background",
        "Spinal Canal Stenosis",
        "Left Neural Foraminal Narrowing",
        "Right Neural Foraminal Narrowing",
        "Left Subarticular Stenosis",
        "Right Subarticular Stenosis",
    ],
    fill_value=0,
)

save_csv(
    transition_matrix.reset_index(),
    "disease_prediction_error_matrix.csv"
)


# ============================================================
# 6. RFNN-SPECIFIC ANALYSIS
# ============================================================

rfnn = errors[
    errors["class_name"]
    == "Right Neural Foraminal Narrowing"
].copy()

rfnn_transition = (
    rfnn["predicted_class_name"]
    .value_counts()
    .rename_axis("predicted_class_name")
    .reset_index(
        name="error_count"
    )
)

rfnn_transition["error_share_percent"] = (
    rfnn_transition["error_count"]
    / len(rfnn)
    * 100
)

save_csv(
    rfnn_transition,
    "rfnn_error_transitions.csv"
)


# RFNN by level

rfnn_level = (
    rfnn["level"]
    .value_counts()
    .reindex(
        LEVEL_ORDER,
        fill_value=0,
    )
    .reset_index()
)

rfnn_level.columns = [
    "level",
    "rfnn_error_count",
]

save_csv(
    rfnn_level,
    "rfnn_error_by_level.csv"
)


# RFNN probability statistics

rfnn_probability = pd.DataFrame([{
    "count": len(rfnn),
    "mean_true_probability": rfnn[
        "true_probability"
    ].mean(),
    "median_true_probability": rfnn[
        "true_probability"
    ].median(),
    "minimum_true_probability": rfnn[
        "true_probability"
    ].min(),
    "maximum_true_probability": rfnn[
        "true_probability"
    ].max(),
}])

save_csv(
    rfnn_probability,
    "rfnn_probability_summary.csv"
)


# ============================================================
# 7. ERROR PROBABILITY ANALYSIS
# ============================================================

probability_summary = pd.DataFrame([{
    "mean_error_probability": errors[
        "true_probability"
    ].mean(),

    "median_error_probability": errors[
        "true_probability"
    ].median(),

    "minimum_error_probability": errors[
        "true_probability"
    ].min(),

    "maximum_error_probability": errors[
        "true_probability"
    ].max(),

    "errors_ge_050": int(
        (errors["true_probability"] >= 0.50).sum()
    ),

    "errors_ge_025": int(
        (errors["true_probability"] >= 0.25).sum()
    ),
}])

save_csv(
    probability_summary,
    "error_probability_summary.csv"
)


# ============================================================
# 8. REPRESENTATIVE ERROR RECORDS
# ============================================================

# Lowest true-class probabilities
lowest_probability = (
    errors
    .sort_values(
        "true_probability",
        ascending=True,
    )
    .head(10)
    .copy()
)

lowest_probability["selection"] = (
    "Lowest true-class probability"
)


# Highest true-class probabilities among errors
highest_probability = (
    errors
    .sort_values(
        "true_probability",
        ascending=False,
    )
    .head(10)
    .copy()
)

highest_probability["selection"] = (
    "Highest true-class probability among errors"
)


representative_errors = pd.concat(
    [
        lowest_probability,
        highest_probability,
    ],
    ignore_index=True,
)

save_csv(
    representative_errors,
    "representative_error_records.csv"
)


# ============================================================
# 9. REPRESENTATIVE RFNN ERRORS
# ============================================================

representative_rfnn = (
    rfnn
    .sort_values(
        "true_probability",
        ascending=False,
    )
    .head(10)
    .copy()
)

representative_rfnn["selection"] = (
    "Representative RFNN errors"
)

save_csv(
    representative_rfnn,
    "representative_rfnn_errors.csv"
)


# ============================================================
# 10. FIGURE — ERRORS BY DISEASE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    disease_error["class_name"],
    disease_error["error_count"],
)

plt.ylabel("Number of Incorrect Predictions")
plt.xlabel("True Disease")
plt.title("Part 4.8 — Errors by Disease")
plt.xticks(
    rotation=25,
    ha="right",
)

save_plot(
    "errors_by_disease.png"
)


# ============================================================
# 11. FIGURE — ERRORS BY LEVEL
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    level_error["level"],
    level_error["error_count"],
)

plt.ylabel("Number of Incorrect Predictions")
plt.xlabel("Spinal Level")
plt.title("Part 4.8 — Errors by Spinal Level")

save_plot(
    "errors_by_level.png"
)


# ============================================================
# 12. FIGURE — TRUE → PREDICTED TRANSITIONS
# ============================================================

top_transitions = transition.head(10).copy()

transition_labels = (
    top_transitions["class_name"]
    + " → "
    + top_transitions["predicted_class_name"]
)

plt.figure(figsize=(11, 7))

plt.barh(
    transition_labels[::-1],
    top_transitions["error_count"][::-1],
)

plt.xlabel("Number of Errors")
plt.ylabel("Error Transition")
plt.title(
    "Part 4.8 — Most Frequent Error Transitions"
)

save_plot(
    "top_error_transitions.png"
)


# ============================================================
# 13. FIGURE — RFNN ERROR TRANSITIONS
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    rfnn_transition["predicted_class_name"],
    rfnn_transition["error_count"],
)

plt.ylabel("Number of RFNN Errors")
plt.xlabel("Predicted Class")
plt.title(
    "Part 4.8 — RFNN Error Transitions"
)

plt.xticks(
    rotation=25,
    ha="right",
)

save_plot(
    "rfnn_error_transitions.png"
)


# ============================================================
# 14. FIGURE — ERROR PROBABILITY
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    errors["true_probability"],
    bins=12,
)

plt.axvline(
    0.50,
    linestyle="--",
    label="0.50 threshold",
)

plt.xlabel("True-class Probability")
plt.ylabel("Number of Errors")
plt.title(
    "Part 4.8 — Probability Distribution of Errors"
)

plt.legend()

save_plot(
    "error_probability_distribution.png"
)


# ============================================================
# 15. FIGURE — DISEASE × PREDICTION ERROR MATRIX
# ============================================================

plt.figure(figsize=(12, 7))

plt.imshow(
    transition_matrix.values,
    aspect="auto",
)

plt.colorbar(
    label="Number of Errors"
)

plt.xticks(
    range(len(transition_matrix.columns)),
    transition_matrix.columns,
    rotation=45,
    ha="right",
)

plt.yticks(
    range(len(transition_matrix.index)),
    transition_matrix.index,
)

plt.xlabel("Predicted Class")
plt.ylabel("True Disease")
plt.title(
    "Part 4.8 — Disease × Predicted Class Error Matrix"
)

for i in range(
    transition_matrix.shape[0]
):
    for j in range(
        transition_matrix.shape[1]
    ):

        value = transition_matrix.iloc[i, j]

        if value != 0:
            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
            )

save_plot(
    "disease_prediction_error_matrix.png"
)


# ============================================================
# 16. FINAL REPORT
# ============================================================

report = []

report.append(
    "PART 4.8 — FINAL ERROR ANALYSIS"
)

report.append("=" * 80)
report.append("")

report.append(
    "Analysis basis:"
)
report.append(
    "Part 4.7 final read-only segmentation evaluation."
)
report.append(
    "No new model inference was performed."
)
report.append(
    "No training was performed."
)
report.append(
    "No checkpoint was modified."
)
report.append(
    "No geometry was modified."
)
report.append(
    "No test split was modified."
)
report.append("")

report.append(
    f"Total test points: {total_points}"
)

report.append(
    f"Total incorrect predictions: {total_errors}"
)

report.append(
    f"Overall error rate: "
    f"{error_rate:.6f} "
    f"({error_rate * 100:.2f}%)"
)

report.append("")

report.append(
    "ERRORS BY DISEASE"
)

report.append("-" * 80)

for _, row in disease_error.iterrows():

    report.append(
        f"{row['class_name']}: "
        f"{int(row['error_count'])} errors "
        f"out of {int(row['count'])} points "
        f"({row['error_rate_within_disease'] * 100:.2f}%)"
    )

report.append("")

report.append(
    "ERRORS BY LEVEL"
)

report.append("-" * 80)

for _, row in level_error.iterrows():

    report.append(
        f"{row['level']}: "
        f"{int(row['error_count'])} errors "
        f"out of {int(row['count'])} points "
        f"({row['error_rate_within_level'] * 100:.2f}%)"
    )

report.append("")

report.append(
    "MOST FREQUENT ERROR TRANSITIONS"
)

report.append("-" * 80)

for _, row in transition.head(10).iterrows():

    report.append(
        f"{row['class_name']} -> "
        f"{row['predicted_class_name']}: "
        f"{int(row['error_count'])} "
        f"({row['error_share_percent']:.2f}% of all errors)"
    )

report.append("")

report.append(
    "RFNN ERROR ANALYSIS"
)

report.append("-" * 80)

report.append(
    f"RFNN errors: {len(rfnn)}"
)

for _, row in rfnn_transition.iterrows():

    report.append(
        f"RFNN -> {row['predicted_class_name']}: "
        f"{int(row['error_count'])} "
        f"({row['error_share_percent']:.2f}% of RFNN errors)"
    )

report.append("")

report.append(
    f"RFNN mean true-class probability: "
    f"{rfnn['true_probability'].mean():.6f}"
)

report.append(
    f"RFNN minimum true-class probability: "
    f"{rfnn['true_probability'].min():.6f}"
)

report.append(
    f"RFNN maximum true-class probability: "
    f"{rfnn['true_probability'].max():.6f}"
)

report.append("")

report.append(
    "ERROR PROBABILITY"
)

report.append("-" * 80)

report.append(
    f"Mean true-class probability among errors: "
    f"{errors['true_probability'].mean():.6f}"
)

report.append(
    f"Median true-class probability among errors: "
    f"{errors['true_probability'].median():.6f}"
)

report.append(
    f"Errors with probability >= 0.50: "
    f"{int((errors['true_probability'] >= 0.50).sum())}"
)

report.append("")

report.append(
    "INTERPRETATION"
)

report.append("-" * 80)

report.append(
    "The error analysis identifies the distribution of "
    "incorrect predictions within the evaluated untouched "
    "test cohort. These findings describe this evaluation "
    "cohort and should not be generalized to unseen clinical "
    "populations without additional validation."
)

report.append(
    "RFNN produced the largest number of incorrect points "
    "in this evaluation cohort."
)

report.append(
    "The dominant RFNN error transition was RFNN to LFNN."
)

report.append(
    "No incorrect point had a recorded true-class probability "
    "of at least 0.50."
)

with open(
    OUTPUT_DIR / "part48_final_error_report.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(report))

print(
    f"Saved: "
    f"{OUTPUT_DIR / 'part48_final_error_report.txt'}"
)


# ============================================================
# JSON SUMMARY
# ============================================================

json_summary = {
    "experiment": "Part 4.8",
    "source": "Part 4.7 final segmentation evaluation",

    "test_points": total_points,
    "total_errors": total_errors,
    "error_rate": error_rate,

    "errors_by_disease": disease_error.to_dict(
        orient="records"
    ),

    "errors_by_level": level_error.to_dict(
        orient="records"
    ),

    "top_error_transitions": transition.head(
        10
    ).to_dict(
        orient="records"
    ),

    "rfnn": {
        "error_count": int(len(rfnn)),
        "mean_true_probability": float(
            rfnn["true_probability"].mean()
        ),
        "min_true_probability": float(
            rfnn["true_probability"].min()
        ),
        "max_true_probability": float(
            rfnn["true_probability"].max()
        ),
        "transitions": rfnn_transition.to_dict(
            orient="records"
        ),
    },

    "error_probability": {
        "mean": float(
            errors["true_probability"].mean()
        ),
        "median": float(
            errors["true_probability"].median()
        ),
        "errors_ge_050": int(
            (
                errors["true_probability"]
                >= 0.50
            ).sum()
        ),
    },

    "integrity": {
        "training_performed": False,
        "checkpoint_modified": False,
        "geometry_modified": False,
        "test_split_modified": False,
        "manual_voxel_masks_fabricated": False,
    },
}

with open(
    OUTPUT_DIR / "part48_final_error_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        json_summary,
        f,
        indent=2,
    )

print(
    f"Saved: "
    f"{OUTPUT_DIR / 'part48_final_error_summary.json'}"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("PART 4.8 COMPLETE")
print("=" * 80)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print(f"\nTotal errors: {total_errors}")
print(f"Error rate: {error_rate * 100:.2f}%")
print(f"RFNN errors: {len(rfnn)}")

print("\nNo training performed.")
print("No checkpoint modified.")
print("No geometry modified.")
print("No test split modified.")
print("No voxel ground truth fabricated.")