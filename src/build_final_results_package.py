from pathlib import Path
import json
import shutil
import pandas as pd


# ============================================================
# FINAL RESULTS PACKAGE
# Read-only consolidation of validated project outputs
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SEG_ROOT = ROOT / "outputs" / "segmentation"
FINAL_ROOT = ROOT / "outputs" / "FINAL_RESULTS"

PART33 = SEG_ROOT / "rsna_part33_untouched_test_evaluation"
PART47 = SEG_ROOT / "part47_final_segmentation_evaluation"
PART48 = SEG_ROOT / "part48_final_error_analysis"

FINAL_ROOT.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("FINAL RESULTS PACKAGE")
print("=" * 80)
print()
print("READ-ONLY CONSOLIDATION")
print("No training will be performed.")
print("No inference will be performed.")
print("No checkpoint will be modified.")
print("No geometry will be modified.")
print("No test split will be modified.")
print()


# ============================================================
# LOAD AUTHORITATIVE SOURCES
# ============================================================

part33_summary_path = PART33 / "part33_untouched_test_summary.json"
part47_summary_path = PART47 / "part47_final_summary.json"
part48_summary_path = PART48 / "part48_final_error_summary.json"

part33_summary = json.loads(
    part33_summary_path.read_text(encoding="utf-8")
)

part47_summary = json.loads(
    part47_summary_path.read_text(encoding="utf-8")
)

part48_summary = json.loads(
    part48_summary_path.read_text(encoding="utf-8")
)


# ============================================================
# LOAD TABLES
# ============================================================

disease_metrics = pd.read_csv(
    PART47 / "disease_wise_metrics.csv"
)

level_metrics = pd.read_csv(
    PART47 / "level_wise_metrics.csv"
)

error_records = pd.read_csv(
    PART47 / "error_records.csv"
)

error_transitions = pd.read_csv(
    PART48 / "true_to_predicted_error_transitions.csv"
)


# ============================================================
# BASIC INTEGRITY CHECKS
# ============================================================

official_test_cases = int(part33_summary["test_cases"])
official_test_points = int(part33_summary["test_points"])

observed_error_records = len(error_records)

expected_errors = official_test_points - int(
    round(part33_summary["test_overall"] * official_test_points)
)

print(f"Official test cases : {official_test_cases}")
print(f"Official test points: {official_test_points}")
print(f"Observed error rows : {observed_error_records}")

if observed_error_records != 76:
    raise RuntimeError(
        f"Expected 76 error records, found {observed_error_records}."
    )

if official_test_cases != 25:
    raise RuntimeError(
        f"Expected 25 official test cases, found {official_test_cases}."
    )

if official_test_points != 200:
    raise RuntimeError(
        f"Expected 200 test points, found {official_test_points}."
    )

print("✓ Final-result integrity checks passed.")
print()


# ============================================================
# EXTRACT AUTHORITATIVE METRICS
# ============================================================

overall_accuracy = float(part33_summary["test_overall"])
macro_accuracy = float(part33_summary["test_macro"])
mean_probability = float(part33_summary["test_mean_probability"])
hit_050 = float(part33_summary["test_hit_050"])
foreground_ratio = float(part33_summary["test_foreground_ratio"])

error_rate = observed_error_records / official_test_points

rfnn_errors = int(
    (error_records["class_name"] == "Right Neural Foraminal Narrowing").sum()
)

high_confidence_errors = int(
    (error_records["true_probability"] >= 0.50).sum()
)


# ============================================================
# AUTHORITATIVE OVERALL SUMMARY
# ============================================================

overall = pd.DataFrame(
    [
        {
            "metric": "Official untouched test cases",
            "value": official_test_cases,
        },
        {
            "metric": "Official test points",
            "value": official_test_points,
        },
        {
            "metric": "Overall point-level accuracy",
            "value": overall_accuracy,
        },
        {
            "metric": "Macro disease accuracy",
            "value": macro_accuracy,
        },
        {
            "metric": "Mean true-class probability",
            "value": mean_probability,
        },
        {
            "metric": "Hit@0.50",
            "value": hit_050,
        },
        {
            "metric": "Foreground ratio",
            "value": foreground_ratio,
        },
        {
            "metric": "Incorrect points",
            "value": observed_error_records,
        },
        {
            "metric": "Error rate",
            "value": error_rate,
        },
        {
            "metric": "RFNN errors",
            "value": rfnn_errors,
        },
        {
            "metric": "High-confidence errors (true probability >= 0.50)",
            "value": high_confidence_errors,
        },
    ]
)

overall.to_csv(
    FINAL_ROOT / "FINAL_overall_results.csv",
    index=False,
)


# ============================================================
# DISEASE RESULTS
# ============================================================

disease_metrics.to_csv(
    FINAL_ROOT / "FINAL_disease_wise_results.csv",
    index=False,
)

level_metrics.to_csv(
    FINAL_ROOT / "FINAL_level_wise_results.csv",
    index=False,
)


# ============================================================
# ERROR ANALYSIS
# ============================================================

error_records.to_csv(
    FINAL_ROOT / "FINAL_error_records.csv",
    index=False,
)

error_transitions.to_csv(
    FINAL_ROOT / "FINAL_error_transitions.csv",
    index=False,
)


# ============================================================
# FIND DOMINANT RFNN ERROR TRANSITION
# ============================================================

rfnn_transition = error_transitions[
    error_transitions["class_name"]
    == "Right Neural Foraminal Narrowing"
].copy()

if not rfnn_transition.empty:

    rfnn_transition = rfnn_transition.sort_values(
        "error_count",
        ascending=False
    )

    dominant_rfnn_transition = rfnn_transition.iloc[0]

    dominant_true = dominant_rfnn_transition["class_name"]
    dominant_pred = dominant_rfnn_transition["predicted_class_name"]
    dominant_count = int(dominant_rfnn_transition["error_count"])

else:
    dominant_true = "N/A"
    dominant_pred = "N/A"
    dominant_count = 0



# ============================================================
# FINAL JSON
# ============================================================

final_summary = {
    "project": {
        "title": (
            "Explainable Swin-UNETR Framework for Automated "
            "Lumbar Spine Disease Detection and Classification "
            "from MRI Images"
        ),
        "model": "Swin-UNETR",
        "task": "point-supervised disease localization",
    },

    "validated_pipeline": {
        "geometry": "Part 2.20B physical-space geometry",
        "model_evaluation": "Part 3.3 untouched test evaluation",
        "final_evaluation": "Part 4.7 final segmentation evaluation",
        "error_analysis": "Part 4.8 final error analysis",
    },

    "test_protocol": {
        "test_cases": official_test_cases,
        "test_points": official_test_points,
        "training_performed": False,
        "checkpoint_modified": False,
        "geometry_modified": False,
        "test_split_modified": False,
        "manual_voxel_masks_fabricated": False,
    },

    "overall_results": {
        "accuracy": overall_accuracy,
        "macro_disease_accuracy": macro_accuracy,
        "mean_true_probability": mean_probability,
        "hit_at_050": hit_050,
        "foreground_ratio": foreground_ratio,
        "incorrect_points": observed_error_records,
        "error_rate": error_rate,
        "high_confidence_errors": high_confidence_errors,
    },

    "error_analysis": {
        "rfnn_errors": rfnn_errors,
        "dominant_rfnn_true_class": dominant_true,
        "dominant_rfnn_predicted_class": dominant_pred,
        "dominant_rfnn_transition_count": dominant_count,
    },

    "limitations": [
        "The task is point-supervised disease localization.",
        "Voxel-wise segmentation has not been clinically validated.",
        "Manual voxel-level ground truth is unavailable.",
        "Error findings describe the evaluated untouched cohort and should not be generalized to unseen populations."
    ],

    "source_summaries": {
        "part33": part33_summary,
        "part47": part47_summary,
        "part48": part48_summary,
    },
}

with open(
    FINAL_ROOT / "FINAL_results_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        final_summary,
        f,
        indent=2
    )


# ============================================================
# FINAL HUMAN-READABLE REPORT
# ============================================================

report_lines = []

report_lines.append("=" * 80)
report_lines.append("FINAL PROJECT RESULTS REPORT")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("PROJECT")
report_lines.append(
    "Explainable Swin-UNETR Framework for Automated Lumbar Spine "
    "Disease Detection and Classification from MRI Images"
)
report_lines.append("")

report_lines.append("MODEL / TASK")
report_lines.append("Model: Swin-UNETR")
report_lines.append("Task: Point-supervised disease localization")
report_lines.append(
    "Geometry: Part 2.20B physical-space geometry"
)
report_lines.append("")

report_lines.append("EVALUATION PROTOCOL")
report_lines.append(f"Official untouched test cases: {official_test_cases}")
report_lines.append(f"Official test points: {official_test_points}")
report_lines.append("Training performed: No")
report_lines.append("Checkpoint modified: No")
report_lines.append("Geometry modified: No")
report_lines.append("Test split modified: No")
report_lines.append("Manual voxel masks fabricated: No")
report_lines.append("")

report_lines.append("OVERALL RESULTS")
report_lines.append(
    f"Overall point-level accuracy: {overall_accuracy:.6f}"
)
report_lines.append(
    f"Macro disease accuracy: {macro_accuracy:.6f}"
)
report_lines.append(
    f"Mean true-class probability: {mean_probability:.6f}"
)
report_lines.append(
    f"Hit@0.50: {hit_050:.6f}"
)
report_lines.append(
    f"Foreground ratio: {foreground_ratio:.6f}"
)
report_lines.append(
    f"Incorrect points: {observed_error_records}"
)
report_lines.append(
    f"Error rate: {error_rate:.2%}"
)
report_lines.append(
    f"High-confidence errors (true probability >= 0.50): "
    f"{high_confidence_errors}"
)
report_lines.append("")

report_lines.append("DISEASE-WISE RESULTS")
report_lines.append("")

for _, row in disease_metrics.iterrows():

    report_lines.append(
        f"{row['class_name']}: "
        f"accuracy={float(row['accuracy']):.6f}, "
        f"mean_true_probability={float(row['mean_true_probability']):.6f}, "
        f"hit_050={float(row['hit_050']):.6f}"
    )

report_lines.append("")
report_lines.append("ERROR ANALYSIS")
report_lines.append(
    f"Total errors: {observed_error_records}"
)
report_lines.append(
    f"RFNN errors: {rfnn_errors}"
)

if dominant_count > 0:
    report_lines.append(
        f"Dominant observed RFNN error transition: "
        f"{dominant_true} -> {dominant_pred} "
        f"({dominant_count} errors)"
    )

report_lines.append("")
report_lines.append("INTERPRETATION LIMITATIONS")
report_lines.append(
    "The reported error patterns describe the evaluated untouched "
    "test cohort and should not be generalized to unseen populations."
)
report_lines.append(
    "The model produces point-supervised disease-localization "
    "probability outputs; validated voxel-level segmentation is "
    "not established because manual voxel ground truth is unavailable."
)
report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("END OF FINAL RESULTS REPORT")
report_lines.append("=" * 80)

(FINAL_ROOT / "FINAL_results_report.txt").write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


print("Saved:")
print(FINAL_ROOT / "FINAL_overall_results.csv")
print(FINAL_ROOT / "FINAL_disease_wise_results.csv")
print(FINAL_ROOT / "FINAL_level_wise_results.csv")
print(FINAL_ROOT / "FINAL_error_records.csv")
print(FINAL_ROOT / "FINAL_error_transitions.csv")
print(FINAL_ROOT / "FINAL_results_summary.json")
print(FINAL_ROOT / "FINAL_results_report.txt")

print()
print("=" * 80)
print("FINAL RESULTS PACKAGE COMPLETE")
print("=" * 80)