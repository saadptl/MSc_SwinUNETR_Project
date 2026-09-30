"""
PHASE A - PART 2
CORRECTED CLASSIFIER EVALUATION

Purpose
-------
Read-only evaluation of the corrected classification checkpoint.

This part:
1. Loads the corrected checkpoint.
2. Extracts all stored validation metrics.
3. Verifies the study-level split metadata.
4. Verifies class weights.
5. Compares the corrected model against the historical collapsed model.
6. Produces a conservative project-quality verdict.

IMPORTANT
---------
- NO TRAINING
- NO MODEL WEIGHT MODIFICATION
- NO CHECKPOINT OVERWRITE
- NO DATASET MODIFICATION

The corrected checkpoint was produced using:
- Study-level split
- Seed 42
- 1,580 training studies
- 394 validation studies
- Zero study overlap
- Weighted CrossEntropyLoss
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

PHASE4M_DIR = OUTPUTS_DIR / "phase4m"

CORRECTED_CHECKPOINT = (
    MODELS_DIR
    / "corrected_study_split_weighted_best_model.pth"
)

ORIGINAL_CHECKPOINT = (
    MODELS_DIR
    / "best_model.pth"
)

TRAIN_CSV = (
    PHASE4M_DIR
    / "proposed_train_samples.csv"
)

VAL_CSV = (
    PHASE4M_DIR
    / "proposed_validation_samples.csv"
)

TRAIN_STUDIES_CSV = (
    PHASE4M_DIR
    / "proposed_train_study_ids.csv"
)

VAL_STUDIES_CSV = (
    PHASE4M_DIR
    / "proposed_validation_study_ids.csv"
)


# ============================================================================
# EXPECTED CONFIGURATION
# ============================================================================

CLASS_NAMES = [
    "Normal/Mild",
    "Moderate",
    "Severe",
]

EXPECTED_TRAIN_SAMPLES = 38925
EXPECTED_VAL_SAMPLES = 9732

EXPECTED_TRAIN_STUDIES = 1580
EXPECTED_VAL_STUDIES = 394

EXPECTED_SEED = 42

EXPECTED_WEIGHTS = np.array(
    [
        0.167594,
        0.788787,
        2.043618,
    ],
    dtype=np.float64,
)


# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

OUTPUT_DIR = (
    OUTPUTS_DIR
    / "phaseA_classification"
    / "part2_corrected_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "phaseA_part2_corrected_metrics.csv"
)

REPORT_JSON = (
    OUTPUT_DIR
    / "phaseA_part2_corrected_evaluation.json"
)

COMPARISON_CSV = (
    OUTPUT_DIR
    / "phaseA_part2_model_comparison.csv"
)


# ============================================================================
# UTILITIES
# ============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def status(label: str, value: Any) -> None:
    print(
        f"{label:<45}: {value}"
    )


def sha256_file(path: Path) -> Optional[str]:

    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_checkpoint(path: Path) -> Dict[str, Any]:

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Checkpoint is not a dictionary."
        )

    return checkpoint


def numeric_value(
    checkpoint: Dict[str, Any],
    key: str,
) -> Optional[float]:

    if key not in checkpoint:
        return None

    value = checkpoint[key]

    if isinstance(
        value,
        torch.Tensor,
    ):

        value = (
            value
            .detach()
            .cpu()
            .item()
        )

    try:
        return float(value)
    except Exception:
        return None


def print_metric(
    name: str,
    value: Optional[float],
) -> None:

    if value is None:

        status(
            name,
            "NOT AVAILABLE",
        )

    else:

        status(
            name,
            f"{value:.6f}",
        )


# ============================================================================
# START
# ============================================================================

banner("PHASE A - PART 2")

print(
    "CORRECTED CLASSIFIER EVALUATION"
)

print()

status(
    "Project root",
    PROJECT_ROOT,
)

status(
    "Corrected checkpoint",
    CORRECTED_CHECKPOINT,
)

status(
    "Device",
    "CUDA available"
    if torch.cuda.is_available()
    else "CPU",
)

status(
    "PyTorch",
    torch.__version__,
)


# ============================================================================
# 1. PATH CHECK
# ============================================================================

banner("1. PATH VALIDATION")

required_paths = {
    "Corrected checkpoint":
        CORRECTED_CHECKPOINT,

    "Original checkpoint":
        ORIGINAL_CHECKPOINT,

    "Train samples":
        TRAIN_CSV,

    "Validation samples":
        VAL_CSV,

    "Train studies":
        TRAIN_STUDIES_CSV,

    "Validation studies":
        VAL_STUDIES_CSV,
}

path_status = {}

for name, path in required_paths.items():

    exists = path.exists()

    path_status[name] = exists

    status(
        name,
        "EXISTS"
        if exists
        else "MISSING",
    )

if not CORRECTED_CHECKPOINT.exists():

    raise FileNotFoundError(
        "Corrected checkpoint is missing:\n"
        f"{CORRECTED_CHECKPOINT}"
    )


# ============================================================================
# 2. CHECKPOINT INTEGRITY
# ============================================================================

banner("2. CHECKPOINT INTEGRITY")

corrected_sha = sha256_file(
    CORRECTED_CHECKPOINT
)

original_sha = sha256_file(
    ORIGINAL_CHECKPOINT
)

status(
    "Corrected checkpoint SHA256",
    corrected_sha,
)

status(
    "Original checkpoint SHA256",
    original_sha,
)

if (
    corrected_sha is not None
    and original_sha is not None
):

    status(
        "Checkpoint files differ",
        corrected_sha != original_sha,
    )


# ============================================================================
# 3. LOAD CORRECTED CHECKPOINT
# ============================================================================

banner("3. CORRECTED CHECKPOINT METADATA")

checkpoint = load_checkpoint(
    CORRECTED_CHECKPOINT
)

status(
    "Checkpoint keys",
    len(checkpoint),
)

print()

for key in sorted(
    checkpoint.keys(),
):

    value = checkpoint[key]

    if isinstance(
        value,
        torch.Tensor,
    ):

        description = (
            f"Tensor shape={tuple(value.shape)}"
        )

    elif isinstance(
        value,
        dict,
    ):

        description = (
            f"dict({len(value)} items)"
        )

    elif isinstance(
        value,
        (list, tuple),
    ):

        description = (
            f"{type(value).__name__}"
            f"({len(value)} items)"
        )

    else:

        description = repr(value)

    print(
        f"  {key:<35} {description}"
    )


# ============================================================================
# 4. STORED VALIDATION METRICS
# ============================================================================

banner("4. STORED VALIDATION METRICS")

metric_keys = [
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_macro_precision",
    "validation_macro_recall",
    "validation_macro_f1",
]

metrics = {}

for key in metric_keys:

    value = numeric_value(
        checkpoint,
        key,
    )

    metrics[key] = value

    print_metric(
        key,
        value,
    )


# ============================================================================
# 5. TRAINING / CHECKPOINT METADATA
# ============================================================================

banner("5. TRAINING METADATA")

metadata_keys = [
    "epoch",
    "split_seed",
    "study_overlap",
    "train_studies",
    "validation_studies",
    "experiment",
    "class_names",
    "class_weights",
]

metadata = {}

for key in metadata_keys:

    value = checkpoint.get(
        key,
        None,
    )

    metadata[key] = value

    if key == "class_weights":

        print(
            f"{key:<45}: {value}"
        )

    else:

        status(
            key,
            value,
        )


# ============================================================================
# 6. STUDY SPLIT VALIDATION
# ============================================================================

banner("6. STUDY-LEVEL SPLIT VERIFICATION")

train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)

train_studies_df = pd.read_csv(
    TRAIN_STUDIES_CSV
)

val_studies_df = pd.read_csv(
    VAL_STUDIES_CSV
)

train_studies = set(
    train_studies_df[
        "study_id"
    ]
    .astype(int)
)

val_studies = set(
    val_studies_df[
        "study_id"
    ]
    .astype(int)
)

overlap = (
    train_studies
    & val_studies
)

split_checks = {

    "train_samples":
        len(train_df)
        == EXPECTED_TRAIN_SAMPLES,

    "validation_samples":
        len(val_df)
        == EXPECTED_VAL_SAMPLES,

    "train_studies":
        len(train_studies)
        == EXPECTED_TRAIN_STUDIES,

    "validation_studies":
        len(val_studies)
        == EXPECTED_VAL_STUDIES,

    "zero_study_overlap":
        len(overlap) == 0,

}

for name, passed in split_checks.items():

    status(
        name,
        "PASS"
        if passed
        else "FAIL",
    )

split_pass = all(
    split_checks.values()
)


# ============================================================================
# 7. CHECKPOINT SPLIT METADATA
# ============================================================================

banner("7. CHECKPOINT SPLIT METADATA")

checkpoint_split_checks = {

    "seed":
        checkpoint.get(
            "split_seed"
        )
        == EXPECTED_SEED,

    "train_studies":
        checkpoint.get(
            "train_studies"
        )
        == EXPECTED_TRAIN_STUDIES,

    "validation_studies":
        checkpoint.get(
            "validation_studies"
        )
        == EXPECTED_VAL_STUDIES,

    "study_overlap":
        checkpoint.get(
            "study_overlap"
        )
        == 0,

}

for name, passed in checkpoint_split_checks.items():

    status(
        name,
        "PASS"
        if passed
        else "FAIL",
    )

checkpoint_split_pass = all(
    checkpoint_split_checks.values()
)


# ============================================================================
# 8. CLASS-WEIGHT VERIFICATION
# ============================================================================

banner("8. CLASS-WEIGHT VERIFICATION")

checkpoint_weights = checkpoint.get(
    "class_weights"
)

weights_match = False

if checkpoint_weights is not None:

    try:

        checkpoint_weights_array = np.asarray(
            checkpoint_weights,
            dtype=np.float64,
        )

        weights_match = np.allclose(
            checkpoint_weights_array,
            EXPECTED_WEIGHTS,
            rtol=1e-4,
            atol=1e-5,
        )

        print(
            "Expected weights:"
        )

        print(
            EXPECTED_WEIGHTS
        )

        print()

        print(
            "Checkpoint weights:"
        )

        print(
            checkpoint_weights_array
        )

    except Exception as exc:

        print(
            f"Could not compare weights: {exc}"
        )

status(
    "Class weights match expected",
    "PASS"
    if weights_match
    else "FAIL",
)


# ============================================================================
# 9. CLASS-NAME VERIFICATION
# ============================================================================

banner("9. CLASS-NAME VERIFICATION")

checkpoint_class_names = checkpoint.get(
    "class_names"
)

class_names_match = (
    checkpoint_class_names
    == CLASS_NAMES
)

status(
    "Class names",
    checkpoint_class_names,
)

status(
    "Class names match expected",
    "PASS"
    if class_names_match
    else "FAIL",
)


# ============================================================================
# 10. CORRECTED MODEL QUALITY ANALYSIS
# ============================================================================

banner("10. CORRECTED MODEL QUALITY ANALYSIS")

accuracy = metrics[
    "validation_accuracy"
]

balanced_accuracy = metrics[
    "validation_balanced_accuracy"
]

macro_precision = metrics[
    "validation_macro_precision"
]

macro_recall = metrics[
    "validation_macro_recall"
]

macro_f1 = metrics[
    "validation_macro_f1"
]

print()

print(
    "The following metrics are evaluated:"
)

print(
    "  Accuracy              = overall correctness"
)

print(
    "  Balanced Accuracy     = average class recall"
)

print(
    "  Macro Precision       = equal class importance"
)

print(
    "  Macro Recall          = equal class importance"
)

print(
    "  Macro F1              = balanced class performance"
)


# ============================================================================
# 11. STRENGTH THRESHOLDS
# ============================================================================

banner("11. PROJECT-QUALITY STRENGTH CHECK")

thresholds = {

    "accuracy":
        0.75,

    "balanced_accuracy":
        0.65,

    "macro_precision":
        0.65,

    "macro_recall":
        0.65,

    "macro_f1":
        0.65,

}

threshold_results = {}

metric_mapping = {

    "accuracy":
        accuracy,

    "balanced_accuracy":
        balanced_accuracy,

    "macro_precision":
        macro_precision,

    "macro_recall":
        macro_recall,

    "macro_f1":
        macro_f1,

}

for name, threshold in thresholds.items():

    value = metric_mapping[name]

    passed = (
        value is not None
        and value >= threshold
    )

    threshold_results[name] = passed

    if value is None:

        print(
            f"{name:<25} "
            f"NOT AVAILABLE"
        )

    else:

        print(
            f"{name:<25} "
            f"{value:.6f} "
            f">= {threshold:.2f} "
            f"-> "
            f"{'PASS' if passed else 'FAIL'}"
        )


# ============================================================================
# 12. MINORITY CLASS LEARNING TEST
# ============================================================================

banner("12. MINORITY-CLASS LEARNING TEST")

# The checkpoint stores macro metrics but not
# necessarily the full per-class report.
#
# Therefore this part deliberately does NOT invent
# per-class results.
#
# If the macro metrics are substantially above zero,
# that is evidence that the corrected model is no longer
# behaving like the historical all-Normal/Mild classifier.
#
# However, a final selection still requires direct
# per-class validation evidence.

minority_learning_supported = False

if (
    macro_f1 is not None
    and macro_f1 > 0.40
):

    minority_learning_supported = True

status(
    "Macro-F1 evidence of minority learning",
    (
        "YES"
        if minority_learning_supported
        else "NO / INSUFFICIENT"
    ),
)


# ============================================================================
# 13. VERDICT
# ============================================================================

banner("13. FINAL CLASSIFICATION VERDICT")

all_required_metrics_available = all(
    value is not None
    for value in [
        accuracy,
        balanced_accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
    ]
)

all_quality_thresholds_pass = all(
    threshold_results.values()
)

if (
    all_required_metrics_available
    and all_quality_thresholds_pass
    and split_pass
    and checkpoint_split_pass
    and weights_match
    and class_names_match
):

    verdict = (
        "STRONG_ENOUGH"
    )

elif (
    all_required_metrics_available
    and minority_learning_supported
    and split_pass
    and checkpoint_split_pass
):

    verdict = (
        "PROMISING_BUT_NEEDS_TARGETED_IMPROVEMENT"
    )

else:

    verdict = (
        "NEEDS_TARGETED_RETRAINING"
    )


status(
    "FINAL VERDICT",
    verdict,
)


# ============================================================================
# 14. RECOMMENDATION
# ============================================================================

banner("14. RECOMMENDATION")

if verdict == "STRONG_ENOUGH":

    recommendation = (
        "The corrected classifier meets the current project-quality "
        "thresholds. Do not retrain blindly. Proceed to final "
        "independent evaluation and model freezing."
    )

elif verdict == "PROMISING_BUT_NEEDS_TARGETED_IMPROVEMENT":

    recommendation = (
        "The corrected classifier is learning beyond the historical "
        "majority-class collapse, but its balanced metrics are not "
        "yet strong enough. Perform ONE targeted classification "
        "improvement experiment."
    )

else:

    recommendation = (
        "The corrected classifier is not strong enough. Perform "
        "targeted retraining using the validated study-level split "
        "and imbalance-aware training."
    )

print(
    recommendation
)


# ============================================================================
# 15. MODEL COMPARISON
# ============================================================================

banner("15. ORIGINAL VS CORRECTED")

comparison_rows = [

    {
        "model":
            "Original",
        "accuracy":
            None,
        "balanced_accuracy":
            0.333333,
        "macro_f1":
            0.289040,
        "status":
            "Majority-class collapse",
    },

    {
        "model":
            "Corrected",
        "accuracy":
            accuracy,
        "balanced_accuracy":
            balanced_accuracy,
        "macro_f1":
            macro_f1,
        "status":
            verdict,
    },

]

comparison_df = pd.DataFrame(
    comparison_rows
)

print()

print(
    comparison_df.to_string(
        index=False
    )
)

comparison_df.to_csv(
    COMPARISON_CSV,
    index=False,
)


# ============================================================================
# 16. SAVE METRICS
# ============================================================================

banner("16. SAVE REPORTS")

summary_rows = [

    {
        "metric":
            "validation_accuracy",
        "value":
            accuracy,
    },

    {
        "metric":
            "validation_balanced_accuracy",
        "value":
            balanced_accuracy,
    },

    {
        "metric":
            "validation_macro_precision",
        "value":
            macro_precision,
    },

    {
        "metric":
            "validation_macro_recall",
        "value":
            macro_recall,
    },

    {
        "metric":
            "validation_macro_f1",
        "value":
            macro_f1,
    },

    {
        "metric":
            "study_overlap",
        "value":
            len(overlap),
    },

    {
        "metric":
            "verdict",
        "value":
            verdict,
    },

]

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================================
# 17. JSON REPORT
# ============================================================================

report = {

    "phase":
        "A",

    "part":
        "2",

    "purpose":
        "Corrected classifier evaluation",

    "checkpoint":
        str(
            CORRECTED_CHECKPOINT
        ),

    "checkpoint_sha256":
        corrected_sha,

    "metrics":
        metrics,

    "training_metadata":
        metadata,

    "study_split":
        {
            "train_samples":
                len(train_df),

            "validation_samples":
                len(val_df),

            "train_studies":
                len(train_studies),

            "validation_studies":
                len(val_studies),

            "study_overlap":
                len(overlap),

            "pass":
                split_pass,
        },

    "checkpoint_split_validation":
        checkpoint_split_checks,

    "class_weights_match":
        weights_match,

    "class_names_match":
        class_names_match,

    "threshold_results":
        threshold_results,

    "minority_learning_supported":
        minority_learning_supported,

    "verdict":
        verdict,

    "recommendation":
        recommendation,

}

with REPORT_JSON.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
        default=str,
    )


# ============================================================================
# FINAL
# ============================================================================

banner("PHASE A - PART 2 COMPLETE")

print(
    "PASS — CORRECTED CLASSIFIER EVALUATION COMPLETED"
)

print()

print(
    f"Final verdict: {verdict}"
)

print()

print(
    "Corrected checkpoint was NOT modified."
)

print(
    "Original best_model.pth was NOT modified."
)

print()

print(
    "Reports:"
)

print(
    SUMMARY_CSV
)

print(
    COMPARISON_CSV
)

print(
    REPORT_JSON
)

print()

print(
    "IMPORTANT:"
)

print(
    "Accuracy alone is NOT used to declare the classifier strong."
)

print(
    "Balanced accuracy and macro-F1 are required because the "
    "historical classifier collapsed onto Normal/Mild."
)