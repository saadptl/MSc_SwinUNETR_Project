"""
PHASE A - PART 1
CLASSIFICATION STRENGTH AUDIT

Purpose
-------
Read-only audit of the existing RSNA lumbar-spine classification branch.

This part:
1. Verifies the project and dataset paths.
2. Verifies the original and corrected checkpoints.
3. Verifies the validated study-level split.
4. Reads existing classification metrics when available.
5. Reads the corrected checkpoint metadata.
6. Evaluates whether minority classes are actually being learned.
7. Produces a clear classification-strength verdict.

IMPORTANT
---------
- NO TRAINING
- NO MODEL WEIGHTS MODIFIED
- NO CHECKPOINT OVERWRITTEN
- NO DATASET MODIFIED

Project root is derived automatically from this file location.
Expected project structure:

MSc_SwinUNETR_Project/
    dataset/
    models/
    outputs/
    src/
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PHASE4M_DIR = OUTPUTS_DIR / "phase4m"

ORIGINAL_CHECKPOINT = MODELS_DIR / "best_model.pth"

CORRECTED_CHECKPOINT = (
    MODELS_DIR / "corrected_study_split_weighted_best_model.pth"
)

# Existing historical metric files
TRAINING_VALIDATION_SUMMARY = (
    OUTPUTS_DIR / "training_validation_summary.csv"
)

FINAL_TRAINING_REPORT = (
    OUTPUTS_DIR / "final_training_report.csv"
)

EVALUATION_METRICS = (
    OUTPUTS_DIR / "evaluation_metrics.csv"
)

CLASSIFICATION_REPORT = (
    OUTPUTS_DIR / "classification_report.csv"
)

# Corrected experiment outputs
PHASE4P_DIR = OUTPUTS_DIR / "phase4p"

CORRECTED_METRICS_CANDIDATES = [
    PHASE4P_DIR / "validation_metrics.csv",
    PHASE4P_DIR / "classification_metrics.csv",
    PHASE4P_DIR / "final_metrics.csv",
    PHASE4P_DIR / "training_history.csv",
    PHASE4P_DIR / "classification_report.csv",
]

TRAIN_CSV = (
    PHASE4M_DIR / "proposed_train_samples.csv"
)

VAL_CSV = (
    PHASE4M_DIR / "proposed_validation_samples.csv"
)

TRAIN_STUDIES_CSV = (
    PHASE4M_DIR / "proposed_train_study_ids.csv"
)

VAL_STUDIES_CSV = (
    PHASE4M_DIR / "proposed_validation_study_ids.csv"
)

# ============================================================================
# EXPECTED VALUES
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

EXPECTED_WEIGHTS = np.array(
    [
        0.167594,
        0.788787,
        2.043618,
    ],
    dtype=np.float64,
)

# ============================================================================
# OUTPUT
# ============================================================================

AUDIT_DIR = (
    OUTPUTS_DIR
    / "phaseA_classification"
    / "part1_strength_audit"
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_JSON = AUDIT_DIR / "phaseA_part1_audit_report.json"
REPORT_CSV = AUDIT_DIR / "phaseA_part1_summary.csv"


# ============================================================================
# UTILITIES
# ============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def status(label: str, value: Any) -> None:
    print(f"{label:<42}: {value}")


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        return None


def load_checkpoint(path: Path) -> Optional[Any]:
    if not path.exists():
        return None

    try:
        return torch.load(
            path,
            map_location="cpu",
        )
    except Exception as exc:
        print(f"WARNING: Could not load checkpoint {path}: {exc}")
        return None


def find_metric_value(
    df: Optional[pd.DataFrame],
    names,
) -> Optional[float]:

    if df is None or df.empty:
        return None

    normalized = {
        str(x).strip().lower(): x
        for x in names
    }

    # Format A:
    # Metric | Value
    if "Metric" in df.columns and "Value" in df.columns:

        for _, row in df.iterrows():

            metric = str(
                row["Metric"]
            ).strip().lower()

            for wanted in normalized:

                if metric == wanted:
                    try:
                        return float(row["Value"])
                    except Exception:
                        pass

    # Format B:
    # metric_name | value
    if "metric" in df.columns and "value" in df.columns:

        for _, row in df.iterrows():

            metric = str(
                row["metric"]
            ).strip().lower()

            for wanted in normalized:

                if metric == wanted:
                    try:
                        return float(row["value"])
                    except Exception:
                        pass

    return None


def checkpoint_metadata(
    checkpoint: Any,
) -> Dict[str, Any]:

    result: Dict[str, Any] = {}

    if not isinstance(checkpoint, dict):
        result["format"] = type(checkpoint).__name__
        return result

    for key in [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "precision",
        "recall",
        "f1_score",
        "best_epoch",
        "best_accuracy",
        "best_validation_accuracy",
    ]:

        if key in checkpoint:

            value = checkpoint[key]

            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()

            try:
                if isinstance(value, np.generic):
                    value = value.item()
            except Exception:
                pass

            result[key] = value

    result["keys"] = sorted(
        [str(k) for k in checkpoint.keys()]
    )

    return result


# ============================================================================
# START
# ============================================================================

banner("PHASE A - PART 1")
print("CLASSIFICATION STRENGTH AUDIT")
print()

status("Project root", PROJECT_ROOT)
status("Dataset root", DATA_ROOT)
status("Device", "CUDA" if torch.cuda.is_available() else "CPU")
status(
    "PyTorch",
    torch.__version__,
)

# ============================================================================
# 1. PATH VALIDATION
# ============================================================================

banner("1. PATH VALIDATION")

paths_to_check = {
    "Dataset root": DATA_ROOT,
    "Train images": DATA_ROOT / "train_images",
    "Models directory": MODELS_DIR,
    "Outputs directory": OUTPUTS_DIR,
    "Original checkpoint": ORIGINAL_CHECKPOINT,
    "Corrected checkpoint": CORRECTED_CHECKPOINT,
    "Phase4M directory": PHASE4M_DIR,
    "Phase4P directory": PHASE4P_DIR,
    "Train samples CSV": TRAIN_CSV,
    "Validation samples CSV": VAL_CSV,
    "Train study IDs CSV": TRAIN_STUDIES_CSV,
    "Validation study IDs CSV": VAL_STUDIES_CSV,
}

path_results = {}

for name, path in paths_to_check.items():

    exists = path.exists()

    path_results[name] = exists

    status(
        name,
        "EXISTS" if exists else "MISSING",
    )

# ============================================================================
# 2. CHECKPOINT HASHES
# ============================================================================

banner("2. CHECKPOINT INTEGRITY")

original_sha = sha256_file(
    ORIGINAL_CHECKPOINT
)

corrected_sha = sha256_file(
    CORRECTED_CHECKPOINT
)

status(
    "Original checkpoint SHA256",
    original_sha if original_sha else "MISSING",
)

status(
    "Corrected checkpoint SHA256",
    corrected_sha if corrected_sha else "MISSING",
)

if (
    original_sha is not None
    and corrected_sha is not None
):

    status(
        "Original != Corrected",
        original_sha != corrected_sha,
    )

# ============================================================================
# 3. STUDY-LEVEL SPLIT AUDIT
# ============================================================================

banner("3. STUDY-LEVEL SPLIT AUDIT")

train_df = safe_read_csv(TRAIN_CSV)
val_df = safe_read_csv(VAL_CSV)

train_studies_df = safe_read_csv(
    TRAIN_STUDIES_CSV
)

val_studies_df = safe_read_csv(
    VAL_STUDIES_CSV
)

split_pass = True

if train_df is not None:

    status(
        "Training samples",
        len(train_df),
    )

    if len(train_df) != EXPECTED_TRAIN_SAMPLES:
        split_pass = False

else:

    status(
        "Training samples",
        "UNAVAILABLE",
    )

    split_pass = False


if val_df is not None:

    status(
        "Validation samples",
        len(val_df),
    )

    if len(val_df) != EXPECTED_VAL_SAMPLES:
        split_pass = False

else:

    status(
        "Validation samples",
        "UNAVAILABLE",
    )

    split_pass = False


if train_studies_df is not None:

    train_studies = set(
        train_studies_df["study_id"]
        .astype(int)
        .tolist()
    )

    status(
        "Training studies",
        len(train_studies),
    )

    if len(train_studies) != EXPECTED_TRAIN_STUDIES:
        split_pass = False

else:

    train_studies = set()

    status(
        "Training studies",
        "UNAVAILABLE",
    )

    split_pass = False


if val_studies_df is not None:

    val_studies = set(
        val_studies_df["study_id"]
        .astype(int)
        .tolist()
    )

    status(
        "Validation studies",
        len(val_studies),
    )

    if len(val_studies) != EXPECTED_VAL_STUDIES:
        split_pass = False

else:

    val_studies = set()

    status(
        "Validation studies",
        "UNAVAILABLE",
    )

    split_pass = False


overlap = train_studies & val_studies

status(
    "Study overlap",
    len(overlap),
)

if len(overlap) != 0:
    split_pass = False

status(
    "Study-level split",
    "PASS" if split_pass else "FAIL",
)

# ============================================================================
# 4. CLASS DISTRIBUTION
# ============================================================================

banner("4. CLASS DISTRIBUTION")

class_distribution = {}

if train_df is not None and "label" in train_df.columns:

    train_counts = (
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("TRAINING DISTRIBUTION")

    for class_id in range(3):

        count = int(
            train_counts.get(
                class_id,
                0,
            )
        )

        class_distribution[
            f"train_class_{class_id}"
        ] = count

        print(
            f"  {class_id} - "
            f"{CLASS_NAMES[class_id]:<15} "
            f"{count:>8}"
        )


if val_df is not None and "label" in val_df.columns:

    val_counts = (
        val_df["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("VALIDATION DISTRIBUTION")

    for class_id in range(3):

        count = int(
            val_counts.get(
                class_id,
                0,
            )
        )

        class_distribution[
            f"val_class_{class_id}"
        ] = count

        print(
            f"  {class_id} - "
            f"{CLASS_NAMES[class_id]:<15} "
            f"{count:>8}"
        )

# ============================================================================
# 5. ORIGINAL HISTORICAL METRICS
# ============================================================================

banner("5. ORIGINAL CLASSIFICATION METRICS")

original_summary = safe_read_csv(
    TRAINING_VALIDATION_SUMMARY
)

original_eval = safe_read_csv(
    EVALUATION_METRICS
)

original_report = safe_read_csv(
    CLASSIFICATION_REPORT
)

original_metrics = {}

metric_aliases = {
    "accuracy": [
        "accuracy",
        "validation accuracy",
    ],
    "precision": [
        "precision",
    ],
    "recall": [
        "recall",
    ],
    "f1": [
        "f1 score",
        "f1-score",
        "f1",
    ],
    "balanced_accuracy": [
        "balanced accuracy",
    ],
    "cohen_kappa": [
        "cohen kappa",
    ],
    "matthews_correlation": [
        "matthews correlation",
    ],
}

for metric_name, aliases in metric_aliases.items():

    value = (
        find_metric_value(
            original_eval,
            aliases,
        )
    )

    if value is None:

        value = (
            find_metric_value(
                original_summary,
                aliases,
            )
        )

    original_metrics[
        metric_name
    ] = value

    status(
        metric_name,
        (
            f"{value:.6f}"
            if value is not None
            else "NOT FOUND"
        ),
    )

# ============================================================================
# 6. ORIGINAL PER-CLASS REPORT
# ============================================================================

banner("6. ORIGINAL PER-CLASS PERFORMANCE")

per_class_original = {}

if original_report is not None:

    report_df = original_report.copy()

    print()

    for _, row in report_df.iterrows():

        label = str(
            row.iloc[0]
        )

        if label in CLASS_NAMES:

            precision = float(
                row["precision"]
            )

            recall = float(
                row["recall"]
            )

            f1 = float(
                row["f1-score"]
            )

            support = int(
                float(row["support"])
            )

            per_class_original[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }

            print(
                f"{label:<15} "
                f"Precision={precision:.4f} "
                f"Recall={recall:.4f} "
                f"F1={f1:.4f} "
                f"Support={support}"
            )

else:

    print("Classification report not found.")

# ============================================================================
# 7. CORRECTED CHECKPOINT
# ============================================================================

banner("7. CORRECTED CHECKPOINT")

corrected_checkpoint = load_checkpoint(
    CORRECTED_CHECKPOINT
)

corrected_meta = checkpoint_metadata(
    corrected_checkpoint
)

if corrected_checkpoint is None:

    status(
        "Corrected checkpoint",
        "UNAVAILABLE",
    )

else:

    status(
        "Checkpoint format",
        (
            corrected_meta.get(
                "format",
                "dictionary",
            )
        ),
    )

    for key, value in corrected_meta.items():

        if key == "keys":
            continue

        status(
            key,
            value,
        )

    if "keys" in corrected_meta:

        print()
        print("Checkpoint keys:")

        for key in corrected_meta["keys"]:

            print(
                f"  - {key}"
            )

# ============================================================================
# 8. CORRECTED EXPERIMENT FILE DISCOVERY
# ============================================================================

banner("8. CORRECTED EXPERIMENT OUTPUTS")

corrected_files = []

if PHASE4P_DIR.exists():

    for path in sorted(
        PHASE4P_DIR.rglob("*")
    ):

        if path.is_file():

            corrected_files.append(
                str(path)
            )

            print(
                f"FOUND: {path}"
            )

else:

    print(
        "Phase4P output directory not found."
    )

# ============================================================================
# 9. DETERMINE CORRECTED METRICS FROM AVAILABLE FILES
# ============================================================================

banner("9. CORRECTED METRIC SEARCH")

corrected_metrics = {}

for candidate in CORRECTED_METRICS_CANDIDATES:

    if not candidate.exists():
        continue

    df = safe_read_csv(candidate)

    if df is None:
        continue

    print()
    print(
        f"Reading: {candidate}"
    )

    for metric_name, aliases in metric_aliases.items():

        value = find_metric_value(
            df,
            aliases,
        )

        if value is not None:

            corrected_metrics[
                metric_name
            ] = value

            print(
                f"  {metric_name:<25} "
                f"{value:.6f}"
            )

# Also inspect checkpoint fields.
checkpoint_metric_mapping = {
    "accuracy": [
        "validation_accuracy",
        "best_accuracy",
        "best_validation_accuracy",
    ],
    "precision": [
        "precision",
    ],
    "recall": [
        "recall",
    ],
    "f1": [
        "f1_score",
    ],
}

for metric_name, keys in checkpoint_metric_mapping.items():

    if metric_name in corrected_metrics:
        continue

    for key in keys:

        if key in corrected_meta:

            try:

                corrected_metrics[
                    metric_name
                ] = float(
                    corrected_meta[key]
                )

                break

            except Exception:
                pass

# ============================================================================
# 10. VERDICT LOGIC
# ============================================================================

banner("10. CLASSIFICATION STRENGTH VERDICT")

accuracy = corrected_metrics.get(
    "accuracy"
)

balanced_accuracy = corrected_metrics.get(
    "balanced_accuracy"
)

macro_f1 = corrected_metrics.get(
    "macro_f1"
)

# Try to discover macro F1 from report.
if macro_f1 is None and original_report is not None:

    for _, row in original_report.iterrows():

        label = str(
            row.iloc[0]
        ).strip().lower()

        if label == "macro avg":

            try:
                macro_f1 = float(
                    row["f1-score"]
                )
            except Exception:
                pass


# Historical original model is known to be collapsed.
original_collapsed = False

if per_class_original:

    moderate_recall = (
        per_class_original
        .get("Moderate", {})
        .get("recall", 0.0)
    )

    severe_recall = (
        per_class_original
        .get("Severe", {})
        .get("recall", 0.0)
    )

    if (
        moderate_recall == 0.0
        and severe_recall == 0.0
    ):
        original_collapsed = True


# Conservative verdict.
#
# We deliberately do NOT call a model strong based on accuracy alone.
#
# Strong enough requires evidence that minority classes are being learned.
#
# Thresholds are project-quality thresholds, not clinical thresholds.

if corrected_metrics:

    # If we have macro-F1 and balanced accuracy,
    # use both.
    if (
        macro_f1 is not None
        and balanced_accuracy is not None
    ):

        if (
            macro_f1 >= 0.70
            and balanced_accuracy >= 0.70
        ):

            verdict = (
                "STRONG_ENOUGH"
            )

        elif (
            macro_f1 >= 0.50
            and balanced_accuracy >= 0.50
        ):

            verdict = (
                "PROMISING_BUT_NEEDS_TARGETED_IMPROVEMENT"
            )

        else:

            verdict = (
                "NEEDS_TARGETED_RETRAINING"
            )

    elif accuracy is not None:

        # Accuracy alone is never sufficient.
        verdict = (
            "NEEDS_TARGETED_RETRAINING"
        )

    else:

        verdict = (
            "INSUFFICIENT_METRICS"
        )

else:

    verdict = (
        "CORRECTED_METRICS_NOT_AVAILABLE"
    )


status(
    "Historical original model",
    (
        "CLASS_COLLAPSE"
        if original_collapsed
        else "NOT_DETERMINED"
    ),
)

status(
    "Corrected accuracy",
    (
        f"{accuracy:.6f}"
        if accuracy is not None
        else "NOT AVAILABLE"
    ),
)

status(
    "Corrected balanced accuracy",
    (
        f"{balanced_accuracy:.6f}"
        if balanced_accuracy is not None
        else "NOT AVAILABLE"
    ),
)

status(
    "Corrected macro F1",
    (
        f"{macro_f1:.6f}"
        if macro_f1 is not None
        else "NOT AVAILABLE"
    ),
)

status(
    "FINAL VERDICT",
    verdict,
)

# ============================================================================
# 11. RECOMMENDATION
# ============================================================================

banner("11. PHASE A RECOMMENDATION")

if verdict == "STRONG_ENOUGH":

    recommendation = (
        "Classification branch is strong enough for the current "
        "MSc project standard. Freeze the checkpoint after a "
        "final independent validation audit."
    )

elif verdict == "PROMISING_BUT_NEEDS_TARGETED_IMPROVEMENT":

    recommendation = (
        "Classification has learned minority classes but is not "
        "yet strong enough. Perform one targeted improvement "
        "experiment using the validated study-level split."
    )

elif verdict == "NEEDS_TARGETED_RETRAINING":

    recommendation = (
        "Classification is not strong enough. Do not use the "
        "current checkpoint as the final classifier. Perform "
        "targeted retraining using the validated study-level "
        "split and imbalance-aware training."
    )

elif verdict == "CORRECTED_METRICS_NOT_AVAILABLE":

    recommendation = (
        "Corrected checkpoint exists but its evaluation metrics "
        "were not found in the expected outputs. Run a dedicated "
        "corrected-model evaluation before retraining."
    )

else:

    recommendation = (
        "The available evidence is insufficient to select a final "
        "classification model."
    )

print(recommendation)

# ============================================================================
# 12. SAVE REPORT
# ============================================================================

banner("12. SAVE AUDIT REPORT")

summary_rows = [
    {
        "item": "project_root",
        "value": str(PROJECT_ROOT),
    },
    {
        "item": "study_split",
        "value": (
            "PASS"
            if split_pass
            else "FAIL"
        ),
    },
    {
        "item": "study_overlap",
        "value": len(overlap),
    },
    {
        "item": "original_checkpoint_sha256",
        "value": original_sha,
    },
    {
        "item": "corrected_checkpoint_sha256",
        "value": corrected_sha,
    },
    {
        "item": "original_accuracy",
        "value": original_metrics.get(
            "accuracy"
        ),
    },
    {
        "item": "original_balanced_accuracy",
        "value": original_metrics.get(
            "balanced_accuracy"
        ),
    },
    {
        "item": "original_f1",
        "value": original_metrics.get(
            "f1"
        ),
    },
    {
        "item": "corrected_accuracy",
        "value": corrected_metrics.get(
            "accuracy"
        ),
    },
    {
        "item": "corrected_balanced_accuracy",
        "value": corrected_metrics.get(
            "balanced_accuracy"
        ),
    },
    {
        "item": "corrected_macro_f1",
        "value": corrected_metrics.get(
            "macro_f1"
        ),
    },
    {
        "item": "verdict",
        "value": verdict,
    },
    {
        "item": "recommendation",
        "value": recommendation,
    },
]

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    REPORT_CSV,
    index=False,
)

report = {
    "phase": "A",
    "part": "1",
    "purpose": "classification strength audit",
    "project_root": str(PROJECT_ROOT),
    "paths": path_results,
    "study_split": {
        "train_samples": (
            len(train_df)
            if train_df is not None
            else None
        ),
        "validation_samples": (
            len(val_df)
            if val_df is not None
            else None
        ),
        "train_studies": len(train_studies),
        "validation_studies": len(val_studies),
        "study_overlap": len(overlap),
        "pass": split_pass,
    },
    "checkpoint": {
        "original": {
            "path": str(
                ORIGINAL_CHECKPOINT
            ),
            "sha256": original_sha,
        },
        "corrected": {
            "path": str(
                CORRECTED_CHECKPOINT
            ),
            "sha256": corrected_sha,
            "metadata": corrected_meta,
        },
    },
    "original_metrics": original_metrics,
    "original_per_class": per_class_original,
    "corrected_metrics": corrected_metrics,
    "verdict": verdict,
    "recommendation": recommendation,
    "corrected_output_files": corrected_files,
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

status(
    "CSV report",
    REPORT_CSV,
)

status(
    "JSON report",
    REPORT_JSON,
)

# ============================================================================
# FINAL
# ============================================================================

banner("PHASE A - PART 1 COMPLETE")

print(
    "PASS — CLASSIFICATION STRENGTH AUDIT COMPLETED"
)

print()
print(
    "IMPORTANT:"
)
print(
    "The original 76.54% accuracy model is NOT considered strong "
    "because its Moderate and Severe recalls were 0%."
)

print()
print(
    "Next decision will be based on the corrected study-level "
    "classifier evidence, not accuracy alone."
)

print()
print(
    "No checkpoint was modified."
)