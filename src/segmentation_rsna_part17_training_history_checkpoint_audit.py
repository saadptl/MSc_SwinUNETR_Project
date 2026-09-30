"""
==============================================================================
PHASE 4 - PART 17
RSNA-ONLY PART 15 TRAINING HISTORY / CHECKPOINT AUDIT
==============================================================================

Purpose
-------
Audit the Part 15 extended controlled training run.

This stage DOES NOT:
    - train the model
    - update model weights
    - use SPIDER
    - use the RSNA test set
    - modify the Part 15 checkpoint

It verifies:
    1. Part 15 training history
    2. Best epoch and best validation Dice
    3. Best checkpoint metadata
    4. Training / validation cohort sizes
    5. Train-validation overlap
    6. Part 16 independent reproduction
    7. Model architecture compatibility
    8. Patch size / classes / seed consistency
    9. Whether the selected checkpoint is actually the best checkpoint
   10. Internal reproducibility of the Part 15 result

Important
---------
All Dice values are against RSNA point-derived pseudo-masks.
They are NOT manual clinical segmentation ground truth.
==============================================================================

"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
    / "rsna_part8_train_manifest.csv"
)

VAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
    / "rsna_part8_validation_manifest.csv"
)

PART11_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part11_controlled_pilot_training"
    / "checkpoints"
    / "best_model.pth"
)

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART15_CHECKPOINT = PART15_DIR / "checkpoints" / "best_model.pth"

PART15_SUMMARY = PART15_DIR / "phase4_part15_extended_training_summary.json"

PART15_HISTORY_CANDIDATES = [
    PART15_DIR / "part15_training_history.csv",
    PART15_DIR / "training_history.csv",
    PART15_DIR / "history.csv",
    PART15_DIR / "part15_history.csv",
]

PART15_COHORT_CANDIDATES = [
    PART15_DIR / "part15_training_cohort.csv",
    PART15_DIR / "part15_cohort.csv",
    PART15_DIR / "training_cohort.csv",
]

PART16_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part16_best_checkpoint_validation"
)

PART16_SUMMARY = (
    PART16_DIR
    / "phase4_part16_best_checkpoint_validation_summary.json"
)

PART16_CASE_METRICS = (
    PART16_DIR
    / "part16_validation_case_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part17_training_history_checkpoint_audit"
)

REPORT_DIR = OUTPUT_DIR / "reports"


# =============================================================================
# CONSTANTS
# =============================================================================

PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12
NUM_CLASSES = 6
BATCH_SIZE = 1
EXPECTED_TRAIN_CASES = 500
EXPECTED_VAL_CASES = 100
EXPECTED_SEED = 42

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# =============================================================================
# PRINT HELPERS
# =============================================================================

def header(title: str):
    print("=" * 78)
    print(title)
    print("=" * 78)


def section(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def status(label: str, value):
    print(f"{label:<35}: {value}")


# =============================================================================
# PATH VALIDATION
# =============================================================================

def require_paths():
    section("PATH VALIDATION")

    required = {
        "RSNA root": RSNA_ROOT,
        "train manifest": TRAIN_MANIFEST,
        "validation manifest": VAL_MANIFEST,
        "Part 15 directory": PART15_DIR,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 16 directory": PART16_DIR,
        "Part 16 summary": PART16_SUMMARY,
        "Part 16 case metrics": PART16_CASE_METRICS,
    }

    missing = []

    for name, path in required.items():
        exists = path.exists()
        status(name, "FOUND" if exists else "MISSING")

        if not exists:
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "Missing required Part 17 input(s):\n"
            + "\n".join(str(p) for p in missing)
        )


# =============================================================================
# GPU INFORMATION
# =============================================================================

def print_environment():
    section("PYTORCH / GPU ENVIRONMENT")

    status("PyTorch version", torch.__version__)
    status("CUDA available", torch.cuda.is_available())

    if torch.cuda.is_available():
        status("Device", torch.cuda.get_device_name(0))
        status(
            "GPU memory",
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB",
        )
    else:
        status("Device", "CPU")


# =============================================================================
# FIND HISTORY FILE
# =============================================================================

def find_history_file():
    for path in PART15_HISTORY_CANDIDATES:
        if path.exists():
            return path

    # Recursive fallback
    candidates = sorted(
        PART15_DIR.rglob("*.csv")
    )

    for path in candidates:
        name = path.name.lower()

        if (
            "history" in name
            or "training" in name
            or "epoch" in name
        ):
            return path

    return None


# =============================================================================
# FIND COHORT FILE
# =============================================================================

def find_cohort_file():
    for path in PART15_COHORT_CANDIDATES:
        if path.exists():
            return path

    candidates = sorted(PART15_DIR.rglob("*.csv"))

    for path in candidates:
        name = path.name.lower()

        if "cohort" in name:
            return path

    return None


# =============================================================================
# LOAD JSON SAFELY
# =============================================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# CHECKPOINT AUDIT
# =============================================================================

def audit_checkpoint():
    section("PART 15 CHECKPOINT METADATA AUDIT")

    checkpoint = torch.load(
        PART15_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        print("Checkpoint type : dictionary")

        keys = list(checkpoint.keys())

        print("Checkpoint keys:")
        for key in keys:
            print(f"  - {key}")

        epoch = checkpoint.get("epoch")
        best_dice = checkpoint.get("best_val_dice")

        seed = checkpoint.get("seed")
        patch_size = checkpoint.get("patch_size")
        classes = checkpoint.get("classes")
        feature_size = checkpoint.get("feature_size")

        print()

        status("Checkpoint epoch", epoch)
        status("Checkpoint best Dice", best_dice)
        status("Checkpoint seed", seed)
        status("Checkpoint patch size", patch_size)
        status("Checkpoint feature size", feature_size)
        status("Checkpoint classes", classes)

        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
        )

        parameter_tensors = None

        if isinstance(state_dict, dict):
            parameter_tensors = len(state_dict)

        status("Parameter tensors", parameter_tensors)

        return {
            "checkpoint_epoch": epoch,
            "checkpoint_best_dice": best_dice,
            "checkpoint_seed": seed,
            "checkpoint_patch_size": patch_size,
            "checkpoint_feature_size": feature_size,
            "checkpoint_classes": classes,
            "parameter_tensors": parameter_tensors,
            "checkpoint_keys": keys,
        }

    raise RuntimeError(
        "Unexpected Part 15 checkpoint format."
    )


# =============================================================================
# HISTORY AUDIT
# =============================================================================

def audit_history():
    section("PART 15 TRAINING HISTORY AUDIT")

    history_path = find_history_file()

    if history_path is None:
        print("No dedicated Part 15 history CSV found.")
        print(
            "The audit will use the Part 15 summary JSON and checkpoint "
            "metadata where available."
        )

        return None, {
            "history_file_found": False,
            "history_rows": 0,
        }

    print(f"History file : {history_path}")

    df = pd.read_csv(history_path)

    print()
    print("History columns:")
    for column in df.columns:
        print(f"  - {column}")

    print()
    print(f"History rows : {len(df)}")

    if len(df) > 0:
        print()
        print(df.to_string(index=False))

    # -------------------------------------------------------------
    # Identify epoch column
    # -------------------------------------------------------------

    epoch_col = None

    for candidate in [
        "epoch",
        "Epoch",
        "epoch_number",
    ]:
        if candidate in df.columns:
            epoch_col = candidate
            break

    # -------------------------------------------------------------
    # Identify validation Dice column
    # -------------------------------------------------------------

    dice_col = None

    preferred_dice_columns = [
        "val_dice",
        "validation_dice",
        "mean_val_dice",
        "best_val_dice",
        "dice",
        "val_mean_dice",
    ]

    for candidate in preferred_dice_columns:
        if candidate in df.columns:
            dice_col = candidate
            break

    # fallback
    if dice_col is None:
        for column in df.columns:
            lower = column.lower()

            if "dice" in lower and (
                "val" in lower
                or "validation" in lower
            ):
                dice_col = column
                break

    best_epoch = None
    best_dice = None

    if dice_col is not None:
        values = pd.to_numeric(
            df[dice_col],
            errors="coerce",
        )

        valid = values.dropna()

        if len(valid) > 0:
            best_index = valid.idxmax()
            best_dice = float(valid.loc[best_index])

            if epoch_col is not None:
                best_epoch = df.loc[best_index, epoch_col]

            print()
            status("Validation Dice column", dice_col)
            status("Best history Dice", f"{best_dice:.6f}")
            status("Best history epoch", best_epoch)

            print()
            print("Validation Dice by epoch:")

            for idx, row in df.iterrows():
                epoch = (
                    row[epoch_col]
                    if epoch_col is not None
                    else idx + 1
                )

                dice = pd.to_numeric(
                    row[dice_col],
                    errors="coerce",
                )

                if pd.notna(dice):
                    marker = "  <-- BEST" if idx == best_index else ""
                    print(
                        f"  Epoch {epoch}: "
                        f"{float(dice):.6f}{marker}"
                    )

    result = {
        "history_file_found": True,
        "history_file": str(history_path),
        "history_rows": int(len(df)),
        "epoch_column": epoch_col,
        "dice_column": dice_col,
        "best_history_epoch": best_epoch,
        "best_history_dice": best_dice,
    }

    return df, result


# =============================================================================
# SUMMARY JSON AUDIT
# =============================================================================

def audit_part15_summary():
    section("PART 15 SUMMARY JSON AUDIT")

    if not PART15_SUMMARY.exists():
        print("Part 15 summary JSON not found.")
        return {}

    data = load_json(PART15_SUMMARY)

    print("Part 15 summary loaded.")

    # Print useful fields without assuming a rigid schema.
    interesting = [
        "best_epoch",
        "best_val_dice",
        "source_epoch",
        "source_best_dice",
        "train_cases",
        "validation_cases",
        "epochs",
        "seed",
        "patch_size",
        "classes",
        "feature_size",
    ]

    print()

    extracted = {}

    for key in interesting:
        if key in data:
            value = data[key]
            extracted[key] = value
            status(key, value)

    return extracted


# =============================================================================
# COHORT AUDIT
# =============================================================================

def audit_cohort():
    section("PART 15 CONTROLLED COHORT AUDIT")

    train_df = pd.read_csv(TRAIN_MANIFEST)
    val_df = pd.read_csv(VAL_MANIFEST)

    print(f"Train manifest total      : {len(train_df)}")
    print(f"Validation manifest total : {len(val_df)}")

    cohort_file = find_cohort_file()

    if cohort_file is not None:
        print()
        print(f"Cohort file : {cohort_file}")

        cohort = pd.read_csv(cohort_file)

        print(f"Cohort rows : {len(cohort)}")

        return {
            "train_manifest_total": len(train_df),
            "validation_manifest_total": len(val_df),
            "cohort_file": str(cohort_file),
            "cohort_rows": len(cohort),
        }

    # -------------------------------------------------------------
    # Reconstruct deterministic cohort using the same seed
    # -------------------------------------------------------------

    print()
    print("No cohort CSV found.")
    print("Reconstructing deterministic controlled cohorts.")

    train_sample = (
        train_df
        .sample(
            n=min(EXPECTED_TRAIN_CASES, len(train_df)),
            random_state=EXPECTED_SEED,
        )
        .reset_index(drop=True)
    )

    val_sample = (
        val_df
        .sample(
            n=min(EXPECTED_VAL_CASES, len(val_df)),
            random_state=EXPECTED_SEED,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Determine IDs
    # -------------------------------------------------------------

    def make_keys(df):
        keys = []

        for _, row in df.iterrows():
            study = row.get("study_id")
            series = row.get("series_id")

            if pd.notna(study) and pd.notna(series):
                keys.append(
                    f"{int(study)}|{int(series)}"
                )

        return set(keys)

    train_keys = make_keys(train_sample)
    val_keys = make_keys(val_sample)

    overlap = train_keys.intersection(val_keys)

    print()
    status("Selected train cases", len(train_sample))
    status("Selected validation cases", len(val_sample))
    status("Train/validation overlap", len(overlap))

    if overlap:
        print()
        print("WARNING: overlap detected:")
        for key in sorted(overlap):
            print(f"  {key}")

    return {
        "train_manifest_total": len(train_df),
        "validation_manifest_total": len(val_df),
        "selected_train_cases": len(train_sample),
        "selected_validation_cases": len(val_sample),
        "train_validation_overlap": len(overlap),
    }


# =============================================================================
# PART 16 REPRODUCTION AUDIT
# =============================================================================

def audit_part16():
    section("PART 16 INDEPENDENT REPRODUCTION AUDIT")

    summary = load_json(PART16_SUMMARY)

    print("Part 16 summary loaded.")

    possible_keys = [
        "mean_validation_dice",
        "validation_dice",
        "mean_val_dice",
        "reproduced_dice",
        "part15_recorded_best_dice",
        "absolute_dice_difference",
    ]

    extracted = {}

    for key in possible_keys:
        if key in summary:
            extracted[key] = summary[key]
            status(key, summary[key])

    # -------------------------------------------------------------
    # Case metrics
    # -------------------------------------------------------------

    case_df = pd.read_csv(PART16_CASE_METRICS)

    print()
    print(f"Part 16 case metrics rows : {len(case_df)}")

    dice_col = None

    for candidate in [
        "dice",
        "validation_dice",
        "mean_dice",
        "foreground_dice",
    ]:
        if candidate in case_df.columns:
            dice_col = candidate
            break

    if dice_col is None:
        for column in case_df.columns:
            if "dice" in column.lower():
                dice_col = column
                break

    reproduced_dice = None

    if dice_col is not None:
        reproduced_dice = float(
            pd.to_numeric(
                case_df[dice_col],
                errors="coerce",
            ).mean()
        )

        print()
        status(
            "Computed Part 16 mean Dice",
            f"{reproduced_dice:.6f}",
        )

    extracted["case_metric_rows"] = len(case_df)
    extracted["computed_part16_mean_dice"] = reproduced_dice

    return extracted


# =============================================================================
# CONSISTENCY DECISION
# =============================================================================

def perform_consistency_checks(
    checkpoint_info,
    history_info,
    summary_info,
    cohort_info,
    part16_info,
):
    section("PART 17 CONSISTENCY CHECKS")

    checks = []

    # -------------------------------------------------------------
    # Check 1: checkpoint exists
    # -------------------------------------------------------------

    checks.append(
        (
            "Part 15 checkpoint exists",
            PART15_CHECKPOINT.exists(),
        )
    )

    # -------------------------------------------------------------
    # Check 2: checkpoint architecture metadata
    # -------------------------------------------------------------

    checkpoint_patch = checkpoint_info.get(
        "checkpoint_patch_size"
    )

    if checkpoint_patch is not None:
        try:
            normalized_patch = tuple(
                int(x) for x in checkpoint_patch
            )

            patch_ok = normalized_patch == PATCH_SIZE

        except Exception:
            patch_ok = False

        checks.append(
            (
                "Checkpoint patch size matches Part 10",
                patch_ok,
            )
        )

    checkpoint_classes = checkpoint_info.get(
        "checkpoint_classes"
    )

    if checkpoint_classes is not None:
        checks.append(
            (
                "Checkpoint class count is 6",
                int(checkpoint_classes) == NUM_CLASSES,
            )
        )

    checkpoint_seed = checkpoint_info.get(
        "checkpoint_seed"
    )

    if checkpoint_seed is not None:
        checks.append(
            (
                "Checkpoint seed is 42",
                int(checkpoint_seed) == EXPECTED_SEED,
            )
        )

    # -------------------------------------------------------------
    # Check 3: history vs checkpoint
    # -------------------------------------------------------------

    checkpoint_dice = checkpoint_info.get(
        "checkpoint_best_dice"
    )

    history_dice = history_info.get(
        "best_history_dice"
    )

    if (
        checkpoint_dice is not None
        and history_dice is not None
    ):
        difference = abs(
            float(checkpoint_dice)
            - float(history_dice)
        )

        checks.append(
            (
                "History best Dice matches checkpoint",
                difference < 1e-6,
            )
        )

        status(
            "History/checkpoint Dice difference",
            f"{difference:.10f}",
        )

    # -------------------------------------------------------------
    # Check 4: Part 16 reproduction
    # -------------------------------------------------------------

    reproduced = part16_info.get(
        "computed_part16_mean_dice"
    )

    if (
        checkpoint_dice is not None
        and reproduced is not None
    ):
        difference = abs(
            float(checkpoint_dice)
            - float(reproduced)
        )

        checks.append(
            (
                "Part 16 reproduces Part 15 Dice",
                difference < 1e-6,
            )
        )

        status(
            "Part 15 ↔ Part 16 Dice difference",
            f"{difference:.10f}",
        )

    # -------------------------------------------------------------
    # Check 5: expected cohort size
    # -------------------------------------------------------------

    selected_train = cohort_info.get(
        "selected_train_cases"
    )

    selected_val = cohort_info.get(
        "selected_validation_cases"
    )

    if selected_train is not None:
        checks.append(
            (
                "Controlled train cohort = 500",
                int(selected_train) == EXPECTED_TRAIN_CASES,
            )
        )

    if selected_val is not None:
        checks.append(
            (
                "Controlled validation cohort = 100",
                int(selected_val) == EXPECTED_VAL_CASES,
            )
        )

    # -------------------------------------------------------------
    # Check 6: no overlap
    # -------------------------------------------------------------

    overlap = cohort_info.get(
        "train_validation_overlap"
    )

    if overlap is not None:
        checks.append(
            (
                "Train/validation overlap = 0",
                int(overlap) == 0,
            )
        )

    # -------------------------------------------------------------
    # Print checks
    # -------------------------------------------------------------

    print()

    for name, passed in checks:
        print(
            f"{'PASS' if passed else 'FAIL':<6} - {name}"
        )

    all_pass = all(result for _, result in checks)

    return checks, all_pass


# =============================================================================
# SAVE REPORT
# =============================================================================

def save_outputs(
    checkpoint_info,
    history_info,
    summary_info,
    cohort_info,
    part16_info,
    checks,
    decision,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Audit CSV
    # -------------------------------------------------------------

    rows = []

    for name, passed in checks:
        rows.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
            }
        )

    audit_df = pd.DataFrame(rows)

    audit_csv = (
        OUTPUT_DIR
        / "part17_consistency_checks.csv"
    )

    audit_df.to_csv(
        audit_csv,
        index=False,
    )

    print(f"Saved: {audit_csv}")

    # -------------------------------------------------------------
    # Main summary
    # -------------------------------------------------------------

    summary = {
        "phase": 4,
        "part": 17,
        "purpose": (
            "RSNA-only Part 15 training history and "
            "checkpoint audit"
        ),
        "decision": decision,
        "spider_used": False,
        "test_set_used": False,
        "training_performed": False,
        "model_weights_changed": False,
        "patch_size": list(PATCH_SIZE),
        "feature_size": FEATURE_SIZE,
        "classes": NUM_CLASSES,
        "expected_train_cases": EXPECTED_TRAIN_CASES,
        "expected_validation_cases": EXPECTED_VAL_CASES,
        "expected_seed": EXPECTED_SEED,
        "checkpoint": checkpoint_info,
        "history": history_info,
        "part15_summary": summary_info,
        "cohort": cohort_info,
        "part16": part16_info,
        "checks": [
            {
                "name": name,
                "passed": bool(passed),
            }
            for name, passed in checks
        ],
    }

    summary_json = (
        OUTPUT_DIR
        / "phase4_part17_training_history_checkpoint_audit_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    print(f"Saved: {summary_json}")

    # -------------------------------------------------------------
    # Text report
    # -------------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "phase4_part17_training_history_checkpoint_audit_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 4 - PART 17\n"
            "RSNA-ONLY PART 15 TRAINING HISTORY / "
            "CHECKPOINT AUDIT\n"
            "=" * 78
            + "\n\n"
        )

        f.write(
            "Purpose:\n"
            "Audit the Part 15 extended controlled training run "
            "without performing training or modifying weights.\n\n"
        )

        f.write(
            f"Decision: {decision}\n\n"
        )

        f.write(
            "CHECK RESULTS\n"
            + "-" * 78
            + "\n"
        )

        for name, passed in checks:
            f.write(
                f"{'PASS' if passed else 'FAIL'} - {name}\n"
            )

        f.write("\n")

        f.write(
            "CHECKPOINT INFORMATION\n"
            + "-" * 78
            + "\n"
        )

        for key, value in checkpoint_info.items():
            if key != "checkpoint_keys":
                f.write(
                    f"{key}: {value}\n"
                )

        f.write("\n")

        f.write(
            "PART 16 REPRODUCTION\n"
            + "-" * 78
            + "\n"
        )

        for key, value in part16_info.items():
            f.write(
                f"{key}: {value}\n"
            )

        f.write("\n")

        f.write(
            "IMPORTANT:\n"
            "Dice values are calculated against RSNA "
            "point-derived pseudo-masks, not manually "
            "delineated clinical segmentation ground truth.\n"
        )

    print(f"Saved: {report_path}")

    return summary_json, report_path


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "PHASE 4 - PART 17\n"
        "RSNA-ONLY PART 15 TRAINING HISTORY / CHECKPOINT AUDIT"
    )

    print()
    print(
        "This stage audits Part 15.\n"
        "No training is performed.\n"
        "No model weights are modified.\n"
        "SPIDER is not used.\n"
        "The RSNA test set is not used."
    )

    print()
    print("PROJECT ROOT")
    print(PROJECT_ROOT)

    print()
    print("RSNA DATASET")
    print(RSNA_ROOT)

    print()
    print("PART 15 CHECKPOINT")
    print(PART15_CHECKPOINT)

    print()
    print("PART 16 VALIDATION")
    print(PART16_DIR)

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    # -------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------

    require_paths()

    # -------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------

    print_environment()

    # -------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------

    checkpoint_info = audit_checkpoint()

    # -------------------------------------------------------------
    # History
    # -------------------------------------------------------------

    history_df, history_info = audit_history()

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    summary_info = audit_part15_summary()

    # -------------------------------------------------------------
    # Cohort
    # -------------------------------------------------------------

    cohort_info = audit_cohort()

    # -------------------------------------------------------------
    # Part 16
    # -------------------------------------------------------------

    part16_info = audit_part16()

    # -------------------------------------------------------------
    # Consistency checks
    # -------------------------------------------------------------

    checks, all_pass = perform_consistency_checks(
        checkpoint_info,
        history_info,
        summary_info,
        cohort_info,
        part16_info,
    )

    # -------------------------------------------------------------
    # Decision
    # -------------------------------------------------------------

    if all_pass:
        decision = (
            "PASS - Part 15 training history and best-checkpoint "
            "metadata are internally consistent, and Part 16 "
            "independently reproduces the recorded Part 15 "
            "best-checkpoint validation result."
        )
    else:
        decision = (
            "ACTION REQUIRED - one or more Part 15 history, "
            "checkpoint, cohort, or reproduction consistency "
            "checks failed."
        )

    section("PART 17 FINAL SUMMARY")

    status(
        "Part 15 checkpoint",
        "FOUND",
    )

    status(
        "Part 15 history",
        "FOUND"
        if history_info.get("history_file_found")
        else "NOT FOUND",
    )

    status(
        "Part 15 best Dice",
        checkpoint_info.get(
            "checkpoint_best_dice"
        ),
    )

    status(
        "Part 16 reproduced Dice",
        part16_info.get(
            "computed_part16_mean_dice"
        ),
    )

    status(
        "Selected train cases",
        cohort_info.get(
            "selected_train_cases"
        ),
    )

    status(
        "Selected validation cases",
        cohort_info.get(
            "selected_validation_cases"
        ),
    )

    status(
        "Train/validation overlap",
        cohort_info.get(
            "train_validation_overlap"
        ),
    )

    print()

    print(
        "SPIDER used            : NO"
    )
    print(
        "Test set used          : NO"
    )
    print(
        "Training performed     : NO"
    )
    print(
        "Model weights changed  : NO"
    )

    print()
    print("FINAL DECISION")
    print(decision)

    print()
    print(
        "IMPORTANT:\n"
        "Dice is calculated against RSNA point-derived "
        "pseudo-masks, not manually delineated clinical "
        "segmentation ground truth."
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    section("SAVING PART 17 RESULTS")

    save_outputs(
        checkpoint_info,
        history_info,
        summary_info,
        cohort_info,
        part16_info,
        checks,
        decision,
    )

    section("PHASE 4 - PART 17 COMPLETE")

    if not all_pass:
        sys.exit(1)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as exc:

        print()
        print("=" * 78)
        print("PART 17 ERROR")
        print("=" * 78)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        sys.exit(1)