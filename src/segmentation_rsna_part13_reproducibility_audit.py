"""
PHASE 4 - PART 13
RSNA-ONLY PILOT REPRODUCIBILITY AND PREPROCESSING CONSISTENCY AUDIT

This is a diagnostic/reproducibility stage following the discrepancy observed
between:
  - Part 11 best validation Dice (~0.140356)
  - Part 12 checkpoint-validation Dice (~0.000436)

Part 13 intentionally reuses the corrected Part 11 implementation rather than
creating a second preprocessing pipeline. It:

1. Loads the Part 11 corrected source as a module (without running training).
2. Reconstructs the EXACT 100/30 pilot selection using seed 42.
3. Loads the EXACT Part 11 best checkpoint.
4. Re-runs validation with Part 11's own load_tensor_case(), preprocessing,
   metric definition, and DiceCELoss.
5. Compares the reproduced result with the checkpoint's embedded history.
6. Compares the reproduced 30 case IDs with Part 12 case-level metrics.
7. Reports whether the discrepancy is:
      A) reproducible,
      B) caused by different case selection,
      C) caused by preprocessing/loader mismatch,
      D) caused by checkpoint metadata/state mismatch, or
      E) unresolved and requiring deeper inspection.
8. Saves machine-readable and human-readable audit outputs.

No training is performed.
No optimizer step is performed.
No model weights are modified.
RSNA only. SPIDER is never used. Test data is never used.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss


# ---------------------------------------------------------------------------
# PATHS / CONFIG
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

PART8_MANIFEST_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
)

TRAIN_MANIFEST = PART8_MANIFEST_DIR / "rsna_part8_train_manifest.csv"
VAL_MANIFEST = PART8_MANIFEST_DIR / "rsna_part8_validation_manifest.csv"

PART11_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part11_controlled_pilot_training"
)

PART11_SOURCE = (
    PROJECT_ROOT
    / "src"
    / "segmentation_rsna_part11_controlled_pilot_training.py"
)

CHECKPOINT = PART11_DIR / "checkpoints" / "best_model.pth"
PART11_RESULT = PART11_DIR / "part11_training_result.json"
PART11_HISTORY = PART11_DIR / "part11_training_history.csv"

PART12_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part12_pilot_checkpoint_validation"
)

PART12_CASE_METRICS = PART12_DIR / "part12_validation_case_metrics.csv"
PART12_SUMMARY = (
    PART12_DIR
    / "phase4_part12_checkpoint_validation_summary.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part13_reproducibility_audit"
)

REPORT_DIR = OUTPUT_DIR / "reports"

SEED = 42
PILOT_TRAIN_CASES = 100
PILOT_VAL_CASES = 30

NUM_CLASSES = 6
PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12

LABELS = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def banner(text: str) -> None:
    print("=" * 78)
    print(text)
    print("=" * 78)


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def stable_case_key(row: pd.Series) -> str:
    study = str(row["study_id"])
    series = str(row["series_id"])
    return f"{study}::{series}"


def require_paths() -> None:
    banner("PATH VALIDATION")

    paths = {
        "RSNA root": RSNA_ROOT,
        "train manifest": TRAIN_MANIFEST,
        "validation manifest": VAL_MANIFEST,
        "Part 11 source": PART11_SOURCE,
        "Part 11 checkpoint": CHECKPOINT,
        "Part 11 result": PART11_RESULT,
        "Part 12 case metrics": PART12_CASE_METRICS,
        "Part 12 summary": PART12_SUMMARY,
    }

    missing = []

    for name, path in paths.items():
        exists = path.exists()
        print(f"{name:<30}: {'FOUND' if exists else 'MISSING'}")
        if not exists:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required audit input(s):\n" + "\n".join(missing)
        )


def import_part11():
    """
    Import corrected Part 11 as a module.

    The corrected Part 11 has a main guard, so importing it does not start
    training. We deliberately reuse its exact functions to avoid silently
    creating a second preprocessing implementation.
    """
    banner("IMPORTING CORRECTED PART 11 IMPLEMENTATION")

    spec = importlib.util.spec_from_file_location(
        "part11_corrected_for_part13",
        PART11_SOURCE,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Unable to load corrected Part 11 source.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["part11_corrected_for_part13"] = module
    spec.loader.exec_module(module)

    required = [
        "select_pilot_rows",
        "load_tensor_case",
        "create_model",
        "dice_from_prediction",
    ]

    missing = [name for name in required if not hasattr(module, name)]

    if missing:
        raise AttributeError(
            "Corrected Part 11 is missing required API: "
            + ", ".join(missing)
        )

    print("âœ“ Corrected Part 11 imported without starting training.")
    print("âœ“ Exact Part 11 sampling function available.")
    print("âœ“ Exact Part 11 preprocessing available.")
    print("âœ“ Exact Part 11 metric available.")
    return module


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CASE SELECTION AUDIT
# ---------------------------------------------------------------------------

def reconstruct_part11_cohort(part11) -> Tuple[pd.DataFrame, pd.DataFrame]:
    banner("RECONSTRUCTING EXACT PART 11 PILOT COHORT")

    train_rows = part11.select_pilot_rows(
        TRAIN_MANIFEST,
        PILOT_TRAIN_CASES,
        SEED,
    )

    val_rows = part11.select_pilot_rows(
        VAL_MANIFEST,
        PILOT_VAL_CASES,
        SEED,
    )

    print(f"Train manifest total : {len(pd.read_csv(TRAIN_MANIFEST))}")
    print(f"Validation total     : {len(pd.read_csv(VAL_MANIFEST))}")
    print(f"Reconstructed train  : {len(train_rows)}")
    print(f"Reconstructed val    : {len(val_rows)}")

    return train_rows, val_rows


def compare_case_cohorts(
    reconstructed_val: pd.DataFrame,
    part12_df: pd.DataFrame,
) -> Dict:
    banner("PART 11 â†” PART 12 CASE COHORT COMPARISON")

    expected = [
        stable_case_key(row)
        for _, row in reconstructed_val.iterrows()
    ]

    actual = []

    if "study_id" in part12_df.columns and "series_id" in part12_df.columns:
        actual = [
            f"{row.study_id}::{row.series_id}"
            for row in part12_df.itertuples()
        ]
    else:
        print("WARNING: Part 12 CSV does not expose study_id/series_id.")

    expected_set = set(expected)
    actual_set = set(actual)

    only_part11 = sorted(expected_set - actual_set)
    only_part12 = sorted(actual_set - expected_set)

    same_order = expected == actual

    result = {
        "part11_cases": len(expected),
        "part12_cases": len(actual),
        "same_case_set": expected_set == actual_set,
        "same_case_order": same_order,
        "part11_only_cases": only_part11,
        "part12_only_cases": only_part12,
    }

    print(f"Part 11 reconstructed cases : {len(expected)}")
    print(f"Part 12 evaluated cases     : {len(actual)}")
    print(f"Same case set               : {result['same_case_set']}")
    print(f"Same case order             : {result['same_case_order']}")

    if only_part11:
        print("Cases only in Part 11 cohort:")
        for x in only_part11:
            print(" ", x)

    if only_part12:
        print("Cases only in Part 12 cohort:")
        for x in only_part12:
            print(" ", x)

    return result


# ---------------------------------------------------------------------------
# CHECKPOINT AUDIT
# ---------------------------------------------------------------------------

def audit_checkpoint() -> Tuple[Dict, Dict]:
    banner("CHECKPOINT METADATA AUDIT")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Part 11 checkpoint is not a dictionary.")

    result = {}

    result["checkpoint_keys"] = sorted(checkpoint.keys())

    result["checkpoint_epoch"] = checkpoint.get("epoch")
    result["checkpoint_best_val_dice"] = checkpoint.get("best_val_dice")

    config = checkpoint.get("config", {})
    result["checkpoint_config"] = config

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    result["checkpoint_parameter_tensors"] = len(state)

    embedded_history = checkpoint.get("history", [])
    result["embedded_history_rows"] = len(embedded_history)

    embedded_best = None
    if embedded_history:
        candidates = []

        for item in embedded_history:
            if isinstance(item, dict):
                value = item.get("val_mean_dice")
                if value is None:
                    value = item.get("val_dice")
                if value is not None:
                    candidates.append(float(value))

        if candidates:
            embedded_best = max(candidates)

    result["embedded_history_best_val_dice"] = embedded_best

    print(f"Checkpoint epoch             : {result['checkpoint_epoch']}")
    print(f"Checkpoint best_val_dice     : {result['checkpoint_best_val_dice']}")
    print(f"Checkpoint parameter tensors : {result['checkpoint_parameter_tensors']}")
    print(f"Embedded history rows        : {result['embedded_history_rows']}")
    print(f"Embedded history best Dice   : {embedded_best}")

    part11_result = load_json(PART11_RESULT)
    print(
        f"Part 11 result best Dice     : "
        f"{part11_result.get('best_validation_dice')}"
    )

    return result, part11_result


# ---------------------------------------------------------------------------
# EXACT PART 11 REPRODUCTION
# ---------------------------------------------------------------------------

@torch.no_grad()
def reproduce_part11_validation(
    part11,
    val_rows: pd.DataFrame,
    device: torch.device,
) -> Tuple[List[Dict], Dict]:
    banner("EXACT PART 11 VALIDATION REPRODUCTION")

    model = part11.create_model(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    state = checkpoint["model_state_dict"]

    cleaned = {}
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=False,
    )

    print(f"Checkpoint missing keys    : {len(missing)}")
    print(f"Checkpoint unexpected keys : {len(unexpected)}")

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not exactly match Part 11 model architecture."
        )

    model.eval()

    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    rows = []
    fallback_count = 0

    start = time.time()

    for step, (_, row) in enumerate(
        val_rows.iterrows(),
        start=1,
    ):
        image, mask, info = part11.load_tensor_case(
            row,
            part11.load_part9_module(),
        )

        if info.get("part11_loader") == "local_robust_fallback":
            fallback_count += 1

        expected_image_shape = (1, *PATCH_SIZE)

        shape_ok = tuple(image.shape) == expected_image_shape

        if not shape_ok:
            raise RuntimeError(
                f"Part 11 preprocessing produced unexpected image shape: "
                f"{tuple(image.shape)}"
            )

        image_device = image.unsqueeze(0).to(device)
        mask_device = mask.unsqueeze(0).to(device)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=device.type == "cuda",
        ):
            logits = model(image_device)
            loss = loss_function(
                logits,
                mask_device.unsqueeze(1),
            )

        mean_dice, class_dice = part11.dice_from_prediction(
            logits,
            mask_device,
        )

        prediction = torch.argmax(
            logits,
            dim=1,
        )[0].detach().cpu().numpy().astype(np.int64)

        target = mask.detach().cpu().numpy().astype(np.int64)
        image_np = image[0].detach().cpu().numpy().astype(np.float32)

        case = {
            "case_index": step,
            "study_id": str(info.get("study_id")),
            "series_id": str(info.get("series_id")),
            "series_description": str(
                info.get("series_description", "")
            ),
            "loader": str(
                info.get("part11_loader", "")
            ),
            "dicom_harmonized": bool(
                info.get("dicom_harmonized", False)
            ),
            "loss": float(loss.item()),
            "mean_foreground_dice": float(mean_dice),
            "target_foreground_voxels": int(
                np.count_nonzero(target > 0)
            ),
            "prediction_foreground_voxels": int(
                np.count_nonzero(prediction > 0)
            ),
            "image_min": float(image_np.min()),
            "image_max": float(image_np.max()),
            "image_mean": float(image_np.mean()),
            "image_std": float(image_np.std()),
            "image_sha256": sha256_array(image_np),
            "mask_sha256": sha256_array(target),
            "prediction_sha256": sha256_array(prediction),
        }

        for class_id, value in enumerate(
            class_dice,
            start=1,
        ):
            case[f"dice_class_{class_id}"] = float(value)

        rows.append(case)

        print(
            f"  [{step:02d}/{len(val_rows)}] "
            f"{case['study_id']} | {case['series_id']} | "
            f"Loss={case['loss']:.4f} "
            f"Dice={case['mean_foreground_dice']:.6f}"
        )

        del (
            image_device,
            mask_device,
            logits,
            loss,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    summary = {
        "cases": len(df),
        "mean_loss": float(df["loss"].mean()),
        "mean_dice": float(df["mean_foreground_dice"].mean()),
        "median_dice": float(df["mean_foreground_dice"].median()),
        "fallback_cases": fallback_count,
        "elapsed_minutes": float((time.time() - start) / 60.0),
    }

    for class_id in range(1, NUM_CLASSES):
        summary[f"dice_class_{class_id}"] = float(
            df[f"dice_class_{class_id}"].mean()
        )

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rows, summary


# ---------------------------------------------------------------------------
# PART 12 COMPARISON
# ---------------------------------------------------------------------------

def compare_metrics(
    exact_df: pd.DataFrame,
    part12_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    banner("PART 11-EXACT ↔ PART 12 METRIC COMPARISON")

    required_exact = {
        "study_id",
        "series_id",
        "loss",
        "mean_foreground_dice",
    }

    required_part12 = {
        "study_id",
        "series_id",
    }

    missing_exact = required_exact - set(exact_df.columns)
    missing_part12 = required_part12 - set(part12_df.columns)

    if missing_exact:
        raise RuntimeError(
            f"Exact Part 11 reproduction is missing columns: "
            f"{sorted(missing_exact)}"
        )

    if missing_part12:
        raise RuntimeError(
            f"Part 12 case metrics are missing columns: "
            f"{sorted(missing_part12)}"
        )

    left = exact_df.copy()
    right = part12_df.copy()

    left["case_key"] = (
        left["study_id"].astype(str)
        + "::"
        + left["series_id"].astype(str)
    )

    right["case_key"] = (
        right["study_id"].astype(str)
        + "::"
        + right["series_id"].astype(str)
    )

    # Explicitly rename the columns before merging so pandas cannot
    # create unexpected suffix names.
    left = left.rename(
        columns={
            "loss": "loss_part11_exact",
            "mean_foreground_dice": "mean_foreground_dice_part11_exact",
        }
    )

    # Part 12 normally stores these under mean_dice/loss.
    if "mean_dice" in right.columns:
        right = right.rename(
            columns={"mean_dice": "mean_dice_part12"}
        )

    if "loss" in right.columns:
        right = right.rename(
            columns={"loss": "loss_part12"}
        )

    comparison = left.merge(
        right,
        on="case_key",
        how="outer",
        indicator=True,
    )

    common = comparison[
        comparison["_merge"] == "both"
    ].copy()

    if not common.empty:

        if (
            "mean_dice_part12" not in common.columns
            or "mean_foreground_dice_part11_exact" not in common.columns
        ):
            raise RuntimeError(
                "Could not identify the Part 11 and Part 12 Dice columns "
                "after explicit merge."
            )

        common["dice_difference"] = (
            common["mean_foreground_dice_part11_exact"]
            - common["mean_dice_part12"]
        )

        if "loss_part12" in common.columns:
            common["loss_difference"] = (
                common["loss_part11_exact"]
                - common["loss_part12"]
            )
        else:
            common["loss_difference"] = np.nan

        common["absolute_dice_difference"] = (
            common["dice_difference"].abs()
        )

    summary = {
        "common_cases": int(len(common)),
        "mean_part11_exact_dice": (
            float(
                common["mean_foreground_dice_part11_exact"].mean()
            )
            if len(common)
            else None
        ),
        "mean_part12_dice": (
            float(common["mean_dice_part12"].mean())
            if len(common) and "mean_dice_part12" in common.columns
            else None
        ),
        "mean_absolute_case_dice_difference": (
            float(common["absolute_dice_difference"].mean())
            if len(common)
            else None
        ),
        "max_absolute_case_dice_difference": (
            float(common["absolute_dice_difference"].max())
            if len(common)
            else None
        ),
    }

    print()
    print(f"Common cases                    : {len(common)}")

    if len(common):
        print(
            "Mean exact Part 11 Dice        : "
            f"{summary['mean_part11_exact_dice']:.6f}"
        )

        print(
            "Mean Part 12 Dice              : "
            f"{summary['mean_part12_dice']:.6f}"
        )

        print(
            "Mean absolute Dice difference  : "
            f"{summary['mean_absolute_case_dice_difference']:.6f}"
        )

        print(
            "Max absolute Dice difference   : "
            f"{summary['max_absolute_case_dice_difference']:.6f}"
        )

    return comparison, summary

def determine_diagnosis(
    cohort_result: Dict,
    checkpoint_result: Dict,
    part11_result: Dict,
    reproduced_summary: Dict,
    comparison_summary: Dict,
) -> Tuple[str, List[str]]:
    findings = []

    # Cohort
    if not cohort_result.get("same_case_set", False):
        findings.append(
            "CASE_SELECTION_MISMATCH: Part 12 did not evaluate the exact "
            "Part 11 30-case cohort."
        )
        return (
            "CASE_SELECTION_MISMATCH",
            findings,
        )

    # Checkpoint metadata
    embedded = checkpoint_result.get(
        "checkpoint_best_val_dice"
    )
    reported = part11_result.get(
        "best_validation_dice"
    )
    reproduced = reproduced_summary.get(
        "mean_dice"
    )

    if embedded is not None and reported is not None:
        if abs(float(embedded) - float(reported)) > 1e-6:
            findings.append(
                "CHECKPOINT_METADATA_MISMATCH: checkpoint and Part 11 "
                "result JSON report different best Dice values."
            )

    # Exact reproduction vs reported best.
    if reported is not None:
        delta = abs(float(reproduced) - float(reported))

        if delta <= 0.01:
            findings.append(
                "PART11_REPRODUCED: exact Part 11 pipeline reproduces the "
                "reported best validation Dice within 0.01."
            )
        else:
            findings.append(
                "PART11_NOT_REPRODUCED: exact Part 11 pipeline does not "
                "reproduce the reported best validation Dice."
            )

    # Part 12 comparison.
    exact_dice = comparison_summary.get(
        "mean_part11_exact_dice"
    )
    p12_dice = comparison_summary.get(
        "mean_part12_dice"
    )

    if exact_dice is not None and p12_dice is not None:
        diff = abs(exact_dice - p12_dice)

        if diff <= 0.005:
            findings.append(
                "PART12_CONSISTENT: Part 12 agrees with exact Part 11 "
                "case-level Dice."
            )
            diagnosis = "CONSISTENT"
        else:
            findings.append(
                "PART12_MISMATCH: Part 12 produces materially different "
                "case-level Dice despite matching the case cohort."
            )
            diagnosis = "PREPROCESSING_OR_METRIC_MISMATCH"
    else:
        diagnosis = "INSUFFICIENT_CASE_OVERLAP"

    return diagnosis, findings


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    set_seed()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    banner("PHASE 4 - PART 13")
    print("RSNA-ONLY PILOT REPRODUCIBILITY / PREPROCESSING CONSISTENCY AUDIT")
    print()

    print("Purpose:")
    print("Resolve the discrepancy between Part 11 and Part 12 validation.")
    print()

    require_paths()

    banner("PYTORCH / GPU ENVIRONMENT")
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"Device          : {device}")

    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(
            f"GPU memory      : "
            f"{props.total_memory / (1024 ** 3):.2f} GB"
        )

    part11 = import_part11()

    train_rows, val_rows = reconstruct_part11_cohort(part11)

    part12_df = pd.read_csv(PART12_CASE_METRICS)

    cohort_result = compare_case_cohorts(
        val_rows,
        part12_df,
    )

    checkpoint_result, part11_result = audit_checkpoint()

    exact_rows, reproduced_summary = reproduce_part11_validation(
        part11,
        val_rows,
        device,
    )

    exact_df = pd.DataFrame(exact_rows)

    exact_csv = OUTPUT_DIR / "part13_exact_part11_reproduction.csv"
    exact_df.to_csv(exact_csv, index=False)

    comparison_df, comparison_summary = compare_metrics(
        exact_df,
        part12_df,
    )

    comparison_csv = OUTPUT_DIR / "part13_part11_vs_part12_case_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    diagnosis, findings = determine_diagnosis(
        cohort_result,
        checkpoint_result,
        part11_result,
        reproduced_summary,
        comparison_summary,
    )

    audit = {
        "phase": "Phase 4 - Part 13",
        "purpose": "RSNA-only pilot reproducibility and preprocessing consistency audit",
        "seed": SEED,
        "patch_size": list(PATCH_SIZE),
        "feature_size": FEATURE_SIZE,
        "num_classes": NUM_CLASSES,
        "rsna_only": True,
        "spider_used": False,
        "training_performed": False,
        "weights_modified": False,
        "test_set_used": False,
        "part11_source": str(PART11_SOURCE),
        "checkpoint": str(CHECKPOINT),
        "part12_case_metrics": str(PART12_CASE_METRICS),
        "cohort_comparison": cohort_result,
        "checkpoint_audit": checkpoint_result,
        "part11_result": part11_result,
        "exact_part11_reproduction": reproduced_summary,
        "part11_vs_part12_comparison": comparison_summary,
        "diagnosis": diagnosis,
        "findings": findings,
    }

    summary_json = (
        OUTPUT_DIR
        / "phase4_part13_reproducibility_audit_summary.json"
    )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            audit,
            f,
            indent=2,
            default=str,
        )

    report_path = (
        REPORT_DIR
        / "phase4_part13_reproducibility_audit_report.txt"
    )

    report = [
        "PHASE 4 - PART 13",
        "RSNA-ONLY PILOT REPRODUCIBILITY / PREPROCESSING CONSISTENCY AUDIT",
        "=" * 78,
        "",
        f"Diagnosis: {diagnosis}",
        "",
        "CASE COHORT",
        f"Part 11 reconstructed validation cases: "
        f"{cohort_result.get('part11_cases')}",
        f"Part 12 validation cases: "
        f"{cohort_result.get('part12_cases')}",
        f"Same case set: "
        f"{cohort_result.get('same_case_set')}",
        f"Same case order: "
        f"{cohort_result.get('same_case_order')}",
        "",
        "CHECKPOINT",
        f"Checkpoint epoch: "
        f"{checkpoint_result.get('checkpoint_epoch')}",
        f"Checkpoint best Dice: "
        f"{checkpoint_result.get('checkpoint_best_val_dice')}",
        f"Part 11 result best Dice: "
        f"{part11_result.get('best_validation_dice')}",
        "",
        "EXACT PART 11 REPRODUCTION",
        f"Cases: {reproduced_summary.get('cases')}",
        f"Mean loss: {reproduced_summary.get('mean_loss'):.6f}",
        f"Mean Dice: {reproduced_summary.get('mean_dice'):.6f}",
        f"Median Dice: {reproduced_summary.get('median_dice'):.6f}",
        f"DICOM fallback cases: "
        f"{reproduced_summary.get('fallback_cases')}",
        "",
        "PART 11 vs PART 12",
        f"Common cases: "
        f"{comparison_summary.get('common_cases')}",
        f"Part 11 exact mean Dice: "
        f"{comparison_summary.get('mean_part11_exact_dice')}",
        f"Part 12 mean Dice: "
        f"{comparison_summary.get('mean_part12_dice')}",
        f"Mean absolute case Dice difference: "
        f"{comparison_summary.get('mean_absolute_case_dice_difference')}",
        f"Maximum absolute case Dice difference: "
        f"{comparison_summary.get('max_absolute_case_dice_difference')}",
        "",
        "FINDINGS",
    ]

    report.extend(
        f"- {item}"
        for item in findings
    )

    report.extend(
        [
            "",
            "SCIENTIFIC CONTROL",
            "RSNA only: YES",
            "SPIDER used: NO",
            "Test set used: NO",
            "Training performed: NO",
            "Model weights modified: NO",
            "",
            "INTERPRETATION",
            "This audit is intended to establish whether the Part 11 result",
            "can be reproduced using the exact Part 11 preprocessing and",
            "metric implementation. Part 12 must not be interpreted as a",
            "final model-performance result until any mismatch is resolved.",
            "",
            "IMPORTANT",
            "RSNA targets are point-derived pseudo-masks, not manual",
            "pixel-wise segmentation ground truth.",
        ]
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    banner("PART 13 FINAL SUMMARY")
    print(f"Diagnosis                  : {diagnosis}")
    print(
        f"Exact Part 11 mean Dice    : "
        f"{reproduced_summary.get('mean_dice'):.6f}"
    )
    print(
        f"Part 11 reported best Dice : "
        f"{part11_result.get('best_validation_dice')}"
    )
    print(
        f"Part 12 mean Dice          : "
        f"{comparison_summary.get('mean_part12_dice')}"
    )
    print(
        f"Same validation case set   : "
        f"{cohort_result.get('same_case_set')}"
    )
    print(
        f"Common cases               : "
        f"{comparison_summary.get('common_cases')}"
    )

    print()
    print(f"Saved: {exact_csv}")
    print(f"Saved: {comparison_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_path}")

    banner("PART 13 DECISION")

    if diagnosis == "CONSISTENT":
        print(
            "PASS - Part 11 validation is reproducible and Part 12 is "
            "consistent with the exact Part 11 implementation."
        )
    elif diagnosis == "CASE_SELECTION_MISMATCH":
        print(
            "ACTION REQUIRED - Part 11 and Part 12 evaluated different "
            "validation cohorts."
        )
    elif diagnosis == "PREPROCESSING_OR_METRIC_MISMATCH":
        print(
            "ACTION REQUIRED - Part 12 does not reproduce the exact Part 11 "
            "preprocessing/metric result."
        )
        print(
            "Do NOT use Part 12's Dice as a final performance number."
        )
    else:
        print(
            "ACTION REQUIRED - reproducibility remains unresolved."
        )

    print()
    print("SPIDER used          : NO")
    print("Test set used        : NO")
    print("Training performed   : NO")
    print("Model weights changed: NO")

    del exact_df, comparison_df
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    banner("PHASE 4 - PART 13 COMPLETE")


if __name__ == "__main__":
    main()



