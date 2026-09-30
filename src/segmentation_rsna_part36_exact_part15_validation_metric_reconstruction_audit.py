"""
==============================================================================
PHASE 4 - PART 36
RSNA-ONLY EXACT PART 15 VALIDATION-METRIC RECONSTRUCTION AUDIT
==============================================================================

Purpose
-------
Resolve the discrepancy discovered in Part 35:

    Part 15 recorded validation Dice : 0.668
    Saved Part 15 checkpoint actual foreground Dice : 0.000
    Saved checkpoint predicts background for all 100 validation cases.

This audit compares the validation API EXPECTED by the Part 15 source with
the CURRENT validated Part 11 API, and independently reconstructs the saved
Part 15 checkpoint's validation metric on the exact deterministic cohort.

This is EVALUATION / SOURCE AUDIT ONLY.

NO:
    - training
    - optimizer
    - optimizer step
    - weight modification
    - SPIDER
    - RSNA test set
"""

from __future__ import annotations

import ast
import inspect
import json
import random
import re
import sys
import traceback
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

VAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
    / "rsna_part8_validation_manifest.csv"
)

PART11_SOURCE = (
    SRC_DIR
    / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

PART15_SOURCE = (
    SRC_DIR
    / "segmentation_rsna_part15_extended_controlled_training.py"
)

PART15_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "checkpoints"
    / "best_model.pth"
)

PART15_HISTORY = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_training_history.csv"
)

PART15_VAL_COHORT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_validation_cohort.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part36_exact_part15_validation_metric_reconstruction_audit"
)

CASE_CSV = OUTPUT_DIR / "part36_case_reconstruction_audit.csv"
CLASS_CSV = OUTPUT_DIR / "part36_class_metric_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "phase4_part36_summary.json"
REPORT_TXT = OUTPUT_DIR / "phase4_part36_report.txt"

SEED = 42
VAL_CASES = 100
NUM_CLASSES = 6
PATCH_SIZE = (64, 96, 96)

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

FOREGROUND = list(range(1, NUM_CLASSES))


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def import_module(path: Path, name: str):
    spec = spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def norm_id(value):
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def ids(row):
    return norm_id(row.get("study_id")), norm_id(row.get("series_id"))


def ensure_image(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = torch.as_tensor(x).float()

    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 4 and x.shape[0] == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 5:
        raise ValueError(f"Unexpected image shape: {tuple(x.shape)}")

    return x.contiguous()


def ensure_mask(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = torch.as_tensor(x).long()

    if x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {tuple(x.shape)}")

    return x.contiguous()


def normalize_loaded_case(image, mask):
    image = ensure_image(image)
    mask = ensure_mask(mask)

    if tuple(image.shape[-3:]) != PATCH_SIZE:
        image = F.interpolate(
            image,
            size=PATCH_SIZE,
            mode="trilinear",
            align_corners=False,
        )

    if tuple(mask.shape) != PATCH_SIZE:
        mask = F.interpolate(
            mask[None, None].float(),
            size=PATCH_SIZE,
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

    return image, mask


# =============================================================================
# METRIC IMPLEMENTATIONS
# =============================================================================

def conventional_foreground_dice(prediction, target):
    """
    The CURRENT corrected Part 11 implementation:
    - argmax already performed
    - foreground classes 1..5
    - empty/empty class is assigned Dice 1.0
    - mean over foreground classes
    """
    class_dice = {}

    for class_id in FOREGROUND:
        pred = prediction == class_id
        true = target == class_id

        intersection = int((pred & true).sum().item())
        denominator = int(pred.sum().item() + true.sum().item())

        if denominator == 0:
            value = 1.0
        else:
            value = 2.0 * intersection / denominator

        class_dice[class_id] = float(value)

    return float(np.mean(list(class_dice.values()))), class_dice


def three_argument_part15_compatible_metric(logits, target, num_classes):
    """
    Compatibility metric ONLY for source-level diagnosis.

    Part 15 calls:

        part11.dice_from_prediction(logits, mask_d, NUM_CLASSES)

    The current Part 11 function accepts only:

        dice_from_prediction(logits, target)

    Therefore this adapter demonstrates what a 3-argument call would need
    to mean without modifying the real Part 11 module.
    """
    if logits.ndim != 5:
        raise ValueError(f"Expected logits [B,C,D,H,W], got {tuple(logits.shape)}")

    prediction = torch.argmax(logits, dim=1)

    values = {}
    for class_id in range(1, int(num_classes)):
        p = prediction == class_id
        t = target == class_id

        tp = int((p & t).sum().item())
        denom = int(p.sum().item() + t.sum().item())

        values[class_id] = 1.0 if denom == 0 else (2.0 * tp / denom)

    return float(np.mean(list(values.values()))), values


# =============================================================================
# STATIC SOURCE AUDIT
# =============================================================================

def extract_function_source(path: Path, function_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                lines = text.splitlines()
                return "\n".join(lines[node.lineno - 1:node.end_lineno])

    return ""


def inspect_part15_metric_call():
    source = extract_function_source(PART15_SOURCE, "evaluate")

    calls = []

    if source:
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None

                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id

                if func_name == "dice_from_prediction":
                    calls.append({
                        "line": node.lineno,
                        "positional_arguments": len(node.args),
                        "keyword_arguments": [
                            kw.arg for kw in node.keywords
                        ],
                    })

    return source, calls


# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint(model, device):
    ckpt = torch.load(
        PART15_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    state = None

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                state = value
                break

    if state is None:
        if isinstance(ckpt, dict) and all(
            isinstance(v, torch.Tensor)
            for v in ckpt.values()
        ):
            state = ckpt
        else:
            raise RuntimeError(
                "Could not locate model_state_dict in checkpoint."
            )

    cleaned = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }

    result = model.load_state_dict(cleaned, strict=False)

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={len(result.missing_keys)}, "
            f"unexpected={len(result.unexpected_keys)}"
        )

    return ckpt


# =============================================================================
# CASE LOADING
# =============================================================================

def load_case(part11, part9, row):
    image, mask, info = part11.load_tensor_case(row, part9)

    if isinstance(image, torch.Tensor):
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        image = image.detach().cpu().numpy()
    else:
        image = np.asarray(image)

    if isinstance(mask, torch.Tensor):
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask[0]
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)

    processed = part11.preprocess_case(
        image,
        mask,
    )

    if isinstance(processed, tuple):
        image, mask = processed[0], processed[1]
    else:
        image = processed

    image, mask = normalize_loaded_case(
        image,
        mask,
    )

    return image, mask, dict(info or {})


# =============================================================================
# MAIN
# =============================================================================

def main():
    seed_everything()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("PHASE 4 - PART 36")
    print("RSNA-ONLY EXACT PART 15 VALIDATION-METRIC RECONSTRUCTION AUDIT")
    print()
    print("Evaluation / source audit only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("No optimizer step is performed.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")

    # -------------------------------------------------------------------------
    # PATHS
    # -------------------------------------------------------------------------
    section("PATH VALIDATION")

    paths = {
        "RSNA root": RSNA_ROOT,
        "validation manifest": VAL_MANIFEST,
        "Part 11 source": PART11_SOURCE,
        "Part 15 source": PART15_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 history": PART15_HISTORY,
    }

    for name, path in paths.items():
        print(f"{name:<36}: {'FOUND' if path.exists() else 'MISSING'}")

    missing = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required input(s) missing:\n" + "\n".join(missing)
        )

    # -------------------------------------------------------------------------
    # ENV
    # -------------------------------------------------------------------------
    section("PYTORCH / GPU ENVIRONMENT")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print(f"PyTorch version                       : {torch.__version__}")
    print(f"CUDA available                        : {torch.cuda.is_available()}")
    print(f"Device                                : {device}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU                                   : {props.name}")
        print(
            f"GPU memory                            : "
            f"{props.total_memory / 1024**3:.2f} GB"
        )

    print(f"Patch size                            : {PATCH_SIZE}")
    print(f"Classes                               : {NUM_CLASSES}")
    print(f"Validation cases                      : {VAL_CASES}")

    # -------------------------------------------------------------------------
    # SOURCE API AUDIT
    # -------------------------------------------------------------------------
    section("PART 15 SOURCE API AUDIT")

    part15_eval_source, metric_calls = inspect_part15_metric_call()

    print("Part 15 evaluate() metric calls:")
    if metric_calls:
        for call in metric_calls:
            print(
                f"  line {call['line']}: "
                f"dice_from_prediction positional args = "
                f"{call['positional_arguments']}"
            )
    else:
        print("  No dice_from_prediction call found.")

    part15_signature_match = False

    if metric_calls:
        part15_signature_match = all(
            call["positional_arguments"] == 2
            for call in metric_calls
        )

    print()
    if part15_signature_match:
        print("SOURCE/API STATUS: COMPATIBLE")
    else:
        print("SOURCE/API STATUS: INCOMPATIBLE")

    # -------------------------------------------------------------------------
    # PART 11
    # -------------------------------------------------------------------------
    section("IMPORTING CURRENT CORRECTED PART 11")

    part11 = import_module(
        PART11_SOURCE,
        "part11_for_part36",
    )

    required = [
        "select_pilot_rows",
        "load_part9_module",
        "load_tensor_case",
        "preprocess_case",
        "create_model",
        "dice_from_prediction",
    ]

    missing_api = [
        name for name in required
        if not hasattr(part11, name)
    ]

    if missing_api:
        raise RuntimeError(
            "Current Part 11 missing required API: "
            + ", ".join(missing_api)
        )

    print("✓ Current corrected Part 11 imported.")

    print(
        "Current Part 11 dice_from_prediction signature : "
        f"{inspect.signature(part11.dice_from_prediction)}"
    )

    # -------------------------------------------------------------------------
    # EXACT COHORT
    # -------------------------------------------------------------------------
    section("RECONSTRUCTING EXACT PART 15 VALIDATION COHORT")

    val_manifest = pd.read_csv(VAL_MANIFEST)

    val_rows = part11.select_pilot_rows(
        VAL_MANIFEST,
        VAL_CASES,
        SEED,
    ).reset_index(drop=True)

    print(f"Validation manifest rows               : {len(val_manifest)}")
    print(f"Reconstructed validation rows          : {len(val_rows)}")

    if len(val_rows) != VAL_CASES:
        raise RuntimeError(
            f"Expected {VAL_CASES} validation cases; "
            f"got {len(val_rows)}."
        )

    if PART15_VAL_COHORT.exists():
        saved = pd.read_csv(PART15_VAL_COHORT)

        reconstructed_ids = list(
            zip(
                val_rows["study_id"].astype(str),
                val_rows["series_id"].astype(str),
            )
        )

        saved_ids = list(
            zip(
                saved["study_id"].astype(str),
                saved["series_id"].astype(str),
            )
        )

        exact_match = reconstructed_ids == saved_ids

        print(
            f"Saved Part 15 cohort rows             : {len(saved)}"
        )
        print(
            f"Exact row-order cohort match           : "
            f"{exact_match}"
        )

        if not exact_match:
            overlap = len(
                set(reconstructed_ids) & set(saved_ids)
            )
            print(
                f"Cohort pair overlap                    : "
                f"{overlap}/{VAL_CASES}"
            )

    # -------------------------------------------------------------------------
    # PART 15 HISTORY
    # -------------------------------------------------------------------------
    section("PART 15 RECORDED METRIC")

    history = pd.read_csv(PART15_HISTORY)

    print(f"History rows                          : {len(history)}")

    if "val_dice" in history.columns:
        recorded_best = float(history["val_dice"].max())
        print(
            f"Recorded best validation Dice        : "
            f"{recorded_best:.6f}"
        )
    else:
        recorded_best = float("nan")
        print("val_dice column not found.")

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    section("LOADING SAVED PART 15 BEST CHECKPOINT")

    model = part11.create_model(device).to(device)
    checkpoint = load_checkpoint(
        model,
        device,
    )
    model.eval()

    print(
        f"Total parameters                      : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    checkpoint_epoch = (
        checkpoint.get("epoch")
        if isinstance(checkpoint, dict)
        else None
    )

    checkpoint_dice = (
        checkpoint.get("best_val_dice")
        if isinstance(checkpoint, dict)
        else None
    )

    print(f"Checkpoint epoch                      : {checkpoint_epoch}")
    print(f"Checkpoint stored best Dice           : {checkpoint_dice}")

    # -------------------------------------------------------------------------
    # PART 9
    # -------------------------------------------------------------------------
    section("LOADING PART 9 THROUGH PART 11")

    part9 = part11.load_part9_module()
    print("✓ Part 9 loader imported through Part 11.")

    # -------------------------------------------------------------------------
    # RECONSTRUCTION
    # -------------------------------------------------------------------------
    section("RUNNING EXACT SAVED-CHECKPOINT RECONSTRUCTION")

    case_rows = []
    class_acc = {
        c: {
            "target": 0,
            "prediction": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for c in FOREGROUND
    }

    failed = []

    with torch.no_grad():
        for i, (_, row) in enumerate(
            val_rows.iterrows(),
            start=1,
        ):
            study, series = ids(row)

            try:
                image, mask, info = load_case(
                    part11,
                    part9,
                    row,
                )

                image = image.to(
                    device,
                    non_blocking=True,
                )

                mask_device = mask.to(
                    device,
                    non_blocking=True,
                )

                logits = model(image)

                prediction = torch.argmax(
                    logits,
                    dim=1,
                ).squeeze(0)

                current_mean, current_class = (
                    conventional_foreground_dice(
                        prediction,
                        mask_device,
                    )
                )

                compatible_mean, compatible_class = (
                    three_argument_part15_compatible_metric(
                        logits,
                        mask_device.unsqueeze(0),
                        NUM_CLASSES,
                    )
                )

                pred_fg = int(
                    sum(
                        int((prediction == c).sum().item())
                        for c in FOREGROUND
                    )
                )

                target_fg = int(
                    sum(
                        int((mask_device == c).sum().item())
                        for c in FOREGROUND
                    )
                )

                case_record = {
                    "case_index": i,
                    "study_id": study,
                    "series_id": series,
                    "status": "OK",
                    "target_foreground_voxels": target_fg,
                    "prediction_foreground_voxels": pred_fg,
                    "prediction_target_foreground_ratio": (
                        pred_fg / target_fg
                        if target_fg else float("nan")
                    ),
                    "current_part11_mean_dice": current_mean,
                    "three_argument_compatible_mean_dice": compatible_mean,
                    "unique_prediction_labels": ",".join(
                        map(
                            str,
                            sorted(
                                int(x)
                                for x in torch.unique(
                                    prediction
                                ).cpu()
                            ),
                        )
                    ),
                }

                for c in FOREGROUND:
                    p = prediction == c
                    t = mask_device == c

                    tp = int((p & t).sum().item())
                    fp = int((p & ~t).sum().item())
                    fn = int((~p & t).sum().item())

                    class_acc[c]["target"] += int(t.sum().item())
                    class_acc[c]["prediction"] += int(p.sum().item())
                    class_acc[c]["tp"] += tp
                    class_acc[c]["fp"] += fp
                    class_acc[c]["fn"] += fn

                    case_record[f"class_{c}_dice"] = (
                        current_class[c]
                    )

                case_rows.append(case_record)

                if (
                    i <= 5
                    or i % 25 == 0
                    or i == len(val_rows)
                ):
                    print(
                        f"  [{i:03d}/{len(val_rows)}] "
                        f"{study} | {series} | "
                        f"pred_fg={pred_fg} | "
                        f"target_fg={target_fg} | "
                        f"Dice={current_mean:.6f} | "
                        f"labels=[{case_record['unique_prediction_labels']}]"
                    )

            except Exception as exc:
                failed.append({
                    "case_index": i,
                    "study_id": study,
                    "series_id": series,
                    "status": "ERROR",
                    "error": repr(exc),
                })

                case_rows.append(
                    failed[-1]
                )

                print(
                    f"  [{i:03d}/{len(val_rows)}] "
                    f"{study} | {series} | ERROR: {exc}"
                )

    # -------------------------------------------------------------------------
    # AGGREGATE
    # -------------------------------------------------------------------------
    section("PART 36 OVERALL DIAGNOSIS")

    successful = len(case_rows) - len(failed)

    total_target_fg = sum(
        class_acc[c]["target"]
        for c in FOREGROUND
    )

    total_prediction_fg = sum(
        class_acc[c]["prediction"]
        for c in FOREGROUND
    )

    aggregate_class = {}
    aggregate_dices = []

    for c in FOREGROUND:
        a = class_acc[c]
        denominator = 2 * a["tp"] + a["fp"] + a["fn"]

        d = (
            float("nan")
            if denominator == 0
            else 2.0 * a["tp"] / denominator
        )

        aggregate_class[c] = d

        if not np.isnan(d):
            aggregate_dices.append(d)

    aggregate_dice = (
        float(np.mean(aggregate_dices))
        if aggregate_dices else float("nan")
    )

    fg_ratio = (
        total_prediction_fg / total_target_fg
        if total_target_fg
        else float("nan")
    )

    all_background = (
        total_prediction_fg == 0
    )

    if not part15_signature_match:
        if all_background:
            diagnosis = (
                "PART15_METRIC_API_MISMATCH_AND_SAVED_CHECKPOINT_FOREGROUND_COLLAPSE"
            )
        else:
            diagnosis = (
                "PART15_METRIC_API_MISMATCH_REQUIRES_METRIC_RECONCILIATION"
            )
    elif all_background:
        diagnosis = "SAVED_CHECKPOINT_FOREGROUND_COLLAPSE"
    else:
        diagnosis = "PART15_METRIC_API_COMPATIBLE"

    print(f"Successful cases                     : {successful}")
    print(f"Failed cases                         : {len(failed)}")
    print(f"Target foreground voxels             : {total_target_fg:,}")
    print(f"Predicted foreground voxels          : {total_prediction_fg:,}")
    print(
        f"Prediction / target foreground ratio : "
        f"{fg_ratio:.6f}"
    )
    print(
        f"Aggregate conventional foreground Dice: "
        f"{aggregate_dice:.6f}"
    )
    print(
        f"Part 15 recorded best Dice            : "
        f"{recorded_best:.6f}"
    )
    print(
        f"Checkpoint stored best Dice           : "
        f"{checkpoint_dice}"
    )
    print(
        f"Part 15 evaluate() API compatible     : "
        f"{part15_signature_match}"
    )
    print(
        f"Saved checkpoint all-background        : "
        f"{all_background}"
    )
    print(f"DIAGNOSIS                            : {diagnosis}")

    # -------------------------------------------------------------------------
    # CLASS SUMMARY
    # -------------------------------------------------------------------------
    section("CLASS-WISE RECONSTRUCTION")

    class_rows = []

    for c in FOREGROUND:
        a = class_acc[c]
        denominator = (
            2 * a["tp"] +
            a["fp"] +
            a["fn"]
        )

        d = (
            float("nan")
            if denominator == 0
            else 2.0 * a["tp"] / denominator
        )

        p = (
            a["tp"] / (a["tp"] + a["fp"])
            if a["tp"] + a["fp"]
            else 0.0
        )

        r = (
            a["tp"] / (a["tp"] + a["fn"])
            if a["tp"] + a["fn"]
            else 0.0
        )

        class_rows.append({
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "target_voxels": a["target"],
            "prediction_voxels": a["prediction"],
            "tp": a["tp"],
            "fp": a["fp"],
            "fn": a["fn"],
            "dice": d,
            "precision": p,
            "recall": r,
        })

    class_df = pd.DataFrame(class_rows)

    print(
        class_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # API DETAIL
    # -------------------------------------------------------------------------
    section("API MISMATCH DETAIL")

    if not part15_signature_match:
        print(
            "The Part 15 source calls dice_from_prediction() "
            "with 3 positional arguments."
        )
        print(
            "The current corrected Part 11 API accepts 2 positional "
            "arguments: logits, target."
        )
        print()
        print(
            "The Part 15 source also expects class_dice to support "
            ".get(class_id), while the current corrected Part 11 "
            "implementation returns a list of foreground class Dice values."
        )
        print()
        print(
            "Therefore the recorded Part 15 0.668 metric cannot be "
            "blindly treated as the metric produced by the current "
            "corrected Part 11 implementation."
        )
    else:
        print(
            "No positional-argument mismatch detected in Part 15 evaluate()."
        )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    section("SAVING PART 36 OUTPUTS")

    case_df = pd.DataFrame(case_rows)
    case_df.to_csv(
        CASE_CSV,
        index=False,
    )

    class_df.to_csv(
        CLASS_CSV,
        index=False,
    )

    summary = {
        "phase": "Phase 4 - Part 36",
        "description": (
            "Exact Part 15 validation metric reconstruction and "
            "Part 11 API compatibility audit"
        ),
        "validation_cases_requested": VAL_CASES,
        "successful_cases": successful,
        "failed_cases": len(failed),
        "part15_recorded_best_dice": (
            None if np.isnan(recorded_best)
            else float(recorded_best)
        ),
        "checkpoint_stored_best_dice": (
            float(checkpoint_dice)
            if checkpoint_dice is not None
            else None
        ),
        "reconstructed_conventional_foreground_dice": (
            None if np.isnan(aggregate_dice)
            else float(aggregate_dice)
        ),
        "target_foreground_voxels": int(total_target_fg),
        "prediction_foreground_voxels": int(total_prediction_fg),
        "prediction_target_foreground_ratio": float(fg_ratio),
        "part15_evaluate_metric_api_compatible": bool(
            part15_signature_match
        ),
        "saved_checkpoint_all_background": bool(
            all_background
        ),
        "diagnosis": diagnosis,
        "checkpoint_epoch": checkpoint_epoch,
        "patch_size": list(PATCH_SIZE),
        "num_classes": NUM_CLASSES,
        "training_performed": False,
        "model_weights_modified": False,
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "spider_used": False,
        "rsna_test_set_used": False,
        "pseudo_masks_are_manual_ground_truth": False,
        "class_summary": class_rows,
        "failed_cases": failed,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "PHASE 4 - PART 36",
        "EXACT PART 15 VALIDATION-METRIC RECONSTRUCTION AUDIT",
        "=" * 78,
        "",
        f"Successful cases: {successful}",
        f"Failed cases: {len(failed)}",
        f"Part 15 recorded best Dice: {recorded_best}",
        f"Checkpoint stored best Dice: {checkpoint_dice}",
        f"Reconstructed conventional foreground Dice: {aggregate_dice}",
        f"Target foreground voxels: {total_target_fg}",
        f"Prediction foreground voxels: {total_prediction_fg}",
        f"Prediction/target foreground ratio: {fg_ratio}",
        f"Part 15 evaluate API compatible: {part15_signature_match}",
        f"Saved checkpoint all background: {all_background}",
        f"DIAGNOSIS: {diagnosis}",
        "",
        "CLASS SUMMARY",
        "-" * 78,
        class_df.to_string(index=False),
        "",
        "API DETAIL",
        "-" * 78,
        (
            "Part 15 evaluate() calls dice_from_prediction with "
            f"{metric_calls[0]['positional_arguments'] if metric_calls else 'unknown'} "
            "positional arguments."
        ),
        (
            "Current corrected Part 11 signature: "
            f"{inspect.signature(part11.dice_from_prediction)}"
        ),
        (
            "Current corrected Part 11 returns mean foreground Dice "
            "plus a list of class Dice values."
        ),
    ]

    REPORT_TXT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"Case audit CSV                        : {CASE_CSV}")
    print(f"Class summary CSV                     : {CLASS_CSV}")
    print(f"Summary JSON                          : {SUMMARY_JSON}")
    print(f"Report TXT                            : {REPORT_TXT}")

    section("PHASE 4 - PART 36 COMPLETE")
    print("✓ Part 15 validation source inspected.")
    print("✓ Part 15 Dice API calls inspected.")
    print("✓ Current corrected Part 11 API inspected.")
    print("✓ Exact 100-case validation cohort reconstructed.")
    print("✓ Saved Part 15 checkpoint evaluated.")
    print("✓ Conventional foreground Dice reconstructed.")
    print("✓ Class-wise TP / FP / FN reconstructed.")
    print("✓ API compatibility diagnosis recorded.")
    print("✓ No training performed.")
    print("✓ No model weights modified.")
    print("✓ No optimizer created.")
    print("✓ No optimizer step performed.")
    print("✓ SPIDER not used.")
    print("✓ RSNA test set not used.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("PART 36 ERROR")
        print("=" * 78)
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
