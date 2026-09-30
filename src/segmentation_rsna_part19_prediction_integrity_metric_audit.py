"""
==============================================================================
PHASE 4 - PART 19
RSNA-ONLY PREDICTION INTEGRITY / DICE METRIC AUDIT
==============================================================================

Purpose
-------
Audit the apparent contradiction found in Part 18:

    - mean Dice = 0.668
    - precision = 0
    - recall = 0
    - empty foreground predictions = 100/100

This stage performs EVALUATION ONLY.

It:
    1. Reconstructs the exact Part 15/16/18 100-case validation cohort.
    2. Loads the verified Part 15 best checkpoint.
    3. Uses the validated Part 11 preprocessing.
    4. Computes argmax predictions explicitly.
    5. Counts predicted/target foreground voxels.
    6. Computes TP/FP/FN explicitly for every foreground class.
    7. Computes conventional foreground Dice from TP/FP/FN.
    8. Separately reports the legacy/recorded Part 18 Dice.
    9. Detects impossible/inconsistent metric combinations.
   10. Saves case-level, class-level and audit summaries.

NO TRAINING.
NO WEIGHT UPDATES.
NO SPIDER.
NO RSNA TEST SET.
"""

from __future__ import annotations

import json
import random
import math
import sys
import traceback
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR


# =============================================================================
# PATHS / CONFIG
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
    SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "checkpoints"
    / "best_model.pth"
)

PART16_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part16_best_checkpoint_validation"
)

PART18_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part18_expanded_validation_stability_analysis"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part19_prediction_integrity_metric_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

SEED = 42
VALIDATION_CASES = 100
PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12
NUM_CLASSES = 6
FOREGROUND_CLASSES = [1, 2, 3, 4, 5]

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

AMP_ENABLED = True


# =============================================================================
# UTILITIES
# =============================================================================

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def import_module(path: Path, name: str):
    spec = spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def get_case_ids(row):
    def norm(v):
        if pd.isna(v):
            return ""
        try:
            return str(int(float(v)))
        except Exception:
            return str(v)

    return norm(row.get("study_id")), norm(row.get("series_id"))


def ensure_image_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    x = x.float()

    # Native [D,H,W] -> [1,1,D,H,W]
    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    # [1,D,H,W] -> [1,1,D,H,W]
    elif x.ndim == 4 and x.shape[0] == 1:
        x = x.unsqueeze(0)
    # already [B,C,D,H,W]
    elif x.ndim != 5:
        raise ValueError(f"Unexpected image shape: {tuple(x.shape)}")

    return x.contiguous()


def ensure_mask_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    x = x.long()

    if x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {tuple(x.shape)}")

    return x.contiguous()


def resize_volume_and_mask(image, mask):
    image = ensure_image_tensor(image)
    mask = ensure_mask_tensor(mask)

    image = F.interpolate(
        image,
        size=PATCH_SIZE,
        mode="trilinear",
        align_corners=False,
    )

    mask = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=PATCH_SIZE,
        mode="nearest",
    ).squeeze(0).squeeze(0).long()

    return image, mask


def load_part9(part11):
    if hasattr(part11, "load_part9_module"):
        return part11.load_part9_module()

    if hasattr(part11, "load_part9"):
        return part11.load_part9()

    path = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"
    return import_module(path, "segmentation_rsna_part9_for_part19")


def load_case_exact(row, part11, part9):
    """
    Reuse corrected Part 11 loading/preprocessing contracts.

    Part 11's load_tensor_case may internally use its robust DICOM fallback.
    Its preprocess_case expects native 3-D image/mask arrays.
    """
    image, mask, info = part11.load_tensor_case(row, part9)

    # Convert loader output to native 3-D arrays for exact preprocessing API.
    if isinstance(image, torch.Tensor):
        if image.ndim == 5:
            image = image.squeeze(0).squeeze(0)
        elif image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)
        image_np = image.detach().cpu().numpy()
    else:
        image_np = np.asarray(image)

    if isinstance(mask, torch.Tensor):
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    try:
        processed = part11.preprocess_case(
            image_np,
            mask_np,
        )

        if isinstance(processed, tuple):
            proc_image = processed[0]
            proc_mask = processed[1] if len(processed) > 1 else mask_np
        else:
            proc_image = processed
            proc_mask = mask_np

    except TypeError:
        proc_image = image_np
        proc_mask = mask_np

    image_t = ensure_image_tensor(proc_image)
    mask_t = ensure_mask_tensor(proc_mask)

    # Enforce model contract only if preprocessing did not return the target.
    if tuple(image_t.shape[-3:]) != PATCH_SIZE or tuple(mask_t.shape) != PATCH_SIZE:
        image_t, mask_t = resize_volume_and_mask(image_t, mask_t)

    return image_t, mask_t, info


def create_model():
    return SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        use_checkpoint=False,
    )


def load_checkpoint(model, device):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    return checkpoint, missing, unexpected


# =============================================================================
# EXPLICIT METRICS
# =============================================================================

def counts_for_class(pred, target, class_id):
    p = pred == class_id
    t = target == class_id

    tp = int((p & t).sum().item())
    fp = int((p & ~t).sum().item())
    fn = int((~p & t).sum().item())

    return tp, fp, fn


def conventional_dice(tp, fp, fn):
    denom = 2 * tp + fp + fn

    # Both prediction and target are empty for this class.
    # Report NaN here so an absent class cannot artificially inflate
    # foreground performance.
    if denom == 0:
        return float("nan")

    return (2.0 * tp) / denom


def precision(tp, fp):
    denom = tp + fp
    if denom == 0:
        return 0.0
    return tp / denom


def recall(tp, fn):
    denom = tp + fn
    if denom == 0:
        return 0.0
    return tp / denom


def audit_prediction(pred, target):
    pred = pred.long()
    target = target.long()

    pred_fg = int((pred > 0).sum().item())
    target_fg = int((target > 0).sum().item())

    unique_pred = sorted(int(x) for x in torch.unique(pred).cpu().tolist())
    unique_target = sorted(int(x) for x in torch.unique(target).cpu().tolist())

    class_rows = []
    dice_values = []
    precision_values = []
    recall_values = []

    for cid in FOREGROUND_CLASSES:
        tp, fp, fn = counts_for_class(pred, target, cid)
        d = conventional_dice(tp, fp, fn)
        p = precision(tp, fp)
        r = recall(tp, fn)

        if not math_is_nan(d):
            dice_values.append(d)

        precision_values.append(p)
        recall_values.append(r)

        class_rows.append({
            "class_id": cid,
            "class_name": CLASS_NAMES[cid],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "target_voxels": tp + fn,
            "prediction_voxels": tp + fp,
            "dice": d,
            "precision": p,
            "recall": r,
        })

    mean_dice = float(np.mean(dice_values)) if dice_values else float("nan")

    return {
        "foreground_prediction_voxels": pred_fg,
        "foreground_target_voxels": target_fg,
        "empty_prediction": pred_fg == 0,
        "unique_prediction_labels": ",".join(map(str, unique_pred)),
        "unique_target_labels": ",".join(map(str, unique_target)),
        "mean_foreground_dice_conventional": mean_dice,
        "mean_foreground_precision": float(np.mean(precision_values)),
        "mean_foreground_recall": float(np.mean(recall_values)),
        "class_rows": class_rows,
    }


def math_is_nan(x):
    return bool(np.isnan(x))


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        section("PHASE 4 - PART 19")
        print("RSNA-ONLY PREDICTION INTEGRITY / DICE METRIC AUDIT")
        print()
        print("Evaluation only.")
        print("No training is performed.")
        print("No model weights are modified.")
        print("SPIDER is not used.")
        print("RSNA test set is not used.")
        print()
        print("Purpose:")
        print("Audit the Part 18 combination of:")
        print("  Dice=0.668, Precision=0, Recall=0, Empty predictions=100.")

        section("PATH VALIDATION")
        paths = {
            "RSNA root": RSNA_ROOT,
            "validation manifest": VAL_MANIFEST,
            "Part 11 corrected source": PART11_SOURCE,
            "Part 15 checkpoint": CHECKPOINT,
            "Part 16 directory": PART16_DIR,
            "Part 18 directory": PART18_DIR,
        }

        for name, path in paths.items():
            print(f"{name:<36}: {'FOUND' if path.exists() else 'MISSING'}")

        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing required Part 19 input(s):\n" + "\n".join(missing))

        section("PYTORCH / GPU ENVIRONMENT")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch version                       : {torch.__version__}")
        print(f"CUDA available                        : {torch.cuda.is_available()}")
        print(f"Device                                : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
        if device.type == "cuda":
            print(f"GPU memory                            : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Patch size                            : {PATCH_SIZE}")
        print(f"Feature size                          : {FEATURE_SIZE}")
        print(f"Classes                               : {NUM_CLASSES}")
        print(f"Validation cases                      : {VALIDATION_CASES}")

        section("IMPORTING VALIDATED PART 11 IMPLEMENTATION")
        part11 = import_module(
            PART11_SOURCE,
            "segmentation_rsna_part11_corrected_for_part19",
        )
        print("✓ Corrected Part 11 imported.")

        if not hasattr(part11, "select_pilot_rows"):
            raise AttributeError("Part 11 select_pilot_rows API missing.")
        if not hasattr(part11, "load_tensor_case"):
            raise AttributeError("Part 11 load_tensor_case API missing.")
        if not hasattr(part11, "preprocess_case"):
            raise AttributeError("Part 11 preprocess_case API missing.")

        print("✓ Exact cohort API available.")
        print("✓ Exact loading API available.")
        print("✓ Exact preprocessing API available.")

        section("LOADING PART 9")
        part9 = load_part9(part11)
        print("✓ Part 9 loader imported.")

        section("RECONSTRUCTING EXACT 100-CASE COHORT")
        cohort = part11.select_pilot_rows(
            str(VAL_MANIFEST),
            VALIDATION_CASES,
            SEED,
        ).reset_index(drop=True)

        print(f"Validation manifest total : {len(pd.read_csv(VAL_MANIFEST))}")
        print(f"Part 19 validation cohort : {len(cohort)}")

        if len(cohort) != VALIDATION_CASES:
            raise RuntimeError("Unexpected validation cohort size.")

        section("LOADING PART 15 BEST CHECKPOINT")
        model = create_model().to(device)
        checkpoint, missing, unexpected = load_checkpoint(model, device)
        model.eval()

        print("✓ Swin-UNETR created.")
        print(f"Total parameters : {sum(p.numel() for p in model.parameters()):,}")
        print(f"Missing keys     : {len(missing)}")
        print(f"Unexpected keys  : {len(unexpected)}")

        if isinstance(checkpoint, dict):
            print(f"Checkpoint epoch : {checkpoint.get('epoch', 'N/A')}")
            print(f"Checkpoint best Dice : {checkpoint.get('best_val_dice', 'N/A')}")

        loss_fn = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        )

        section("STARTING EXPLICIT PREDICTION INTEGRITY AUDIT")

        case_rows = []
        class_rows_all = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for i, row in cohort.iterrows():
            study_id, series_id = get_case_ids(row)

            image, mask, info = load_case_exact(
                row,
                part11,
                part9,
            )

            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.no_grad():
                if device.type == "cuda" and AMP_ENABLED:
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        logits = model(image)
                        loss = loss_fn(
                            logits,
                            mask.unsqueeze(0).unsqueeze(1),
                        )
                else:
                    logits = model(image)
                    loss = loss_fn(
                        logits,
                        mask.unsqueeze(0).unsqueeze(1),
                    )

            prediction = torch.argmax(logits, dim=1).squeeze(0)

            audit = audit_prediction(prediction, mask)

            case_rows.append({
                "index": i,
                "study_id": study_id,
                "series_id": series_id,
                "loss": float(loss.item()),
                **{k: v for k, v in audit.items() if k != "class_rows"},
            })

            for cr in audit["class_rows"]:
                class_rows_all.append({
                    "index": i,
                    "study_id": study_id,
                    "series_id": series_id,
                    **cr,
                })

            if i < 5 or (i + 1) % 25 == 0 or i + 1 == len(cohort):
                print(
                    f"  [{i+1:03d}/{len(cohort)}] "
                    f"{study_id} | {series_id} | "
                    f"pred_fg={audit['foreground_prediction_voxels']} | "
                    f"target_fg={audit['foreground_target_voxels']} | "
                    f"labels=[{audit['unique_prediction_labels']}] | "
                    f"Dice={audit['mean_foreground_dice_conventional']:.6f}"
                )

        case_df = pd.DataFrame(case_rows)
        class_df = pd.DataFrame(class_rows_all)

        section("PART 19 EXPLICIT METRIC SUMMARY")

        empty_count = int(case_df["empty_prediction"].sum())
        mean_dice = float(case_df["mean_foreground_dice_conventional"].mean())
        mean_precision = float(case_df["mean_foreground_precision"].mean())
        mean_recall = float(case_df["mean_foreground_recall"].mean())

        print(f"Validation cases                 : {len(case_df)}")
        print(f"Mean conventional foreground Dice: {mean_dice:.6f}")
        print(f"Mean precision                   : {mean_precision:.6f}")
        print(f"Mean recall                      : {mean_recall:.6f}")
        print(f"Empty foreground predictions     : {empty_count}")
        print(f"Non-empty foreground predictions : {len(case_df) - empty_count}")

        print("\nPrediction label distribution:")
        label_counts = {}
        for labels in case_df["unique_prediction_labels"]:
            for x in str(labels).split(","):
                label_counts[x] = label_counts.get(x, 0) + 1

        for label, count in sorted(label_counts.items(), key=lambda z: int(z[0])):
            print(f"  Label {label}: present in {count}/{len(case_df)} cases")

        section("PER-CLASS EXPLICIT METRICS")

        class_summary = []
        for cid in FOREGROUND_CLASSES:
            sub = class_df[class_df["class_id"] == cid].copy()

            class_summary.append({
                "class_id": cid,
                "class_name": CLASS_NAMES[cid],
                "mean_dice": float(sub["dice"].mean()),
                "mean_precision": float(sub["precision"].mean()),
                "mean_recall": float(sub["recall"].mean()),
                "mean_target_voxels": float(sub["target_voxels"].mean()),
                "mean_prediction_voxels": float(sub["prediction_voxels"].mean()),
                "zero_prediction_cases": int((sub["prediction_voxels"] == 0).sum()),
                "zero_dice_cases": int((sub["dice"].fillna(0) == 0).sum()),
            })

            print(f"{cid}: {CLASS_NAMES[cid]}")
            print(f"  Dice       : {class_summary[-1]['mean_dice']:.6f}")
            print(f"  Precision  : {class_summary[-1]['mean_precision']:.6f}")
            print(f"  Recall     : {class_summary[-1]['mean_recall']:.6f}")
            print(f"  Mean target voxels     : {class_summary[-1]['mean_target_voxels']:.1f}")
            print(f"  Mean prediction voxels: {class_summary[-1]['mean_prediction_voxels']:.1f}")
            print(f"  Zero-prediction cases : {class_summary[-1]['zero_prediction_cases']}")

        class_summary_df = pd.DataFrame(class_summary)

        section("INCONSISTENCY DETECTION")

        impossible_mask = (
            (case_df["empty_prediction"])
            & (case_df["mean_foreground_precision"] == 0)
            & (case_df["mean_foreground_recall"] == 0)
            & (case_df["foreground_target_voxels"] > 0)
            & (case_df["mean_foreground_dice_conventional"] > 0)
        )

        impossible_count = int(impossible_mask.sum())

        # Load Part 18 recorded metrics if available.
        part18_csv = PART18_DIR / "part18_expanded_validation_case_metrics.csv"
        legacy_mean = None

        if part18_csv.exists():
            p18 = pd.read_csv(part18_csv)

            # Try common possible names.
            candidates = [
                "dice",
                "mean_foreground_dice",
                "mean_dice",
                "dice_part18",
            ]
            dice_col = next((c for c in candidates if c in p18.columns), None)

            if dice_col is not None:
                legacy_mean = float(p18[dice_col].mean())
                print(f"Part 18 recorded Dice column : {dice_col}")
                print(f"Part 18 recorded mean Dice   : {legacy_mean:.6f}")

        print(f"Cases with mathematically inconsistent metrics : {impossible_count}")

        if impossible_count > 0:
            diagnosis = "METRIC_IMPLEMENTATION_INCONSISTENCY"
            print("DIAGNOSIS: explicit TP/FP/FN metrics contradict the reported Dice.")
        elif empty_count == len(case_df):
            diagnosis = "MODEL_FOREGROUND_COLLAPSE_OR_THRESHOLDING_ISSUE"
            print("DIAGNOSIS: all predictions are foreground-empty; inspect checkpoint/model outputs.")
        else:
            diagnosis = "NO_INTERNAL_METRIC_CONTRADICTION_DETECTED"
            print("DIAGNOSIS: no direct contradiction detected.")

        if legacy_mean is not None:
            abs_diff = abs(legacy_mean - mean_dice)
            print(f"Legacy Part 18 vs explicit Dice difference : {abs_diff:.6f}")
        else:
            abs_diff = None

        section("SAVING PART 19 RESULTS")

        case_path = OUTPUT_DIR / "part19_case_prediction_integrity_metrics.csv"
        class_path = OUTPUT_DIR / "part19_class_prediction_integrity_metrics.csv"
        summary_path = OUTPUT_DIR / "part19_class_summary.csv"
        audit_json = OUTPUT_DIR / "phase4_part19_prediction_integrity_audit_summary.json"
        report_path = OUTPUT_DIR / "reports" / "phase4_part19_prediction_integrity_audit_report.txt"

        case_df.to_csv(case_path, index=False)
        class_df.to_csv(class_path, index=False)
        class_summary_df.to_csv(summary_path, index=False)

        peak_alloc = None
        peak_reserved = None
        if device.type == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

        summary = {
            "phase": "4 - Part 19",
            "validation_cases": int(len(case_df)),
            "mean_conventional_foreground_dice": mean_dice,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "empty_prediction_cases": empty_count,
            "non_empty_prediction_cases": int(len(case_df) - empty_count),
            "inconsistent_metric_cases": impossible_count,
            "part18_recorded_mean_dice": legacy_mean,
            "part18_vs_explicit_dice_absolute_difference": abs_diff,
            "diagnosis": diagnosis,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "checkpoint_best_dice": checkpoint.get("best_val_dice") if isinstance(checkpoint, dict) else None,
            "patch_size": PATCH_SIZE,
            "feature_size": FEATURE_SIZE,
            "num_classes": NUM_CLASSES,
            "spider_used": False,
            "test_set_used": False,
            "training_performed": False,
            "model_weights_changed": False,
            "peak_gpu_allocated_gb": peak_alloc,
            "peak_gpu_reserved_gb": peak_reserved,
        }

        audit_json.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        report_lines = [
            "PHASE 4 - PART 19",
            "RSNA-ONLY PREDICTION INTEGRITY / DICE METRIC AUDIT",
            "",
            f"Validation cases: {len(case_df)}",
            f"Explicit mean foreground Dice: {mean_dice:.6f}",
            f"Mean precision: {mean_precision:.6f}",
            f"Mean recall: {mean_recall:.6f}",
            f"Empty prediction cases: {empty_count}",
            f"Inconsistent metric cases: {impossible_count}",
            f"Part 18 recorded mean Dice: {legacy_mean}",
            f"Diagnosis: {diagnosis}",
            "",
            "IMPORTANT:",
            "Metrics are against RSNA point-derived pseudo-masks, not manual clinical ground truth.",
            "No training, test evaluation, SPIDER data, or weight modification was performed.",
        ]

        report_path.write_text(
            "\n".join(report_lines),
            encoding="utf-8",
        )

        print(f"Saved: {case_path}")
        print(f"Saved: {class_path}")
        print(f"Saved: {summary_path}")
        print(f"Saved: {audit_json}")
        print(f"Saved: {report_path}")

        section("PART 19 FINAL SUMMARY")
        print(f"Validation cases       : {len(case_df)}")
        print(f"Explicit mean Dice     : {mean_dice:.6f}")
        print(f"Mean precision         : {mean_precision:.6f}")
        print(f"Mean recall            : {mean_recall:.6f}")
        print(f"Empty predictions      : {empty_count}")
        print(f"Metric contradictions : {impossible_count}")
        print(f"Diagnosis              : {diagnosis}")

        if impossible_count > 0:
            print("\nFINAL DECISION")
            print("ACTION REQUIRED - Part 18's Dice metric is inconsistent with")
            print("the explicit TP/FP/FN prediction counts. Do not report 0.668")
            print("as segmentation performance until the metric implementation")
            print("is resolved.")
        elif empty_count == len(case_df):
            print("\nFINAL DECISION")
            print("CAUTION - all validation predictions are foreground-empty.")
            print("Investigate model output/logit behavior before further training.")
        else:
            print("\nFINAL DECISION")
            print("PASS - explicit prediction integrity and metric calculations")
            print("are internally consistent.")

        print("\nSPIDER used          : NO")
        print("Test set used        : NO")
        print("Training performed   : NO")
        print("Model weights changed: NO")

        section("PHASE 4 - PART 19 COMPLETE")

    except Exception as exc:
        print("\n" + "=" * 78)
        print("PART 19 ERROR")
        print("=" * 78)
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
