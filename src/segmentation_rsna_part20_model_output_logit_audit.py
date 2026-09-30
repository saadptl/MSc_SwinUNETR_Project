"""
PHASE 4 - PART 20
RSNA-ONLY MODEL OUTPUT / LOGIT BEHAVIOR AUDIT

Purpose:
Investigate the Part 19 finding that the Part 15 checkpoint produces
foreground-empty argmax predictions on all 100 validation cases.

Evaluation only.
No training.
No weight updates.
RSNA only.
SPIDER is not used.
Test set is not used.

This audit does NOT change the checkpoint or the model.

It examines:
1. Raw logits per class.
2. Softmax probabilities.
3. Argmax label distribution.
4. Foreground-vs-background probability.
5. Per-class maximum probability.
6. Probability margins between background and best foreground class.
7. Whether the output is numerically collapsed.
8. Whether alternative probability thresholds would recover foreground.
9. Consistency with the Part 19 explicit foreground result.

The objective is to distinguish:
    MODEL_FOREGROUND_COLLAPSE
from:
    ARGMAX_BACKGROUND_DOMINANCE
from:
    OUTPUT/POSTPROCESSING_ISSUE
from:
    HEALTHY_NON-COLLAPSED_LOGITS_WITH_PSEUDO-MASK_MISMATCH
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

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

PART11_SOURCE = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"

PART15_CHECKPOINT = (
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

PART19_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part19_prediction_integrity_metric_audit"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part20_model_output_logit_audit"
)

CASE_CSV = OUT_DIR / "part20_case_logit_metrics.csv"
CLASS_CSV = OUT_DIR / "part20_class_logit_summary.csv"
THRESHOLD_CSV = OUT_DIR / "part20_threshold_sensitivity.csv"
SUMMARY_JSON = OUT_DIR / "phase4_part20_model_output_logit_audit_summary.json"
REPORT_TXT = OUT_DIR / "reports" / "phase4_part20_model_output_logit_audit_report.txt"


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12
IN_CHANNELS = 1
NUM_CLASSES = 6
BATCH_SIZE = 1
AMP_ENABLED = True
SEED = 42
VAL_CASES = 100

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_paths() -> None:
    paths = {
        "RSNA root": RSNA_ROOT,
        "validation manifest": VAL_MANIFEST,
        "Part 11 corrected source": PART11_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 16 directory": PART16_DIR,
        "Part 18 directory": PART18_DIR,
        "Part 19 directory": PART19_DIR,
    }

    missing = []

    for name, path in paths.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{name:<35}: {status}")
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required Part 20 input(s):\n" + "\n".join(missing)
        )


def import_part11():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "part11_corrected_for_part20",
        str(PART11_SOURCE),
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {PART11_SOURCE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "select_pilot_rows",
        "load_part9_module",
        "load_tensor_case",
        "preprocess_case",
        "create_model",
        "dice_from_prediction",
    ]

    for name in required:
        if not hasattr(module, name):
            raise AttributeError(
                f"Corrected Part 11 does not expose required API: {name}"
            )

    return module


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def get_state_dict(checkpoint: dict) -> dict:
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    raise KeyError("Could not find model state_dict in checkpoint.")


def create_model(part11, device: torch.device):
    model = part11.create_model(
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
    )
    return model.to(device)


def scalar(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def safe_mean(values):
    values = [float(x) for x in values if np.isfinite(x)]
    return float(np.mean(values)) if values else 0.0


def safe_std(values):
    values = [float(x) for x in values if np.isfinite(x)]
    return float(np.std(values)) if values else 0.0


def safe_median(values):
    values = [float(x) for x in values if np.isfinite(x)]
    return float(np.median(values)) if values else 0.0


# ---------------------------------------------------------------------
# ROBUST MODEL INPUT / OUTPUT HELPERS
# ---------------------------------------------------------------------

def ensure_image_5d(image: torch.Tensor) -> torch.Tensor:
    """
    Convert common image layouts to [B,C,D,H,W].

    Expected final form:
        [1,1,64,96,96]
    """
    image = image.float()

    if image.ndim == 3:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 4:
        # [C,D,H,W] -> [B,C,D,H,W]
        image = image.unsqueeze(0)
    elif image.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {tuple(image.shape)}")

    return image


def ensure_mask_4d(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        return mask.unsqueeze(0).long()
    if mask.ndim == 4:
        return mask.long()
    raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")


def normalize_processed_case(processed):
    """
    Accept the known Part 11 return styles.

    Expected:
        image, mask
    or:
        image, mask, info
    """
    if isinstance(processed, (tuple, list)):
        if len(processed) >= 2:
            image = processed[0]
            mask = processed[1]
            info = processed[2] if len(processed) >= 3 else {}
            return image, mask, info

    if isinstance(processed, dict):
        image = processed.get("image")
        mask = processed.get("mask")
        info = processed.get("info", {})
        if image is not None and mask is not None:
            return image, mask, info

    raise ValueError("Unsupported preprocess_case return format.")


# ---------------------------------------------------------------------
# OUTPUT ANALYSIS
# ---------------------------------------------------------------------

def analyze_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    thresholds: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
):
    """
    Analyze one case.

    logits:
        [1,6,D,H,W]

    target:
        [1,D,H,W]
    """
    if logits.ndim != 5:
        raise ValueError(f"Expected logits [B,C,D,H,W], got {tuple(logits.shape)}")

    if target.ndim != 4:
        raise ValueError(f"Expected target [B,D,H,W], got {tuple(target.shape)}")

    probs = torch.softmax(logits.float(), dim=1)

    pred = torch.argmax(logits, dim=1)

    target = target.long()

    foreground_pred = pred > 0
    foreground_target = target > 0

    pred_fg_voxels = int(foreground_pred.sum().item())
    target_fg_voxels = int(foreground_target.sum().item())

    # Conventional foreground Dice.
    intersection = int((foreground_pred & foreground_target).sum().item())
    denom = pred_fg_voxels + target_fg_voxels

    if denom == 0:
        foreground_dice = 1.0
    else:
        foreground_dice = 2.0 * intersection / denom

    precision = (
        intersection / pred_fg_voxels
        if pred_fg_voxels > 0
        else 0.0
    )

    recall = (
        intersection / target_fg_voxels
        if target_fg_voxels > 0
        else 0.0
    )

    background_prob = probs[:, 0]
    best_fg_prob, best_fg_class = torch.max(probs[:, 1:], dim=1)
    best_fg_class = best_fg_class + 1

    bg_mean = scalar(background_prob.mean())
    bg_min = scalar(background_prob.min())
    bg_max = scalar(background_prob.max())

    fg_mean = scalar(probs[:, 1:].mean())
    fg_max = scalar(probs[:, 1:].max())

    best_fg_mean = scalar(best_fg_prob.mean())
    best_fg_max = scalar(best_fg_prob.max())

    margin = best_fg_prob - background_prob

    margin_mean = scalar(margin.mean())
    margin_max = scalar(margin.max())
    positive_margin_voxels = int((margin > 0).sum().item())

    # Entropy across six classes.
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

    entropy_mean = scalar(entropy.mean())
    entropy_min = scalar(entropy.min())
    entropy_max = scalar(entropy.max())

    # Logit statistics.
    logits_cpu = logits.float().detach().cpu()

    class_logit_means = []
    class_logit_stds = []
    class_prob_means = []
    class_prob_maxs = []

    for c in range(NUM_CLASSES):
        class_logits = logits_cpu[:, c]
        class_probs = probs[:, c]

        class_logit_means.append(scalar(class_logits.mean()))
        class_logit_stds.append(scalar(class_logits.std()))
        class_prob_means.append(scalar(class_probs.mean()))
        class_prob_maxs.append(scalar(class_probs.max()))

    # Threshold sensitivity:
    # A threshold is applied to the best foreground probability.
    # Voxels below threshold become background; otherwise best foreground class.
    threshold_rows = []

    for threshold in thresholds:
        threshold_pred = torch.where(
            best_fg_prob >= threshold,
            best_fg_class,
            torch.zeros_like(best_fg_class),
        )

        t_fg = threshold_pred > 0
        p_voxels = int(t_fg.sum().item())
        inter = int((t_fg & foreground_target).sum().item())

        if p_voxels + target_fg_voxels == 0:
            d = 1.0
        else:
            d = 2.0 * inter / (p_voxels + target_fg_voxels)

        p = inter / p_voxels if p_voxels else 0.0
        r = inter / target_fg_voxels if target_fg_voxels else 0.0

        threshold_rows.append(
            {
                "threshold": float(threshold),
                "pred_foreground_voxels": p_voxels,
                "target_foreground_voxels": target_fg_voxels,
                "intersection": inter,
                "dice": float(d),
                "precision": float(p),
                "recall": float(r),
            }
        )

    return {
        "pred_foreground_voxels": pred_fg_voxels,
        "target_foreground_voxels": target_fg_voxels,
        "intersection": intersection,
        "foreground_dice": float(foreground_dice),
        "precision": float(precision),
        "recall": float(recall),
        "empty_argmax_prediction": int(pred_fg_voxels == 0),
        "background_probability_mean": bg_mean,
        "background_probability_min": bg_min,
        "background_probability_max": bg_max,
        "foreground_probability_mean": fg_mean,
        "foreground_probability_max": fg_max,
        "best_foreground_probability_mean": best_fg_mean,
        "best_foreground_probability_max": best_fg_max,
        "best_foreground_minus_background_mean": margin_mean,
        "best_foreground_minus_background_max": margin_max,
        "positive_foreground_margin_voxels": positive_margin_voxels,
        "positive_foreground_margin_fraction": (
            positive_margin_voxels / margin.numel()
        ),
        "entropy_mean": entropy_mean,
        "entropy_min": entropy_min,
        "entropy_max": entropy_max,
        "class_logit_means": class_logit_means,
        "class_logit_stds": class_logit_stds,
        "class_prob_means": class_prob_means,
        "class_prob_maxs": class_prob_maxs,
        "argmax_label_counts": {
            str(c): int((pred == c).sum().item())
            for c in range(NUM_CLASSES)
        },
        "threshold_rows": threshold_rows,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 78)
    print("PHASE 4 - PART 20")
    print("RSNA-ONLY MODEL OUTPUT / LOGIT BEHAVIOR AUDIT")
    print("=" * 78)

    print(
        """
Evaluation only.
No training is performed.
No model weights are modified.
SPIDER is not used.
RSNA test set is not used.

Purpose:
Investigate the Part 19 finding:
  foreground prediction voxels = 0 for all 100 cases.
"""
    )

    print("PROJECT ROOT")
    print(PROJECT_ROOT)

    print("\nRSNA DATASET")
    print(RSNA_ROOT)

    print("\nPART 15 CHECKPOINT")
    print(PART15_CHECKPOINT)

    print("\nOUTPUT DIRECTORY")
    print(OUT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

    header("PATH VALIDATION")
    require_paths()

    header("PYTORCH / GPU ENVIRONMENT")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"PyTorch version                       : {torch.__version__}")
    print(f"CUDA available                        : {torch.cuda.is_available()}")
    print(f"Device                                : {device}")

    if device.type == "cuda":
        print(f"GPU                                   : {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory                            : "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )

    print(f"Patch size                            : {PATCH_SIZE}")
    print(f"Feature size                          : {FEATURE_SIZE}")
    print(f"Classes                               : {NUM_CLASSES}")
    print(f"Validation cases                      : {VAL_CASES}")
    print(f"AMP                                   : {AMP_ENABLED}")

    set_seed()

    header("IMPORTING VALIDATED PART 11 IMPLEMENTATION")

    part11 = import_part11()

    print("✓ Corrected Part 11 imported.")
    print("✓ Exact cohort API available.")
    print("✓ Exact loading API available.")
    print("✓ Exact preprocessing API available.")
    print("✓ Exact model API available.")

    header("LOADING PART 9")

    part9 = part11.load_part9_module()
    print("✓ Part 9 loader imported through Part 11.")

    header("RECONSTRUCTING EXACT PART 15 VALIDATION COHORT")

    val_df = pd.read_csv(VAL_MANIFEST)

    val_cohort = part11.select_pilot_rows(
        VAL_MANIFEST,
        VAL_CASES,
        SEED,
    )

    if not isinstance(val_cohort, pd.DataFrame):
        val_cohort = pd.DataFrame(val_cohort)

    print(f"Validation manifest total : {len(val_df)}")
    print(f"Part 20 validation cohort : {len(val_cohort)}")

    if len(val_cohort) != VAL_CASES:
        raise RuntimeError(
            f"Expected {VAL_CASES} validation cases, got {len(val_cohort)}."
        )

    # Study/series identity.
    identity_cols = [
        c for c in ["study_id", "series_id"] if c in val_cohort.columns
    ]

    if identity_cols:
        duplicate_count = int(val_cohort.duplicated(identity_cols).sum())
        print(f"Duplicate study/series rows : {duplicate_count}")
    else:
        duplicate_count = 0

    header("LOADING PART 15 BEST CHECKPOINT")

    checkpoint = load_checkpoint(PART15_CHECKPOINT, device)
    state_dict = get_state_dict(checkpoint)

    model = create_model(part11, device)

    load_result = model.load_state_dict(state_dict, strict=False)

    print("✓ Swin-UNETR created.")
    print(f"Total parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Missing keys     : {len(load_result.missing_keys)}")
    print(f"Unexpected keys  : {len(load_result.unexpected_keys)}")

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError("Checkpoint/model state mismatch.")

    checkpoint_epoch = checkpoint.get("epoch", "unknown")
    checkpoint_best_dice = checkpoint.get("best_val_dice", "unknown")

    print(f"Checkpoint epoch : {checkpoint_epoch}")
    print(f"Checkpoint best Dice : {checkpoint_best_dice}")

    model.eval()

    header("STARTING RAW LOGIT / PROBABILITY AUDIT")

    print(
        """
For every case Part 20 records:
  - raw class logits
  - class probabilities
  - argmax prediction distribution
  - background probability dominance
  - best foreground probability
  - foreground/background probability margin
  - entropy
  - explicit foreground Dice
  - threshold sensitivity

No parameter or checkpoint modification occurs.
"""
    )

    case_rows: List[dict] = []
    class_accumulator = {
        c: {
            "logit_mean": [],
            "logit_std": [],
            "prob_mean": [],
            "prob_max": [],
        }
        for c in range(NUM_CLASSES)
    }

    threshold_accumulator = {
        float(t): {
            "dice": [],
            "precision": [],
            "recall": [],
            "pred_fg": [],
        }
        for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    }

    all_empty = 0
    all_argmax_counts = {c: 0 for c in range(NUM_CLASSES)}

    with torch.no_grad():
        for idx, (_, row) in enumerate(val_cohort.iterrows(), start=1):

            image, mask, info = part11.load_tensor_case(
                row,
                part9,
            )

            # Reproduce the exact Part 11 preprocessing contract.
            processed = part11.preprocess_case(
                image,
                mask,
                PATCH_SIZE,
            )

            image_p, mask_p, info_p = normalize_processed_case(processed)

            image_p = ensure_image_5d(image_p).to(device)
            mask_p = ensure_mask_4d(mask_p).to(device)

            if tuple(image_p.shape[1:]) != (1,) + PATCH_SIZE:
                raise RuntimeError(
                    f"Unexpected model input shape: {tuple(image_p.shape)}"
                )

            autocast_enabled = AMP_ENABLED and device.type == "cuda"

            if autocast_enabled:
                with torch.amp.autocast("cuda"):
                    logits = model(image_p)
            else:
                logits = model(image_p)

            result = analyze_logits(logits, mask_p)

            # Identify IDs.
            study_id = str(row.get("study_id", row.get("study", "")))
            series_id = str(row.get("series_id", row.get("series", "")))

            row_out = {
                "case_index": idx,
                "study_id": study_id,
                "series_id": series_id,
                "pred_foreground_voxels": result["pred_foreground_voxels"],
                "target_foreground_voxels": result["target_foreground_voxels"],
                "intersection": result["intersection"],
                "explicit_foreground_dice": result["foreground_dice"],
                "precision": result["precision"],
                "recall": result["recall"],
                "empty_argmax_prediction": result["empty_argmax_prediction"],
                "background_probability_mean": result["background_probability_mean"],
                "background_probability_min": result["background_probability_min"],
                "background_probability_max": result["background_probability_max"],
                "foreground_probability_mean": result["foreground_probability_mean"],
                "foreground_probability_max": result["foreground_probability_max"],
                "best_foreground_probability_mean": result["best_foreground_probability_mean"],
                "best_foreground_probability_max": result["best_foreground_probability_max"],
                "best_foreground_minus_background_mean": result[
                    "best_foreground_minus_background_mean"
                ],
                "best_foreground_minus_background_max": result[
                    "best_foreground_minus_background_max"
                ],
                "positive_foreground_margin_voxels": result[
                    "positive_foreground_margin_voxels"
                ],
                "positive_foreground_margin_fraction": result[
                    "positive_foreground_margin_fraction"
                ],
                "entropy_mean": result["entropy_mean"],
                "entropy_min": result["entropy_min"],
                "entropy_max": result["entropy_max"],
            }

            for c in range(NUM_CLASSES):
                row_out[f"logit_mean_class_{c}"] = result["class_logit_means"][c]
                row_out[f"logit_std_class_{c}"] = result["class_logit_stds"][c]
                row_out[f"prob_mean_class_{c}"] = result["class_prob_means"][c]
                row_out[f"prob_max_class_{c}"] = result["class_prob_maxs"][c]

                class_accumulator[c]["logit_mean"].append(
                    result["class_logit_means"][c]
                )
                class_accumulator[c]["logit_std"].append(
                    result["class_logit_stds"][c]
                )
                class_accumulator[c]["prob_mean"].append(
                    result["class_prob_means"][c]
                )
                class_accumulator[c]["prob_max"].append(
                    result["class_prob_maxs"][c]
                )

                all_argmax_counts[c] += result["argmax_label_counts"][str(c)]

            for threshold_row in result["threshold_rows"]:
                t = threshold_row["threshold"]
                threshold_accumulator[t]["dice"].append(threshold_row["dice"])
                threshold_accumulator[t]["precision"].append(
                    threshold_row["precision"]
                )
                threshold_accumulator[t]["recall"].append(
                    threshold_row["recall"]
                )
                threshold_accumulator[t]["pred_fg"].append(
                    threshold_row["pred_foreground_voxels"]
                )

            if result["empty_argmax_prediction"]:
                all_empty += 1

            case_rows.append(row_out)

            if idx <= 5 or idx in (25, 50, 75, 100):
                print(
                    f"  [{idx:03d}/{len(val_cohort)}] "
                    f"{study_id} | {series_id} | "
                    f"pred_fg={result['pred_foreground_voxels']} | "
                    f"target_fg={result['target_foreground_voxels']} | "
                    f"bg_mean={result['background_probability_mean']:.4f} | "
                    f"best_fg_max={result['best_foreground_probability_max']:.4f} | "
                    f"margin_max={result['best_foreground_minus_background_max']:.4f}"
                )

    case_df = pd.DataFrame(case_rows)

    header("PART 20 LOGIT / OUTPUT SUMMARY")

    mean_bg = float(case_df["background_probability_mean"].mean())
    mean_best_fg = float(
        case_df["best_foreground_probability_mean"].mean()
    )
    mean_best_fg_max = float(
        case_df["best_foreground_probability_max"].mean()
    )
    mean_margin = float(
        case_df["best_foreground_minus_background_mean"].mean()
    )
    max_margin = float(
        case_df["best_foreground_minus_background_max"].max()
    )
    mean_entropy = float(case_df["entropy_mean"].mean())

    print(f"Validation cases                    : {len(case_df)}")
    print(f"Empty argmax predictions             : {all_empty}")
    print(f"Non-empty argmax predictions         : {len(case_df) - all_empty}")
    print(f"Mean background probability          : {mean_bg:.6f}")
    print(f"Mean best foreground probability     : {mean_best_fg:.6f}")
    print(f"Mean best foreground maximum         : {mean_best_fg_max:.6f}")
    print(f"Mean best-FG minus background margin : {mean_margin:.6f}")
    print(f"Maximum observed margin              : {max_margin:.6f}")
    print(f"Mean prediction entropy              : {mean_entropy:.6f}")

    print("\nGlobal argmax voxel distribution:")
    total_voxels = sum(all_argmax_counts.values())

    for c in range(NUM_CLASSES):
        count = all_argmax_counts[c]
        fraction = count / total_voxels if total_voxels else 0.0
        print(
            f"  Label {c}: {count:,} voxels "
            f"({fraction * 100:.4f}%) - {CLASS_NAMES[c]}"
        )

    header("PER-CLASS LOGIT / PROBABILITY SUMMARY")

    class_rows = []

    for c in range(NUM_CLASSES):
        row = {
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "mean_logit": safe_mean(
                class_accumulator[c]["logit_mean"]
            ),
            "std_logit_across_cases": safe_std(
                class_accumulator[c]["logit_mean"]
            ),
            "mean_within_case_logit_std": safe_mean(
                class_accumulator[c]["logit_std"]
            ),
            "mean_probability": safe_mean(
                class_accumulator[c]["prob_mean"]
            ),
            "mean_case_max_probability": safe_mean(
                class_accumulator[c]["prob_max"]
            ),
            "max_probability_observed": max(
                class_accumulator[c]["prob_max"]
            ),
        }
        class_rows.append(row)

        print(
            f"{c}: {CLASS_NAMES[c]}\n"
            f"  Mean logit               : {row['mean_logit']:.6f}\n"
            f"  Mean probability        : {row['mean_probability']:.6f}\n"
            f"  Mean case max probability: "
            f"{row['mean_case_max_probability']:.6f}\n"
            f"  Max probability observed: "
            f"{row['max_probability_observed']:.6f}"
        )

    class_df = pd.DataFrame(class_rows)

    header("THRESHOLD SENSITIVITY")

    threshold_rows = []

    best_threshold_row = None

    for threshold, values in threshold_accumulator.items():
        row = {
            "threshold": threshold,
            "mean_dice": safe_mean(values["dice"]),
            "mean_precision": safe_mean(values["precision"]),
            "mean_recall": safe_mean(values["recall"]),
            "mean_pred_foreground_voxels": safe_mean(values["pred_fg"]),
            "cases_with_nonzero_foreground": int(
                sum(1 for x in values["pred_fg"] if x > 0)
            ),
        }
        threshold_rows.append(row)

        if (
            best_threshold_row is None
            or row["mean_dice"] > best_threshold_row["mean_dice"]
        ):
            best_threshold_row = row

        print(
            f"Threshold={threshold:.2f} | "
            f"Dice={row['mean_dice']:.6f} | "
            f"Precision={row['mean_precision']:.6f} | "
            f"Recall={row['mean_recall']:.6f} | "
            f"Non-empty={row['cases_with_nonzero_foreground']}/"
            f"{len(case_df)}"
        )

    threshold_df = pd.DataFrame(threshold_rows)

    header("DIAGNOSIS")

    # Evidence-based diagnosis.
    if all_empty == len(case_df):
        if mean_best_fg < 0.20 and max_margin <= 0:
            diagnosis = "STRONG_FOREGROUND_LOGIT_COLLAPSE"
            decision = (
                "CAUTION - foreground logits/probabilities are consistently "
                "dominated by background; investigate training/labels/loss."
            )
        elif max_margin > 0:
            diagnosis = "ARGMAX_BACKGROUND_DOMINANCE_OR_THRESHOLDING"
            decision = (
                "CAUTION - foreground probability exceeds background in "
                "some voxels, but argmax remains entirely background; inspect "
                "postprocessing and class calibration."
            )
        else:
            diagnosis = "BACKGROUND_DOMINANT_OUTPUT"
            decision = (
                "CAUTION - model output is background dominant; further "
                "logit/loss analysis is required."
            )
    else:
        diagnosis = "NONEMPTY_FOREGROUND_OUTPUT_CONFIRMED"
        decision = (
            "PASS - checkpoint produces non-empty foreground predictions; "
            "Part 19's foreground-collapse finding is not reproduced."
        )

    print(f"Diagnosis : {diagnosis}")
    print(f"Decision  : {decision}")

    header("SAVING PART 20 RESULTS")

    case_df.to_csv(CASE_CSV, index=False)
    class_df.to_csv(CLASS_CSV, index=False)
    threshold_df.to_csv(THRESHOLD_CSV, index=False)

    summary = {
        "phase": "Phase 4 - Part 20",
        "purpose": "Model output and logit behavior audit",
        "validation_cases": int(len(case_df)),
        "checkpoint": str(PART15_CHECKPOINT),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_best_dice": (
            float(checkpoint_best_dice)
            if isinstance(checkpoint_best_dice, (int, float))
            else str(checkpoint_best_dice)
        ),
        "empty_argmax_predictions": int(all_empty),
        "nonempty_argmax_predictions": int(len(case_df) - all_empty),
        "mean_background_probability": mean_bg,
        "mean_best_foreground_probability": mean_best_fg,
        "mean_best_foreground_max_probability": mean_best_fg_max,
        "mean_best_foreground_minus_background_margin": mean_margin,
        "maximum_observed_foreground_background_margin": max_margin,
        "mean_entropy": mean_entropy,
        "global_argmax_voxel_distribution": all_argmax_counts,
        "best_threshold_by_mean_dice": best_threshold_row,
        "diagnosis": diagnosis,
        "decision": decision,
        "spider_used": False,
        "test_set_used": False,
        "training_performed": False,
        "model_weights_changed": False,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "PHASE 4 - PART 20",
        "RSNA-ONLY MODEL OUTPUT / LOGIT BEHAVIOR AUDIT",
        "",
        "Evaluation only. No training or weight updates.",
        "",
        f"Validation cases: {len(case_df)}",
        f"Checkpoint epoch: {checkpoint_epoch}",
        f"Checkpoint best Dice: {checkpoint_best_dice}",
        f"Empty argmax predictions: {all_empty}",
        f"Non-empty argmax predictions: {len(case_df) - all_empty}",
        f"Mean background probability: {mean_bg:.8f}",
        f"Mean best foreground probability: {mean_best_fg:.8f}",
        f"Mean best foreground max probability: {mean_best_fg_max:.8f}",
        f"Mean FG-background margin: {mean_margin:.8f}",
        f"Maximum FG-background margin: {max_margin:.8f}",
        f"Mean entropy: {mean_entropy:.8f}",
        "",
        "Global argmax voxel distribution:",
    ]

    for c in range(NUM_CLASSES):
        report_lines.append(
            f"  Label {c} ({CLASS_NAMES[c]}): {all_argmax_counts[c]:,}"
        )

    report_lines.extend(
        [
            "",
            "Threshold sensitivity:",
        ]
    )

    for row in threshold_rows:
        report_lines.append(
            f"  {row['threshold']:.2f}: "
            f"Dice={row['mean_dice']:.8f}, "
            f"Precision={row['mean_precision']:.8f}, "
            f"Recall={row['mean_recall']:.8f}, "
            f"Non-empty={row['cases_with_nonzero_foreground']}/{len(case_df)}"
        )

    report_lines.extend(
        [
            "",
            f"Diagnosis: {diagnosis}",
            f"Decision: {decision}",
            "",
            "SPIDER used: NO",
            "Test set used: NO",
            "Training performed: NO",
            "Model weights changed: NO",
        ]
    )

    REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved: {CASE_CSV}")
    print(f"Saved: {CLASS_CSV}")
    print(f"Saved: {THRESHOLD_CSV}")
    print(f"Saved: {SUMMARY_JSON}")
    print(f"Saved: {REPORT_TXT}")

    header("PART 20 FINAL SUMMARY")

    print(f"Validation cases                 : {len(case_df)}")
    print(f"Empty argmax predictions         : {all_empty}")
    print(f"Non-empty argmax predictions     : {len(case_df) - all_empty}")
    print(f"Mean background probability      : {mean_bg:.6f}")
    print(f"Mean best foreground probability : {mean_best_fg:.6f}")
    print(f"Mean FG-background margin        : {mean_margin:.6f}")
    print(f"Diagnosis                         : {diagnosis}")
    print(f"Decision                          : {decision}")

    print(
        """
SPIDER used          : NO
Test set used        : NO
Training performed   : NO
Model weights changed: NO
"""
    )

    print("=" * 78)
    print("PHASE 4 - PART 20 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 78)
        print("PART 20 ERROR")
        print("=" * 78)
        print(f"{type(exc).__name__}: {exc}")
        raise
