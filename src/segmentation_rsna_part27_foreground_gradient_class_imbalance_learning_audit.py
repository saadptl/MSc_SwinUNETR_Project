"""
PHASE 4 - PART 27
RSNA-ONLY FOREGROUND GRADIENT / CLASS-IMBALANCE / LEARNING AUDIT

Evaluation / audit only.
No training is performed.
No model weights are modified.
No optimizer is created.
SPIDER is not used.
RSNA test set is not used.

Purpose
-------
Investigate why the Part 15 checkpoint produces foreground-empty predictions
for all audited validation cases despite the historical macro Dice of 0.668.

Part 27 audits:
  1. Part 15 training history and checkpoint metadata.
  2. Exact Part 15 train/validation cohort sizes.
  3. Training pseudo-mask foreground/background distribution.
  4. Per-class positive voxel counts and occupancy.
  5. Validation pseudo-mask distribution.
  6. DiceCELoss configuration as defined by Part 11.
  7. Checkpoint parameter statistics.
  8. Raw checkpoint logits on validation cases.
  9. Per-class logit/probability dominance.
 10. Gradient sensitivity of DiceCELoss with respect to logits
     on representative real validation targets.
 11. A synthetic foreground-gradient sanity test.
 12. A background-only collapse diagnosis.

IMPORTANT:
This script never calls optimizer.step(), backward() on model parameters,
or modifies/saves model weights. autograd is used only on detached clone
logits for loss-gradient diagnostics.
"""

from __future__ import annotations

import ast
import csv
import inspect
import json
import math
import os
import sys
import traceback
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# =============================================================================
# PATHS / CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RSNA_ROOT = PROJECT_ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification"

PART11_SOURCE = PROJECT_ROOT / "src" / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"

PART15_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
PART15_CHECKPOINT = PART15_DIR / "checkpoints" / "best_model.pth"
PART15_HISTORY = PART15_DIR / "part15_training_history.csv"
PART15_TRAIN_COHORT = PART15_DIR / "part15_train_cohort.csv"
PART15_VAL_COHORT = PART15_DIR / "part15_validation_cohort.csv"

PART8_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
    / "rsna_part8_validation_manifest.csv"
)

PART16_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part16_best_checkpoint_validation"
PART18_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part18_expanded_validation_stability_analysis"
PART19_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part19_prediction_integrity_metric_audit"
PART26_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part26_exact_part15_dice_reconstruction_audit"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part27_foreground_gradient_class_imbalance_learning_audit"
REPORT_DIR = OUTPUT_DIR / "reports"

PATCH_SIZE = (64, 96, 96)
NUM_CLASSES = 6
FEATURE_SIZE = 12
AUDIT_CASES = 10

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

EPS = 1e-8


# =============================================================================
# UTILITIES
# =============================================================================

def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )


def import_part11():
    spec = importlib.util.spec_from_file_location("part11_corrected", PART11_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {PART11_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["part11_corrected"] = module
    spec.loader.exec_module(module)
    return module


def find_part9_source() -> Path:
    candidates = [
        PROJECT_ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py",
        PROJECT_ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader_corrected.py",
    ]
    for p in candidates:
        if p.exists():
            return p

    matches = list((PROJECT_ROOT / "src").glob("*part9*3d*dataset*loader*.py"))
    if matches:
        return matches[0]

    raise FileNotFoundError("Could not locate Part 9 3D dataset loader source.")


def import_part9_through_part11(part11):
    if hasattr(part11, "load_part9"):
        return part11.load_part9()

    source = find_part9_source()
    spec = importlib.util.spec_from_file_location("part9_loader", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Part 9 from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["part9_loader"] = module
    spec.loader.exec_module(module)
    return module


def normalize_tensor_case(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize common Part 11 tensor shapes without changing values.

    Image returned as [B,C,D,H,W].
    Mask returned as [B,D,H,W] with integer class labels.
    """
    image = image.detach()
    mask = mask.detach()

    if image.ndim == 3:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 4:
        # Usually [C,D,H,W].
        image = image.unsqueeze(0)
    elif image.ndim != 5:
        raise ValueError(f"Unexpected image shape: {tuple(image.shape)}")

    if mask.ndim == 4:
        # Usually [B,D,H,W].
        pass
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 5 and mask.shape[1] == 1:
        mask = mask[:, 0]
    else:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")

    return image.float(), mask.long()


def load_case(part11, part9, row: pd.Series):
    """
    Use the validated Part 11 API. The known validated signature is:
      load_tensor_case(row, part9)
    """
    value = part11.load_tensor_case(row, part9)
    if not isinstance(value, tuple) or len(value) < 2:
        raise RuntimeError(f"Unexpected load_tensor_case return: {type(value)}")
    image, mask = value[0], value[1]
    info = value[2] if len(value) >= 3 else {}
    return normalize_tensor_case(image, mask), info


def create_model(part11, device):
    sig = inspect.signature(part11.create_model)
    print(f"Part 11 create_model signature : {sig}")
    return part11.create_model(device)


def load_checkpoint(model, path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break

    missing, unexpected = model.load_state_dict(state, strict=False)
    return checkpoint, list(missing), list(unexpected)


def resolve_history():
    if not PART15_HISTORY.exists():
        return None
    return pd.read_csv(PART15_HISTORY)


def resolve_cohort(path: Path, fallback: Path | None = None):
    if path.exists():
        return pd.read_csv(path)
    if fallback is not None and fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(path)


# =============================================================================
# SOURCE / LOSS INSPECTION
# =============================================================================

def inspect_part11_loss_source() -> Dict[str, Any]:
    text = PART11_SOURCE.read_text(encoding="utf-8", errors="replace")

    result: Dict[str, Any] = {
        "diceceloss_occurrences": [],
        "loss_function_occurrences": [],
        "source_sha256": None,
    }

    import hashlib
    result["source_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    for line_no, line in enumerate(text.splitlines(), start=1):
        if "DiceCELoss" in line:
            result["diceceloss_occurrences"].append(
                {"line": line_no, "text": line.strip()}
            )
        if "loss_function" in line.lower() or "criterion" in line.lower():
            result["loss_function_occurrences"].append(
                {"line": line_no, "text": line.strip()}
            )

    return result


def construct_dice_ce_loss():
    try:
        from monai.losses import DiceCELoss

        candidates = [
            {},
            {"to_onehot_y": True, "softmax": True},
            {"to_onehot_y": True, "softmax": True, "include_background": True},
        ]

        for kwargs in candidates:
            try:
                loss = DiceCELoss(**kwargs)
                return loss, kwargs, None
            except Exception:
                continue

        return None, None, "Could not instantiate DiceCELoss with tested configurations."
    except Exception as e:
        return None, None, repr(e)


# =============================================================================
# MASK DISTRIBUTION
# =============================================================================

def mask_distribution(mask: torch.Tensor) -> Dict[str, Any]:
    arr = mask.detach().cpu().numpy().astype(np.int64)
    flat = arr.reshape(-1)

    counts = np.bincount(flat, minlength=NUM_CLASSES)
    total = int(flat.size)

    classes = {}
    for cid in range(NUM_CLASSES):
        count = int(counts[cid])
        classes[str(cid)] = {
            "name": CLASS_NAMES[cid],
            "voxels": count,
            "fraction": float(count / total) if total else 0.0,
            "present": bool(count > 0),
        }

    fg = int(counts[1:].sum())

    return {
        "total_voxels": total,
        "background_voxels": int(counts[0]),
        "foreground_voxels": fg,
        "foreground_fraction": float(fg / total) if total else 0.0,
        "classes": classes,
    }


def audit_cohort_masks(part11, part9, cohort: pd.DataFrame, label: str, max_cases: int | None):
    rows = []
    limit = len(cohort) if max_cases is None else min(max_cases, len(cohort))

    for i in range(limit):
        row = cohort.iloc[i]
        (image, mask), info = load_case(part11, part9, row)
        d = mask_distribution(mask)

        record = {
            "index": i + 1,
            "study_id": str(row.get("study_id", "")),
            "series_id": str(row.get("series_id", "")),
            "image_shape": str(tuple(image.shape)),
            "mask_shape": str(tuple(mask.shape)),
            "total_voxels": d["total_voxels"],
            "background_voxels": d["background_voxels"],
            "foreground_voxels": d["foreground_voxels"],
            "foreground_fraction": d["foreground_fraction"],
        }

        for cid in range(1, NUM_CLASSES):
            record[f"class_{cid}_voxels"] = d["classes"][str(cid)]["voxels"]
            record[f"class_{cid}_fraction"] = d["classes"][str(cid)]["fraction"]

        rows.append(record)

        if i < 5 or (i + 1) == limit:
            print(
                f"  [{i+1:03d}/{limit}] "
                f"FG={record['foreground_voxels']} "
                f"BG={record['background_voxels']} "
                f"classes="
                + ",".join(
                    str(record[f"class_{cid}_voxels"]) for cid in range(1, NUM_CLASSES)
                )
            )

    return pd.DataFrame(rows)


# =============================================================================
# LOGIT AUDIT
# =============================================================================

def softmax_stats(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
    """
    logits: [B,C,D,H,W]
    target: [B,D,H,W]
    """
    probs = torch.softmax(logits.float(), dim=1)
    pred = torch.argmax(logits, dim=1)

    class_prob_mean = probs.mean(dim=(0, 2, 3, 4))
    class_logit_mean = logits.float().mean(dim=(0, 2, 3, 4))

    counts = torch.bincount(
        pred.reshape(-1),
        minlength=NUM_CLASSES,
    ).cpu().numpy()

    total = int(pred.numel())
    fg_count = int(counts[1:].sum())

    entropy = -(probs.clamp_min(EPS) * probs.clamp_min(EPS).log()).sum(dim=1).mean()

    best_fg_prob = probs[:, 1:].max(dim=1).values
    bg_prob = probs[:, 0]

    return {
        "logit_mean": class_logit_mean.cpu().tolist(),
        "probability_mean": class_prob_mean.cpu().tolist(),
        "prediction_voxels": counts.tolist(),
        "prediction_fractions": (counts / total).tolist(),
        "foreground_prediction_voxels": fg_count,
        "foreground_prediction_fraction": float(fg_count / total),
        "background_prediction_fraction": float(counts[0] / total),
        "mean_background_probability": float(bg_prob.mean()),
        "mean_best_foreground_probability": float(best_fg_prob.mean()),
        "foreground_background_probability_margin": float(
            (best_fg_prob - bg_prob).mean()
        ),
        "mean_entropy": float(entropy.item()),
        "logit_background_minus_best_foreground": float(
            (logits[:, 0] - logits[:, 1:].max(dim=1).values).mean().item()
        ),
    }


def explicit_class_dice(logits: torch.Tensor, target: torch.Tensor):
    pred = torch.argmax(logits, dim=1)
    out = {}

    for cid in range(1, NUM_CLASSES):
        p = pred == cid
        t = target == cid
        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        dice = 1.0 if denom == 0 else (2.0 * inter / denom)

        out[str(cid)] = {
            "dice": float(dice),
            "prediction_voxels": int(p.sum().item()),
            "target_voxels": int(t.sum().item()),
        }

    return out


# =============================================================================
# GRADIENT SENSITIVITY AUDIT
# =============================================================================

def dice_ce_gradient_audit(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_fn,
) -> Dict[str, Any]:
    """
    Compute loss gradients only on a detached clone of logits.
    The model graph and model parameters are never involved.
    """
    if loss_fn is None:
        return {"available": False, "error": "DiceCELoss unavailable"}

    x = logits.detach().float().clone().requires_grad_(True)
    y = target.detach().clone()

    try:
        loss = loss_fn(x, y)
        loss.backward()

        grad = x.grad.detach()
        per_class_abs = grad.abs().mean(dim=(0, 2, 3, 4))
        per_class_max = grad.abs().amax(dim=(0, 2, 3, 4))

        return {
            "available": True,
            "loss": float(loss.detach().item()),
            "mean_abs_gradient_per_class": per_class_abs.cpu().tolist(),
            "max_abs_gradient_per_class": per_class_max.cpu().tolist(),
            "total_mean_abs_gradient": float(grad.abs().mean().item()),
            "total_max_abs_gradient": float(grad.abs().max().item()),
            "foreground_mean_abs_gradient": float(
                per_class_abs[1:].mean().item()
            ),
            "background_mean_abs_gradient": float(
                per_class_abs[0].item()
            ),
        }
    except Exception as e:
        return {
            "available": False,
            "error": repr(e),
        }


def synthetic_gradient_sanity(loss_fn, device):
    if loss_fn is None:
        return {"available": False, "error": "DiceCELoss unavailable"}

    # Small synthetic volume to verify that foreground targets generate
    # non-zero gradients in the loss implementation itself.
    shape = (1, NUM_CLASSES, 8, 8, 8)
    target = torch.zeros((1, 8, 8, 8), dtype=torch.long, device=device)

    target[:, 2:6, 2:6, 2:6] = 1
    target[:, 4:7, 4:7, 4:7] = 2

    logits = torch.zeros(shape, dtype=torch.float32, device=device, requires_grad=True)
    logits.data[:, 0] = 3.0

    try:
        loss = loss_fn(logits, target)
        loss.backward()

        g = logits.grad.detach().abs().mean(dim=(0, 2, 3, 4))

        return {
            "available": True,
            "loss": float(loss.detach().item()),
            "mean_abs_gradient_per_class": g.cpu().tolist(),
            "foreground_gradient_nonzero": bool((g[1:] > 0).any().item()),
            "background_gradient_nonzero": bool(g[0] > 0),
        }
    except Exception as e:
        return {"available": False, "error": repr(e)}


# =============================================================================
# PARAMETER AUDIT
# =============================================================================

def parameter_audit(model: torch.nn.Module) -> Dict[str, Any]:
    total = 0
    nonzero = 0
    abs_sum = 0.0
    abs_max = 0.0
    tensors = 0

    for p in model.parameters():
        a = p.detach().float().abs()
        total += p.numel()
        nonzero += int((a > 0).sum().item())
        abs_sum += float(a.sum().item())
        abs_max = max(abs_max, float(a.max().item()))
        tensors += 1

    return {
        "parameter_tensors": tensors,
        "parameters": total,
        "nonzero_parameters": nonzero,
        "nonzero_fraction": float(nonzero / total) if total else 0.0,
        "mean_abs_parameter": float(abs_sum / total) if total else 0.0,
        "max_abs_parameter": abs_max,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    banner("PHASE 4 - PART 27")
    print("RSNA-ONLY FOREGROUND GRADIENT / CLASS-IMBALANCE / LEARNING AUDIT")
    print()
    print("Evaluation / audit only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")
    print()
    print("Purpose:")
    print("Investigate the Part 20 foreground-logit collapse and")
    print("determine whether labels/loss/gradients can explain it.")

    banner("PATH VALIDATION")
    paths = {
        "RSNA root": RSNA_ROOT,
        "Part 11 source": PART11_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 history": PART15_HISTORY,
        "Part 15 train cohort": PART15_TRAIN_COHORT,
        "Part 15 validation cohort": PART15_VAL_COHORT,
        "Part 8 validation manifest": PART8_MANIFEST,
        "Part 16 directory": PART16_DIR,
        "Part 18 directory": PART18_DIR,
        "Part 19 directory": PART19_DIR,
        "Part 26 directory": PART26_DIR,
    }

    for name, p in paths.items():
        print(f"{name:<38}: {'FOUND' if p.exists() else 'MISSING'}")

    required = [
        RSNA_ROOT,
        PART11_SOURCE,
        PART15_CHECKPOINT,
        PART15_HISTORY,
        PART15_TRAIN_COHORT,
        PART15_VAL_COHORT,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Part 27 input(s):\n" + "\n".join(missing))

    banner("PYTORCH / GPU ENVIRONMENT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version                       : {torch.__version__}")
    print(f"CUDA available                        : {torch.cuda.is_available()}")
    print(f"Device                                : {device}")
    if torch.cuda.is_available():
        print(f"GPU                                   : {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory                            : "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )
    print(f"Patch size                            : {PATCH_SIZE}")
    print(f"Feature size                          : {FEATURE_SIZE}")
    print(f"Classes                               : {NUM_CLASSES}")
    print(f"Audit validation cases                : {AUDIT_CASES}")

    banner("IMPORTING VALIDATED PART 11")
    part11 = import_part11()
    print("✓ Corrected Part 11 imported.")
    print(f"create_model        : {inspect.signature(part11.create_model)}")
    print(f"preprocess_case     : {inspect.signature(part11.preprocess_case)}")
    print(f"load_tensor_case    : {inspect.signature(part11.load_tensor_case)}")
    print(f"dice_from_prediction: {inspect.signature(part11.dice_from_prediction)}")

    banner("LOADING PART 15 HISTORY / COHORTS")
    history = pd.read_csv(PART15_HISTORY)
    train_cohort = pd.read_csv(PART15_TRAIN_COHORT)
    val_cohort = pd.read_csv(PART15_VAL_COHORT)

    print(f"History rows                 : {len(history)}")
    print(f"Part 15 train cohort rows    : {len(train_cohort)}")
    print(f"Part 15 validation rows      : {len(val_cohort)}")

    if "val_dice" in history.columns:
        print(f"Maximum recorded val Dice    : {history['val_dice'].max():.6f}")
    if "val_loss" in history.columns:
        print(f"Minimum recorded val loss    : {history['val_loss'].min():.6f}")

    print()
    print(history.to_string(index=False))

    banner("INSPECTING PART 11 LOSS SOURCE")
    source_info = inspect_part11_loss_source()
    print(f"Part 11 SHA256 : {source_info['source_sha256']}")
    print("DiceCELoss references:")
    for item in source_info["diceceloss_occurrences"][:20]:
        print(f"  {item['line']}: {item['text']}")

    loss_fn, loss_kwargs, loss_error = construct_dice_ce_loss()
    print(f"Test DiceCELoss configuration : {loss_kwargs}")
    if loss_error:
        print(f"DiceCELoss construction note  : {loss_error}")
    else:
        print(f"DiceCELoss type               : {type(loss_fn)}")

    banner("RESOLVING PART 9")
    part9 = import_part9_through_part11(part11)
    print("✓ Part 9 resolved through the validated Part 11 environment.")
    print(f"Part 9 loader type             : {type(part9)}")

    banner("AUDITING PART 15 TRAINING PSEUDO-MASKS")
    train_mask_df = audit_cohort_masks(
        part11,
        part9,
        train_cohort,
        "train",
        max_cases=None,
    )

    banner("TRAINING PSEUDO-MASK AGGREGATE")
    train_summary = {}
    for cid in range(1, NUM_CLASSES):
        col = f"class_{cid}_voxels"
        values = train_mask_df[col].to_numpy(dtype=np.float64)
        train_summary[str(cid)] = {
            "name": CLASS_NAMES[cid],
            "cases_with_positive_voxels": int((values > 0).sum()),
            "cases_total": int(len(values)),
            "mean_positive_voxels": float(values.mean()),
            "median_positive_voxels": float(np.median(values)),
            "min_positive_voxels": float(values.min()),
            "max_positive_voxels": float(values.max()),
        }

    train_fg = train_mask_df["foreground_voxels"].to_numpy(dtype=np.float64)
    train_bg = train_mask_df["background_voxels"].to_numpy(dtype=np.float64)

    train_aggregate = {
        "cases": int(len(train_mask_df)),
        "mean_foreground_voxels": float(train_fg.mean()),
        "mean_background_voxels": float(train_bg.mean()),
        "mean_foreground_fraction": float(train_mask_df["foreground_fraction"].mean()),
        "median_foreground_fraction": float(train_mask_df["foreground_fraction"].median()),
        "class_summary": train_summary,
    }

    print(f"Training cases                         : {len(train_mask_df)}")
    print(f"Mean foreground voxels                 : {train_fg.mean():.3f}")
    print(f"Mean background voxels                 : {train_bg.mean():.3f}")
    print(f"Mean foreground fraction               : {train_mask_df['foreground_fraction'].mean():.8f}")
    for cid in range(1, NUM_CLASSES):
        x = train_summary[str(cid)]
        print(
            f"{cid}: {x['name']:<35} "
            f"positive cases={x['cases_with_positive_voxels']}/{x['cases_total']} "
            f"mean voxels={x['mean_positive_voxels']:.2f}"
        )

    banner("AUDITING PART 15 VALIDATION MASKS")
    val_mask_df = audit_cohort_masks(
        part11,
        part9,
        val_cohort,
        "validation",
        max_cases=None,
    )

    val_summary = {}
    for cid in range(1, NUM_CLASSES):
        col = f"class_{cid}_voxels"
        values = val_mask_df[col].to_numpy(dtype=np.float64)
        val_summary[str(cid)] = {
            "name": CLASS_NAMES[cid],
            "cases_with_positive_voxels": int((values > 0).sum()),
            "cases_total": int(len(values)),
            "mean_positive_voxels": float(values.mean()),
            "median_positive_voxels": float(np.median(values)),
            "min_positive_voxels": float(values.min()),
            "max_positive_voxels": float(values.max()),
        }

    banner("CREATING PART 15 MODEL / LOADING CHECKPOINT")
    model = create_model(part11, device)
    model = model.to(device)
    model.eval()

    checkpoint, missing_keys, unexpected_keys = load_checkpoint(
        model,
        PART15_CHECKPOINT,
        device,
    )

    print(f"Total parameters       : {sum(p.numel() for p in model.parameters())}")
    print(f"Missing keys           : {len(missing_keys)}")
    print(f"Unexpected keys        : {len(unexpected_keys)}")

    if isinstance(checkpoint, dict):
        print(f"Checkpoint epoch       : {checkpoint.get('epoch', None)}")
        print(f"Checkpoint best Dice   : {checkpoint.get('best_dice', None)}")

    param_stats = parameter_audit(model)
    print(f"Nonzero parameter fraction: {param_stats['nonzero_fraction']:.8f}")
    print(f"Mean absolute parameter   : {param_stats['mean_abs_parameter']:.8e}")

    banner("SYNTHETIC DICECELoss GRADIENT SANITY")
    synthetic_gradient = synthetic_gradient_sanity(loss_fn, device)
    print(json.dumps(synthetic_gradient, indent=2, default=json_default))

    banner(f"RAW CHECKPOINT LOGIT + GRADIENT AUDIT ({AUDIT_CASES} CASES)")
    audit_rows = []
    audit_gradient_rows = []

    n = min(AUDIT_CASES, len(val_cohort))

    with torch.no_grad():
        for i in range(n):
            row = val_cohort.iloc[i]
            (image, mask), info = load_case(part11, part9, row)

            image = image.to(device)
            mask = mask.to(device)

            logits = model(image)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            if logits.ndim != 5:
                raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")

            if tuple(logits.shape[-3:]) != tuple(mask.shape[-3:]):
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=mask.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )

            stats = softmax_stats(logits, mask)
            dices = explicit_class_dice(logits, mask)

            row_out = {
                "index": i + 1,
                "study_id": str(row.get("study_id", "")),
                "series_id": str(row.get("series_id", "")),
                "foreground_target_voxels": int((mask > 0).sum().item()),
                **stats,
            }

            for cid in range(1, NUM_CLASSES):
                row_out[f"class_{cid}_dice"] = dices[str(cid)]["dice"]
                row_out[f"class_{cid}_pred_voxels"] = dices[str(cid)]["prediction_voxels"]
                row_out[f"class_{cid}_target_voxels"] = dices[str(cid)]["target_voxels"]

            audit_rows.append(row_out)

            # Gradient audit is deliberately isolated from the model graph.
            gradient = dice_ce_gradient_audit(logits, mask, loss_fn)
            gradient["index"] = i + 1
            gradient["study_id"] = str(row.get("study_id", ""))
            gradient["series_id"] = str(row.get("series_id", ""))
            audit_gradient_rows.append(gradient)

            print(
                f"[{i+1:03d}/{n}] "
                f"PredFG={stats['foreground_prediction_voxels']} "
                f"TargetFG={row_out['foreground_target_voxels']} "
                f"BGprob={stats['mean_background_probability']:.6f} "
                f"BestFGprob={stats['mean_best_foreground_probability']:.6f} "
                f"Margin={stats['foreground_background_probability_margin']:.6f} "
                f"Entropy={stats['mean_entropy']:.6f}"
            )

    banner("PART 27 AGGREGATED LOGIT RESULTS")
    audit_df = pd.DataFrame(audit_rows)

    mean_bg_prob = float(audit_df["mean_background_probability"].mean())
    mean_best_fg = float(audit_df["mean_best_foreground_probability"].mean())
    mean_margin = float(audit_df["foreground_background_probability_margin"].mean())
    mean_entropy = float(audit_df["mean_entropy"].mean())
    empty_predictions = int((audit_df["foreground_prediction_voxels"] == 0).sum())

    print(f"Audited validation cases              : {len(audit_df)}")
    print(f"Empty foreground predictions          : {empty_predictions}/{len(audit_df)}")
    print(f"Mean background probability           : {mean_bg_prob:.8f}")
    print(f"Mean best foreground probability      : {mean_best_fg:.8f}")
    print(f"Mean foreground/background margin     : {mean_margin:.8f}")
    print(f"Mean voxel entropy                    : {mean_entropy:.8f}")

    banner("PART 27 GRADIENT RESULTS")
    gradient_df = pd.DataFrame(audit_gradient_rows)

    if len(gradient_df) and gradient_df["available"].any():
        available = gradient_df[gradient_df["available"] == True].copy()

        print(f"Gradient-audited cases                : {len(available)}")
        print(
            f"Mean DiceCELoss on checkpoint logits  : "
            f"{available['loss'].mean():.8f}"
        )
        print(
            f"Mean total abs logit gradient         : "
            f"{available['total_mean_abs_gradient'].mean():.8e}"
        )
        print(
            f"Mean foreground abs gradient          : "
            f"{available['foreground_mean_abs_gradient'].mean():.8e}"
        )
        print(
            f"Mean background abs gradient          : "
            f"{available['background_mean_abs_gradient'].mean():.8e}"
        )

        per_class_grad = np.array(
            available["mean_abs_gradient_per_class"].tolist(),
            dtype=np.float64,
        ).mean(axis=0)

        print("Mean absolute gradient per class:")
        for cid in range(NUM_CLASSES):
            print(
                f"  {cid}: {CLASS_NAMES[cid]:<35} "
                f"{per_class_grad[cid]:.8e}"
            )
    else:
        print("No usable DiceCELoss gradient results were produced.")

    banner("PART 27 DIAGNOSIS")

    train_fg_fraction = train_aggregate["mean_foreground_fraction"]

    diagnosis = []

    if empty_predictions == len(audit_df):
        diagnosis.append("CHECKPOINT_FOREGROUND_COLLAPSE_CONFIRMED")

    if train_fg_fraction > 0:
        diagnosis.append("TRAINING_LABELS_CONTAIN_FOREGROUND")
    else:
        diagnosis.append("TRAINING_LABELS_HAVE_NO_FOREGROUND")

    if synthetic_gradient.get("available") and synthetic_gradient.get("foreground_gradient_nonzero"):
        diagnosis.append("DICECELOSS_FOREGROUND_GRADIENT_SANITY_PASS")
    elif synthetic_gradient.get("available"):
        diagnosis.append("DICECELOSS_FOREGROUND_GRADIENT_SANITY_FAIL")

    if mean_bg_prob > mean_best_fg:
        diagnosis.append("BACKGROUND_PROBABILITY_DOMINATES_FOREGROUND")

    print("Diagnosis flags:")
    for item in diagnosis:
        print(f"  - {item}")

    if train_fg_fraction > 0 and empty_predictions == len(audit_df):
        final_diagnosis = (
            "LABELS_ARE_NOT_EMPTY_BUT_CHECKPOINT_LEARNED_BACKGROUND_ONLY; "
            "INVESTIGATE_TRAINING_OBJECTIVE_WEIGHTING_AND_DATA_TO_LABEL_ALIGNMENT"
        )
    elif train_fg_fraction == 0:
        final_diagnosis = "FOREGROUND_COLLAPSE_EXPLAINED_BY_EMPTY_TRAINING_LABELS"
    else:
        final_diagnosis = "FURTHER_TRAINING_PIPELINE_AUDIT_REQUIRED"

    print()
    print(f"FINAL DIAGNOSIS: {final_diagnosis}")

    banner("SAVING PART 27 RESULTS")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_mask_path = OUTPUT_DIR / "part27_training_pseudomask_distribution.csv"
    val_mask_path = OUTPUT_DIR / "part27_validation_pseudomask_distribution.csv"
    logits_path = OUTPUT_DIR / "part27_checkpoint_logit_audit.csv"
    gradients_path = OUTPUT_DIR / "part27_dicece_gradient_audit.csv"
    summary_path = OUTPUT_DIR / "phase4_part27_foreground_gradient_class_imbalance_learning_summary.json"
    report_path = REPORT_DIR / "phase4_part27_foreground_gradient_class_imbalance_learning_report.txt"

    train_mask_df.to_csv(train_mask_path, index=False)
    val_mask_df.to_csv(val_mask_path, index=False)
    audit_df.to_csv(logits_path, index=False)
    gradient_df.to_csv(gradients_path, index=False)

    summary = {
        "phase": 4,
        "part": 27,
        "project_root": str(PROJECT_ROOT),
        "rsna_root": str(RSNA_ROOT),
        "checkpoint": str(PART15_CHECKPOINT),
        "evaluation_only": True,
        "training_performed": False,
        "weights_modified": False,
        "optimizer_created": False,
        "spider_used": False,
        "test_set_used": False,
        "device": str(device),
        "torch_version": torch.__version__,
        "train_cohort_cases": int(len(train_cohort)),
        "validation_cohort_cases": int(len(val_cohort)),
        "audit_cases": int(n),
        "history_max_val_dice": float(history["val_dice"].max()) if "val_dice" in history.columns else None,
        "train_mask_summary": train_aggregate,
        "validation_mask_summary": {
            "cases": int(len(val_mask_df)),
            "mean_foreground_fraction": float(val_mask_df["foreground_fraction"].mean()),
            "class_summary": val_summary,
        },
        "parameter_audit": param_stats,
        "synthetic_gradient_sanity": synthetic_gradient,
        "checkpoint_logit_summary": {
            "empty_foreground_predictions": empty_predictions,
            "mean_background_probability": mean_bg_prob,
            "mean_best_foreground_probability": mean_best_fg,
            "mean_foreground_background_margin": mean_margin,
            "mean_entropy": mean_entropy,
        },
        "gradient_audit_summary": (
            {
                "available_cases": int(
                    gradient_df["available"].sum()
                    if "available" in gradient_df.columns
                    else 0
                ),
                "mean_loss": float(
                    gradient_df.loc[
                        gradient_df["available"] == True, "loss"
                    ].mean()
                )
                if "available" in gradient_df.columns
                and gradient_df["available"].any()
                else None,
            }
        ),
        "diagnosis_flags": diagnosis,
        "final_diagnosis": final_diagnosis,
        "part11_source_sha256": source_info["source_sha256"],
        "loss_source_references": source_info["diceceloss_occurrences"],
    }

    write_json(summary_path, summary)

    report_lines = [
        "PHASE 4 - PART 27",
        "RSNA-ONLY FOREGROUND GRADIENT / CLASS-IMBALANCE / LEARNING AUDIT",
        "",
        "Evaluation only. No training. No weight modification. No optimizer.",
        "",
        f"Train cohort cases: {len(train_cohort)}",
        f"Validation cohort cases: {len(val_cohort)}",
        f"Audit cases: {n}",
        f"Historical maximum val Dice: {summary['history_max_val_dice']}",
        "",
        "TRAINING PSEUDO-MASK SUMMARY",
        f"Mean foreground fraction: {train_aggregate['mean_foreground_fraction']:.10f}",
        f"Mean foreground voxels: {train_aggregate['mean_foreground_voxels']:.3f}",
        f"Mean background voxels: {train_aggregate['mean_background_voxels']:.3f}",
        "",
        "CHECKPOINT LOGIT SUMMARY",
        f"Empty foreground predictions: {empty_predictions}/{len(audit_df)}",
        f"Mean background probability: {mean_bg_prob:.10f}",
        f"Mean best foreground probability: {mean_best_fg:.10f}",
        f"Mean foreground/background margin: {mean_margin:.10f}",
        f"Mean entropy: {mean_entropy:.10f}",
        "",
        "SYNTHETIC GRADIENT SANITY",
        json.dumps(synthetic_gradient, indent=2, default=json_default),
        "",
        "DIAGNOSIS",
        *[f"- {x}" for x in diagnosis],
        f"FINAL DIAGNOSIS: {final_diagnosis}",
        "",
        "No model parameters were changed by this audit.",
    ]

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved: {train_mask_path}")
    print(f"Saved: {val_mask_path}")
    print(f"Saved: {logits_path}")
    print(f"Saved: {gradients_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")

    banner("PHASE 4 - PART 27 COMPLETE")
    print("No training performed.")
    print("No model weights modified.")
    print("No checkpoint modified.")
    print(f"Diagnosis: {final_diagnosis}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPART 27 INTERRUPTED BY USER.")
        raise
    except Exception as exc:
        banner("PART 27 ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
