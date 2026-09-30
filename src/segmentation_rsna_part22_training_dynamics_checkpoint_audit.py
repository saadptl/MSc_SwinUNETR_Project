
"""
PHASE 4 - PART 22
RSNA-ONLY TRAINING DYNAMICS / CHECKPOINT COLLAPSE AUDIT

Evaluation / audit only.
No training is performed.
No model weights are modified.
No optimizer is created.
SPIDER is not used.
RSNA test set is not used.

Purpose:
Investigate WHEN and WHERE the Part 15 foreground collapse occurred.

This audit uses:
  - Part 15 training history
  - Part 15 train/validation cohorts
  - Part 15 best checkpoint
  - validated Part 11 preprocessing/model API
  - explicit foreground metrics

Important:
The legacy Part 15 Dice value is NOT treated as trustworthy foreground
segmentation performance. Part 21 established that the checkpoint produces
zero foreground predictions despite non-empty training masks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

SRC_DIR = PROJECT_ROOT / "src"

PART11_SOURCE = (
    SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART15_CHECKPOINT = PART15_DIR / "checkpoints" / "best_model.pth"
PART15_HISTORY = PART15_DIR / "part15_training_history.csv"
PART15_TRAIN_COHORT = PART15_DIR / "part15_train_cohort.csv"
PART15_VAL_COHORT = PART15_DIR / "part15_validation_cohort.csv"

PART16_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part16_best_checkpoint_validation"
)

PART19_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part19_prediction_integrity_metric_audit"
)

PART21_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part21_training_label_loss_integrity_audit"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part22_training_dynamics_checkpoint_audit"
)

REPORT_DIR = OUTPUT_DIR / "reports"

PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12
NUM_CLASSES = 6
BATCH_SIZE = 1
AMP_ENABLED = True

CLASS_NAMES = [
    "Background",
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]


# ============================================================================
# UTILITIES
# ============================================================================

def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def print_kv(label: str, value: Any, width: int = 38) -> None:
    print(f"{label:<{width}} : {value}")


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PATH VALIDATION
# ============================================================================

def validate_paths() -> None:
    required = {
        "RSNA root": RSNA_ROOT,
        "Part 11 source": PART11_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 history": PART15_HISTORY,
        "Part 15 train cohort": PART15_TRAIN_COHORT,
        "Part 15 validation cohort": PART15_VAL_COHORT,
    }

    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required Part 22 input(s):\n" + "\n".join(missing)
        )


# ============================================================================
# PART 11 IMPORT
# ============================================================================

def import_part11():
    spec = importlib.util.spec_from_file_location("part11_corrected", PART11_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Part 11 source: {PART11_SOURCE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["part11_corrected"] = module
    spec.loader.exec_module(module)

    required = [
        "select_pilot_rows",
        "load_part9_module",
        "load_tensor_case",
        "preprocess_case",
        "create_model",
        "DiceCELoss",
    ]

    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Part 11 missing required API: {missing}")

    return module


# ============================================================================
# DATA / API ADAPTERS
# ============================================================================

def normalize_3d_numpy(x: Any, name: str) -> np.ndarray:
    """
    Convert common loader representations to [D,H,W].
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    while x.ndim > 3:
        if x.shape[0] == 1:
            x = x[0]
        elif x.shape[-1] == 1:
            x = x[..., 0]
        else:
            raise ValueError(f"{name} cannot safely reduce shape {x.shape} to [D,H,W]")

    if x.ndim != 3:
        raise ValueError(f"{name} must be 3-D after normalization; got {x.shape}")

    return x


def normalize_processed_case(processed: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Part 11 returns image, mask tensors. Normalize them to:
      image [1,1,D,H,W]
      mask  [1,D,H,W]
    """
    if not isinstance(processed, (tuple, list)) or len(processed) < 2:
        raise ValueError(f"Unexpected preprocess_case return: {type(processed)}")

    image, mask = processed[0], processed[1]

    image = torch.as_tensor(image)
    mask = torch.as_tensor(mask)

    # Image -> [1,1,D,H,W]
    if image.ndim == 3:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 4:
        if image.shape[0] == 1:
            image = image.unsqueeze(0)
        elif image.shape[-1] == 1:
            image = image.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected image tensor shape: {tuple(image.shape)}")
    elif image.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected image tensor shape: {tuple(image.shape)}")

    # Mask -> [1,D,H,W]
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4:
        if mask.shape[0] != 1:
            raise ValueError(f"Unexpected mask batch shape: {tuple(mask.shape)}")
    elif mask.ndim == 5 and mask.shape[0] == 1 and mask.shape[1] == 1:
        mask = mask[:, 0]
    else:
        raise ValueError(f"Unexpected mask tensor shape: {tuple(mask.shape)}")

    return image.float(), mask.long()


def call_preprocess(part11, image_np: np.ndarray, mask_np: np.ndarray):
    image_np = normalize_3d_numpy(image_np, "image")
    mask_np = normalize_3d_numpy(mask_np, "mask")
    return part11.preprocess_case(image_np, mask_np)


def create_model(part11, device: torch.device):
    factory = part11.create_model
    sig = inspect.signature(factory)

    kwargs = {}
    args = []

    if len(sig.parameters) == 1:
        # Validated Part 11 signature: create_model(device)
        args = [device]
    else:
        # Defensive fallback for compatible variants.
        for name, p in sig.parameters.items():
            if name == "device":
                kwargs[name] = device
            elif name in ("in_channels", "in_chans"):
                kwargs[name] = 1
            elif name in ("out_channels", "num_classes"):
                kwargs[name] = NUM_CLASSES
            elif name in ("feature_size",):
                kwargs[name] = FEATURE_SIZE

    try:
        model = factory(*args, **kwargs)
    except TypeError as e:
        raise TypeError(
            f"Could not call Part 11 create_model(); signature={sig}; "
            f"error={e}"
        ) from e

    return model


# ============================================================================
# HISTORY AUDIT
# ============================================================================

def audit_history() -> Dict[str, Any]:
    df = pd.read_csv(PART15_HISTORY)

    print_kv("History rows", len(df))
    print_kv("History columns", list(df.columns))

    print("\nPART 15 HISTORY")
    print("-" * 78)

    print(df.to_string(index=False))

    numeric_candidates = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_dice",
        "val_dice",
        "best_val_dice",
        "learning_rate",
    ]

    summary = {}

    for col in numeric_candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            summary[col] = {
                "count": int(vals.notna().sum()),
                "min": float(vals.min()) if vals.notna().any() else None,
                "max": float(vals.max()) if vals.notna().any() else None,
                "first": float(vals.dropna().iloc[0]) if vals.notna().any() else None,
                "last": float(vals.dropna().iloc[-1]) if vals.notna().any() else None,
            }

    return {
        "columns": list(df.columns),
        "rows": len(df),
        "summary": summary,
        "records": df.to_dict(orient="records"),
    }


# ============================================================================
# CHECKPOINT AUDIT
# ============================================================================

def audit_checkpoint(part11, device: torch.device):
    checkpoint = torch.load(
        PART15_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model = create_model(part11, device)

    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    missing, unexpected = model.load_state_dict(state, strict=False)

    print_kv("Checkpoint epoch", checkpoint.get("epoch"))
    print_kv("Recorded best Dice", checkpoint.get("best_val_dice"))
    print_kv("Checkpoint seed", checkpoint.get("seed"))
    print_kv("Missing keys", len(missing))
    print_kv("Unexpected keys", len(unexpected))
    print_kv("SHA256", sha256_file(PART15_CHECKPOINT))

    if missing or unexpected:
        print("WARNING: checkpoint/model key mismatch detected.")

    return checkpoint, model


def find_output_head(model: torch.nn.Module):
    """
    Locate likely final segmentation classifier layers.
    We report candidates rather than assuming one exact MONAI internals path.
    """
    candidates = []

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
            if getattr(module, "out_channels", None) == NUM_CLASSES:
                candidates.append((name, module))

    return candidates


def parameter_stats(tensor: torch.Tensor) -> Dict[str, float]:
    t = tensor.detach().float().cpu()
    return {
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "min": float(t.min()),
        "max": float(t.max()),
        "abs_mean": float(t.abs().mean()),
    }


def audit_output_heads(model):
    candidates = find_output_head(model)

    print(f"Candidate {NUM_CLASSES}-class Conv3d heads : {len(candidates)}")

    rows = []

    for name, module in candidates:
        row = {
            "module": name,
            "type": module.__class__.__name__,
            "out_channels": int(module.out_channels),
            "in_channels": int(module.in_channels),
            "kernel_size": tuple(module.kernel_size),
        }

        if module.weight is not None:
            row.update({f"weight_{k}": v for k, v in parameter_stats(module.weight).items()})

        if module.bias is not None:
            bias_stats = parameter_stats(module.bias)
            row.update({f"bias_{k}": v for k, v in bias_stats.items()})
            row["bias_values"] = module.bias.detach().float().cpu().tolist()

        rows.append(row)

        print(f"\nHEAD: {name}")
        print(json.dumps(json_safe(row), indent=2))

    return rows


# ============================================================================
# CASE EVALUATION
# ============================================================================

def explicit_foreground_metrics(pred: torch.Tensor, target: torch.Tensor):
    """
    pred/target: [D,H,W] integer labels.
    Foreground = labels > 0.
    """
    pred_fg = pred > 0
    target_fg = target > 0

    tp = int((pred_fg & target_fg).sum().item())
    fp = int((pred_fg & ~target_fg).sum().item())
    fn = int((~pred_fg & target_fg).sum().item())

    pred_n = tp + fp
    target_n = tp + fn

    dice = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
    precision = tp / pred_n if pred_n else 0.0
    recall = tp / target_n if target_n else 0.0

    return dice, precision, recall, pred_n, target_n


def logit_probability_stats(logits: torch.Tensor) -> Dict[str, Any]:
    """
    logits: [1,C,D,H,W]
    """
    probs = torch.softmax(logits.float(), dim=1)
    pred = torch.argmax(logits, dim=1)

    bg_prob = probs[:, 0]
    fg_probs = probs[:, 1:]

    best_fg_prob = fg_probs.max(dim=1).values
    margin = best_fg_prob - bg_prob

    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)

    return {
        "background_probability_mean": float(bg_prob.mean()),
        "best_foreground_probability_mean": float(best_fg_prob.mean()),
        "fg_bg_margin_mean": float(margin.mean()),
        "entropy_mean": float(entropy.mean()),
        "pred_foreground_voxels": int((pred > 0).sum().item()),
        "pred_background_voxels": int((pred == 0).sum().item()),
        "total_voxels": int(pred.numel()),
        "pred_labels": {
            str(int(k)): int(v)
            for k, v in zip(*torch.unique(pred, return_counts=True))
        },
    }


def evaluate_cases(
    part11,
    part9,
    model,
    cohort: pd.DataFrame,
    device: torch.device,
    max_cases: int = 100,
):
    rows = []

    n = min(len(cohort), max_cases)

    model.eval()

    loss_fn = part11.DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
    )

    with torch.no_grad():
        for i in range(n):
            row = cohort.iloc[i]

            case_id = str(
                row.get("study_id", row.get("study", row.get("case_id", i)))
            )
            series_id = str(
                row.get("series_id", row.get("series", ""))
            )

            result = {
                "index": i + 1,
                "case_id": case_id,
                "series_id": series_id,
            }

            try:
                image, mask, info = part11.load_tensor_case(row, part9)

                image_np = normalize_3d_numpy(image, "loaded image")
                mask_np = normalize_3d_numpy(mask, "loaded mask")

                raw_fg = int((mask_np > 0).sum())

                processed = call_preprocess(part11, image_np, mask_np)
                image_p, mask_p = normalize_processed_case(processed)

                processed_fg = int((mask_p > 0).sum().item())

                image_p = image_p.to(device)
                mask_p = mask_p.to(device)

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=AMP_ENABLED and device.type == "cuda",
                ):
                    logits = model(image_p)

                target = mask_p.long()

                loss = loss_fn(logits, target)

                pred = torch.argmax(logits, dim=1)

                dice, precision, recall, pred_fg, target_fg = (
                    explicit_foreground_metrics(pred[0], target[0])
                )

                stats = logit_probability_stats(logits)

                result.update(
                    {
                        "status": "OK",
                        "raw_foreground_voxels": raw_fg,
                        "processed_foreground_voxels": processed_fg,
                        "foreground_retention": (
                            processed_fg / raw_fg if raw_fg else np.nan
                        ),
                        "loss": float(loss.detach().cpu()),
                        "dice_explicit": dice,
                        "precision": precision,
                        "recall": recall,
                        "target_foreground_voxels": target_fg,
                        "predicted_foreground_voxels": pred_fg,
                        **stats,
                    }
                )

            except Exception as exc:
                result.update(
                    {
                        "status": "ERROR",
                        "error": repr(exc),
                    }
                )

            rows.append(result)

            if i < 5 or (i + 1) % 25 == 0 or i + 1 == n:
                print(
                    f"  [{i+1:03d}/{n}] {case_id} | "
                    f"status={result['status']} | "
                    f"pred_fg={result.get('predicted_foreground_voxels', 'NA')} | "
                    f"Dice={result.get('dice_explicit', np.nan):.6f}"
                )

    return pd.DataFrame(rows)


# ============================================================================
# COHORT / HISTORY INTERPRETATION
# ============================================================================

def analyze_history_for_collapse(history: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame(history["records"])

    findings = []

    for col in ["train_dice", "val_dice"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            vals = vals.dropna()

            if len(vals) >= 2:
                if float(vals.iloc[-1]) <= float(vals.iloc[0]):
                    findings.append(
                        f"{col} did not improve from first to last recorded epoch."
                    )

    if "train_loss" in df.columns and "val_loss" in df.columns:
        tr = pd.to_numeric(df["train_loss"], errors="coerce").dropna()
        va = pd.to_numeric(df["val_loss"], errors="coerce").dropna()

        if len(tr) and len(va):
            findings.append(
                f"Final recorded train/validation loss: "
                f"{float(tr.iloc[-1]):.6f}/{float(va.iloc[-1]):.6f}"
            )

    return {"findings": findings}


# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        ensure_output_dirs()

        header("PHASE 4 - PART 22")
        print("RSNA-ONLY TRAINING DYNAMICS / CHECKPOINT COLLAPSE AUDIT")
        print()
        print("Evaluation / audit only.")
        print("No training is performed.")
        print("No model weights are modified.")
        print("No optimizer is created.")
        print("SPIDER is not used.")
        print("RSNA test set is not used.")
        print()
        print("Purpose:")
        print("Determine when/where the Part 15 foreground collapse occurred.")

        print()
        print_kv("PROJECT ROOT", PROJECT_ROOT)
        print_kv("RSNA DATASET", RSNA_ROOT)
        print_kv("PART 11 SOURCE", PART11_SOURCE)
        print_kv("PART 15 CHECKPOINT", PART15_CHECKPOINT)
        print_kv("PART 15 HISTORY", PART15_HISTORY)
        print_kv("OUTPUT DIRECTORY", OUTPUT_DIR)

        header("PATH VALIDATION")
        validate_paths()
        print("All required Part 22 paths found.")

        header("PYTORCH / GPU ENVIRONMENT")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print_kv("PyTorch version", torch.__version__)
        print_kv("CUDA available", torch.cuda.is_available())
        print_kv("Device", device)
        if device.type == "cuda":
            print_kv("GPU", torch.cuda.get_device_name(0))
            print_kv(
                "GPU memory",
                f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB",
            )
        print_kv("Patch size", PATCH_SIZE)
        print_kv("Feature size", FEATURE_SIZE)
        print_kv("Classes", NUM_CLASSES)
        print_kv("AMP", AMP_ENABLED)

        header("IMPORTING VALIDATED PART 11 IMPLEMENTATION")
        part11 = import_part11()
        print("✓ Corrected Part 11 imported.")
        print_kv("create_model signature", inspect.signature(part11.create_model))
        print_kv("preprocess_case signature", inspect.signature(part11.preprocess_case))
        print_kv("load_tensor_case signature", inspect.signature(part11.load_tensor_case))

        header("LOADING PART 9 THROUGH PART 11")
        part9 = part11.load_part9_module()
        print("✓ Part 9 loader imported.")

        header("PART 15 TRAINING HISTORY AUDIT")
        history = audit_history()

        header("PART 15 COHORT AUDIT")
        train_cohort = pd.read_csv(PART15_TRAIN_COHORT)
        val_cohort = pd.read_csv(PART15_VAL_COHORT)

        print_kv("Part 15 train cohort", len(train_cohort))
        print_kv("Part 15 validation cohort", len(val_cohort))

        train_keys = set(
            zip(
                train_cohort.get("study_id", pd.Series(dtype=str)).astype(str),
                train_cohort.get("series_id", pd.Series(dtype=str)).astype(str),
            )
        )
        val_keys = set(
            zip(
                val_cohort.get("study_id", pd.Series(dtype=str)).astype(str),
                val_cohort.get("series_id", pd.Series(dtype=str)).astype(str),
            )
        )

        print_kv("Train/validation overlap", len(train_keys & val_keys))

        header("PART 15 CHECKPOINT AUDIT")
        checkpoint, model = audit_checkpoint(part11, device)
        model.eval()

        header("SEGMENTATION OUTPUT HEAD AUDIT")
        head_rows = audit_output_heads(model)

        header("PART 15 CHECKPOINT OUTPUT AUDIT - TRAINING COHORT")
        print("Evaluating up to 100 exact Part 15 training cases.")
        train_results = evaluate_cases(
            part11,
            part9,
            model,
            train_cohort,
            device,
            max_cases=100,
        )

        header("PART 15 CHECKPOINT OUTPUT AUDIT - VALIDATION COHORT")
        print("Evaluating up to 100 exact Part 15 validation cases.")
        val_results = evaluate_cases(
            part11,
            part9,
            model,
            val_cohort,
            device,
            max_cases=100,
        )

        header("TRAINING / VALIDATION EXPLICIT METRICS")

        def summarize(df: pd.DataFrame, name: str):
            ok = df[df["status"] == "OK"].copy()

            print(f"\n{name}")
            print("-" * 78)
            print_kv("Successful cases", len(ok))
            print_kv(
                "Mean explicit Dice",
                f"{ok['dice_explicit'].mean():.6f}" if len(ok) else "NA",
            )
            print_kv(
                "Mean precision",
                f"{ok['precision'].mean():.6f}" if len(ok) else "NA",
            )
            print_kv(
                "Mean recall",
                f"{ok['recall'].mean():.6f}" if len(ok) else "NA",
            )
            print_kv(
                "Mean predicted foreground voxels",
                f"{ok['predicted_foreground_voxels'].mean():.2f}" if len(ok) else "NA",
            )
            print_kv(
                "Mean background probability",
                f"{ok['background_probability_mean'].mean():.6f}" if len(ok) else "NA",
            )
            print_kv(
                "Mean best foreground probability",
                f"{ok['best_foreground_probability_mean'].mean():.6f}" if len(ok) else "NA",
            )
            print_kv(
                "Mean FG-BG margin",
                f"{ok['fg_bg_margin_mean'].mean():.6f}" if len(ok) else "NA",
            )
            print_kv(
                "Empty predictions",
                int((ok["predicted_foreground_voxels"] == 0).sum())
                if len(ok) else "NA",
            )

            return {
                "cases": len(ok),
                "mean_dice": float(ok["dice_explicit"].mean()) if len(ok) else None,
                "mean_precision": float(ok["precision"].mean()) if len(ok) else None,
                "mean_recall": float(ok["recall"].mean()) if len(ok) else None,
                "mean_pred_fg": (
                    float(ok["predicted_foreground_voxels"].mean())
                    if len(ok) else None
                ),
                "mean_bg_probability": (
                    float(ok["background_probability_mean"].mean())
                    if len(ok) else None
                ),
                "mean_best_fg_probability": (
                    float(ok["best_foreground_probability_mean"].mean())
                    if len(ok) else None
                ),
                "mean_fg_bg_margin": (
                    float(ok["fg_bg_margin_mean"].mean())
                    if len(ok) else None
                ),
                "empty_predictions": (
                    int((ok["predicted_foreground_voxels"] == 0).sum())
                    if len(ok) else None
                ),
            }

        train_summary = summarize(train_results, "TRAINING COHORT")
        val_summary = summarize(val_results, "VALIDATION COHORT")

        header("PART 15 HISTORY INTERPRETATION")
        history_interpretation = analyze_history_for_collapse(history)

        for finding in history_interpretation["findings"]:
            print(f"- {finding}")

        header("PART 22 DIAGNOSIS")

        train_collapsed = (
            train_summary["cases"] > 0
            and train_summary["empty_predictions"] == train_summary["cases"]
        )

        val_collapsed = (
            val_summary["cases"] > 0
            and val_summary["empty_predictions"] == val_summary["cases"]
        )

        if train_collapsed and val_collapsed:
            diagnosis = "CHECKPOINT_FOREGROUND_COLLAPSE_CONFIRMED"
            decision = (
                "CAUTION - Part 15 best checkpoint predicts background only "
                "on both sampled training and validation cohorts."
            )
        elif train_collapsed:
            diagnosis = "TRAINING_COHORT_FOREGROUND_COLLAPSE"
            decision = (
                "CAUTION - checkpoint is foreground-collapsed on training "
                "cases; inspect training dynamics and output head."
            )
        elif val_collapsed:
            diagnosis = "VALIDATION_ONLY_FOREGROUND_COLLAPSE"
            decision = (
                "INVESTIGATE - checkpoint is foreground-collapsed on "
                "validation but not training cases."
            )
        else:
            diagnosis = "NO_UNIVERSAL_FOREGROUND_COLLAPSE"
            decision = (
                "PARTIAL - sampled checkpoint outputs are not universally "
                "foreground-empty; continue targeted diagnosis."
            )

        print_kv("Diagnosis", diagnosis)
        print_kv("Decision", decision)

        header("SAVING PART 22 RESULTS")

        train_results.to_csv(
            OUTPUT_DIR / "part22_training_checkpoint_case_metrics.csv",
            index=False,
        )

        val_results.to_csv(
            OUTPUT_DIR / "part22_validation_checkpoint_case_metrics.csv",
            index=False,
        )

        pd.DataFrame(head_rows).to_csv(
            OUTPUT_DIR / "part22_segmentation_head_audit.csv",
            index=False,
        )

        history_df = pd.DataFrame(history["records"])
        history_df.to_csv(
            OUTPUT_DIR / "part22_part15_history_copy.csv",
            index=False,
        )

        summary = {
            "phase": "Phase 4 - Part 22",
            "diagnosis": diagnosis,
            "decision": decision,
            "paths": {
                "project_root": str(PROJECT_ROOT),
                "rsna_root": str(RSNA_ROOT),
                "part11_source": str(PART11_SOURCE),
                "part15_checkpoint": str(PART15_CHECKPOINT),
            },
            "environment": {
                "pytorch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if device.type == "cuda"
                    else None
                ),
                "patch_size": PATCH_SIZE,
                "feature_size": FEATURE_SIZE,
                "classes": NUM_CLASSES,
                "amp": AMP_ENABLED,
            },
            "checkpoint": {
                "epoch": checkpoint.get("epoch"),
                "best_val_dice": checkpoint.get("best_val_dice"),
                "seed": checkpoint.get("seed"),
                "sha256": sha256_file(PART15_CHECKPOINT),
            },
            "history": history,
            "train_summary": train_summary,
            "validation_summary": val_summary,
            "head_audit": head_rows,
            "history_interpretation": history_interpretation,
            "integrity": {
                "training_performed": False,
                "weights_changed": False,
                "spider_used": False,
                "test_set_used": False,
            },
        }

        summary_path = (
            OUTPUT_DIR / "phase4_part22_training_dynamics_checkpoint_audit_summary.json"
        )

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(json_safe(summary), f, indent=2)

        report_path = (
            REPORT_DIR
            / "phase4_part22_training_dynamics_checkpoint_audit_report.txt"
        )

        with report_path.open("w", encoding="utf-8") as f:
            f.write("PHASE 4 - PART 22\n")
            f.write("RSNA-ONLY TRAINING DYNAMICS / CHECKPOINT COLLAPSE AUDIT\n\n")
            f.write(f"Diagnosis: {diagnosis}\n")
            f.write(f"Decision: {decision}\n\n")
            f.write("PART 15 CHECKPOINT\n")
            f.write(f"Epoch: {checkpoint.get('epoch')}\n")
            f.write(f"Recorded best Dice: {checkpoint.get('best_val_dice')}\n")
            f.write(f"SHA256: {sha256_file(PART15_CHECKPOINT)}\n\n")
            f.write("TRAIN SUMMARY\n")
            f.write(json.dumps(json_safe(train_summary), indent=2))
            f.write("\n\nVALIDATION SUMMARY\n")
            f.write(json.dumps(json_safe(val_summary), indent=2))
            f.write("\n\nHEAD AUDIT\n")
            f.write(json.dumps(json_safe(head_rows), indent=2))
            f.write("\n\nHISTORY INTERPRETATION\n")
            f.write(json.dumps(json_safe(history_interpretation), indent=2))
            f.write("\n")

        print(f"Saved: {OUTPUT_DIR / 'part22_training_checkpoint_case_metrics.csv'}")
        print(f"Saved: {OUTPUT_DIR / 'part22_validation_checkpoint_case_metrics.csv'}")
        print(f"Saved: {OUTPUT_DIR / 'part22_segmentation_head_audit.csv'}")
        print(f"Saved: {OUTPUT_DIR / 'part22_part15_history_copy.csv'}")
        print(f"Saved: {summary_path}")
        print(f"Saved: {report_path}")

        header("PART 22 FINAL SUMMARY")
        print_kv("Training audit cases", train_summary["cases"])
        print_kv("Training explicit Dice", train_summary["mean_dice"])
        print_kv("Training empty predictions", train_summary["empty_predictions"])
        print_kv("Validation audit cases", val_summary["cases"])
        print_kv("Validation explicit Dice", val_summary["mean_dice"])
        print_kv("Validation empty predictions", val_summary["empty_predictions"])
        print_kv("Diagnosis", diagnosis)

        print()
        print("FINAL DECISION")
        print(decision)

        print()
        print("SPIDER used          : NO")
        print("Test set used        : NO")
        print("Training performed   : NO")
        print("Model weights changed: NO")

        header("PHASE 4 - PART 22 COMPLETE")

    except Exception as exc:
        header("PART 22 ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
