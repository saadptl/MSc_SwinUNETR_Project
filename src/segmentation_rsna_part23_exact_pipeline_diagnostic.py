"""
PHASE 4 - PART 23
RSNA-ONLY EXACT SINGLE-CASE PIPELINE DIAGNOSTIC

Purpose
-------
Diagnose the Part 22 situation where all sampled case evaluations returned
status=ERROR and therefore produced no actual model-output statistics.

This script intentionally evaluates only:
  * one exact Part 15 training-cohort case
  * one exact Part 15 validation-cohort case

It does NOT train, update weights, create an optimizer, use SPIDER, or use
the RSNA test set.

Unlike Part 22, exceptions are NOT swallowed.  Every pipeline stage prints
type, shape, dtype, finite/range information and then continues only if the
stage succeeded.

The script uses the exact validated Part 11 APIs:
  create_model(device)
  load_tensor_case(row, part9)
  preprocess_case(image, mask)
  dice_from_prediction(logits, target)

It also contains a conservative adapter for common tensor/NumPy layouts.
The adapter never changes the semantic contents of the data; it only removes
a redundant batch/channel dimension when it is unambiguously present.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

PART11_SOURCE = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART15_CHECKPOINT = PART15_DIR / "checkpoints" / "best_model.pth"
PART15_TRAIN_COHORT = PART15_DIR / "part15_train_cohort.csv"
PART15_VAL_COHORT = PART15_DIR / "part15_validation_cohort.csv"

PART8_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
)

TRAIN_MANIFEST = PART8_DIR / "rsna_part8_train_manifest.csv"
VAL_MANIFEST = PART8_DIR / "rsna_part8_validation_manifest.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part23_exact_pipeline_diagnostic"
)

REPORT_DIR = OUTPUT_DIR / "reports"


# ---------------------------------------------------------------------------
# Fixed project configuration
# ---------------------------------------------------------------------------

PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12
NUM_CLASSES = 6
AMP_ENABLED = True

CLASS_NAMES = [
    "Background",
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def header(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def line(label: str, value: Any) -> None:
    print(f"{label:<38}: {value}")


def fail(message: str) -> None:
    raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def import_module_from_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_part11():
    module = import_module_from_file(
        PART11_SOURCE,
        "part11_corrected_for_part23",
    )

    required = [
        "create_model",
        "load_tensor_case",
        "preprocess_case",
        "dice_from_prediction",
        "select_pilot_rows",
    ]

    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise AttributeError(f"Part 11 missing required APIs: {missing}")

    return module


# ---------------------------------------------------------------------------
# Path / manifest helpers
# ---------------------------------------------------------------------------

def validate_paths() -> None:
    paths = {
        "RSNA root": RSNA_ROOT,
        "Part 11 source": PART11_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 train cohort": PART15_TRAIN_COHORT,
        "Part 15 validation cohort": PART15_VAL_COHORT,
        "Part 8 train manifest": TRAIN_MANIFEST,
        "Part 8 validation manifest": VAL_MANIFEST,
    }

    missing = [f"{k}: {v}" for k, v in paths.items() if not v.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required Part 23 input(s):\n" + "\n".join(missing)
        )


def load_cohort(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Cohort is empty: {path}")
    return df.copy()


def choose_row(df: pd.DataFrame, index: int = 0) -> pd.Series:
    if index >= len(df):
        raise IndexError(f"Cohort contains only {len(df)} rows")
    return df.iloc[index].copy()


# ---------------------------------------------------------------------------
# Generic diagnostics
# ---------------------------------------------------------------------------

def describe_value(name: str, value: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "name": name,
        "python_type": type(value).__name__,
    }

    if torch.is_tensor(value):
        x = value.detach()
        info.update(
            {
                "kind": "torch",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "numel": int(x.numel()),
                "requires_grad": bool(x.requires_grad),
            }
        )

        if x.numel():
            xf = x.float()
            finite = torch.isfinite(xf)
            info["finite"] = bool(finite.all().item())
            info["min"] = float(xf[finite].min().item()) if finite.any() else None
            info["max"] = float(xf[finite].max().item()) if finite.any() else None
            info["mean"] = float(xf[finite].mean().item()) if finite.any() else None
            info["unique_sample"] = (
                torch.unique(xf).detach().cpu().tolist()[:20]
                if x.numel() <= 100000
                else None
            )

    elif isinstance(value, np.ndarray):
        info.update(
            {
                "kind": "numpy",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": int(value.size),
            }
        )
        if value.size:
            finite = np.isfinite(value)
            info["finite"] = bool(finite.all())
            info["min"] = float(value[finite].min()) if finite.any() else None
            info["max"] = float(value[finite].max()) if finite.any() else None
            info["mean"] = float(value[finite].mean()) if finite.any() else None
            if value.size <= 100000:
                info["unique_sample"] = np.unique(value).tolist()[:20]

    elif isinstance(value, (list, tuple)):
        info["length"] = len(value)
        info["items"] = [
            describe_value(f"{name}[{i}]", x)
            for i, x in enumerate(value[:6])
        ]

    elif isinstance(value, dict):
        info["keys"] = list(value.keys())

    else:
        info["repr"] = repr(value)[:500]

    print(f"\n{name}")
    for k, v in info.items():
        if k != "name":
            print(f"  {k:<20}: {v}")

    return info


def normalize_image_for_numpy(value: Any) -> np.ndarray:
    """
    Convert an image-like result to a 3-D NumPy array [D,H,W].

    Only removes dimensions of size 1 when they are clearly batch/channel
    dimensions. A non-singleton 4-D layout is rejected instead of guessed.
    """
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]

    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 3:
        raise ValueError(
            f"Cannot safely convert image to [D,H,W]; got shape {arr.shape}"
        )

    return np.asarray(arr)


def normalize_mask_for_numpy(value: Any) -> np.ndarray:
    """
    Convert a mask-like result to a 3-D NumPy array [D,H,W].
    """
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]

    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 3:
        raise ValueError(
            f"Cannot safely convert mask to [D,H,W]; got shape {arr.shape}"
        )

    return np.asarray(arr)


def to_model_image(x: Any, device: torch.device) -> torch.Tensor:
    """
    Required model input: [B,1,D,H,W].
    """
    if not torch.is_tensor(x):
        x = torch.from_numpy(np.asarray(x))

    x = x.float()

    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 4:
        if x.shape[0] == 1:
            x = x.unsqueeze(0)
        elif x.shape[-1] == 1:
            x = x.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            raise ValueError(
                f"Ambiguous 4-D image layout for model: {tuple(x.shape)}"
            )
    elif x.ndim == 5:
        if x.shape[0] != 1 or x.shape[1] != 1:
            raise ValueError(
                f"Expected [B,1,D,H,W], got {tuple(x.shape)}"
            )
    else:
        raise ValueError(f"Unsupported image dimensions: {tuple(x.shape)}")

    return x.to(device)


def to_model_target(x: Any, device: torch.device) -> torch.Tensor:
    """
    Required target form for the project's Dice API: [B,D,H,W].
    """
    if not torch.is_tensor(x):
        x = torch.from_numpy(np.asarray(x))

    x = x.long()

    if x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"Expected B=1 target, got {tuple(x.shape)}")
    elif x.ndim == 5:
        if x.shape[0] == 1 and x.shape[1] == 1:
            x = x[:, 0]
        else:
            raise ValueError(
                f"Unsupported 5-D target layout: {tuple(x.shape)}"
            )
    else:
        raise ValueError(f"Unsupported target dimensions: {tuple(x.shape)}")

    return x.to(device)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "weights",
        ):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate

        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise ValueError(
        "Could not locate a model state_dict inside checkpoint."
    )


def checkpoint_summary(checkpoint: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {"type": type(checkpoint).__name__}

    result = {}
    for key, value in checkpoint.items():
        if key in {"model_state_dict", "state_dict", "model", "weights"}:
            if isinstance(value, dict):
                result[key] = f"{len(value)} tensors"
            else:
                result[key] = type(value).__name__
        elif isinstance(value, (int, float, str, bool, type(None))):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = f"{type(value).__name__}(len={len(value)})"
        else:
            result[key] = type(value).__name__
    return result


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def run_load_tensor_case(part11, row: pd.Series, part9) -> Tuple[Any, Any, Any]:
    """
    Call the exact validated API and print its raw return structure.
    """
    result = part11.load_tensor_case(row, part9)

    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise ValueError(
            "load_tensor_case did not return the expected tuple/list."
        )

    image = result[0]
    mask = result[1]
    info = result[2] if len(result) >= 3 else {}

    return image, mask, info


def run_preprocess_case(part11, image: np.ndarray, mask: np.ndarray):
    """
    Exact validated Part 11 call:
        preprocess_case(image, mask)
    """
    return part11.preprocess_case(image, mask)


# ---------------------------------------------------------------------------
# Model audit
# ---------------------------------------------------------------------------

@torch.no_grad()
def model_forward(
    model: torch.nn.Module,
    image: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(
        device_type="cuda",
        enabled=(AMP_ENABLED and image.device.type == "cuda"),
    ):
        logits = model(image)

    if not torch.is_tensor(logits):
        raise TypeError(f"Model returned {type(logits).__name__}, not Tensor.")

    if logits.ndim != 5:
        raise ValueError(
            f"Expected logits [B,C,D,H,W], got {tuple(logits.shape)}"
        )

    return logits


def explicit_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Any]:
    probs = torch.softmax(logits.float(), dim=1)
    pred = torch.argmax(probs, dim=1)

    pred_fg = pred > 0
    target_fg = target > 0

    intersection = (pred_fg & target_fg).sum().item()
    pred_count = pred_fg.sum().item()
    target_count = target_fg.sum().item()

    dice = (
        (2.0 * intersection) / (pred_count + target_count)
        if pred_count + target_count > 0
        else 1.0
    )

    precision = (
        intersection / pred_count
        if pred_count > 0
        else 0.0
    )

    recall = (
        intersection / target_count
        if target_count > 0
        else 0.0
    )

    bg_prob = probs[:, 0]
    fg_prob = probs[:, 1:]
    best_fg_prob = fg_prob.max(dim=1).values

    margin = best_fg_prob - bg_prob

    entropy = -(
        probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()
    ).sum(dim=1)

    labels, counts = torch.unique(pred, return_counts=True)

    return {
        "pred_fg_voxels": int(pred_fg.sum().item()),
        "target_fg_voxels": int(target_fg.sum().item()),
        "intersection_fg_voxels": int(intersection),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "mean_background_probability": float(bg_prob.mean().item()),
        "mean_best_foreground_probability": float(best_fg_prob.mean().item()),
        "mean_fg_bg_margin": float(margin.mean().item()),
        "mean_entropy": float(entropy.mean().item()),
        "pred_labels": {
            int(k): int(v)
            for k, v in zip(labels.cpu().tolist(), counts.cpu().tolist())
        },
        "logit_mean_per_class": logits.float().mean(
            dim=(0, 2, 3, 4)
        ).cpu().tolist(),
        "logit_max_per_class": logits.float().amax(
            dim=(0, 2, 3, 4)
        ).cpu().tolist(),
        "prob_mean_per_class": probs.mean(
            dim=(0, 2, 3, 4)
        ).cpu().tolist(),
        "prob_max_per_class": probs.amax(
            dim=(0, 2, 3, 4)
        ).cpu().tolist(),
    }


def audit_case(
    case_name: str,
    row: pd.Series,
    part11,
    part9,
    model,
    device: torch.device,
) -> Dict[str, Any]:

    header(f"CASE DIAGNOSTIC: {case_name}")

    result: Dict[str, Any] = {
        "case_name": case_name,
        "status": "STARTED",
    }

    print("Row fields:")
    for key, value in row.items():
        print(f"  {key}: {value}")

    # Stage 1 ---------------------------------------------------------------
    header("STAGE 1 - EXACT load_tensor_case()")

    try:
        raw_image, raw_mask, info = run_load_tensor_case(part11, row, part9)
        result["load_status"] = "OK"

        describe_value("raw_image", raw_image)
        describe_value("raw_mask", raw_mask)
        describe_value("loader_info", info)

        result["loader_info"] = str(info)[:2000]

    except Exception as exc:
        result["status"] = "LOAD_ERROR"
        result["error_stage"] = "load_tensor_case"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        traceback.print_exc()
        return result

    # Stage 2 ---------------------------------------------------------------
    header("STAGE 2 - CONVERT LOADER OUTPUT TO [D,H,W]")

    try:
        image_np = normalize_image_for_numpy(raw_image)
        mask_np = normalize_mask_for_numpy(raw_mask)

        describe_value("normalized_image_np", image_np)
        describe_value("normalized_mask_np", mask_np)

        unique_mask = np.unique(mask_np)
        print(f"\nMask labels: {unique_mask[:50].tolist()}")

        if not np.isfinite(image_np).all():
            raise ValueError("Normalized image contains non-finite values.")

        result["image_shape_before_preprocess"] = list(image_np.shape)
        result["mask_shape_before_preprocess"] = list(mask_np.shape)
        result["mask_unique_labels"] = unique_mask.tolist()[:50]

    except Exception as exc:
        result["status"] = "NORMALIZATION_ERROR"
        result["error_stage"] = "loader_output_normalization"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        traceback.print_exc()
        return result

    # Stage 3 ---------------------------------------------------------------
    header("STAGE 3 - EXACT preprocess_case(image, mask)")

    try:
        processed = run_preprocess_case(part11, image_np, mask_np)

        describe_value("preprocess_case_return", processed)

        if not isinstance(processed, (tuple, list)) or len(processed) < 2:
            raise ValueError(
                "preprocess_case did not return at least (image, mask)."
            )

        processed_image = processed[0]
        processed_mask = processed[1]

        describe_value("processed_image", processed_image)
        describe_value("processed_mask", processed_mask)

        result["preprocess_status"] = "OK"

    except Exception as exc:
        result["status"] = "PREPROCESS_ERROR"
        result["error_stage"] = "preprocess_case"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        traceback.print_exc()
        return result

    # Stage 4 ---------------------------------------------------------------
    header("STAGE 4 - NORMALIZE MODEL INPUT/TARGET CONTRACT")

    try:
        image = to_model_image(processed_image, device)
        target = to_model_target(processed_mask, device)

        describe_value("model_image", image)
        describe_value("model_target", target)

        expected = (1, 1, *PATCH_SIZE)
        if tuple(image.shape) != expected:
            raise ValueError(
                f"Model image contract mismatch: expected {expected}, "
                f"got {tuple(image.shape)}"
            )

        expected_target = (1, *PATCH_SIZE)
        if tuple(target.shape) != expected_target:
            raise ValueError(
                f"Target contract mismatch: expected {expected_target}, "
                f"got {tuple(target.shape)}"
            )

        result["model_input_shape"] = list(image.shape)
        result["target_shape"] = list(target.shape)

    except Exception as exc:
        result["status"] = "CONTRACT_ERROR"
        result["error_stage"] = "model_contract"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        traceback.print_exc()
        return result

    # Stage 5 ---------------------------------------------------------------
    header("STAGE 5 - MODEL FORWARD / RAW LOGITS")

    try:
        logits = model_forward(model, image)

        describe_value("raw_logits", logits)

        if tuple(logits.shape[0:2]) != (1, NUM_CLASSES):
            raise ValueError(
                f"Unexpected output batch/class dimensions: {tuple(logits.shape)}"
            )

        result["logits_shape"] = list(logits.shape)

    except Exception as exc:
        result["status"] = "MODEL_FORWARD_ERROR"
        result["error_stage"] = "model_forward"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        traceback.print_exc()
        return result

    # Stage 6 ---------------------------------------------------------------
    header("STAGE 6 - EXPLICIT PROBABILITY / PREDICTION METRICS")

    try:
        metrics = explicit_metrics(logits, target)

        print(f"Predicted foreground voxels : {metrics['pred_fg_voxels']}")
        print(f"Target foreground voxels    : {metrics['target_fg_voxels']}")
        print(f"Foreground Dice             : {metrics['dice']:.6f}")
        print(f"Precision                   : {metrics['precision']:.6f}")
        print(f"Recall                      : {metrics['recall']:.6f}")
        print(
            "Mean background probability : "
            f"{metrics['mean_background_probability']:.6f}"
        )
        print(
            "Mean best foreground prob.  : "
            f"{metrics['mean_best_foreground_probability']:.6f}"
        )
        print(
            "Mean FG-BG margin           : "
            f"{metrics['mean_fg_bg_margin']:.6f}"
        )
        print(f"Mean entropy                : {metrics['mean_entropy']:.6f}")
        print(f"Prediction labels            : {metrics['pred_labels']}")

        print("\nClass mean logits:")
        for name, value in zip(
            CLASS_NAMES,
            metrics["logit_mean_per_class"],
        ):
            print(f"  {name:<35}: {value:.6f}")

        print("\nClass mean probabilities:")
        for name, value in zip(
            CLASS_NAMES,
            metrics["prob_mean_per_class"],
        ):
            print(f"  {name:<35}: {value:.6f}")

        result.update(metrics)
        result["status"] = "SUCCESS"

    except Exception as exc:
        result["status"] = "METRIC_ERROR"
        result["error_stage"] = "explicit_metrics"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        traceback.print_exc()
        return result

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    header("PHASE 4 - PART 23")
    print("RSNA-ONLY EXACT SINGLE-CASE PIPELINE DIAGNOSTIC")
    print()
    print("Evaluation / diagnostic only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")
    print()
    print("Purpose:")
    print("Diagnose the Part 22 case-evaluation failures before any further")
    print("training or metric interpretation is attempted.")

    header("PROJECT PATHS")
    line("PROJECT ROOT", PROJECT_ROOT)
    line("RSNA DATASET", RSNA_ROOT)
    line("PART 11 SOURCE", PART11_SOURCE)
    line("PART 15 CHECKPOINT", PART15_CHECKPOINT)
    line("PART 15 TRAIN COHORT", PART15_TRAIN_COHORT)
    line("PART 15 VALIDATION COHORT", PART15_VAL_COHORT)
    line("OUTPUT DIRECTORY", OUTPUT_DIR)

    header("PATH VALIDATION")
    validate_paths()
    print("All required Part 23 paths found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    header("PYTORCH / GPU ENVIRONMENT")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    line("PyTorch version", torch.__version__)
    line("CUDA available", torch.cuda.is_available())
    line("Device", device)
    if device.type == "cuda":
        line("GPU", torch.cuda.get_device_name(device))
        line(
            "GPU memory",
            f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB",
        )
    line("Patch size", PATCH_SIZE)
    line("Feature size", FEATURE_SIZE)
    line("Classes", NUM_CLASSES)

    header("IMPORTING VALIDATED PART 11")
    part11 = import_part11()
    print("✓ Corrected Part 11 imported.")
    line("create_model signature", inspect.signature(part11.create_model))
    line("preprocess_case signature", inspect.signature(part11.preprocess_case))
    line("load_tensor_case signature", inspect.signature(part11.load_tensor_case))
    line("dice_from_prediction signature", inspect.signature(part11.dice_from_prediction))

    header("LOADING PART 9 THROUGH PART 11")
    part9 = part11.load_part9_module()
    print("✓ Part 9 loader imported.")

    header("LOADING PART 15 COHORTS")
    train_cohort = load_cohort(PART15_TRAIN_COHORT)
    val_cohort = load_cohort(PART15_VAL_COHORT)

    line("Part 15 train cohort", len(train_cohort))
    line("Part 15 validation cohort", len(val_cohort))

    train_row = choose_row(train_cohort, 0)
    val_row = choose_row(val_cohort, 0)

    line("Selected training case index", 0)
    line("Selected validation case index", 0)

    header("LOADING PART 15 BEST CHECKPOINT")
    checkpoint = torch.load(
        PART15_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    print("Checkpoint metadata:")
    print(json.dumps(checkpoint_summary(checkpoint), indent=2, default=str))

    state_dict = extract_state_dict(checkpoint)

    header("CREATING SWIN-UNETR")
    model = part11.create_model(device)
    model = model.to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    line("Total parameters", sum(p.numel() for p in model.parameters()))
    line("Missing keys", len(missing))
    line("Unexpected keys", len(unexpected))

    if missing or unexpected:
        print("Missing:", missing)
        print("Unexpected:", unexpected)
        raise RuntimeError("Checkpoint/model state mismatch.")

    model.eval()
    print("✓ Checkpoint loaded successfully.")

    line("Checkpoint SHA256", sha256_file(PART15_CHECKPOINT))

    header("RUNNING EXACT TWO-CASE DIAGNOSTIC")

    train_result = audit_case(
        "PART15_TRAIN_CASE_001",
        train_row,
        part11,
        part9,
        model,
        device,
    )

    val_result = audit_case(
        "PART15_VALIDATION_CASE_001",
        val_row,
        part11,
        part9,
        model,
        device,
    )

    header("PART 23 FINAL DIAGNOSTIC SUMMARY")

    for result in (train_result, val_result):
        print()
        print(result["case_name"])
        print("-" * 60)
        print("Status       :", result.get("status"))
        print("Error stage  :", result.get("error_stage", "None"))
        print("Error type   :", result.get("error_type", "None"))
        print("Error message:", result.get("error_message", "None"))

        if result.get("status") == "SUCCESS":
            print("Dice         :", f"{result['dice']:.6f}")
            print("Pred FG voxels:", result["pred_fg_voxels"])
            print("Target FG voxels:", result["target_fg_voxels"])
            print(
                "Mean BG prob.:",
                f"{result['mean_background_probability']:.6f}",
            )
            print(
                "Best FG prob.:",
                f"{result['mean_best_foreground_probability']:.6f}",
            )
            print(
                "FG-BG margin :",
                f"{result['mean_fg_bg_margin']:.6f}",
            )

    successful = [
        x for x in (train_result, val_result)
        if x.get("status") == "SUCCESS"
    ]

    if not successful:
        diagnosis = "PIPELINE_ERROR_REQUIRES_FIX"
        decision = (
            "FAIL-DIAGNOSTIC - neither case completed the exact pipeline; "
            "do not interpret Part 22 metrics or retrain yet."
        )
    else:
        empty = [
            x for x in successful
            if x.get("pred_fg_voxels", 0) == 0
        ]

        if len(empty) == len(successful):
            diagnosis = "FOREGROUND_COLLAPSE_CONFIRMED_ON_SUCCESSFUL_CASES"
            decision = (
                "CAUTION - successful exact cases are foreground-empty; "
                "inspect logits/probabilities and training labels next."
            )
        else:
            diagnosis = "FOREGROUND_NOT_UNIVERSALLY_EMPTY"
            decision = (
                "PARTIAL - at least one exact case produces foreground; "
                "continue with targeted output/threshold analysis."
            )

    print()
    print(f"Diagnosis : {diagnosis}")
    print(f"Decision  : {decision}")

    payload = {
        "phase": 4,
        "part": 23,
        "purpose": "exact_single_case_pipeline_diagnostic",
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint_sha256": sha256_file(PART15_CHECKPOINT),
        "train_result": train_result,
        "validation_result": val_result,
        "diagnosis": diagnosis,
        "decision": decision,
        "spider_used": False,
        "test_set_used": False,
        "training_performed": False,
        "model_weights_changed": False,
    }

    json_path = OUTPUT_DIR / "phase4_part23_exact_pipeline_diagnostic_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    rows = [train_result, val_result]
    csv_rows = []
    for r in rows:
        flat = {
            "case_name": r.get("case_name"),
            "status": r.get("status"),
            "error_stage": r.get("error_stage"),
            "error_type": r.get("error_type"),
            "error_message": r.get("error_message"),
            "dice": r.get("dice"),
            "precision": r.get("precision"),
            "recall": r.get("recall"),
            "pred_fg_voxels": r.get("pred_fg_voxels"),
            "target_fg_voxels": r.get("target_fg_voxels"),
            "mean_background_probability": r.get(
                "mean_background_probability"
            ),
            "mean_best_foreground_probability": r.get(
                "mean_best_foreground_probability"
            ),
            "mean_fg_bg_margin": r.get("mean_fg_bg_margin"),
            "mean_entropy": r.get("mean_entropy"),
            "image_shape_before_preprocess": str(
                r.get("image_shape_before_preprocess")
            ),
            "mask_shape_before_preprocess": str(
                r.get("mask_shape_before_preprocess")
            ),
            "model_input_shape": str(r.get("model_input_shape")),
            "target_shape": str(r.get("target_shape")),
            "logits_shape": str(r.get("logits_shape")),
            "pred_labels": json.dumps(
                r.get("pred_labels", {}),
                default=str,
            ),
            "logit_mean_per_class": json.dumps(
                r.get("logit_mean_per_class", []),
                default=str,
            ),
            "prob_mean_per_class": json.dumps(
                r.get("prob_mean_per_class", []),
                default=str,
            ),
        }
        csv_rows.append(flat)

    csv_path = OUTPUT_DIR / "part23_two_case_diagnostic_metrics.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    report_path = REPORT_DIR / "phase4_part23_exact_pipeline_diagnostic_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("PHASE 4 - PART 23\n")
        f.write("RSNA-ONLY EXACT SINGLE-CASE PIPELINE DIAGNOSTIC\n\n")
        f.write("No training performed.\n")
        f.write("No model weights modified.\n")
        f.write("SPIDER not used.\n")
        f.write("RSNA test set not used.\n\n")
        f.write(f"Diagnosis: {diagnosis}\n")
        f.write(f"Decision: {decision}\n\n")
        f.write(json.dumps(payload, indent=2, default=str))

    print()
    print("Saved:", csv_path)
    print("Saved:", json_path)
    print("Saved:", report_path)

    header("PHASE 4 - PART 23 COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPART 23 interrupted by user.")
        raise
    except Exception:
        header("PART 23 FATAL ERROR")
        traceback.print_exc()
        raise
