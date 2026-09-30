"""
==============================================================================
PHASE 4 - PART 21
RSNA-ONLY TRAINING LABEL / LOSS INTEGRITY AUDIT
==============================================================================

Purpose
-------
Investigate the Part 20 finding:

    100/100 validation cases -> empty foreground predictions
    strong background logit dominance

This audit determines whether the problem originates from:

    1. Training pseudo-mask construction
    2. Foreground/background class imbalance
    3. Preprocessing / resizing
    4. Loss configuration
    5. Checkpoint foreground logits
    6. Training-label availability

IMPORTANT
---------
Evaluation / audit only.

NO training is performed.
NO optimizer is created.
NO gradients are calculated.
NO model weights are modified.
NO checkpoint is overwritten.
SPIDER is not used.
RSNA test set is not used.

The Part 15 checkpoint is loaded read-only.
==============================================================================

"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part21_training_label_loss_integrity_audit"
)

REPORT_DIR = OUTPUT_DIR / "reports"

TRAIN_MANIFEST_CANDIDATES = [
    RSNA_ROOT / "train_manifest.csv",
    RSNA_ROOT / "manifests" / "train_manifest.csv",
]

VAL_MANIFEST_CANDIDATES = [
    RSNA_ROOT / "validation_manifest.csv",
    RSNA_ROOT / "manifests" / "validation_manifest.csv",
]

TRAIN_CASES = 500
VAL_CASES = 100

PATCH_SIZE = (64, 96, 96)
NUM_CLASSES = 6
BACKGROUND_CLASS = 0
FOREGROUND_CLASSES = tuple(range(1, NUM_CLASSES))

SEED = 42
AMP_ENABLED = True

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# =============================================================================
# PRINTING
# =============================================================================

def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subheader(title: str) -> None:
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def line(label: str, value: Any) -> None:
    print(f"{label:<38}: {value}")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# =============================================================================
# FILE HELPERS
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# =============================================================================
# PATH VALIDATION
# =============================================================================

def validate_paths() -> Tuple[Path, Path]:
    train_manifest = find_first_existing(TRAIN_MANIFEST_CANDIDATES)
    val_manifest = find_first_existing(VAL_MANIFEST_CANDIDATES)

    header("PATH VALIDATION")

    line("RSNA root", "FOUND" if RSNA_ROOT.exists() else "MISSING")
    line(
        "train manifest",
        str(train_manifest) if train_manifest else "MISSING",
    )
    line(
        "validation manifest",
        str(val_manifest) if val_manifest else "MISSING",
    )
    line(
        "Part 11 corrected source",
        "FOUND" if PART11_SOURCE.exists() else "MISSING",
    )
    line(
        "Part 15 checkpoint",
        "FOUND" if PART15_CHECKPOINT.exists() else "MISSING",
    )

    missing = []

    if not RSNA_ROOT.exists():
        missing.append(RSNA_ROOT)

    if train_manifest is None:
        missing.append(Path("train_manifest.csv"))

    if val_manifest is None:
        missing.append(Path("validation_manifest.csv"))

    if not PART11_SOURCE.exists():
        missing.append(PART11_SOURCE)

    if not PART15_CHECKPOINT.exists():
        missing.append(PART15_CHECKPOINT)

    if missing:
        raise FileNotFoundError(
            "Missing required Part 21 input(s):\n"
            + "\n".join(str(x) for x in missing)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    return train_manifest, val_manifest


# =============================================================================
# PYTORCH ENVIRONMENT
# =============================================================================

def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_environment(device: torch.device) -> None:
    header("PYTORCH / GPU ENVIRONMENT")

    line("PyTorch version", torch.__version__)
    line("CUDA available", torch.cuda.is_available())
    line("Device", str(device))

    if torch.cuda.is_available():
        line("GPU", torch.cuda.get_device_name(device))
        props = torch.cuda.get_device_properties(device)
        line("GPU memory", f"{props.total_memory / 1024**3:.2f} GB")
    else:
        line("GPU", "N/A")
        line("GPU memory", "N/A")

    line("Patch size", PATCH_SIZE)
    line("Feature size", 12)
    line("Classes", NUM_CLASSES)
    line("Training cases", TRAIN_CASES)
    line("Validation cases", VAL_CASES)
    line("AMP", AMP_ENABLED)


# =============================================================================
# PART 11 IMPORT
# =============================================================================

def import_part11():
    header("IMPORTING VALIDATED PART 11 IMPLEMENTATION")

    spec = importlib.util.spec_from_file_location(
        "part11_corrected",
        str(PART11_SOURCE),
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import Part 11: {PART11_SOURCE}")

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
        raise AttributeError(
            "Part 11 missing required API: " + ", ".join(missing)
        )

    line("Corrected Part 11", "IMPORTED")

    for name in required:
        line(name, "available")

    line(
        "create_model signature",
        inspect.signature(module.create_model),
    )

    line(
        "preprocess_case signature",
        inspect.signature(module.preprocess_case),
    )

    line(
        "load_tensor_case signature",
        inspect.signature(module.load_tensor_case),
    )

    return module


# =============================================================================
# ROBUST FUNCTION CALLERS
# =============================================================================

def call_create_model(part11, device):
    factory = part11.create_model
    sig = inspect.signature(factory)

    kwargs = {}

    for name, parameter in sig.parameters.items():
        if name == "device":
            kwargs[name] = device
        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(
                f"Unsupported required create_model argument: {name}"
            )

    return factory(**kwargs)


def call_preprocess_case(part11, loaded_case):
    """
    Part 11 validated API uses preprocess_case(case).

    Do NOT pass patch_size as a second positional argument.
    """

    fn = part11.preprocess_case
    sig = inspect.signature(fn)

    params = list(sig.parameters.values())

    if len(params) == 1:
        return fn(loaded_case)

    # Keyword-compatible fallback.
    kwargs = {}

    for p in params:
        if p.name in {"case", "case_data", "data", "loaded_case"}:
            kwargs[p.name] = loaded_case
        elif p.default is inspect.Parameter.empty:
            raise TypeError(
                f"Unsupported required preprocess_case argument: {p.name}"
            )

    return fn(**kwargs)


def call_load_tensor_case(part11, row):
    fn = part11.load_tensor_case
    sig = inspect.signature(fn)

    params = list(sig.parameters.values())

    if len(params) == 1:
        return fn(row)

    kwargs = {}

    for p in params:
        if p.name in {"row", "case", "case_row", "record"}:
            kwargs[p.name] = row
        elif p.default is inspect.Parameter.empty:
            raise TypeError(
                f"Unsupported required load_tensor_case argument: {p.name}"
            )

    return fn(**kwargs)


# =============================================================================
# TENSOR NORMALIZATION
# =============================================================================

def unwrap_tensor(value):
    if torch.is_tensor(value):
        return value

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)

    if isinstance(value, dict):
        preferred = [
            "image",
            "mask",
            "image_tensor",
            "mask_tensor",
            "img",
            "label",
            "target",
        ]

        for key in preferred:
            if key in value:
                candidate = value[key]
                if torch.is_tensor(candidate) or isinstance(
                    candidate, np.ndarray
                ):
                    return candidate

        for candidate in value.values():
            if torch.is_tensor(candidate) or isinstance(candidate, np.ndarray):
                return candidate

    if isinstance(value, (list, tuple)):
        for candidate in value:
            if torch.is_tensor(candidate) or isinstance(candidate, np.ndarray):
                return candidate

    return None


def extract_image_mask(loaded):
    """
    Best-effort extraction while preserving Part 11 output.

    Returns:
        image, mask
    """

    image = None
    mask = None

    if isinstance(loaded, dict):
        for key in ["image", "image_tensor", "img"]:
            if key in loaded:
                image = unwrap_tensor(loaded[key])
                if image is not None:
                    break

        for key in ["mask", "mask_tensor", "label", "target"]:
            if key in loaded:
                mask = unwrap_tensor(loaded[key])
                if mask is not None:
                    break

    elif isinstance(loaded, (tuple, list)):
        tensors = []

        for item in loaded:
            t = unwrap_tensor(item)
            if t is not None:
                tensors.append(t)

        if len(tensors) >= 2:
            image = tensors[0]
            mask = tensors[1]
        elif len(tensors) == 1:
            image = tensors[0]

    else:
        image = unwrap_tensor(loaded)

    return image, mask


# =============================================================================
# SHAPE NORMALIZATION
# =============================================================================

def normalize_image_for_model(image: torch.Tensor) -> torch.Tensor:
    image = torch.as_tensor(image)

    # Remove only leading singleton dimensions.
    while image.ndim > 5 and image.shape[0] == 1:
        image = image.squeeze(0)

    # Expected model input: [B, C, D, H, W]
    if image.ndim == 3:
        image = image.unsqueeze(0).unsqueeze(0)

    elif image.ndim == 4:
        # [C,D,H,W] or [B,D,H,W]
        if image.shape[0] == 1:
            image = image.unsqueeze(0)
        else:
            image = image.unsqueeze(1)

    elif image.ndim == 5:
        pass

    else:
        raise ValueError(
            f"Unsupported image shape: {tuple(image.shape)}"
        )

    return image.float()


def normalize_mask_for_metrics(mask: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(mask)

    while mask.ndim > 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    # Expected: [D,H,W]
    if mask.ndim == 3:
        return mask.long()

    if mask.ndim == 4:
        if mask.shape[0] == 1:
            return mask.squeeze(0).long()

        # If one-hot: [C,D,H,W]
        if mask.shape[0] == NUM_CLASSES:
            return torch.argmax(mask, dim=0).long()

    raise ValueError(
        f"Unsupported mask shape: {tuple(mask.shape)}"
    )


# =============================================================================
# LABEL STATISTICS
# =============================================================================

def calculate_label_statistics(mask: torch.Tensor) -> Dict[str, Any]:
    mask = normalize_mask_for_metrics(mask)

    flat = mask.reshape(-1)

    counts = torch.bincount(
        flat.clamp(min=0, max=NUM_CLASSES - 1),
        minlength=NUM_CLASSES,
    )

    total = int(flat.numel())

    result = {
        "total_voxels": total,
        "background_voxels": int(counts[0]),
        "foreground_voxels": int(counts[1:].sum()),
        "foreground_fraction": (
            float(counts[1:].sum()) / total if total else 0.0
        ),
        "empty_foreground": bool(int(counts[1:].sum()) == 0),
        "unique_labels": sorted(
            [int(x) for x in torch.unique(flat).detach().cpu().tolist()]
        ),
    }

    for c in range(NUM_CLASSES):
        result[f"class_{c}_voxels"] = int(counts[c])

    return result


# =============================================================================
# EXPLICIT METRICS
# =============================================================================

def explicit_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Any]:

    prediction = prediction.long()
    target = target.long()

    fg_pred = prediction > 0
    fg_target = target > 0

    tp = int((fg_pred & fg_target).sum())
    fp = int((fg_pred & ~fg_target).sum())
    fn = int((~fg_pred & fg_target).sum())

    pred_fg = int(fg_pred.sum())
    target_fg = int(fg_target.sum())

    dice = (
        2.0 * tp / (2.0 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    return {
        "pred_fg_voxels": pred_fg,
        "target_fg_voxels": target_fg,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "empty_prediction": pred_fg == 0,
        "empty_target": target_fg == 0,
    }


# =============================================================================
# MODEL OUTPUT STATISTICS
# =============================================================================

def logit_statistics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Any]:

    # [B,C,D,H,W]
    if logits.ndim != 5:
        raise ValueError(
            f"Expected logits [B,C,D,H,W], got {tuple(logits.shape)}"
        )

    logits = logits.detach().float()

    probabilities = torch.softmax(logits, dim=1)

    prediction = torch.argmax(logits, dim=1)

    # Remove batch dimension.
    pred = prediction[0].cpu()
    target = normalize_mask_for_metrics(target).cpu()

    probs = probabilities[0].cpu()

    bg_prob = probs[BACKGROUND_CLASS]

    fg_probs = probs[1:]

    best_fg_prob = fg_probs.max(dim=0).values

    margin = best_fg_prob - bg_prob

    entropy = -(
        probabilities
        * torch.log(probabilities.clamp_min(1e-12))
    ).sum(dim=1)

    pred_counts = torch.bincount(
        pred.reshape(-1),
        minlength=NUM_CLASSES,
    )

    result = {
        "logit_global_min": float(logits.min()),
        "logit_global_max": float(logits.max()),
        "logit_global_mean": float(logits.mean()),
        "background_logit_mean": float(
            logits[:, BACKGROUND_CLASS].mean()
        ),
        "foreground_logit_mean": float(
            logits[:, 1:].mean()
        ),
        "background_probability_mean": float(
            bg_prob.mean()
        ),
        "best_foreground_probability_mean": float(
            best_fg_prob.mean()
        ),
        "foreground_background_margin_mean": float(
            margin.mean()
        ),
        "foreground_background_margin_max": float(
            margin.max()
        ),
        "foreground_background_margin_min": float(
            margin.min()
        ),
        "entropy_mean": float(entropy.mean()),
        "prediction_fg_voxels": int((pred > 0).sum()),
        "prediction_total_voxels": int(pred.numel()),
        "prediction_background_fraction": float(
            (pred == 0).float().mean()
        ),
    }

    for c in range(NUM_CLASSES):
        result[f"pred_class_{c}_voxels"] = int(pred_counts[c])

        result[f"class_{c}_probability_mean"] = float(
            probs[c].mean()
        )

        result[f"class_{c}_logit_mean"] = float(
            logits[:, c].mean()
        )

    result.update(
        {
            "explicit_dice": explicit_metrics(pred, target)["dice"],
            "explicit_precision": explicit_metrics(pred, target)[
                "precision"
            ],
            "explicit_recall": explicit_metrics(pred, target)[
                "recall"
            ],
        }
    )

    return result


# =============================================================================
# THRESHOLD ANALYSIS
# =============================================================================

def threshold_analysis(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:

    probabilities = torch.softmax(logits.detach().float(), dim=1)

    fg_probability = probabilities[:, 1:].max(dim=1).values[0]

    target = normalize_mask_for_metrics(target).cpu()

    results = {}

    for threshold in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]:
        pred = (
            fg_probability.cpu() >= threshold
        ).long()

        target_fg = (target > 0).long()

        tp = int(((pred == 1) & (target_fg == 1)).sum())
        fp = int(((pred == 1) & (target_fg == 0)).sum())
        fn = int(((pred == 0) & (target_fg == 1)).sum())

        dice = (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn) > 0
            else 1.0
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        key = f"{threshold:.2f}"

        results[f"threshold_{key}_dice"] = dice
        results[f"threshold_{key}_precision"] = precision
        results[f"threshold_{key}_recall"] = recall
        results[f"threshold_{key}_positive_voxels"] = int(
            pred.sum()
        )

    return results


# =============================================================================
# SAFE LOSS AUDIT
# =============================================================================

def inspect_loss_configuration(part11):
    header("LOSS CONFIGURATION AUDIT")

    loss_cls = getattr(part11, "DiceCELoss", None)

    if loss_cls is None:
        print("DiceCELoss not exposed by Part 11.")
        return {
            "available": False,
            "configuration": {},
        }

    configuration = {
        "class": str(loss_cls),
    }

    try:
        loss_fn = loss_cls(
            to_onehot_y=True,
            softmax=True,
        )

        configuration["default_audit_instance"] = repr(loss_fn)

    except Exception as exc:
        configuration["default_audit_instance_error"] = str(exc)

    for attr in [
        "to_onehot_y",
        "softmax",
        "sigmoid",
        "include_background",
        "lambda_dice",
        "lambda_ce",
        "smooth_nr",
        "smooth_dr",
    ]:
        if hasattr(loss_cls, attr):
            try:
                configuration[attr] = getattr(loss_cls, attr)
            except Exception:
                pass

    print(json.dumps(configuration, indent=2, default=str))

    return {
        "available": True,
        "configuration": configuration,
    }


# =============================================================================
# CHECKPOINT AUDIT
# =============================================================================

def load_checkpoint(model: torch.nn.Module, device: torch.device):
    header("PART 15 CHECKPOINT AUDIT")

    checkpoint = torch.load(
        PART15_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    state_dict = None

    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "weights",
        ]:
            if key in checkpoint and isinstance(
                checkpoint[key], dict
            ):
                state_dict = checkpoint[key]
                break

    if state_dict is None and isinstance(checkpoint, dict):
        # Detect a raw state dictionary.
        if all(
            isinstance(v, torch.Tensor)
            for v in checkpoint.values()
        ):
            state_dict = checkpoint

    if state_dict is None:
        raise RuntimeError(
            "Could not identify model state_dict in checkpoint."
        )

    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    line(
        "Checkpoint epoch",
        checkpoint.get("epoch", "unknown")
        if isinstance(checkpoint, dict)
        else "unknown",
    )

    line(
        "Recorded best Dice",
        checkpoint.get("best_val_dice", "unknown")
        if isinstance(checkpoint, dict)
        else "unknown",
    )

    line(
        "Missing keys",
        len(incompatible.missing_keys),
    )

    line(
        "Unexpected keys",
        len(incompatible.unexpected_keys),
    )

    return checkpoint


# =============================================================================
# TRAINING COHORT RECONSTRUCTION
# =============================================================================

def reconstruct_train_cohort(
    part11,
    train_manifest: Path,
):
    header("RECONSTRUCTING PART 15 TRAINING COHORT")

    df = pd.read_csv(train_manifest)

    line("Training manifest total", len(df))

    try:
        cohort = part11.select_pilot_rows(
            str(train_manifest),
            TRAIN_CASES,
            SEED,
        )
    except TypeError:
        try:
            cohort = part11.select_pilot_rows(
                str(train_manifest),
                TRAIN_CASES,
            )
        except TypeError:
            cohort = part11.select_pilot_rows(
                df,
                TRAIN_CASES,
                SEED,
            )

    if not isinstance(cohort, pd.DataFrame):
        cohort = pd.DataFrame(cohort)

    line("Selected training cases", len(cohort))

    if len(cohort) != TRAIN_CASES:
        print(
            f"WARNING: expected {TRAIN_CASES}, "
            f"got {len(cohort)}"
        )

    return cohort


# =============================================================================
# CASE IDENTIFICATION
# =============================================================================

def case_identifier(row: pd.Series, index: int) -> str:
    preferred = [
        "study_id",
        "studyId",
        "study",
        "patient_id",
        "patientId",
        "id",
    ]

    values = []

    for key in preferred:
        if key in row.index:
            value = row[key]

            if pd.notna(value):
                values.append(str(value))

    if values:
        return "|".join(values)

    return f"row_{index:04d}"


# =============================================================================
# SINGLE CASE AUDIT
# =============================================================================

def audit_training_case(
    part11,
    row: pd.Series,
    index: int,
    model: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Any]:

    case_id = case_identifier(row, index)

    result = {
        "index": index,
        "case_id": case_id,
        "status": "OK",
    }

    try:
        loaded = call_load_tensor_case(part11, row)

        image, mask = extract_image_mask(loaded)

        if image is None:
            raise RuntimeError(
                "Could not extract image tensor from load_tensor_case output."
            )

        if mask is None:
            raise RuntimeError(
                "Could not extract mask tensor from load_tensor_case output."
            )

        raw_image_shape = tuple(image.shape)
        raw_mask_shape = tuple(mask.shape)

        raw_stats = calculate_label_statistics(mask)

        result["raw_image_shape"] = str(raw_image_shape)
        result["raw_mask_shape"] = str(raw_mask_shape)

        for key, value in raw_stats.items():
            result[f"raw_{key}"] = value

        # -------------------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------------------

        processed = call_preprocess_case(
            part11,
            loaded,
        )

        processed_image = None
        processed_mask = None

        if isinstance(processed, dict):
            processed_image = unwrap_tensor(
                processed.get("image")
                or processed.get("image_tensor")
                if "image" in processed
                else processed.get("image_tensor")
            )

            if processed_image is None:
                for key in ["image", "image_tensor", "img"]:
                    if key in processed:
                        processed_image = unwrap_tensor(
                            processed[key]
                        )
                        if processed_image is not None:
                            break

            for key in [
                "mask",
                "mask_tensor",
                "label",
                "target",
            ]:
                if key in processed:
                    processed_mask = unwrap_tensor(
                        processed[key]
                    )
                    if processed_mask is not None:
                        break

        elif isinstance(processed, (tuple, list)):
            tensors = []

            for item in processed:
                t = unwrap_tensor(item)

                if t is not None:
                    tensors.append(t)

            if len(tensors) >= 2:
                processed_image = tensors[0]
                processed_mask = tensors[1]

        else:
            processed_image = unwrap_tensor(processed)

        if processed_image is None:
            # Some Part 11 versions return only an image.
            processed_image = unwrap_tensor(image)

        if processed_mask is None:
            processed_mask = unwrap_tensor(mask)

        processed_image = normalize_image_for_model(
            processed_image
        )

        processed_mask = normalize_mask_for_metrics(
            processed_mask
        )

        result["processed_image_shape"] = str(
            tuple(processed_image.shape)
        )

        result["processed_mask_shape"] = str(
            tuple(processed_mask.shape)
        )

        processed_stats = calculate_label_statistics(
            processed_mask
        )

        for key, value in processed_stats.items():
            result[f"processed_{key}"] = value

        # -------------------------------------------------------------
        # LABEL RETENTION
        # -------------------------------------------------------------

        raw_fg = raw_stats["foreground_voxels"]
        proc_fg = processed_stats["foreground_voxels"]

        result["foreground_retention_ratio"] = (
            proc_fg / raw_fg if raw_fg > 0 else 0.0
        )

        result["foreground_destroyed_by_preprocessing"] = (
            raw_fg > 0 and proc_fg == 0
        )

        # -------------------------------------------------------------
        # MODEL AUDIT
        # -------------------------------------------------------------

        image_device = processed_image.to(device)

        with torch.inference_mode():
            if device.type == "cuda" and AMP_ENABLED:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = model(image_device)
            else:
                logits = model(image_device)

        # Ensure target dimensions.
        target = processed_mask

        # Handle possible [B,D,H,W].
        if target.ndim == 3:
            target = target.unsqueeze(0)

        model_stats = logit_statistics(
            logits,
            target,
        )

        result.update(model_stats)

        threshold_stats = threshold_analysis(
            logits,
            target,
        )

        result.update(threshold_stats)

        result["status"] = "OK"

        del image_device
        del logits

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = repr(exc)

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    seed_everything(SEED)

    try:

        header("PHASE 4 - PART 21")
        print("RSNA-ONLY TRAINING LABEL / LOSS INTEGRITY AUDIT")
        print()
        print("Evaluation / audit only.")
        print("No training is performed.")
        print("No model weights are modified.")
        print("No optimizer is created.")
        print("SPIDER is not used.")
        print("RSNA test set is not used.")
        print()
        print("Purpose:")
        print("Investigate the Part 20 foreground-logit collapse.")
        print()
        print("Part 21 checks:")
        print("  - training pseudo-mask distribution")
        print("  - foreground/background balance")
        print("  - per-class positive voxels")
        print("  - preprocessing label retention")
        print("  - DiceCELoss configuration")
        print("  - Part 15 checkpoint output behavior")
        print("  - explicit foreground metrics")

        line("PROJECT ROOT", PROJECT_ROOT)
        line("RSNA DATASET", RSNA_ROOT)
        line("PART 11 SOURCE", PART11_SOURCE)
        line("PART 15 CHECKPOINT", PART15_CHECKPOINT)
        line("OUTPUT DIRECTORY", OUTPUT_DIR)

        train_manifest, val_manifest = validate_paths()

        device = get_device()

        print_environment(device)

        part11 = import_part11()

        # -------------------------------------------------------------
        # PART 9
        # -------------------------------------------------------------

        header("LOADING PART 9 THROUGH PART 11")

        try:
            part9 = part11.load_part9_module()
            print("✓ Part 9 loader imported.")
        except TypeError:
            part9 = part11.load_part9_module(
                RSNA_ROOT
            )
            print("✓ Part 9 loader imported with RSNA root.")

        # -------------------------------------------------------------
        # TRAIN COHORT
        # -------------------------------------------------------------

        train_cohort = reconstruct_train_cohort(
            part11,
            train_manifest,
        )

        # -------------------------------------------------------------
        # LOSS
        # -------------------------------------------------------------

        loss_audit = inspect_loss_configuration(
            part11
        )

        # -------------------------------------------------------------
        # MODEL
        # -------------------------------------------------------------

        header("CREATING SWIN-UNETR")

        model = call_create_model(
            part11,
            device,
        )

        model = model.to(device)
        model.eval()

        line(
            "Total parameters",
            sum(p.numel() for p in model.parameters()),
        )

        # -------------------------------------------------------------
        # CHECKPOINT
        # -------------------------------------------------------------

        checkpoint = load_checkpoint(
            model,
            device,
        )

        # -------------------------------------------------------------
        # CHECKPOINT HASH
        # -------------------------------------------------------------

        checkpoint_hash = sha256_file(
            PART15_CHECKPOINT
        )

        line(
            "Checkpoint SHA256",
            checkpoint_hash,
        )

        # -------------------------------------------------------------
        # AUDIT CASES
        # -------------------------------------------------------------

        header("STARTING TRAINING-LABEL INTEGRITY AUDIT")

        print(
            f"Auditing {len(train_cohort)} training cases."
        )

        rows = []

        for i, (_, row) in enumerate(
            train_cohort.iterrows(),
            start=1,
        ):

            result = audit_training_case(
                part11,
                row,
                i,
                model,
                device,
            )

            rows.append(result)

            if (
                i <= 5
                or i % 25 == 0
                or i == len(train_cohort)
            ):
                case_id = result.get(
                    "case_id",
                    f"case_{i}",
                )

                status = result.get(
                    "status",
                    "UNKNOWN",
                )

                raw_fg = result.get(
                    "raw_foreground_voxels",
                    "NA",
                )

                proc_fg = result.get(
                    "processed_foreground_voxels",
                    "NA",
                )

                pred_fg = result.get(
                    "prediction_fg_voxels",
                    "NA",
                )

                dice = result.get(
                    "explicit_dice",
                    float("nan"),
                )

                print(
                    f"  [{i:03d}/{len(train_cohort)}] "
                    f"{case_id} | "
                    f"status={status} | "
                    f"raw_fg={raw_fg} | "
                    f"processed_fg={proc_fg} | "
                    f"pred_fg={pred_fg} | "
                    f"Dice={dice:.6f}"
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()

            gc.collect()

        df_results = pd.DataFrame(rows)

        # -------------------------------------------------------------
        # AGGREGATE LABEL AUDIT
        # -------------------------------------------------------------

        header("TRAINING LABEL DISTRIBUTION SUMMARY")

        successful = df_results[
            df_results["status"] == "OK"
        ].copy()

        errors = df_results[
            df_results["status"] != "OK"
        ].copy()

        line("Training cases audited", len(df_results))
        line("Successful cases", len(successful))
        line("Error cases", len(errors))

        if len(successful):

            raw_fg = successful[
                "raw_foreground_voxels"
            ]

            proc_fg = successful[
                "processed_foreground_voxels"
            ]

            line(
                "Mean raw foreground voxels",
                f"{raw_fg.mean():.2f}",
            )

            line(
                "Median raw foreground voxels",
                f"{raw_fg.median():.2f}",
            )

            line(
                "Mean processed foreground voxels",
                f"{proc_fg.mean():.2f}",
            )

            line(
                "Median processed foreground voxels",
                f"{proc_fg.median():.2f}",
            )

            line(
                "Empty raw masks",
                int(
                    successful[
                        "raw_empty_foreground"
                    ].sum()
                ),
            )

            line(
                "Empty processed masks",
                int(
                    successful[
                        "processed_empty_foreground"
                    ].sum()
                ),
            )

            line(
                "Masks losing all foreground",
                int(
                    successful[
                        "foreground_destroyed_by_preprocessing"
                    ].sum()
                ),
            )

            for c in FOREGROUND_CLASSES:

                column = f"processed_class_{c}_voxels"

                if column in successful.columns:
                    line(
                        f"Mean class {c} voxels",
                        f"{successful[column].mean():.2f}",
                    )

        # -------------------------------------------------------------
        # MODEL OUTPUT SUMMARY
        # -------------------------------------------------------------

        header("PART 15 CHECKPOINT OUTPUT SUMMARY ON TRAINING COHORT")

        if len(successful):

            pred_fg = successful[
                "prediction_fg_voxels"
            ]

            bg_fraction = successful[
                "prediction_background_fraction"
            ]

            fg_prob = successful[
                "best_foreground_probability_mean"
            ]

            bg_prob = successful[
                "background_probability_mean"
            ]

            margin = successful[
                "foreground_background_margin_mean"
            ]

            dice = successful[
                "explicit_dice"
            ]

            precision = successful[
                "explicit_precision"
            ]

            recall = successful[
                "explicit_recall"
            ]

            line(
                "Mean predicted foreground voxels",
                f"{pred_fg.mean():.2f}",
            )

            line(
                "Empty predictions",
                int(
                    successful[
                        "prediction_fg_voxels"
                    ].eq(0).sum()
                ),
            )

            line(
                "Mean background fraction",
                f"{bg_fraction.mean():.6f}",
            )

            line(
                "Mean background probability",
                f"{bg_prob.mean():.6f}",
            )

            line(
                "Mean best foreground probability",
                f"{fg_prob.mean():.6f}",
            )

            line(
                "Mean FG-BG probability margin",
                f"{margin.mean():.6f}",
            )

            line(
                "Explicit foreground Dice",
                f"{dice.mean():.6f}",
            )

            line(
                "Explicit precision",
                f"{precision.mean():.6f}",
            )

            line(
                "Explicit recall",
                f"{recall.mean():.6f}",
            )

        # -------------------------------------------------------------
        # PREPROCESSING RETENTION
        # -------------------------------------------------------------

        header("PREPROCESSING FOREGROUND RETENTION")

        if len(successful):

            retention = successful[
                "foreground_retention_ratio"
            ].replace(
                [np.inf, -np.inf],
                np.nan,
            )

            line(
                "Mean foreground retention",
                f"{retention.mean():.6f}",
            )

            line(
                "Median foreground retention",
                f"{retention.median():.6f}",
            )

            line(
                "Cases with complete foreground loss",
                int(
                    successful[
                        "foreground_destroyed_by_preprocessing"
                    ].sum()
                ),
            )

        # -------------------------------------------------------------
        # DIAGNOSIS
        # -------------------------------------------------------------

        header("PART 21 DIAGNOSIS")

        diagnosis = "INCONCLUSIVE"

        if len(successful):

            empty_masks = int(
                successful[
                    "processed_empty_foreground"
                ].sum()
            )

            destroyed = int(
                successful[
                    "foreground_destroyed_by_preprocessing"
                ].sum()
            )

            empty_predictions = int(
                successful[
                    "prediction_fg_voxels"
                ].eq(0).sum()
            )

            mean_retention = float(
                successful[
                    "foreground_retention_ratio"
                ].replace(
                    [np.inf, -np.inf],
                    np.nan,
                ).mean()
            )

            mean_margin = float(
                successful[
                    "foreground_background_margin_mean"
                ].mean()
            )

            if empty_masks > 0:
                diagnosis = (
                    "TRAINING_LABELS_CONTAIN_EMPTY_FOREGROUND"
                )

            elif destroyed > 0:
                diagnosis = (
                    "PREPROCESSING_DESTROYS_FOREGROUND_LABELS"
                )

            elif (
                empty_predictions == len(successful)
                and mean_margin < 0
            ):
                diagnosis = (
                    "MODEL_FOREGROUND_COLLAPSE_CONFIRMED"
                )

            elif mean_retention < 0.25:
                diagnosis = (
                    "SEVERE_FOREGROUND_LABEL_LOSS_DURING_PREPROCESSING"
                )

            else:
                diagnosis = (
                    "LABEL_PIPELINE_APPEARS_INTACT; "
                    "INVESTIGATE LOSS/OPTIMIZATION"
                )

        print(f"Diagnosis: {diagnosis}")

        # -------------------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------------------

        header("SAVING PART 21 RESULTS")

        case_csv = (
            OUTPUT_DIR
            / "part21_training_case_integrity_metrics.csv"
        )

        class_rows = []

        if len(successful):

            for c in range(NUM_CLASSES):

                class_rows.append(
                    {
                        "class_id": c,
                        "class_name": CLASS_NAMES[c],
                        "mean_raw_voxels": successful[
                            f"raw_class_{c}_voxels"
                        ].mean(),
                        "median_raw_voxels": successful[
                            f"raw_class_{c}_voxels"
                        ].median(),
                        "mean_processed_voxels": successful[
                            f"processed_class_{c}_voxels"
                        ].mean(),
                        "median_processed_voxels": successful[
                            f"processed_class_{c}_voxels"
                        ].median(),
                        "zero_raw_cases": int(
                            (
                                successful[
                                    f"raw_class_{c}_voxels"
                                ]
                                == 0
                            ).sum()
                        ),
                        "zero_processed_cases": int(
                            (
                                successful[
                                    f"processed_class_{c}_voxels"
                                ]
                                == 0
                            ).sum()
                        ),
                    }
                )

        class_csv = (
            OUTPUT_DIR
            / "part21_training_class_distribution.csv"
        )

        summary = {
            "phase": "Phase 4 - Part 21",
            "purpose": (
                "Training label / loss integrity audit"
            ),
            "project_root": str(PROJECT_ROOT),
            "rsna_root": str(RSNA_ROOT),
            "part11_source": str(PART11_SOURCE),
            "part15_checkpoint": str(PART15_CHECKPOINT),
            "part15_checkpoint_sha256": checkpoint_hash,
            "train_manifest": str(train_manifest),
            "validation_manifest": str(val_manifest),
            "training_cases_requested": TRAIN_CASES,
            "training_cases_audited": int(len(df_results)),
            "successful_cases": int(len(successful)),
            "error_cases": int(len(errors)),
            "diagnosis": diagnosis,
            "loss_audit": loss_audit,
            "checkpoint_epoch": (
                checkpoint.get("epoch")
                if isinstance(checkpoint, dict)
                else None
            ),
            "checkpoint_best_dice": (
                checkpoint.get("best_val_dice")
                if isinstance(checkpoint, dict)
                else None
            ),
            "no_training": True,
            "weights_modified": False,
            "spider_used": False,
            "test_set_used": False,
        }

        if len(successful):

            summary.update(
                {
                    "mean_raw_foreground_voxels": float(
                        successful[
                            "raw_foreground_voxels"
                        ].mean()
                    ),
                    "mean_processed_foreground_voxels": float(
                        successful[
                            "processed_foreground_voxels"
                        ].mean()
                    ),
                    "empty_raw_masks": int(
                        successful[
                            "raw_empty_foreground"
                        ].sum()
                    ),
                    "empty_processed_masks": int(
                        successful[
                            "processed_empty_foreground"
                        ].sum()
                    ),
                    "foreground_destroyed_by_preprocessing": int(
                        successful[
                            "foreground_destroyed_by_preprocessing"
                        ].sum()
                    ),
                    "mean_foreground_retention": float(
                        successful[
                            "foreground_retention_ratio"
                        ].replace(
                            [np.inf, -np.inf],
                            np.nan,
                        ).mean()
                    ),
                    "empty_predictions": int(
                        successful[
                            "prediction_fg_voxels"
                        ].eq(0).sum()
                    ),
                    "mean_background_probability": float(
                        successful[
                            "background_probability_mean"
                        ].mean()
                    ),
                    "mean_best_foreground_probability": float(
                        successful[
                            "best_foreground_probability_mean"
                        ].mean()
                    ),
                    "mean_foreground_background_margin": float(
                        successful[
                            "foreground_background_margin_mean"
                        ].mean()
                    ),
                    "mean_explicit_foreground_dice": float(
                        successful[
                            "explicit_dice"
                        ].mean()
                    ),
                    "mean_explicit_precision": float(
                        successful[
                            "explicit_precision"
                        ].mean()
                    ),
                    "mean_explicit_recall": float(
                        successful[
                            "explicit_recall"
                        ].mean()
                    ),
                }
            )

        df_results.to_csv(
            case_csv,
            index=False,
        )

        pd.DataFrame(class_rows).to_csv(
            class_csv,
            index=False,
        )

        summary_json = (
            OUTPUT_DIR
            / "phase4_part21_training_label_loss_integrity_summary.json"
        )

        with summary_json.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                default=str,
            )

        report_txt = (
            REPORT_DIR
            / "phase4_part21_training_label_loss_integrity_report.txt"
        )

        with report_txt.open(
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "PHASE 4 - PART 21\n"
                "RSNA-ONLY TRAINING LABEL / LOSS INTEGRITY AUDIT\n"
                "=" * 78
                + "\n\n"
            )

            f.write(
                "This is an evaluation/audit-only stage.\n"
                "No training was performed.\n"
                "No model weights were modified.\n"
                "SPIDER was not used.\n"
                "RSNA test set was not used.\n\n"
            )

            f.write(
                json.dumps(
                    summary,
                    indent=2,
                    default=str,
                )
            )

            f.write("\n\n")

            if len(errors):
                f.write("CASE ERRORS\n")
                f.write("-" * 78 + "\n")
                f.write(
                    errors.to_string(index=False)
                )
                f.write("\n")

        print(f"Saved: {case_csv}")
        print(f"Saved: {class_csv}")
        print(f"Saved: {summary_json}")
        print(f"Saved: {report_txt}")

        # -------------------------------------------------------------
        # FINAL
        # -------------------------------------------------------------

        header("PART 21 FINAL SUMMARY")

        line(
            "Training cases audited",
            len(df_results),
        )

        line(
            "Successful cases",
            len(successful),
        )

        line(
            "Error cases",
            len(errors),
        )

        if len(successful):

            line(
                "Empty processed masks",
                int(
                    successful[
                        "processed_empty_foreground"
                    ].sum()
                ),
            )

            line(
                "Foreground destroyed by preprocessing",
                int(
                    successful[
                        "foreground_destroyed_by_preprocessing"
                    ].sum()
                ),
            )

            line(
                "Empty model predictions",
                int(
                    successful[
                        "prediction_fg_voxels"
                    ].eq(0).sum()
                ),
            )

            line(
                "Mean explicit foreground Dice",
                f"{successful['explicit_dice'].mean():.6f}",
            )

            line(
                "Mean explicit precision",
                f"{successful['explicit_precision'].mean():.6f}",
            )

            line(
                "Mean explicit recall",
                f"{successful['explicit_recall'].mean():.6f}",
            )

            line(
                "Mean FG-BG probability margin",
                f"{successful['foreground_background_margin_mean'].mean():.6f}",
            )

        line("Diagnosis", diagnosis)

        print()
        print("FINAL DECISION")

        if diagnosis == "MODEL_FOREGROUND_COLLAPSE_CONFIRMED":
            print(
                "CAUTION - training labels appear available, "
                "but the Part 15 model remains foreground-collapsed."
            )
            print(
                "Do NOT interpret the legacy 0.668 Dice as genuine "
                "foreground segmentation performance."
            )

        elif (
            diagnosis
            == "PREPROCESSING_DESTROYS_FOREGROUND_LABELS"
        ):
            print(
                "ACTION REQUIRED - preprocessing is destroying "
                "foreground labels."
            )
            print(
                "Fix preprocessing before any further training."
            )

        elif (
            diagnosis
            == "TRAINING_LABELS_CONTAIN_EMPTY_FOREGROUND"
        ):
            print(
                "ACTION REQUIRED - training labels contain "
                "empty foreground masks."
            )
            print(
                "Inspect pseudo-mask construction before training."
            )

        else:
            print(
                "INVESTIGATION REQUIRED - Part 21 did not "
                "establish a single root cause."
            )

        print()
        print("SPIDER used          : NO")
        print("Test set used        : NO")
        print("Training performed   : NO")
        print("Model weights changed: NO")

        header("PHASE 4 - PART 21 COMPLETE")

    except Exception as exc:

        header("PART 21 ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()