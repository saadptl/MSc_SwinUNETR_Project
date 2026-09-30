"""
Part 83 — Historical Metric Reconstruction
==========================================

Purpose
-------
Final forensic audit for the historical Part71 validation Dice value.

Known issue:
- Part71 reported class-balanced E3 foreground Dice ≈ 0.044026.
- Part82 directly evaluated the saved class-balanced E3 checkpoint and did
  not reproduce 0.044026 using global voxel-pooled or mean-case foreground Dice.
- Therefore this part DOES NOT train anything.
- It reads the ORIGINAL Part71 source code, extracts the exact metric/report
  implementation, then evaluates the SAVED class-balanced E3 checkpoint
  through several reconstruction paths, including the original
  dice_from_prediction() implementation when available.

This script is designed to be run from the Windows project root:
C:\\Saad\\Msc Major Project Swin Unetr Framework\\MSc_SwinUNETR_Project

Expected outputs:
- reports/part83_historical_metric_reconstruction_report.txt
- reports/part83_historical_metric_reconstruction_summary.json

IMPORTANT
---------
No new training is performed.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"
REPORT_DIR = PROJECT_ROOT / "reports"

PART71_SOURCE = SRC_DIR / "segmentation_rsna_part71_class_balanced_dicece_training.py"

PART71_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part71_class_balanced_dicece_training"
    / "checkpoints"
    / "part71_r2_full_class_balanced_epoch3.pth"
)

PART71_SUMMARY = REPORT_DIR / "part71_summary.json"
PART71_REPORT = REPORT_DIR / "part71_report.txt"

# We reuse the validated Part9 and Part11 infrastructure where possible.
PART9_SOURCE = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"
PART11_SOURCE = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training.py"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return safe_float(obj)
    return obj


def print_section(title: str, lines: List[str]) -> None:
    lines.append("")
    lines.append("=" * 78)
    lines.append(title)
    lines.append("=" * 78)


# ---------------------------------------------------------------------
# Source inspection
# ---------------------------------------------------------------------

def read_source() -> str:
    if not PART71_SOURCE.exists():
        raise FileNotFoundError(f"Part71 source not found: {PART71_SOURCE}")
    return PART71_SOURCE.read_text(encoding="utf-8", errors="replace")


def source_function_source(source: str, function_name: str) -> Optional[str]:
    try:
        tree = ast.parse(source)
    except Exception:
        return None

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                lines = source.splitlines()
                return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return None


def inspect_metric_code(source: str) -> Dict[str, Any]:
    names = [
        "dice_from_prediction",
        "evaluate",
        "validation",
        "validate",
        "train_one_epoch",
        "create_model",
        "preprocess_case",
    ]

    result: Dict[str, Any] = {
        "functions_found": {},
        "metric_keywords": {},
        "dice_related_lines": [],
        "prediction_related_lines": [],
        "aggregation_related_lines": [],
        "loss_related_lines": [],
    }

    for name in names:
        text = source_function_source(source, name)
        if text is not None:
            result["functions_found"][name] = text

    lines = source.splitlines()

    metric_patterns = [
        r"\bdice\b",
        r"foreground",
        r"per[_ ]class",
        r"classwise",
        r"mean",
        r"sum",
        r"argmax",
        r"softmax",
        r"sigmoid",
        r"threshold",
        r"prediction",
        r"pred",
        r"target",
        r"intersection",
        r"union",
        r"torch\.save",
        r"json",
        r"summary",
        r"report",
    ]

    for i, line in enumerate(lines, start=1):
        lower = line.lower()
        if "dice" in lower:
            result["dice_related_lines"].append({"line": i, "text": line})
        if any(x in lower for x in ["argmax", "softmax", "sigmoid", "threshold", "prediction", "pred ="]):
            result["prediction_related_lines"].append({"line": i, "text": line})
        if any(x in lower for x in ["mean(", ".mean", "sum(", ".sum", "average", "aggregate"]):
            result["aggregation_related_lines"].append({"line": i, "text": line})
        if any(x in lower for x in ["dicece", "loss", "crossentropy", "weight"]):
            result["loss_related_lines"].append({"line": i, "text": line})

    for pat in metric_patterns:
        result["metric_keywords"][pat] = [
            i + 1 for i, line in enumerate(lines) if re.search(pat, line, re.I)
        ]

    return result


def extract_historical_reference(source: str) -> Dict[str, Any]:
    candidates = []
    for m in re.finditer(r"0\.044026(?:\d*)?", source):
        candidates.append({
            "value": m.group(0),
            "start": m.start(),
            "context": source[max(0, m.start() - 350):m.end() + 350],
        })

    report_candidates = []
    for path in [PART71_SUMMARY, PART71_REPORT]:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"0\.044026(?:\d*)?", text):
                report_candidates.append({
                    "file": str(path),
                    "value": m.group(0),
                    "context": text[max(0, m.start() - 250):m.end() + 250],
                })

    return {
        "source_candidates": candidates,
        "report_candidates": report_candidates,
    }


# ---------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------

def import_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------
# Generic Dice reconstructions
# ---------------------------------------------------------------------

def binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    p = int(pred.sum())
    t = int(target.sum())

    if p == 0 and t == 0:
        return 1.0
    if p + t == 0:
        return 1.0

    inter = int(np.logical_and(pred, target).sum())
    return float((2.0 * inter + eps) / (p + t + eps))


def global_foreground_dice(pred: np.ndarray, target: np.ndarray) -> float:
    return binary_dice(pred != 0, target != 0)


def mean_case_foreground_dice(
    preds: List[np.ndarray],
    targets: List[np.ndarray],
) -> float:
    vals = [
        global_foreground_dice(p, t)
        for p, t in zip(preds, targets)
    ]
    return float(np.mean(vals)) if vals else 0.0


def classwise_global_dice(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> Dict[int, float]:
    out = {}
    for c in range(1, num_classes):
        out[c] = binary_dice(pred == c, target == c)
    return out


def macro_foreground_dice(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> float:
    vals = list(classwise_global_dice(pred, target, num_classes).values())
    return float(np.mean(vals)) if vals else 0.0


def weighted_macro_from_support(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> float:
    dices = classwise_global_dice(pred, target, num_classes)
    weights = []
    vals = []
    for c, d in dices.items():
        support = int((target == c).sum())
        if support > 0:
            vals.append(d)
            weights.append(support)
    if not vals:
        return 0.0
    return float(np.average(vals, weights=weights))


def per_case_macro_dice(
    preds: List[np.ndarray],
    targets: List[np.ndarray],
    num_classes: int,
) -> float:
    vals = []
    for p, t in zip(preds, targets):
        cls = classwise_global_dice(p, t, num_classes)
        vals.append(float(np.mean(list(cls.values()))))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------
# Original Part71 metric discovery
# ---------------------------------------------------------------------

def find_callable(module: Any, names: List[str]) -> Optional[Any]:
    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj
    return None


def describe_callable(fn: Any) -> Dict[str, Any]:
    if fn is None:
        return {"found": False}
    try:
        import inspect
        return {
            "found": True,
            "name": getattr(fn, "__name__", str(fn)),
            "signature": str(inspect.signature(fn)),
            "module": getattr(fn, "__module__", None),
        }
    except Exception:
        return {
            "found": True,
            "name": getattr(fn, "__name__", str(fn)),
        }


# ---------------------------------------------------------------------
# Model/checkpoint
# ---------------------------------------------------------------------

def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict):
        return ckpt

    return {"raw_checkpoint": ckpt}


def get_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in [
        "model_state_dict",
        "state_dict",
        "model",
        "network_state_dict",
    ]:
        value = ckpt.get(key)
        if isinstance(value, dict):
            # Some checkpoints store the module itself under model.
            if all(isinstance(v, torch.Tensor) for v in value.values()):
                return value

    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    raise RuntimeError("Could not identify a model state_dict in checkpoint.")


def create_model_from_part71_or_part11(
    part71: Any,
    part11: Any,
    device: torch.device,
):
    """
    Create the model using the validated Part11 API.

    IMPORTANT:
    Part71's create_model() is a wrapper around its imported `part11`
    module and, in the historical source, calling it with a torch.device
    can incorrectly bind that device object to the wrapper's internal
    `part11` parameter. Therefore Part83 deliberately uses Part11's
    validated create_model(device) API directly.
    """
    fn11 = find_callable(part11, ["create_model"])
    if fn11 is not None:
        try:
            return fn11(device)
        except TypeError:
            try:
                return fn11()
            except Exception as exc:
                raise RuntimeError(
                    f"Part11 create_model() failed: {exc}"
                ) from exc

    # Fallback only if Part11 does not expose create_model.
    fn71 = find_callable(part71, ["create_model"])
    if fn71 is not None:
        # Historical Part71 wrapper may use a module/global DEVICE rather
        # than accepting the device as its first positional argument.
        try:
            return fn71()
        except TypeError:
            try:
                return fn71(device=device)
            except Exception as exc:
                raise RuntimeError(
                    f"Part71 create_model() fallback failed: {exc}"
                ) from exc

    raise RuntimeError("Neither Part11 nor Part71 exposes create_model().")


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_cohort_from_part71_or_known_path(part71: Any) -> pd.DataFrame:
    # Prefer an explicit Part71 cohort path exposed by the module.
    candidate_names = [
        "VALIDATION_COHORT_CSV",
        "VAL_COHORT_CSV",
        "VALIDATION_COHORT",
        "VAL_COHORT",
    ]

    for name in candidate_names:
        value = getattr(part71, name, None)
        if value:
            p = Path(value)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return pd.read_csv(p)

    # Known historical Part15/Part71 cohort location.
    candidates = [
        PROJECT_ROOT / "outputs" / "segmentation"
        / "rsna_part15_extended_controlled_training"
        / "part15_validation_cohort.csv",
        PROJECT_ROOT / "outputs" / "segmentation"
        / "rsna_part15_extended_controlled_training"
        / "part15_train_cohort.csv",
    ]

    for p in candidates:
        if p.exists():
            return pd.read_csv(p)

    # Search for likely validation cohort files without guessing contents.
    found = list(
        (PROJECT_ROOT / "outputs" / "segmentation").rglob("*validation*cohort*.csv")
    )
    if found:
        return pd.read_csv(found[0])

    raise FileNotFoundError("Could not locate the historical validation cohort CSV.")


def resolve_part71_validation_rows(df: pd.DataFrame, max_cases: int = 50) -> pd.DataFrame:
    # Historical Part71 first-50 validation evaluation used the validation cohort.
    # Preserve CSV order rather than sorting by any new criterion.
    if len(df) > max_cases:
        return df.iloc[:max_cases].copy()
    return df.copy()


# ---------------------------------------------------------------------
# Prediction conversion audit
# ---------------------------------------------------------------------

def prediction_variants(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Return several plausible historical prediction conversions.

    logits: [1,C,D,H,W]
    """
    probs = torch.softmax(logits, dim=1)
    out = {
        "argmax_logits": torch.argmax(logits, dim=1),
        "argmax_softmax": torch.argmax(probs, dim=1),
    }

    # A foreground threshold on max foreground probability, followed by
    # foreground class selection. This is only an audit variant.
    fg_probs = probs[:, 1:]
    fg_values, fg_idx = torch.max(fg_probs, dim=1)
    threshold_pred = fg_idx + 1
    threshold_pred = torch.where(
        fg_values >= 0.5,
        threshold_pred,
        torch.zeros_like(threshold_pred),
    )
    out["foreground_threshold_0.50"] = threshold_pred

    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        pred = torch.where(
            fg_values >= threshold,
            threshold_pred,
            torch.zeros_like(threshold_pred),
        )
        out[f"foreground_threshold_{threshold:.2f}"] = pred

    return out


# ---------------------------------------------------------------------
# Exact Part11/Part9 loading helpers
# ---------------------------------------------------------------------

def prepare_case(
    row: pd.Series,
    part71: Any,
    part11: Any,
    part9: Any,
):
    """
    Prefer Part71's loader if it exists. Otherwise use validated Part11 loader.
    """
    candidates = [
        "load_tensor_case",
        "load_case",
        "load_case_robust",
    ]

    fn = find_callable(part71, candidates)
    if fn is not None:
        attempts = [
            lambda: fn(row, part9),
            lambda: fn(row),
        ]
        for attempt in attempts:
            try:
                result = attempt()
                if isinstance(result, tuple) and len(result) >= 2:
                    return result[0], result[1]
            except Exception:
                pass

    fn11 = find_callable(part11, ["load_tensor_case", "load_case_robust"])
    if fn11 is None:
        raise RuntimeError("No usable case loader found.")

    attempts = [
        lambda: fn11(row, part9),
        lambda: fn11(row),
    ]

    last_exc = None
    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, tuple) and len(result) >= 2:
                return result[0], result[1]
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(f"Case loading failed: {last_exc}")


def normalize_image_mask(
    image: Any,
    mask: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = torch.as_tensor(image)
    mask = torch.as_tensor(mask)

    # image expected [1,D,H,W] or [D,H,W]
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    elif image.ndim == 5 and image.shape[0] == 1 and image.shape[1] == 1:
        image = image[0, 0]

    # mask expected [D,H,W], sometimes [1,D,H,W]
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 5 and mask.shape[0] == 1 and mask.shape[1] == 1:
        mask = mask[0, 0]

    return image.float(), mask.long()


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report: List[str] = []
    summary: Dict[str, Any] = {
        "part": "83",
        "title": "Historical Metric Reconstruction",
        "training_performed": False,
        "status": None,
    }

    print_section("PART 83 — HISTORICAL METRIC RECONSTRUCTION", report)
    report.append("Training is DISABLED. This is a forensic audit only.")

    # -------------------------------------------------------------
    # File fingerprints
    # -------------------------------------------------------------
    print_section("1. FILE FINGERPRINTS", report)

    for label, path in [
        ("Part71 source", PART71_SOURCE),
        ("Part71 class-balanced E3 checkpoint", PART71_CHECKPOINT),
        ("Part71 summary", PART71_SUMMARY),
        ("Part71 report", PART71_REPORT),
    ]:
        report.append(f"{label}: {path}")
        report.append(f"  exists={path.exists()}")
        if path.exists():
            report.append(f"  size={path.stat().st_size}")
            report.append(f"  sha256={sha256_file(path)}")

    summary["files"] = {
        "part71_source": {
            "path": str(PART71_SOURCE),
            "exists": PART71_SOURCE.exists(),
            "sha256": sha256_file(PART71_SOURCE),
        },
        "class_balanced_epoch3": {
            "path": str(PART71_CHECKPOINT),
            "exists": PART71_CHECKPOINT.exists(),
            "sha256": sha256_file(PART71_CHECKPOINT),
        },
    }

    # -------------------------------------------------------------
    # Source and report references
    # -------------------------------------------------------------
    source = read_source()
    source_info = inspect_metric_code(source)
    refs = extract_historical_reference(source)

    print_section("2. HISTORICAL REFERENCE DISCOVERY", report)
    report.append(f"Source literal 0.044026 occurrences: {len(refs['source_candidates'])}")
    report.append(f"Report/summary 0.044026 occurrences: {len(refs['report_candidates'])}")

    for item in refs["report_candidates"]:
        report.append(f"  {item['file']}: {item['value']}")
        report.append("  context:")
        report.extend("    " + x for x in item["context"].splitlines()[:8])

    summary["historical_reference"] = refs

    # -------------------------------------------------------------
    # Exact Part71 source function extraction
    # -------------------------------------------------------------
    print_section("3. ORIGINAL PART71 METRIC-RELATED CODE", report)

    metric_function_names = [
        "dice_from_prediction",
        "evaluate",
        "validate",
        "validation",
        "train_one_epoch",
    ]

    for name in metric_function_names:
        text = source_info["functions_found"].get(name)
        if text:
            report.append(f"\n--- FUNCTION: {name} ---")
            report.extend(text.splitlines())

    report.append("\n--- DICE-RELATED SOURCE LINES ---")
    for item in source_info["dice_related_lines"]:
        report.append(f"{item['line']:5d}: {item['text']}")

    report.append("\n--- PREDICTION-RELATED SOURCE LINES ---")
    for item in source_info["prediction_related_lines"]:
        report.append(f"{item['line']:5d}: {item['text']}")

    report.append("\n--- AGGREGATION-RELATED SOURCE LINES ---")
    for item in source_info["aggregation_related_lines"]:
        report.append(f"{item['line']:5d}: {item['text']}")

    summary["source_metric_inspection"] = source_info

    # -------------------------------------------------------------
    # Import original modules
    # -------------------------------------------------------------
    print_section("4. DYNAMIC MODULE IMPORT", report)

    part11 = None
    part9 = None
    part71 = None

    try:
        part9 = import_module_from_path("part83_part9", PART9_SOURCE)
        report.append("Part9 import: OK")
    except Exception as exc:
        report.append(f"Part9 import: FAILED: {exc}")

    try:
        part11 = import_module_from_path("part83_part11", PART11_SOURCE)
        report.append("Part11 import: OK")
    except Exception as exc:
        report.append(f"Part11 import: FAILED: {exc}")

    try:
        part71 = import_module_from_path("part83_part71", PART71_SOURCE)
        report.append("Part71 import: OK")
    except Exception as exc:
        report.append(f"Part71 import: FAILED: {exc}")

    if part71 is not None:
        for name in [
            "dice_from_prediction",
            "evaluate",
            "validate",
            "validation",
            "create_model",
            "load_tensor_case",
        ]:
            fn = find_callable(part71, [name])
            report.append(f"Part71 callable {name}: {describe_callable(fn)}")

    summary["imports"] = {
        "part9": part9 is not None,
        "part11": part11 is not None,
        "part71": part71 is not None,
    }

    if part71 is None or part11 is None or part9 is None:
        summary["status"] = "CANNOT_COMPLETE_ORIGINAL_METRIC_RECONSTRUCTION_MODULE_IMPORT_FAILED"
        report.append(f"\nFINAL STATUS: {summary['status']}")
        (REPORT_DIR / "part83_historical_metric_reconstruction_report.txt").write_text(
            "\n".join(report), encoding="utf-8"
        )
        (REPORT_DIR / "part83_historical_metric_reconstruction_summary.json").write_text(
            json.dumps(json_safe(summary), indent=2), encoding="utf-8"
        )
        print("\n".join(report))
        return

    # -------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------
    print_section("5. CLASS-BALANCED EPOCH3 CHECKPOINT", report)

    ckpt = load_checkpoint(PART71_CHECKPOINT)
    state_dict = get_state_dict(ckpt)

    report.append(f"Checkpoint top-level keys: {list(ckpt.keys())}")
    report.append(f"State dict tensors: {len(state_dict)}")

    for key in ["epoch", "seed", "best_val_dice", "val_dice", "feature_size",
                "num_classes", "train_cases", "validation_cases",
                "source_checkpoint"]:
        if key in ckpt:
            report.append(f"metadata[{key}] = {ckpt[key]!r}")

    summary["checkpoint_metadata"] = {
        k: json_safe(ckpt.get(k))
        for k in [
            "epoch",
            "seed",
            "best_val_dice",
            "val_dice",
            "feature_size",
            "num_classes",
            "train_cases",
            "validation_cases",
            "source_checkpoint",
        ]
        if k in ckpt
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report.append(f"Device: {device}")

    model = create_model_from_part71_or_part11(part71, part11, device)

    try:
        load_result = model.load_state_dict(state_dict, strict=True)
        report.append(f"Strict checkpoint load: OK: {load_result}")
    except Exception as exc:
        report.append(f"Strict checkpoint load FAILED: {exc}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        report.append(f"Non-strict missing: {missing}")
        report.append(f"Non-strict unexpected: {unexpected}")
        summary["status"] = "CHECKPOINT_MODEL_ARCHITECTURE_MISMATCH"
        report.append(f"\nFINAL STATUS: {summary['status']}")
        (REPORT_DIR / "part83_historical_metric_reconstruction_report.txt").write_text(
            "\n".join(report), encoding="utf-8"
        )
        (REPORT_DIR / "part83_historical_metric_reconstruction_summary.json").write_text(
            json.dumps(json_safe(summary), indent=2), encoding="utf-8"
        )
        print("\n".join(report))
        return

    model.to(device)
    model.eval()

    # -------------------------------------------------------------
    # Validation cohort
    # -------------------------------------------------------------
    print_section("6. VALIDATION COHORT", report)

    val_df = load_cohort_from_part71_or_known_path(part71)
    val_df = resolve_part71_validation_rows(val_df, 50)

    report.append(f"Validation rows evaluated: {len(val_df)}")
    report.append(f"Validation columns: {list(val_df.columns)}")

    summary["validation_rows"] = len(val_df)

    # -------------------------------------------------------------
    # Evaluate checkpoint
    # -------------------------------------------------------------
    print_section("7. CHECKPOINT PREDICTION RECONSTRUCTION", report)

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    case_records: List[Dict[str, Any]] = []

    original_metric_fn = find_callable(part71, ["dice_from_prediction"])

    with torch.no_grad():
        for idx, (_, row) in enumerate(val_df.iterrows(), start=1):
            try:
                image, target = prepare_case(row, part71, part11, part9)
                image, target = normalize_image_mask(image, target)

                # Expected project convention: centered 32x64x64 crop.
                # If Part71 itself exposes preprocess_case, use it first.
                preprocess = find_callable(part71, ["preprocess_case"])

                if preprocess is not None:
                    try:
                        processed = preprocess(image, target)
                        if isinstance(processed, tuple) and len(processed) >= 2:
                            image, target = normalize_image_mask(
                                processed[0], processed[1]
                            )
                    except Exception:
                        # Do not silently stop; retain already loaded tensors.
                        pass

                x = image.unsqueeze(0).unsqueeze(0).to(device)
                y = target.unsqueeze(0).to(device)

                logits = model(x)

                if logits.shape[-3:] != y.shape[-3:]:
                    # Center-crop model output to target only if dimensions differ.
                    d = min(logits.shape[-3], y.shape[-3])
                    h = min(logits.shape[-2], y.shape[-2])
                    w = min(logits.shape[-1], y.shape[-1])

                    logits = logits[..., :d, :h, :w]
                    y = y[..., :d, :h, :w]

                variants = prediction_variants(logits)

                # Primary historical-style prediction = argmax.
                pred = variants["argmax_logits"]

                pred_np = pred[0].detach().cpu().numpy().astype(np.int64)
                target_np = y[0].detach().cpu().numpy().astype(np.int64)

                all_preds.append(pred_np)
                all_targets.append(target_np)

                record: Dict[str, Any] = {
                    "case_index": idx,
                    "image_shape": list(image.shape),
                    "target_shape": list(target_np.shape),
                    "target_fg": int((target_np != 0).sum()),
                    "pred_fg": int((pred_np != 0).sum()),
                    "global_fg_dice": global_foreground_dice(pred_np, target_np),
                    "macro_fg_dice": macro_foreground_dice(
                        pred_np, target_np, logits.shape[1]
                    ),
                }

                # Re-run exact Part71 metric if a callable is exposed.
                if original_metric_fn is not None:
                    exact_attempts = [
                        lambda: original_metric_fn(logits, y),
                        lambda: original_metric_fn(
                            logits.detach().cpu(),
                            y.detach().cpu(),
                        ),
                        lambda: original_metric_fn(
                            logits,
                            y,
                            logits.shape[1],
                        ),
                    ]

                    exact_value = None
                    exact_raw_repr = None

                    for attempt in exact_attempts:
                        try:
                            raw = attempt()
                            exact_raw_repr = repr(raw)
                            if isinstance(raw, tuple):
                                # Search tuple recursively for numeric Dice values.
                                nums = []
                                for v in raw:
                                    if isinstance(v, (float, int, np.floating, np.integer)):
                                        nums.append(float(v))
                                    elif isinstance(v, (list, tuple, np.ndarray)):
                                        for z in np.asarray(v).reshape(-1):
                                            if np.isscalar(z):
                                                try:
                                                    nums.append(float(z))
                                                except Exception:
                                                    pass
                                if nums:
                                    exact_value = float(nums[0])
                            elif isinstance(raw, (float, int, np.floating, np.integer)):
                                exact_value = float(raw)
                            elif isinstance(raw, dict):
                                # Search common names.
                                for key in [
                                    "dice",
                                    "fg_dice",
                                    "foreground_dice",
                                    "val_dice",
                                    "mean_dice",
                                ]:
                                    if key in raw:
                                        exact_value = safe_float(raw[key])
                                        if exact_value is not None:
                                            break
                            break
                        except Exception:
                            continue

                    record["original_metric_raw"] = exact_raw_repr
                    record["original_metric_value_first_numeric"] = exact_value

                case_records.append(record)

            except Exception as exc:
                case_records.append({
                    "case_index": idx,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=2),
                })

    successful = [r for r in case_records if "error" not in r]

    report.append(f"Successful cases: {len(successful)}/{len(val_df)}")

    if not successful:
        summary["status"] = "NO_SUCCESSFUL_VALIDATION_CASES"
        report.append(f"FINAL STATUS: {summary['status']}")
        (REPORT_DIR / "part83_historical_metric_reconstruction_report.txt").write_text(
            "\n".join(report), encoding="utf-8"
        )
        (REPORT_DIR / "part83_historical_metric_reconstruction_summary.json").write_text(
            json.dumps(json_safe(summary), indent=2), encoding="utf-8"
        )
        print("\n".join(report))
        return

    # -------------------------------------------------------------
    # Aggregations
    # -------------------------------------------------------------
    print_section("8. METRIC RECONSTRUCTION RESULTS", report)

    preds = all_preds
    targets = all_targets

    # All voxels pooled.
    pooled_pred = np.concatenate([p.reshape(-1) for p in preds])
    pooled_target = np.concatenate([t.reshape(-1) for t in targets])

    results: Dict[str, Any] = {
        "historical_reference": 0.044026262773,
        "global_voxel_pooled_foreground_dice": global_foreground_dice(
            pooled_pred, pooled_target
        ),
        "mean_case_foreground_dice": mean_case_foreground_dice(preds, targets),
        "global_macro_foreground_dice": macro_foreground_dice(
            pooled_pred, pooled_target, 6
        ),
        "global_support_weighted_macro_foreground_dice": weighted_macro_from_support(
            pooled_pred, pooled_target, 6
        ),
        "mean_case_macro_foreground_dice": per_case_macro_dice(
            preds, targets, 6
        ),
        "target_fg_voxels": int((pooled_target != 0).sum()),
        "pred_fg_voxels": int((pooled_pred != 0).sum()),
    }

    # Mean of any exact Part71 metric values successfully returned.
    exact_values = [
        r.get("original_metric_value_first_numeric")
        for r in successful
        if r.get("original_metric_value_first_numeric") is not None
    ]
    if exact_values:
        results["mean_original_part71_metric_first_numeric"] = float(
            np.mean(exact_values)
        )
        results["median_original_part71_metric_first_numeric"] = float(
            np.median(exact_values)
        )

    for key, value in results.items():
        report.append(f"{key}: {value}")

    # -------------------------------------------------------------
    # Threshold audit — only to see whether 0.044026 appears under a
    # plausible prediction conversion. This is NOT a replacement metric.
    # -------------------------------------------------------------
    print_section("9. PREDICTION-CONVERSION AUDIT", report)

    threshold_results = {}

    # Recompute threshold variants from stored model would require retaining
    # logits. Instead use the exact primary predictions and report the
    # historical threshold variants only when the source explicitly contains
    # a threshold. The purpose is source tracing, not parameter searching.
    threshold_lines = [
        x["text"]
        for x in source_info["prediction_related_lines"]
        if "threshold" in x["text"].lower()
    ]

    threshold_results["source_contains_threshold_logic"] = bool(threshold_lines)
    threshold_results["source_threshold_lines"] = threshold_lines

    report.append(
        f"Part71 source explicitly contains threshold prediction logic: "
        f"{bool(threshold_lines)}"
    )
    for line in threshold_lines:
        report.append(f"  {line}")

    # -------------------------------------------------------------
    # Compare with reference
    # -------------------------------------------------------------
    print_section("10. HISTORICAL REFERENCE COMPARISON", report)

    reference = 0.044026262773
    comparison = {}

    for key, value in results.items():
        if isinstance(value, (int, float)) and key not in [
            "historical_reference",
            "target_fg_voxels",
            "pred_fg_voxels",
        ]:
            diff = abs(float(value) - reference)
            comparison[key] = {
                "value": float(value),
                "absolute_difference": diff,
                "matches_within_1e-6": diff <= 1e-6,
                "matches_within_1e-4": diff <= 1e-4,
                "matches_within_1e-3": diff <= 1e-3,
            }
            report.append(
                f"{key}: value={float(value):.12f}, "
                f"abs_diff={diff:.12f}"
            )

    summary["results"] = results
    summary["comparison"] = comparison
    summary["case_records"] = case_records

    # -------------------------------------------------------------
    # Final diagnosis
    # -------------------------------------------------------------
    exact_match = any(
        item["matches_within_1e-6"]
        for item in comparison.values()
        if isinstance(item, dict)
    )

    close_match = any(
        item["matches_within_1e-3"]
        for item in comparison.values()
        if isinstance(item, dict)
    )

    if exact_match:
        status = "HISTORICAL_0_044026_RECONSTRUCTED"
    elif close_match:
        status = "HISTORICAL_0_044026_APPROXIMATELY_RECONSTRUCTED"
    else:
        status = "HISTORICAL_0_044026_NOT_RECONSTRUCTED_FROM_SAVED_CHECKPOINT"

    summary["status"] = status

    print_section("11. FINAL DIAGNOSIS", report)
    report.append(status)

    if status == "HISTORICAL_0_044026_RECONSTRUCTED":
        report.append(
            "The historical value can be reproduced from the saved "
            "class-balanced epoch3 checkpoint using the reconstructed metric path."
        )
        report.append(
            "Do NOT retrain. The next step is to lock the exact metric definition "
            "and proceed with the final model pipeline."
        )
    elif status == "HISTORICAL_0_044026_APPROXIMATELY_RECONSTRUCTED":
        report.append(
            "A metric path is close to the historical reference but does not "
            "exactly reproduce it. Preserve the exact reconstructed implementation "
            "and document the numerical discrepancy."
        )
    else:
        report.append(
            "No tested metric reconstruction reproduces the historical 0.044026 "
            "from the saved class-balanced epoch3 checkpoint."
        )
        report.append(
            "This means the historical report value cannot currently be treated "
            "as a reproducible checkpoint-level validation metric."
        )
        report.append(
            "Do NOT start another training ablation. The correct next action is "
            "to inspect the historical Part71 report-generation path and preserve "
            "the discrepancy in the project report if the original value cannot "
            "be recovered."
        )

    # -------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------
    report_path = REPORT_DIR / "part83_historical_metric_reconstruction_report.txt"
    summary_path = REPORT_DIR / "part83_historical_metric_reconstruction_summary.json"

    report_path.write_text("\n".join(report), encoding="utf-8")
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )

    print("\n".join(report))
    print("\nSaved:")
    print(report_path)
    print(summary_path)


if __name__ == "__main__":
    main()
