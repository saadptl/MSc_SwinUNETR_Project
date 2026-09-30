"""
==============================================================================
PHASE 4 - PART 25
RSNA-ONLY DICE API / METRIC IMPLEMENTATION AUDIT
==============================================================================

Evaluation / audit only.

NO training.
NO optimizer.
NO checkpoint modification.
NO SPIDER.
NO RSNA test set.

Purpose
-------
Part 24 established a critical contradiction:

    Part 15 recorded Dice ~= 0.668
    Part 24 explicit foreground Dice ~= 0.700
    Part 24 prediction foreground voxels = 0

Therefore Part 25 directly audits the implementation of:

    part11.dice_from_prediction()

The audit compares:
    1. Raw model argmax prediction
    2. Conventional foreground Dice
    3. Per-class Dice
    4. Background Dice
    5. All-class macro Dice
    6. Part 11 dice_from_prediction()
    7. Several possible metric interpretations
    8. Whether the Part 15 recorded Dice can be reproduced

No model weights are changed.
=============================================================================="""

from __future__ import annotations

import ast
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


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

SRC_DIR = PROJECT_ROOT / "src"

PART11_SOURCE = (
    SRC_DIR
    / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

PART15_SOURCE = (
    SRC_DIR
    / "segmentation_rsna_part15_extended_controlled_training.py"
)

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART15_CHECKPOINT = PART15_DIR / "checkpoints" / "best_model.pth"
PART15_HISTORY = PART15_DIR / "part15_training_history.csv"
PART15_VAL_COHORT = PART15_DIR / "part15_validation_cohort.csv"

PART8_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
)

PART8_MANIFEST = (
    PART8_DIR
    / "manifests"
    / "rsna_part8_validation_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part25_dice_api_metric_implementation_audit"
)

REPORT_DIR = OUTPUT_DIR / "reports"


PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12
NUM_CLASSES = 6
VALIDATION_CASES = 10
SEED = 42

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

def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def line(label: str, value: Any) -> None:
    print(f"{label:<38}: {value}")


# =============================================================================
# HASH
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# IMPORT PART 11
# =============================================================================

def import_part11():
    if not PART11_SOURCE.exists():
        raise FileNotFoundError(PART11_SOURCE)

    spec = importlib.util.spec_from_file_location(
        "part11_corrected",
        str(PART11_SOURCE),
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not create Part 11 import specification.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["part11_corrected"] = module
    spec.loader.exec_module(module)

    required = [
        "create_model",
        "preprocess_case",
        "load_tensor_case",
        "select_pilot_rows",
        "dice_from_prediction",
        "load_part9_module",
    ]

    missing = [x for x in required if not hasattr(module, x)]

    if missing:
        raise AttributeError(
            "Part 11 missing required API: " + ", ".join(missing)
        )

    return module


# =============================================================================
# PATH VALIDATION
# =============================================================================

def validate_paths() -> None:
    banner("PATH VALIDATION")

    line("RSNA root", "FOUND" if RSNA_ROOT.exists() else "MISSING")
    line(
        "Part 11 source",
        "FOUND" if PART11_SOURCE.exists() else "MISSING",
    )
    line(
        "Part 15 checkpoint",
        "FOUND" if PART15_CHECKPOINT.exists() else "MISSING",
    )
    line(
        "Part 15 history",
        "FOUND" if PART15_HISTORY.exists() else "MISSING",
    )
    line(
        "Part 15 validation cohort",
        "FOUND" if PART15_VAL_COHORT.exists() else "MISSING",
    )
    line(
        "Part 8 validation manifest",
        "FOUND" if PART8_MANIFEST.exists() else "MISSING",
    )

    required = [
        RSNA_ROOT,
        PART11_SOURCE,
        PART15_CHECKPOINT,
        PART15_HISTORY,
        PART15_VAL_COHORT,
    ]

    missing = [str(x) for x in required if not x.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required Part 25 input(s):\n"
            + "\n".join(missing)
        )


# =============================================================================
# PART 11 SOURCE INSPECTION
# =============================================================================

def inspect_dice_function_source() -> Dict[str, Any]:
    banner("INSPECTING PART 11 DICE IMPLEMENTATION")

    source = PART11_SOURCE.read_text(encoding="utf-8", errors="replace")

    tree = ast.parse(source)

    found = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "dice_from_prediction":
                found.append(node)

    if not found:
        print("dice_from_prediction function not found in AST.")
        return {
            "found": False,
            "source": "",
            "sha256": sha256_file(PART11_SOURCE),
        }

    node = found[0]

    try:
        lines = source.splitlines()
        start = node.lineno
        end = getattr(node, "end_lineno", start)

        snippet = "\n".join(
            f"{i:04d}: {lines[i - 1]}"
            for i in range(start, min(end, start + 100))
        )

    except Exception:
        snippet = ast.unparse(node)

    print("\nActual Part 11 dice_from_prediction():\n")
    print(snippet)

    return {
        "found": True,
        "source": snippet,
        "sha256": sha256_file(PART11_SOURCE),
        "start_line": node.lineno,
        "end_line": getattr(node, "end_lineno", node.lineno),
    }


# =============================================================================
# MODEL LOADING
# =============================================================================

def create_and_load_model(part11, device: torch.device):
    banner("CREATING SWIN-UNETR")

    signature = inspect.signature(part11.create_model)

    print(f"Part 11 create_model signature : {signature}")

    model = part11.create_model(device)

    print(f"Total parameters               : {sum(p.numel() for p in model.parameters())}")

    checkpoint = torch.load(
        PART15_CHECKPOINT,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unexpected checkpoint format.")

    state = checkpoint.get("model_state_dict")

    if state is None:
        raise KeyError("Checkpoint has no model_state_dict.")

    result = model.load_state_dict(state, strict=False)

    print(f"Missing keys                   : {len(result.missing_keys)}")
    print(f"Unexpected keys                : {len(result.unexpected_keys)}")

    model.eval()

    print("✓ Checkpoint loaded successfully.")

    return model, checkpoint


# =============================================================================
# SAFE CASE LOADING
# =============================================================================

def safe_case(part11, part9, row):
    image, mask, info = part11.load_tensor_case(row, part9)

    # Part 11 returns:
    # image = [C,D,H,W] or occasionally [1,C,D,H,W]
    # mask  = [D,H,W] or [1,D,H,W]

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if image.ndim == 5:
        if image.shape[0] != 1:
            raise ValueError(
                f"Unexpected image shape: {tuple(image.shape)}"
            )
        image = image[0]

    if mask.ndim == 4:
        if mask.shape[0] != 1:
            raise ValueError(
                f"Unexpected mask shape: {tuple(mask.shape)}"
            )
        mask = mask[0]

    image = image.float().contiguous()
    mask = mask.long().contiguous()

    return image, mask, info


# =============================================================================
# CONVENTIONAL METRICS
# =============================================================================

def dice_binary(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.bool()
    target = target.bool()

    p = int(pred.sum().item())
    t = int(target.sum().item())

    if p == 0 and t == 0:
        return 1.0

    if p == 0 or t == 0:
        return 0.0

    intersection = int((pred & target).sum().item())

    return (2.0 * intersection) / (p + t)


def per_class_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[int, Dict[str, float]]:

    result = {}

    for c in range(NUM_CLASSES):
        p = pred == c
        t = target == c

        result[c] = {
            "dice": dice_binary(p, t),
            "pred_voxels": float(p.sum().item()),
            "target_voxels": float(t.sum().item()),
            "intersection": float((p & t).sum().item()),
        }

    return result


def foreground_macro_dice(pred, target) -> float:
    values = []

    for c in range(1, NUM_CLASSES):
        values.append(
            dice_binary(pred == c, target == c)
        )

    return float(np.mean(values))


def all_class_macro_dice(pred, target) -> float:
    values = []

    for c in range(NUM_CLASSES):
        values.append(
            dice_binary(pred == c, target == c)
        )

    return float(np.mean(values))


def conventional_foreground_binary_dice(pred, target) -> float:
    return dice_binary(pred > 0, target > 0)


# =============================================================================
# POSSIBLE "EMPTY PREDICTION" DICE INTERPRETATIONS
# =============================================================================

def empty_prediction_analysis(pred, target):
    pred_fg = pred > 0
    target_fg = target > 0

    pred_n = int(pred_fg.sum().item())
    target_n = int(target_fg.sum().item())

    intersection = int((pred_fg & target_fg).sum().item())

    conventional = dice_binary(pred_fg, target_fg)

    # Background included in a two-class macro metric.
    bg_pred = pred == 0
    bg_target = target == 0

    bg_dice = dice_binary(bg_pred, bg_target)

    macro_binary = float(
        np.mean(
            [
                bg_dice,
                conventional,
            ]
        )
    )

    # A common smoothing formulation:
    smooth = 1.0
    smoothed = (
        2.0 * intersection + smooth
    ) / (
        pred_n + target_n + smooth
    )

    # Another common smoothing formulation:
    smoothed_2 = (
        2.0 * intersection + smooth
    ) / (
        pred_n + target_n + 2.0 * smooth
    )

    return {
        "pred_fg": pred_n,
        "target_fg": target_n,
        "intersection_fg": intersection,
        "conventional_fg_dice": conventional,
        "background_dice": bg_dice,
        "binary_macro_dice": macro_binary,
        "smoothed_dice_s1": smoothed,
        "smoothed_dice_s2": smoothed_2,
    }


# =============================================================================
# PART 11 DICE CALL
# =============================================================================

def call_part11_dice(part11, logits, mask):
    """Call the validated Part 11 Dice API exactly as implemented."""
    attempts = []

    try:
        value = part11.dice_from_prediction(
            logits,
            mask,
        )

        attempts.append(
            {
                "attempt": "logits, mask",
                "success": True,
                "type": str(type(value)),
            }
        )

        return value, attempts

    except Exception as e:
        attempts.append(
            {
                "attempt": "logits, mask",
                "success": False,
                "error": repr(e),
            }
        )

    raise RuntimeError(
        "Could not call Part 11 dice_from_prediction().\n"
        + json.dumps(attempts, indent=2)
    )


def normalize_part11_result(value):
    """
    Convert common return formats to:

        mean_dice
        class_dice

    without changing the underlying value.
    """

    mean = None
    classes = {}

    if isinstance(value, tuple):
        if len(value) >= 1:
            mean = value[0]

        if len(value) >= 2 and isinstance(value[1], dict):
            classes = value[1]

    elif isinstance(value, dict):
        for key in [
            "mean_dice",
            "dice",
            "foreground_dice",
            "macro_dice",
        ]:
            if key in value:
                mean = value[key]
                break

        possible_classes = value.get("class_dice")

        if isinstance(possible_classes, dict):
            classes = possible_classes

    elif isinstance(value, (float, int)):
        mean = value

    if torch.is_tensor(mean):
        mean = float(mean.detach().cpu())

    if mean is not None:
        mean = float(mean)

    normalized = {}

    for key, val in classes.items():
        try:
            k = int(key)
        except Exception:
            continue

        if torch.is_tensor(val):
            val = float(val.detach().cpu())

        try:
            normalized[k] = float(val)
        except Exception:
            pass

    return mean, normalized


# =============================================================================
# CASE AUDIT
# =============================================================================

def audit_case(part11, part9, model, row, device):
    image, mask, info = safe_case(part11, part9, row)

    image_b = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_b)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if logits.ndim != 5:
        raise ValueError(
            f"Unexpected model output shape: {tuple(logits.shape)}"
        )

    probs = torch.softmax(logits.float(), dim=1)

    pred = torch.argmax(probs, dim=1)[0].detach().cpu()

    target = mask.detach().cpu()

    explicit = per_class_metrics(pred, target)

    fg = conventional_foreground_binary_dice(pred, target)
    fg_macro = foreground_macro_dice(pred, target)
    all_macro = all_class_macro_dice(pred, target)

    empty = empty_prediction_analysis(pred, target)

    part11_raw, call_attempts = call_part11_dice(
        part11,
        logits,
        mask.unsqueeze(0).to(device),
    )

    part11_mean, part11_classes = normalize_part11_result(
        part11_raw
    )

    logits_cpu = logits[0].detach().float().cpu()

    class_logit_mean = [
        float(logits_cpu[c].mean())
        for c in range(NUM_CLASSES)
    ]

    class_prob_mean = [
        float(probs[0, c].detach().float().cpu().mean())
        for c in range(NUM_CLASSES)
    ]

    pred_counts = {
        str(c): int((pred == c).sum().item())
        for c in range(NUM_CLASSES)
    }

    result = {
        "study_id": str(row.get("study_id", "")),
        "series_id": str(row.get("series_id", "")),
        "part11_mean_dice": part11_mean,

        "explicit_foreground_binary_dice": fg,
        "explicit_foreground_macro_dice": fg_macro,
        "explicit_all_class_macro_dice": all_macro,

        "background_dice": explicit[0]["dice"],

        "pred_foreground_voxels": empty["pred_fg"],
        "target_foreground_voxels": empty["target_fg"],
        "intersection_foreground_voxels": empty["intersection_fg"],

        "binary_macro_dice": empty["binary_macro_dice"],
        "smoothed_dice_s1": empty["smoothed_dice_s1"],
        "smoothed_dice_s2": empty["smoothed_dice_s2"],

        "part11_class_1": part11_classes.get(1),
        "part11_class_2": part11_classes.get(2),
        "part11_class_3": part11_classes.get(3),
        "part11_class_4": part11_classes.get(4),
        "part11_class_5": part11_classes.get(5),

        "pred_labels": json.dumps(pred_counts),
        "class_logit_mean": json.dumps(class_logit_mean),
        "class_probability_mean": json.dumps(class_prob_mean),

        "part11_call_attempts": json.dumps(
            call_attempts
        ),

        "loader": info.get("part11_loader", ""),
    }

    return result


# =============================================================================
# DIAGNOSIS
# =============================================================================

def diagnose(df):
    banner("PART 25 DIAGNOSIS")

    p15 = 0.668

    mean_part11 = df["part11_mean_dice"].dropna().mean()

    mean_fg = df[
        "explicit_foreground_binary_dice"
    ].mean()

    mean_fg_macro = df[
        "explicit_foreground_macro_dice"
    ].mean()

    mean_all = df[
        "explicit_all_class_macro_dice"
    ].mean()

    mean_bg = df[
        "background_dice"
    ].mean()

    empty = int(
        (df["pred_foreground_voxels"] == 0).sum()
    )

    print(f"Part 15 recorded Dice             : {p15:.6f}")

    if not math.isnan(mean_part11):
        print(
            f"Part 11 API mean Dice             : "
            f"{mean_part11:.6f}"
        )
    else:
        print("Part 11 API mean Dice             : NaN")

    print(
        f"Explicit foreground binary Dice   : "
        f"{mean_fg:.6f}"
    )

    print(
        f"Explicit foreground macro Dice    : "
        f"{mean_fg_macro:.6f}"
    )

    print(
        f"Explicit all-class macro Dice     : "
        f"{mean_all:.6f}"
    )

    print(
        f"Explicit background Dice          : "
        f"{mean_bg:.6f}"
    )

    print(
        f"Empty foreground predictions      : "
        f"{empty}/{len(df)}"
    )

    candidates = {
        "Part11_API": mean_part11,
        "ForegroundBinary": mean_fg,
        "ForegroundMacro": mean_fg_macro,
        "AllClassMacro": mean_all,
        "Background": mean_bg,
    }

    print("\nDistance from Part 15 recorded Dice:")

    distances = {}

    for name, value in candidates.items():
        if value is None or (
            isinstance(value, float)
            and math.isnan(value)
        ):
            continue

        distance = abs(float(value) - p15)
        distances[name] = distance

        print(
            f"  {name:<24}: "
            f"{value:.6f} "
            f"(difference={distance:.6f})"
        )

    closest = (
        min(distances, key=distances.get)
        if distances
        else "NONE"
    )

    closest_distance = (
        distances[closest]
        if closest != "NONE"
        else float("inf")
    )

    print(
        f"\nClosest tested interpretation          : "
        f"{closest}"
    )

    print(
        f"Closest absolute difference            : "
        f"{closest_distance:.6f}"
    )

    if empty == len(df) and mean_fg == 0.0:
        print(
            "\nCRITICAL OBSERVATION:"
        )
        print(
            "The conventional foreground Dice is exactly zero "
            "for the audited cases."
        )
        print(
            "Any positive Dice therefore comes from inclusion of "
            "background, class handling, smoothing, or another "
            "metric implementation."
        )

    if not math.isnan(mean_part11):
        if abs(mean_part11 - mean_fg) < 1e-6:
            diagnosis = "PART11_API_MATCHES_FOREGROUND_DICE"
        elif abs(mean_part11 - mean_all) < 1e-6:
            diagnosis = "PART11_API_MATCHES_ALL_CLASS_DICE"
        elif abs(mean_part11 - mean_bg) < 1e-6:
            diagnosis = "PART11_API_MATCHES_BACKGROUND_DICE"
        else:
            diagnosis = "PART11_API_USES_DIFFERENT_IMPLEMENTATION"
    else:
        diagnosis = "PART11_API_COULD_NOT_BE_EVALUATED"

    print(f"\nDiagnosis                              : {diagnosis}")

    return {
        "part15_recorded_dice": p15,
        "part11_api_mean": None
        if math.isnan(mean_part11)
        else float(mean_part11),
        "foreground_binary_mean": float(mean_fg),
        "foreground_macro_mean": float(mean_fg_macro),
        "all_class_macro_mean": float(mean_all),
        "background_mean": float(mean_bg),
        "empty_predictions": empty,
        "closest_interpretation": closest,
        "closest_difference": float(closest_distance),
        "diagnosis": diagnosis,
    }


# =============================================================================
# SAVE
# =============================================================================

def save_results(df, diagnosis, source_info):
    banner("SAVING PART 25 RESULTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    case_path = (
        OUTPUT_DIR
        / "part25_dice_api_metric_case_audit.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "phase4_part25_dice_api_metric_audit_summary.json"
    )

    report_path = (
        REPORT_DIR
        / "phase4_part25_dice_api_metric_audit_report.txt"
    )

    df.to_csv(case_path, index=False)

    summary = {
        "phase": 4,
        "part": 25,
        "purpose": (
            "Audit dice_from_prediction implementation "
            "against explicit mathematically defined Dice metrics."
        ),
        "checkpoint_sha256": sha256_file(PART15_CHECKPOINT),
        "part11_source_sha256": source_info["sha256"],
        "cases": int(len(df)),
        **diagnosis,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report = []

    report.append("PHASE 4 - PART 25")
    report.append(
        "RSNA-ONLY DICE API / METRIC IMPLEMENTATION AUDIT"
    )
    report.append("")
    report.append(
        "No training performed."
    )
    report.append(
        "No model weights modified."
    )
    report.append(
        "SPIDER not used."
    )
    report.append(
        "RSNA test set not used."
    )
    report.append("")
    report.append(
        f"Part 15 recorded Dice: "
        f"{diagnosis['part15_recorded_dice']:.6f}"
    )
    report.append(
        f"Part 11 API mean Dice: "
        f"{diagnosis['part11_api_mean']}"
    )
    report.append(
        f"Foreground binary Dice: "
        f"{diagnosis['foreground_binary_mean']:.6f}"
    )
    report.append(
        f"Foreground macro Dice: "
        f"{diagnosis['foreground_macro_mean']:.6f}"
    )
    report.append(
        f"All-class macro Dice: "
        f"{diagnosis['all_class_macro_mean']:.6f}"
    )
    report.append(
        f"Background Dice: "
        f"{diagnosis['background_mean']:.6f}"
    )
    report.append(
        f"Empty predictions: "
        f"{diagnosis['empty_predictions']}/{diagnosis['cases']}"
    )
    report.append("")
    report.append(
        f"Diagnosis: {diagnosis['diagnosis']}"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"Saved: {case_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        banner("PHASE 4 - PART 25")
        print(
            "RSNA-ONLY DICE API / METRIC IMPLEMENTATION AUDIT"
        )

        print("\nEvaluation / audit only.")
        print("No training is performed.")
        print("No model weights are modified.")
        print("No optimizer is created.")
        print("SPIDER is not used.")
        print("RSNA test set is not used.")

        print(
            "\nPurpose:"
            "\nDetermine exactly why the Part 15 recorded Dice "
            "does not match conventional foreground Dice."
        )

        print("\nPROJECT ROOT")
        print(PROJECT_ROOT)

        print("\nPART 15 CHECKPOINT")
        print(PART15_CHECKPOINT)

        validate_paths()

        banner("PYTORCH / GPU ENVIRONMENT")

        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        line("PyTorch version", torch.__version__)
        line(
            "CUDA available",
            torch.cuda.is_available(),
        )
        line("Device", device)

        if device.type == "cuda":
            line(
                "GPU",
                torch.cuda.get_device_name(0),
            )

        line("Patch size", PATCH_SIZE)
        line("Feature size", FEATURE_SIZE)
        line("Classes", NUM_CLASSES)

        banner("IMPORTING VALIDATED PART 11")

        part11 = import_part11()

        print("✓ Corrected Part 11 imported.")
        print(
            "create_model       :",
            inspect.signature(part11.create_model),
        )
        print(
            "preprocess_case    :",
            inspect.signature(part11.preprocess_case),
        )
        print(
            "load_tensor_case   :",
            inspect.signature(part11.load_tensor_case),
        )
        print(
            "dice_from_prediction:",
            inspect.signature(
                part11.dice_from_prediction
            ),
        )

        source_info = inspect_dice_function_source()

        banner("LOADING PART 15 HISTORY")

        history = pd.read_csv(PART15_HISTORY)

        print(f"History rows : {len(history)}")
        print(
            f"Part 15 best recorded Dice : "
            f"{history['val_dice'].min() if 'val_dice' in history else 'N/A'}"
        )

        if "val_dice" in history.columns:
            print(
                history[
                    [
                        "epoch",
                        "train_loss",
                        "train_dice",
                        "val_loss",
                        "val_dice",
                    ]
                ].to_string(index=False)
            )

        banner("LOADING EXACT PART 15 VALIDATION COHORT")

        rows = pd.read_csv(PART15_VAL_COHORT)

        print(f"Validation cohort rows : {len(rows)}")

        if len(rows) > VALIDATION_CASES:
            rows = rows.iloc[
                :VALIDATION_CASES
            ].copy()

        print(
            f"Part 25 audit cases     : {len(rows)}"
        )

        banner("LOADING PART 15 BEST CHECKPOINT")

        model, checkpoint = create_and_load_model(
            part11,
            device,
        )

        print(
            f"Checkpoint epoch        : "
            f"{checkpoint.get('epoch')}"
        )

        print(
            f"Checkpoint best Dice    : "
            f"{checkpoint.get('best_val_dice')}"
        )

        banner("LOADING PART 9")

        part9 = part11.load_part9_module()

        print("✓ Part 9 loader created through Part 11.")

        banner("RUNNING DICE IMPLEMENTATION AUDIT")

        results = []

        for i, (_, row) in enumerate(
            rows.iterrows(),
            start=1,
        ):

            result = audit_case(
                part11,
                part9,
                model,
                row,
                device,
            )

            results.append(result)

            print(
                f"[{i:03d}/{len(rows):03d}] "
                f"{result['study_id']} | "
                f"{result['series_id']} | "
                f"PredFG="
                f"{result['pred_foreground_voxels']} | "
                f"TargetFG="
                f"{result['target_foreground_voxels']} | "
                f"FGDice="
                f"{result['explicit_foreground_binary_dice']:.6f} | "
                f"FGMacro="
                f"{result['explicit_foreground_macro_dice']:.6f} | "
                f"AllMacro="
                f"{result['explicit_all_class_macro_dice']:.6f} | "
                f"Part11="
                f"{result['part11_mean_dice']}"
            )

        df = pd.DataFrame(results)

        banner("PART 25 AGGREGATED METRICS")

        print(
            f"Validation cases                  : "
            f"{len(df)}"
        )

        print(
            f"Mean Part 11 API Dice             : "
            f"{df['part11_mean_dice'].mean():.6f}"
        )

        print(
            f"Mean foreground binary Dice       : "
            f"{df['explicit_foreground_binary_dice'].mean():.6f}"
        )

        print(
            f"Mean foreground macro Dice        : "
            f"{df['explicit_foreground_macro_dice'].mean():.6f}"
        )

        print(
            f"Mean all-class macro Dice         : "
            f"{df['explicit_all_class_macro_dice'].mean():.6f}"
        )

        print(
            f"Mean background Dice              : "
            f"{df['background_dice'].mean():.6f}"
        )

        print(
            f"Empty foreground predictions      : "
            f"{int((df['pred_foreground_voxels'] == 0).sum())}"
        )

        diagnosis = diagnose(df)

        diagnosis["cases"] = len(df)

        save_results(
            df,
            diagnosis,
            source_info,
        )

        banner("PHASE 4 - PART 25 COMPLETE")

        print("No training performed.")
        print("No model weights modified.")
        print(
            f"Diagnosis: {diagnosis['diagnosis']}"
        )

    except Exception as exc:
        banner("PART 25 ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()