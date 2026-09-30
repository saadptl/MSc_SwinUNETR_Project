"""
PHASE 4 - PART 26

RSNA-ONLY EXACT PART 15 DICE RECONSTRUCTION AUDIT

Purpose
-------
Determine exactly how the Part 15 recorded validation Dice of 0.668
is produced.

This is an AUDIT ONLY.

No training.
No optimizer.
No weight updates.
No checkpoint modification.
No SPIDER.
No RSNA test set.

The audit compares:

1. Part 11 dice_from_prediction()
2. Per-class Dice
3. Empty-class handling
4. Foreground-only Dice
5. All-class Dice
6. Per-case averaging
7. Global voxel aggregation
8. Part 15 recorded validation Dice
9. Part 16 reproduced Dice
10. Mathematical reconstruction of 0.668
"""

from __future__ import annotations

import json
import inspect
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

SRC_ROOT = PROJECT_ROOT / "src"

PART11_PATH = (
    SRC_ROOT
    / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART15_CHECKPOINT = (
    PART15_DIR
    / "checkpoints"
    / "best_model.pth"
)

PART15_HISTORY = (
    PART15_DIR
    / "part15_training_history.csv"
)

PART15_VAL_COHORT = (
    PART15_DIR
    / "part15_validation_cohort.csv"
)

PART16_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part16_best_checkpoint_validation"
)

PART16_CASE_METRICS = (
    PART16_DIR
    / "part16_validation_case_metrics.csv"
)

PART8_VAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
    / "rsna_part8_validation_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part26_exact_part15_dice_reconstruction_audit"
)

REPORT_DIR = OUTPUT_DIR / "reports"


PATCH_SIZE = (64, 96, 96)
NUM_CLASSES = 6
FEATURE_SIZE = 12

SEED = 42


CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ============================================================================
# PRINTING
# ============================================================================

def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def line(label: str, value: Any) -> None:
    print(f"{label:<42}: {value}")


# ============================================================================
# IMPORT PART 11
# ============================================================================

def import_part11():
    import importlib.util

    if not PART11_PATH.exists():
        raise FileNotFoundError(PART11_PATH)

    spec = importlib.util.spec_from_file_location(
        "part11_corrected",
        PART11_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not create Part 11 import specification.")

    module = importlib.util.module_from_spec(spec)

    sys.modules["part11_corrected"] = module

    spec.loader.exec_module(module)

    return module


# ============================================================================
# PATH VALIDATION
# ============================================================================

def validate_paths() -> None:
    required = {
        "RSNA root": RSNA_ROOT,
        "Part 11 source": PART11_PATH,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 history": PART15_HISTORY,
        "Part 15 validation cohort": PART15_VAL_COHORT,
        "Part 16 case metrics": PART16_CASE_METRICS,
        "Part 8 validation manifest": PART8_VAL_MANIFEST,
    }

    missing = []

    for name, path in required.items():
        status = "FOUND" if path.exists() else "MISSING"
        line(name, status)

        if not path.exists():
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            "Missing required Part 26 input(s):\n"
            + "\n".join(missing)
        )


# ============================================================================
# DEVICE
# ============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


# ============================================================================
# CHECKPOINT
# ============================================================================

def load_checkpoint(model: torch.nn.Module) -> Dict[str, Any]:
    checkpoint = torch.load(
        PART15_CHECKPOINT,
        map_location="cpu",
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(f"Missing keys     : {len(missing)}")
    print(f"Unexpected keys  : {len(unexpected)}")

    if missing:
        print("Missing:", missing)

    if unexpected:
        print("Unexpected:", unexpected)

    return checkpoint


# ============================================================================
# TENSOR NORMALIZATION
# ============================================================================

def normalize_image_tensor(
    image: torch.Tensor,
) -> torch.Tensor:

    x = image

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    x = x.float()

    # Expected model input:
    # B,C,D,H,W

    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)

    elif x.ndim == 4:
        # Usually C,D,H,W
        x = x.unsqueeze(0)

    elif x.ndim != 5:
        raise ValueError(
            f"Unexpected image tensor shape: {tuple(x.shape)}"
        )

    return x


def normalize_target_tensor(
    target: torch.Tensor,
) -> torch.Tensor:

    y = target

    if not torch.is_tensor(y):
        y = torch.as_tensor(y)

    y = y.long()

    # Expected:
    # B,D,H,W

    if y.ndim == 3:
        y = y.unsqueeze(0)

    elif y.ndim == 4:
        pass

    elif y.ndim == 5:
        # Possible one-hot representation.
        # Convert only when channel dimension equals NUM_CLASSES.
        if y.shape[1] == NUM_CLASSES:
            y = torch.argmax(y, dim=1)
        else:
            raise ValueError(
                f"Unexpected target shape: {tuple(y.shape)}"
            )

    else:
        raise ValueError(
            f"Unexpected target tensor shape: {tuple(y.shape)}"
        )

    return y


# ============================================================================
# DICE IMPLEMENTATIONS
# ============================================================================

def binary_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:

    prediction = prediction.bool()
    target = target.bool()

    intersection = torch.logical_and(
        prediction,
        target,
    ).sum().float()

    denominator = (
        prediction.sum()
        + target.sum()
    ).float()

    if denominator.item() == 0:
        return 1.0

    return float(
        (
            2.0
            * intersection
            / denominator
        ).item()
    )


def explicit_foreground_binary_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:

    pred_fg = prediction != 0
    true_fg = target != 0

    intersection = torch.logical_and(
        pred_fg,
        true_fg,
    ).sum().float()

    denominator = (
        pred_fg.sum()
        + true_fg.sum()
    ).float()

    if denominator.item() == 0:
        return 1.0

    return float(
        (
            2.0
            * intersection
            / denominator
        ).item()
    )


def per_class_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> List[float]:

    values = []

    for class_id in range(1, NUM_CLASSES):

        pred = prediction == class_id
        true = target == class_id

        intersection = (
            pred & true
        ).sum().float()

        denominator = (
            pred.sum()
            + true.sum()
        ).float()

        if denominator.item() == 0:
            dice = 1.0
        else:
            dice = float(
                (
                    2.0
                    * intersection
                    / denominator
                ).item()
            )

        values.append(dice)

    return values


def part11_style_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[float, List[float]]:

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    values = per_class_dice(
        prediction,
        target,
    )

    return float(np.mean(values)), values


def all_class_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> List[float]:

    values = []

    for class_id in range(NUM_CLASSES):

        pred = prediction == class_id
        true = target == class_id

        intersection = (
            pred & true
        ).sum().float()

        denominator = (
            pred.sum()
            + true.sum()
        ).float()

        if denominator.item() == 0:
            dice = 1.0
        else:
            dice = float(
                (
                    2.0
                    * intersection
                    / denominator
                ).item()
            )

        values.append(dice)

    return values


# ============================================================================
# ROBUST PART 9 RESOLUTION
# ============================================================================

def resolve_part9(part11):
    """Resolve the Part 9 loader required by Part 11."""
    import importlib
    import importlib.util

    # Part 11 globals/attributes first.
    try:
        g = part11.load_tensor_case.__globals__
        for name in ("part9", "loader", "dataset_loader", "rsna_loader"):
            obj = g.get(name)
            if obj is not None and hasattr(obj, "load_case"):
                print(f"✓ Resolved Part 9 through Part 11 global: {name}")
                return obj

        for name, obj in g.items():
            if obj is not None:
                try:
                    if hasattr(obj, "load_case"):
                        print(f"✓ Resolved dataset loader through Part 11 global: {name}")
                        return obj
                except Exception:
                    pass
    except Exception as exc:
        print("WARNING: Part 11 global inspection failed:", repr(exc))

    # Direct project-src imports.
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    for module_name in (
        "segmentation_rsna_part9_dataset_loader",
        "segmentation_rsna_part9_loader",
    ):
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "load_case"):
                print(f"✓ Resolved Part 9 by import: {module_name}")
                return module
        except Exception:
            pass

    # Direct source-file loading.
    for path in (
        SRC_ROOT / "segmentation_rsna_part9_dataset_loader.py",
        SRC_ROOT / "segmentation_rsna_part9_loader.py",
        SRC_ROOT / "segmentation_rsna_part9_dataset_construction.py",
    ):
        if not path.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"_part9_{path.stem}",
                path,
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                if hasattr(module, "load_case"):
                    print(f"✓ Resolved Part 9 directly: {path.name}")
                    return module
        except Exception as exc:
            print(f"WARNING: Could not load {path.name}: {exc!r}")

    # Already loaded modules.
    for module_name, module in list(sys.modules.items()):
        if module is not None and "part9" in str(module_name).lower():
            try:
                if hasattr(module, "load_case"):
                    print(f"✓ Resolved already-loaded Part 9: {module_name}")
                    return module
            except Exception:
                pass

    raise RuntimeError(
        "Could not resolve the Part 9 loader required by Part 11."
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_part15_cohort() -> pd.DataFrame:

    df = pd.read_csv(
        PART15_VAL_COHORT
    )

    return df.reset_index(drop=True)


def load_part16_metrics() -> pd.DataFrame:

    return pd.read_csv(
        PART16_CASE_METRICS
    )


# ============================================================================
# SAFE ROW LOADING
# ============================================================================

def load_case(
    part11,
    part9,
    row: pd.Series,
):

    if part9 is None:
        raise RuntimeError(
            "Part 9 loader is None. "
            "resolve_part9() should never return None."
        )

    if not hasattr(part9, "load_case"):
        raise TypeError(
            "Resolved Part 9 object does not expose load_case(): "
            f"{type(part9)}"
        )

    result = part11.load_tensor_case(
        row,
        part9,
    )

    if not isinstance(result, tuple):
        raise TypeError(
            "Part 11 load_tensor_case() did not return a tuple."
        )

    if len(result) != 3:
        raise ValueError(
            f"Expected 3 values from load_tensor_case(), got {len(result)}."
        )

    image, mask, info = result

    return (
        normalize_image_tensor(image),
        normalize_target_tensor(mask),
        info,
    )


# ============================================================================
# MODEL FORWARD
# ============================================================================

def forward_model(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:

    image = image.to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                output = model(image)

        else:
            output = model(image)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError(
            f"Unexpected model output type: {type(output)}"
        )

    return output


# ============================================================================
# AUDIT CASE
# ============================================================================

def audit_case(
    part11,
    part9,
    model,
    device,
    row: pd.Series,
) -> Dict[str, Any]:

    image, target, info = load_case(
        part11,
        part9,
        row,
    )

    logits = forward_model(
        model,
        image,
        device,
    )

    target = target.to(
        device,
        non_blocking=True,
    )

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    part11_dice, class_dice = part11_style_dice(
        logits,
        target,
    )

    foreground_dice = explicit_foreground_binary_dice(
        prediction,
        target,
    )

    all_dice_values = all_class_dice(
        prediction,
        target,
    )

    all_macro = float(
        np.mean(all_dice_values)
    )

    background_dice = all_dice_values[0]

    foreground_macro = float(
        np.mean(class_dice)
    )

    pred_fg = int(
        (prediction != 0).sum().item()
    )

    target_fg = int(
        (target != 0).sum().item()
    )

    empty_classes = []

    for class_id in range(
        1,
        NUM_CLASSES,
    ):

        pred_count = int(
            (prediction == class_id).sum().item()
        )

        target_count = int(
            (target == class_id).sum().item()
        )

        if pred_count == 0 and target_count == 0:
            empty_classes.append(class_id)

    # Attempt the actual Part 11 function as an independent check.
    actual_api_value = None
    actual_api_classes = None

    try:

        actual_api_value, actual_api_classes = (
            part11.dice_from_prediction(
                logits,
                target,
            )
        )

    except Exception as exc:

        print(
            "WARNING: Part 11 API call failed:",
            repr(exc),
        )

    result = {
        "study_id": str(
            row.get(
                "study_id",
                row.get("studyId", ""),
            )
        ),
        "series_id": str(
            row.get(
                "series_id",
                row.get("seriesId", ""),
            )
        ),
        "part11_manual_dice": float(part11_dice),
        "part11_api_dice": (
            float(actual_api_value)
            if actual_api_value is not None
            else np.nan
        ),
        "foreground_binary_dice": float(
            foreground_dice
        ),
        "foreground_macro_dice": float(
            foreground_macro
        ),
        "all_class_macro_dice": float(
            all_macro
        ),
        "background_dice": float(
            background_dice
        ),
        "pred_fg_voxels": pred_fg,
        "target_fg_voxels": target_fg,
        "empty_foreground_classes": len(
            empty_classes
        ),
        "class_1_dice": float(class_dice[0]),
        "class_2_dice": float(class_dice[1]),
        "class_3_dice": float(class_dice[2]),
        "class_4_dice": float(class_dice[3]),
        "class_5_dice": float(class_dice[4]),
    }

    return result


# ============================================================================
# COLUMN DETECTION
# ============================================================================

def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> str | None:

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in lookup:
            return lookup[key]

    return None


# ============================================================================
# PART 16 COMPARISON
# ============================================================================

def compare_part16(
    part15_df: pd.DataFrame,
    part16_df: pd.DataFrame,
) -> Dict[str, Any]:

    study15 = find_column(
        part15_df,
        [
            "study_id",
            "studyId",
        ],
    )

    series15 = find_column(
        part15_df,
        [
            "series_id",
            "seriesId",
        ],
    )

    study16 = find_column(
        part16_df,
        [
            "study_id",
            "studyId",
        ],
    )

    series16 = find_column(
        part16_df,
        [
            "series_id",
            "seriesId",
        ],
    )

    if (
        study15 is None
        or series15 is None
        or study16 is None
        or series16 is None
    ):
        return {
            "comparison_available": False,
            "reason": "Could not identify study/series columns.",
        }

    a = part15_df[
        [study15, series15]
    ].astype(str)

    b = part16_df[
        [study16, series16]
    ].astype(str)

    keys15 = set(
        zip(
            a.iloc[:, 0],
            a.iloc[:, 1],
        )
    )

    keys16 = set(
        zip(
            b.iloc[:, 0],
            b.iloc[:, 1],
        )
    )

    return {
        "comparison_available": True,
        "part15_cases": len(keys15),
        "part16_cases": len(keys16),
        "intersection": len(
            keys15 & keys16
        ),
        "identical": keys15 == keys16,
    }


# ============================================================================
# MATHEMATICAL RECONSTRUCTION
# ============================================================================

def reconstruct_recorded_dice(
    audit_df: pd.DataFrame,
    recorded: float,
) -> Dict[str, Any]:

    result = {}

    for column in [
        "part11_api_dice",
        "part11_manual_dice",
        "foreground_binary_dice",
        "foreground_macro_dice",
        "all_class_macro_dice",
        "background_dice",
    ]:

        if column not in audit_df:
            continue

        value = float(
            audit_df[column].mean()
        )

        result[column] = {
            "mean": value,
            "difference": abs(
                value - recorded
            ),
        }

    # Exact combinations of the five foreground
    # class scores using simple arithmetic are also tested.

    class_columns = [
        "class_1_dice",
        "class_2_dice",
        "class_3_dice",
        "class_4_dice",
        "class_5_dice",
    ]

    if all(
        c in audit_df.columns
        for c in class_columns
    ):

        class_means = [
            float(audit_df[c].mean())
            for c in class_columns
        ]

        result["class_mean_vector"] = (
            class_means
        )

        result["class_mean_average"] = {
            "value": float(
                np.mean(class_means)
            ),
            "difference": abs(
                float(np.mean(class_means))
                - recorded
            ),
        }

    return result


# ============================================================================
# SAVE
# ============================================================================

def save_results(
    audit_df: pd.DataFrame,
    summary: Dict[str, Any],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_path = (
        OUTPUT_DIR
        / "part26_case_dice_reconstruction_audit.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "phase4_part26_exact_part15_dice_reconstruction_summary.json"
    )

    report_path = (
        REPORT_DIR
        / "phase4_part26_exact_part15_dice_reconstruction_report.txt"
    )

    audit_df.to_csv(
        case_path,
        index=False,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    report_lines = []

    report_lines.append(
        "PHASE 4 - PART 26\n"
    )

    report_lines.append(
        "EXACT PART 15 DICE RECONSTRUCTION AUDIT\n"
    )

    report_lines.append(
        "\nNo training was performed.\n"
        "No model weights were modified.\n"
        "No optimizer was created.\n"
    )

    report_lines.append(
        "\nSUMMARY\n"
    )

    for key, value in summary.items():
        report_lines.append(
            f"{key}: {value}\n"
        )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines(report_lines)

    print("Saved:", case_path)
    print("Saved:", json_path)
    print("Saved:", report_path)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    header(
        "PHASE 4 - PART 26\n"
        "RSNA-ONLY EXACT PART 15 DICE RECONSTRUCTION AUDIT"
    )

    print()
    print("Evaluation / audit only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")

    print()
    print(
        "Purpose:"
    )
    print(
        "Determine exactly how the Part 15 recorded "
        "Dice of 0.668 is produced."
    )

    line("PROJECT ROOT", PROJECT_ROOT)
    line("RSNA DATASET", RSNA_ROOT)
    line("PART 15 CHECKPOINT", PART15_CHECKPOINT)
    line("PART 15 HISTORY", PART15_HISTORY)
    line("PART 15 VALIDATION COHORT", PART15_VAL_COHORT)

    header("PATH VALIDATION")

    validate_paths()

    header("PYTORCH / GPU ENVIRONMENT")

    device = get_device()

    line(
        "PyTorch version",
        torch.__version__,
    )

    line(
        "CUDA available",
        torch.cuda.is_available(),
    )

    line(
        "Device",
        device,
    )

    if torch.cuda.is_available():

        line(
            "GPU",
            torch.cuda.get_device_name(0),
        )

        line(
            "GPU memory",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
        )

    line("Patch size", PATCH_SIZE)
    line("Feature size", FEATURE_SIZE)
    line("Classes", NUM_CLASSES)

    header("IMPORTING VALIDATED PART 11")

    part11 = import_part11()

    print("✓ Corrected Part 11 imported.")

    print(
        "create_model :",
        __import__("inspect").signature(
            part11.create_model
        ),
    )

    print(
        "dice_from_prediction :",
        __import__("inspect").signature(
            part11.dice_from_prediction
        ),
    )

    header("LOADING PART 15 HISTORY")

    history = pd.read_csv(
        PART15_HISTORY
    )

    print(
        "History rows :",
        len(history),
    )

    if "val_dice" in history.columns:

        print(
            history[
                [
                    c
                    for c in [
                        "epoch",
                        "train_loss",
                        "train_dice",
                        "val_loss",
                        "val_dice",
                    ]
                    if c in history.columns
                ]
            ].to_string(
                index=False
            )
        )

        recorded_history_dice = float(
            history["val_dice"].max()
        )

    else:

        recorded_history_dice = 0.668

    print()
    print(
        "Part 15 maximum recorded val Dice:",
        recorded_history_dice,
    )

    header(
        "LOADING EXACT PART 15 VALIDATION COHORT"
    )

    cohort = load_part15_cohort()

    print(
        "Validation cohort rows :",
        len(cohort),
    )

    # Audit all available cases, rather than only 10.
    audit_cohort = cohort.copy()

    print(
        "Part 26 audit cases     :",
        len(audit_cohort),
    )

    header("LOADING PART 15 BEST CHECKPOINT")

    model = part11.create_model(
        device
    )

    model = model.to(device)

    model.eval()

    checkpoint = load_checkpoint(
        model
    )

    print(
        "Checkpoint epoch :",
        checkpoint.get("epoch"),
    )

    checkpoint_best_dice = checkpoint.get("best_dice")
    print(
        "Checkpoint best Dice :",
        checkpoint_best_dice,
    )

    if checkpoint_best_dice is None:
        print(
            "Note: checkpoint contains no 'best_dice' field; "
            "Part 15 history val_dice is used as the recorded metric."
        )

    header("RESOLVING PART 9 THROUGH VALIDATED PART 11")

    part9 = resolve_part9(part11)

    print(
        "✓ Part 9 loader successfully resolved."
    )

    print(
        "Part 9 loader type:",
        type(part9),
    )

    header(
        "RUNNING EXACT DICE RECONSTRUCTION"
    )

    rows = []

    for idx, (_, row) in enumerate(
        audit_cohort.iterrows(),
        start=1,
    ):

        try:

            result = audit_case(
                part11,
                part9,
                model,
                device,
                row,
            )

            rows.append(result)

            print(
                f"[{idx:03d}/{len(audit_cohort):03d}] "
                f"{result['study_id']} | "
                f"{result['series_id']} | "
                f"PredFG={result['pred_fg_voxels']} | "
                f"TargetFG={result['target_fg_voxels']} | "
                f"Part11={result['part11_api_dice']:.6f} | "
                f"FGBinary={result['foreground_binary_dice']:.6f} | "
                f"FGMacro={result['foreground_macro_dice']:.6f} | "
                f"AllMacro={result['all_class_macro_dice']:.6f}"
            )

        except Exception as exc:

            print(
                f"[{idx:03d}/{len(audit_cohort):03d}] ERROR:",
                repr(exc),
            )

    if not rows:

        raise RuntimeError(
            "No audit cases completed successfully."
        )

    audit_df = pd.DataFrame(rows)

    header("PART 26 AGGREGATED RESULTS")

    for column in [
        "part11_api_dice",
        "foreground_binary_dice",
        "foreground_macro_dice",
        "all_class_macro_dice",
        "background_dice",
    ]:

        value = float(
            audit_df[column].mean()
        )

        print(
            f"{column:<34}: {value:.6f}"
        )

    print()

    print(
        "Empty foreground predictions:",
        int(
            (
                audit_df[
                    "pred_fg_voxels"
                ]
                == 0
            ).sum()
        ),
        "/",
        len(audit_df),
    )

    header(
        "PART 16 / PART 15 COHORT COMPARISON"
    )

    part16_df = load_part16_metrics()

    comparison = compare_part16(
        cohort,
        part16_df,
    )

    print(
        json.dumps(
            comparison,
            indent=2,
        )
    )

    header(
        "MATHEMATICAL RECONSTRUCTION OF 0.668"
    )

    reconstruction = reconstruct_recorded_dice(
        audit_df,
        recorded_history_dice,
    )

    for key, value in reconstruction.items():

        if isinstance(value, dict):

            if "mean" in value:

                print(
                    f"{key:<34}: "
                    f"{value['mean']:.6f} "
                    f"(difference={value['difference']:.6f})"
                )

            elif "value" in value:

                print(
                    f"{key:<34}: "
                    f"{value['value']:.6f} "
                    f"(difference={value['difference']:.6f})"
                )

        else:

            print(
                key,
                ":",
                value,
            )

    header("PART 26 DIAGNOSIS")

    mean_part11 = float(
        audit_df[
            "part11_api_dice"
        ].mean()
    )

    mean_foreground = float(
        audit_df[
            "foreground_binary_dice"
        ].mean()
    )

    mean_foreground_macro = float(
        audit_df[
            "foreground_macro_dice"
        ].mean()
    )

    mean_all_macro = float(
        audit_df[
            "all_class_macro_dice"
        ].mean()
    )

    differences = {
        "Part11_API": abs(
            mean_part11
            - recorded_history_dice
        ),
        "ForegroundBinary": abs(
            mean_foreground
            - recorded_history_dice
        ),
        "ForegroundMacro": abs(
            mean_foreground_macro
            - recorded_history_dice
        ),
        "AllClassMacro": abs(
            mean_all_macro
            - recorded_history_dice
        ),
    }

    closest = min(
        differences,
        key=differences.get,
    )

    print(
        f"Part 15 recorded Dice : "
        f"{recorded_history_dice:.6f}"
    )

    print(
        f"Part 11 API mean Dice  : "
        f"{mean_part11:.6f}"
    )

    print(
        f"Foreground binary Dice: "
        f"{mean_foreground:.6f}"
    )

    print(
        f"Foreground macro Dice : "
        f"{mean_foreground_macro:.6f}"
    )

    print(
        f"All-class macro Dice  : "
        f"{mean_all_macro:.6f}"
    )

    print()

    print(
        "Closest tested interpretation:",
        closest,
    )

    print(
        "Closest difference:",
        f"{differences[closest]:.6f}",
    )

    if mean_foreground == 0.0:

        print()
        print(
            "CRITICAL:"
        )

        print(
            "Conventional foreground Dice is zero."
        )

        print(
            "Therefore the recorded positive Dice "
            "cannot be interpreted as conventional "
            "foreground segmentation performance."
        )

    summary = {
        "phase": "4",
        "part": 26,
        "evaluation_only": True,
        "training_performed": False,
        "weights_modified": False,
        "spider_used": False,
        "test_set_used": False,
        "validation_cases": len(audit_df),
        "part15_recorded_dice": recorded_history_dice,
        "mean_part11_api_dice": mean_part11,
        "mean_foreground_binary_dice": mean_foreground,
        "mean_foreground_macro_dice": mean_foreground_macro,
        "mean_all_class_macro_dice": mean_all_macro,
        "closest_interpretation": closest,
        "closest_difference": differences[closest],
        "cohort_comparison": comparison,
        "mathematical_reconstruction": reconstruction,
        "diagnosis": (
            "RECORDED_DICE_IS_NOT_CONVENTIONAL_FOREGROUND_DICE"
            if mean_foreground == 0.0
            else "FURTHER_INVESTIGATION_REQUIRED"
        ),
    }

    header("SAVING PART 26 RESULTS")

    save_results(
        audit_df,
        summary,
    )

    header("PHASE 4 - PART 26 COMPLETE")

    print(
        "No training performed."
    )

    print(
        "No model weights modified."
    )

    print(
        "No checkpoint modified."
    )

    print(
        "Diagnosis:",
        summary["diagnosis"],
    )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "PART 26 INTERRUPTED BY USER."
        )

        raise

    except Exception as exc:

        header("PART 26 ERROR")

        print(
            type(exc).__name__ + ":",
            str(exc),
        )

        raise