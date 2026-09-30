"""
==============================================================================
PHASE 4 - PART 18
RSNA-ONLY EXPANDED VALIDATION / TRAINING STABILITY ANALYSIS
==============================================================================

Purpose
-------
Evaluate the verified Part 15 best checkpoint on the controlled 100-case
validation cohort and produce a deeper stability / error analysis.

This stage DOES NOT:
    - train the model
    - update model weights
    - use SPIDER
    - use the RSNA test set
    - modify the Part 15 checkpoint

It evaluates:
    1. Overall Dice
    2. Per-class Dice
    3. Precision
    4. Recall / sensitivity
    5. Prediction foreground statistics
    6. Empty-prediction cases
    7. Per-case stability
    8. Class-wise failure patterns
    9. Comparison with Part 15 / Part 16
   10. GPU memory usage
   11. Reproducibility of the verified checkpoint

IMPORTANT
---------
All segmentation metrics are measured against RSNA point-derived
pseudo-masks, NOT manually delineated clinical segmentation masks.

Therefore these results are experimental segmentation validation metrics,
not clinical diagnostic performance.
==============================================================================

"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import traceback
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from monai.losses import DiceCELoss
    from monai.networks.nets import SwinUNETR
except Exception as exc:
    raise RuntimeError(
        "MONAI is required for Part 18."
    ) from exc


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

PART11_SOURCE = (
    SRC_DIR
    / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
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

PART16_SUMMARY = (
    PART16_DIR
    / "phase4_part16_best_checkpoint_validation_summary.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part18_expanded_validation_stability_analysis"
)

REPORT_DIR = OUTPUT_DIR / "reports"


# =============================================================================
# MODEL CONFIGURATION — VALIDATED IN PART 10 / PART 15
# =============================================================================

PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12
NUM_CLASSES = 6
BATCH_SIZE = 1
AMP_ENABLED = True

SEED = 42

EXPECTED_VALIDATION_CASES = 100

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

FOREGROUND_CLASSES = list(range(1, NUM_CLASSES))


# =============================================================================
# PRINT HELPERS
# =============================================================================

def section(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def status(label: str, value):
    print(f"{label:<38}: {value}")


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed: int = SEED):

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
# PATH VALIDATION
# =============================================================================

def require_paths():

    section("PATH VALIDATION")

    required = {
        "RSNA root": RSNA_ROOT,
        "train manifest": TRAIN_MANIFEST,
        "validation manifest": VAL_MANIFEST,
        "Part 11 corrected source": PART11_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 history": PART15_HISTORY,
        "Part 16 case metrics": PART16_CASE_METRICS,
        "Part 16 summary": PART16_SUMMARY,
    }

    missing = []

    for name, path in required.items():

        exists = path.exists()

        status(
            name,
            "FOUND" if exists else "MISSING"
        )

        if not exists:
            missing.append(path)

    if missing:

        raise FileNotFoundError(
            "Missing required Part 18 input(s):\n"
            + "\n".join(str(p) for p in missing)
        )


# =============================================================================
# ENVIRONMENT
# =============================================================================

def print_environment():

    section("PYTORCH / GPU ENVIRONMENT")

    status("PyTorch version", torch.__version__)
    status("CUDA available", torch.cuda.is_available())

    if torch.cuda.is_available():

        status(
            "Device",
            torch.cuda.get_device_name(0)
        )

        props = torch.cuda.get_device_properties(0)

        status(
            "GPU memory",
            f"{props.total_memory / (1024 ** 3):.2f} GB"
        )

    else:

        status(
            "Device",
            "CPU"
        )

    status("Patch size", PATCH_SIZE)
    status("Feature size", FEATURE_SIZE)
    status("Classes", NUM_CLASSES)
    status("Batch size", BATCH_SIZE)
    status("AMP", AMP_ENABLED)


# =============================================================================
# IMPORT PART 11
# =============================================================================

def import_part11():

    section("IMPORTING VALIDATED PART 11 IMPLEMENTATION")

    spec = spec_from_file_location(
        "segmentation_rsna_part11_corrected_part18",
        PART11_SOURCE,
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            "Could not create import specification for Part 11."
        )

    module = module_from_spec(spec)

    sys.modules[
        "segmentation_rsna_part11_corrected_part18"
    ] = module

    spec.loader.exec_module(module)

    required_api = [
        "select_pilot_rows",
        "load_part9_module",
        "load_tensor_case",
        "preprocess_case",
        "dice_from_prediction",
        "create_model",
        "DiceCELoss",
    ]

    for name in required_api:

        if not hasattr(module, name):

            raise AttributeError(
                f"Part 11 does not expose required API: {name}"
            )

    print("✓ Corrected Part 11 imported.")
    print("✓ Part 11 preprocessing retained.")
    print("✓ Part 11 Dice API retained.")
    print("✓ Part 11 model API retained.")

    return module


# =============================================================================
# LOAD PART 9
# =============================================================================

def load_part9(part11):

    section("LOADING PART 9 THROUGH CORRECTED PART 11")

    part9 = part11.load_part9_module()

    print("✓ Part 9 loader imported.")

    return part9


# =============================================================================
# LOAD CHECKPOINT
# =============================================================================

def load_checkpoint():

    checkpoint = torch.load(
        PART15_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):

        raise RuntimeError(
            "Unexpected Part 15 checkpoint format."
        )

    return checkpoint


# =============================================================================
# CREATE MODEL
# =============================================================================

def create_model():

    model = SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        use_checkpoint=False,
    )

    return model


# =============================================================================
# LOAD MODEL WEIGHTS
# =============================================================================

def load_model(checkpoint, device):

    section("CREATING SWIN-UNETR")

    model = create_model()

    print("✓ Swin-UNETR created.")

    print(
        "Total parameters : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:

        state_dict = checkpoint.get(
            "state_dict"
        )

    if state_dict is None:

        raise KeyError(
            "Could not find model_state_dict/state_dict "
            "inside Part 15 checkpoint."
        )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"Missing keys     : {len(missing)}"
    )

    print(
        f"Unexpected keys  : {len(unexpected)}"
    )

    if missing or unexpected:

        raise RuntimeError(
            "Part 15 checkpoint is not fully compatible "
            "with the validated model architecture."
        )

    model.to(device)
    model.eval()

    return model


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_image(part11, image, mask):

    """
    Use the exact Part 11 preprocessing function.

    The function signature can differ slightly between corrected versions,
    therefore this wrapper handles the validated common forms.
    """

    try:

        result = part11.preprocess_case(
            image,
            mask,
        )

    except TypeError:

        result = part11.preprocess_case(
            image
        )

    if isinstance(result, tuple):

        processed_image = result[0]

        if len(result) > 1:
            processed_mask = result[1]
        else:
            processed_mask = mask

    else:

        processed_image = result
        processed_mask = mask

    return processed_image, processed_mask


# =============================================================================
# TENSOR CONTRACT
# =============================================================================

def ensure_image_tensor(image):

    if not isinstance(image, torch.Tensor):

        image = torch.as_tensor(
            image,
            dtype=torch.float32,
        )

    image = image.float()

    # Expected [1,D,H,W]
    if image.ndim == 3:
        image = image.unsqueeze(0)

    # Expected [B,1,D,H,W]
    if image.ndim == 4:
        image = image.unsqueeze(0)

    if image.ndim != 5:

        raise RuntimeError(
            f"Unexpected image tensor shape: {tuple(image.shape)}"
        )

    return image


def ensure_mask_tensor(mask):

    if not isinstance(mask, torch.Tensor):

        mask = torch.as_tensor(
            mask,
            dtype=torch.long,
        )

    mask = mask.long()

    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    if mask.ndim != 3:

        raise RuntimeError(
            f"Unexpected mask tensor shape: {tuple(mask.shape)}"
        )

    return mask


# =============================================================================
# PATCH / RESIZE TO VALIDATED TRAINING SIZE
# =============================================================================

def resize_volume_and_mask(image, mask):

    """
    Resize every validation case to the Part 10 / Part 15 validated
    Swin-UNETR input contract.

    image : [B,1,D,H,W]
    mask  : [D,H,W]
    """

    target_d, target_h, target_w = PATCH_SIZE

    image = image.float()

    image = torch.nn.functional.interpolate(
        image,
        size=PATCH_SIZE,
        mode="trilinear",
        align_corners=False,
    )

    mask_tensor = mask.unsqueeze(0).unsqueeze(0).float()

    mask_tensor = torch.nn.functional.interpolate(
        mask_tensor,
        size=PATCH_SIZE,
        mode="nearest",
    )

    mask = mask_tensor.squeeze(0).squeeze(0).long()

    return image, mask


# =============================================================================
# METRIC HELPERS
# =============================================================================

def class_confusion(pred, target, class_id):

    pred_class = pred == class_id
    target_class = target == class_id

    tp = int(
        torch.logical_and(
            pred_class,
            target_class,
        ).sum().item()
    )

    fp = int(
        torch.logical_and(
            pred_class,
            torch.logical_not(target_class),
        ).sum().item()
    )

    fn = int(
        torch.logical_and(
            torch.logical_not(pred_class),
            target_class,
        ).sum().item()
    )

    tn = int(
        torch.logical_and(
            torch.logical_not(pred_class),
            torch.logical_not(target_class),
        ).sum().item()
    )

    return tp, fp, fn, tn


def dice_from_counts(tp, fp, fn):

    denominator = (
        2 * tp + fp + fn
    )

    if denominator == 0:

        return 1.0

    return (
        2.0 * tp
        / denominator
    )


def precision_from_counts(tp, fp):

    denominator = tp + fp

    if denominator == 0:

        return 0.0

    return tp / denominator


def recall_from_counts(tp, fn):

    denominator = tp + fn

    if denominator == 0:

        return 0.0

    return tp / denominator


def compute_case_metrics(pred, target):

    class_metrics = {}

    foreground_dices = []

    foreground_precisions = []

    foreground_recalls = []

    total_foreground_pred = int(
        (pred > 0).sum().item()
    )

    total_foreground_target = int(
        (target > 0).sum().item()
    )

    for class_id in FOREGROUND_CLASSES:

        tp, fp, fn, tn = class_confusion(
            pred,
            target,
            class_id,
        )

        dice = dice_from_counts(
            tp,
            fp,
            fn,
        )

        precision = precision_from_counts(
            tp,
            fp,
        )

        recall = recall_from_counts(
            tp,
            fn,
        )

        class_metrics[class_id] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "target_voxels": tp + fn,
            "prediction_voxels": tp + fp,
        }

        foreground_dices.append(dice)
        foreground_precisions.append(precision)
        foreground_recalls.append(recall)

    mean_dice = float(
        np.mean(foreground_dices)
    )

    mean_precision = float(
        np.mean(foreground_precisions)
    )

    mean_recall = float(
        np.mean(foreground_recalls)
    )

    return {
        "mean_foreground_dice": mean_dice,
        "mean_foreground_precision": mean_precision,
        "mean_foreground_recall": mean_recall,
        "foreground_prediction_voxels": total_foreground_pred,
        "foreground_target_voxels": total_foreground_target,
        "class_metrics": class_metrics,
    }


# =============================================================================
# CASE ID
# =============================================================================

def get_case_ids(row):

    study = row.get("study_id")
    series = row.get("series_id")

    if pd.isna(study):
        study = ""

    if pd.isna(series):
        series = ""

    try:
        study = str(int(float(study)))
    except Exception:
        study = str(study)

    try:
        series = str(int(float(series)))
    except Exception:
        series = str(series)

    return study, series


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_case(
    row,
    index,
    total,
    part11,
    part9,
    model,
    device,
    loss_fn,
):

    study_id, series_id = get_case_ids(row)

    image, mask, info = part11.load_tensor_case(
        row,
        part9,
    )

    # IMPORTANT: Part 11 preprocess_case expects the loader's native
    # image dimensionality (typically [C,D,H,W]). Do NOT convert the
    # image to [B,C,D,H,W] before calling preprocess_case, otherwise
    # Part 11's internal resize_3d receives five dimensions and fails.

    # -------------------------------------------------------------
    # Apply exact Part 11 preprocessing BEFORE adding batch dimension
    # -------------------------------------------------------------

    try:

        processed = part11.preprocess_case(
            image,
            mask,
        )

        if isinstance(processed, tuple):

            image = processed[0]

            if len(processed) > 1:
                mask = processed[1]

    except TypeError:

        # Some corrected Part 11 versions perform preprocessing
        # inside load_tensor_case.
        pass

    image = ensure_image_tensor(image)
    mask = ensure_mask_tensor(mask)

    # -------------------------------------------------------------
    # Ensure validated spatial contract
    # -------------------------------------------------------------

    image, mask = resize_volume_and_mask(
        image,
        mask,
    )

    image = image.to(
        device,
        non_blocking=True,
    )

    mask = mask.to(
        device,
        non_blocking=True,
    )

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------

    with torch.no_grad():

        if device.type == "cuda" and AMP_ENABLED:

            with torch.amp.autocast(
                device_type="cuda",
                enabled=True,
            ):

                logits = model(image)

                loss = loss_fn(
                    logits,
                    mask.unsqueeze(0),
                )

        else:

            logits = model(image)

            loss = loss_fn(
                logits,
                mask.unsqueeze(0),
            )

    prediction = torch.argmax(
        logits,
        dim=1,
    ).squeeze(0)

    metrics = compute_case_metrics(
        prediction,
        mask,
    )

    result = {
        "index": index,
        "study_id": study_id,
        "series_id": series_id,
        "loss": float(loss.item()),
        "dice": metrics[
            "mean_foreground_dice"
        ],
        "precision": metrics[
            "mean_foreground_precision"
        ],
        "recall": metrics[
            "mean_foreground_recall"
        ],
        "foreground_prediction_voxels": metrics[
            "foreground_prediction_voxels"
        ],
        "foreground_target_voxels": metrics[
            "foreground_target_voxels"
        ],
        "prediction_empty": (
            metrics["foreground_prediction_voxels"] == 0
        ),
        "target_empty": (
            metrics["foreground_target_voxels"] == 0
        ),
    }

    for class_id in FOREGROUND_CLASSES:

        cm = metrics[
            "class_metrics"
        ][class_id]

        prefix = f"class_{class_id}"

        result[f"{prefix}_dice"] = cm["dice"]
        result[f"{prefix}_precision"] = cm["precision"]
        result[f"{prefix}_recall"] = cm["recall"]
        result[f"{prefix}_tp"] = cm["tp"]
        result[f"{prefix}_fp"] = cm["fp"]
        result[f"{prefix}_fn"] = cm["fn"]
        result[f"{prefix}_target_voxels"] = cm[
            "target_voxels"
        ]
        result[f"{prefix}_prediction_voxels"] = cm[
            "prediction_voxels"
        ]

    if index <= 5 or index % 25 == 0 or index == total:

        print(
            f"  [{index:03d}/{total}] "
            f"{study_id} | {series_id} | "
            f"Loss={result['loss']:.4f} "
            f"Dice={result['dice']:.6f} "
            f"Precision={result['precision']:.6f} "
            f"Recall={result['recall']:.6f}"
        )

    return result


# =============================================================================
# PER-CLASS SUMMARY
# =============================================================================

def create_class_summary(case_df):

    rows = []

    for class_id in FOREGROUND_CLASSES:

        name = CLASS_NAMES[class_id]

        dice_col = f"class_{class_id}_dice"
        precision_col = f"class_{class_id}_precision"
        recall_col = f"class_{class_id}_recall"
        target_col = f"class_{class_id}_target_voxels"
        pred_col = f"class_{class_id}_prediction_voxels"

        dice_values = pd.to_numeric(
            case_df[dice_col],
            errors="coerce",
        )

        precision_values = pd.to_numeric(
            case_df[precision_col],
            errors="coerce",
        )

        recall_values = pd.to_numeric(
            case_df[recall_col],
            errors="coerce",
        )

        target_values = pd.to_numeric(
            case_df[target_col],
            errors="coerce",
        )

        pred_values = pd.to_numeric(
            case_df[pred_col],
            errors="coerce",
        )

        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "mean_dice": float(
                    dice_values.mean()
                ),
                "median_dice": float(
                    dice_values.median()
                ),
                "std_dice": float(
                    dice_values.std(
                        ddof=0
                    )
                ),
                "mean_precision": float(
                    precision_values.mean()
                ),
                "mean_recall": float(
                    recall_values.mean()
                ),
                "mean_target_voxels": float(
                    target_values.mean()
                ),
                "mean_prediction_voxels": float(
                    pred_values.mean()
                ),
                "zero_dice_cases": int(
                    (dice_values <= 1e-8).sum()
                ),
                "low_dice_cases": int(
                    (dice_values < 0.5).sum()
                ),
                "high_dice_cases": int(
                    (dice_values >= 0.8).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# STABILITY SUMMARY
# =============================================================================

def create_stability_summary(case_df):

    dice = pd.to_numeric(
        case_df["dice"],
        errors="coerce",
    )

    precision = pd.to_numeric(
        case_df["precision"],
        errors="coerce",
    )

    recall = pd.to_numeric(
        case_df["recall"],
        errors="coerce",
    )

    loss = pd.to_numeric(
        case_df["loss"],
        errors="coerce",
    )

    return {
        "cases": int(len(case_df)),
        "mean_loss": float(loss.mean()),
        "median_loss": float(loss.median()),
        "std_loss": float(loss.std(ddof=0)),
        "mean_dice": float(dice.mean()),
        "median_dice": float(dice.median()),
        "std_dice": float(dice.std(ddof=0)),
        "min_dice": float(dice.min()),
        "max_dice": float(dice.max()),
        "mean_precision": float(precision.mean()),
        "median_precision": float(precision.median()),
        "mean_recall": float(recall.mean()),
        "median_recall": float(recall.median()),
        "zero_dice_cases": int(
            (dice <= 1e-8).sum()
        ),
        "low_dice_cases": int(
            (dice < 0.5).sum()
        ),
        "medium_dice_cases": int(
            ((dice >= 0.5) & (dice < 0.8)).sum()
        ),
        "high_dice_cases": int(
            (dice >= 0.8).sum()
        ),
        "prediction_empty_cases": int(
            case_df["prediction_empty"].sum()
        ),
        "target_empty_cases": int(
            case_df["target_empty"].sum()
        ),
    }


# =============================================================================
# COMPARE WITH PART 16
# =============================================================================

def compare_part16(case_df):

    section("PART 16 ↔ PART 18 REPRODUCIBILITY CHECK")

    part16_df = pd.read_csv(
        PART16_CASE_METRICS
    )

    print(
        f"Part 16 rows : {len(part16_df)}"
    )

    print(
        f"Part 18 rows : {len(case_df)}"
    )

    # -------------------------------------------------------------
    # Find common identifiers
    # -------------------------------------------------------------

    id_candidates = [
        ("study_id", "series_id"),
        ("study", "series"),
    ]

    common_pair = None

    for a, b in id_candidates:

        if (
            a in case_df.columns
            and b in case_df.columns
            and a in part16_df.columns
            and b in part16_df.columns
        ):

            common_pair = (a, b)
            break

    if common_pair is None:

        print(
            "Could not identify common study/series columns."
        )

        return {
            "common_cases": 0,
            "mean_abs_difference": None,
            "max_abs_difference": None,
            "exact_match": False,
        }

    a, b = common_pair

    part18 = case_df.copy()
    part16 = part16_df.copy()

    part18[a] = part18[a].astype(str)
    part18[b] = part18[b].astype(str)

    part16[a] = part16[a].astype(str)
    part16[b] = part16[b].astype(str)

    # -------------------------------------------------------------
    # Find Dice column
    # -------------------------------------------------------------

    def find_dice_col(df):

        for col in [
            "dice",
            "mean_foreground_dice",
            "validation_dice",
        ]:

            if col in df.columns:
                return col

        for col in df.columns:

            if "dice" in col.lower():

                return col

        return None

    p18_dice = find_dice_col(part18)
    p16_dice = find_dice_col(part16)

    if p18_dice is None or p16_dice is None:

        print(
            "Could not identify Dice columns."
        )

        return {
            "common_cases": 0,
            "mean_abs_difference": None,
            "max_abs_difference": None,
            "exact_match": False,
        }

    common = part18.merge(
        part16[
            [a, b, p16_dice]
        ],
        on=[a, b],
        how="inner",
        suffixes=(
            "_part18",
            "_part16",
        ),
    )

    if len(common) == 0:

        print(
            "No common validation cases found."
        )

        return {
            "common_cases": 0,
            "mean_abs_difference": None,
            "max_abs_difference": None,
            "exact_match": False,
        }

    differences = (
        pd.to_numeric(
            common[f"{p18_dice}_part18"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            common[p16_dice],
            errors="coerce",
        )
    ).abs()

    mean_difference = float(
        differences.mean()
    )

    max_difference = float(
        differences.max()
    )

    exact = (
        mean_difference < 1e-6
        and max_difference < 1e-6
    )

    print()
    print(
        f"Common cases                  : {len(common)}"
    )

    print(
        f"Mean absolute Dice difference : "
        f"{mean_difference:.10f}"
    )

    print(
        f"Max absolute Dice difference  : "
        f"{max_difference:.10f}"
    )

    print(
        f"Exact reproduction            : "
        f"{'YES' if exact else 'NO'}"
    )

    return {
        "common_cases": int(len(common)),
        "mean_abs_difference": mean_difference,
        "max_abs_difference": max_difference,
        "exact_match": exact,
    }


# =============================================================================
# DECISION
# =============================================================================

def make_decision(
    stability,
    class_summary,
    reproduction,
):

    mean_dice = stability["mean_dice"]

    zero_cases = stability[
        "zero_dice_cases"
    ]

    empty_predictions = stability[
        "prediction_empty_cases"
    ]

    reproducible = reproduction[
        "exact_match"
    ]

    # -------------------------------------------------------------
    # Reproduction is mandatory
    # -------------------------------------------------------------

    if not reproducible:

        return (
            "CAUTION - Part 18 does not exactly reproduce "
            "the Part 16 case-level Dice results. Investigate "
            "preprocessing, resizing, metric or cohort consistency "
            "before using this stage for final performance reporting."
        )

    # -------------------------------------------------------------
    # Structural sanity
    # -------------------------------------------------------------

    if empty_predictions > 0:

        return (
            "CAUTION - checkpoint reproduction is consistent, "
            "but empty foreground predictions were detected in "
            f"{empty_predictions} validation case(s)."
        )

    if zero_cases > stability["cases"] * 0.25:

        return (
            "CAUTION - more than 25% of validation cases have "
            "zero foreground Dice. Continue with detailed failure "
            "analysis before scaling training."
        )

    if mean_dice >= 0.60:

        return (
            "PASS - expanded validation is stable and reproduces "
            "the verified Part 15/Part 16 checkpoint behaviour. "
            "The checkpoint is suitable for the next controlled "
            "experimental stage."
        )

    return (
        "CAUTION - validation reproduction is internally "
        "consistent, but the observed Dice level remains low. "
        "Perform additional failure analysis before scaling."
    )


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(
    case_df,
    class_df,
    stability,
    reproduction,
    decision,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Case metrics
    # -------------------------------------------------------------

    case_path = (
        OUTPUT_DIR
        / "part18_expanded_validation_case_metrics.csv"
    )

    case_df.to_csv(
        case_path,
        index=False,
    )

    print(f"Saved: {case_path}")

    # -------------------------------------------------------------
    # Class metrics
    # -------------------------------------------------------------

    class_path = (
        OUTPUT_DIR
        / "part18_expanded_validation_per_class_metrics.csv"
    )

    class_df.to_csv(
        class_path,
        index=False,
    )

    print(f"Saved: {class_path}")

    # -------------------------------------------------------------
    # Stability
    # -------------------------------------------------------------

    stability_path = (
        OUTPUT_DIR
        / "part18_validation_stability_summary.csv"
    )

    pd.DataFrame(
        [stability]
    ).to_csv(
        stability_path,
        index=False,
    )

    print(f"Saved: {stability_path}")

    # -------------------------------------------------------------
    # Reproduction
    # -------------------------------------------------------------

    reproduction_path = (
        OUTPUT_DIR
        / "part18_part16_reproduction_comparison.csv"
    )

    pd.DataFrame(
        [reproduction]
    ).to_csv(
        reproduction_path,
        index=False,
    )

    print(f"Saved: {reproduction_path}")

    # -------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------

    summary = {
        "phase": 4,
        "part": 18,
        "purpose": (
            "RSNA-only expanded validation and "
            "training stability analysis"
        ),
        "decision": decision,
        "patch_size": list(PATCH_SIZE),
        "feature_size": FEATURE_SIZE,
        "classes": NUM_CLASSES,
        "validation_cases": EXPECTED_VALIDATION_CASES,
        "seed": SEED,
        "spider_used": False,
        "test_set_used": False,
        "training_performed": False,
        "model_weights_changed": False,
        "stability": stability,
        "reproduction": reproduction,
        "per_class": class_df.to_dict(
            orient="records"
        ),
        "important_note": (
            "Dice, precision and recall are calculated against "
            "RSNA point-derived pseudo-masks, not manually "
            "delineated clinical segmentation ground truth."
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "phase4_part18_expanded_validation_stability_summary.json"
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
        )

    print(f"Saved: {json_path}")

    # -------------------------------------------------------------
    # Text report
    # -------------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "phase4_part18_expanded_validation_stability_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 4 - PART 18\n"
            "RSNA-ONLY EXPANDED VALIDATION / "
            "TRAINING STABILITY ANALYSIS\n"
        )

        f.write("=" * 78 + "\n\n")

        f.write(
            "FINAL DECISION\n"
            f"{decision}\n\n"
        )

        f.write(
            "OVERALL VALIDATION\n"
            + "-" * 78
            + "\n"
        )

        for key, value in stability.items():

            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            "\nPER-CLASS RESULTS\n"
            + "-" * 78
            + "\n"
        )

        for _, row in class_df.iterrows():

            f.write(
                f"{int(row['class_id'])}: "
                f"{row['class_name']}\n"
            )

            f.write(
                f"  Dice       : "
                f"{row['mean_dice']:.6f}\n"
            )

            f.write(
                f"  Precision  : "
                f"{row['mean_precision']:.6f}\n"
            )

            f.write(
                f"  Recall     : "
                f"{row['mean_recall']:.6f}\n"
            )

            f.write(
                f"  Zero Dice  : "
                f"{int(row['zero_dice_cases'])}\n"
            )

        f.write(
            "\nPART 16 REPRODUCTION\n"
            + "-" * 78
            + "\n"
        )

        for key, value in reproduction.items():

            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            "\nIMPORTANT\n"
            + "-" * 78
            + "\n"
        )

        f.write(
            "All segmentation metrics are calculated against "
            "RSNA point-derived pseudo-masks. They are not "
            "manual clinical segmentation ground-truth metrics.\n"
        )

        f.write(
            "\nSPIDER used: NO\n"
            "Test set used: NO\n"
            "Training performed: NO\n"
            "Model weights changed: NO\n"
        )

    print(f"Saved: {report_path}")

    return json_path, report_path


# =============================================================================
# MAIN
# =============================================================================

def main():

    section(
        "PHASE 4 - PART 18\n"
        "RSNA-ONLY EXPANDED VALIDATION / TRAINING STABILITY ANALYSIS"
    )

    print()
    print(
        "Evaluation only.\n"
        "No training is performed.\n"
        "No model weights are modified.\n"
        "SPIDER is not used.\n"
        "RSNA test set is not used."
    )

    print()
    print("PROJECT ROOT")
    print(PROJECT_ROOT)

    print()
    print("RSNA DATASET")
    print(RSNA_ROOT)

    print()
    print("PART 15 BEST CHECKPOINT")
    print(PART15_CHECKPOINT)

    print()
    print("PART 16 REFERENCE")
    print(PART16_DIR)

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    # -------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------

    set_seed()

    require_paths()

    print_environment()

    # -------------------------------------------------------------
    # Load Part 11
    # -------------------------------------------------------------

    part11 = import_part11()

    part9 = load_part9(
        part11
    )

    # -------------------------------------------------------------
    # Load manifests
    # -------------------------------------------------------------

    section("LOADING VALIDATION MANIFEST")

    val_df = pd.read_csv(
        VAL_MANIFEST
    )

    print(
        f"Validation manifest total : {len(val_df)}"
    )

    # -------------------------------------------------------------
    # Reconstruct exact Part 15 validation cohort
    # -------------------------------------------------------------

    section(
        "RECONSTRUCTING PART 15 VALIDATION COHORT"
    )

    # The validated Part 11 select_pilot_rows() API expects the manifest
    # path, not an already-loaded DataFrame. Use the exact manifest path
    # so the Part 15/16 deterministic cohort is reconstructed correctly.
    val_cohort = part11.select_pilot_rows(
        str(VAL_MANIFEST),
        EXPECTED_VALIDATION_CASES,
        SEED,
    )

    val_cohort = val_cohort.reset_index(
        drop=True
    )

    print(
        f"Part 18 validation cohort : "
        f"{len(val_cohort)}"
    )

    if len(val_cohort) != EXPECTED_VALIDATION_CASES:

        raise RuntimeError(
            "Part 18 validation cohort size does not "
            "match expected Part 15/Part 16 cohort."
        )

    # -------------------------------------------------------------
    # Check duplicates
    # -------------------------------------------------------------

    if (
        "study_id" in val_cohort.columns
        and "series_id" in val_cohort.columns
    ):

        duplicate_count = int(
            val_cohort.duplicated(
                subset=[
                    "study_id",
                    "series_id",
                ]
            ).sum()
        )

        print(
            f"Duplicate study/series rows : "
            f"{duplicate_count}"
        )

        if duplicate_count > 0:

            raise RuntimeError(
                "Duplicate study/series cases detected "
                "in validation cohort."
            )

    # -------------------------------------------------------------
    # Check Part 16 cohort
    # -------------------------------------------------------------

    if PART16_CASE_METRICS.exists():

        part16_cases = pd.read_csv(
            PART16_CASE_METRICS
        )

        print(
            f"Part 16 evaluated cases      : "
            f"{len(part16_cases)}"
        )

        if len(part16_cases) != len(val_cohort):

            raise RuntimeError(
                "Part 16 and Part 18 validation cohort sizes differ."
            )

        # Prevent a same-size but different cohort from being compared.
        if (
            "study_id" in val_cohort.columns
            and "series_id" in val_cohort.columns
            and "study_id" in part16_cases.columns
            and "series_id" in part16_cases.columns
        ):
            p18_ids = set(zip(
                val_cohort["study_id"].astype(str),
                val_cohort["series_id"].astype(str),
            ))
            p16_ids = set(zip(
                part16_cases["study_id"].astype(str),
                part16_cases["series_id"].astype(str),
            ))

            if p18_ids != p16_ids:
                raise RuntimeError(
                    "Part 16 and Part 18 validation cohorts are not identical. "
                    f"Missing from Part 18: {len(p16_ids - p18_ids)}; "
                    f"Extra in Part 18: {len(p18_ids - p16_ids)}."
                )

            print("✓ Part 16 and Part 18 validation case sets are identical.")

    # -------------------------------------------------------------
    # Check checkpoint
    # -------------------------------------------------------------

    section(
        "LOADING PART 15 BEST CHECKPOINT"
    )

    checkpoint = load_checkpoint()

    checkpoint_epoch = checkpoint.get(
        "epoch"
    )

    checkpoint_best_dice = checkpoint.get(
        "best_val_dice"
    )

    print(
        f"Checkpoint epoch       : "
        f"{checkpoint_epoch}"
    )

    print(
        f"Checkpoint best Dice   : "
        f"{checkpoint_best_dice}"
    )

    print(
        f"Checkpoint seed        : "
        f"{checkpoint.get('seed')}"
    )

    print(
        f"Checkpoint patch size  : "
        f"{checkpoint.get('patch_size')}"
    )

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = load_model(
        checkpoint,
        device,
    )

    # -------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------

    section("LOSS / EVALUATION CONFIGURATION")

    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    print("Loss         : DiceCELoss")
    print("Mode         : evaluation only")
    print("Gradient     : disabled")
    print("Weight update: disabled")

    # -------------------------------------------------------------
    # GPU reset
    # -------------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # -------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------

    section(
        "STARTING EXPANDED 100-CASE VALIDATION"
    )

    print(
        "RSNA only."
    )

    print(
        "SPIDER is not used."
    )

    print(
        "Test set is not used."
    )

    print(
        "No training is performed."
    )

    case_results = []

    total = len(val_cohort)

    for idx, (_, row) in enumerate(
        val_cohort.iterrows(),
        start=1,
    ):

        result = evaluate_case(
            row=row,
            index=idx,
            total=total,
            part11=part11,
            part9=part9,
            model=model,
            device=device,
            loss_fn=loss_fn,
        )

        case_results.append(
            result
        )

    case_df = pd.DataFrame(
        case_results
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    section(
        "PART 18 EXPANDED VALIDATION SUMMARY"
    )

    stability = create_stability_summary(
        case_df
    )

    print(
        f"Validation cases       : "
        f"{stability['cases']}"
    )

    print(
        f"Mean validation loss   : "
        f"{stability['mean_loss']:.6f}"
    )

    print(
        f"Mean validation Dice   : "
        f"{stability['mean_dice']:.6f}"
    )

    print(
        f"Median validation Dice : "
        f"{stability['median_dice']:.6f}"
    )

    print(
        f"Std validation Dice    : "
        f"{stability['std_dice']:.6f}"
    )

    print(
        f"Min validation Dice    : "
        f"{stability['min_dice']:.6f}"
    )

    print(
        f"Max validation Dice    : "
        f"{stability['max_dice']:.6f}"
    )

    print(
        f"Mean precision         : "
        f"{stability['mean_precision']:.6f}"
    )

    print(
        f"Mean recall            : "
        f"{stability['mean_recall']:.6f}"
    )

    print(
        f"Zero-Dice cases        : "
        f"{stability['zero_dice_cases']}"
    )

    print(
        f"Low-Dice cases (<0.5)  : "
        f"{stability['low_dice_cases']}"
    )

    print(
        f"High-Dice cases (≥0.8) : "
        f"{stability['high_dice_cases']}"
    )

    print(
        f"Empty predictions      : "
        f"{stability['prediction_empty_cases']}"
    )

    # -------------------------------------------------------------
    # Class summary
    # -------------------------------------------------------------

    section("PER-CLASS VALIDATION ANALYSIS")

    class_df = create_class_summary(
        case_df
    )

    for _, row in class_df.iterrows():

        print(
            f"{int(row['class_id'])}: "
            f"{row['class_name']}"
        )

        print(
            f"  Dice       : "
            f"{row['mean_dice']:.6f}"
        )

        print(
            f"  Precision  : "
            f"{row['mean_precision']:.6f}"
        )

        print(
            f"  Recall     : "
            f"{row['mean_recall']:.6f}"
        )

        print(
            f"  Zero Dice  : "
            f"{int(row['zero_dice_cases'])}"
        )

        print(
            f"  Low Dice   : "
            f"{int(row['low_dice_cases'])}"
        )

    # -------------------------------------------------------------
    # Reproduction
    # -------------------------------------------------------------

    reproduction = compare_part16(
        case_df
    )

    # -------------------------------------------------------------
    # GPU
    # -------------------------------------------------------------

    peak_allocated = None
    peak_reserved = None

    if device.type == "cuda":

        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )

    # -------------------------------------------------------------
    # Decision
    # -------------------------------------------------------------

    decision = make_decision(
        stability,
        class_df,
        reproduction,
    )

    section("PART 18 DECISION")

    print(decision)

    print()
    print(
        f"Peak GPU allocated : "
        f"{peak_allocated:.3f} GB"
        if peak_allocated is not None
        else "Peak GPU allocated : CPU"
    )

    print(
        f"Peak GPU reserved  : "
        f"{peak_reserved:.3f} GB"
        if peak_reserved is not None
        else "Peak GPU reserved  : CPU"
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    section("SAVING PART 18 RESULTS")

    save_outputs(
        case_df=case_df,
        class_df=class_df,
        stability=stability,
        reproduction=reproduction,
        decision=decision,
    )

    section("PART 18 FINAL SUMMARY")

    print(
        f"Validation cases       : "
        f"{stability['cases']}"
    )

    print(
        f"Mean validation Dice   : "
        f"{stability['mean_dice']:.6f}"
    )

    print(
        f"Median validation Dice : "
        f"{stability['median_dice']:.6f}"
    )

    print(
        f"Std validation Dice    : "
        f"{stability['std_dice']:.6f}"
    )

    print(
        f"Mean precision         : "
        f"{stability['mean_precision']:.6f}"
    )

    print(
        f"Mean recall            : "
        f"{stability['mean_recall']:.6f}"
    )

    print(
        f"Part 16 exact match    : "
        f"{'YES' if reproduction['exact_match'] else 'NO'}"
    )

    print()
    print(
        "SPIDER used            : NO"
    )

    print(
        "Test set used          : NO"
    )

    print(
        "Training performed     : NO"
    )

    print(
        "Model weights changed  : NO"
    )

    print()
    print("FINAL DECISION")
    print(decision)

    print()
    print(
        "IMPORTANT:\n"
        "Dice, precision and recall are calculated against "
        "RSNA point-derived pseudo-masks, not manually "
        "delineated clinical segmentation ground truth."
    )

    section(
        "PHASE 4 - PART 18 COMPLETE"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print()
        print("=" * 78)
        print("PART 18 ERROR")
        print("=" * 78)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        sys.exit(1)