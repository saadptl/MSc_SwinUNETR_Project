"""
==============================================================================
PHASE 4 - PART 38
RSNA-ONLY EPOCH-BY-EPOCH CHECKPOINT PREDICTION FORENSIC AUDIT
==============================================================================

Purpose
-------
Part 35: saved Part 15 best checkpoint -> complete foreground collapse.
Part 36: exact cohort reconstruction -> conventional foreground Dice = 0.
Part 37: historical Part 15 metric/API mismatch confirmed.

Part 38 now evaluates EVERY available Part 15 checkpoint on the EXACT SAME
100-case validation cohort.

Checkpoint set:
    part15_initialization_from_part11.pth
    epoch_01.pth ... epoch_10.pth
    best_model.pth

For every checkpoint:
    - stored epoch / Dice metadata
    - parameter/state integrity
    - prediction foreground voxels
    - target foreground voxels
    - prediction/target foreground ratio
    - unique predicted labels
    - background-only case count
    - foreground-prediction case count
    - per-class TP / FP / FN
    - aggregate conventional foreground Dice
    - macro class Dice
    - precision / recall
    - comparison with stored historical Dice

IMPORTANT:
    Evaluation only.
    No training.
    No optimizer.
    No weight modification.
    No checkpoint modification.
    No SPIDER.
    No RSNA test set.
"""

from __future__ import annotations

import hashlib
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
import torch.nn.functional as F


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

PART8_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
)

VAL_MANIFEST = (
    PART8_DIR
    / "manifests"
    / "rsna_part8_validation_manifest.csv"
)

PART11_SOURCE = (
    SRC_DIR
    / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

PART15_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

CHECKPOINT_DIR = PART15_OUTPUT / "checkpoints"

PART15_HISTORY = (
    PART15_OUTPUT
    / "part15_training_history.csv"
)

PART15_VAL_COHORT = (
    PART15_OUTPUT
    / "part15_validation_cohort.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part38_epoch_checkpoint_prediction_forensic_audit"
)

CHECKPOINT_SUMMARY_CSV = (
    OUTPUT_DIR
    / "part38_checkpoint_summary.csv"
)

CHECKPOINT_CLASS_CSV = (
    OUTPUT_DIR
    / "part38_checkpoint_class_summary.csv"
)

CASE_CSV = (
    OUTPUT_DIR
    / "part38_case_checkpoint_predictions.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "phase4_part38_summary.json"
)

REPORT_TXT = (
    OUTPUT_DIR
    / "phase4_part38_report.txt"
)


# =============================================================================
# LOCKED CONFIGURATION
# =============================================================================

SEED = 42
VAL_CASES = 100
NUM_CLASSES = 6
PATCH_SIZE = (64, 96, 96)
FOREGROUND = [1, 2, 3, 4, 5]

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def import_module(path: Path, name: str):
    spec = spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import module: {path}"
        )

    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def sha256_file(path: Path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def norm_id(value):
    if pd.isna(value):
        return ""

    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def row_ids(row):
    return (
        norm_id(row.get("study_id")),
        norm_id(row.get("series_id")),
    )


def safe_float(value):
    try:
        value = float(value)

        if math.isnan(value):
            return None

        return value

    except Exception:
        return None


# =============================================================================
# CHECKPOINT DISCOVERY
# =============================================================================

def checkpoint_sort_key(path: Path):
    name = path.name.lower()

    if name == "part15_initialization_from_part11.pth":
        return (0, 0)

    if name == "epoch_01.pth":
        return (1, 1)

    if name.startswith("epoch_"):
        try:
            number = int(
                name.replace(
                    "epoch_",
                    "",
                ).replace(
                    ".pth",
                    "",
                )
            )
            return (1, number)
        except Exception:
            pass

    if name == "best_model.pth":
        return (2, 0)

    return (3, name)


def discover_checkpoints():
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: "
            f"{CHECKPOINT_DIR}"
        )

    paths = [
        p
        for p in CHECKPOINT_DIR.glob("*.pth")
        if p.is_file()
    ]

    paths.sort(
        key=checkpoint_sort_key
    )

    return paths


# =============================================================================
# CHECKPOINT METADATA
# =============================================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            value = checkpoint.get(key)

            if isinstance(value, dict):
                return value

        if checkpoint and all(
            isinstance(v, torch.Tensor)
            for v in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not locate model state_dict in checkpoint."
    )


def inspect_checkpoint(path: Path):
    info = {
        "filename": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        info["python_type"] = str(
            type(checkpoint)
        )

        if isinstance(checkpoint, dict):
            info["keys"] = ",".join(
                sorted(
                    str(k)
                    for k in checkpoint.keys()
                )
            )

            for key in (
                "epoch",
                "best_val_dice",
                "val_dice",
                "best_dice",
                "feature_size",
                "num_classes",
                "seed",
                "train_cases",
                "validation_cases",
                "source_checkpoint",
            ):
                if key in checkpoint:
                    value = checkpoint[key]

                    if isinstance(
                        value,
                        (
                            str,
                            int,
                            float,
                            bool,
                        ),
                    ) or value is None:
                        info[key] = value
                    else:
                        info[key] = str(
                            type(value)
                        )

        state = extract_state_dict(
            checkpoint
        )

        info["state_tensor_count"] = int(
            sum(
                1
                for v in state.values()
                if isinstance(
                    v,
                    torch.Tensor,
                )
            )
        )

        info["state_parameter_count"] = int(
            sum(
                v.numel()
                for v in state.values()
                if isinstance(
                    v,
                    torch.Tensor,
                )
            )
        )

        return (
            info,
            checkpoint,
        )

    except Exception as exc:
        info["load_error"] = repr(exc)
        raise


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_from_checkpoint(
    part11,
    checkpoint_path,
    device,
):
    model = part11.create_model(
        device
    )

    model = model.to(
        device
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state = extract_state_dict(
        checkpoint
    )

    cleaned = {
        (
            key[7:]
            if key.startswith("module.")
            else key
        ): value
        for key, value in state.items()
    }

    result = model.load_state_dict(
        cleaned,
        strict=False,
    )

    model.eval()

    return (
        model,
        checkpoint,
        result,
    )


# =============================================================================
# DATA SHAPE HELPERS
# =============================================================================

def ensure_image(x):
    if isinstance(
        x,
        np.ndarray,
    ):
        x = torch.from_numpy(x)

    x = torch.as_tensor(
        x
    ).float()

    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)

    elif x.ndim == 4:
        if x.shape[0] == 1:
            x = x.unsqueeze(0)
        elif x.shape[1] == 1:
            x = x.unsqueeze(0)

    if x.ndim != 5:
        raise ValueError(
            f"Unexpected image tensor shape: "
            f"{tuple(x.shape)}"
        )

    return x.contiguous()


def ensure_mask(x):
    if isinstance(
        x,
        np.ndarray,
    ):
        x = torch.from_numpy(x)

    x = torch.as_tensor(
        x
    ).long()

    if x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim != 3:
        raise ValueError(
            f"Unexpected mask tensor shape: "
            f"{tuple(x.shape)}"
        )

    return x.contiguous()


def normalize_case(
    image,
    mask,
):
    image = ensure_image(
        image
    )

    mask = ensure_mask(
        mask
    )

    if tuple(
        image.shape[-3:]
    ) != PATCH_SIZE:
        image = F.interpolate(
            image,
            size=PATCH_SIZE,
            mode="trilinear",
            align_corners=False,
        )

    if tuple(
        mask.shape
    ) != PATCH_SIZE:
        mask = F.interpolate(
            mask[None, None].float(),
            size=PATCH_SIZE,
            mode="nearest",
        ).squeeze(
            0
        ).squeeze(
            0
        ).long()

    return (
        image.contiguous(),
        mask.contiguous(),
    )


def load_case(
    part11,
    part9,
    row,
):
    image, mask, info = (
        part11.load_tensor_case(
            row,
            part9,
        )
    )

    if isinstance(
        image,
        torch.Tensor,
    ):
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]

        image = (
            image
            .detach()
            .cpu()
            .numpy()
        )
    else:
        image = np.asarray(
            image
        )

    if isinstance(
        mask,
        torch.Tensor,
    ):
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask[0]

        mask = (
            mask
            .detach()
            .cpu()
            .numpy()
        )
    else:
        mask = np.asarray(
            mask
        )

    processed = part11.preprocess_case(
        image,
        mask,
    )

    if not isinstance(
        processed,
        tuple,
    ) or len(processed) < 2:
        raise RuntimeError(
            "Part 11 preprocess_case() did not "
            "return image/mask."
        )

    image = processed[0]
    mask = processed[1]

    return normalize_case(
        image,
        mask,
    )


# =============================================================================
# METRIC
# =============================================================================

def evaluate_prediction(
    logits,
    target,
):
    prediction = torch.argmax(
        logits,
        dim=1,
    ).squeeze(0)

    class_stats = {}

    for c in FOREGROUND:
        p = prediction == c
        t = target == c

        tp = int(
            (p & t).sum().item()
        )

        fp = int(
            (p & ~t).sum().item()
        )

        fn = int(
            (~p & t).sum().item()
        )

        target_voxels = int(
            t.sum().item()
        )

        prediction_voxels = int(
            p.sum().item()
        )

        denominator = (
            2 * tp
            + fp
            + fn
        )

        dice = (
            0.0
            if denominator == 0
            else 2.0 * tp / denominator
        )

        precision = (
            0.0
            if tp + fp == 0
            else tp / (tp + fp)
        )

        recall = (
            0.0
            if tp + fn == 0
            else tp / (tp + fn)
        )

        class_stats[c] = {
            "target_voxels": target_voxels,
            "prediction_voxels": prediction_voxels,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }

    total_target = sum(
        class_stats[c]["target_voxels"]
        for c in FOREGROUND
    )

    total_prediction = sum(
        class_stats[c]["prediction_voxels"]
        for c in FOREGROUND
    )

    total_tp = sum(
        class_stats[c]["tp"]
        for c in FOREGROUND
    )

    total_fp = sum(
        class_stats[c]["fp"]
        for c in FOREGROUND
    )

    total_fn = sum(
        class_stats[c]["fn"]
        for c in FOREGROUND
    )

    aggregate_denominator = (
        2 * total_tp
        + total_fp
        + total_fn
    )

    aggregate_dice = (
        0.0
        if aggregate_denominator == 0
        else 2.0 * total_tp
        / aggregate_denominator
    )

    macro_dice = float(
        np.mean(
            [
                class_stats[c]["dice"]
                for c in FOREGROUND
            ]
        )
    )

    unique_labels = sorted(
        int(x)
        for x in torch.unique(
            prediction
        ).detach().cpu().tolist()
    )

    return {
        "prediction": prediction,
        "class_stats": class_stats,
        "target_foreground_voxels": total_target,
        "prediction_foreground_voxels": total_prediction,
        "aggregate_tp": total_tp,
        "aggregate_fp": total_fp,
        "aggregate_fn": total_fn,
        "aggregate_dice": aggregate_dice,
        "macro_dice": macro_dice,
        "prediction_target_ratio": (
            total_prediction / total_target
            if total_target
            else float("nan")
        ),
        "unique_labels": unique_labels,
        "background_only": (
            unique_labels == [0]
        ),
        "has_foreground_prediction": (
            any(
                c in unique_labels
                for c in FOREGROUND
            )
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    section("PHASE 4 - PART 38")

    print(
        "RSNA-ONLY EPOCH-BY-EPOCH CHECKPOINT "
        "PREDICTION FORENSIC AUDIT"
    )

    print()
    print("Evaluation only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("No optimizer step is performed.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")

    # -------------------------------------------------------------------------
    # PATH VALIDATION
    # -------------------------------------------------------------------------
    section("PATH VALIDATION")

    paths = {
        "RSNA root": RSNA_ROOT,
        "Part 8 validation manifest": VAL_MANIFEST,
        "Part 11 source": PART11_SOURCE,
        "Part 15 checkpoint directory": CHECKPOINT_DIR,
        "Part 15 history": PART15_HISTORY,
        "Part 15 validation cohort": PART15_VAL_COHORT,
    }

    for name, path in paths.items():
        print(
            f"{name:<38}: "
            f"{'FOUND' if path.exists() else 'MISSING'}"
        )

    required = [
        RSNA_ROOT,
        VAL_MANIFEST,
        PART11_SOURCE,
        CHECKPOINT_DIR,
        PART15_HISTORY,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required paths:\n"
            + "\n".join(missing)
        )

    # -------------------------------------------------------------------------
    # ENVIRONMENT
    # -------------------------------------------------------------------------
    section("PYTORCH / GPU ENVIRONMENT")

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"PyTorch version                    : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available                     : "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"Device                             : "
        f"{device}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU                               : "
            f"{torch.cuda.get_device_name(0)}"
        )

        props = torch.cuda.get_device_properties(
            0
        )

        print(
            f"GPU memory                        : "
            f"{props.total_memory / (1024 ** 3):.2f} GB"
        )

    print(
        f"Patch size                         : "
        f"{PATCH_SIZE}"
    )

    print(
        f"Feature/classes                    : "
        f"{NUM_CLASSES}"
    )

    print(
        f"Validation cases                   : "
        f"{VAL_CASES}"
    )

    # -------------------------------------------------------------------------
    # IMPORT PART 11
    # -------------------------------------------------------------------------
    section("IMPORTING VALIDATED PART 11")

    part11 = import_module(
        PART11_SOURCE,
        "part11_for_part38",
    )

    print(
        "✓ Corrected Part 11 imported."
    )

    print(
        "create_model                       : "
        f"{inspect_signature(part11, 'create_model')}"
    )

    print(
        "preprocess_case                    : "
        f"{inspect_signature(part11, 'preprocess_case')}"
    )

    print(
        "load_tensor_case                  : "
        f"{inspect_signature(part11, 'load_tensor_case')}"
    )

    # -------------------------------------------------------------------------
    # EXACT COHORT
    # -------------------------------------------------------------------------
    section("RECONSTRUCTING EXACT PART 15 VALIDATION COHORT")

    val_manifest = pd.read_csv(
        VAL_MANIFEST
    )

    cohort = part11.select_pilot_rows(
        VAL_MANIFEST,
        VAL_CASES,
        SEED,
    ).reset_index(drop=True)

    print(
        f"Validation manifest rows             : "
        f"{len(val_manifest)}"
    )

    print(
        f"Reconstructed validation rows        : "
        f"{len(cohort)}"
    )

    cohort_match = None

    if PART15_VAL_COHORT.exists():
        saved = pd.read_csv(
            PART15_VAL_COHORT
        )

        reconstructed_pairs = [
            row_ids(row)
            for _, row in cohort.iterrows()
        ]

        saved_pairs = [
            row_ids(row)
            for _, row in saved.iterrows()
        ]

        cohort_match = (
            reconstructed_pairs
            == saved_pairs
        )

        print(
            f"Saved Part 15 cohort rows           : "
            f"{len(saved)}"
        )

        print(
            f"Exact row-order cohort match        : "
            f"{cohort_match}"
        )

    if len(cohort) != VAL_CASES:
        raise RuntimeError(
            f"Expected {VAL_CASES} validation cases; "
            f"got {len(cohort)}."
        )

    # -------------------------------------------------------------------------
    # HISTORY
    # -------------------------------------------------------------------------
    section("LOADING PART 15 HISTORY")

    history = pd.read_csv(
        PART15_HISTORY
    )

    recorded_best = None

    if "val_dice" in history.columns:
        recorded_best = float(
            history["val_dice"].max()
        )

    print(
        f"History rows                      : "
        f"{len(history)}"
    )

    print(
        f"Recorded best validation Dice     : "
        f"{recorded_best}"
    )

    # -------------------------------------------------------------------------
    # CHECKPOINT INVENTORY
    # -------------------------------------------------------------------------
    section("DISCOVERING PART 15 CHECKPOINTS")

    checkpoints = discover_checkpoints()

    print(
        f"Checkpoint files discovered        : "
        f"{len(checkpoints)}"
    )

    for path in checkpoints:
        print(
            f"  {path.name}"
        )

    expected_names = [
        "part15_initialization_from_part11.pth"
    ] + [
        f"epoch_{i:02d}.pth"
        for i in range(1, 11)
    ] + [
        "best_model.pth"
    ]

    missing_expected = [
        name
        for name in expected_names
        if not (
            CHECKPOINT_DIR / name
        ).exists()
    ]

    if missing_expected:
        print()
        print(
            "WARNING - Expected checkpoint(s) missing:"
        )

        for name in missing_expected:
            print(
                f"  {name}"
            )

    # -------------------------------------------------------------------------
    # INVENTORY METADATA
    # -------------------------------------------------------------------------
    section("CHECKPOINT METADATA INVENTORY")

    inventory_records = []

    for path in checkpoints:
        info, _ = inspect_checkpoint(
            path
        )

        inventory_records.append(
            info
        )

        print(
            f"{path.name:<42} "
            f"epoch={info.get('epoch', 'N/A')} "
            f"stored_dice={info.get('best_val_dice', 'N/A')} "
            f"params={info.get('state_parameter_count', 'N/A'):,}"
        )

    inventory_df = pd.DataFrame(
        inventory_records
    )

    inventory_df.to_csv(
        CHECKPOINT_SUMMARY_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # PART 9
    # -------------------------------------------------------------------------
    section("LOADING PART 9 THROUGH PART 11")

    part9 = part11.load_part9_module()

    print(
        "✓ Part 9 loaded through corrected Part 11."
    )

    # -------------------------------------------------------------------------
    # CASE DATASET LOAD
    # -------------------------------------------------------------------------
    section("LOADING EXACT VALIDATION TENSORS")

    cases = []

    failed_cases = []

    for index, row in cohort.iterrows():
        case_index = index + 1
        study, series = row_ids(row)

        try:
            image, target = load_case(
                part11,
                part9,
                row,
            )

            cases.append({
                "case_index": case_index,
                "study_id": study,
                "series_id": series,
                "image": image,
                "target": target,
            })

        except Exception as exc:
            failed_cases.append({
                "case_index": case_index,
                "study_id": study,
                "series_id": series,
                "error": repr(exc),
            })

    print(
        f"Successful tensor loads              : "
        f"{len(cases)}"
    )

    print(
        f"Failed tensor loads                  : "
        f"{len(failed_cases)}"
    )

    if failed_cases:
        print()
        print(
            "WARNING: Some validation cases could not be loaded."
        )

    if not cases:
        raise RuntimeError(
            "No validation cases could be loaded."
        )

    # -------------------------------------------------------------------------
    # EVALUATE EVERY CHECKPOINT
    # -------------------------------------------------------------------------
    section("RUNNING EPOCH-BY-EPOCH CHECKPOINT EVALUATION")

    checkpoint_summary = []
    class_summary = []
    case_records = []

    for checkpoint_number, checkpoint_path in enumerate(
        checkpoints,
        start=1,
    ):
        print()
        print(
            "-" * 78
        )

        print(
            f"CHECKPOINT [{checkpoint_number}/{len(checkpoints)}] "
            f"{checkpoint_path.name}"
        )

        info, checkpoint_metadata = inspect_checkpoint(
            checkpoint_path
        )

        try:
            model, checkpoint, load_result = (
                load_model_from_checkpoint(
                    part11,
                    checkpoint_path,
                    device,
                )
            )

            print(
                f"  epoch metadata                    : "
                f"{checkpoint.get('epoch', 'N/A') if isinstance(checkpoint, dict) else 'N/A'}"
            )

            print(
                f"  stored best Dice                 : "
                f"{checkpoint.get('best_val_dice', 'N/A') if isinstance(checkpoint, dict) else 'N/A'}"
            )

            print(
                f"  missing state keys                : "
                f"{len(load_result.missing_keys)}"
            )

            print(
                f"  unexpected state keys             : "
                f"{len(load_result.unexpected_keys)}"
            )

        except Exception as exc:
            print(
                f"  CHECKPOINT LOAD ERROR: {exc}"
            )

            checkpoint_summary.append({
                "checkpoint": checkpoint_path.name,
                "checkpoint_path": str(
                    checkpoint_path
                ),
                "checkpoint_epoch": info.get(
                    "epoch"
                ),
                "stored_best_val_dice": safe_float(
                    info.get(
                        "best_val_dice"
                    )
                ),
                "status": "LOAD_ERROR",
                "error": repr(exc),
            })

            continue

        totals = {
            c: {
                "target": 0,
                "prediction": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
            }
            for c in FOREGROUND
        }

        background_only_cases = 0
        foreground_prediction_cases = 0
        all_prediction_labels = set()

        successful_inference_cases = 0
        failed_inference_cases = 0

        with torch.no_grad():
            for case in cases:
                try:
                    image = case["image"].to(
                        device
                    )

                    target = case["target"].to(
                        device
                    )

                    logits = model(
                        image
                    )

                    metrics = evaluate_prediction(
                        logits,
                        target,
                    )

                    successful_inference_cases += 1

                    if metrics[
                        "background_only"
                    ]:
                        background_only_cases += 1

                    if metrics[
                        "has_foreground_prediction"
                    ]:
                        foreground_prediction_cases += 1

                    all_prediction_labels.update(
                        metrics[
                            "unique_labels"
                        ]
                    )

                    for c in FOREGROUND:
                        s = metrics[
                            "class_stats"
                        ][c]

                        totals[c]["target"] += s[
                            "target_voxels"
                        ]

                        totals[c]["prediction"] += s[
                            "prediction_voxels"
                        ]

                        totals[c]["tp"] += s[
                            "tp"
                        ]

                        totals[c]["fp"] += s[
                            "fp"
                        ]

                        totals[c]["fn"] += s[
                            "fn"
                        ]

                    case_records.append({
                        "checkpoint": checkpoint_path.name,
                        "checkpoint_epoch": (
                            checkpoint.get("epoch")
                            if isinstance(
                                checkpoint,
                                dict,
                            )
                            else None
                        ),
                        "case_index": case[
                            "case_index"
                        ],
                        "study_id": case[
                            "study_id"
                        ],
                        "series_id": case[
                            "series_id"
                        ],
                        "target_foreground_voxels": metrics[
                            "target_foreground_voxels"
                        ],
                        "prediction_foreground_voxels": metrics[
                            "prediction_foreground_voxels"
                        ],
                        "prediction_target_ratio": safe_float(
                            metrics[
                                "prediction_target_ratio"
                            ]
                        ),
                        "aggregate_dice": metrics[
                            "aggregate_dice"
                        ],
                        "macro_dice": metrics[
                            "macro_dice"
                        ],
                        "unique_prediction_labels": ",".join(
                            map(
                                str,
                                metrics[
                                    "unique_labels"
                                ],
                            )
                        ),
                        "background_only": metrics[
                            "background_only"
                        ],
                    })

                except Exception as exc:
                    failed_inference_cases += 1

                    case_records.append({
                        "checkpoint": checkpoint_path.name,
                        "checkpoint_epoch": (
                            checkpoint.get("epoch")
                            if isinstance(
                                checkpoint,
                                dict,
                            )
                            else None
                        ),
                        "case_index": case[
                            "case_index"
                        ],
                        "study_id": case[
                            "study_id"
                        ],
                        "series_id": case[
                            "series_id"
                        ],
                        "status": "ERROR",
                        "error": repr(exc),
                    })

        total_target = sum(
            totals[c]["target"]
            for c in FOREGROUND
        )

        total_prediction = sum(
            totals[c]["prediction"]
            for c in FOREGROUND
        )

        total_tp = sum(
            totals[c]["tp"]
            for c in FOREGROUND
        )

        total_fp = sum(
            totals[c]["fp"]
            for c in FOREGROUND
        )

        total_fn = sum(
            totals[c]["fn"]
            for c in FOREGROUND
        )

        denominator = (
            2 * total_tp
            + total_fp
            + total_fn
        )

        aggregate_dice = (
            0.0
            if denominator == 0
            else 2.0 * total_tp
            / denominator
        )

        per_class_dice = {}

        for c in FOREGROUND:
            s = totals[c]

            d = (
                0.0
                if (
                    2 * s["tp"]
                    + s["fp"]
                    + s["fn"]
                ) == 0
                else (
                    2.0 * s["tp"]
                    / (
                        2 * s["tp"]
                        + s["fp"]
                        + s["fn"]
                    )
                )
            )

            p = (
                0.0
                if s["tp"] + s["fp"] == 0
                else s["tp"]
                / (
                    s["tp"]
                    + s["fp"]
                )
            )

            r = (
                0.0
                if s["tp"] + s["fn"] == 0
                else s["tp"]
                / (
                    s["tp"]
                    + s["fn"]
                )
            )

            per_class_dice[c] = d

            class_summary.append({
                "checkpoint": checkpoint_path.name,
                "checkpoint_epoch": (
                    checkpoint.get("epoch")
                    if isinstance(
                        checkpoint,
                        dict,
                    )
                    else None
                ),
                "class_id": c,
                "class_name": CLASS_NAMES[c],
                "target_voxels": s["target"],
                "prediction_voxels": s["prediction"],
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
                "dice": d,
                "precision": p,
                "recall": r,
            })

        macro_dice = float(
            np.mean(
                list(
                    per_class_dice.values()
                )
            )
        )

        stored_dice = safe_float(
            checkpoint.get(
                "best_val_dice"
            )
            if isinstance(
                checkpoint,
                dict,
            )
            else None
        )

        epoch_metadata = (
            checkpoint.get("epoch")
            if isinstance(
                checkpoint,
                dict,
            )
            else None
        )

        history_val_dice = None

        if epoch_metadata is not None:
            try:
                matches = history[
                    history["epoch"].astype(int)
                    == int(epoch_metadata)
                ]

                if not matches.empty:
                    history_val_dice = float(
                        matches.iloc[0]["val_dice"]
                    )

            except Exception:
                pass

        if (
            stored_dice is not None
            and not math.isnan(
                stored_dice
            )
        ):
            stored_minus_reconstructed = (
                stored_dice
                - aggregate_dice
            )
        else:
            stored_minus_reconstructed = float(
                "nan"
            )

        summary_record = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_path": str(
                checkpoint_path
            ),
            "checkpoint_epoch": epoch_metadata,
            "status": "OK",
            "successful_inference_cases": successful_inference_cases,
            "failed_inference_cases": failed_inference_cases,
            "stored_best_val_dice": stored_dice,
            "history_epoch_val_dice": history_val_dice,
            "reconstructed_aggregate_foreground_dice": (
                aggregate_dice
            ),
            "reconstructed_macro_class_dice": (
                macro_dice
            ),
            "stored_minus_reconstructed_dice": (
                stored_minus_reconstructed
            ),
            "target_foreground_voxels": total_target,
            "prediction_foreground_voxels": total_prediction,
            "prediction_target_foreground_ratio": (
                total_prediction / total_target
                if total_target
                else float("nan")
            ),
            "background_only_cases": background_only_cases,
            "foreground_prediction_cases": (
                foreground_prediction_cases
            ),
            "unique_prediction_labels": ",".join(
                map(
                    str,
                    sorted(
                        all_prediction_labels
                    ),
                )
            ),
            "all_background": (
                total_prediction == 0
            ),
            "load_missing_keys": len(
                load_result.missing_keys
            ),
            "load_unexpected_keys": len(
                load_result.unexpected_keys
            ),
            "sha256": info.get(
                "sha256"
            ),
            "size_bytes": info.get(
                "size_bytes"
            ),
        }

        checkpoint_summary.append(
            summary_record
        )

        print(
            f"  target foreground voxels         : "
            f"{total_target:,}"
        )

        print(
            f"  predicted foreground voxels      : "
            f"{total_prediction:,}"
        )

        print(
            f"  prediction/target ratio          : "
            f"{summary_record['prediction_target_foreground_ratio']:.6f}"
        )

        print(
            f"  background-only cases            : "
            f"{background_only_cases}/{successful_inference_cases}"
        )

        print(
            f"  foreground-prediction cases      : "
            f"{foreground_prediction_cases}/{successful_inference_cases}"
        )

        print(
            f"  reconstructed aggregate Dice     : "
            f"{aggregate_dice:.6f}"
        )

        print(
            f"  reconstructed macro Dice         : "
            f"{macro_dice:.6f}"
        )

        print(
            f"  stored checkpoint Dice           : "
            f"{stored_dice}"
        )

        print(
            f"  unique prediction labels         : "
            f"{sorted(all_prediction_labels)}"
        )

    # -------------------------------------------------------------------------
    # SAVE TABLES
    # -------------------------------------------------------------------------
    section("SAVING PART 38 OUTPUTS")

    checkpoint_df = pd.DataFrame(
        checkpoint_summary
    )

    class_df = pd.DataFrame(
        class_summary
    )

    case_df = pd.DataFrame(
        case_records
    )

    checkpoint_df.to_csv(
        CHECKPOINT_SUMMARY_CSV,
        index=False,
    )

    class_df.to_csv(
        CHECKPOINT_CLASS_CSV,
        index=False,
    )

    case_df.to_csv(
        CASE_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # FINAL DIAGNOSIS
    # -------------------------------------------------------------------------
    section("PART 38 OVERALL DIAGNOSIS")

    ok_df = checkpoint_df[
        checkpoint_df["status"] == "OK"
    ].copy()

    if ok_df.empty:
        diagnosis = (
            "NO_CHECKPOINT_COULD_BE_EVALUATED"
        )

        all_background_checkpoints = 0
        foreground_capable_checkpoints = 0

    else:
        all_background_checkpoints = int(
            ok_df[
                "all_background"
            ].fillna(False).sum()
        )

        foreground_capable_checkpoints = int(
            (
                ok_df[
                    "foreground_prediction_cases"
                ].fillna(0)
                > 0
            ).sum()
        )

        epoch_rows = ok_df[
            ok_df["checkpoint"].str.match(
                r"epoch_\d+\.pth"
            )
        ].copy()

        if (
            len(epoch_rows) > 0
            and all(
                epoch_rows[
                    "all_background"
                ].fillna(False)
            )
        ):
            diagnosis = (
                "FOREGROUND_COLLAPSE_PRESENT_ACROSS_ALL_"
                "EPOCH_CHECKPOINTS"
            )

        elif (
            foreground_capable_checkpoints == 0
        ):
            diagnosis = (
                "FOREGROUND_COLLAPSE_PRESENT_ACROSS_"
                "AVAILABLE_CHECKPOINTS"
            )

        else:
            diagnosis = (
                "CHECKPOINT_EVOLUTION_REQUIRES_"
                "EPOCH_SPECIFIC_ANALYSIS"
            )

    print(
        f"Checkpoint files evaluated          : "
        f"{len(ok_df)}"
    )

    print(
        f"All-background checkpoints          : "
        f"{all_background_checkpoints}"
    )

    print(
        f"Foreground-capable checkpoints      : "
        f"{foreground_capable_checkpoints}"
    )

    if not ok_df.empty:
        print()
        print(
            "CHECKPOINT COMPARISON"
        )

        cols = [
            "checkpoint",
            "checkpoint_epoch",
            "stored_best_val_dice",
            "history_epoch_val_dice",
            "reconstructed_aggregate_foreground_dice",
            "reconstructed_macro_class_dice",
            "prediction_foreground_voxels",
            "prediction_target_foreground_ratio",
            "background_only_cases",
            "foreground_prediction_cases",
            "unique_prediction_labels",
        ]

        print(
            ok_df[
                [
                    c
                    for c in cols
                    if c in ok_df.columns
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        f"DIAGNOSIS                           : "
        f"{diagnosis}"
    )

    # -------------------------------------------------------------------------
    # IMPORTANT HISTORICAL COMPARISON
    # -------------------------------------------------------------------------
    section("HISTORICAL DICE VS CHECKPOINT BEHAVIOR")

    print(
        f"Part 15 recorded best Dice          : "
        f"{recorded_best}"
    )

    if not ok_df.empty:
        best_reconstructed_idx = ok_df[
            "reconstructed_aggregate_foreground_dice"
        ].idxmax()

        best_reconstructed = ok_df.loc[
            best_reconstructed_idx
        ]

        print(
            f"Best reconstructed checkpoint       : "
            f"{best_reconstructed['checkpoint']}"
        )

        print(
            f"Best reconstructed foreground Dice  : "
            f"{best_reconstructed['reconstructed_aggregate_foreground_dice']:.6f}"
        )

        print(
            f"Best reconstructed foreground voxels: "
            f"{best_reconstructed['prediction_foreground_voxels']:,}"
        )

    print()
    print(
        "Interpretation:"
    )

    if (
        not ok_df.empty
        and all_background_checkpoints
        == len(ok_df)
    ):
        print(
            "  Every evaluated checkpoint predicts only background "
            "on the exact validation cohort."
        )

        print(
            "  Therefore the 0.668 historical value is not reproduced "
            "by conventional foreground Dice from any saved checkpoint."
        )

        print(
            "  The next investigation should focus on the historical "
            "training-time metric implementation and the training "
            "learning signal, not on checkpoint selection alone."
        )

    elif foreground_capable_checkpoints > 0:
        print(
            "  At least one checkpoint produces foreground predictions."
        )

        print(
            "  Therefore checkpoint evolution is informative and "
            "the next step should compare the first foreground-capable "
            "epoch against its loss/metric history."
        )

    # -------------------------------------------------------------------------
    # SUMMARY JSON
    # -------------------------------------------------------------------------
    summary = {
        "phase": "Phase 4 - Part 38",
        "description": (
            "RSNA-only epoch-by-epoch Part 15 checkpoint "
            "prediction forensic audit"
        ),
        "validation_cases_requested": VAL_CASES,
        "validation_cases_loaded": len(cases),
        "validation_cases_failed_to_load": len(
            failed_cases
        ),
        "exact_cohort_match": cohort_match,
        "checkpoint_files_discovered": [
            p.name
            for p in checkpoints
        ],
        "checkpoint_files_evaluated": int(
            len(ok_df)
        ),
        "all_background_checkpoints": (
            all_background_checkpoints
        ),
        "foreground_capable_checkpoints": (
            foreground_capable_checkpoints
        ),
        "part15_recorded_best_dice": safe_float(
            recorded_best
        ),
        "diagnosis": diagnosis,
        "checkpoint_summary": (
            checkpoint_summary
        ),
        "failed_tensor_cases": failed_cases,
        "training_performed": False,
        "model_weights_modified": False,
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "spider_used": False,
        "rsna_test_set_used": False,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "PHASE 4 - PART 38",
        "RSNA-ONLY EPOCH-BY-EPOCH CHECKPOINT "
        "PREDICTION FORENSIC AUDIT",
        "=" * 78,
        "",
        f"Validation cases requested: {VAL_CASES}",
        f"Validation cases loaded: {len(cases)}",
        f"Validation load failures: {len(failed_cases)}",
        f"Exact cohort match: {cohort_match}",
        f"Checkpoint files discovered: {len(checkpoints)}",
        f"Checkpoint files evaluated: {len(ok_df)}",
        f"All-background checkpoints: "
        f"{all_background_checkpoints}",
        f"Foreground-capable checkpoints: "
        f"{foreground_capable_checkpoints}",
        f"Part 15 recorded best Dice: "
        f"{recorded_best}",
        f"DIAGNOSIS: {diagnosis}",
        "",
        "CHECKPOINT SUMMARY",
        "-" * 78,
        (
            ok_df.to_string(index=False)
            if not ok_df.empty
            else "No checkpoints successfully evaluated."
        ),
        "",
        "CLASS SUMMARY",
        "-" * 78,
        (
            class_df.to_string(index=False)
            if not class_df.empty
            else "No class metrics available."
        ),
        "",
        "FAILED CASES",
        "-" * 78,
        json.dumps(
            failed_cases,
            indent=2,
        ),
        "",
        "No training performed.",
        "No model weights modified.",
        "No optimizer created.",
        "No optimizer step performed.",
        "SPIDER not used.",
        "RSNA test set not used.",
    ]

    REPORT_TXT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(
        f"Checkpoint summary CSV              : "
        f"{CHECKPOINT_SUMMARY_CSV}"
    )

    print(
        f"Checkpoint class CSV                : "
        f"{CHECKPOINT_CLASS_CSV}"
    )

    print(
        f"Case prediction CSV                : "
        f"{CASE_CSV}"
    )

    print(
        f"Summary JSON                        : "
        f"{SUMMARY_JSON}"
    )

    print(
        f"Report TXT                          : "
        f"{REPORT_TXT}"
    )

    section("PHASE 4 - PART 38 COMPLETE")

    print(
        "✓ All available Part 15 checkpoints inventoried."
    )

    print(
        "✓ Exact validation cohort reconstructed."
    )

    print(
        "✓ Same validation tensors reused for checkpoint comparison."
    )

    print(
        "✓ Epoch-by-epoch predictions evaluated."
    )

    print(
        "✓ Foreground/background prediction distributions calculated."
    )

    print(
        "✓ Aggregate conventional Dice calculated."
    )

    print(
        "✓ Per-class TP / FP / FN calculated."
    )

    print(
        "✓ Historical stored Dice compared with reconstructed Dice."
    )

    print(
        "✓ No training performed."
    )

    print(
        "✓ No model weights modified."
    )

    print(
        "✓ No optimizer created."
    )

    print(
        "✓ No optimizer step performed."
    )

    print(
        "✓ SPIDER not used."
    )

    print(
        "✓ RSNA test set not used."
    )


def inspect_signature(module, function_name):
    try:
        import inspect

        function = getattr(
            module,
            function_name,
        )

        return str(
            inspect.signature(
                function
            )
        )

    except Exception:
        return "N/A"


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("=" * 78)
        print("PART 38 ERROR")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
