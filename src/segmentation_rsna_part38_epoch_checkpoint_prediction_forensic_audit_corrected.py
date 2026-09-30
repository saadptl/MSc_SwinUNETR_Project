"""
==============================================================================
PHASE 4 - PART 38 CORRECTED
RSNA-ONLY EPOCH-BY-EPOCH CHECKPOINT PREDICTION FORENSIC AUDIT
==============================================================================

Purpose
-------
Corrected version of Part 38.

The previous Part 38 evaluation failed after the initialization checkpoint
because every subsequent model was moved to the 4-GB GPU while CUDA memory
from the previous model remained allocated.

This version evaluates ONE checkpoint at a time and explicitly releases:
    - model
    - checkpoint objects
    - logits
    - tensors
    - CUDA cache

It also:
    - loads checkpoint weights on CPU first
    - evaluates on GPU when possible
    - automatically falls back to CPU if GPU evaluation runs out of memory
    - never keeps multiple models alive simultaneously
    - continues to the next checkpoint after an individual failure
    - uses the exact Part 15 100-case validation cohort
    - does not train or modify checkpoints

No training is performed.
No optimizer is created.
No optimizer step is performed.
No model weights are modified.
SPIDER is not used.
RSNA test set is not used.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import random
import sys
import traceback
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
    / "rsna_part38_corrected_epoch_checkpoint_prediction_forensic_audit"
)

CHECKPOINT_SUMMARY_CSV = (
    OUTPUT_DIR
    / "part38_checkpoint_summary.csv"
)

CLASS_SUMMARY_CSV = (
    OUTPUT_DIR
    / "part38_checkpoint_class_summary.csv"
)

CASE_CSV = (
    OUTPUT_DIR
    / "part38_case_checkpoint_predictions.csv"
)

FAILED_CHECKPOINT_CSV = (
    OUTPUT_DIR
    / "part38_failed_checkpoints.csv"
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
# LOCKED PROJECT CONFIGURATION
# =============================================================================

SEED = 42
VAL_CASES = 100

NUM_CLASSES = 6
PATCH_SIZE = (64, 96, 96)

FOREGROUND_CLASSES = (1, 2, 3, 4, 5)

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# =============================================================================
# PRINT HELPERS
# =============================================================================

def section(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


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


def cleanup_cuda():
    """
    Aggressively release Python references and CUDA cached allocations.

    This function never changes model parameters.
    """
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        gc.collect()


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


def import_module(path: Path, name: str):
    spec = spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import module: {path}"
        )

    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def sha256_file(path: Path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# =============================================================================
# CHECKPOINT DISCOVERY
# =============================================================================

def checkpoint_sort_key(path: Path):
    name = path.name.lower()

    if name == "part15_initialization_from_part11.pth":
        return (0, 0)

    if name.startswith("epoch_") and name.endswith(".pth"):
        try:
            number = int(
                name[len("epoch_"):-len(".pth")]
            )

            return (1, number)

        except Exception:
            pass

    if name == "best_model.pth":
        return (2, 0)

    return (3, name)


def discover_checkpoints():
    paths = sorted(
        [
            p
            for p in CHECKPOINT_DIR.glob("*.pth")
            if p.is_file()
        ],
        key=checkpoint_sort_key,
    )

    return paths


# =============================================================================
# CHECKPOINT STATE EXTRACTION
# =============================================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            candidate = checkpoint.get(key)

            if isinstance(candidate, dict):
                return candidate

        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not locate model state_dict in checkpoint."
    )


def load_checkpoint_cpu(path: Path):
    """
    IMPORTANT:
    Checkpoint is loaded on CPU.

    This prevents torch.load() from allocating the checkpoint directly
    on CUDA and avoids retaining CUDA allocations between checkpoints.
    """
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def inspect_checkpoint(path: Path):
    checkpoint = load_checkpoint_cpu(path)

    info = {
        "filename": path.name,
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "python_type": str(type(checkpoint)),
    }

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

    state = extract_state_dict(
        checkpoint
    )

    info["state_tensor_count"] = int(
        sum(
            1
            for value in state.values()
            if isinstance(
                value,
                torch.Tensor,
            )
        )
    )

    info["state_parameter_count"] = int(
        sum(
            value.numel()
            for value in state.values()
            if isinstance(
                value,
                torch.Tensor,
            )
        )
    )

    return info


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_from_cpu_checkpoint(
    part11,
    checkpoint_path: Path,
    device: torch.device,
):
    """
    Load exactly one model.

    The checkpoint itself remains CPU-side until the state dictionary is
    copied into the newly created model.
    """
    checkpoint = load_checkpoint_cpu(
        checkpoint_path
    )

    state = extract_state_dict(
        checkpoint
    )

    cleaned_state = {
        (
            key[7:]
            if key.startswith("module.")
            else key
        ): value
        for key, value in state.items()
    }

    model = part11.create_model(
        device=torch.device("cpu")
    )

    model = model.to(
        device
    )

    load_result = model.load_state_dict(
        cleaned_state,
        strict=False,
    )

    model.eval()

    return (
        model,
        checkpoint,
        load_result,
    )


# =============================================================================
# TENSOR NORMALIZATION
# =============================================================================

def ensure_image_tensor(image):
    if isinstance(
        image,
        np.ndarray,
    ):
        image = torch.from_numpy(
            image
        )

    image = torch.as_tensor(
        image
    ).float()

    if image.ndim == 3:
        image = image.unsqueeze(0).unsqueeze(0)

    elif image.ndim == 4:
        if image.shape[0] == 1:
            image = image.unsqueeze(0)

        elif image.shape[1] == 1:
            image = image.unsqueeze(0)

    if image.ndim != 5:
        raise ValueError(
            "Unexpected image shape: "
            f"{tuple(image.shape)}"
        )

    return image.contiguous()


def ensure_mask_tensor(mask):
    if isinstance(
        mask,
        np.ndarray,
    ):
        mask = torch.from_numpy(
            mask
        )

    mask = torch.as_tensor(
        mask
    ).long()

    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    if mask.ndim != 3:
        raise ValueError(
            "Unexpected mask shape: "
            f"{tuple(mask.shape)}"
        )

    return mask.contiguous()


def normalize_case(
    image,
    mask,
):
    image = ensure_image_tensor(
        image
    )

    mask = ensure_mask_tensor(
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
        image = (
            image.detach()
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
        mask = (
            mask.detach()
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

    if (
        not isinstance(
            processed,
            tuple,
        )
        or len(processed) < 2
    ):
        raise RuntimeError(
            "Part 11 preprocess_case() did not "
            "return image and mask."
        )

    return normalize_case(
        processed[0],
        processed[1],
    )


# =============================================================================
# EXACT PART 15 COHORT
# =============================================================================

def reconstruct_validation_cohort(
    part11,
):
    manifest = pd.read_csv(
        VAL_MANIFEST
    )

    cohort = part11.select_pilot_rows(
        VAL_MANIFEST,
        VAL_CASES,
        SEED,
    ).reset_index(
        drop=True
    )

    exact_match = None
    saved_rows = None

    if PART15_VAL_COHORT.exists():
        saved = pd.read_csv(
            PART15_VAL_COHORT
        )

        saved_rows = len(saved)

        reconstructed_pairs = [
            row_ids(row)
            for _, row in cohort.iterrows()
        ]

        saved_pairs = [
            row_ids(row)
            for _, row in saved.iterrows()
        ]

        exact_match = (
            reconstructed_pairs
            == saved_pairs
        )

    print(
        f"Validation manifest rows             : "
        f"{len(manifest)}"
    )

    print(
        f"Reconstructed validation rows        : "
        f"{len(cohort)}"
    )

    if saved_rows is not None:
        print(
            f"Saved Part 15 cohort rows           : "
            f"{saved_rows}"
        )

        print(
            f"Exact row-order cohort match        : "
            f"{exact_match}"
        )

    return (
        cohort,
        exact_match,
    )


# =============================================================================
# CASE METRICS
# =============================================================================

def calculate_case_metrics(
    logits,
    target,
):
    prediction = torch.argmax(
        logits,
        dim=1,
    ).squeeze(0)

    stats = {}

    for class_id in FOREGROUND_CLASSES:
        pred = prediction == class_id
        true = target == class_id

        tp = int(
            (pred & true).sum().item()
        )

        fp = int(
            (pred & ~true).sum().item()
        )

        fn = int(
            (~pred & true).sum().item()
        )

        target_voxels = int(
            true.sum().item()
        )

        prediction_voxels = int(
            pred.sum().item()
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

        stats[class_id] = {
            "target_voxels": target_voxels,
            "prediction_voxels": prediction_voxels,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }

    target_foreground = sum(
        stats[c]["target_voxels"]
        for c in FOREGROUND_CLASSES
    )

    prediction_foreground = sum(
        stats[c]["prediction_voxels"]
        for c in FOREGROUND_CLASSES
    )

    tp_total = sum(
        stats[c]["tp"]
        for c in FOREGROUND_CLASSES
    )

    fp_total = sum(
        stats[c]["fp"]
        for c in FOREGROUND_CLASSES
    )

    fn_total = sum(
        stats[c]["fn"]
        for c in FOREGROUND_CLASSES
    )

    aggregate_denominator = (
        2 * tp_total
        + fp_total
        + fn_total
    )

    aggregate_dice = (
        0.0
        if aggregate_denominator == 0
        else 2.0 * tp_total
        / aggregate_denominator
    )

    macro_dice = float(
        np.mean(
            [
                stats[c]["dice"]
                for c in FOREGROUND_CLASSES
            ]
        )
    )

    unique_labels = sorted(
        int(value)
        for value in torch.unique(
            prediction
        ).detach().cpu().tolist()
    )

    return {
        "class_stats": stats,
        "target_foreground_voxels": target_foreground,
        "prediction_foreground_voxels": prediction_foreground,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "aggregate_dice": aggregate_dice,
        "macro_dice": macro_dice,
        "prediction_target_ratio": (
            prediction_foreground / target_foreground
            if target_foreground > 0
            else float("nan")
        ),
        "unique_labels": unique_labels,
        "background_only": (
            unique_labels == [0]
        ),
        "foreground_prediction": any(
            c in unique_labels
            for c in FOREGROUND_CLASSES
        ),
    }


# =============================================================================
# SINGLE-CHECKPOINT EVALUATION
# =============================================================================

def evaluate_checkpoint(
    part11,
    checkpoint_path,
    cases,
    preferred_device,
):
    """
    Evaluate exactly one checkpoint.

    GPU OOM policy:
        1. Attempt GPU.
        2. If CUDA OOM occurs, fully clean CUDA state.
        3. Reload the SAME checkpoint on CPU.
        4. Evaluate on CPU.
        5. Return successful metrics.

    This makes one bad GPU allocation unable to prevent the audit from
    evaluating the remaining checkpoints.
    """

    checkpoint_name = checkpoint_path.name

    print()
    print("-" * 78)
    print(
        f"CHECKPOINT: {checkpoint_name}"
    )

    metadata = inspect_checkpoint(
        checkpoint_path
    )

    stored_dice = safe_float(
        metadata.get(
            "best_val_dice"
        )
    )

    epoch = metadata.get(
        "epoch"
    )

    print(
        f"  epoch metadata                    : "
        f"{epoch}"
    )

    print(
        f"  stored best Dice                 : "
        f"{stored_dice}"
    )

    print(
        f"  checkpoint parameters             : "
        f"{metadata['state_parameter_count']:,}"
    )

    # -------------------------------------------------------------------------
    # Internal evaluator
    # -------------------------------------------------------------------------
    def run_on_device(device):
        model = None
        checkpoint = None
        logits = None

        try:
            cleanup_cuda()

            print(
                f"  evaluation device                 : "
                f"{device}"
            )

            model, checkpoint, load_result = (
                load_model_from_cpu_checkpoint(
                    part11,
                    checkpoint_path,
                    device,
                )
            )

            print(
                f"  missing state keys                : "
                f"{len(load_result.missing_keys)}"
            )

            print(
                f"  unexpected state keys             : "
                f"{len(load_result.unexpected_keys)}"
            )

            totals = {
                c: {
                    "target": 0,
                    "prediction": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                }
                for c in FOREGROUND_CLASSES
            }

            background_only_cases = 0
            foreground_prediction_cases = 0

            unique_prediction_labels = set()

            case_rows = []

            successful = 0
            failed = 0

            with torch.inference_mode():
                for case in cases:
                    try:
                        image = case[
                            "image"
                        ].to(
                            device,
                            non_blocking=False,
                        )

                        target = case[
                            "target"
                        ].to(
                            device,
                            non_blocking=False,
                        )

                        logits = model(
                            image
                        )

                        metrics = calculate_case_metrics(
                            logits,
                            target,
                        )

                        successful += 1

                        if metrics[
                            "background_only"
                        ]:
                            background_only_cases += 1

                        if metrics[
                            "foreground_prediction"
                        ]:
                            foreground_prediction_cases += 1

                        unique_prediction_labels.update(
                            metrics[
                                "unique_labels"
                            ]
                        )

                        for c in FOREGROUND_CLASSES:
                            item = metrics[
                                "class_stats"
                            ][c]

                            totals[c]["target"] += (
                                item[
                                    "target_voxels"
                                ]
                            )

                            totals[c]["prediction"] += (
                                item[
                                    "prediction_voxels"
                                ]
                            )

                            totals[c]["tp"] += (
                                item["tp"]
                            )

                            totals[c]["fp"] += (
                                item["fp"]
                            )

                            totals[c]["fn"] += (
                                item["fn"]
                            )

                        case_rows.append({
                            "checkpoint": checkpoint_name,
                            "checkpoint_epoch": epoch,
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

                        # Release case-level CUDA outputs immediately.
                        del image
                        del target
                        del logits
                        logits = None

                    except RuntimeError as exc:
                        message = str(
                            exc
                        ).lower()

                        if (
                            device.type == "cuda"
                            and (
                                "out of memory"
                                in message
                                or "cuda error"
                                in message
                            )
                        ):
                            raise

                        failed += 1

                        case_rows.append({
                            "checkpoint": checkpoint_name,
                            "checkpoint_epoch": epoch,
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

                    except Exception as exc:
                        failed += 1

                        case_rows.append({
                            "checkpoint": checkpoint_name,
                            "checkpoint_epoch": epoch,
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

            target_total = sum(
                totals[c]["target"]
                for c in FOREGROUND_CLASSES
            )

            prediction_total = sum(
                totals[c]["prediction"]
                for c in FOREGROUND_CLASSES
            )

            tp_total = sum(
                totals[c]["tp"]
                for c in FOREGROUND_CLASSES
            )

            fp_total = sum(
                totals[c]["fp"]
                for c in FOREGROUND_CLASSES
            )

            fn_total = sum(
                totals[c]["fn"]
                for c in FOREGROUND_CLASSES
            )

            denominator = (
                2 * tp_total
                + fp_total
                + fn_total
            )

            aggregate_dice = (
                0.0
                if denominator == 0
                else 2.0 * tp_total
                / denominator
            )

            per_class_dice = {}

            class_rows = []

            for c in FOREGROUND_CLASSES:
                item = totals[c]

                d = (
                    0.0
                    if (
                        2 * item["tp"]
                        + item["fp"]
                        + item["fn"]
                    ) == 0
                    else (
                        2.0 * item["tp"]
                        / (
                            2 * item["tp"]
                            + item["fp"]
                            + item["fn"]
                        )
                    )
                )

                p = (
                    0.0
                    if item["tp"] + item["fp"] == 0
                    else item["tp"]
                    / (
                        item["tp"]
                        + item["fp"]
                    )
                )

                r = (
                    0.0
                    if item["tp"] + item["fn"] == 0
                    else item["tp"]
                    / (
                        item["tp"]
                        + item["fn"]
                    )
                )

                per_class_dice[c] = d

                class_rows.append({
                    "checkpoint": checkpoint_name,
                    "checkpoint_epoch": epoch,
                    "class_id": c,
                    "class_name": CLASS_NAMES[c],
                    "target_voxels": item[
                        "target"
                    ],
                    "prediction_voxels": item[
                        "prediction"
                    ],
                    "tp": item["tp"],
                    "fp": item["fp"],
                    "fn": item["fn"],
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

            record = {
                "checkpoint": checkpoint_name,
                "checkpoint_path": str(
                    checkpoint_path
                ),
                "checkpoint_epoch": epoch,
                "status": "OK",
                "evaluation_device": str(
                    device
                ),
                "successful_inference_cases": successful,
                "failed_inference_cases": failed,
                "stored_best_val_dice": stored_dice,
                "reconstructed_aggregate_foreground_dice": (
                    aggregate_dice
                ),
                "reconstructed_macro_class_dice": (
                    macro_dice
                ),
                "stored_minus_reconstructed_dice": (
                    (
                        stored_dice
                        - aggregate_dice
                    )
                    if stored_dice is not None
                    else float("nan")
                ),
                "target_foreground_voxels": target_total,
                "prediction_foreground_voxels": prediction_total,
                "prediction_target_foreground_ratio": (
                    prediction_total / target_total
                    if target_total > 0
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
                            unique_prediction_labels
                        ),
                    )
                ),
                "all_background": (
                    prediction_total == 0
                ),
                "load_missing_keys": len(
                    load_result.missing_keys
                ),
                "load_unexpected_keys": len(
                    load_result.unexpected_keys
                ),
                "sha256": metadata[
                    "sha256"
                ],
                "size_bytes": metadata[
                    "size_bytes"
                ],
                "state_parameter_count": metadata[
                    "state_parameter_count"
                ],
            }

            return (
                record,
                class_rows,
                case_rows,
            )

        finally:
            # -------------------------------------------------------------
            # Critical cleanup.
            # -------------------------------------------------------------
            del logits

            if model is not None:
                del model

            if checkpoint is not None:
                del checkpoint

            cleanup_cuda()

    # -------------------------------------------------------------------------
    # FIRST TRY: GPU
    # -------------------------------------------------------------------------
    if preferred_device.type == "cuda":
        try:
            return run_on_device(
                preferred_device
            )

        except RuntimeError as exc:
            message = str(
                exc
            ).lower()

            if (
                "out of memory"
                not in message
                and "cuda error" not in message
            ):
                raise

            print(
                "  CUDA OOM detected."
            )

            print(
                "  Cleaning CUDA memory and "
                "retrying this checkpoint on CPU."
            )

            cleanup_cuda()

            # -------------------------------------------------------------
            # CPU fallback.
            # -------------------------------------------------------------
            return run_on_device(
                torch.device("cpu")
            )

    # -------------------------------------------------------------------------
    # CPU only
    # -------------------------------------------------------------------------
    return run_on_device(
        torch.device("cpu")
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    section("PHASE 4 - PART 38 CORRECTED")

    print(
        "RSNA-ONLY EPOCH-BY-EPOCH CHECKPOINT "
        "PREDICTION FORENSIC AUDIT"
    )

    print()
    print(
        "This corrected version evaluates one checkpoint at a time."
    )
    print(
        "CUDA memory is explicitly released between checkpoints."
    )
    print(
        "CPU fallback is enabled for GPU OOM."
    )

    print()
    print(
        "Evaluation only."
    )
    print(
        "No training is performed."
    )
    print(
        "No model weights are modified."
    )
    print(
        "No optimizer is created."
    )
    print(
        "No optimizer step is performed."
    )
    print(
        "SPIDER is not used."
    )
    print(
        "RSNA test set is not used."
    )

    # -------------------------------------------------------------------------
    # PATH VALIDATION
    # -------------------------------------------------------------------------
    section("PATH VALIDATION")

    path_items = [
        (
            "RSNA root",
            RSNA_ROOT,
        ),
        (
            "Part 8 validation manifest",
            VAL_MANIFEST,
        ),
        (
            "Part 11 source",
            PART11_SOURCE,
        ),
        (
            "Part 15 checkpoint directory",
            CHECKPOINT_DIR,
        ),
        (
            "Part 15 history",
            PART15_HISTORY,
        ),
        (
            "Part 15 validation cohort",
            PART15_VAL_COHORT,
        ),
    ]

    for name, path in path_items:
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
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required paths:\n"
            + "\n".join(missing)
        )

    # -------------------------------------------------------------------------
    # PYTORCH
    # -------------------------------------------------------------------------
    section("PYTORCH / GPU ENVIRONMENT")

    gpu_available = torch.cuda.is_available()

    preferred_device = torch.device(
        "cuda:0"
        if gpu_available
        else "cpu"
    )

    print(
        f"PyTorch version                    : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available                     : "
        f"{gpu_available}"
    )

    print(
        f"Preferred device                   : "
        f"{preferred_device}"
    )

    if gpu_available:
        print(
            f"GPU                               : "
            f"{torch.cuda.get_device_name(0)}"
        )

        properties = (
            torch.cuda.get_device_properties(
                0
            )
        )

        print(
            f"GPU memory                        : "
            f"{properties.total_memory / (1024 ** 3):.2f} GB"
        )

    print(
        f"Patch size                         : "
        f"{PATCH_SIZE}"
    )

    print(
        f"Classes                            : "
        f"{NUM_CLASSES}"
    )

    print(
        f"Validation cases                   : "
        f"{VAL_CASES}"
    )

    # -------------------------------------------------------------------------
    # PART 11
    # -------------------------------------------------------------------------
    section("IMPORTING VALIDATED PART 11")

    part11 = import_module(
        PART11_SOURCE,
        "part11_for_part38_corrected",
    )

    print(
        "✓ Corrected Part 11 imported."
    )

    for function_name in (
        "create_model",
        "preprocess_case",
        "load_tensor_case",
        "select_pilot_rows",
    ):
        function = getattr(
            part11,
            function_name,
            None,
        )

        if function is not None:
            print(
                f"{function_name:<36}: "
                f"{inspect.signature(function)}"
            )

    # -------------------------------------------------------------------------
    # EXACT COHORT
    # -------------------------------------------------------------------------
    section(
        "RECONSTRUCTING EXACT PART 15 VALIDATION COHORT"
    )

    cohort, cohort_match = (
        reconstruct_validation_cohort(
            part11
        )
    )

    if len(cohort) != VAL_CASES:
        raise RuntimeError(
            f"Expected {VAL_CASES} validation rows; "
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
    # CHECKPOINT DISCOVERY
    # -------------------------------------------------------------------------
    section("DISCOVERING PART 15 CHECKPOINTS")

    checkpoints = discover_checkpoints()

    print(
        f"Checkpoint files discovered        : "
        f"{len(checkpoints)}"
    )

    for checkpoint in checkpoints:
        print(
            f"  {checkpoint.name}"
        )

    if not checkpoints:
        raise RuntimeError(
            "No .pth checkpoints were found."
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
    # LOAD ALL DATA CASES ON CPU
    # -------------------------------------------------------------------------
    section("LOADING EXACT VALIDATION TENSORS")

    cases = []
    failed_cases = []

    for index, row in cohort.iterrows():
        case_index = index + 1
        study_id, series_id = row_ids(
            row
        )

        try:
            image, target = load_case(
                part11,
                part9,
                row,
            )

            # Keep validation tensors on CPU.
            # They are copied to GPU one case at a time.
            cases.append({
                "case_index": case_index,
                "study_id": study_id,
                "series_id": series_id,
                "image": image.cpu(),
                "target": target.cpu(),
            })

        except Exception as exc:
            failed_cases.append({
                "case_index": case_index,
                "study_id": study_id,
                "series_id": series_id,
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
        print(
            "WARNING: validation tensor failures occurred."
        )

    if not cases:
        raise RuntimeError(
            "No validation cases could be loaded."
        )

    # -------------------------------------------------------------------------
    # EVALUATE ALL CHECKPOINTS
    # -------------------------------------------------------------------------
    section(
        "RUNNING MEMORY-SAFE EPOCH-BY-EPOCH CHECKPOINT EVALUATION"
    )

    checkpoint_records = []
    class_records = []
    case_records = []
    failed_checkpoint_records = []

    for number, checkpoint_path in enumerate(
        checkpoints,
        start=1,
    ):
        print()
        print(
            f"[{number}/{len(checkpoints)}] "
            f"Evaluating {checkpoint_path.name}"
        )

        try:
            record, classes, case_rows = (
                evaluate_checkpoint(
                    part11,
                    checkpoint_path,
                    cases,
                    preferred_device,
                )
            )

            checkpoint_records.append(
                record
            )

            class_records.extend(
                classes
            )

            case_records.extend(
                case_rows
            )

            print(
                f"  ✓ Completed on {record['evaluation_device']}"
            )

            print(
                f"  predicted foreground voxels      : "
                f"{record['prediction_foreground_voxels']:,}"
            )

            print(
                f"  prediction/target ratio          : "
                f"{record['prediction_target_foreground_ratio']:.6f}"
            )

            print(
                f"  background-only cases            : "
                f"{record['background_only_cases']}/"
                f"{record['successful_inference_cases']}"
            )

            print(
                f"  foreground-prediction cases      : "
                f"{record['foreground_prediction_cases']}/"
                f"{record['successful_inference_cases']}"
            )

            print(
                f"  conventional foreground Dice    : "
                f"{record['reconstructed_aggregate_foreground_dice']:.6f}"
            )

            print(
                f"  macro class Dice                : "
                f"{record['reconstructed_macro_class_dice']:.6f}"
            )

        except Exception as exc:
            error_record = {
                "checkpoint": checkpoint_path.name,
                "checkpoint_path": str(
                    checkpoint_path
                ),
                "status": "ERROR",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

            failed_checkpoint_records.append(
                error_record
            )

            print(
                f"  ✗ CHECKPOINT FAILED: {exc}"
            )

        finally:
            cleanup_cuda()

    # -------------------------------------------------------------------------
    # DATAFRAMES
    # -------------------------------------------------------------------------
    checkpoint_df = pd.DataFrame(
        checkpoint_records
    )

    class_df = pd.DataFrame(
        class_records
    )

    case_df = pd.DataFrame(
        case_records
    )

    failed_checkpoint_df = pd.DataFrame(
        failed_checkpoint_records
    )

    # -------------------------------------------------------------------------
    # SAVE OUTPUTS
    # -------------------------------------------------------------------------
    section("SAVING PART 38 CORRECTED OUTPUTS")

    checkpoint_df.to_csv(
        CHECKPOINT_SUMMARY_CSV,
        index=False,
    )

    class_df.to_csv(
        CLASS_SUMMARY_CSV,
        index=False,
    )

    case_df.to_csv(
        CASE_CSV,
        index=False,
    )

    failed_checkpoint_df.to_csv(
        FAILED_CHECKPOINT_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # DIAGNOSIS
    # -------------------------------------------------------------------------
    section("PART 38 CORRECTED OVERALL DIAGNOSIS")

    evaluated = checkpoint_df[
        checkpoint_df["status"] == "OK"
    ].copy()

    evaluated_count = len(
        evaluated
    )

    all_background_count = 0
    foreground_capable_count = 0
    epoch_count = 0
    epoch_all_background = None

    if not evaluated.empty:
        all_background_count = int(
            evaluated[
                "all_background"
            ].astype(bool).sum()
        )

        foreground_capable_count = int(
            (
                evaluated[
                    "foreground_prediction_cases"
                ]
                .fillna(0)
                > 0
            ).sum()
        )

        epoch_df = evaluated[
            evaluated[
                "checkpoint"
            ].str.match(
                r"^epoch_\d+\.pth$"
            )
        ].copy()

        epoch_count = len(
            epoch_df
        )

        if epoch_count > 0:
            epoch_all_background = bool(
                epoch_df[
                    "all_background"
                ].astype(bool).all()
            )

    if evaluated_count == 0:
        diagnosis = (
            "NO_CHECKPOINT_SUCCESSFULLY_EVALUATED"
        )

    elif (
        epoch_count > 0
        and epoch_all_background
    ):
        diagnosis = (
            "FOREGROUND_COLLAPSE_ACROSS_ALL_"
            "EPOCH_CHECKPOINTS"
        )

    elif foreground_capable_count == 0:
        diagnosis = (
            "FOREGROUND_COLLAPSE_ACROSS_"
            "ALL_EVALUATED_CHECKPOINTS"
        )

    elif all_background_count == evaluated_count:
        diagnosis = (
            "ALL_EVALUATED_CHECKPOINTS_BACKGROUND_ONLY"
        )

    else:
        diagnosis = (
            "CHECKPOINT_EVOLUTION_REVEALED_"
            "EPOCH_SPECIFIC_BEHAVIOR"
        )

    print(
        f"Checkpoint files discovered        : "
        f"{len(checkpoints)}"
    )

    print(
        f"Checkpoint files evaluated          : "
        f"{evaluated_count}"
    )

    print(
        f"Checkpoint evaluation failures     : "
        f"{len(failed_checkpoint_records)}"
    )

    print(
        f"Epoch checkpoints evaluated        : "
        f"{epoch_count}"
    )

    print(
        f"All-background checkpoints         : "
        f"{all_background_count}"
    )

    print(
        f"Foreground-capable checkpoints     : "
        f"{foreground_capable_count}"
    )

    print(
        f"Part 15 recorded best Dice         : "
        f"{recorded_best}"
    )

    print(
        f"DIAGNOSIS                           : "
        f"{diagnosis}"
    )

    # -------------------------------------------------------------------------
    # COMPLETE COMPARISON
    # -------------------------------------------------------------------------
    if not evaluated.empty:
        section("COMPLETE CHECKPOINT COMPARISON")

        display_columns = [
            "checkpoint",
            "checkpoint_epoch",
            "evaluation_device",
            "stored_best_val_dice",
            "reconstructed_aggregate_foreground_dice",
            "reconstructed_macro_class_dice",
            "stored_minus_reconstructed_dice",
            "prediction_foreground_voxels",
            "prediction_target_foreground_ratio",
            "background_only_cases",
            "foreground_prediction_cases",
            "unique_prediction_labels",
        ]

        available_columns = [
            c
            for c in display_columns
            if c in evaluated.columns
        ]

        print(
            evaluated[
                available_columns
            ].to_string(
                index=False
            )
        )

        best_idx = evaluated[
            "reconstructed_aggregate_foreground_dice"
        ].idxmax()

        best = evaluated.loc[
            best_idx
        ]

        print()
        print(
            "BEST RECONSTRUCTED CHECKPOINT"
        )

        print(
            f"  checkpoint                       : "
            f"{best['checkpoint']}"
        )

        print(
            f"  epoch                            : "
            f"{best['checkpoint_epoch']}"
        )

        print(
            f"  conventional foreground Dice     : "
            f"{best['reconstructed_aggregate_foreground_dice']:.6f}"
        )

        print(
            f"  predicted foreground voxels      : "
            f"{int(best['prediction_foreground_voxels']):,}"
        )

    # -------------------------------------------------------------------------
    # HISTORICAL INTERPRETATION
    # -------------------------------------------------------------------------
    section("PART 15 HISTORICAL METRIC COMPARISON")

    print(
        f"Part 15 recorded best validation Dice: "
        f"{recorded_best}"
    )

    if not evaluated.empty:
        max_reconstructed = float(
            evaluated[
                "reconstructed_aggregate_foreground_dice"
            ].max()
        )

        print(
            f"Maximum reconstructed conventional foreground Dice: "
            f"{max_reconstructed:.6f}"
        )

        if recorded_best is not None:
            print(
                f"Difference between recorded best and maximum "
                f"reconstructed Dice: "
                f"{recorded_best - max_reconstructed:.6f}"
            )

    if (
        len(failed_checkpoint_records)
        > 0
    ):
        print()
        print(
            "Some checkpoints still failed after the CPU fallback."
        )

        for item in failed_checkpoint_records:
            print(
                f"  {item['checkpoint']}: "
                f"{item['error']}"
            )

    # -------------------------------------------------------------------------
    # SUMMARY JSON
    # -------------------------------------------------------------------------
    summary = {
        "phase": "Phase 4 - Part 38 Corrected",
        "description": (
            "Memory-safe epoch-by-epoch Part 15 checkpoint "
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
        "checkpoint_files_evaluated": evaluated_count,
        "checkpoint_evaluation_failures": len(
            failed_checkpoint_records
        ),
        "epoch_checkpoints_evaluated": epoch_count,
        "all_background_checkpoints": (
            all_background_count
        ),
        "foreground_capable_checkpoints": (
            foreground_capable_count
        ),
        "part15_recorded_best_dice": safe_float(
            recorded_best
        ),
        "diagnosis": diagnosis,
        "gpu_available": gpu_available,
        "preferred_device": str(
            preferred_device
        ),
        "cpu_fallback_enabled": True,
        "one_checkpoint_at_a_time": True,
        "explicit_cuda_cleanup": True,
        "training_performed": False,
        "model_weights_modified": False,
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "spider_used": False,
        "rsna_test_set_used": False,
        "failed_tensor_cases": failed_cases,
        "failed_checkpoint_records": failed_checkpoint_records,
        "checkpoint_results": checkpoint_records,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
            default=str,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------
    report = []

    report.append(
        "PHASE 4 - PART 38 CORRECTED"
    )

    report.append(
        "RSNA-ONLY EPOCH-BY-EPOCH CHECKPOINT "
        "PREDICTION FORENSIC AUDIT"
    )

    report.append(
        "=" * 78
    )

    report.append(
        ""
    )

    report.append(
        f"Validation cases requested: {VAL_CASES}"
    )

    report.append(
        f"Validation cases loaded: {len(cases)}"
    )

    report.append(
        f"Validation tensor failures: {len(failed_cases)}"
    )

    report.append(
        f"Exact cohort match: {cohort_match}"
    )

    report.append(
        f"Checkpoint files discovered: {len(checkpoints)}"
    )

    report.append(
        f"Checkpoint files evaluated: {evaluated_count}"
    )

    report.append(
        f"Checkpoint failures: {len(failed_checkpoint_records)}"
    )

    report.append(
        f"Epoch checkpoints evaluated: {epoch_count}"
    )

    report.append(
        f"All-background checkpoints: {all_background_count}"
    )

    report.append(
        f"Foreground-capable checkpoints: "
        f"{foreground_capable_count}"
    )

    report.append(
        f"Part 15 recorded best Dice: {recorded_best}"
    )

    report.append(
        f"DIAGNOSIS: {diagnosis}"
    )

    report.append(
        ""
    )

    report.append(
        "CHECKPOINT COMPARISON"
    )

    report.append(
        "-" * 78
    )

    if not evaluated.empty:
        report.append(
            evaluated.to_string(
                index=False
            )
        )
    else:
        report.append(
            "No checkpoint successfully evaluated."
        )

    report.append(
        ""
    )

    report.append(
        "FAILED CHECKPOINTS"
    )

    report.append(
        "-" * 78
    )

    if failed_checkpoint_records:
        report.append(
            json.dumps(
                failed_checkpoint_records,
                indent=2,
                default=str,
            )
        )
    else:
        report.append(
            "None."
        )

    report.append(
        ""
    )

    report.append(
        "No training performed."
    )

    report.append(
        "No model weights modified."
    )

    report.append(
        "No optimizer created."
    )

    report.append(
        "No optimizer step performed."
    )

    report.append(
        "SPIDER not used."
    )

    report.append(
        "RSNA test set not used."
    )

    REPORT_TXT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------------------------------------
    section("PART 38 CORRECTED OUTPUTS")

    print(
        f"Checkpoint summary CSV              : "
        f"{CHECKPOINT_SUMMARY_CSV}"
    )

    print(
        f"Class summary CSV                   : "
        f"{CLASS_SUMMARY_CSV}"
    )

    print(
        f"Case prediction CSV                 : "
        f"{CASE_CSV}"
    )

    print(
        f"Failed checkpoints CSV              : "
        f"{FAILED_CHECKPOINT_CSV}"
    )

    print(
        f"Summary JSON                        : "
        f"{SUMMARY_JSON}"
    )

    print(
        f"Report TXT                          : "
        f"{REPORT_TXT}"
    )

    section("PHASE 4 - PART 38 CORRECTED COMPLETE")

    print(
        "✓ One checkpoint evaluated at a time."
    )

    print(
        "✓ CPU checkpoint loading used."
    )

    print(
        "✓ CUDA cache explicitly released."
    )

    print(
        "✓ GPU OOM CPU fallback enabled."
    )

    print(
        "✓ Exact validation cohort reconstructed."
    )

    print(
        "✓ Same validation tensors reused."
    )

    print(
        "✓ All available checkpoints attempted."
    )

    print(
        "✓ Conventional foreground Dice calculated."
    )

    print(
        "✓ Per-class TP / FP / FN calculated."
    )

    print(
        "✓ Historical Part 15 Dice compared."
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


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("=" * 78)
        print("PART 38 CORRECTED ERROR")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
