"""
PHASE 4 - PART 14
RSNA-ONLY EXACT PART 11 VALIDATION REVALIDATION

Corrected to use the actual public API of:
segmentation_rsna_part11_controlled_pilot_training_corrected.py

No training is performed. No weights are modified.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES = RSNA_ROOT / "train_images"

MANIFEST_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
)

TRAIN_MANIFEST = MANIFEST_DIR / "rsna_part8_train_manifest.csv"
VAL_MANIFEST = MANIFEST_DIR / "rsna_part8_validation_manifest.csv"

# IMPORTANT: use the corrected Part 11 implementation.
PART11_SOURCE = (
    SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

PART11_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part11_controlled_pilot_training"
)

CHECKPOINT = PART11_DIR / "checkpoints" / "best_model.pth"

PART13_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part13_reproducibility_audit"
)

PART13_EXACT_CSV = PART13_DIR / "part13_exact_part11_reproduction.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part14_exact_validation_revalidation"
)

REPORT_DIR = OUTPUT_DIR / "reports"


# ============================================================================
# LOCKED CONFIGURATION - SAME AS CORRECTED PART 11
# ============================================================================

PATCH_SIZE = (64, 96, 96)
IN_CHANNELS = 1
NUM_CLASSES = 6
FEATURE_SIZE = 12
BATCH_SIZE = 1
PILOT_TRAIN_CASES = 100
PILOT_VAL_CASES = 30
SEED = 42
USE_AMP = True

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ============================================================================
# HELPERS
# ============================================================================

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def require_paths() -> None:
    section("PATH VALIDATION")

    required = {
        "RSNA root": RSNA_ROOT,
        "train_images": TRAIN_IMAGES,
        "train manifest": TRAIN_MANIFEST,
        "validation manifest": VAL_MANIFEST,
        "CORRECTED Part 11 source": PART11_SOURCE,
        "Part 11 checkpoint": CHECKPOINT,
        "Part 13 exact reproduction": PART13_EXACT_CSV,
    }

    missing = []

    for label, path in required.items():
        state = "FOUND" if path.exists() else "MISSING"
        print(f"{label:<34}: {state}")
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required Part 14 input(s):\n" + "\n".join(missing)
        )


# ============================================================================
# IMPORT CORRECTED PART 11
# ============================================================================

def import_corrected_part11():
    section("IMPORTING CORRECTED PART 11 IMPLEMENTATION")

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    spec = importlib.util.spec_from_file_location(
        "part11_corrected_for_part14",
        PART11_SOURCE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {PART11_SOURCE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    print("✓ Corrected Part 11 imported without starting training.")

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
                f"Corrected Part 11 does not expose required API: {name}"
            )
        print(f"✓ {name} available.")

    return module


# ============================================================================
# COHORT
# ============================================================================

def reconstruct_exact_cohort(part11) -> pd.DataFrame:
    section("RECONSTRUCTING EXACT PART 11 PILOT COHORT")

    train_rows = part11.select_pilot_rows(
        TRAIN_MANIFEST,
        PILOT_TRAIN_CASES,
        SEED,
    )

    val_rows = part11.select_pilot_rows(
        VAL_MANIFEST,
        PILOT_VAL_CASES,
        SEED,
    )

    print(f"Train manifest total : {len(pd.read_csv(TRAIN_MANIFEST))}")
    print(f"Validation total     : {len(pd.read_csv(VAL_MANIFEST))}")
    print(f"Reconstructed train  : {len(train_rows)}")
    print(f"Reconstructed val    : {len(val_rows)}")

    return val_rows


# ============================================================================
# CHECKPOINT
# ============================================================================

def load_checkpoint() -> Dict[str, Any]:
    section("CHECKPOINT METADATA AUDIT")

    payload = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise RuntimeError("Checkpoint is not a dictionary.")

    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        state_dict = payload.get("state_dict")

    if state_dict is None:
        raise RuntimeError(
            "Checkpoint has no model_state_dict/state_dict."
        )

    print(f"Checkpoint epoch          : {payload.get('epoch')}")
    print(
        "Checkpoint best_val_dice  : "
        f"{payload.get('best_val_dice')}"
    )
    print(f"Checkpoint parameter tensors : {len(state_dict)}")

    history = payload.get("history")
    if isinstance(history, list):
        print(f"Embedded history rows     : {len(history)}")

    return {
        "payload": payload,
        "state_dict": state_dict,
    }


# ============================================================================
# EXACT VALIDATION
# ============================================================================

@torch.no_grad()
def evaluate_exact_part11(
    part11,
    part9,
    model,
    val_rows: pd.DataFrame,
    device: torch.device,
):
    section("EXACT PART 11 VALIDATION REVALIDATION")

    model.eval()

    # Use the exact same loss construction as corrected Part 11.
    loss_function = part11.DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    case_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []

    losses = []
    dices = []
    class_dices = []

    fallback_count = 0

    for step, (_, row) in enumerate(
        val_rows.iterrows(),
        start=1,
    ):
        study_id = str(row["study_id"])
        series_id = str(row["series_id"])

        t0 = time.time()

        image, mask, info = part11.load_tensor_case(
            row,
            part9,
        )

        if info.get("part11_loader") == "local_robust_fallback":
            fallback_count += 1

        image = image.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        if tuple(image.shape[1:]) != (1, *PATCH_SIZE):
            raise RuntimeError(
                f"Unexpected image shape for "
                f"{study_id}/{series_id}: {tuple(image.shape)}"
            )

        if tuple(mask.shape[1:]) != PATCH_SIZE:
            raise RuntimeError(
                f"Unexpected mask shape for "
                f"{study_id}/{series_id}: {tuple(mask.shape)}"
            )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(USE_AMP and device.type == "cuda"),
        ):
            logits = model(image)

            loss = loss_function(
                logits,
                mask.unsqueeze(1),
            )

        # THIS IS THE EXACT CORRECTED PART 11 METRIC FUNCTION.
        mean_dice, per_class = part11.dice_from_prediction(
            logits,
            mask,
        )

        losses.append(float(loss.item()))
        dices.append(float(mean_dice))
        class_dices.append(per_class)

        case_rows.append(
            {
                "index": step,
                "study_id": study_id,
                "series_id": series_id,
                "series_description": info.get(
                    "series_description",
                    row.get("series_description", ""),
                ),
                "loss": float(loss.item()),
                "mean_foreground_dice": float(mean_dice),
                "foreground_voxels": int((mask > 0).sum().item()),
                "image_shape": str(tuple(image.shape[1:])),
                "mask_shape": str(tuple(mask.shape[1:])),
                "loader_mode": info.get("part11_loader", ""),
                "elapsed_seconds": float(time.time() - t0),
                "status": "PASS",
            }
        )

        class_row = {
            "index": step,
            "study_id": study_id,
            "series_id": series_id,
        }

        for class_id, value in enumerate(per_class, start=1):
            class_row[f"dice_class_{class_id}"] = float(value)

        class_rows.append(class_row)

        del image, mask, logits, loss

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"  [{step:02d}/{len(val_rows)}] "
            f"{study_id} | {series_id} | "
            f"Loss={losses[-1]:.4f} "
            f"Dice={dices[-1]:.6f}"
        )

    return (
        pd.DataFrame(case_rows),
        pd.DataFrame(class_rows),
        fallback_count,
    )


# ============================================================================
# PART 13 COMPARISON
# ============================================================================

def compare_with_part13(
    part14_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:

    section("PART 14 ↔ PART 13 EXACT REPRODUCTION COMPARISON")

    p13 = pd.read_csv(PART13_EXACT_CSV)

    def make_key(df):
        return (
            df["study_id"].astype(str)
            + "::"
            + df["series_id"].astype(str)
        )

    p13["case_key"] = make_key(p13)
    part14_df["case_key"] = make_key(part14_df)

    # Part 13 exact reproduction files generated by the audit use
    # mean_foreground_dice as the per-case exact metric.
    candidate_cols = [
        "mean_foreground_dice",
        "dice",
        "mean_dice",
    ]

    p13_dice_col = next(
        (c for c in candidate_cols if c in p13.columns),
        None,
    )

    if p13_dice_col is None:
        raise RuntimeError(
            "Could not identify the Part 13 exact Dice column. "
            f"Available columns: {list(p13.columns)}"
        )

    common = p13.merge(
        part14_df,
        on="case_key",
        how="inner",
        suffixes=("_part13", "_part14"),
    )

    if common.empty:
        raise RuntimeError(
            "No common validation cases between Part 13 and Part 14."
        )

    common["dice_part13"] = pd.to_numeric(
        common[f"{p13_dice_col}_part13"],
        errors="coerce",
    )

    common["dice_part14"] = pd.to_numeric(
        common["mean_foreground_dice_part14"],
        errors="coerce",
    )

    common["difference"] = (
        common["dice_part14"] - common["dice_part13"]
    )

    common["absolute_difference"] = common["difference"].abs()

    summary = {
        "common_cases": int(len(common)),
        "part13_mean_dice": float(common["dice_part13"].mean()),
        "part14_mean_dice": float(common["dice_part14"].mean()),
        "mean_absolute_difference": float(
            common["absolute_difference"].mean()
        ),
        "max_absolute_difference": float(
            common["absolute_difference"].max()
        ),
    }

    print(f"Common cases                  : {summary['common_cases']}")
    print(
        f"Mean Part 13 exact Dice      : "
        f"{summary['part13_mean_dice']:.6f}"
    )
    print(
        f"Mean Part 14 exact Dice      : "
        f"{summary['part14_mean_dice']:.6f}"
    )
    print(
        f"Mean absolute Dice difference: "
        f"{summary['mean_absolute_difference']:.10f}"
    )
    print(
        f"Max absolute Dice difference : "
        f"{summary['max_absolute_difference']:.10f}"
    )

    return common, summary


# ============================================================================
# OUTPUTS
# ============================================================================

def save_outputs(
    case_df: pd.DataFrame,
    class_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    comparison_summary: Dict[str, float],
    checkpoint_info: Dict[str, Any],
    peak_allocated: float,
    peak_reserved: float,
):
    section("SAVING PART 14 RESULTS")

    case_path = OUTPUT_DIR / "part14_exact_validation_case_metrics.csv"
    class_path = OUTPUT_DIR / "part14_exact_validation_per_class_metrics.csv"
    comparison_path = OUTPUT_DIR / "part14_vs_part13_exact_comparison.csv"

    case_df.to_csv(case_path, index=False)
    class_df.to_csv(class_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    class_summary = []

    for class_id in range(1, NUM_CLASSES):
        col = f"dice_class_{class_id}"
        values = pd.to_numeric(
            class_df[col],
            errors="coerce",
        ).dropna()

        class_summary.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "mean_dice": float(values.mean()),
                "median_dice": float(values.median()),
                "cases": int(len(values)),
            }
        )

    class_summary_df = pd.DataFrame(class_summary)

    class_summary_path = (
        OUTPUT_DIR / "part14_per_class_summary.csv"
    )
    class_summary_df.to_csv(
        class_summary_path,
        index=False,
    )

    valid = pd.to_numeric(
        case_df["mean_foreground_dice"],
        errors="coerce",
    ).dropna()

    mean_dice = float(valid.mean())
    median_dice = float(valid.median())

    # Internal reproducibility tolerance.
    exact_match = (
        comparison_summary["mean_absolute_difference"] <= 1e-5
        and comparison_summary["max_absolute_difference"] <= 1e-5
    )

    summary = {
        "phase": "Phase 4 - Part 14",
        "description": "Exact Part 11 validation revalidation",
        "dataset": "RSNA only",
        "spider_used": False,
        "test_set_used": False,
        "training_performed": False,
        "model_weights_modified": False,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": checkpoint_info["payload"].get("epoch"),
        "checkpoint_best_val_dice": checkpoint_info["payload"].get(
            "best_val_dice"
        ),
        "seed": SEED,
        "patch_size": list(PATCH_SIZE),
        "validation_cases": int(len(case_df)),
        "mean_validation_loss": float(case_df["loss"].mean()),
        "mean_validation_dice": mean_dice,
        "median_validation_dice": median_dice,
        "peak_gpu_allocated_gb": peak_allocated,
        "peak_gpu_reserved_gb": peak_reserved,
        "part13_comparison": comparison_summary,
        "exact_reproduction_pass": exact_match,
        "warning": (
            "Dice is measured against RSNA point-derived pseudo-masks, "
            "not manually delineated clinical ground truth."
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "phase4_part14_exact_validation_revalidation_summary.json"
    )

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report_path = (
        REPORT_DIR
        / "phase4_part14_exact_validation_revalidation_report.txt"
    )

    lines = [
        "=" * 78,
        "PHASE 4 - PART 14",
        "RSNA-ONLY EXACT PART 11 VALIDATION REVALIDATION",
        "=" * 78,
        "",
        f"Checkpoint: {CHECKPOINT}",
        f"Checkpoint epoch: {summary['checkpoint_epoch']}",
        f"Checkpoint best Dice: {summary['checkpoint_best_val_dice']}",
        "",
        "RESULTS",
        "-" * 78,
        f"Validation cases: {len(case_df)}",
        f"Mean validation loss: {summary['mean_validation_loss']:.8f}",
        f"Mean validation Dice: {mean_dice:.8f}",
        f"Median validation Dice: {median_dice:.8f}",
        "",
        "PART 13 COMPARISON",
        "-" * 78,
        f"Common cases: {comparison_summary['common_cases']}",
        f"Part 13 mean Dice: {comparison_summary['part13_mean_dice']:.8f}",
        f"Part 14 mean Dice: {comparison_summary['part14_mean_dice']:.8f}",
        (
            "Mean absolute difference: "
            f"{comparison_summary['mean_absolute_difference']:.10f}"
        ),
        (
            "Max absolute difference: "
            f"{comparison_summary['max_absolute_difference']:.10f}"
        ),
        "",
        "EXPERIMENT STATUS",
        "-" * 78,
        "RSNA used: YES",
        "SPIDER used: NO",
        "Test set used: NO",
        "Training performed: NO",
        "Model weights modified: NO",
        "",
        "SCIENTIFIC WARNING",
        "-" * 78,
        "These are pseudo-mask validation metrics.",
        "They are not final clinical segmentation metrics against manual masks.",
        "",
        "FINAL DECISION",
        "-" * 78,
    ]

    if exact_match:
        lines.extend(
            [
                "PASS - Part 14 reproduces the Part 13 exact validation result.",
                "The Part 11 evaluation pipeline is internally reproducible.",
            ]
        )
    else:
        lines.extend(
            [
                "ACTION REQUIRED - Part 14 still differs from Part 13.",
                "Do NOT use the Part 14 value as the final performance number.",
            ]
        )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved: {case_path}")
    print(f"Saved: {class_path}")
    print(f"Saved: {class_summary_path}")
    print(f"Saved: {comparison_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {report_path}")

    return exact_match, summary, class_summary_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    section(
        "PHASE 4 - PART 14\n"
        "RSNA-ONLY EXACT PART 11 VALIDATION REVALIDATION"
    )

    print(
        "\nPurpose:\n"
        "Revalidate the Part 11 best checkpoint using the actual "
        "corrected Part 11 implementation, exact pilot cohort, "
        "preprocessing and Dice metric."
    )

    print("\nPROJECT ROOT")
    print(PROJECT_ROOT)

    print("\nRSNA DATASET")
    print(RSNA_ROOT)

    print("\nCORRECTED PART 11 SOURCE")
    print(PART11_SOURCE)

    print("\nPART 11 CHECKPOINT")
    print(CHECKPOINT)

    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    ensure_dirs()
    require_paths()
    set_seed()

    section("PYTORCH / GPU ENVIRONMENT")

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        print(f"Device          : {props.name}")
        print(
            f"GPU memory      : "
            f"{props.total_memory / (1024 ** 3):.2f} GB"
        )
    else:
        device = torch.device("cpu")
        print("Device          : CPU")

    print(f"Patch size      : {PATCH_SIZE}")
    print(f"Classes         : {NUM_CLASSES}")
    print(f"AMP             : {USE_AMP}")

    part11 = import_corrected_part11()

    # Load the exact Part 9 loader through Part 11's own function.
    section("LOADING PART 9 THROUGH CORRECTED PART 11")

    part9 = part11.load_part9_module()

    print("✓ Part 9 loader imported.")
    print("✓ Corrected Part 11 local robust DICOM fallback retained.")

    val_rows = reconstruct_exact_cohort(part11)

    checkpoint_info = load_checkpoint()

    section("CREATING SWIN-UNETR")

    model = part11.create_model(device)

    total_params = sum(p.numel() for p in model.parameters())

    print("✓ Swin-UNETR created.")
    print(f"Total parameters : {total_params:,}")

    missing, unexpected = model.load_state_dict(
        checkpoint_info["state_dict"],
        strict=False,
    )

    print(f"Missing keys     : {len(missing)}")
    print(f"Unexpected keys  : {len(unexpected)}")

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"Missing={missing[:10]}, Unexpected={unexpected[:10]}"
        )

    model.eval()

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    case_df, class_df, fallback_count = evaluate_exact_part11(
        part11,
        part9,
        model,
        val_rows,
        device,
    )

    if torch.cuda.is_available():
        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )
    else:
        peak_allocated = 0.0
        peak_reserved = 0.0

    comparison_df, comparison_summary = compare_with_part13(
        case_df
    )

    section("PART 14 FINAL SUMMARY")

    print(f"Validation cases       : {len(case_df)}")
    print(
        f"Mean validation loss   : "
        f"{case_df['loss'].mean():.6f}"
    )
    print(
        f"Mean validation Dice   : "
        f"{case_df['mean_foreground_dice'].mean():.6f}"
    )
    print(
        f"Median validation Dice : "
        f"{case_df['mean_foreground_dice'].median():.6f}"
    )

    print("\nPer-class Dice:")

    for class_id in range(1, NUM_CLASSES):
        col = f"dice_class_{class_id}"
        print(
            f"{CLASS_NAMES[class_id]:35} : "
            f"{class_df[col].mean():.6f}"
        )

    print()
    print(
        f"Part 13 mean Dice      : "
        f"{comparison_summary['part13_mean_dice']:.6f}"
    )
    print(
        f"Part 14 mean Dice      : "
        f"{comparison_summary['part14_mean_dice']:.6f}"
    )
    print(
        f"Mean absolute diff     : "
        f"{comparison_summary['mean_absolute_difference']:.10f}"
    )
    print(
        f"Max absolute diff      : "
        f"{comparison_summary['max_absolute_difference']:.10f}"
    )
    print(f"Part 11 robust fallbacks used : {fallback_count}")
    print(f"Peak GPU allocated     : {peak_allocated:.3f} GB")
    print(f"Peak GPU reserved      : {peak_reserved:.3f} GB")

    exact_match, _, _ = save_outputs(
        case_df,
        class_df,
        comparison_df,
        comparison_summary,
        checkpoint_info,
        peak_allocated,
        peak_reserved,
    )

    section("PART 14 DECISION")

    if exact_match:
        print(
            "PASS - Part 14 exactly reproduces the Part 13 "
            "exact Part 11 validation result."
        )
        print(
            "The Part 11 evaluation pipeline is internally reproducible."
        )
    else:
        print(
            "ACTION REQUIRED - Part 14 still differs from Part 13."
        )
        print(
            "Do NOT use the Part 14 Dice as the final performance number."
        )

    print()
    print("SPIDER used          : NO")
    print("Test set used        : NO")
    print("Training performed   : NO")
    print("Model weights changed: NO")
    print()
    print(
        "IMPORTANT: Dice is against RSNA point-derived pseudo-masks, "
        "not manually delineated clinical segmentation ground truth."
    )

    section("PHASE 4 - PART 14 COMPLETE")


if __name__ == "__main__":
    main()
