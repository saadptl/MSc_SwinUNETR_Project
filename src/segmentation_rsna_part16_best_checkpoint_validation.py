"""
==============================================================================
PHASE 4 - PART 16
RSNA-ONLY PART 15 BEST CHECKPOINT VALIDATION
==============================================================================

Purpose:
    Independently validate the best checkpoint produced by Part 15.

Scientific role:
    - Load the Part 15 best checkpoint.
    - Reconstruct the exact Part 15 validation cohort.
    - Reuse the validated Part 11 preprocessing / Part 9 loading pathway.
    - Recalculate validation loss and Dice.
    - Calculate per-class Dice.
    - Compare the reproduced result against the Part 15 recorded result.

Rules:
    - RSNA only.
    - SPIDER is NOT used.
    - Test set is NOT used.
    - No training.
    - No optimizer.
    - No weight updates.
    - No checkpoint modification.
    - Metrics are against point-derived pseudo-masks.

Expected Part 15 configuration:
    Patch size  : (64, 96, 96)
    Feature size: 12
    Classes     : 6
    Validation  : 100 cases
    Seed        : 42
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

RSNA_DIR = (
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
    SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

# Part 15 v2 may have produced best_model.pth here.
PART15_CHECKPOINT_CANDIDATES = [
    PART15_DIR / "checkpoints" / "best_model.pth",
    PART15_DIR / "checkpoints" / "part15_best_model.pth",
]

PART15_HISTORY_CSV = PART15_DIR / "part15_training_history.csv"
PART15_VAL_COHORT = PART15_DIR / "part15_validation_cohort.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part16_best_checkpoint_validation"
)

REPORT_DIR = OUTPUT_DIR / "reports"


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

VAL_CASES = 100

PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12
NUM_CLASSES = 6

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
# UTILITIES
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


def banner(text: str) -> None:
    print("=" * 78)
    print(text)
    print("=" * 78)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "None"

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


# =============================================================================
# CHECKPOINT DISCOVERY
# =============================================================================

def find_part15_checkpoint() -> Path:
    for path in PART15_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path

    # Fallback: search recursively for likely Part 15 checkpoints.
    candidates = sorted(
        PART15_DIR.rglob("*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    preferred = [
        p
        for p in candidates
        if "best" in p.name.lower()
    ]

    if preferred:
        return preferred[0]

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No Part 15 checkpoint found under:\n"
        f"{PART15_DIR}"
    )


# =============================================================================
# PATH VALIDATION
# =============================================================================

def require_paths(part15_checkpoint: Path) -> None:
    paths = {
        "RSNA root": RSNA_DIR,
        "train_images": RSNA_DIR / "train_images",
        "train manifest": TRAIN_MANIFEST,
        "validation manifest": VAL_MANIFEST,
        "Part 11 corrected source": PART11_SOURCE,
        "Part 15 directory": PART15_DIR,
        "Part 15 checkpoint": part15_checkpoint,
    }

    missing = [
        f"{name}: {path}"
        for name, path in paths.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required Part 16 input(s):\n"
            + "\n".join(missing)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# IMPORT PART 11
# =============================================================================

def import_part11():
    spec = importlib.util.spec_from_file_location(
        "part11_for_part16",
        PART11_SOURCE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import corrected Part 11:\n{PART11_SOURCE}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["part11_for_part16"] = module
    spec.loader.exec_module(module)

    required = [
        "select_pilot_rows",
        "load_part9_module",
        "load_tensor_case",
        "create_model",
        "dice_from_prediction",
    ]

    missing = [
        name for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise AttributeError(
            "Corrected Part 11 implementation is missing required API: "
            + ", ".join(missing)
        )

    return module


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state = None

    if isinstance(ckpt, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            if isinstance(ckpt.get(key), dict):
                state = ckpt[key]
                break

    if state is None:
        if (
            isinstance(ckpt, dict)
            and ckpt
            and all(isinstance(v, torch.Tensor) for v in ckpt.values())
        ):
            state = ckpt
        else:
            raise RuntimeError(
                "Could not locate model state dictionary in Part 15 checkpoint."
            )

    cleaned = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }

    result = model.load_state_dict(
        cleaned,
        strict=False,
    )

    print(f"Missing keys     : {len(result.missing_keys)}")
    print(f"Unexpected keys  : {len(result.unexpected_keys)}")

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Part 15 checkpoint architecture mismatch."
        )

    metadata = {
        "checkpoint_epoch": (
            ckpt.get("epoch")
            if isinstance(ckpt, dict)
            else None
        ),
        "checkpoint_best_val_dice": (
            ckpt.get("best_val_dice")
            if isinstance(ckpt, dict)
            else None
        ),
        "checkpoint_seed": (
            ckpt.get("seed")
            if isinstance(ckpt, dict)
            else None
        ),
        "checkpoint_patch_size": (
            ckpt.get("patch_size")
            if isinstance(ckpt, dict)
            else None
        ),
        "checkpoint_feature_size": (
            ckpt.get("feature_size")
            if isinstance(ckpt, dict)
            else None
        ),
        "checkpoint_num_classes": (
            ckpt.get("num_classes")
            if isinstance(ckpt, dict)
            else None
        ),
        "checkpoint_train_cases": (
            ckpt.get("train_cases")
            if isinstance(ckpt, dict)
            else None
        ),
        "checkpoint_validation_cases": (
            ckpt.get("validation_cases")
            if isinstance(ckpt, dict)
            else None
        ),
        "history_rows": (
            len(ckpt.get("history", []))
            if isinstance(ckpt, dict)
            and isinstance(ckpt.get("history"), list)
            else None
        ),
    }

    return metadata


# =============================================================================
# TENSOR NORMALIZATION
# =============================================================================

def normalize_3d_case(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    image = torch.as_tensor(image)
    mask = torch.as_tensor(mask)

    # Remove ONLY singleton dimensions.
    while image.ndim > 3:
        singleton = [
            i for i, size in enumerate(image.shape)
            if size == 1
        ]

        if not singleton:
            raise RuntimeError(
                "Part 16 expected 3-D image [D,H,W], got "
                f"{tuple(image.shape)}"
            )

        image = image.squeeze(singleton[0])

    while mask.ndim > 3:
        singleton = [
            i for i, size in enumerate(mask.shape)
            if size == 1
        ]

        if not singleton:
            raise RuntimeError(
                "Part 16 expected 3-D mask [D,H,W], got "
                f"{tuple(mask.shape)}"
            )

        mask = mask.squeeze(singleton[0])

    if image.ndim != 3:
        raise RuntimeError(
            f"Image must be [D,H,W], got {tuple(image.shape)}"
        )

    if mask.ndim != 3:
        raise RuntimeError(
            f"Mask must be [D,H,W], got {tuple(mask.shape)}"
        )

    if tuple(image.shape) != tuple(mask.shape):
        raise RuntimeError(
            "Image/mask spatial mismatch: "
            f"image={tuple(image.shape)}, "
            f"mask={tuple(mask.shape)}"
        )

    return (
        image.float().contiguous(),
        mask.long().contiguous(),
    )


# =============================================================================
# SAFE CASE LOADING
# =============================================================================

def safe_case(
    part11,
    part9,
    row,
):
    """
    Reuse the exact Part 11 loading/preprocessing API.

    The corrected Part 15 implementation calls:

        part11.load_tensor_case(row, part9)

    and then enforces [D,H,W] before batching.
    """

    image, mask, info = part11.load_tensor_case(
        row,
        part9,
    )

    image, mask = normalize_3d_case(
        image,
        mask,
    )

    # Part 15 safe_case returns [1,D,H,W] before final batching.
    image = image.unsqueeze(0)

    return image, mask, info


# =============================================================================
# VALIDATION
# =============================================================================

def evaluate(
    model,
    rows: pd.DataFrame,
    part11,
    part9,
    loss_fn,
    device: torch.device,
    amp_enabled: bool,
):
    model.eval()

    losses = []
    dices = []

    class_acc = {
        c: []
        for c in range(1, NUM_CLASSES)
    }

    fallback_count = 0

    case_records = []

    total = len(rows)

    with torch.no_grad():

        for i, (_, row) in enumerate(
            rows.iterrows(),
            start=1,
        ):

            image, mask, info = safe_case(
                part11,
                part9,
                row,
            )

            if info.get("part11_loader") == "local_robust_fallback":
                fallback_count += 1

            # safe_case returns [1,D,H,W]
            # Add batch dimension -> [B,1,D,H,W]
            image = image.unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            mask_d = mask.unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            if image.ndim != 5 or image.shape[1] != 1:
                raise RuntimeError(
                    "Part 16 model input must be [B,1,D,H,W], got "
                    f"{tuple(image.shape)}"
                )

            if mask_d.ndim != 4:
                raise RuntimeError(
                    "Part 16 mask must be [B,D,H,W], got "
                    f"{tuple(mask_d.shape)}"
                )

            start = time.perf_counter()

            with torch.autocast(
                device_type="cuda",
                enabled=(
                    amp_enabled
                    and device.type == "cuda"
                ),
            ):
                logits = model(image)

                loss = loss_fn(
                    logits,
                    mask_d.unsqueeze(1),
                )

            elapsed = time.perf_counter() - start

            # IMPORTANT:
            # Correct Part 11 API:
            # dice_from_prediction(logits, target)
            mean_dice, class_dice = (
                part11.dice_from_prediction(
                    logits,
                    mask_d,
                )
            )

            mean_dice = float(mean_dice)

            losses.append(
                float(loss.detach().cpu())
            )

            dices.append(mean_dice)

            for c in range(1, NUM_CLASSES):
                class_value = float(
                    class_dice[c - 1]
                )

                class_acc[c].append(
                    class_value
                )

            study_id = str(
                row.get(
                    "study_id",
                    info.get("study_id", ""),
                )
            )

            series_id = str(
                row.get(
                    "series_id",
                    info.get("series_id", ""),
                )
            )

            series_description = str(
                row.get(
                    "series_description",
                    info.get(
                        "series_description",
                        "",
                    ),
                )
            )

            case_records.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "series_description": series_description,
                    "loss": float(
                        loss.detach().cpu()
                    ),
                    "mean_foreground_dice": mean_dice,
                    "spinal_canal_stenosis_dice": float(
                        class_dice[0]
                    ),
                    "left_neural_foraminal_narrowing_dice": float(
                        class_dice[1]
                    ),
                    "right_neural_foraminal_narrowing_dice": float(
                        class_dice[2]
                    ),
                    "left_subarticular_stenosis_dice": float(
                        class_dice[3]
                    ),
                    "right_subarticular_stenosis_dice": float(
                        class_dice[4]
                    ),
                    "inference_seconds": elapsed,
                    "loader_fallback": (
                        info.get("part11_loader")
                        == "local_robust_fallback"
                    ),
                }
            )

            if (
                i <= 5
                or i % 25 == 0
                or i == total
            ):
                print(
                    f"  [{i:03d}/{total}] "
                    f"{study_id} | {series_id} | "
                    f"Loss={float(loss.detach().cpu()):.4f} "
                    f"Dice={mean_dice:.6f}"
                )

    summary = {
        "validation_cases": total,
        "mean_validation_loss": float(
            np.mean(losses)
        ),
        "mean_validation_dice": float(
            np.mean(dices)
        ),
        "median_validation_dice": float(
            np.median(dices)
        ),
        "std_validation_dice": float(
            np.std(dices)
        ),
        "min_validation_dice": float(
            np.min(dices)
        ),
        "max_validation_dice": float(
            np.max(dices)
        ),
        "per_class_dice": {
            str(c): float(
                np.mean(class_acc[c])
            )
            for c in range(1, NUM_CLASSES)
        },
        "fallback_count": fallback_count,
        "case_metrics": case_records,
    }

    return summary


# =============================================================================
# PART 15 RECORDED RESULT
# =============================================================================

def load_recorded_part15_result() -> Dict[str, Any]:

    result = {
        "source": None,
        "best_val_dice": None,
        "final_val_dice": None,
        "best_epoch": None,
    }

    if PART15_HISTORY_CSV.exists():

        try:
            df = pd.read_csv(
                PART15_HISTORY_CSV
            )

            result["source"] = str(
                PART15_HISTORY_CSV
            )

            if "val_dice" in df.columns:

                values = pd.to_numeric(
                    df["val_dice"],
                    errors="coerce",
                ).dropna()

                if len(values):

                    best_idx = values.idxmax()

                    result["best_val_dice"] = (
                        float(values.loc[best_idx])
                    )

                    result["final_val_dice"] = (
                        float(values.iloc[-1])
                    )

                    if "epoch" in df.columns:
                        result["best_epoch"] = int(
                            df.loc[
                                best_idx,
                                "epoch",
                            ]
                        )

                    return result

        except Exception as exc:
            print(
                f"Warning: could not read Part 15 history: {exc}"
            )

    return result


# =============================================================================
# COHORT RECONSTRUCTION
# =============================================================================

def reconstruct_validation_cohort(
    part11,
) -> pd.DataFrame:

    manifest = pd.read_csv(
        VAL_MANIFEST
    )

    # If Part 15 saved its exact validation cohort,
    # use it and verify against deterministic reconstruction.
    if PART15_VAL_COHORT.exists():

        saved = pd.read_csv(
            PART15_VAL_COHORT
        )

        reconstructed = part11.select_pilot_rows(
            VAL_MANIFEST,
            VAL_CASES,
            SEED,
        ).reset_index(drop=True)

        if len(saved) != len(reconstructed):
            raise RuntimeError(
                "Part 15 saved validation cohort size differs "
                "from deterministic reconstruction."
            )

        key_cols = [
            c
            for c in [
                "study_id",
                "series_id",
            ]
            if c in saved.columns
            and c in reconstructed.columns
        ]

        if key_cols:

            saved_keys = list(
                zip(
                    *[
                        saved[c].astype(str)
                        for c in key_cols
                    ]
                )
            )

            reconstructed_keys = list(
                zip(
                    *[
                        reconstructed[c].astype(str)
                        for c in key_cols
                    ]
                )
            )

            if saved_keys != reconstructed_keys:
                raise RuntimeError(
                    "Saved Part 15 validation cohort does not "
                    "match deterministic seed-42 reconstruction."
                )

        return saved.reset_index(drop=True)

    return part11.select_pilot_rows(
        VAL_MANIFEST,
        VAL_CASES,
        SEED,
    ).reset_index(drop=True)


# =============================================================================
# REPORT
# =============================================================================

def save_outputs(
    summary: Dict[str, Any],
    checkpoint_meta: Dict[str, Any],
    recorded: Dict[str, Any],
    cohort: pd.DataFrame,
):

    case_df = pd.DataFrame(
        summary["case_metrics"]
    )

    case_df.to_csv(
        OUTPUT_DIR
        / "part16_validation_case_metrics.csv",
        index=False,
    )

    class_rows = []

    for c in range(1, NUM_CLASSES):

        class_rows.append(
            {
                "class_id": c,
                "class_name": CLASS_NAMES[c],
                "dice": summary[
                    "per_class_dice"
                ][str(c)],
            }
        )

    class_df = pd.DataFrame(
        class_rows
    )

    class_df.to_csv(
        OUTPUT_DIR
        / "part16_per_class_metrics.csv",
        index=False,
    )

    cohort.to_csv(
        OUTPUT_DIR
        / "part16_validation_cohort.csv",
        index=False,
    )

    reproduced = summary[
        "mean_validation_dice"
    ]

    recorded_best = recorded[
        "best_val_dice"
    ]

    if recorded_best is not None:

        abs_diff = abs(
            reproduced - recorded_best
        )

        relative_diff = (
            abs_diff / max(
                abs(recorded_best),
                1e-12,
            )
        )

    else:

        abs_diff = None
        relative_diff = None

    comparison = pd.DataFrame(
        [
            {
                "part15_recorded_best_dice": recorded_best,
                "part16_reproduced_dice": reproduced,
                "absolute_difference": abs_diff,
                "relative_difference": relative_diff,
                "reproduction_pass": (
                    abs_diff is not None
                    and abs_diff <= 1e-5
                ),
            }
        ]
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "part16_part15_comparison.csv",
        index=False,
    )

    final_summary = {
        "phase": "4 - Part 16",
        "purpose": (
            "Independent validation of Part 15 best checkpoint"
        ),
        "validation_cases": summary[
            "validation_cases"
        ],
        "mean_validation_loss": summary[
            "mean_validation_loss"
        ],
        "mean_validation_dice": summary[
            "mean_validation_dice"
        ],
        "median_validation_dice": summary[
            "median_validation_dice"
        ],
        "std_validation_dice": summary[
            "std_validation_dice"
        ],
        "min_validation_dice": summary[
            "min_validation_dice"
        ],
        "max_validation_dice": summary[
            "max_validation_dice"
        ],
        "per_class_dice": summary[
            "per_class_dice"
        ],
        "fallback_count": summary[
            "fallback_count"
        ],
        "checkpoint_metadata": checkpoint_meta,
        "part15_recorded_result": recorded,
        "reproduced_vs_recorded_absolute_difference": abs_diff,
        "reproduction_pass": (
            abs_diff is not None
            and abs_diff <= 1e-5
        ),
        "spider_used": False,
        "test_set_used": False,
        "training_performed": False,
        "weights_modified": False,
        "metrics_against": (
            "RSNA point-derived pseudo-masks"
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "phase4_part16_best_checkpoint_validation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_summary,
            f,
            indent=2,
        )

    report_path = (
        REPORT_DIR
        / "phase4_part16_best_checkpoint_validation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 4 - PART 16\n"
            "RSNA-ONLY PART 15 BEST CHECKPOINT VALIDATION\n"
            "\n"
        )

        f.write(
            "=" * 78 + "\n"
        )

        f.write(
            f"Validation cases       : "
            f"{summary['validation_cases']}\n"
        )

        f.write(
            f"Mean validation loss   : "
            f"{summary['mean_validation_loss']:.6f}\n"
        )

        f.write(
            f"Mean validation Dice   : "
            f"{summary['mean_validation_dice']:.6f}\n"
        )

        f.write(
            f"Median validation Dice : "
            f"{summary['median_validation_dice']:.6f}\n"
        )

        f.write(
            f"Std validation Dice    : "
            f"{summary['std_validation_dice']:.6f}\n"
        )

        f.write(
            f"Min validation Dice    : "
            f"{summary['min_validation_dice']:.6f}\n"
        )

        f.write(
            f"Max validation Dice    : "
            f"{summary['max_validation_dice']:.6f}\n"
        )

        f.write("\nPER-CLASS DICE\n")

        for c in range(1, NUM_CLASSES):

            f.write(
                f"{CLASS_NAMES[c]:35s}: "
                f"{summary['per_class_dice'][str(c)]:.6f}\n"
            )

        f.write("\nCHECKPOINT\n")

        f.write(
            f"Epoch                 : "
            f"{checkpoint_meta['checkpoint_epoch']}\n"
        )

        f.write(
            f"Checkpoint best Dice  : "
            f"{checkpoint_meta['checkpoint_best_val_dice']}\n"
        )

        f.write(
            f"Checkpoint seed       : "
            f"{checkpoint_meta['checkpoint_seed']}\n"
        )

        f.write(
            f"Checkpoint patch size : "
            f"{checkpoint_meta['checkpoint_patch_size']}\n"
        )

        f.write(
            f"Checkpoint classes    : "
            f"{checkpoint_meta['checkpoint_num_classes']}\n"
        )

        f.write("\nPART 15 COMPARISON\n")

        f.write(
            f"Part 15 recorded best : "
            f"{recorded['best_val_dice']}\n"
        )

        f.write(
            f"Part 16 reproduced    : "
            f"{summary['mean_validation_dice']:.6f}\n"
        )

        f.write(
            f"Absolute difference   : "
            f"{abs_diff}\n"
        )

        if (
            abs_diff is not None
            and abs_diff <= 1e-5
        ):
            decision = (
                "PASS - Part 15 best-checkpoint validation "
                "was independently reproduced."
            )
        else:
            decision = (
                "ACTION REQUIRED - Part 16 does not reproduce "
                "the recorded Part 15 validation result."
            )

        f.write("\nDECISION\n")
        f.write(decision + "\n")

        f.write(
            "\nIMPORTANT:\n"
            "Dice is calculated against RSNA point-derived "
            "pseudo-masks, not manually delineated clinical "
            "segmentation ground truth.\n"
        )

    return final_summary


# =============================================================================
# MAIN
# =============================================================================

def main():

    seed_everything()

    part15_checkpoint = find_part15_checkpoint()

    banner("PHASE 4 - PART 16")
    print(
        "RSNA-ONLY PART 15 BEST CHECKPOINT VALIDATION"
    )
    print()
    print(
        "Evaluation only. No training or weight updates."
    )
    print()

    print("PROJECT ROOT")
    print(PROJECT_ROOT)
    print()

    print("RSNA DATASET")
    print(RSNA_DIR)
    print()

    print("PART 15 BEST CHECKPOINT")
    print(part15_checkpoint)
    print()

    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    banner("PATH VALIDATION")

    require_paths(
        part15_checkpoint
    )

    print("All required paths found.")

    banner("PYTORCH / GPU ENVIRONMENT")

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"PyTorch version : {torch.__version__}"
    )

    print(
        f"CUDA available  : "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"Device          : {device}"
    )

    if device.type == "cuda":

        props = torch.cuda.get_device_properties(0)

        print(
            f"GPU             : {props.name}"
        )

        print(
            f"GPU memory      : "
            f"{props.total_memory / 1024**3:.2f} GB"
        )

    print(
        f"Patch size      : {PATCH_SIZE}"
    )

    print(
        f"Feature size    : {FEATURE_SIZE}"
    )

    print(
        f"Classes         : {NUM_CLASSES}"
    )

    print(
        f"Validation cases: {VAL_CASES}"
    )

    print(
        f"AMP             : "
        f"{AMP_ENABLED and device.type == 'cuda'}"
    )

    banner(
        "IMPORTING VALIDATED PART 11 IMPLEMENTATION"
    )

    part11 = import_part11()

    part9 = part11.load_part9_module()

    print(
        "✓ Corrected Part 11 imported."
    )

    print(
        "✓ Part 9 loader imported through Part 11."
    )

    print(
        "✓ Part 11 preprocessing retained."
    )

    print(
        "✓ Part 11 Dice API: "
        "dice_from_prediction(logits, target)."
    )

    banner(
        "RECONSTRUCTING PART 15 VALIDATION COHORT"
    )

    val_manifest = pd.read_csv(
        VAL_MANIFEST
    )

    val_rows = reconstruct_validation_cohort(
        part11
    )

    print(
        f"Validation manifest total : "
        f"{len(val_manifest)}"
    )

    print(
        f"Part 16 validation cohort : "
        f"{len(val_rows)}"
    )

    if len(val_rows) != VAL_CASES:
        raise RuntimeError(
            f"Expected {VAL_CASES} validation cases, "
            f"got {len(val_rows)}."
        )

    banner(
        "CREATING SWIN-UNETR"
    )

    model = part11.create_model(
        device
    )

    model = model.to(device)

    print(
        "✓ Swin-UNETR created."
    )

    print(
        f"Total parameters : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    banner(
        "LOADING PART 15 BEST CHECKPOINT"
    )

    checkpoint_meta = load_checkpoint(
        model,
        part15_checkpoint,
        device,
    )

    print(
        f"Checkpoint epoch          : "
        f"{checkpoint_meta['checkpoint_epoch']}"
    )

    print(
        f"Checkpoint best Dice      : "
        f"{checkpoint_meta['checkpoint_best_val_dice']}"
    )

    print(
        f"Checkpoint seed           : "
        f"{checkpoint_meta['checkpoint_seed']}"
    )

    print(
        f"Checkpoint patch size    : "
        f"{checkpoint_meta['checkpoint_patch_size']}"
    )

    print(
        f"Checkpoint classes        : "
        f"{checkpoint_meta['checkpoint_num_classes']}"
    )

    # Verify architecture/configuration metadata when available.
    if (
        checkpoint_meta["checkpoint_feature_size"]
        is not None
        and int(
            checkpoint_meta["checkpoint_feature_size"]
        ) != FEATURE_SIZE
    ):
        raise RuntimeError(
            "Part 15 checkpoint feature size does not match Part 16."
        )

    if (
        checkpoint_meta["checkpoint_num_classes"]
        is not None
        and int(
            checkpoint_meta["checkpoint_num_classes"]
        ) != NUM_CLASSES
    ):
        raise RuntimeError(
            "Part 15 checkpoint class count does not match Part 16."
        )

    banner(
        "LOSS / EVALUATION CONFIGURATION"
    )

    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
    )

    print(
        "Loss         : DiceCELoss"
    )

    print(
        "Mode         : evaluation only"
    )

    print(
        "Gradient     : disabled"
    )

    print(
        "Weight update: disabled"
    )

    banner(
        "STARTING PART 15 BEST CHECKPOINT VALIDATION"
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

    print()

    if device.type == "cuda":

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            device
        )

    summary = evaluate(
        model=model,
        rows=val_rows,
        part11=part11,
        part9=part9,
        loss_fn=loss_fn,
        device=device,
        amp_enabled=AMP_ENABLED,
    )

    if device.type == "cuda":

        peak_allocated = (
            torch.cuda.max_memory_allocated(device)
            / 1024**3
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved(device)
            / 1024**3
        )

    else:

        peak_allocated = 0.0
        peak_reserved = 0.0

    recorded = load_recorded_part15_result()

    reproduced_dice = (
        summary["mean_validation_dice"]
    )

    recorded_dice = (
        recorded["best_val_dice"]
    )

    if recorded_dice is not None:

        abs_diff = abs(
            reproduced_dice
            - recorded_dice
        )

    else:

        abs_diff = None

    banner(
        "PART 16 FINAL SUMMARY"
    )

    print(
        f"Validation cases       : "
        f"{summary['validation_cases']}"
    )

    print(
        f"Mean validation loss   : "
        f"{summary['mean_validation_loss']:.6f}"
    )

    print(
        f"Mean validation Dice   : "
        f"{summary['mean_validation_dice']:.6f}"
    )

    print(
        f"Median validation Dice : "
        f"{summary['median_validation_dice']:.6f}"
    )

    print(
        f"Std validation Dice    : "
        f"{summary['std_validation_dice']:.6f}"
    )

    print(
        f"Min validation Dice    : "
        f"{summary['min_validation_dice']:.6f}"
    )

    print(
        f"Max validation Dice    : "
        f"{summary['max_validation_dice']:.6f}"
    )

    print()

    print("PER-CLASS DICE")

    for c in range(1, NUM_CLASSES):

        print(
            f"{CLASS_NAMES[c]:35s}: "
            f"{summary['per_class_dice'][str(c)]:.6f}"
        )

    print()

    print(
        f"Loader fallbacks       : "
        f"{summary['fallback_count']}"
    )

    print(
        f"Peak GPU allocated     : "
        f"{peak_allocated:.3f} GB"
    )

    print(
        f"Peak GPU reserved      : "
        f"{peak_reserved:.3f} GB"
    )

    print()

    print(
        f"Part 15 recorded best Dice : "
        f"{recorded_dice}"
    )

    print(
        f"Part 16 reproduced Dice    : "
        f"{reproduced_dice:.6f}"
    )

    print(
        f"Absolute Dice difference   : "
        f"{abs_diff}"
    )

    if (
        abs_diff is not None
        and abs_diff <= 1e-5
    ):

        decision = (
            "PASS - Part 16 independently reproduces "
            "the recorded Part 15 best-checkpoint "
            "validation result."
        )

    else:

        decision = (
            "ACTION REQUIRED - Part 16 does not reproduce "
            "the recorded Part 15 validation result."
        )

    banner(
        "SAVING PART 16 RESULTS"
    )

    final_summary = save_outputs(
        summary=summary,
        checkpoint_meta=checkpoint_meta,
        recorded=recorded,
        cohort=val_rows,
    )

    final_summary[
        "peak_gpu_allocated_gb"
    ] = peak_allocated

    final_summary[
        "peak_gpu_reserved_gb"
    ] = peak_reserved

    # Rewrite JSON with GPU information included.
    summary_path = (
        OUTPUT_DIR
        / "phase4_part16_best_checkpoint_validation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_summary,
            f,
            indent=2,
        )

    print(
        "Saved:",
        OUTPUT_DIR
        / "part16_validation_case_metrics.csv",
    )

    print(
        "Saved:",
        OUTPUT_DIR
        / "part16_per_class_metrics.csv",
    )

    print(
        "Saved:",
        OUTPUT_DIR
        / "part16_validation_cohort.csv",
    )

    print(
        "Saved:",
        OUTPUT_DIR
        / "part16_part15_comparison.csv",
    )

    print(
        "Saved:",
        summary_path,
    )

    print(
        "Saved:",
        REPORT_DIR
        / "phase4_part16_best_checkpoint_validation_report.txt",
    )

    banner(
        "PART 16 DECISION"
    )

    print(decision)

    print()

    print(
        "SPIDER used          : NO"
    )

    print(
        "Test set used        : NO"
    )

    print(
        "Training performed   : NO"
    )

    print(
        "Model weights changed: NO"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Dice is calculated against RSNA point-derived "
        "pseudo-masks, not manually delineated clinical "
        "segmentation ground truth."
    )

    banner(
        "PHASE 4 - PART 16 COMPLETE"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:

        banner("PART 16 ERROR")

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise