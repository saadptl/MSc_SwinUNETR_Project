"""
================================================================================
PART 100 — ROBUST CLINICAL-ORIENTED SWIN-UNETR FINE-TUNING
================================================================================

Purpose
-------
Continue fine-tuning the strongest available Part98 Swin-UNETR checkpoint using
a robust clinical-oriented segmentation objective.

IMPORTANT CORRECTION
--------------------
The previous Part100 implementation incorrectly required exactly 600 unique
discoverable segmentation cases.

The actual discoverable RSNA segmentation pool contains only 517 unique cases.

Part98 already demonstrated that a 500/100 cohort can be constructed and trained
successfully. Therefore this version:

1. Uses existing Part98 500/100 cohort files when available.
2. Otherwise reconstructs cohorts from the established Part15 cohorts.
3. Performs strict study-level leakage auditing.
4. Never requires an artificial 600-case filesystem requirement.
5. If fewer than 600 unique cases are available, uses the maximum feasible
   leakage-free cohort.
6. Continues from the Part98 best checkpoint.
7. Uses Dice + Focal + Tversky loss.
8. Uses foreground-aware crop sampling.
9. Uses gradient accumulation and AMP.
10. Independently reloads and verifies the final best checkpoint.

Classes
-------
0 Background
1 Spinal Canal Stenosis
2 Left Neural Foraminal Narrowing
3 Right Neural Foraminal Narrowing
4 Left Subarticular Stenosis
5 Right Subarticular Stenosis

This is advanced research/development training. It does NOT constitute clinical
validation, regulatory approval, or clinical deployment readiness.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part100_robust_clinical_finetuning"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

EPOCHS = 10

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4

LEARNING_RATE = 2.0e-5
WEIGHT_DECAY = 1.0e-5

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)

FEATURE_SIZE = 12
NUM_CLASSES = 6

USE_AMP = True

# Clinical-oriented composite objective
DICE_WEIGHT = 0.45
FOCAL_WEIGHT = 0.30
TVERSKY_WEIGHT = 0.25

FOCAL_GAMMA = 2.0

TVERSKY_ALPHA = 0.30
TVERSKY_BETA = 0.70

SMOOTH = 1e-6

# Foreground-aware training crop probability.
FOREGROUND_CROP_PROBABILITY = 0.80

# Existing Part98 intended cohort.
REQUESTED_TRAIN = 500
REQUESTED_VAL = 100

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# =============================================================================
# CHECKPOINTS / COHORTS
# =============================================================================

PART98_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part98_strong_full_cohort_training"
)

PART98_CHECKPOINT = (
    PART98_DIR
    / "checkpoints"
    / "part98_best_model.pth"
)

PART98_TRAIN_COHORT = (
    PART98_DIR
    / "part98_train_cohort.csv"
)

PART98_VAL_COHORT = (
    PART98_DIR
    / "part98_validation_cohort.csv"
)

PART15_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART15_INIT = (
    PART15_DIR
    / "checkpoints"
    / "part15_initialization_from_part11.pth"
)

PART15_TRAIN_COHORT = (
    PART15_DIR
    / "part15_train_cohort.csv"
)

PART15_VAL_COHORT = (
    PART15_DIR
    / "part15_validation_cohort.csv"
)


BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "part100_best_model.pth"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "part100_final_model.pth"
)

HISTORY_CSV = (
    OUTPUT_DIR
    / "part100_training_history.csv"
)

VAL_CASE_CSV = (
    OUTPUT_DIR
    / "part100_validation_case_metrics.csv"
)

TRAIN_COHORT_OUTPUT = (
    OUTPUT_DIR
    / "part100_train_cohort.csv"
)

VAL_COHORT_OUTPUT = (
    OUTPUT_DIR
    / "part100_validation_cohort.csv"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "part100_robust_clinical_summary.json"
)

REPORT_TXT = (
    REPORT_DIR
    / "part100_robust_clinical_report.txt"
)


# =============================================================================
# PRINT HELPERS
# =============================================================================

def banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def section(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# HASH
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


# =============================================================================
# DYNAMIC IMPORT
# =============================================================================

def import_module_from_path(
    name: str,
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import module from {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def import_project_modules():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    part9_path = (
        SRC
        / "segmentation_rsna_part9_3d_dataset_loader.py"
    )

    part11_path = (
        SRC
        / "segmentation_rsna_part11_controlled_pilot_training.py"
    )

    part9 = import_module_from_path(
        "part100_part9",
        part9_path,
    )

    part11 = import_module_from_path(
        "part100_part11",
        part11_path,
    )

    return part9, part11


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint(
    path: Path,
    device: torch.device,
):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


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
        "Could not locate model state_dict."
    )


def load_model_checkpoint(
    model: nn.Module,
    path: Path,
    device: torch.device,
):
    checkpoint = load_checkpoint(
        path,
        device,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=True,
    )

    return checkpoint, missing, unexpected


# =============================================================================
# COHORT UTILITIES
# =============================================================================

def load_csv(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(
            f"Cohort is empty: {path}"
        )

    return df.copy()


def find_identifier_column(
    df: pd.DataFrame,
) -> Optional[str]:

    candidates = [
        "study_id",
        "study",
        "studyid",
        "id",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    return None


def normalize_id(value) -> str:

    if pd.isna(value):
        return ""

    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def study_ids(
    df: pd.DataFrame,
) -> List[str]:

    col = find_identifier_column(df)

    if col is None:
        return [
            str(i)
            for i in range(len(df))
        ]

    return [
        normalize_id(x)
        for x in df[col]
    ]


def audit_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> int:

    train_ids = set(
        study_ids(train_df)
    )

    val_ids = set(
        study_ids(val_df)
    )

    overlap = train_ids & val_ids

    if overlap:
        raise RuntimeError(
            "TRAIN/VALIDATION LEAKAGE DETECTED: "
            f"{len(overlap)} overlapping studies."
        )

    return len(overlap)


# =============================================================================
# ROBUST COHORT CONSTRUCTION
# =============================================================================

def resolve_cohorts() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    str,
]:
    """
    Resolve the strongest available leakage-free cohort.

    Priority:

    1. Existing Part98 500/100 cohorts.
    2. Deterministic reconstruction from Part15.
    3. If requested 500/100 is impossible at study level,
       use the maximum feasible leakage-free split.

    This avoids the incorrect 600-case hard requirement.
    """

    # -------------------------------------------------------------------------
    # OPTION 1 — EXISTING PART98 COHORTS
    # -------------------------------------------------------------------------

    if (
        PART98_TRAIN_COHORT.exists()
        and PART98_VAL_COHORT.exists()
    ):

        train_df = load_csv(
            PART98_TRAIN_COHORT
        )

        val_df = load_csv(
            PART98_VAL_COHORT
        )

        overlap = audit_leakage(
            train_df,
            val_df,
        )

        print(
            "Existing Part98 cohort files found."
        )

        print(
            f"Part98 training rows : {len(train_df)}"
        )

        print(
            f"Part98 validation rows: {len(val_df)}"
        )

        print(
            f"Study overlap         : {overlap}"
        )

        if len(train_df) > 0 and len(val_df) > 0:
            return (
                train_df.reset_index(drop=True),
                val_df.reset_index(drop=True),
                "PART98_EXISTING_COHORT",
            )

    # -------------------------------------------------------------------------
    # OPTION 2 — PART15 RECONSTRUCTION
    # -------------------------------------------------------------------------

    print(
        "Part98 cohort CSV files not found."
    )

    print(
        "Reconstructing from established Part15 cohorts."
    )

    base_train = load_csv(
        PART15_TRAIN_COHORT
    )

    base_val = load_csv(
        PART15_VAL_COHORT
    )

    train_ids = study_ids(
        base_train
    )

    val_ids = study_ids(
        base_val
    )

    # -------------------------------------------------------------------------
    # Remove duplicate study rows deterministically.
    # -------------------------------------------------------------------------

    train_unique_mask = ~pd.Series(
        train_ids
    ).duplicated(
        keep="first"
    )

    val_unique_mask = ~pd.Series(
        val_ids
    ).duplicated(
        keep="first"
    )

    train_unique = (
        base_train
        .loc[
            train_unique_mask.values
        ]
        .reset_index(drop=True)
    )

    val_unique = (
        base_val
        .loc[
            val_unique_mask.values
        ]
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Remove any cross-partition overlap.
    # -------------------------------------------------------------------------

    train_set = set(
        study_ids(train_unique)
    )

    val_set = set(
        study_ids(val_unique)
    )

    cross_overlap = (
        train_set & val_set
    )

    if cross_overlap:

        print(
            "WARNING: overlapping study IDs found "
            "between source cohorts."
        )

        print(
            f"Removing {len(cross_overlap)} "
            "overlapping validation studies."
        )

        val_unique = val_unique[
            ~val_unique.apply(
                lambda row:
                normalize_id(
                    row[
                        find_identifier_column(
                            val_unique
                        )
                    ]
                )
                in cross_overlap,
                axis=1,
            )
        ].reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Deterministic sampling.
    # -------------------------------------------------------------------------

    rng_train = np.random.RandomState(
        SEED
    )

    rng_val = np.random.RandomState(
        SEED + 1
    )

    train_available = len(
        train_unique
    )

    val_available = len(
        val_unique
    )

    train_target = min(
        REQUESTED_TRAIN,
        train_available,
    )

    val_target = min(
        REQUESTED_VAL,
        val_available,
    )

    # -------------------------------------------------------------------------
    # If the requested total is impossible, use available data.
    #
    # The important rule is that validation remains separate.
    # -------------------------------------------------------------------------

    if train_target <= 0:
        raise RuntimeError(
            "No usable training studies available."
        )

    if val_target <= 0:
        raise RuntimeError(
            "No usable validation studies available."
        )

    train_indices = rng_train.choice(
        train_available,
        size=train_target,
        replace=False,
    )

    train_df = (
        train_unique
        .iloc[
            sorted(train_indices)
        ]
        .reset_index(drop=True)
    )

    # Validation must not overlap with selected training IDs.
    selected_train_ids = set(
        study_ids(train_df)
    )

    val_candidates = val_unique[
        ~val_unique.apply(
            lambda row:
            normalize_id(
                row[
                    find_identifier_column(
                        val_unique
                    )
                ]
            )
            in selected_train_ids,
            axis=1,
        )
    ].reset_index(drop=True)

    val_available_after_leakage = len(
        val_candidates
    )

    val_target = min(
        REQUESTED_VAL,
        val_available_after_leakage,
    )

    if val_target <= 0:
        raise RuntimeError(
            "No leakage-free validation studies "
            "remain after cohort construction."
        )

    val_indices = rng_val.choice(
        val_available_after_leakage,
        size=val_target,
        replace=False,
    )

    val_df = (
        val_candidates
        .iloc[
            sorted(val_indices)
        ]
        .reset_index(drop=True)
    )

    overlap = audit_leakage(
        train_df,
        val_df,
    )

    if overlap != 0:
        raise RuntimeError(
            "Internal leakage audit failed."
        )

    return (
        train_df,
        val_df,
        "PART15_RECONSTRUCTED_ADAPTIVE",
    )


# =============================================================================
# IMAGE / MASK NORMALIZATION
# =============================================================================

def ensure_3d(
    tensor,
    name: str,
) -> torch.Tensor:

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(
            tensor
        )

    if not isinstance(
        tensor,
        torch.Tensor,
    ):
        tensor = torch.as_tensor(
            tensor
        )

    tensor = tensor.detach().cpu()

    # [1,D,H,W]
    if tensor.ndim == 4:
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.shape[-1] == 1:
            tensor = tensor[..., 0]

    if tensor.ndim != 3:
        raise RuntimeError(
            f"{name} must be 3D, got "
            f"{tuple(tensor.shape)}"
        )

    return tensor


def resize_image(
    image: torch.Tensor,
) -> torch.Tensor:

    image = ensure_3d(
        image,
        "image",
    ).float()

    image = F.interpolate(
        image[None, None],
        size=FULL_SHAPE,
        mode="trilinear",
        align_corners=False,
    )[0, 0]

    return image


def resize_mask(
    mask: torch.Tensor,
) -> torch.Tensor:

    mask = ensure_3d(
        mask,
        "mask",
    ).long()

    mask = F.interpolate(
        mask[None, None].float(),
        size=FULL_SHAPE,
        mode="nearest",
    )[0, 0].long()

    mask = torch.clamp(
        mask,
        min=0,
        max=NUM_CLASSES - 1,
    )

    return mask


# =============================================================================
# CASE LOADING
# =============================================================================

def load_case(
    row: pd.Series,
    part9,
    part11,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:

    result = part11.load_tensor_case(
        row,
        part9,
    )

    if len(result) >= 2:
        image = result[0]
        mask = result[1]
    else:
        raise RuntimeError(
            "Unexpected load_tensor_case result."
        )

    image = ensure_3d(
        image,
        "image",
    )

    mask = ensure_3d(
        mask,
        "mask",
    )

    image = resize_image(
        image
    )

    mask = resize_mask(
        mask
    )

    image = torch.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    # Normalize robustly.
    min_value = float(
        image.min()
    )

    max_value = float(
        image.max()
    )

    if max_value > min_value:

        image = (
            image - min_value
        ) / (
            max_value - min_value
        )

    else:

        image = torch.zeros_like(
            image
        )

    return (
        image.float(),
        mask.long(),
    )


# =============================================================================
# FOREGROUND CENTER
# =============================================================================

def mask_centroid(
    mask: torch.Tensor,
) -> Tuple[int, int, int]:

    coords = torch.nonzero(
        mask > 0,
        as_tuple=False,
    )

    if coords.numel() == 0:

        return tuple(
            (n - 1) // 2
            for n in FULL_SHAPE
        )

    center = (
        coords.float()
        .mean(dim=0)
        .round()
        .long()
        .tolist()
    )

    return (
        int(center[0]),
        int(center[1]),
        int(center[2]),
    )


def random_center() -> Tuple[
    int,
    int,
    int,
]:

    return tuple(
        random.randint(
            crop // 2,
            full - crop // 2,
        )
        for full, crop in zip(
            FULL_SHAPE,
            CROP_SHAPE,
        )
    )


def crop_bounds(
    center: Tuple[int, int, int],
):

    bounds = []

    for c, full, crop in zip(
        center,
        FULL_SHAPE,
        CROP_SHAPE,
    ):

        start = int(
            c - crop // 2
        )

        start = max(
            0,
            min(
                start,
                full - crop,
            ),
        )

        end = start + crop

        bounds.append(
            (start, end)
        )

    return bounds


def crop_case(
    image: torch.Tensor,
    mask: torch.Tensor,
    training: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:

    if (
        training
        and torch.rand(1).item()
        < FOREGROUND_CROP_PROBABILITY
        and torch.any(mask > 0)
    ):

        center = mask_centroid(
            mask
        )

        # Add small deterministic/random perturbation
        # so the model does not always see exactly
        # the same crop.
        perturbed = []

        for idx, (
            c,
            full,
            crop,
        ) in enumerate(
            zip(
                center,
                FULL_SHAPE,
                CROP_SHAPE,
            )
        ):

            jitter = random.randint(
                -max(1, crop // 8),
                max(1, crop // 8),
            )

            value = c + jitter

            value = max(
                crop // 2,
                min(
                    value,
                    full - crop // 2,
                ),
            )

            perturbed.append(
                value
            )

        center = tuple(
            perturbed
        )

    else:

        if torch.any(mask > 0):

            center = mask_centroid(
                mask
            )

        else:

            center = tuple(
                n // 2
                for n in FULL_SHAPE
            )

    bounds = crop_bounds(
        center
    )

    slices = tuple(
        slice(start, end)
        for start, end in bounds
    )

    return (
        image[slices],
        mask[slices],
    )


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:

    probs = torch.softmax(
        logits,
        dim=1,
    )

    target_onehot = F.one_hot(
        target.long(),
        num_classes=NUM_CLASSES,
    )

    target_onehot = (
        target_onehot
        .permute(
            0,
            4,
            1,
            2,
            3,
        )
        .float()
    )

    probs_fg = probs[:, 1:]
    target_fg = target_onehot[:, 1:]

    intersection = (
        probs_fg * target_fg
    ).sum(
        dim=(0, 2, 3, 4)
    )

    denominator = (
        probs_fg.sum(
            dim=(0, 2, 3, 4)
        )
        +
        target_fg.sum(
            dim=(0, 2, 3, 4)
        )
    )

    dice = (
        2.0 * intersection
        + SMOOTH
    ) / (
        denominator
        + SMOOTH
    )

    valid = (
        target_fg.sum(
            dim=(0, 2, 3, 4)
        ) > 0
    )

    if valid.any():

        return (
            1.0
            - dice[valid].mean()
        )

    return 1.0 - dice.mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:

    log_probs = F.log_softmax(
        logits,
        dim=1,
    )

    probs = torch.softmax(
        logits,
        dim=1,
    )

    target_log_prob = (
        log_probs
        .gather(
            1,
            target.unsqueeze(1),
        )
        .squeeze(1)
    )

    target_prob = (
        probs
        .gather(
            1,
            target.unsqueeze(1),
        )
        .squeeze(1)
    )

    focal_factor = (
        1.0 - target_prob
    ).pow(
        FOCAL_GAMMA
    )

    loss = (
        -focal_factor
        * target_log_prob
    )

    # Background remains important but foreground
    # voxels receive stronger effective contribution.
    foreground = (
        target > 0
    ).float()

    weight = (
        1.0
        + 2.0 * foreground
    )

    return (
        loss * weight
    ).mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:

    probs = torch.softmax(
        logits,
        dim=1,
    )

    target_onehot = F.one_hot(
        target.long(),
        num_classes=NUM_CLASSES,
    )

    target_onehot = (
        target_onehot
        .permute(
            0,
            4,
            1,
            2,
            3,
        )
        .float()
    )

    probs_fg = probs[:, 1:]
    target_fg = target_onehot[:, 1:]

    tp = (
        probs_fg * target_fg
    ).sum(
        dim=(0, 2, 3, 4)
    )

    fp = (
        probs_fg * (1.0 - target_fg)
    ).sum(
        dim=(0, 2, 3, 4)
    )

    fn = (
        (1.0 - probs_fg) * target_fg
    ).sum(
        dim=(0, 2, 3, 4)
    )

    tversky = (
        tp + SMOOTH
    ) / (
        tp
        + TVERSKY_ALPHA * fp
        + TVERSKY_BETA * fn
        + SMOOTH
    )

    valid = (
        target_fg.sum(
            dim=(0, 2, 3, 4)
        ) > 0
    )

    if valid.any():

        return (
            1.0
            - tversky[valid].mean()
        )

    return 1.0 - tversky.mean()


def clinical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
):

    dice = multiclass_dice_loss(
        logits,
        target,
    )

    focal = focal_loss(
        logits,
        target,
    )

    tversky = tversky_loss(
        logits,
        target,
    )

    total = (
        DICE_WEIGHT * dice
        +
        FOCAL_WEIGHT * focal
        +
        TVERSKY_WEIGHT * tversky
    )

    return (
        total,
        dice.detach(),
        focal.detach(),
        tversky.detach(),
    )


# =============================================================================
# METRICS
# =============================================================================

def dice_for_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
) -> float:

    p = pred == class_id
    t = target == class_id

    p_sum = int(
        p.sum().item()
    )

    t_sum = int(
        t.sum().item()
    )

    if (
        p_sum == 0
        and t_sum == 0
    ):
        return float("nan")

    intersection = int(
        (p & t).sum().item()
    )

    return (
        2.0 * intersection
    ) / (
        p_sum
        + t_sum
        + 1e-8
    )


def foreground_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:

    p = pred > 0
    t = target > 0

    p_sum = int(
        p.sum().item()
    )

    t_sum = int(
        t.sum().item()
    )

    if (
        p_sum == 0
        and t_sum == 0
    ):
        return 1.0

    intersection = int(
        (p & t).sum().item()
    )

    return (
        2.0 * intersection
    ) / (
        p_sum
        + t_sum
        + 1e-8
    )


def case_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict:

    result = {
        "foreground_dice":
            foreground_dice(
                pred,
                target,
            ),
        "target_foreground_voxels":
            int(
                (target > 0)
                .sum()
                .item()
            ),
        "predicted_foreground_voxels":
            int(
                (pred > 0)
                .sum()
                .item()
            ),
    }

    for class_id in range(
        1,
        NUM_CLASSES,
    ):

        result[
            f"class_{class_id}_dice"
        ] = dice_for_class(
            pred,
            target,
            class_id,
        )

    return result


# =============================================================================
# MODEL
# =============================================================================

def create_model(
    part11,
) -> nn.Module:

    try:

        model = part11.create_model(
            in_channels=1,
            out_channels=NUM_CLASSES,
            feature_size=FEATURE_SIZE,
        )

    except TypeError:

        model = part11.create_model(
            DEVICE
        )

    return model.to(
        DEVICE
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    train_df: pd.DataFrame,
    part9,
    part11,
    optimizer,
    scaler,
) -> Dict:

    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_focal = 0.0
    total_tversky = 0.0

    successful = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    amp_enabled = (
        USE_AMP
        and DEVICE.type == "cuda"
    )

    for index, (_, row) in enumerate(
        train_df.iterrows()
    ):

        try:

            image, mask = load_case(
                row,
                part9,
                part11,
            )

            image, mask = crop_case(
                image,
                mask,
                training=True,
            )

            image = image.unsqueeze(0).unsqueeze(0)
            mask = mask.unsqueeze(0)

            image = image.to(
                DEVICE,
                non_blocking=True,
            )

            mask = mask.to(
                DEVICE,
                non_blocking=True,
            )

            if amp_enabled:

                with torch.amp.autocast(
                    "cuda"
                ):

                    logits = model(
                        image
                    )

                    loss, dice_loss, focal, tversky = (
                        clinical_loss(
                            logits,
                            mask,
                        )
                    )

            else:

                logits = model(
                    image
                )

                loss, dice_loss, focal, tversky = (
                    clinical_loss(
                        logits,
                        mask,
                    )
                )

            scaled_loss = (
                loss
                / GRADIENT_ACCUMULATION
            )

            if scaler is not None:

                scaler.scale(
                    scaled_loss
                ).backward()

            else:

                scaled_loss.backward()

            should_step = (
                (
                    (index + 1)
                    % GRADIENT_ACCUMULATION
                    == 0
                )
                or
                index
                == len(train_df) - 1
            )

            if should_step:

                if scaler is not None:

                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )

                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            total_loss += float(
                loss.detach()
                .cpu()
                .item()
            )

            total_dice += float(
                dice_loss
                .cpu()
                .item()
            )

            total_focal += float(
                focal
                .cpu()
                .item()
            )

            total_tversky += float(
                tversky
                .cpu()
                .item()
            )

            successful += 1

        except Exception as exc:

            print(
                f"\nWARNING: training case "
                f"{index + 1}/{len(train_df)} failed: "
                f"{exc}"
            )

    if successful == 0:
        raise RuntimeError(
            "No training cases completed."
        )

    return {
        "loss": total_loss / successful,
        "dice_loss":
            total_dice / successful,
        "focal_loss":
            total_focal / successful,
        "tversky_loss":
            total_tversky / successful,
        "successful_cases":
            successful,
    }


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    val_df: pd.DataFrame,
    part9,
    part11,
    save_cases: bool = True,
) -> Tuple[
    Dict,
    List[Dict],
]:

    model.eval()

    case_rows = []

    global_intersection = 0
    global_pred_fg = 0
    global_target_fg = 0

    class_intersection = {
        c: 0
        for c in range(
            1,
            NUM_CLASSES,
        )
    }

    class_pred = {
        c: 0
        for c in range(
            1,
            NUM_CLASSES,
        )
    }

    class_target = {
        c: 0
        for c in range(
            1,
            NUM_CLASSES,
        )
    }

    amp_enabled = (
        USE_AMP
        and DEVICE.type == "cuda"
    )

    successful = 0
    losses = []

    for index, (_, row) in enumerate(
        val_df.iterrows()
    ):

        try:

            image, mask = load_case(
                row,
                part9,
                part11,
            )

            image, mask = crop_case(
                image,
                mask,
                training=False,
            )

            input_tensor = (
                image
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            target = (
                mask
                .unsqueeze(0)
                .to(DEVICE)
            )

            if amp_enabled:

                with torch.amp.autocast(
                    "cuda"
                ):

                    logits = model(
                        input_tensor
                    )

                    loss, _, _, _ = clinical_loss(
                        logits,
                        target,
                    )

            else:

                logits = model(
                    input_tensor
                )

                loss, _, _, _ = clinical_loss(
                    logits,
                    target,
                )

            pred = torch.argmax(
                logits,
                dim=1,
            )[0].cpu()

            target_cpu = (
                target[0]
                .cpu()
            )

            metrics = case_metrics(
                pred,
                target_cpu,
            )

            case_row = {
                "case_index":
                    index,
                "study_id":
                    normalize_id(
                        row.get(
                            "study_id",
                            index,
                        )
                    ),
                "validation_loss":
                    float(
                        loss
                        .cpu()
                        .item()
                    ),
                **metrics,
            }

            case_rows.append(
                case_row
            )

            losses.append(
                float(
                    loss
                    .cpu()
                    .item()
                )
            )

            pred_fg = pred > 0
            target_fg = (
                target_cpu > 0
            )

            global_intersection += int(
                (
                    pred_fg
                    & target_fg
                )
                .sum()
                .item()
            )

            global_pred_fg += int(
                pred_fg.sum()
                .item()
            )

            global_target_fg += int(
                target_fg.sum()
                .item()
            )

            for c in range(
                1,
                NUM_CLASSES,
            ):

                p = pred == c
                t = target_cpu == c

                class_intersection[c] += int(
                    (p & t)
                    .sum()
                    .item()
                )

                class_pred[c] += int(
                    p.sum()
                    .item()
                )

                class_target[c] += int(
                    t.sum()
                    .item()
                )

            successful += 1

        except Exception as exc:

            print(
                f"\nWARNING: validation case "
                f"{index + 1}/{len(val_df)} failed: "
                f"{exc}"
            )

    if successful == 0:
        raise RuntimeError(
            "No validation cases completed."
        )

    global_dice = (
        2.0 * global_intersection
        + SMOOTH
    ) / (
        global_pred_fg
        + global_target_fg
        + SMOOTH
    )

    case_dices = [
        float(
            x["foreground_dice"]
        )
        for x in case_rows
        if np.isfinite(
            x["foreground_dice"]
        )
    ]

    class_dices = {}

    for c in range(
        1,
        NUM_CLASSES,
    ):

        denom = (
            class_pred[c]
            + class_target[c]
        )

        if denom == 0:

            class_dices[c] = float(
                "nan"
            )

        else:

            class_dices[c] = (
                2.0
                * class_intersection[c]
                + SMOOTH
            ) / (
                denom
                + SMOOTH
            )

    macro_dice_values = [
        value
        for value in class_dices.values()
        if np.isfinite(value)
    ]

    summary = {
        "validation_loss":
            float(
                np.mean(losses)
            ),
        "global_foreground_dice":
            float(global_dice),
        "mean_case_foreground_dice":
            float(
                np.mean(case_dices)
            )
            if case_dices
            else float("nan"),
        "median_case_foreground_dice":
            float(
                np.median(case_dices)
            )
            if case_dices
            else float("nan"),
        "macro_foreground_dice":
            float(
                np.mean(
                    macro_dice_values
                )
            )
            if macro_dice_values
            else float("nan"),
        "successful_cases":
            successful,
        "failed_cases":
            len(val_df) - successful,
        "target_foreground_voxels":
            int(global_target_fg),
        "predicted_foreground_voxels":
            int(global_pred_fg),
    }

    for c in range(
        1,
        NUM_CLASSES,
    ):

        summary[
            f"class_{c}_dice"
        ] = class_dices[c]

        summary[
            f"class_{c}_target_voxels"
        ] = class_target[c]

        summary[
            f"class_{c}_predicted_voxels"
        ] = class_pred[c]

    if save_cases:

        pd.DataFrame(
            case_rows
        ).to_csv(
            VAL_CASE_CSV,
            index=False,
        )

    return (
        summary,
        case_rows,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed()

    banner(
        "PART 100 — ROBUST CLINICAL-ORIENTED "
        "SWIN-UNETR FINE-TUNING"
    )

    print(
        f"Project root : {ROOT}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "GPU          : CPU execution"
        )

    section(
        "CONFIGURATION"
    )

    print(
        f"Epochs                 : {EPOCHS}"
    )

    print(
        f"Batch size             : {BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation  : "
        f"{GRADIENT_ACCUMULATION}"
    )

    print(
        f"Learning rate          : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay           : "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Requested train cohort : "
        f"{REQUESTED_TRAIN}"
    )

    print(
        f"Requested validation   : "
        f"{REQUESTED_VAL}"
    )

    print(
        f"Full volume            : "
        f"{FULL_SHAPE}"
    )

    print(
        f"Training crop          : "
        f"{CROP_SHAPE}"
    )

    print(
        f"Feature size           : "
        f"{FEATURE_SIZE}"
    )

    print(
        f"Foreground crop prob.  : "
        f"{FOREGROUND_CROP_PROBABILITY}"
    )

    print(
        f"AMP                    : "
        f"{USE_AMP}"
    )

    section(
        "LOSS"
    )

    print(
        f"Dice weight             : "
        f"{DICE_WEIGHT}"
    )

    print(
        f"Focal weight            : "
        f"{FOCAL_WEIGHT}"
    )

    print(
        f"Tversky weight          : "
        f"{TVERSKY_WEIGHT}"
    )

    print(
        f"Focal gamma             : "
        f"{FOCAL_GAMMA}"
    )

    print(
        f"Tversky alpha/beta      : "
        f"{TVERSKY_ALPHA}/{TVERSKY_BETA}"
    )

    # -------------------------------------------------------------------------
    # Required artifacts
    # -------------------------------------------------------------------------

    section(
        "CHECKING INITIAL CHECKPOINT"
    )

    print(
        f"Part98 best checkpoint : "
        f"{'PASS' if PART98_CHECKPOINT.exists() else 'FAIL'}"
    )

    if not PART98_CHECKPOINT.exists():
        raise FileNotFoundError(
            PART98_CHECKPOINT
        )

    part98_hash = sha256_file(
        PART98_CHECKPOINT
    )

    print(
        f"SHA256 : {part98_hash}"
    )

    # -------------------------------------------------------------------------
    # Modules
    # -------------------------------------------------------------------------

    section(
        "IMPORTING EXISTING PROJECT MODULES"
    )

    part9, part11 = (
        import_project_modules()
    )

    print(
        f"Part9 module  : {part9.__name__}"
    )

    print(
        f"Part11 module : {part11.__name__}"
    )

    # -------------------------------------------------------------------------
    # Cohorts
    # -------------------------------------------------------------------------

    section(
        "BUILDING ROBUST LEAKAGE-FREE COHORT"
    )

    train_df, val_df, cohort_source = (
        resolve_cohorts()
    )

    overlap = audit_leakage(
        train_df,
        val_df,
    )

    print(
        f"Cohort source              : "
        f"{cohort_source}"
    )

    print(
        f"Training cohort rows       : "
        f"{len(train_df)}"
    )

    print(
        f"Validation cohort rows     : "
        f"{len(val_df)}"
    )

    print(
        f"Study-level overlap        : "
        f"{overlap}"
    )

    unique_total = len(
        set(
            study_ids(train_df)
        )
        |
        set(
            study_ids(val_df)
        )
    )

    print(
        f"Unique combined studies    : "
        f"{unique_total}"
    )

    if overlap != 0:
        raise RuntimeError(
            "Leakage audit failed."
        )

    train_df.to_csv(
        TRAIN_COHORT_OUTPUT,
        index=False,
    )

    val_df.to_csv(
        VAL_COHORT_OUTPUT,
        index=False,
    )

    print(
        "Cohort files saved: PASS"
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    section(
        "CREATING SWIN-UNETR"
    )

    model = create_model(
        part11
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters : "
        f"{total_parameters:,}"
    )

    if total_parameters != 4_078_116:
        print(
            "WARNING: parameter count differs "
            "from the validated Part84/98 model."
        )

    # -------------------------------------------------------------------------
    # Load Part98
    # -------------------------------------------------------------------------

    section(
        "LOADING PART98 STARTING CHECKPOINT"
    )

    checkpoint, missing, unexpected = (
        load_model_checkpoint(
            model,
            PART98_CHECKPOINT,
            DEVICE,
        )
    )

    print(
        f"Missing keys           : "
        f"{len(missing)}"
    )

    print(
        f"Unexpected keys        : "
        f"{len(unexpected)}"
    )

    if missing or unexpected:
        raise RuntimeError(
            "Part98 checkpoint strict load failed."
        )

    starting_epoch = (
        checkpoint.get(
            "epoch",
            None,
        )
        if isinstance(
            checkpoint,
            dict,
        )
        else None
    )

    print(
        "Part98 initialization load : PASS"
    )

    print(
        f"Starting epoch metadata : "
        f"{starting_epoch}"
    )

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    amp_enabled = (
        USE_AMP
        and DEVICE.type == "cuda"
    )

    if amp_enabled:

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    else:

        scaler = None

    # -------------------------------------------------------------------------
    # Starting evaluation
    # -------------------------------------------------------------------------

    section(
        "EVALUATING STARTING PART98 CHECKPOINT"
    )

    start_eval, _ = validate(
        model,
        val_df,
        part9,
        part11,
        save_cases=False,
    )

    print(
        f"Starting global foreground Dice : "
        f"{start_eval['global_foreground_dice']:.6f}"
    )

    print(
        f"Starting mean-case Dice          : "
        f"{start_eval['mean_case_foreground_dice']:.6f}"
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    banner(
        "STARTING ROBUST CLINICAL FINE-TUNING"
    )

    history = []

    best_dice = float(
        start_eval[
            "global_foreground_dice"
        ]
    )

    best_epoch = 0

    # Save initial best state.
    torch.save(
        {
            "epoch": 0,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "global_foreground_dice":
                best_dice,
            "configuration": {
                "epochs": EPOCHS,
                "learning_rate":
                    LEARNING_RATE,
                "weight_decay":
                    WEIGHT_DECAY,
                "train_cases":
                    len(train_df),
                "val_cases":
                    len(val_df),
                "crop_shape":
                    CROP_SHAPE,
                "full_shape":
                    FULL_SHAPE,
                "feature_size":
                    FEATURE_SIZE,
                "dice_weight":
                    DICE_WEIGHT,
                "focal_weight":
                    FOCAL_WEIGHT,
                "tversky_weight":
                    TVERSKY_WEIGHT,
                "foreground_crop_probability":
                    FOREGROUND_CROP_PROBABILITY,
                "seed":
                    SEED,
            },
            "source_checkpoint":
                str(
                    PART98_CHECKPOINT
                ),
            "source_checkpoint_sha256":
                part98_hash,
        },
        BEST_CHECKPOINT,
    )

    training_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = time.time()

        train_stats = train_one_epoch(
            model,
            train_df,
            part9,
            part11,
            optimizer,
            scaler,
        )

        val_stats, _ = validate(
            model,
            val_df,
            part9,
            part11,
            save_cases=False,
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        record = {
            "epoch":
                epoch,
            "train_loss":
                train_stats["loss"],
            "train_dice_loss":
                train_stats["dice_loss"],
            "train_focal_loss":
                train_stats["focal_loss"],
            "train_tversky_loss":
                train_stats["tversky_loss"],
            "train_successful_cases":
                train_stats[
                    "successful_cases"
                ],
            "val_loss":
                val_stats[
                    "validation_loss"
                ],
            "val_global_foreground_dice":
                val_stats[
                    "global_foreground_dice"
                ],
            "val_mean_case_dice":
                val_stats[
                    "mean_case_foreground_dice"
                ],
            "val_median_case_dice":
                val_stats[
                    "median_case_foreground_dice"
                ],
            "val_macro_foreground_dice":
                val_stats[
                    "macro_foreground_dice"
                ],
            "val_successful_cases":
                val_stats[
                    "successful_cases"
                ],
            "val_failed_cases":
                val_stats[
                    "failed_cases"
                ],
            "time_seconds":
                elapsed,
        }

        history.append(
            record
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"TrainLoss "
            f"{record['train_loss']:.6f} | "
            f"ValLoss "
            f"{record['val_loss']:.6f} | "
            f"ValGlobalDice "
            f"{record['val_global_foreground_dice']:.6f} | "
            f"ValMeanCase "
            f"{record['val_mean_case_dice']:.6f} | "
            f"Time "
            f"{elapsed:.1f}s"
        )

        # ---------------------------------------------------------------------
        # Best checkpoint selected by GLOBAL foreground Dice.
        # ---------------------------------------------------------------------

        current_dice = (
            record[
                "val_global_foreground_dice"
            ]
        )

        if (
            np.isfinite(current_dice)
            and current_dice > best_dice
        ):

            best_dice = float(
                current_dice
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch":
                        epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "global_foreground_dice":
                        best_dice,
                    "validation_metrics":
                        val_stats,
                    "configuration": {
                        "epochs": EPOCHS,
                        "learning_rate":
                            LEARNING_RATE,
                        "weight_decay":
                            WEIGHT_DECAY,
                        "train_cases":
                            len(train_df),
                        "val_cases":
                            len(val_df),
                        "crop_shape":
                            CROP_SHAPE,
                        "full_shape":
                            FULL_SHAPE,
                        "feature_size":
                            FEATURE_SIZE,
                        "dice_weight":
                            DICE_WEIGHT,
                        "focal_weight":
                            FOCAL_WEIGHT,
                        "tversky_weight":
                            TVERSKY_WEIGHT,
                        "foreground_crop_probability":
                            FOREGROUND_CROP_PROBABILITY,
                        "seed":
                            SEED,
                    },
                    "source_checkpoint":
                        str(
                            PART98_CHECKPOINT
                        ),
                    "source_checkpoint_sha256":
                        part98_hash,
                    "cohort_source":
                        cohort_source,
                },
                BEST_CHECKPOINT,
            )

            print(
                f"  NEW BEST CHECKPOINT — "
                f"global Dice "
                f"{best_dice:.6f}"
            )

    # -------------------------------------------------------------------------
    # Final checkpoint
    # -------------------------------------------------------------------------

    torch.save(
        {
            "epoch":
                EPOCHS,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "best_epoch":
                best_epoch,
            "best_global_foreground_dice":
                best_dice,
            "history":
                history,
            "source_checkpoint":
                str(
                    PART98_CHECKPOINT
                ),
            "source_checkpoint_sha256":
                part98_hash,
            "cohort_source":
                cohort_source,
        },
        FINAL_CHECKPOINT,
    )

    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------

    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Independent best-checkpoint verification
    # -------------------------------------------------------------------------

    section(
        "VERIFYING BEST CHECKPOINT"
    )

    verification_model = create_model(
        part11
    )

    _, missing, unexpected = (
        load_model_checkpoint(
            verification_model,
            BEST_CHECKPOINT,
            DEVICE,
        )
    )

    print(
        f"Strict load missing    : "
        f"{len(missing)}"
    )

    print(
        f"Strict load unexpected : "
        f"{len(unexpected)}"
    )

    if missing or unexpected:
        raise RuntimeError(
            "Independent checkpoint verification failed."
        )

    verification, case_rows = validate(
        verification_model,
        val_df,
        part9,
        part11,
        save_cases=True,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    total_training_time = (
        time.time()
        - training_start
    )

    summary = {
        "part": 100,
        "title":
            "Robust Clinical-Oriented "
            "Swin-UNETR Fine-Tuning",
        "status":
            "PASS — ROBUST CLINICAL-ORIENTED "
            "FINE-TUNING COMPLETED",
        "device":
            str(DEVICE),
        "gpu":
            (
                torch.cuda.get_device_name(0)
                if DEVICE.type == "cuda"
                else "CPU"
            ),
        "train_cases":
            len(train_df),
        "validation_cases":
            len(val_df),
        "unique_combined_studies":
            unique_total,
        "study_overlap":
            overlap,
        "cohort_source":
            cohort_source,
        "requested_train":
            REQUESTED_TRAIN,
        "requested_validation":
            REQUESTED_VAL,
        "actual_train":
            len(train_df),
        "actual_validation":
            len(val_df),
        "epochs_completed":
            EPOCHS,
        "best_epoch":
            best_epoch,
        "best_global_foreground_dice":
            best_dice,
        "verification_global_foreground_dice":
            verification[
                "global_foreground_dice"
            ],
        "verification_mean_case_dice":
            verification[
                "mean_case_foreground_dice"
            ],
        "verification_median_case_dice":
            verification[
                "median_case_foreground_dice"
            ],
        "verification_macro_foreground_dice":
            verification[
                "macro_foreground_dice"
            ],
        "verification_metrics":
            verification,
        "part98_checkpoint":
            str(
                PART98_CHECKPOINT
            ),
        "part98_checkpoint_sha256":
            part98_hash,
        "part100_best_checkpoint":
            str(
                BEST_CHECKPOINT
            ),
        "part100_final_checkpoint":
            str(
                FINAL_CHECKPOINT
            ),
        "total_training_time_seconds":
            total_training_time,
        "clinical_orientation":
            {
                "foreground_aware_crop_probability":
                    FOREGROUND_CROP_PROBABILITY,
                "dice_weight":
                    DICE_WEIGHT,
                "focal_weight":
                    FOCAL_WEIGHT,
                "tversky_weight":
                    TVERSKY_WEIGHT,
                "focal_gamma":
                    FOCAL_GAMMA,
                "tversky_alpha":
                    TVERSKY_ALPHA,
                "tversky_beta":
                    TVERSKY_BETA,
            },
        "clinical_status":
            "Research/development only; "
            "not clinically validated.",
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    report = []

    report.append(
        "PART 100 — ROBUST CLINICAL-ORIENTED "
        "SWIN-UNETR FINE-TUNING"
    )

    report.append("")
    report.append(
        "This experiment continued fine-tuning "
        "from the Part98 best Swin-UNETR checkpoint."
    )

    report.append("")
    report.append(
        "The previous implementation incorrectly "
        "required 600 unique discoverable cases."
    )

    report.append(
        "The corrected implementation uses the "
        "maximum available leakage-free cohort."
    )

    report.append("")

    report.append(
        f"Cohort source: {cohort_source}"
    )

    report.append(
        f"Training cases: {len(train_df)}"
    )

    report.append(
        f"Validation cases: {len(val_df)}"
    )

    report.append(
        f"Study overlap: {overlap}"
    )

    report.append("")

    report.append(
        "STARTING PART98 CHECKPOINT"
    )

    report.append(
        f"SHA256: {part98_hash}"
    )

    report.append("")

    report.append(
        "RESULTS"
    )

    report.append(
        f"Best epoch: {best_epoch}"
    )

    report.append(
        f"Best global foreground Dice: "
        f"{best_dice:.6f}"
    )

    report.append(
        f"Independent verification global Dice: "
        f"{verification['global_foreground_dice']:.6f}"
    )

    report.append(
        f"Mean-case Dice: "
        f"{verification['mean_case_foreground_dice']:.6f}"
    )

    report.append(
        f"Median-case Dice: "
        f"{verification['median_case_foreground_dice']:.6f}"
    )

    report.append(
        f"Macro foreground Dice: "
        f"{verification['macro_foreground_dice']:.6f}"
    )

    report.append("")

    report.append(
        "CLASS-WISE DICE"
    )

    for c in range(
        1,
        NUM_CLASSES,
    ):

        value = verification[
            f"class_{c}_dice"
        ]

        report.append(
            f"{c} — "
            f"{CLASS_NAMES[c]}: "
            f"{value:.6f}"
        )

    report.append("")

    report.append(
        "CHECKPOINT VERIFICATION"
    )

    report.append(
        f"Missing keys: {len(missing)}"
    )

    report.append(
        f"Unexpected keys: {len(unexpected)}"
    )

    report.append(
        "Strict checkpoint load: PASS"
    )

    report.append("")

    report.append(
        "IMPORTANT CLINICAL LIMITATION"
    )

    report.append(
        "This model remains a research/development "
        "system. Successful training does not establish "
        "clinical validity, regulatory approval, "
        "generalization to external institutions, "
        "or deployment readiness."
    )

    REPORT_TXT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Final console output
    # -------------------------------------------------------------------------

    banner(
        "PART 100 FINAL RESULT"
    )

    print(
        f"Training cases          : "
        f"{len(train_df)}"
    )

    print(
        f"Validation cases        : "
        f"{len(val_df)}"
    )

    print(
        f"Study overlap           : "
        f"{overlap}"
    )

    print(
        f"Epochs completed        : "
        f"{EPOCHS}"
    )

    print(
        f"Best epoch              : "
        f"{best_epoch}"
    )

    print(
        f"Best global Dice        : "
        f"{best_dice:.6f}"
    )

    print(
        f"Verification global Dice: "
        f"{verification['global_foreground_dice']:.6f}"
    )

    print(
        f"Mean-case Dice          : "
        f"{verification['mean_case_foreground_dice']:.6f}"
    )

    print(
        f"Median-case Dice        : "
        f"{verification['median_case_foreground_dice']:.6f}"
    )

    print(
        f"Macro foreground Dice   : "
        f"{verification['macro_foreground_dice']:.6f}"
    )

    print(
        f"Checkpoint strict load  : "
        f"{'PASS' if not missing and not unexpected else 'FAIL'}"
    )

    print("")
    print(
        "FINAL STATUS:"
    )

    print(
        "PASS — ROBUST CLINICAL-ORIENTED "
        "FINE-TUNING COMPLETED"
    )

    print("")
    print(
        "BEST CHECKPOINT:"
    )

    print(
        BEST_CHECKPOINT
    )

    print("")
    print(
        "FINAL CHECKPOINT:"
    )

    print(
        FINAL_CHECKPOINT
    )

    print("")
    print(
        "REPORT:"
    )

    print(
        REPORT_TXT
    )

    print("")
    print(
        "SUMMARY:"
    )

    print(
        SUMMARY_JSON
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()