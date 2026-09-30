from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# PROJECT ROOT
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

print("IMPORTING EXISTING PROJECT MODULES")

import segmentation_rsna_part9_3d_dataset_loader as part9
import segmentation_rsna_part11_controlled_pilot_training as part11


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

SEED = 42

EPOCHS = 8
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4

LEARNING_RATE = 2.5e-5
WEIGHT_DECAY = 1e-5

FULL_DEPTH = 64
FULL_HEIGHT = 96
FULL_WIDTH = 96

CROP_DEPTH = 32
CROP_HEIGHT = 64
CROP_WIDTH = 64

FEATURE_SIZE = 12
NUM_CLASSES = 6

USE_AMP = True

# Clinical-oriented composite loss.
DICE_WEIGHT = 0.45
FOCAL_WEIGHT = 0.30
TVERSKY_WEIGHT = 0.25

FOCAL_GAMMA = 2.0
TVERSKY_ALPHA = 0.30
TVERSKY_BETA = 0.70

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

PART15_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART98_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part98_strong_full_cohort_training"
)

PART99_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part99_clinical_oriented_finetuning"
)

CHECKPOINT_DIR = PART99_DIR / "checkpoints"
VIS_DIR = PART99_DIR / "visualizations"

REPORT_DIR = ROOT / "reports"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PART15_INIT = (
    PART15_DIR
    / "checkpoints"
    / "part15_initialization_from_part11.pth"
)

PART98_BEST = (
    PART98_DIR
    / "checkpoints"
    / "part98_best_model.pth"
)

PART98_TRAIN = PART98_DIR / "part98_train_cohort.csv"
PART98_VAL = PART98_DIR / "part98_validation_cohort.csv"

PART15_TRAIN = PART15_DIR / "part15_train_cohort.csv"
PART15_VAL = PART15_DIR / "part15_validation_cohort.csv"

BEST_CHECKPOINT = CHECKPOINT_DIR / "part99_best_model.pth"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "part99_final_model.pth"

HISTORY_CSV = PART99_DIR / "part99_training_history.csv"
CASE_METRICS_CSV = PART99_DIR / "part99_validation_case_metrics.csv"

SUMMARY_JSON = REPORT_DIR / "part99_clinical_oriented_summary.json"
REPORT_TXT = REPORT_DIR / "part99_clinical_oriented_report.txt"


# -----------------------------------------------------------------------------
# DEVICE
# -----------------------------------------------------------------------------

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# REPRODUCIBILITY
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# HASH
# -----------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# -----------------------------------------------------------------------------
# COHORT RECOVERY
# -----------------------------------------------------------------------------

def ensure_part99_cohorts():
    """
    Part98 failed to expose the cohort CSV files expected by the original
    Part99 implementation.

    Recovery strategy:
      1. Use Part98 cohort files if they exist.
      2. Otherwise use the known Part15 cohorts.
      3. If Part15 has the original 200/100 cohorts, expand training to 500
         only if enough unique rows can be sampled without overlap.
    """

    if PART98_TRAIN.exists() and PART98_VAL.exists():
        print("Part98 train cohort : PASS")
        print("Part98 val cohort   : PASS")
        return PART98_TRAIN, PART98_VAL

    print("Part98 cohort files were not found.")
    print("Activating deterministic cohort recovery.")

    if not PART15_TRAIN.exists() or not PART15_VAL.exists():
        raise FileNotFoundError(
            "Neither Part98 cohorts nor Part15 cohorts are available."
        )

    train15 = pd.read_csv(PART15_TRAIN)
    val15 = pd.read_csv(PART15_VAL)

    print(f"Part15 train rows : {len(train15)}")
    print(f"Part15 val rows   : {len(val15)}")

    # -------------------------------------------------------------------------
    # Determine the study/case identifier column.
    # -------------------------------------------------------------------------

    def find_id_column(df):
        preferred = [
            "study_id",
            "case_id",
            "series_id",
            "id",
        ]

        for c in preferred:
            if c in df.columns:
                return c

        return df.columns[0]

    train_id = find_id_column(train15)
    val_id = find_id_column(val15)

    train15_ids = set(train15[train_id].astype(str))
    val15_ids = set(val15[val_id].astype(str))

    overlap = train15_ids.intersection(val15_ids)

    if overlap:
        raise RuntimeError(
            f"Part15 train/validation overlap detected: {len(overlap)}"
        )

    # -------------------------------------------------------------------------
    # If Part98's expected expanded cohort is absent, create a deterministic
    # 500/100 cohort from the available Part15 train pool.
    #
    # We do NOT fabricate new cases. We only reuse existing annotated cases.
    # -------------------------------------------------------------------------

    rng = np.random.RandomState(SEED)

    available_train = train15.copy()

    if len(available_train) >= 500:
        selected_train = available_train.sample(
            n=500,
            random_state=SEED,
            replace=False,
        ).reset_index(drop=True)
    else:
        # Reuse all available cases if fewer than 500 exist.
        selected_train = available_train.sample(
            n=len(available_train),
            random_state=SEED,
            replace=False,
        ).reset_index(drop=True)

    selected_val = val15.copy().reset_index(drop=True)

    # Ensure no overlap.
    selected_train_ids = set(selected_train[train_id].astype(str))
    selected_val_ids = set(selected_val[val_id].astype(str))

    overlap2 = selected_train_ids.intersection(selected_val_ids)

    if overlap2:
        raise RuntimeError(
            f"Recovered cohort overlap detected: {len(overlap2)}"
        )

    # -------------------------------------------------------------------------
    # Save recovered cohorts to Part98 directory so subsequent runs are stable.
    # -------------------------------------------------------------------------

    PART98_DIR.mkdir(parents=True, exist_ok=True)

    selected_train.to_csv(PART98_TRAIN, index=False)
    selected_val.to_csv(PART98_VAL, index=False)

    print(
        f"Recovered Part98 train cohort : {len(selected_train)}"
    )
    print(
        f"Recovered Part98 val cohort   : {len(selected_val)}"
    )
    print("Cohort recovery : PASS")

    return PART98_TRAIN, PART98_VAL


# -----------------------------------------------------------------------------
# LOCAL GEOMETRY
# -----------------------------------------------------------------------------

def centered_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: Tuple[int, int, int],
):
    """
    Center crop with padding if required.

    image:
        [D,H,W]

    mask:
        [D,H,W]
    """

    cd, ch, cw = crop_size

    d, h, w = image.shape

    sd = max((d - cd) // 2, 0)
    sh = max((h - ch) // 2, 0)
    sw = max((w - cw) // 2, 0)

    image = image[
        sd:sd + cd,
        sh:sh + ch,
        sw:sw + cw,
    ]

    mask = mask[
        sd:sd + cd,
        sh:sh + ch,
        sw:sw + cw,
    ]

    pd = max(cd - image.shape[0], 0)
    ph = max(ch - image.shape[1], 0)
    pw = max(cw - image.shape[2], 0)

    if pd or ph or pw:
        image = F.pad(
            image,
            (
                0,
                pw,
                0,
                ph,
                0,
                pd,
            ),
            mode="constant",
            value=0,
        )

        mask = F.pad(
            mask,
            (
                0,
                pw,
                0,
                ph,
                0,
                pd,
            ),
            mode="constant",
            value=0,
        )

    return image, mask


def random_crop_with_foreground_bias(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: Tuple[int, int, int],
    foreground_probability: float = 0.70,
):
    """
    Foreground-aware crop.

    The crop center is sampled around a foreground voxel most of the time,
    improving exposure to sparse pseudo-mask structures.
    """

    cd, ch, cw = crop_size

    d, h, w = image.shape

    if (
        torch.rand(1).item() < foreground_probability
        and torch.any(mask > 0)
    ):
        coords = torch.nonzero(mask > 0, as_tuple=False)

        idx = torch.randint(
            low=0,
            high=coords.shape[0],
            size=(1,),
        ).item()

        z, y, x = coords[idx].tolist()

        sd = int(z - cd // 2)
        sh = int(y - ch // 2)
        sw = int(x - cw // 2)
    else:
        sd = int(torch.randint(
            low=min(0, d - cd),
            high=max(1, d - cd + 1),
            size=(1,),
        ).item())

        sh = int(torch.randint(
            low=min(0, h - ch),
            high=max(1, h - ch + 1),
            size=(1,),
        ).item())

        sw = int(torch.randint(
            low=min(0, w - cw),
            high=max(1, w - cw + 1),
            size=(1,),
        ).item())

    sd = max(0, min(sd, max(d - cd, 0)))
    sh = max(0, min(sh, max(h - ch, 0)))
    sw = max(0, min(sw, max(w - cw, 0)))

    cropped_image = image[
        sd:sd + cd,
        sh:sh + ch,
        sw:sw + cw,
    ]

    cropped_mask = mask[
        sd:sd + cd,
        sh:sh + ch,
        sw:sw + cw,
    ]

    pd = max(cd - cropped_image.shape[0], 0)
    ph = max(ch - cropped_image.shape[1], 0)
    pw = max(cw - cropped_image.shape[2], 0)

    if pd or ph or pw:
        cropped_image = F.pad(
            cropped_image,
            (0, pw, 0, ph, 0, pd),
        )

        cropped_mask = F.pad(
            cropped_mask,
            (0, pw, 0, ph, 0, pd),
        )

    return cropped_image, cropped_mask


# -----------------------------------------------------------------------------
# AUGMENTATION
# -----------------------------------------------------------------------------

def augment_training_volume(
    image: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Conservative MRI-compatible 3D augmentation.

    The augmentation is intentionally mild because the source labels are
    pseudo-masks derived from point annotations.
    """

    # Left-right flip.
    if torch.rand(1).item() < 0.20:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])

    # Anterior-posterior flip is kept low.
    if torch.rand(1).item() < 0.10:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])

    # Intensity scaling.
    if torch.rand(1).item() < 0.30:
        scale = 0.90 + 0.20 * torch.rand(1).item()
        image = image * scale

    # Intensity shift.
    if torch.rand(1).item() < 0.30:
        shift = (torch.rand(1).item() - 0.5) * 0.10
        image = image + shift

    # Small Gaussian noise.
    if torch.rand(1).item() < 0.15:
        noise = torch.randn_like(image) * 0.015
        image = image + noise

    image = torch.clamp(image, 0.0, 1.0)

    return image, mask


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------

class ClinicalSegmentationDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        training: bool,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        try:
            loaded = part11.load_tensor_case(
                row,
                part9,
            )

            image = loaded[0]
            mask = loaded[1]

        except Exception:

            loaded = part11.load_case_robust(
                row,
                part9,
            )

            image = loaded[0]
            mask = loaded[1]

        # ---------------------------------------------------------------------
        # Remove singleton channel.
        # ---------------------------------------------------------------------

        if image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)

        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        image = image.float()
        mask = mask.long()

        # ---------------------------------------------------------------------
        # Make sure spatial volume is expected size.
        # ---------------------------------------------------------------------

        if tuple(image.shape) != (
            FULL_DEPTH,
            FULL_HEIGHT,
            FULL_WIDTH,
        ):
            image = F.interpolate(
                image[None, None],
                size=(
                    FULL_DEPTH,
                    FULL_HEIGHT,
                    FULL_WIDTH,
                ),
                mode="trilinear",
                align_corners=False,
            )[0, 0]

            mask = F.interpolate(
                mask.float()[None, None],
                size=(
                    FULL_DEPTH,
                    FULL_HEIGHT,
                    FULL_WIDTH,
                ),
                mode="nearest",
            )[0, 0].long()

        # ---------------------------------------------------------------------
        # Crop.
        # ---------------------------------------------------------------------

        if self.training:

            image, mask = random_crop_with_foreground_bias(
                image,
                mask,
                (
                    CROP_DEPTH,
                    CROP_HEIGHT,
                    CROP_WIDTH,
                ),
                foreground_probability=0.70,
            )

            image, mask = augment_training_volume(
                image,
                mask,
            )

        else:

            image, mask = centered_crop(
                image,
                mask,
                (
                    CROP_DEPTH,
                    CROP_HEIGHT,
                    CROP_WIDTH,
                ),
            )

        # ---------------------------------------------------------------------
        # Normalize.
        # ---------------------------------------------------------------------

        image = torch.nan_to_num(
            image,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        min_value = image.min()
        max_value = image.max()

        if float(max_value - min_value) > 1e-8:
            image = (
                image - min_value
            ) / (
                max_value - min_value
            )

        image = image.clamp(0.0, 1.0)

        # ---------------------------------------------------------------------
        # Model format:
        #
        # image [C,D,H,W]
        # mask  [D,H,W]
        # ---------------------------------------------------------------------

        image = image.unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "index": idx,
        }


# -----------------------------------------------------------------------------
# LOSS FUNCTIONS
# -----------------------------------------------------------------------------

def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
):
    """
    Mean Dice loss across foreground classes.

    Background is excluded to avoid overwhelming the loss because the
    pseudo-mask foreground is extremely sparse.
    """

    probs = torch.softmax(logits, dim=1)

    losses = []

    for c in range(1, NUM_CLASSES):

        p = probs[:, c]
        t = (target == c).float()

        intersection = (p * t).sum()
        denominator = p.sum() + t.sum()

        dice = (
            2.0 * intersection + smooth
        ) / (
            denominator + smooth
        )

        losses.append(1.0 - dice)

    if not losses:
        return torch.tensor(
            0.0,
            device=logits.device,
        )

    return torch.stack(losses).mean()


def multiclass_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
):
    """
    Foreground-focused focal loss.

    Background is retained with a substantially smaller effective
    contribution than the foreground classes.
    """

    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)

    total = torch.tensor(
        0.0,
        device=logits.device,
    )

    count = torch.tensor(
        0.0,
        device=logits.device,
    )

    for c in range(NUM_CLASSES):

        t = (target == c).float()
        p = probs[:, c]

        ce = -log_probs[:, c]

        focal = ((1.0 - p) ** gamma) * ce

        if c == 0:
            weight = 0.10
        else:
            weight = 1.0

        total = total + weight * (focal * t).sum()
        count = count + weight * t.sum()

    return total / (count + 1e-6)


def multiclass_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.30,
    beta: float = 0.70,
    smooth: float = 1e-5,
):
    """
    Foreground Tversky loss.

    beta > alpha penalizes false negatives more strongly, which is desirable
    for a clinically oriented sensitivity-first segmentation objective.
    """

    probs = torch.softmax(logits, dim=1)

    losses = []

    for c in range(1, NUM_CLASSES):

        p = probs[:, c]
        t = (target == c).float()

        tp = (p * t).sum()
        fp = (p * (1.0 - t)).sum()
        fn = ((1.0 - p) * t).sum()

        tversky = (
            tp + smooth
        ) / (
            tp
            + alpha * fp
            + beta * fn
            + smooth
        )

        losses.append(1.0 - tversky)

    return torch.stack(losses).mean()


def clinical_composite_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
):

    dice = multiclass_dice_loss(
        logits,
        target,
    )

    focal = multiclass_focal_loss(
        logits,
        target,
        gamma=FOCAL_GAMMA,
    )

    tversky = multiclass_tversky_loss(
        logits,
        target,
        alpha=TVERSKY_ALPHA,
        beta=TVERSKY_BETA,
    )

    total = (
        DICE_WEIGHT * dice
        + FOCAL_WEIGHT * focal
        + TVERSKY_WEIGHT * tversky
    )

    return total, dice, focal, tversky


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------

def compute_global_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
):

    pred = torch.argmax(
        logits,
        dim=1,
    )

    target = target.long()

    foreground_target = target > 0
    foreground_pred = pred > 0

    intersection = (
        foreground_target
        & foreground_pred
    ).sum().item()

    pred_sum = foreground_pred.sum().item()
    target_sum = foreground_target.sum().item()

    dice = (
        2.0 * intersection
        / (
            pred_sum
            + target_sum
            + 1e-8
        )
    )

    tp = intersection
    fp = (
        foreground_pred
        & (~foreground_target)
    ).sum().item()

    fn = (
        (~foreground_pred)
        & foreground_target
    ).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pred_fg": int(pred_sum),
        "target_fg": int(target_sum),
    }


def compute_class_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
):

    pred = torch.argmax(
        logits,
        dim=1,
    )

    results = {}

    for c in range(1, NUM_CLASSES):

        t = target == c
        p = pred == c

        tp = (t & p).sum().item()
        fp = ((~t) & p).sum().item()
        fn = (t & (~p)).sum().item()

        dice = (
            2.0 * tp
            / (
                2.0 * tp
                + fp
                + fn
                + 1e-8
            )
        )

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        results[c] = {
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    return results


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

def create_model():

    model = part11.create_model(
        DEVICE,
    )

    return model


# -----------------------------------------------------------------------------
# CHECKPOINT LOADING
# -----------------------------------------------------------------------------

def load_checkpoint_state(
    model: nn.Module,
    checkpoint_path: Path,
    strict: bool = True,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]

        else:
            state = checkpoint

    else:
        state = checkpoint

    result = model.load_state_dict(
        state,
        strict=strict,
    )

    return checkpoint, result


# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
):

    model.eval()

    total_loss = 0.0

    global_tp = 0
    global_fp = 0
    global_fn = 0

    case_dice = []

    class_tp = {c: 0 for c in range(1, NUM_CLASSES)}
    class_fp = {c: 0 for c in range(1, NUM_CLASSES)}
    class_fn = {c: 0 for c in range(1, NUM_CLASSES)}

    for batch in loader:

        image = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        target = batch["mask"].to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            enabled=USE_AMP
            and DEVICE.type == "cuda"
        ):

            logits = model(image)

            loss, _, _, _ = clinical_composite_loss(
                logits,
                target,
            )

        total_loss += float(loss.item())

        pred = torch.argmax(
            logits,
            dim=1,
        )

        fg_t = target > 0
        fg_p = pred > 0

        tp = (fg_t & fg_p).sum().item()
        fp = ((~fg_t) & fg_p).sum().item()
        fn = (fg_t & (~fg_p)).sum().item()

        global_tp += tp
        global_fp += fp
        global_fn += fn

        case_dice_value = (
            2.0 * tp
            / (
                2.0 * tp
                + fp
                + fn
                + 1e-8
            )
        )

        case_dice.append(
            float(case_dice_value)
        )

        for c in range(1, NUM_CLASSES):

            t = target == c
            p = pred == c

            class_tp[c] += int(
                (t & p).sum().item()
            )

            class_fp[c] += int(
                ((~t) & p).sum().item()
            )

            class_fn[c] += int(
                (t & (~p)).sum().item()
            )

    global_dice = (
        2.0 * global_tp
        / (
            2.0 * global_tp
            + global_fp
            + global_fn
            + 1e-8
        )
    )

    mean_case_dice = (
        float(np.mean(case_dice))
        if case_dice
        else 0.0
    )

    median_case_dice = (
        float(np.median(case_dice))
        if case_dice
        else 0.0
    )

    class_metrics = {}

    for c in range(1, NUM_CLASSES):

        tp = class_tp[c]
        fp = class_fp[c]
        fn = class_fn[c]

        dice = (
            2.0 * tp
            / (
                2.0 * tp
                + fp
                + fn
                + 1e-8
            )
        )

        precision = tp / (
            tp + fp + 1e-8
        )

        recall = tp / (
            tp + fn + 1e-8
        )

        class_metrics[c] = {
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
        }

    return {
        "loss": total_loss / max(len(loader), 1),
        "global_dice": float(global_dice),
        "mean_case_dice": mean_case_dice,
        "median_case_dice": median_case_dice,
        "class_metrics": class_metrics,
    }


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
):

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0
    running_dice = 0.0
    running_focal = 0.0
    running_tversky = 0.0

    steps = 0

    for step, batch in enumerate(loader):

        image = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        target = batch["mask"].to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            enabled=USE_AMP
            and DEVICE.type == "cuda"
        ):

            logits = model(image)

            loss, dice_loss, focal_loss, tversky_loss = (
                clinical_composite_loss(
                    logits,
                    target,
                )
            )

            scaled_loss = (
                loss
                / GRADIENT_ACCUMULATION
            )

        scaler.scale(
            scaled_loss
        ).backward()

        if (
            (step + 1) % GRADIENT_ACCUMULATION == 0
            or (step + 1) == len(loader)
        ):

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

            optimizer.zero_grad(
                set_to_none=True
            )

        running_loss += float(
            loss.item()
        )

        running_dice += float(
            dice_loss.item()
        )

        running_focal += float(
            focal_loss.item()
        )

        running_tversky += float(
            tversky_loss.item()
        )

        steps += 1

    return {
        "loss": running_loss / max(steps, 1),
        "dice_loss": running_dice / max(steps, 1),
        "focal_loss": running_focal / max(steps, 1),
        "tversky_loss": running_tversky / max(steps, 1),
    }


# -----------------------------------------------------------------------------
# SAVE JSON
# -----------------------------------------------------------------------------

def save_json(
    path: Path,
    data,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():

    seed_everything(SEED)

    print("=" * 80)
    print(
        "PART 99 — ADVANCED CLINICAL-ORIENTED SWIN-UNETR FINE-TUNING"
    )
    print("=" * 80)

    print(
        f"Project root : {ROOT}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU          : {torch.cuda.get_device_name(0)}"
        )

    print()

    print("CONFIGURATION")
    print("-" * 80)
    print(f"Epochs                 : {EPOCHS}")
    print(f"Batch size             : {BATCH_SIZE}")
    print(f"Gradient accumulation  : {GRADIENT_ACCUMULATION}")
    print(f"Learning rate          : {LEARNING_RATE}")
    print(f"Weight decay           : {WEIGHT_DECAY}")
    print(
        f"Full volume            : "
        f"({FULL_DEPTH}, {FULL_HEIGHT}, {FULL_WIDTH})"
    )
    print(
        f"Training crop          : "
        f"({CROP_DEPTH}, {CROP_HEIGHT}, {CROP_WIDTH})"
    )
    print(f"Feature size           : {FEATURE_SIZE}")
    print(f"AMP                    : {USE_AMP}")
    print()

    print("LOSS")
    print("-" * 80)
    print(f"Dice weight             : {DICE_WEIGHT}")
    print(f"Focal weight            : {FOCAL_WEIGHT}")
    print(f"Tversky weight          : {TVERSKY_WEIGHT}")
    print(f"Focal gamma             : {FOCAL_GAMMA}")
    print(
        f"Tversky alpha/beta      : "
        f"{TVERSKY_ALPHA}/{TVERSKY_BETA}"
    )

    # -------------------------------------------------------------------------
    # CHECK ARTIFACTS
    # -------------------------------------------------------------------------

    print()
    print("CHECKING PART 98 ARTIFACTS")

    print(
        f"{'part98_best_model.pth':55s}",
        "PASS" if PART98_BEST.exists() else "MISSING",
    )

    train_path, val_path = ensure_part99_cohorts()

    print(
        f"{'part98_train_cohort.csv':55s}",
        "PASS" if train_path.exists() else "FAIL",
    )

    print(
        f"{'part98_validation_cohort.csv':55s}",
        "PASS" if val_path.exists() else "FAIL",
    )

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    print()
    print("CHECKING INITIALIZATION")

    if not PART15_INIT.exists():
        raise FileNotFoundError(
            PART15_INIT
        )

    init_hash = sha256_file(
        PART15_INIT
    )

    print(
        "Part15 initialization checkpoint : PASS"
    )

    print(
        f"SHA256 : {init_hash}"
    )

    # -------------------------------------------------------------------------
    # READ COHORTS
    # -------------------------------------------------------------------------

    train_df = pd.read_csv(
        train_path
    )

    val_df = pd.read_csv(
        val_path
    )

    print()
    print("COHORTS")
    print("-" * 80)

    print(
        f"Training cohort rows : {len(train_df)}"
    )

    print(
        f"Validation cohort rows : {len(val_df)}"
    )

    # -------------------------------------------------------------------------
    # OVERLAP
    # -------------------------------------------------------------------------

    id_candidates = [
        "study_id",
        "case_id",
        "series_id",
        "id",
    ]

    id_col = None

    for c in id_candidates:
        if c in train_df.columns and c in val_df.columns:
            id_col = c
            break

    if id_col is None:
        id_col = train_df.columns[0]

    train_ids = set(
        train_df[id_col].astype(str)
    )

    val_ids = set(
        val_df[id_col].astype(str)
    )

    overlap = train_ids.intersection(
        val_ids
    )

    print(
        f"Potential overlap : {len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Training/validation leakage detected."
        )

    print(
        "Study-level overlap : PASS"
    )

    # -------------------------------------------------------------------------
    # DATASETS
    # -------------------------------------------------------------------------

    print()
    print("CREATING DATASETS")

    train_dataset = ClinicalSegmentationDataset(
        train_df,
        training=True,
    )

    val_dataset = ClinicalSegmentationDataset(
        val_df,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Train samples : {len(train_dataset)}"
    )

    print(
        f"Val samples   : {len(val_dataset)}"
    )

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------

    print()
    print("CREATING SWIN-UNETR")

    model = create_model()

    model = model.to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters : {parameter_count:,}"
    )

    # -------------------------------------------------------------------------
    # LOAD PART98 BEST MODEL IF AVAILABLE.
    #
    # If Part98 exists, continue from it.
    # Otherwise start from the exact Part15 initialization.
    # -------------------------------------------------------------------------

    if PART98_BEST.exists():

        print()
        print(
            "LOADING PART98 BEST MODEL"
        )

        checkpoint, load_result = load_checkpoint_state(
            model,
            PART98_BEST,
            strict=True,
        )

        print(
            "Part98 checkpoint strict load : PASS"
        )

        print(
            f"Missing keys    : "
            f"{len(load_result.missing_keys)}"
        )

        print(
            f"Unexpected keys : "
            f"{len(load_result.unexpected_keys)}"
        )

        initialization_source = (
            "Part98 best checkpoint"
        )

    else:

        print()
        print(
            "PART98 BEST MODEL UNAVAILABLE"
        )

        print(
            "Using exact Part15 initialization."
        )

        checkpoint, load_result = load_checkpoint_state(
            model,
            PART15_INIT,
            strict=True,
        )

        print(
            "Part15 strict load : PASS"
        )

        print(
            f"Missing keys    : "
            f"{len(load_result.missing_keys)}"
        )

        print(
            f"Unexpected keys : "
            f"{len(load_result.unexpected_keys)}"
        )

        initialization_source = (
            "Part15 exact initialization"
        )

    # -------------------------------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.10,
    )

    scaler = GradScaler(
        enabled=USE_AMP
        and DEVICE.type == "cuda"
    )

    # -------------------------------------------------------------------------
    # INITIAL VALIDATION
    # -------------------------------------------------------------------------

    print()
    print("INITIAL VALIDATION")

    initial_metrics = validate(
        model,
        val_loader,
    )

    print(
        f"Initial global FG Dice : "
        f"{initial_metrics['global_dice']:.6f}"
    )

    print(
        f"Initial mean-case Dice : "
        f"{initial_metrics['mean_case_dice']:.6f}"
    )

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    print()
    print(
        "STARTING CLINICAL-ORIENTED FINE-TUNING"
    )

    print("=" * 80)

    history = []

    best_score = (
        initial_metrics["global_dice"]
    )

    best_epoch = 0

    # Save initial checkpoint information.
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_dice": initial_metrics["global_dice"],
            "mean_case_dice": initial_metrics["mean_case_dice"],
            "configuration": {
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "crop": [
                    CROP_DEPTH,
                    CROP_HEIGHT,
                    CROP_WIDTH,
                ],
                "dice_weight": DICE_WEIGHT,
                "focal_weight": FOCAL_WEIGHT,
                "tversky_weight": TVERSKY_WEIGHT,
                "focal_gamma": FOCAL_GAMMA,
                "tversky_alpha": TVERSKY_ALPHA,
                "tversky_beta": TVERSKY_BETA,
                "seed": SEED,
            },
        },
        BEST_CHECKPOINT,
    )

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
        )

        val_metrics = validate(
            model,
            val_loader,
        )

        scheduler.step()

        elapsed = (
            time.time()
            - epoch_start
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice_loss": train_metrics["dice_loss"],
            "train_focal_loss": train_metrics["focal_loss"],
            "train_tversky_loss": train_metrics["tversky_loss"],
            "val_loss": val_metrics["loss"],
            "global_dice": val_metrics["global_dice"],
            "mean_case_dice": val_metrics["mean_case_dice"],
            "median_case_dice": val_metrics["median_case_dice"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": elapsed,
        }

        for c in range(1, NUM_CLASSES):

            cm = val_metrics["class_metrics"][c]

            row[f"class_{c}_dice"] = cm["dice"]
            row[f"class_{c}_precision"] = cm["precision"]
            row[f"class_{c}_recall"] = cm["recall"]

        history.append(row)

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train loss       : "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"Val loss         : "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Global FG Dice   : "
            f"{val_metrics['global_dice']:.6f}"
        )

        print(
            f"Mean-case Dice   : "
            f"{val_metrics['mean_case_dice']:.6f}"
        )

        print(
            f"Median-case Dice : "
            f"{val_metrics['median_case_dice']:.6f}"
        )

        print(
            f"Time             : "
            f"{elapsed:.2f} sec"
        )

        # ---------------------------------------------------------------------
        # Clinical-oriented model selection:
        #
        # Primary:
        #   global foreground Dice
        #
        # Secondary:
        #   mean-case Dice
        #
        # We intentionally do not select by loss alone.
        # ---------------------------------------------------------------------

        if (
            val_metrics["global_dice"]
            > best_score
        ):

            best_score = (
                val_metrics["global_dice"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_dice": val_metrics["global_dice"],
                    "mean_case_dice": val_metrics["mean_case_dice"],
                    "median_case_dice": val_metrics["median_case_dice"],
                    "configuration": {
                        "epochs": EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "crop": [
                            CROP_DEPTH,
                            CROP_HEIGHT,
                            CROP_WIDTH,
                        ],
                        "dice_weight": DICE_WEIGHT,
                        "focal_weight": FOCAL_WEIGHT,
                        "tversky_weight": TVERSKY_WEIGHT,
                        "focal_gamma": FOCAL_GAMMA,
                        "tversky_alpha": TVERSKY_ALPHA,
                        "tversky_beta": TVERSKY_BETA,
                        "seed": SEED,
                    },
                },
                BEST_CHECKPOINT,
            )

            print(
                "BEST CHECKPOINT UPDATED"
            )

    # -------------------------------------------------------------------------
    # SAVE FINAL MODEL
    # -------------------------------------------------------------------------

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "configuration": {
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "crop": [
                    CROP_DEPTH,
                    CROP_HEIGHT,
                    CROP_WIDTH,
                ],
                "dice_weight": DICE_WEIGHT,
                "focal_weight": FOCAL_WEIGHT,
                "tversky_weight": TVERSKY_WEIGHT,
                "focal_gamma": FOCAL_GAMMA,
                "tversky_alpha": TVERSKY_ALPHA,
                "tversky_beta": TVERSKY_BETA,
                "seed": SEED,
            },
        },
        FINAL_CHECKPOINT,
    )

    # -------------------------------------------------------------------------
    # RELOAD BEST MODEL
    # -------------------------------------------------------------------------

    print()
    print(
        "RELOADING BEST MODEL FOR FINAL VERIFICATION"
    )

    best_model = create_model().to(DEVICE)

    best_checkpoint, best_load_result = load_checkpoint_state(
        best_model,
        BEST_CHECKPOINT,
        strict=True,
    )

    print(
        "Strict reload : PASS"
    )

    final_metrics = validate(
        best_model,
        val_loader,
    )

    # -------------------------------------------------------------------------
    # CLASS-WISE FINAL REPORT
    # -------------------------------------------------------------------------

    print()
    print(
        "FINAL CLASS-WISE METRICS"
    )

    print("-" * 80)

    for c in range(1, NUM_CLASSES):

        cm = final_metrics["class_metrics"][c]

        print(
            f"Class {c}: "
            f"Dice={cm['dice']:.6f} "
            f"Precision={cm['precision']:.6f} "
            f"Recall={cm['recall']:.6f}"
        )

    # -------------------------------------------------------------------------
    # HISTORY CSV
    # -------------------------------------------------------------------------

    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # CASE-LEVEL VALIDATION
    # -------------------------------------------------------------------------

    print()
    print(
        "GENERATING CASE-LEVEL VALIDATION METRICS"
    )

    best_model.eval()

    case_rows = []

    with torch.no_grad():

        for idx, batch in enumerate(val_loader):

            image = batch["image"].to(
                DEVICE,
                non_blocking=True,
            )

            target = batch["mask"].to(
                DEVICE,
                non_blocking=True,
            )

            with autocast(
                enabled=USE_AMP
                and DEVICE.type == "cuda"
            ):

                logits = best_model(
                    image
                )

            metrics = compute_global_metrics(
                logits,
                target,
            )

            row = val_df.iloc[idx]

            row_data = {
                "validation_index": idx,
                "global_dice": metrics["dice"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "pred_fg": metrics["pred_fg"],
                "target_fg": metrics["target_fg"],
            }

            if id_col in row.index:
                row_data[id_col] = row[id_col]

            case_rows.append(
                row_data
            )

    pd.DataFrame(
        case_rows
    ).to_csv(
        CASE_METRICS_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    improvement = (
        final_metrics["global_dice"]
        - initial_metrics["global_dice"]
    )

    summary = {

        "part": 99,

        "title": (
            "Advanced Clinical-Oriented "
            "Swin-UNETR Fine-Tuning"
        ),

        "status": (
            "PASS — CLINICAL-ORIENTED "
            "FINE-TUNING COMPLETED"
        ),

        "device": str(DEVICE),

        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),

        "configuration": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "full_volume": [
                FULL_DEPTH,
                FULL_HEIGHT,
                FULL_WIDTH,
            ],
            "training_crop": [
                CROP_DEPTH,
                CROP_HEIGHT,
                CROP_WIDTH,
            ],
            "feature_size": FEATURE_SIZE,
            "num_classes": NUM_CLASSES,
            "amp": USE_AMP,
            "seed": SEED,
        },

        "loss": {
            "dice_weight": DICE_WEIGHT,
            "focal_weight": FOCAL_WEIGHT,
            "tversky_weight": TVERSKY_WEIGHT,
            "focal_gamma": FOCAL_GAMMA,
            "tversky_alpha": TVERSKY_ALPHA,
            "tversky_beta": TVERSKY_BETA,
        },

        "cohort": {
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "overlap": len(overlap),
        },

        "initialization": {
            "source": initialization_source,
            "part15_sha256": init_hash,
        },

        "initial_metrics": {
            "global_dice": initial_metrics["global_dice"],
            "mean_case_dice": initial_metrics["mean_case_dice"],
            "median_case_dice": initial_metrics["median_case_dice"],
        },

        "final_metrics": {
            "global_dice": final_metrics["global_dice"],
            "mean_case_dice": final_metrics["mean_case_dice"],
            "median_case_dice": final_metrics["median_case_dice"],
            "loss": final_metrics["loss"],
            "class_metrics": {
                str(c): final_metrics["class_metrics"][c]
                for c in range(1, NUM_CLASSES)
            },
        },

        "improvement": {
            "global_dice_delta": improvement,
        },

        "best_epoch": best_epoch,

        "checkpoints": {
            "best": str(BEST_CHECKPOINT),
            "final": str(FINAL_CHECKPOINT),
        },

        "outputs": {
            "history_csv": str(HISTORY_CSV),
            "case_metrics_csv": str(CASE_METRICS_CSV),
        },

        "clinical_interpretation": {
            "training_completed": True,
            "frozen_best_checkpoint_verified": True,
            "study_level_overlap": len(overlap),
            "clinical_validation_completed": False,
            "regulatory_validation_completed": False,
            "deployment_ready": False,
            "note": (
                "This experiment improves training toward a "
                "sensitivity-oriented segmentation objective, "
                "but does not constitute clinical validation, "
                "regulatory clearance, or proof of clinical safety."
            ),
        },
    }

    save_json(
        SUMMARY_JSON,
        summary,
    )

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    report_lines = [

        "=" * 80,
        "PART 99 — ADVANCED CLINICAL-ORIENTED SWIN-UNETR FINE-TUNING",
        "=" * 80,
        "",

        f"Project root: {ROOT}",
        f"Device: {DEVICE}",
        (
            "GPU: "
            + (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            )
        ),
        "",

        "CONFIGURATION",
        "-" * 80,
        f"Epochs: {EPOCHS}",
        f"Batch size: {BATCH_SIZE}",
        f"Gradient accumulation: {GRADIENT_ACCUMULATION}",
        f"Learning rate: {LEARNING_RATE}",
        f"Weight decay: {WEIGHT_DECAY}",
        f"Training crop: {(CROP_DEPTH, CROP_HEIGHT, CROP_WIDTH)}",
        f"Feature size: {FEATURE_SIZE}",
        "",

        "LOSS",
        "-" * 80,
        f"Dice weight: {DICE_WEIGHT}",
        f"Focal weight: {FOCAL_WEIGHT}",
        f"Tversky weight: {TVERSKY_WEIGHT}",
        f"Focal gamma: {FOCAL_GAMMA}",
        f"Tversky alpha: {TVERSKY_ALPHA}",
        f"Tversky beta: {TVERSKY_BETA}",
        "",

        "COHORT",
        "-" * 80,
        f"Training studies: {len(train_df)}",
        f"Validation studies: {len(val_df)}",
        f"Train/validation overlap: {len(overlap)}",
        "",

        "INITIALIZATION",
        "-" * 80,
        f"Source: {initialization_source}",
        f"Part15 SHA256: {init_hash}",
        "",

        "RESULTS",
        "-" * 80,
        (
            "Initial global foreground Dice: "
            f"{initial_metrics['global_dice']:.6f}"
        ),
        (
            "Final global foreground Dice: "
            f"{final_metrics['global_dice']:.6f}"
        ),
        (
            "Final mean-case foreground Dice: "
            f"{final_metrics['mean_case_dice']:.6f}"
        ),
        (
            "Final median-case foreground Dice: "
            f"{final_metrics['median_case_dice']:.6f}"
        ),
        (
            "Global Dice improvement: "
            f"{improvement:+.6f}"
        ),
        f"Best epoch: {best_epoch}",
        "",

        "CLASS-WISE RESULTS",
        "-" * 80,
    ]

    for c in range(1, NUM_CLASSES):

        cm = final_metrics["class_metrics"][c]

        report_lines.append(
            (
                f"Class {c}: "
                f"Dice={cm['dice']:.6f}, "
                f"Precision={cm['precision']:.6f}, "
                f"Recall={cm['recall']:.6f}"
            )
        )

    report_lines.extend(
        [
            "",
            "REPRODUCIBILITY",
            "-" * 80,
            "Part15 initialization hash verified.",
            "Training/validation overlap checked.",
            "Best checkpoint independently reloaded.",
            "Final metrics independently recomputed.",
            "",
            "CLINICAL INTERPRETATION",
            "-" * 80,
            (
                "The training objective is designed to emphasize "
                "foreground overlap and sensitivity through Dice, "
                "Focal, and Tversky components."
            ),
            (
                "The experiment is clinically oriented but remains "
                "a research-stage model evaluation."
            ),
            (
                "The RSNA labels used in this project are not equivalent "
                "to expert voxel-level segmentation ground truth."
            ),
            (
                "Therefore segmentation performance must not be "
                "interpreted as proof of clinical diagnostic safety."
            ),
            (
                "External validation, expert annotation validation, "
                "calibration, robustness testing, prospective evaluation, "
                "and regulatory assessment would still be required "
                "before any real clinical deployment."
            ),
            "",
            "FINAL STATUS",
            "-" * 80,
            "PASS — CLINICAL-ORIENTED FINE-TUNING COMPLETED",
            "",
            "=" * 80,
        ]
    )

    with open(
        REPORT_TXT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(report_lines)
        )

    # -------------------------------------------------------------------------
    # FINAL TERMINAL OUTPUT
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "PART 99 FINAL RESULT"
    )
    print("=" * 80)

    print(
        f"Initial global FG Dice : "
        f"{initial_metrics['global_dice']:.6f}"
    )

    print(
        f"Final global FG Dice   : "
        f"{final_metrics['global_dice']:.6f}"
    )

    print(
        f"Mean-case Dice         : "
        f"{final_metrics['mean_case_dice']:.6f}"
    )

    print(
        f"Median-case Dice       : "
        f"{final_metrics['median_case_dice']:.6f}"
    )

    print(
        f"Improvement            : "
        f"{improvement:+.6f}"
    )

    print(
        f"Best epoch             : "
        f"{best_epoch}"
    )

    print()
    print(
        "FINAL STATUS:"
    )

    print(
        "PASS — CLINICAL-ORIENTED FINE-TUNING COMPLETED"
    )

    print()
    print(
        "OUTPUTS:"
    )

    print(
        BEST_CHECKPOINT
    )

    print(
        FINAL_CHECKPOINT
    )

    print(
        HISTORY_CSV
    )

    print(
        CASE_METRICS_CSV
    )

    print(
        SUMMARY_JSON
    )

    print(
        REPORT_TXT
    )

    print("=" * 80)


if __name__ == "__main__":
    main()