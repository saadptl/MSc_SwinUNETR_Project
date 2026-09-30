"""
PART 98 — STRONG FULL-COHORT SWIN-UNETR SEGMENTATION TRAINING

Goal
----
Train a stronger Swin-UNETR segmentation model using the largest
reproducible RSNA segmentation cohort available in the project.

Important:
- RSNA point/localizer annotations are converted into pseudo-masks.
- These are NOT dense expert segmentation masks.
- This script does NOT claim clinical validation.
- No external dataset is used.
- Study-level separation is preserved.
- The final model is selected using validation foreground Dice.

Strategy
--------
1. Use the RSNA training/validation cohort already established by Part 15
   when available.
2. If the cohort is larger than the historical 100/50 subset, use it.
3. Preserve the existing Part11 preprocessing and model factory.
4. Use centered foreground-aware crops.
5. Use class-balanced Dice + weighted cross entropy.
6. Use stronger spatial/intensity augmentation during training.
7. Use gradient accumulation to fit RTX 2050 4GB.
8. Save every epoch and the best validation checkpoint.
9. Independently reload the best checkpoint and verify it.
10. Compare against Part84 final segmentation performance.

This is intended as the main segmentation improvement attempt before
moving to the dashboard.
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# PROJECT PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part98_strong_full_cohort_training"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

EPOCHS = 12

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5

NUM_CLASSES = 6

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)

FEATURE_SIZE = 12

NUM_WORKERS = 0

USE_AMP = True

# Stronger augmentation
AUGMENT_PROB = 0.75

# Early stopping is intentionally conservative.
EARLY_STOPPING_PATIENCE = 5

# Minimum foreground needed for centered crop selection.
MIN_FOREGROUND_VOXELS = 20

# Number of representative validation cases used for independent verification.
VERIFY_CASES = 50


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# IMPORT EXISTING PROJECT PIPELINE
# ============================================================================

print("=" * 90)
print("PART 98 — STRONG FULL-COHORT SWIN-UNETR SEGMENTATION TRAINING")
print("=" * 90)

print(f"Project root : {ROOT}")
print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
else:
    print("GPU          : CPU")


print("\nIMPORTING EXISTING PROJECT MODULES")

try:
    import segmentation_rsna_part11_controlled_pilot_training as part11
    import segmentation_rsna_part9_3d_dataset_loader as part9
except Exception as exc:
    print("\nERROR importing segmentation modules:")
    print(exc)
    raise


# ============================================================================
# CHECKPOINTS
# ============================================================================

PART15_INIT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "checkpoints"
    / "part15_initialization_from_part11.pth"
)

PART84_BEST = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part84_final_reproducible_training"
    / "checkpoints"
    / "part84_best_model.pth"
)

PART15_TRAIN = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_train_cohort.csv"
)

PART15_VAL = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_validation_cohort.csv"
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint_state(path: Path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"], checkpoint

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"], checkpoint

    return checkpoint, {}


def normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    """
    Convert image to [D,H,W].
    """

    image = image.float()

    while image.ndim > 3:
        if image.shape[0] == 1:
            image = image[0]
        elif image.shape[-1] == 1:
            image = image[..., 0]
        else:
            break

    if image.ndim != 3:
        raise RuntimeError(
            f"Unexpected image shape after normalization: {tuple(image.shape)}"
        )

    return image


def normalize_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert mask to [D,H,W] integer labels.
    """

    mask = mask.long()

    while mask.ndim > 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            break

    if mask.ndim != 3:
        raise RuntimeError(
            f"Unexpected mask shape after normalization: {tuple(mask.shape)}"
        )

    return mask


# ============================================================================
# CENTERED FOREGROUND CROP
# ============================================================================

def compute_foreground_center(mask: torch.Tensor) -> Tuple[int, int, int]:
    """
    Calculate center of non-background voxels.
    """

    coords = torch.nonzero(mask > 0, as_tuple=False)

    if coords.numel() == 0:
        d, h, w = mask.shape

        return d // 2, h // 2, w // 2

    center = coords.float().mean(dim=0)

    return (
        int(round(float(center[0]))),
        int(round(float(center[1]))),
        int(round(float(center[2]))),
    )


def crop_3d(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_shape: Tuple[int, int, int],
    center: Tuple[int, int, int],
):
    cd, ch, cw = crop_shape

    d, h, w = image.shape

    cz, cy, cx = center

    z0 = max(0, min(cz - cd // 2, d - cd))
    y0 = max(0, min(cy - ch // 2, h - ch))
    x0 = max(0, min(cx - cw // 2, w - cw))

    z1 = min(d, z0 + cd)
    y1 = min(h, y0 + ch)
    x1 = min(w, x0 + cw)

    image_crop = image[z0:z1, y0:y1, x0:x1]
    mask_crop = mask[z0:z1, y0:y1, x0:x1]

    # Pad if needed.
    pd = cd - image_crop.shape[0]
    ph = ch - image_crop.shape[1]
    pw = cw - image_crop.shape[2]

    if pd > 0 or ph > 0 or pw > 0:
        image_crop = F.pad(
            image_crop,
            (
                0,
                max(0, pw),
                0,
                max(0, ph),
                0,
                max(0, pd),
            ),
            mode="constant",
            value=0,
        )

        mask_crop = F.pad(
            mask_crop,
            (
                0,
                max(0, pw),
                0,
                max(0, ph),
                0,
                max(0, pd),
            ),
            mode="constant",
            value=0,
        )

    return image_crop, mask_crop


# ============================================================================
# FULL VOLUME RESIZE
# ============================================================================

def resize_3d(
    image: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:

    image = image.float()

    x = image.unsqueeze(0).unsqueeze(0)

    x = F.interpolate(
        x,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )

    return x[0, 0]


def resize_mask_3d(
    mask: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:

    x = mask.float().unsqueeze(0).unsqueeze(0)

    x = F.interpolate(
        x,
        size=target_shape,
        mode="nearest",
    )

    return x[0, 0].long()


# ============================================================================
# IMAGE NORMALIZATION
# ============================================================================

def normalize_mri(image: torch.Tensor) -> torch.Tensor:

    image = image.float()

    finite = torch.isfinite(image)

    if not finite.any():
        return torch.zeros_like(image)

    valid = image[finite]

    low = torch.quantile(valid, 0.01)
    high = torch.quantile(valid, 0.99)

    image = torch.clamp(image, low, high)

    min_value = image.min()
    max_value = image.max()

    if float(max_value - min_value) > 1e-8:
        image = (image - min_value) / (max_value - min_value)
    else:
        image = torch.zeros_like(image)

    return image


# ============================================================================
# STRONG AUGMENTATION
# ============================================================================

def augment_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if random.random() > AUGMENT_PROB:
        return image, mask

    # Flip along width.
    if random.random() < 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])

    # Small in-plane vertical flip is intentionally disabled because
    # superior/inferior anatomical orientation must remain meaningful.

    # Intensity scale.
    if random.random() < 0.60:
        scale = random.uniform(0.85, 1.15)
        image = image * scale

    # Intensity shift.
    if random.random() < 0.50:
        shift = random.uniform(-0.08, 0.08)
        image = image + shift

    # Gaussian noise.
    if random.random() < 0.35:
        noise = torch.randn_like(image) * random.uniform(0.005, 0.025)
        image = image + noise

    # Gamma-style contrast.
    if random.random() < 0.30:
        gamma = random.uniform(0.8, 1.2)
        image = torch.clamp(image, 0.0, 1.0)
        image = image.pow(gamma)

    image = torch.clamp(image, 0.0, 1.0)

    return image, mask


# ============================================================================
# DATASET
# ============================================================================

class StrongSegmentationDataset(Dataset):

    def __init__(
        self,
        cohort_path: Path,
        training: bool,
    ):
        self.cohort_path = Path(cohort_path)
        self.training = training

        self.df = pd.read_csv(self.cohort_path)

        if len(self.df) == 0:
            raise RuntimeError(
                f"Cohort is empty: {self.cohort_path}"
            )

    def __len__(self):
        return len(self.df)

    def _load_case(self, row):

        loaded = part11.load_tensor_case(
            row,
            part9,
        )

        if not isinstance(loaded, (tuple, list)):
            raise RuntimeError(
                "Part11 load_tensor_case() did not return a tuple/list."
            )

        if len(loaded) < 2:
            raise RuntimeError(
                "Part11 load_tensor_case() returned fewer than 2 values."
            )

        image = loaded[0]
        mask = loaded[1]

        image = normalize_image_tensor(image)
        mask = normalize_mask_tensor(mask)

        return image, mask

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image, mask = self._load_case(row)

        image = normalize_mri(image)

        image = resize_3d(
            image,
            FULL_SHAPE,
        )

        mask = resize_mask_3d(
            mask,
            FULL_SHAPE,
        )

        center = compute_foreground_center(mask)

        image, mask = crop_3d(
            image,
            mask,
            CROP_SHAPE,
            center,
        )

        if self.training:
            image, mask = augment_pair(
                image,
                mask,
            )

        image = image.unsqueeze(0)

        mask = mask.unsqueeze(0)

        return image.float(), mask.long()


# ============================================================================
# MODEL
# ============================================================================

def create_model():

    model = part11.create_model(
        DEVICE
    )

    return model


# ============================================================================
# CLASS-BALANCED LOSS
# ============================================================================

CLASS_WEIGHTS = torch.tensor(
    [
        0.015069,
        0.722595,
        0.935995,
        1.017003,
        1.178304,
        1.146102,
    ],
    dtype=torch.float32,
)


class CombinedSegmentationLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.ce_weights = CLASS_WEIGHTS / CLASS_WEIGHTS.mean()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ):

        target = target.squeeze(1).long()

        ce = F.cross_entropy(
            logits,
            target,
            weight=self.ce_weights.to(logits.device),
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        one_hot = F.one_hot(
            target,
            num_classes=NUM_CLASSES,
        )

        one_hot = one_hot.permute(
            0,
            4,
            1,
            2,
            3,
        ).float()

        intersection = (
            probabilities * one_hot
        ).sum(dim=(0, 2, 3, 4))

        denominator = (
            probabilities.sum(dim=(0, 2, 3, 4))
            + one_hot.sum(dim=(0, 2, 3, 4))
            + 1e-6
        )

        dice = (
            2.0 * intersection + 1e-6
        ) / denominator

        # Ignore background when calculating the primary Dice loss.
        foreground_dice = dice[1:]

        dice_loss = 1.0 - foreground_dice.mean()

        total = dice_loss + ce

        return total


# ============================================================================
# METRICS
# ============================================================================

@torch.no_grad()
def calculate_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
):

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    target = target.squeeze(1)

    scores = []

    for class_id in range(1, NUM_CLASSES):

        pred_class = prediction == class_id
        target_class = target == class_id

        intersection = (
            pred_class & target_class
        ).sum().float()

        pred_sum = pred_class.sum().float()
        target_sum = target_class.sum().float()

        denominator = pred_sum + target_sum

        if denominator == 0:
            score = 1.0
        else:
            score = (
                2.0 * intersection / denominator
            )

        scores.append(float(score))

    foreground_dice = float(
        np.mean(scores)
    )

    return foreground_dice, scores


# ============================================================================
# ONE EPOCH
# ============================================================================

def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    training: bool,
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dice = 0.0

    class_scores = []

    optimizer.zero_grad(
        set_to_none=True
    )

    for step, (images, masks) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            enabled=(USE_AMP and DEVICE.type == "cuda")
        ):

            logits = model(
                images
            )

            loss = criterion(
                logits,
                masks,
            )

            loss_for_backward = (
                loss / GRADIENT_ACCUMULATION
                if training
                else loss
            )

        if training:

            scaler.scale(
                loss_for_backward
            ).backward()

            should_step = (
                (step + 1) % GRADIENT_ACCUMULATION == 0
                or (step + 1) == len(loader)
            )

            if should_step:

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

        dice, class_dice = calculate_dice(
            logits,
            masks,
        )

        total_loss += float(loss.item())
        total_dice += dice

        class_scores.append(
            class_dice
        )

    n = max(1, len(loader))

    mean_loss = total_loss / n
    mean_dice = total_dice / n

    class_scores = np.asarray(
        class_scores,
        dtype=np.float32,
    )

    classwise = (
        class_scores.mean(axis=0)
        if len(class_scores)
        else np.zeros(NUM_CLASSES - 1)
    )

    return (
        mean_loss,
        mean_dice,
        classwise.tolist(),
    )


# ============================================================================
# SAVE CHECKPOINT
# ============================================================================

def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    epoch,
    metrics,
):

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "seed": SEED,
        "feature_size": FEATURE_SIZE,
        "num_classes": NUM_CLASSES,
        "full_shape": FULL_SHAPE,
        "crop_shape": CROP_SHAPE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "class_weights": CLASS_WEIGHTS.tolist(),
    }

    torch.save(
        payload,
        path,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("\nCONFIGURATION")
    print("-" * 90)

    print(f"Epochs                 : {EPOCHS}")
    print(f"Batch size             : {BATCH_SIZE}")
    print(f"Gradient accumulation  : {GRADIENT_ACCUMULATION}")
    print(f"Learning rate          : {LEARNING_RATE}")
    print(f"Weight decay           : {WEIGHT_DECAY}")
    print(f"Full volume            : {FULL_SHAPE}")
    print(f"Training crop          : {CROP_SHAPE}")
    print(f"Feature size           : {FEATURE_SIZE}")
    print(f"AMP                    : {USE_AMP}")
    print(f"Seed                   : {SEED}")

    print("\nCHECKING COHORTS")

    if not PART15_TRAIN.exists():
        raise FileNotFoundError(
            f"Training cohort not found:\n{PART15_TRAIN}"
        )

    if not PART15_VAL.exists():
        raise FileNotFoundError(
            f"Validation cohort not found:\n{PART15_VAL}"
        )

    train_df = pd.read_csv(
        PART15_TRAIN
    )

    val_df = pd.read_csv(
        PART15_VAL
    )

    print(
        f"Training cohort rows    : {len(train_df)}"
    )

    print(
        f"Validation cohort rows  : {len(val_df)}"
    )

    overlap = set(
        train_df.astype(str).iloc[:, 0]
    ).intersection(
        set(val_df.astype(str).iloc[:, 0])
    )

    print(
        f"Potential first-column overlap : {len(overlap)}"
    )

    print("\nIMPORTANT")

    if len(train_df) < 500:
        print(
            "WARNING: Existing Part15 segmentation cohort is still small."
        )
        print(
            "Part98 will still perform a stronger training run, "
            "but this does not yet constitute full-dataset segmentation."
        )
    else:
        print(
            "Expanded segmentation cohort detected."
        )

    print("\nCHECKING INITIALIZATION")

    if not PART15_INIT.exists():
        raise FileNotFoundError(
            f"Part15 initialization checkpoint not found:\n{PART15_INIT}"
        )

    print(
        "Part15 initialization checkpoint : PASS"
    )

    print(
        f"SHA256 : {sha256_file(PART15_INIT)}"
    )

    print("\nCREATING DATASETS")

    train_dataset = StrongSegmentationDataset(
        PART15_TRAIN,
        training=True,
    )

    val_dataset = StrongSegmentationDataset(
        PART15_VAL,
        training=False,
    )

    print(
        f"Train samples : {len(train_dataset)}"
    )

    print(
        f"Val samples   : {len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("\nCREATING SWIN-UNETR")

    model = create_model()

    print(
        f"Parameters : {count_parameters(model):,}"
    )

    # ------------------------------------------------------------------------
    # LOAD PART15 INITIALIZATION
    # ------------------------------------------------------------------------

    state_dict, metadata = load_checkpoint_state(
        PART15_INIT
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"Initialization missing keys    : {len(missing)}"
    )

    print(
        f"Initialization unexpected keys : {len(unexpected)}"
    )

    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(
            "Part15 initialization does not match the current Swin-UNETR."
        )

    print(
        "Part15 initialization load : PASS"
    )

    # ------------------------------------------------------------------------
    # LOSS / OPTIMIZER
    # ------------------------------------------------------------------------

    criterion = CombinedSegmentationLoss()

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
        enabled=(USE_AMP and DEVICE.type == "cuda")
    )

    history = []

    best_dice = -math.inf
    best_epoch = -1
    patience_counter = 0

    print("\nSTARTING STRONG TRAINING")
    print("=" * 90)

    for epoch in range(1, EPOCHS + 1):

        start = time.time()

        train_loss, train_dice, train_classwise = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            training=True,
        )

        val_loss, val_dice, val_classwise = run_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            criterion,
            training=False,
        )

        scheduler.step()

        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_fg_dice": train_dice,
            "val_loss": val_loss,
            "val_fg_dice": val_dice,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": elapsed,
        }

        for i, value in enumerate(
            train_classwise,
            start=1,
        ):
            row[
                f"train_class_{i}_dice"
            ] = value

        for i, value in enumerate(
            val_classwise,
            start=1,
        ):
            row[
                f"val_class_{i}_dice"
            ] = value

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"TrainLoss {train_loss:.6f} | "
            f"TrainDice {train_dice:.6f} | "
            f"ValLoss {val_loss:.6f} | "
            f"ValDice {val_dice:.6f} | "
            f"Time {elapsed:.1f}s"
        )

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"part98_epoch_{epoch:02d}.pth"
        )

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scaler,
            epoch,
            row,
        )

        if val_dice > best_dice:

            best_dice = val_dice
            best_epoch = epoch
            patience_counter = 0

            best_path = (
                CHECKPOINT_DIR
                / "part98_best_model.pth"
            )

            shutil.copy2(
                checkpoint_path,
                best_path,
            )

            print(
                f"  -> NEW BEST MODEL: Dice={best_dice:.6f}"
            )

        else:

            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print(
                f"Early stopping triggered after "
                f"{patience_counter} non-improving epochs."
            )

            break

    # ------------------------------------------------------------------------
    # SAVE HISTORY
    # ------------------------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        OUTPUT_DIR
        / "part98_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # VERIFY BEST CHECKPOINT
    # ------------------------------------------------------------------------

    print("\nVERIFYING BEST CHECKPOINT")

    best_path = (
        CHECKPOINT_DIR
        / "part98_best_model.pth"
    )

    if not best_path.exists():
        raise RuntimeError(
            "Best checkpoint was not created."
        )

    verify_model = create_model()

    best_state, best_metadata = load_checkpoint_state(
        best_path
    )

    missing, unexpected = verify_model.load_state_dict(
        best_state,
        strict=True,
    )

    print(
        f"Strict load missing    : {len(missing)}"
    )

    print(
        f"Strict load unexpected : {len(unexpected)}"
    )

    verify_model.eval()

    verification_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    verify_losses = []
    verify_dices = []
    verify_classwise = []

    with torch.no_grad():

        for index, (images, masks) in enumerate(
            verification_loader
        ):

            if index >= VERIFY_CASES:
                break

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            with autocast(
                enabled=(
                    USE_AMP
                    and DEVICE.type == "cuda"
                )
            ):

                logits = verify_model(
                    images
                )

                loss = criterion(
                    logits,
                    masks,
                )

            dice, classwise = calculate_dice(
                logits,
                masks,
            )

            verify_losses.append(
                float(loss.item())
            )

            verify_dices.append(
                dice
            )

            verify_classwise.append(
                classwise
            )

    verification_dice = float(
        np.mean(verify_dices)
    )

    verification_loss = float(
        np.mean(verify_losses)
    )

    verification_classwise = (
        np.asarray(
            verify_classwise
        ).mean(axis=0)
    )

    # ------------------------------------------------------------------------
    # SAVE VERIFICATION
    # ------------------------------------------------------------------------

    verification = {
        "verification_cases": len(verify_dices),
        "verification_loss": verification_loss,
        "verification_foreground_dice": verification_dice,
        "classwise_dice": {
            str(i + 1): float(
                verification_classwise[i]
            )
            for i in range(NUM_CLASSES - 1)
        },
        "best_epoch": best_epoch,
        "training_best_dice": best_dice,
        "part15_initialization_sha256": sha256_file(
            PART15_INIT
        ),
        "best_checkpoint": str(
            best_path
        ),
        "best_checkpoint_sha256": sha256_file(
            best_path
        ),
    }

    verification_path = (
        OUTPUT_DIR
        / "part98_best_checkpoint_verification.json"
    )

    with open(
        verification_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            verification,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------------------------------

    summary = {
        "part": 98,
        "title": (
            "Strong Full-Cohort Swin-UNETR "
            "Segmentation Training"
        ),
        "device": str(DEVICE),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "pytorch": torch.__version__,
        "seed": SEED,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "feature_size": FEATURE_SIZE,
        "num_classes": NUM_CLASSES,
        "full_shape": FULL_SHAPE,
        "crop_shape": CROP_SHAPE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
        "verification_dice": verification_dice,
        "verification_loss": verification_loss,
        "verification_cases": len(verify_dices),
        "classwise_verification_dice": {
            str(i + 1): float(
                verification_classwise[i]
            )
            for i in range(NUM_CLASSES - 1)
        },
        "part15_initialization_sha256": sha256_file(
            PART15_INIT
        ),
        "best_checkpoint": str(
            best_path
        ),
        "best_checkpoint_sha256": sha256_file(
            best_path
        ),
        "limitations": [
            "Segmentation masks are derived from RSNA localizer/point annotations.",
            "The dataset does not provide dense expert voxel-wise segmentation masks.",
            "This experiment does not establish clinical validity.",
            "External prospective validation is required before clinical deployment.",
        ],
    }

    summary_path = (
        REPORT_DIR
        / "part98_strong_segmentation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    report_path = (
        REPORT_DIR
        / "part98_strong_segmentation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PART 98 — STRONG FULL-COHORT "
            "SWIN-UNETR SEGMENTATION TRAINING\n"
        )

        f.write("=" * 90 + "\n\n")

        f.write(
            f"Training samples       : {len(train_dataset)}\n"
        )

        f.write(
            f"Validation samples     : {len(val_dataset)}\n"
        )

        f.write(
            f"Epochs completed       : {len(history)}\n"
        )

        f.write(
            f"Best epoch             : {best_epoch}\n"
        )

        f.write(
            f"Best validation Dice   : {best_dice:.6f}\n"
        )

        f.write(
            f"Verification Dice      : {verification_dice:.6f}\n"
        )

        f.write(
            f"Verification loss      : {verification_loss:.6f}\n"
        )

        f.write(
            "\nClasswise verification Dice:\n"
        )

        for i, value in enumerate(
            verification_classwise,
            start=1,
        ):

            f.write(
                f"Class {i}: {value:.6f}\n"
            )

        f.write(
            "\nInterpretation:\n"
        )

        f.write(
            "Part98 is intended to strengthen the segmentation model "
            "before dashboard integration.\n"
        )

        f.write(
            "The RSNA segmentation supervision is based on localizer/"
            "point-derived pseudo-masks rather than dense expert "
            "voxel-wise segmentation masks.\n"
        )

        f.write(
            "Therefore Dice performance should be interpreted as "
            "pseudo-mask agreement rather than clinical segmentation "
            "accuracy.\n"
        )

        f.write(
            "\nClinical limitation:\n"
        )

        f.write(
            "This experiment does not establish clinical safety, "
            "clinical efficacy, regulatory compliance, or readiness "
            "for autonomous diagnosis.\n"
        )

    # ------------------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("PART 98 FINAL RESULT")
    print("=" * 90)

    print(
        f"Training samples       : {len(train_dataset)}"
    )

    print(
        f"Validation samples     : {len(val_dataset)}"
    )

    print(
        f"Epochs completed       : {len(history)}"
    )

    print(
        f"Best epoch             : {best_epoch}"
    )

    print(
        f"Best validation Dice   : {best_dice:.6f}"
    )

    print(
        f"Verification Dice      : {verification_dice:.6f}"
    )

    print(
        f"Verification cases     : {len(verify_dices)}"
    )

    print(
        "\nBEST CHECKPOINT:"
    )

    print(
        best_path
    )

    print(
        "\nREPORT:"
    )

    print(
        report_path
    )

    print(
        "\nSUMMARY:"
    )

    print(
        summary_path
    )

    print(
        "\nSTATUS:"
    )

    print(
        "PASS — STRONG SEGMENTATION TRAINING COMPLETED"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()