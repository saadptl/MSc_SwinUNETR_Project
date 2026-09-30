"""
PART 125 — TARGETED SEGMENTATION RETRAINING
============================================

Purpose
-------
Controlled recovery experiment for the Part104 Swin-UNETR segmentation model.

Design
------
1. Part104 is loaded ONLY as initialization.
2. Part104 is SHA-256 protected and is never overwritten.
3. Uses the established Part9 -> corrected Part11 loading path.
4. Loads the FULL preprocessed volume before cropping.
5. Training uses a genuine 50/50 mixture of:
      - foreground-centred crops
      - random/background-aware crops
6. Training crop: (32, 64, 64)
7. Full preprocessing volume: (64, 96, 96)
8. Loss: unweighted cross-entropy + foreground soft Dice.
9. Validation is performed on the FULL volume using MONAI sliding-window inference.
10. Canonical Swin-UNETR is preserved:
       in_channels=1, out_channels=6, feature_size=12, spatial_dims=3
11. Outputs are isolated under:
       outputs/segmentation/rsna_part125_targeted_retraining

IMPORTANT
---------
Targets are the established development/pseudo-mask targets. This is not
clinical validation or a production/clinical performance claim.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

try:
    from monai.inferers import sliding_window_inference
except Exception as exc:
    raise RuntimeError(
        "MONAI is required for Part125 sliding-window validation."
    ) from exc


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

PART98 = SRC_DIR / "segmentation_rsna_part98_strong_full_cohort_training.py"
PART9 = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"
PART11 = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"

TRAIN_COHORT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part98_strong_full_cohort_training"
    / "part98_train_cohort.csv"
)

VAL_COHORT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part98_strong_full_cohort_training"
    / "part98_validation_cohort.csv"
)

PART104 = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part104_final_checkpoint_selection"
    / "checkpoints"
    / "final_segmentation_model.pth"
)

OUT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part125_targeted_retraining"
)

CHECKPOINT_DIR = OUT / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "reports"

BEST_CKPT = CHECKPOINT_DIR / "best_part125_model.pth"
FINAL_CKPT = CHECKPOINT_DIR / "final_part125_model.pth"
HISTORY_CSV = OUT / "part125_training_history.csv"
METRICS_JSON = OUT / "part125_final_metrics.json"
REPORT_TXT = REPORT_DIR / "part125_targeted_retraining_report.txt"
REPORT_JSON = REPORT_DIR / "part125_targeted_retraining_summary.json"


# ============================================================================
# LOCKED CONFIGURATION
# ============================================================================

EXPECTED_PART104_SHA = (
    "fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"
)

SEED = 42

NUM_CLASSES = 6
IN_CHANNELS = 1
FEATURE_SIZE = 12

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)

EPOCHS = 12
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5

WORKERS = 0
AMP = True

FOREGROUND_CROP_PROBABILITY = 0.50
RANDOM_CROP_PROBABILITY = 0.50

AUGMENT_PROBABILITY = 0.75

MIN_FOREGROUND_VOXELS = 20

# Validation / deployment-compatible inference.
SW_ROI = CROP_SHAPE
SW_OVERLAP = 0.50
SW_BATCH_SIZE = 1
SW_MODE = "gaussian"
SW_SIGMA_SCALE = 0.125

NUM_CLASSES_FOREGROUND = tuple(range(1, NUM_CLASSES))


# ============================================================================
# PRINTING / HELPERS
# ============================================================================

def banner(text: str) -> None:
    print()
    print("=" * 96)
    print(text)
    print("=" * 96)


def section(text: str) -> None:
    print()
    print("-" * 96)
    print(text)
    print("-" * 96)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(type(obj).__name__)


def set_seed() -> None:
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


def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================================
# PREPROCESSING
# ============================================================================

def ensure_3d(x: Any, name: str) -> torch.Tensor:
    """
    Convert a single-channel volume/mask to [D,H,W].

    Accepts:
        [D,H,W]
        [1,D,H,W]
        [1,1,D,H,W]
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    x = x.detach().cpu()

    while x.ndim > 3:
        if x.shape[0] == 1:
            x = x[0]
        elif x.shape[-1] == 1:
            x = x[..., 0]
        else:
            raise RuntimeError(
                f"{name} has unsupported shape {tuple(x.shape)}; "
                "cannot safely reduce to [D,H,W]."
            )

    if x.ndim != 3:
        raise RuntimeError(
            f"{name} must be 3D after normalization; got {tuple(x.shape)}"
        )

    return x


def resize_3d(
    x: torch.Tensor,
    target_shape: Tuple[int, int, int],
    is_mask: bool,
) -> torch.Tensor:
    """
    Resize [D,H,W] to target_shape.

    Image: trilinear.
    Mask: nearest-neighbour.
    """
    x = ensure_3d(x, "resize input").float()

    if tuple(x.shape) == tuple(target_shape):
        if is_mask:
            return torch.round(x).long()
        return x.float()

    y = x.unsqueeze(0).unsqueeze(0)

    if is_mask:
        y = F.interpolate(
            y,
            size=target_shape,
            mode="nearest",
        )
        return torch.round(y[0, 0]).long()

    y = F.interpolate(
        y,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )
    return y[0, 0].float()


def normalize_mri(image: torch.Tensor) -> torch.Tensor:
    """
    Robust percentile normalization matching the established project intent:
    percentile 1-99 followed by [0,1] clipping.
    """
    x = ensure_3d(image, "MRI image").float()

    finite = torch.isfinite(x)
    if not bool(finite.any()):
        return torch.zeros_like(x)

    vals = x[finite]

    p1 = torch.quantile(vals, 0.01)
    p99 = torch.quantile(vals, 0.99)

    if float(p99 - p1) <= 1e-8:
        lo = vals.min()
        hi = vals.max()
        if float(hi - lo) <= 1e-8:
            return torch.zeros_like(x)
        p1, p99 = lo, hi

    x = (x - p1) / (p99 - p1)
    x = torch.clamp(x, 0.0, 1.0)
    x[~finite] = 0.0

    return x.float()


def compute_foreground_center(mask: torch.Tensor) -> Tuple[int, int, int]:
    m = ensure_3d(mask, "foreground mask")
    coords = torch.nonzero(m > 0, as_tuple=False)

    if coords.numel() == 0:
        return tuple(int(s // 2) for s in m.shape)

    center = coords.float().mean(dim=0).round().long()
    return tuple(int(v) for v in center.tolist())


def crop_3d(
    image: torch.Tensor,
    mask: torch.Tensor,
    start: Tuple[int, int, int],
    crop_shape: Tuple[int, int, int] = CROP_SHAPE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract an exact crop. Pads if needed.
    """
    image = ensure_3d(image, "crop image").float()
    mask = ensure_3d(mask, "crop mask").long()

    d, h, w = image.shape
    cd, ch, cw = crop_shape
    z0, y0, x0 = start

    z0 = max(0, min(z0, max(0, d - cd)))
    y0 = max(0, min(y0, max(0, h - ch)))
    x0 = max(0, min(x0, max(0, w - cw)))

    z1 = min(z0 + cd, d)
    y1 = min(y0 + ch, h)
    x1 = min(x0 + cw, w)

    image_crop = image[z0:z1, y0:y1, x0:x1]
    mask_crop = mask[z0:z1, y0:y1, x0:x1]

    pd = cd - image_crop.shape[0]
    ph = ch - image_crop.shape[1]
    pw = cw - image_crop.shape[2]

    if pd or ph or pw:
        image_crop = F.pad(
            image_crop,
            (0, pw, 0, ph, 0, pd),
            mode="constant",
            value=0.0,
        )

        mask_crop = F.pad(
            mask_crop,
            (0, pw, 0, ph, 0, pd),
            mode="constant",
            value=0,
        )

    return image_crop.float(), mask_crop.long()


def foreground_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = compute_foreground_center(mask)

    cd, ch, cw = CROP_SHAPE
    cz, cy, cx = center

    start = (
        cz - cd // 2,
        cy - ch // 2,
        cx - cw // 2,
    )

    return crop_3d(image, mask, start, CROP_SHAPE)


def random_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = ensure_3d(image, "random image")
    mask = ensure_3d(mask, "random mask")

    d, h, w = image.shape
    cd, ch, cw = CROP_SHAPE

    z_max = max(0, d - cd)
    y_max = max(0, h - ch)
    x_max = max(0, w - cw)

    start = (
        random.randint(0, z_max) if z_max else 0,
        random.randint(0, y_max) if y_max else 0,
        random.randint(0, x_max) if x_max else 0,
    )

    return crop_3d(image, mask, start, CROP_SHAPE)


# ============================================================================
# AUGMENTATION
# ============================================================================

def augment_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lightweight spatial/intensity augmentation compatible with the established
    Part98 intent. Masks use nearest-neighbour semantics throughout.
    """
    image = image.float()
    mask = mask.long()

    if random.random() < 0.5:
        image = torch.flip(image, dims=[0])
        mask = torch.flip(mask, dims=[0])

    if random.random() < 0.5:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])

    if random.random() < 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])

    # Small intensity scale/shift.
    if random.random() < 0.5:
        scale = random.uniform(0.90, 1.10)
        shift = random.uniform(-0.05, 0.05)
        image = torch.clamp(image * scale + shift, 0.0, 1.0)

    # Small Gaussian noise.
    if random.random() < 0.35:
        noise_std = random.uniform(0.005, 0.025)
        image = torch.clamp(
            image + torch.randn_like(image) * noise_std,
            0.0,
            1.0,
        )

    # Mild gamma adjustment.
    if random.random() < 0.35:
        gamma = random.uniform(0.85, 1.15)
        image = torch.clamp(image, 0.0, 1.0) ** gamma

    return image.float(), mask.long()


# ============================================================================
# DATASET
# ============================================================================

class Part125FullVolumeDataset(Dataset):
    """
    Loads each case directly from the established Part11 loader, preprocesses
    to FULL_SHAPE, then creates a true training crop.

    Crucially, this does NOT call Part98 __getitem__, because Part98 already
    returns its crop. That would make a second crop operate on an already
    cropped tensor and would not produce genuine random/full-volume sampling.
    """

    def __init__(
        self,
        cohort_path: Path,
        part9_module,
        part11_module,
        training: bool,
    ):
        self.rows = pd.read_csv(cohort_path).reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Empty cohort: {cohort_path}")

        self.part9 = part9_module
        self.part11 = part11_module
        self.training = bool(training)

    def __len__(self):
        return len(self.rows)

    def _load_full_case(
        self,
        row: pd.Series,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loaded = self.part11.load_tensor_case(row, self.part9)

        if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
            raise RuntimeError("Unexpected Part11 load_tensor_case return value.")

        image = ensure_3d(loaded[0], "Part11 image").float()
        mask = ensure_3d(loaded[1], "Part11 mask").long()

        image = normalize_mri(image)
        image = resize_3d(image, FULL_SHAPE, is_mask=False)
        mask = resize_3d(mask, FULL_SHAPE, is_mask=True)

        mask = torch.clamp(mask, 0, NUM_CLASSES - 1).long()

        if tuple(image.shape) != FULL_SHAPE:
            raise RuntimeError(
                f"Image preprocessing produced {tuple(image.shape)}, "
                f"expected {FULL_SHAPE}"
            )

        if tuple(mask.shape) != FULL_SHAPE:
            raise RuntimeError(
                f"Mask preprocessing produced {tuple(mask.shape)}, "
                f"expected {FULL_SHAPE}"
            )

        return image.float(), mask.long()

    def __getitem__(self, index):
        row = self.rows.iloc[index]

        image, mask = self._load_full_case(row)

        if self.training:
            if random.random() < FOREGROUND_CROP_PROBABILITY:
                image, mask = foreground_crop(image, mask)
                crop_mode = "foreground"
            else:
                image, mask = random_crop(image, mask)
                crop_mode = "random"

            if random.random() < AUGMENT_PROBABILITY:
                image, mask = augment_pair(image, mask)
                augmented = True
            else:
                augmented = False
        else:
            # Validation dataset is intentionally FULL volume.
            crop_mode = "full"
            augmented = False

        image = image.unsqueeze(0).float()       # [1,D,H,W]
        mask = mask.long()                       # [D,H,W]

        metadata = {
            "index": int(index),
            "study_id": str(row.get("study_id", "")),
            "series_id": str(row.get("series_id", "")),
            "series_description": str(row.get("series_description", "")),
            "crop_mode": crop_mode,
            "augmented": augmented,
            "foreground_voxels": int((mask > 0).sum().item()),
        }

        return image, mask, metadata


# ============================================================================
# MODEL
# ============================================================================

def normalize_state_keys(state: Dict[str, torch.Tensor]):
    if not state:
        return state

    keys = list(state.keys())

    if all(k.startswith("module.") for k in keys):
        return {
            k[len("module."):]: v
            for k, v in state.items()
        }

    return state


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return normalize_state_keys(value)

        if checkpoint and all(
            isinstance(v, torch.Tensor)
            for v in checkpoint.values()
        ):
            return normalize_state_keys(checkpoint)

    raise RuntimeError("Could not locate model state_dict in Part104 checkpoint.")


def build_model(device: torch.device) -> nn.Module:
    # Use the validated corrected Part11 implementation so the canonical
    # SwinUNETR definition remains identical to the established project.
    model = part11.create_model(device)

    if not isinstance(model, nn.Module):
        raise RuntimeError("Part11 create_model did not return torch.nn.Module.")

    return model


# ============================================================================
# LOSS
# ============================================================================

def foreground_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Foreground-only soft Dice loss over classes 1..5.
    """
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    target = target.long()

    probs = torch.softmax(logits, dim=1)

    target_one_hot = F.one_hot(
        target,
        num_classes=NUM_CLASSES,
    ).permute(0, 4, 1, 2, 3).float()

    dices = []

    for c in NUM_CLASSES_FOREGROUND:
        p = probs[:, c]
        t = target_one_hot[:, c]

        intersection = (p * t).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))

        dice = (2.0 * intersection + smooth) / (denom + smooth)
        dices.append(dice)

    dice_mean = torch.stack(dices, dim=0).mean()
    return 1.0 - dice_mean


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    ce = F.cross_entropy(
        logits,
        target.long(),
        reduction="mean",
    )

    dice = foreground_soft_dice_loss(logits, target)

    total = ce + dice

    return total, float(ce.detach().item()), float(dice.detach().item())


# ============================================================================
# METRICS
# ============================================================================

def class_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> List[float]:
    prediction = prediction.long()
    target = target.long()

    scores = []

    for c in NUM_CLASSES_FOREGROUND:
        p = prediction == c
        t = target == c

        inter = int((p & t).sum().item())
        p_n = int(p.sum().item())
        t_n = int(t.sum().item())

        denom = p_n + t_n

        if denom == 0:
            score = 1.0
        else:
            score = (2.0 * inter) / denom

        scores.append(float(score))

    return scores


def metric_summary(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Any]:
    dices = class_dice(prediction, target)

    pred_fg = int((prediction > 0).sum().item())
    target_fg = int((target > 0).sum().item())

    total = int(prediction.numel())

    return {
        "mean_foreground_dice": float(np.mean(dices)),
        "class_dice": dices,
        "pred_foreground_voxels": pred_fg,
        "target_foreground_voxels": target_fg,
        "pred_foreground_fraction": pred_fg / total if total else 0.0,
        "target_foreground_fraction": target_fg / total if total else 0.0,
        "foreground_volume_ratio": (
            pred_fg / target_fg if target_fg else float("inf")
        ),
        "prediction_class_counts": [
            int((prediction == c).sum().item())
            for c in range(NUM_CLASSES)
        ],
        "target_class_counts": [
            int((target == c).sum().item())
            for c in range(NUM_CLASSES)
        ],
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
) -> Dict[str, Any]:

    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0

    dices = []
    pred_fg_ratios = []

    optimizer.zero_grad(set_to_none=True)

    amp_enabled = AMP and device.type == "cuda"

    start_time = time.time()

    for step, batch in enumerate(loader):
        images, masks, metadata = batch

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            enabled=amp_enabled,
        ):
            logits = model(images)

            loss, ce_value, dice_loss_value = combined_loss(
                logits,
                masks,
            )

            scaled_loss = loss / GRADIENT_ACCUMULATION

        if scaler is not None and amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = (
            (step + 1) % GRADIENT_ACCUMULATION == 0
            or (step + 1) == len(loader)
        )

        if should_step:
            if scaler is not None and amp_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            mm = metric_summary(pred, masks)

        total_loss += float(loss.detach().item())
        total_ce += ce_value
        total_dice_loss += dice_loss_value
        dices.append(mm["mean_foreground_dice"])
        pred_fg_ratios.append(mm["foreground_volume_ratio"])

        if (step + 1) % 25 == 0 or (step + 1) == len(loader):
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch:02d} | "
                f"step {step + 1:03d}/{len(loader):03d} | "
                f"loss {float(loss.detach().item()):.5f} | "
                f"Dice {mm['mean_foreground_dice']:.5f} | "
                f"FG ratio {mm['foreground_volume_ratio']:.3f} | "
                f"{elapsed/60.0:.1f} min"
            )

    n = max(1, len(loader))

    return {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "dice_loss": total_dice_loss / n,
        "mean_foreground_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_foreground_volume_ratio": float(np.mean(pred_fg_ratios))
        if pred_fg_ratios
        else 0.0,
    }


# ============================================================================
# FULL-VOLUME VALIDATION
# ============================================================================

@torch.no_grad()
def validate_full_volume(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
) -> Dict[str, Any]:

    model.eval()

    case_dices = []
    class_dice_values = []
    fg_ratios = []

    aggregate_pred = np.zeros(NUM_CLASSES, dtype=np.int64)
    aggregate_target = np.zeros(NUM_CLASSES, dtype=np.int64)

    failures = []

    for index in range(len(dataset)):
        try:
            image, target, metadata = dataset[index]

            # Dataset validation output is [1,D,H,W].
            if image.ndim != 4:
                raise RuntimeError(
                    f"Validation image expected [1,D,H,W], got {tuple(image.shape)}"
                )

            image_batched = image.unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            target = target.long()

            logits = sliding_window_inference(
                inputs=image_batched,
                roi_size=SW_ROI,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=model,
                overlap=SW_OVERLAP,
                mode=SW_MODE,
                sigma_scale=SW_SIGMA_SCALE,
            )

            prediction = torch.argmax(logits, dim=1)[0].cpu()

            mm = metric_summary(
                prediction,
                target,
            )

            case_dices.append(mm["mean_foreground_dice"])
            class_dice_values.append(mm["class_dice"])
            fg_ratios.append(mm["foreground_volume_ratio"])

            aggregate_pred += np.asarray(
                mm["prediction_class_counts"],
                dtype=np.int64,
            )

            aggregate_target += np.asarray(
                mm["target_class_counts"],
                dtype=np.int64,
            )

            if (index + 1) % 10 == 0 or index == 0:
                print(
                    f"  Validation {index + 1:03d}/{len(dataset):03d} | "
                    f"Dice {mm['mean_foreground_dice']:.5f} | "
                    f"FG ratio {mm['foreground_volume_ratio']:.3f}"
                )

        except Exception as exc:
            failures.append({
                "index": int(index),
                "error": repr(exc),
            })
            print(
                f"  Validation case {index + 1} FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

    if not case_dices:
        raise RuntimeError(
            "Full-volume validation produced zero successful cases."
        )

    class_dice_array = np.asarray(class_dice_values, dtype=np.float64)

    return {
        "validation_cases_requested": int(len(dataset)),
        "validation_cases_completed": int(len(case_dices)),
        "validation_failures": failures,
        "mean_foreground_dice": float(np.mean(case_dices)),
        "median_foreground_dice": float(np.median(case_dices)),
        "class_dice": [
            float(v)
            for v in np.mean(class_dice_array, axis=0)
        ],
        "mean_foreground_volume_ratio": float(np.mean(fg_ratios)),
        "median_foreground_volume_ratio": float(np.median(fg_ratios)),
        "aggregate_prediction_class_counts": aggregate_pred.tolist(),
        "aggregate_target_class_counts": aggregate_target.tolist(),
    }


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    history: List[Dict[str, Any]],
    best_dice: float,
) -> None:

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": (
                scaler.state_dict()
                if scaler is not None
                else None
            ),
            "epoch": int(epoch),
            "best_validation_dice": float(best_dice),
            "history": history,
            "part104_sha256": EXPECTED_PART104_SHA,
            "architecture": {
                "in_channels": IN_CHANNELS,
                "out_channels": NUM_CLASSES,
                "feature_size": FEATURE_SIZE,
                "spatial_dims": 3,
            },
            "full_shape": FULL_SHAPE,
            "crop_shape": CROP_SHAPE,
            "sw_roi": SW_ROI,
            "sw_overlap": SW_OVERLAP,
        },
        path,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    global part9, part11

    set_seed()

    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    banner("PART 125 — TARGETED SEGMENTATION RETRAINING")
    print("Purpose: repair Part104 foreground over-segmentation")
    print()
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device       : {device}")

    if device.type == "cuda":
        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------------------------
    # Required artifacts
    # ------------------------------------------------------------------------

    section("CHECKING REQUIRED ARTIFACTS")

    required = {
        "Part98 source": PART98,
        "Part9 source": PART9,
        "Corrected Part11 source": PART11,
        "Part98 train cohort": TRAIN_COHORT,
        "Part98 validation cohort": VAL_COHORT,
        "Part104 checkpoint": PART104,
    }

    missing = []

    for name, path in required.items():
        ok = path.exists()
        print(f"{name:<42} {'PASS' if ok else 'FAIL'}")
        if not ok:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required artifact(s):\n"
            + "\n".join(missing)
        )

    # ------------------------------------------------------------------------
    # Part104 protection
    # ------------------------------------------------------------------------

    section("CHECKING PART104 BASELINE")

    part104_sha = sha256(PART104)

    print(f"Part104 SHA256           : {part104_sha}")
    print(
        f"Part104 SHA verification : "
        f"{'PASS' if part104_sha == EXPECTED_PART104_SHA else 'FAIL'}"
    )

    if part104_sha != EXPECTED_PART104_SHA:
        raise RuntimeError(
            "Part104 SHA mismatch. Aborting before model construction."
        )

    # ------------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------------

    section("IMPORTING ESTABLISHED PROJECT PIPELINE")

    # Part9 is a dependency used by corrected Part11.
    part9 = import_module_from_path(
        "part125_part9",
        PART9,
    )

    part11 = import_module_from_path(
        "part125_part11_corrected",
        PART11,
    )

    print("Part9 import      : PASS")
    print("Part11 import     : PASS")
    print(
        f"Part11 create_model : "
        f"{'PASS' if hasattr(part11, 'create_model') else 'FAIL'}"
    )
    print(
        f"Part11 load_tensor_case : "
        f"{'PASS' if hasattr(part11, 'load_tensor_case') else 'FAIL'}"
    )

    if not hasattr(part11, "create_model"):
        raise RuntimeError("Corrected Part11 create_model is unavailable.")

    if not hasattr(part11, "load_tensor_case"):
        raise RuntimeError(
            "Corrected Part11 load_tensor_case is unavailable."
        )

    # ------------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------------

    section("BUILDING DATASETS")

    train_dataset = Part125FullVolumeDataset(
        TRAIN_COHORT,
        part9,
        part11,
        training=True,
    )

    val_dataset = Part125FullVolumeDataset(
        VAL_COHORT,
        part9,
        part11,
        training=False,
    )

    print(f"Training cases                    : {len(train_dataset)}")
    print(f"Validation cases                  : {len(val_dataset)}")
    print(f"Full preprocessing shape          : {FULL_SHAPE}")
    print(f"Training crop shape               : {CROP_SHAPE}")
    print(f"Foreground crop probability       : {FOREGROUND_CROP_PROBABILITY}")
    print(f"Random crop probability           : {RANDOM_CROP_PROBABILITY}")
    print(f"Validation                        : FULL VOLUME + SLIDING WINDOW")
    print(f"Sliding-window ROI                : {SW_ROI}")
    print(f"Sliding-window overlap            : {SW_OVERLAP}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ------------------------------------------------------------------------
    # Dataset smoke test BEFORE GPU training
    # ------------------------------------------------------------------------

    section("DATASET SMOKE TEST")

    smoke_image, smoke_mask, smoke_meta = train_dataset[0]

    print(f"Train image shape : {tuple(smoke_image.shape)}")
    print(f"Train mask shape  : {tuple(smoke_mask.shape)}")
    print(f"Crop mode         : {smoke_meta['crop_mode']}")
    print(f"Augmented         : {smoke_meta['augmented']}")
    print(
        f"Foreground voxels : "
        f"{smoke_meta['foreground_voxels']}"
    )

    if tuple(smoke_image.shape) != (1,) + CROP_SHAPE:
        raise RuntimeError(
            "Training smoke-test image has wrong shape: "
            f"{tuple(smoke_image.shape)}"
        )

    if tuple(smoke_mask.shape) != CROP_SHAPE:
        raise RuntimeError(
            "Training smoke-test mask has wrong shape: "
            f"{tuple(smoke_mask.shape)}"
        )

    val_image, val_mask, _ = val_dataset[0]

    print(f"Val full image shape : {tuple(val_image.shape)}")
    print(f"Val full mask shape  : {tuple(val_mask.shape)}")

    if tuple(val_image.shape) != (1,) + FULL_SHAPE:
        raise RuntimeError(
            "Validation image is not FULL_SHAPE. "
            f"Got {tuple(val_image.shape)}"
        )

    if tuple(val_mask.shape) != FULL_SHAPE:
        raise RuntimeError(
            "Validation mask is not FULL_SHAPE. "
            f"Got {tuple(val_mask.shape)}"
        )

    print("Dataset smoke test : PASS")

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    section("BUILDING CANONICAL SWIN-UNETR")

    model = build_model(device)

    print(f"Model parameters : {sum(p.numel() for p in model.parameters())}")

    checkpoint = torch.load(
        PART104,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Part104 initialization did not load cleanly.\n"
            f"Missing keys: {missing_keys[:10]}\n"
            f"Unexpected keys: {unexpected_keys[:10]}"
        )

    model.to(device)

    print("Part104 initialization : PASS")
    print("Part104 remains protected and will never be overwritten.")

    # ------------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.10,
    )

    scaler = (
        torch.cuda.amp.GradScaler()
        if AMP and device.type == "cuda"
        else None
    )

    print()
    print("Loss                 : unweighted CE + foreground soft Dice")
    print(f"Learning rate        : {LEARNING_RATE}")
    print(f"Weight decay         : {WEIGHT_DECAY}")
    print(f"Epochs               : {EPOCHS}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"AMP                  : {AMP}")
    print(f"Feature size         : {FEATURE_SIZE}")

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    section("STARTING PART125 TRAINING")

    history: List[Dict[str, Any]] = []

    best_dice = -float("inf")
    best_epoch = 0

    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        banner(f"PART125 EPOCH {epoch:02d}/{EPOCHS}")

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        print(f"Learning rate : {current_lr:.8e}")

        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        print()
        print("TRAINING SUMMARY")
        print(
            f"Loss                    : "
            f"{train_metrics['loss']:.6f}"
        )
        print(
            f"CE loss                 : "
            f"{train_metrics['ce_loss']:.6f}"
        )
        print(
            f"Dice loss               : "
            f"{train_metrics['dice_loss']:.6f}"
        )
        print(
            f"Mean foreground Dice   : "
            f"{train_metrics['mean_foreground_dice']:.6f}"
        )
        print(
            f"Mean FG volume ratio  : "
            f"{train_metrics['mean_foreground_volume_ratio']:.6f}"
        )

        section(
            f"FULL-VOLUME VALIDATION — EPOCH {epoch:02d}"
        )

        val_metrics = validate_full_volume(
            model=model,
            dataset=val_dataset,
            device=device,
        )

        val_dice = float(
            val_metrics["mean_foreground_dice"]
        )

        val_ratio = float(
            val_metrics["mean_foreground_volume_ratio"]
        )

        print()
        print("FULL-VOLUME VALIDATION SUMMARY")
        print(
            f"Completed cases        : "
            f"{val_metrics['validation_cases_completed']}/"
            f"{val_metrics['validation_cases_requested']}"
        )
        print(
            f"Mean foreground Dice  : "
            f"{val_dice:.6f}"
        )
        print(
            f"Median foreground Dice: "
            f"{val_metrics['median_foreground_dice']:.6f}"
        )
        print(
            f"Mean FG volume ratio  : "
            f"{val_ratio:.6f}"
        )
        print(
            f"Median FG volume ratio: "
            f"{val_metrics['median_foreground_volume_ratio']:.6f}"
        )
        print(
            "Class Dice [1..5]     : "
            + ", ".join(
                f"{v:.6f}"
                for v in val_metrics["class_dice"]
            )
        )

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_dice_loss": train_metrics["dice_loss"],
            "train_foreground_dice": train_metrics[
                "mean_foreground_dice"
            ],
            "train_foreground_volume_ratio": train_metrics[
                "mean_foreground_volume_ratio"
            ],
            "val_foreground_dice": val_dice,
            "val_median_foreground_dice": val_metrics[
                "median_foreground_dice"
            ],
            "val_class1_dice": val_metrics["class_dice"][0],
            "val_class2_dice": val_metrics["class_dice"][1],
            "val_class3_dice": val_metrics["class_dice"][2],
            "val_class4_dice": val_metrics["class_dice"][3],
            "val_class5_dice": val_metrics["class_dice"][4],
            "val_foreground_volume_ratio": val_ratio,
            "val_median_foreground_volume_ratio": val_metrics[
                "median_foreground_volume_ratio"
            ],
            "val_cases_completed": val_metrics[
                "validation_cases_completed"
            ],
        }

        history.append(row)

        # Save epoch checkpoint separately.
        epoch_ckpt = CHECKPOINT_DIR / f"part125_epoch_{epoch:02d}.pth"

        save_checkpoint(
            epoch_ckpt,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            history,
            max(best_dice, val_dice),
        )

        # Best checkpoint selection.
        #
        # Primary criterion is full-volume foreground Dice.
        # A strong explosion in predicted foreground volume is recorded but
        # does not silently replace the primary metric.
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch

            save_checkpoint(
                BEST_CKPT,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                history,
                best_dice,
            )

            print()
            print(
                f"*** NEW BEST PART125 CHECKPOINT — "
                f"epoch {epoch}, full-volume Dice {best_dice:.6f} ***"
            )

        scheduler.step()

        print()
        print(
            f"Epoch {epoch:02d} complete. "
            f"Best full-volume Dice: {best_dice:.6f} "
            f"(epoch {best_epoch})"
        )

    # ------------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------------

    section("FINAL PART125 CHECKPOINT")

    save_checkpoint(
        FINAL_CKPT,
        model,
        optimizer,
        scheduler,
        scaler,
        EPOCHS,
        history,
        best_dice,
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        HISTORY_CSV,
        index=False,
    )

    elapsed = time.time() - training_start

    final_summary = {
        "status": "COMPLETED",
        "part": "125",
        "purpose": "targeted segmentation retraining",
        "training_performed": True,
        "epochs_completed": EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_foreground_dice": best_dice,
        "training_time_seconds": elapsed,
        "part104_sha256_expected": EXPECTED_PART104_SHA,
        "part104_sha256_at_start": part104_sha,
        "part104_protected": True,
        "architecture": {
            "in_channels": IN_CHANNELS,
            "out_channels": NUM_CLASSES,
            "feature_size": FEATURE_SIZE,
            "spatial_dims": 3,
        },
        "full_shape": FULL_SHAPE,
        "crop_shape": CROP_SHAPE,
        "foreground_crop_probability": FOREGROUND_CROP_PROBABILITY,
        "random_crop_probability": RANDOM_CROP_PROBABILITY,
        "loss": "unweighted cross-entropy + foreground soft Dice",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "sliding_window": {
            "roi_size": SW_ROI,
            "overlap": SW_OVERLAP,
            "sw_batch_size": SW_BATCH_SIZE,
            "mode": SW_MODE,
            "sigma_scale": SW_SIGMA_SCALE,
        },
        "best_checkpoint": str(BEST_CKPT),
        "final_checkpoint": str(FINAL_CKPT),
        "history_csv": str(HISTORY_CSV),
        "clinical_validation": False,
        "note": (
            "Targets are development/pseudo-mask labels. "
            "Results must not be interpreted as clinical validation."
        ),
    }

    METRICS_JSON.write_text(
        json.dumps(
            final_summary,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    report = f"""
PART 125 — TARGETED SEGMENTATION RETRAINING REPORT

Status:
    COMPLETED

Purpose:
    Repair Part104 foreground over-segmentation using a controlled
    mixed-crop retraining strategy.

Part104 SHA256:
    {part104_sha}

Expected Part104 SHA256:
    {EXPECTED_PART104_SHA}

Part104 protected:
    YES

Architecture:
    Swin-UNETR
    in_channels = {IN_CHANNELS}
    out_channels = {NUM_CLASSES}
    feature_size = {FEATURE_SIZE}
    spatial_dims = 3

Data:
    training cases = {len(train_dataset)}
    validation cases = {len(val_dataset)}
    full shape = {FULL_SHAPE}
    crop shape = {CROP_SHAPE}

Sampling:
    foreground-centred probability = {FOREGROUND_CROP_PROBABILITY}
    random crop probability = {RANDOM_CROP_PROBABILITY}

Loss:
    unweighted cross-entropy + foreground soft Dice

Optimization:
    epochs = {EPOCHS}
    batch size = {BATCH_SIZE}
    gradient accumulation = {GRADIENT_ACCUMULATION}
    learning rate = {LEARNING_RATE}
    weight decay = {WEIGHT_DECAY}

Validation:
    full-volume validation = YES
    sliding-window ROI = {SW_ROI}
    overlap = {SW_OVERLAP}
    mode = {SW_MODE}
    sigma_scale = {SW_SIGMA_SCALE}

Best epoch:
    {best_epoch}

Best full-volume foreground Dice:
    {best_dice:.6f}

Best checkpoint:
    {BEST_CKPT}

Final checkpoint:
    {FINAL_CKPT}

Training time:
    {elapsed / 60.0:.2f} minutes

Important:
    This is development/pseudo-mask evaluation, not clinical validation.
""".strip() + "\n"

    REPORT_TXT.write_text(
        report,
        encoding="utf-8",
    )

    REPORT_JSON.write_text(
        json.dumps(
            final_summary,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    banner("PART125 COMPLETE")

    print(f"Best epoch                 : {best_epoch}")
    print(
        f"Best full-volume Dice     : "
        f"{best_dice:.6f}"
    )
    print(f"Best checkpoint            : {BEST_CKPT}")
    print(f"Final checkpoint           : {FINAL_CKPT}")
    print(f"Training history           : {HISTORY_CSV}")
    print(f"Report                     : {REPORT_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
