# ============================================================================
# PART 126
# FOREGROUND-CALIBRATED SEGMENTATION FINE-TUNING
#
# Purpose:
#   Controlled continuation from the protected Part104 Swin-UNETR checkpoint.
#
# Main changes from Part125:
#   1. 75% foreground-centered / 25% random crop sampling
#   2. Cross-Entropy + false-positive-aware Tversky loss
#   3. Full-volume sliding-window validation
#   4. Validation metric does NOT reward absent classes with Dice=1
#   5. Foreground-volume calibration is explicitly reported
#
# The canonical Swin-UNETR architecture is NOT changed.
# ============================================================================

from __future__ import annotations

import csv
from email.mime import image
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

PART9 = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"
PART11 = (
    SRC_DIR
    / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

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
    / "rsna_part126_foreground_calibrated_finetuning"
)

CHECKPOINT_DIR = OUT / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "reports"

BEST_CKPT = CHECKPOINT_DIR / "best_part126_model.pth"
FINAL_CKPT = CHECKPOINT_DIR / "final_part126_model.pth"

HISTORY_CSV = OUT / "part126_training_history.csv"
METRICS_JSON = OUT / "part126_final_metrics.json"

REPORT_TXT = REPORT_DIR / "part126_foreground_calibrated_report.txt"
REPORT_JSON = REPORT_DIR / "part126_foreground_calibrated_summary.json"


# ============================================================================
# PROTECTED BASELINE
# ============================================================================

EXPECTED_PART104_SHA = (
    "fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"
)


# ============================================================================
# LOCKED ARCHITECTURE / DATA CONFIGURATION
# ============================================================================

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

# ---------------------------------------------------------------------------
# PART126 SAMPLING
# ---------------------------------------------------------------------------

FOREGROUND_CROP_PROBABILITY = 0.75
RANDOM_CROP_PROBABILITY = 0.25

AUGMENT_PROBABILITY = 0.75
MIN_FOREGROUND_VOXELS = 20


# ============================================================================
# SLIDING WINDOW VALIDATION
# ============================================================================

SW_ROI = CROP_SHAPE
SW_OVERLAP = 0.50
SW_BATCH_SIZE = 1
SW_MODE = "gaussian"
SW_SIGMA_SCALE = 0.125


FOREGROUND_CLASSES = tuple(range(1, NUM_CLASSES))


# ============================================================================
# PART126 LOSS CONFIGURATION
# ============================================================================

# Cross-entropy keeps the model grounded in voxel classification.
CE_WEIGHT = 1.0

# Tversky is deliberately FP-aware.
TV_WEIGHT = 1.0

# alpha > beta means false positives are penalized more strongly.
TV_ALPHA = 0.70
TV_BETA = 0.30

TV_SMOOTH = 1e-5


# ============================================================================
# PRINT HELPERS
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


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def seed_everything(seed: int) -> None:
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TENSOR HELPERS
# ============================================================================

def ensure_3d(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """
    Convert common singleton-channel forms to [D,H,W].
    """

    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)

    if tensor.ndim == 5 and tensor.shape[0] == 1 and tensor.shape[1] == 1:
        tensor = tensor[0, 0]

    elif tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]

    elif tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]

    if tensor.ndim != 3:
        raise RuntimeError(
            f"{name} must be 3D after normalization; "
            f"got shape {tuple(tensor.shape)}"
        )

    return tensor


# ============================================================================
# IMAGE NORMALIZATION
# ============================================================================

def normalize_mri(image: torch.Tensor) -> torch.Tensor:
    """
    Robust percentile normalization used for the training pipeline.
    """

    image = image.float()

    finite = torch.isfinite(image)

    if not finite.any():
        return torch.zeros_like(image)

    values = image[finite]

    p1 = torch.quantile(values, 0.01)
    p99 = torch.quantile(values, 0.99)

    if float((p99 - p1).abs().item()) < 1e-8:
        min_v = values.min()
        max_v = values.max()

        if float((max_v - min_v).abs().item()) < 1e-8:
            return torch.zeros_like(image)

        out = (image - min_v) / (max_v - min_v)
    else:
        out = (image - p1) / (p99 - p1)

    out = torch.clamp(out, 0.0, 1.0)
    out[~finite] = 0.0

    return out


# ============================================================================
# RESIZE
# ============================================================================

def resize_image_3d(
    image: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:

    image = ensure_3d(image, "image")

    x = image.float().unsqueeze(0).unsqueeze(0)

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

    mask = ensure_3d(mask, "mask")

    x = mask.float().unsqueeze(0).unsqueeze(0)

    x = F.interpolate(
        x,
        size=target_shape,
        mode="nearest",
    )

    return x[0, 0].long()


# ============================================================================
# FOREGROUND CENTER
# ============================================================================

def compute_foreground_center(
    mask: torch.Tensor,
) -> Tuple[int, int, int] | None:

    mask = ensure_3d(mask, "mask")

    coords = torch.nonzero(mask > 0, as_tuple=False)

    if coords.numel() == 0:
        return None

    center = coords.float().mean(dim=0)

    return tuple(int(round(float(v))) for v in center)


# ============================================================================
# CROPPING
# ============================================================================

def crop_3d(
    image: torch.Tensor,
    mask: torch.Tensor,
    start: Tuple[int, int, int],
    crop_shape: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:

    image = ensure_3d(image, "image")
    mask = ensure_3d(mask, "mask")

    starts = list(start)
    ends = [
        starts[i] + crop_shape[i]
        for i in range(3)
    ]

    slices = tuple(
        slice(starts[i], ends[i])
        for i in range(3)
    )

    return image[slices], mask[slices]


def bounded_start(
    center: int,
    crop_size: int,
    full_size: int,
) -> int:

    start = center - crop_size // 2

    start = max(0, start)
    start = min(start, full_size - crop_size)

    return int(start)


def foreground_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    center = compute_foreground_center(mask)

    if center is None:
        return random_crop(image, mask)

    starts = tuple(
        bounded_start(
            center[i],
            CROP_SHAPE[i],
            FULL_SHAPE[i],
        )
        for i in range(3)
    )

    return crop_3d(
        image,
        mask,
        starts,
        CROP_SHAPE,
    )


def random_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    image = ensure_3d(image, "random image")
    mask = ensure_3d(mask, "random mask")

    starts = []

    for dim, crop in zip(FULL_SHAPE, CROP_SHAPE):
        max_start = max(0, dim - crop)

        if max_start == 0:
            starts.append(0)
        else:
            starts.append(
                random.randint(0, max_start)
            )

    return crop_3d(
        image,
        mask,
        tuple(starts),
        CROP_SHAPE,
    )


# ============================================================================
# AUGMENTATION
# ============================================================================

def augment_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    image = image.clone()
    mask = mask.clone()

    # Random flips.
    for dim in range(3):
        if random.random() < 0.5:
            image = torch.flip(image, dims=[dim])
            mask = torch.flip(mask, dims=[dim])

    # Intensity scale.
    if random.random() < 0.5:
        scale = random.uniform(0.90, 1.10)
        image = image * scale

    # Intensity shift.
    if random.random() < 0.5:
        shift = random.uniform(-0.05, 0.05)
        image = image + shift

    # Small Gaussian noise.
    if random.random() < 0.5:
        noise = torch.randn_like(image) * 0.02
        image = image + noise

    # Gamma.
    if random.random() < 0.5:
        gamma = random.uniform(0.85, 1.15)
        image = torch.clamp(image, 0.0, 1.0)
        image = image.pow(gamma)

    image = torch.clamp(image, 0.0, 1.0)

    return image, mask


# ============================================================================
# DATASET
# ============================================================================

class Part126Dataset(Dataset):
    """
    Loads cases directly through corrected Part11.

    Training:
        full volume -> 75% foreground crop / 25% random crop.

    Validation:
        full preprocessed volume only.
    """

    def __init__(
        self,
        cohort_csv: Path,
        part9,
        part11,
        training: bool,
    ):

        self.df = pd.read_csv(cohort_csv)

        self.part9 = part9
        self.part11 = part11
        self.training = training

    def __len__(self):
        return len(self.df)

    def _load_case(
        self,
        row: pd.Series,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:

        image, mask, metadata = self.part11.load_tensor_case(
            row,
            self.part9,
        )

        image = ensure_3d(image, "loaded image")
        mask = ensure_3d(mask, "loaded mask")

        image = normalize_mri(image)

        image = resize_image_3d(
            image,
            FULL_SHAPE,
        )

        mask = resize_mask_3d(
            mask,
            FULL_SHAPE,
        )

        return image, mask, metadata

    def __getitem__(self, index: int):

        row = self.df.iloc[index]

        image, mask, metadata = self._load_case(row)

        crop_mode = "full_volume"
        augmented = False

        if self.training:

            if random.random() < FOREGROUND_CROP_PROBABILITY:
                image, mask = foreground_crop(
                    image,
                    mask,
                )

                crop_mode = "foreground"

            else:
                image, mask = random_crop(
                    image,
                    mask,
                )

                crop_mode = "random"

            if random.random() < AUGMENT_PROBABILITY:
                image, mask = augment_pair(
                    image,
                    mask,
                )

                augmented = True

        image = image.float().unsqueeze(0)
        mask = mask.long()
        return {
        "image": image,
        "mask": mask,
        "crop_mode": crop_mode,
        "augmented": augmented,
    }


# ============================================================================
# LOSS — FOREGROUND SOFT DICE
# ============================================================================

def foreground_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    target = target.long()

    probs = torch.softmax(
        logits,
        dim=1,
    )

    target_one_hot = F.one_hot(
        target,
        num_classes=NUM_CLASSES,
    ).permute(0, 4, 1, 2, 3).float()

    dices = []

    for c in FOREGROUND_CLASSES:

        p = probs[:, c]
        t = target_one_hot[:, c]

        intersection = (
            p * t
        ).sum(dim=(1, 2, 3))

        denominator = (
            p.sum(dim=(1, 2, 3))
            + t.sum(dim=(1, 2, 3))
        )

        dice = (
            2.0 * intersection
            + smooth
        ) / (
            denominator
            + smooth
        )

        dices.append(dice)

    return 1.0 - torch.stack(
        dices,
        dim=0,
    ).mean()


# ============================================================================
# LOSS — FOREGROUND TVERSKY
# ============================================================================

def foreground_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = TV_ALPHA,
    beta: float = TV_BETA,
    smooth: float = TV_SMOOTH,
) -> torch.Tensor:

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    target = target.long()

    probs = torch.softmax(
        logits,
        dim=1,
    )

    target_one_hot = F.one_hot(
        target,
        num_classes=NUM_CLASSES,
    ).permute(0, 4, 1, 2, 3).float()

    losses = []

    for c in FOREGROUND_CLASSES:

        p = probs[:, c]
        t = target_one_hot[:, c]

        tp = (p * t).sum(
            dim=(1, 2, 3)
        )

        fp = (
            p * (1.0 - t)
        ).sum(
            dim=(1, 2, 3)
        )

        fn = (
            (1.0 - p) * t
        ).sum(
            dim=(1, 2, 3)
        )

        tversky = (
            tp + smooth
        ) / (
            tp
            + alpha * fp
            + beta * fn
            + smooth
        )

        losses.append(
            1.0 - tversky
        )

    return torch.stack(
        losses,
        dim=0,
    ).mean()


# ============================================================================
# COMBINED PART126 LOSS
# ============================================================================

def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    target = target.long()

    ce = F.cross_entropy(
        logits,
        target,
        reduction="mean",
    )

    tversky = foreground_tversky_loss(
        logits,
        target,
    )

    total = (
        CE_WEIGHT * ce
        + TV_WEIGHT * tversky
    )

    return (
        total,
        float(ce.detach().item()),
        float(tversky.detach().item()),
    )


# ============================================================================
# MODEL
# ============================================================================

def normalize_state_keys(
    state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:

    if not state:
        return state

    keys = list(state.keys())

    if all(
        k.startswith("module.")
        for k in keys
    ):

        return {
            k[len("module."):]: v
            for k, v in state.items()
        }

    return state


def extract_state_dict(
    checkpoint: Any,
) -> Dict[str, torch.Tensor]:

    if isinstance(checkpoint, dict):

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):

            value = checkpoint.get(key)

            if isinstance(value, dict):

                return normalize_state_keys(
                    value
                )

        if checkpoint and all(
            isinstance(v, torch.Tensor)
            for v in checkpoint.values()
        ):

            return normalize_state_keys(
                checkpoint
            )

    raise RuntimeError(
        "Could not locate model state_dict "
        "in Part104 checkpoint."
    )


def build_model(
    device: torch.device,
    part11,
) -> nn.Module:

    model = part11.create_model(
        device
    )

    if not isinstance(
        model,
        nn.Module,
    ):

        raise RuntimeError(
            "Part11 create_model did not "
            "return torch.nn.Module."
        )

    return model


# ============================================================================
# METRICS
# ============================================================================

def dice_for_class(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
) -> float | None:

    p = prediction == class_id
    t = target == class_id

    p_n = int(p.sum().item())
    t_n = int(t.sum().item())

    # If neither exists, do not award a perfect score.
    if p_n == 0 and t_n == 0:
        return None

    intersection = int(
        (p & t).sum().item()
    )

    denominator = p_n + t_n

    if denominator == 0:
        return 0.0

    return (
        2.0 * intersection
        / denominator
    )


def foreground_dice_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Any]:

    class_scores = []

    for c in FOREGROUND_CLASSES:

        score = dice_for_class(
            prediction,
            target,
            c,
        )

        class_scores.append(score)

    valid_scores = [
        x for x in class_scores
        if x is not None
    ]

    if valid_scores:
        mean_dice = float(
            np.mean(valid_scores)
        )
    else:
        mean_dice = 0.0

    pred_fg = int(
        (prediction > 0).sum().item()
    )

    target_fg = int(
        (target > 0).sum().item()
    )

    total_voxels = int(
        target.numel()
    )

    pred_ratio = (
        pred_fg / total_voxels
        if total_voxels
        else 0.0
    )

    target_ratio = (
        target_fg / total_voxels
        if total_voxels
        else 0.0
    )

    if target_fg > 0:
        volume_ratio = (
            pred_fg / target_fg
        )
    else:
        volume_ratio = 0.0 if pred_fg == 0 else float("inf")

    return {
        "mean_foreground_dice": mean_dice,
        "class_dice": class_scores,
        "pred_foreground_voxels": pred_fg,
        "target_foreground_voxels": target_fg,
        "pred_foreground_ratio": pred_ratio,
        "target_foreground_ratio": target_ratio,
        "foreground_volume_ratio": volume_ratio,
        "target_has_foreground": target_fg > 0,
        "prediction_has_foreground": pred_fg > 0,
    }


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def validate_full_volume(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
) -> Dict[str, Any]:

    model.eval()

    case_dice = []
    class_values = [
        [] for _ in FOREGROUND_CLASSES
    ]

    pred_ratios = []
    target_ratios = []
    volume_ratios = []

    target_fg_cases = 0
    prediction_fg_cases = 0

    completed = 0

    for index in range(len(dataset)):

        sample = dataset[index]

        image = sample["image"]
        target = sample["mask"]

        image = image.unsqueeze(0).to(
            device,
            non_blocking=True,
        )

        target = target.to(
            device,
            non_blocking=True,
        )

        logits = sliding_window_inference(
            inputs=image,
            roi_size=SW_ROI,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=SW_OVERLAP,
            mode=SW_MODE,
            sigma_scale=SW_SIGMA_SCALE,
        )

        prediction = torch.argmax(
            logits,
            dim=1,
        )[0]

        metrics = foreground_dice_metrics(
            prediction,
            target,
        )

        case_dice.append(
            metrics["mean_foreground_dice"]
        )

        for i, score in enumerate(
            metrics["class_dice"]
        ):

            if score is not None:
                class_values[i].append(
                    score
                )

        pred_ratios.append(
            metrics["pred_foreground_ratio"]
        )

        target_ratios.append(
            metrics["target_foreground_ratio"]
        )

        vr = metrics[
            "foreground_volume_ratio"
        ]

        if math.isfinite(vr):
            volume_ratios.append(vr)

        if metrics[
            "target_has_foreground"
        ]:
            target_fg_cases += 1

        if metrics[
            "prediction_has_foreground"
        ]:
            prediction_fg_cases += 1

        completed += 1

        if (
            completed % 10 == 0
            or completed == len(dataset)
        ):

            print(
                f"  Validation: "
                f"{completed}/{len(dataset)}"
            )

    mean_class_dice = []

    for values in class_values:

        if values:
            mean_class_dice.append(
                float(np.mean(values))
            )
        else:
            mean_class_dice.append(
                None
            )

    return {
        "mean_foreground_dice": float(
            np.mean(case_dice)
        ) if case_dice else 0.0,

        "median_foreground_dice": float(
            np.median(case_dice)
        ) if case_dice else 0.0,

        "class_dice": mean_class_dice,

        "mean_pred_foreground_ratio": float(
            np.mean(pred_ratios)
        ) if pred_ratios else 0.0,

        "mean_target_foreground_ratio": float(
            np.mean(target_ratios)
        ) if target_ratios else 0.0,

        "mean_foreground_volume_ratio": float(
            np.mean(volume_ratios)
        ) if volume_ratios else 0.0,

        "median_foreground_volume_ratio": float(
            np.median(volume_ratios)
        ) if volume_ratios else 0.0,

        "target_foreground_cases": target_fg_cases,

        "prediction_foreground_cases": prediction_fg_cases,

        "prediction_foreground_case_rate": (
            prediction_fg_cases / len(dataset)
            if len(dataset)
            else 0.0
        ),

        "validation_cases_completed": completed,
    }


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    history: List[Dict[str, Any]],
    best_score: float,
) -> None:

    torch.save(
        {
            "part": 126,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": (
                scaler.state_dict()
                if scaler is not None
                else None
            ),
            "best_score": best_score,
            "history": history,
            "architecture": {
                "in_channels": IN_CHANNELS,
                "out_channels": NUM_CLASSES,
                "feature_size": FEATURE_SIZE,
                "spatial_dims": 3,
            },
        },
        path,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    training_start = time.time()

    seed_everything(SEED)
    ensure_dirs()

    banner(
        "PART 126 — FOREGROUND-CALIBRATED "
        "SEGMENTATION FINE-TUNING"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Output       : {OUT}"
    )

    print(
        f"PyTorch      : {torch.__version__}"
    )

    print(
        f"CUDA         : {torch.cuda.is_available()}"
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device       : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------------------------
    # Required artifacts
    # ------------------------------------------------------------------------

    section(
        "CHECKING REQUIRED ARTIFACTS"
    )

    required = {
        "Part9 source": PART9,
        "Corrected Part11 source": PART11,
        "Training cohort": TRAIN_COHORT,
        "Validation cohort": VAL_COHORT,
        "Part104 checkpoint": PART104,
    }

    missing = []

    for name, path in required.items():

        ok = path.exists()

        print(
            f"{name:<35} "
            f"{'PASS' if ok else 'FAIL'}"
        )

        if not ok:
            missing.append(str(path))

    if missing:

        raise FileNotFoundError(
            "Missing required artifacts:\n"
            + "\n".join(missing)
        )

    # ------------------------------------------------------------------------
    # Part104 protection
    # ------------------------------------------------------------------------

    section(
        "VERIFYING PROTECTED PART104 BASELINE"
    )

    part104_sha = sha256(
        PART104
    )

    print(
        f"Part104 SHA256 : {part104_sha}"
    )

    if part104_sha != EXPECTED_PART104_SHA:

        print(
            "Part104 SHA verification : FAIL"
        )

        raise RuntimeError(
            "Part104 checkpoint SHA mismatch. "
            "Part126 will not continue."
        )

    print(
        "Part104 SHA verification : PASS"
    )

    # ------------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------------

    section(
        "IMPORTING ESTABLISHED PROJECT PIPELINE"
    )

    part9 = import_module_from_path(
        "part126_part9",
        PART9,
    )

    part11 = import_module_from_path(
        "part126_part11",
        PART11,
    )

    print(
        "Part9 import              : PASS"
    )

    print(
        "Corrected Part11 import   : PASS"
    )

    if not hasattr(
        part11,
        "create_model",
    ):

        raise RuntimeError(
            "Part11 create_model unavailable."
        )

    if not hasattr(
        part11,
        "load_tensor_case",
    ):

        raise RuntimeError(
            "Part11 load_tensor_case unavailable."
        )

    print(
        "Part11 create_model       : PASS"
    )

    print(
        "Part11 load_tensor_case   : PASS"
    )

    # ------------------------------------------------------------------------
    # Cohorts
    # ------------------------------------------------------------------------

    section(
        "LOADING COHORTS"
    )

    train_df = pd.read_csv(
        TRAIN_COHORT
    )

    val_df = pd.read_csv(
        VAL_COHORT
    )

    print(
        f"Training cases   : {len(train_df)}"
    )

    print(
        f"Validation cases : {len(val_df)}"
    )

    if len(train_df) != 500:
        print(
            "WARNING: training cohort is not "
            "the expected 500-case cohort."
        )

    if len(val_df) != 100:
        print(
            "WARNING: validation cohort is not "
            "the expected 100-case cohort."
        )

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    section(
        "BUILDING PART126 DATASETS"
    )

    train_dataset = Part126Dataset(
        TRAIN_COHORT,
        part9,
        part11,
        training=True,
    )

    val_dataset = Part126Dataset(
        VAL_COHORT,
        part9,
        part11,
        training=False,
    )

    print(
        f"Full preprocessing shape : "
        f"{FULL_SHAPE}"
    )

    print(
        f"Training crop shape      : "
        f"{CROP_SHAPE}"
    )

    print(
        f"Foreground crop prob.    : "
        f"{FOREGROUND_CROP_PROBABILITY}"
    )

    print(
        f"Random crop probability  : "
        f"{RANDOM_CROP_PROBABILITY}"
    )

    # ------------------------------------------------------------------------
    # Smoke test
    # ------------------------------------------------------------------------

    section(
        "RUNNING DATASET SMOKE TEST"
    )

    smoke_train = train_dataset[0]
    smoke_val = val_dataset[0]

    print(
        f"Train image shape : "
        f"{tuple(smoke_train['image'].shape)}"
    )

    print(
        f"Train mask shape  : "
        f"{tuple(smoke_train['mask'].shape)}"
    )

    print(
        f"Train crop mode   : "
        f"{smoke_train['crop_mode']}"
    )

    print(
        f"Train augmented   : "
        f"{smoke_train['augmented']}"
    )

    print(
        f"Train foreground  : "
        f"{int((smoke_train['mask'] > 0).sum())}"
    )

    print(
        f"Val image shape   : "
        f"{tuple(smoke_val['image'].shape)}"
    )

    print(
        f"Val mask shape    : "
        f"{tuple(smoke_val['mask'].shape)}"
    )

    if tuple(
        smoke_train["image"].shape
    ) != (
        1,
        *CROP_SHAPE,
    ):

        raise RuntimeError(
            "Training smoke-test image shape "
            "is incorrect."
        )

    if tuple(
        smoke_val["image"].shape
    ) != (
        1,
        *FULL_SHAPE,
    ):

        raise RuntimeError(
            "Validation smoke-test image shape "
            "is incorrect."
        )

    print(
        "Dataset smoke test : PASS"
    )

    # ------------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    section(
        "BUILDING CANONICAL SWIN-UNETR"
    )

    model = build_model(
        device,
        part11,
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Trainable parameters : "
        f"{parameter_count:,}"
    )

    if parameter_count != 4_078_116:

        raise RuntimeError(
            "Unexpected model parameter count. "
            f"Expected 4,078,116; got "
            f"{parameter_count:,}"
        )

    # ------------------------------------------------------------------------
    # Load protected Part104 initialization
    # ------------------------------------------------------------------------

    section(
        "LOADING PROTECTED PART104 INITIALIZATION"
    )

    checkpoint = torch.load(
        PART104,
        map_location=device,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    load_result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        f"Missing keys    : "
        f"{len(load_result.missing_keys)}"
    )

    print(
        f"Unexpected keys : "
        f"{len(load_result.unexpected_keys)}"
    )

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):

        raise RuntimeError(
            "Part104 state_dict did not load "
            "strictly."
        )

    print(
        "Part104 initialization : PASS"
    )

    # ------------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------------

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

    scaler = (
        torch.amp.GradScaler("cuda")
        if (
            AMP
            and device.type == "cuda"
        )
        else None
    )

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    section(
        "STARTING PART126 TRAINING"
    )

    print(
        "Loss                     : "
        "Cross-Entropy + Tversky"
    )

    print(
        f"CE weight                : "
        f"{CE_WEIGHT}"
    )

    print(
        f"Tversky weight           : "
        f"{TV_WEIGHT}"
    )

    print(
        f"Tversky alpha            : "
        f"{TV_ALPHA}"
    )

    print(
        f"Tversky beta             : "
        f"{TV_BETA}"
    )

    print(
        "False-positive emphasis  : ENABLED"
    )

    print(
        "Validation               : "
        "FULL VOLUME + SLIDING WINDOW"
    )

    history = []

    best_dice = -float("inf")
    best_epoch = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        epoch_total = 0.0
        epoch_ce = 0.0
        epoch_tv = 0.0

        train_dice_values = []

        optimizer.zero_grad(
            set_to_none=True
        )

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            targets = batch["mask"].to(
                device,
                non_blocking=True,
            )

            if scaler is not None:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    logits = model(
                        images
                    )

                    loss, ce_value, tv_value = (
                        combined_loss(
                            logits,
                            targets,
                        )
                    )

                    scaled_loss = (
                        loss
                        / GRADIENT_ACCUMULATION
                    )

                scaler.scale(
                    scaled_loss
                ).backward()

            else:

                logits = model(
                    images
                )

                loss, ce_value, tv_value = (
                    combined_loss(
                        logits,
                        targets,
                    )
                )

                scaled_loss = (
                    loss
                    / GRADIENT_ACCUMULATION
                )

                scaled_loss.backward()

            epoch_total += float(
                loss.detach().item()
            )

            epoch_ce += ce_value
            epoch_tv += tv_value

            with torch.no_grad():

                prediction = torch.argmax(
                    logits,
                    dim=1,
                )

                batch_metrics = (
                    foreground_dice_metrics(
                        prediction[0],
                        targets[0],
                    )
                )

                train_dice_values.append(
                    batch_metrics[
                        "mean_foreground_dice"
                    ]
                )

            if (
                step % GRADIENT_ACCUMULATION == 0
                or step == len(train_loader)
            ):

                if scaler is not None:

                    scaler.unscale_(
                        optimizer
                    )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                if scaler is not None:

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

        train_batches = len(
            train_loader
        )

        train_loss = (
            epoch_total
            / train_batches
        )

        train_ce = (
            epoch_ce
            / train_batches
        )

        train_tv = (
            epoch_tv
            / train_batches
        )

        train_dice = (
            float(np.mean(train_dice_values))
            if train_dice_values
            else 0.0
        )

        # --------------------------------------------------------------------
        # Full-volume validation
        # --------------------------------------------------------------------

        print()

        print(
            f"Epoch {epoch:02d} "
            f"full-volume validation..."
        )

        val_metrics = validate_full_volume(
            model,
            val_dataset,
            device,
        )

        val_dice = (
            val_metrics[
                "mean_foreground_dice"
            ]
        )

        val_ratio = (
            val_metrics[
                "mean_foreground_volume_ratio"
            ]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_tversky": train_tv,
            "train_foreground_dice": train_dice,
            "val_foreground_dice": val_dice,
            "val_median_foreground_dice":
                val_metrics[
                    "median_foreground_dice"
                ],
            "val_class1_dice":
                val_metrics["class_dice"][0],
            "val_class2_dice":
                val_metrics["class_dice"][1],
            "val_class3_dice":
                val_metrics["class_dice"][2],
            "val_class4_dice":
                val_metrics["class_dice"][3],
            "val_class5_dice":
                val_metrics["class_dice"][4],
            "val_pred_foreground_ratio":
                val_metrics[
                    "mean_pred_foreground_ratio"
                ],
            "val_target_foreground_ratio":
                val_metrics[
                    "mean_target_foreground_ratio"
                ],
            "val_foreground_volume_ratio":
                val_ratio,
            "val_median_foreground_volume_ratio":
                val_metrics[
                    "median_foreground_volume_ratio"
                ],
            "val_prediction_foreground_cases":
                val_metrics[
                    "prediction_foreground_cases"
                ],
            "val_target_foreground_cases":
                val_metrics[
                    "target_foreground_cases"
                ],
            "learning_rate": current_lr,
        }

        history.append(row)

        print()
        print(
            f"Epoch {epoch:02d}"
        )

        print(
            f"  Train loss       : "
            f"{train_loss:.6f}"
        )

        print(
            f"  Train CE         : "
            f"{train_ce:.6f}"
        )

        print(
            f"  Train Tversky    : "
            f"{train_tv:.6f}"
        )

        print(
            f"  Train FG Dice    : "
            f"{train_dice:.6f}"
        )

        print(
            f"  Val FG Dice      : "
            f"{val_dice:.6f}"
        )

        print(
            f"  Val median Dice  : "
            f"{val_metrics['median_foreground_dice']:.6f}"
        )

        print(
            f"  Val FG ratio     : "
            f"{val_metrics['mean_pred_foreground_ratio']:.8f}"
        )

        print(
            f"  Target FG ratio  : "
            f"{val_metrics['mean_target_foreground_ratio']:.8f}"
        )

        print(
            f"  Pred/target FG   : "
            f"{val_ratio:.6f}"
        )

        print(
            f"  Pred FG cases    : "
            f"{val_metrics['prediction_foreground_cases']}"
            f"/{len(val_dataset)}"
        )

        # --------------------------------------------------------------------
        # Save epoch checkpoint
        # --------------------------------------------------------------------

        epoch_ckpt = (
            CHECKPOINT_DIR
            / f"part126_epoch_{epoch:02d}.pth"
        )

        save_checkpoint(
            epoch_ckpt,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            history,
            max(
                best_dice,
                val_dice,
            ),
        )

        # --------------------------------------------------------------------
        # Best checkpoint
        #
        # Dice remains the primary criterion, but unlike Part125:
        #   - absent classes are not awarded Dice=1
        #   - foreground volume behavior is always recorded
        #   - zero-foreground collapse is visible
        # --------------------------------------------------------------------

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
                "*** NEW BEST PART126 CHECKPOINT "
                f"— epoch {epoch}, "
                f"validation FG Dice "
                f"{best_dice:.6f} ***"
            )

        scheduler.step()

    # ------------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------------

    section(
        "SAVING FINAL PART126 CHECKPOINT"
    )

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

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_CSV,
        index=False,
    )

    elapsed = (
        time.time()
        - training_start
    )

    final_summary = {
        "status": "COMPLETED",
        "part": "126",
        "purpose":
            "foreground-calibrated segmentation fine-tuning",
        "training_performed": True,
        "epochs_completed": EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_foreground_dice":
            best_dice,
        "training_time_seconds":
            elapsed,

        "part104_sha256_expected":
            EXPECTED_PART104_SHA,

        "part104_sha256_at_start":
            part104_sha,

        "part104_protected":
            True,

        "architecture": {
            "in_channels": IN_CHANNELS,
            "out_channels": NUM_CLASSES,
            "feature_size": FEATURE_SIZE,
            "spatial_dims": 3,
            "parameter_count": 4_078_116,
        },

        "full_shape":
            FULL_SHAPE,

        "crop_shape":
            CROP_SHAPE,

        "foreground_crop_probability":
            FOREGROUND_CROP_PROBABILITY,

        "random_crop_probability":
            RANDOM_CROP_PROBABILITY,

        "loss": {
            "cross_entropy_weight":
                CE_WEIGHT,
            "tversky_weight":
                TV_WEIGHT,
            "tversky_alpha":
                TV_ALPHA,
            "tversky_beta":
                TV_BETA,
            "description":
                "cross-entropy + foreground Tversky "
                "with stronger false-positive penalty",
        },

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "sliding_window": {
            "roi_size":
                SW_ROI,
            "overlap":
                SW_OVERLAP,
            "sw_batch_size":
                SW_BATCH_SIZE,
            "mode":
                SW_MODE,
            "sigma_scale":
                SW_SIGMA_SCALE,
        },

        "best_checkpoint":
            str(BEST_CKPT),

        "final_checkpoint":
            str(FINAL_CKPT),

        "history_csv":
            str(HISTORY_CSV),

        "clinical_validation":
            False,

        "note":
            "Targets are development/pseudo-mask labels. "
            "Results must not be interpreted as clinical validation.",
    }

    METRICS_JSON.write_text(
        json.dumps(
            final_summary,
            indent=2,
            default=json_default,
        ),
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

    report = f"""
PART 126 — FOREGROUND-CALIBRATED SEGMENTATION
FINE-TUNING REPORT

Status:
    COMPLETED

Purpose:
    Controlled fine-tuning from the protected Part104 checkpoint
    using foreground-aware sampling and false-positive-aware
    Tversky loss.

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
    parameters = 4,078,116

Data:
    training cases = {len(train_dataset)}
    validation cases = {len(val_dataset)}
    full shape = {FULL_SHAPE}
    crop shape = {CROP_SHAPE}

Sampling:
    foreground-centred probability =
        {FOREGROUND_CROP_PROBABILITY}

    random crop probability =
        {RANDOM_CROP_PROBABILITY}

Loss:
    cross-entropy weight = {CE_WEIGHT}
    Tversky weight = {TV_WEIGHT}
    Tversky alpha = {TV_ALPHA}
    Tversky beta = {TV_BETA}

Validation:
    full-volume sliding-window inference
    ROI = {SW_ROI}
    overlap = {SW_OVERLAP}
    mode = {SW_MODE}
    sigma_scale = {SW_SIGMA_SCALE}

Best epoch:
    {best_epoch}

Best validation foreground Dice:
    {best_dice:.6f}

Best checkpoint:
    {BEST_CKPT}

Final checkpoint:
    {FINAL_CKPT}

Clinical validation:
    NO

Important:
    Development pseudo-mask labels were used.
    These results must not be interpreted as clinical validation.
"""

    REPORT_TXT.write_text(
        report.strip(),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------------

    banner(
        "PART126 COMPLETED"
    )

    print(
        f"Best epoch              : "
        f"{best_epoch}"
    )

    print(
        f"Best validation Dice    : "
        f"{best_dice:.6f}"
    )

    print()
    print(
        f"Best checkpoint         : "
        f"{BEST_CKPT}"
    )

    print(
        f"Final checkpoint        : "
        f"{FINAL_CKPT}"
    )

    print(
        f"Training history        : "
        f"{HISTORY_CSV}"
    )

    print(
        f"Metrics JSON            : "
        f"{METRICS_JSON}"
    )

    print(
        f"Report                  : "
        f"{REPORT_TXT}"
    )

    print()
    print(
        "Part104 protected : YES"
    )

    print(
        "Clinical validation : NO"
    )


if __name__ == "__main__":
    main()