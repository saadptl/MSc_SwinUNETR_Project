"""
PHASE 4 - PART 11
RSNA-ONLY SWIN-UNETR CONTROLLED PILOT TRAINING
ROBUST DICOM-SHAPE COMPATIBILITY VERSION

Purpose
-------
Run a controlled RSNA-only pilot training experiment using the
configuration validated by Part 10.

This version keeps Part 9 unchanged. If Part 9 rejects a DICOM series
because its slices have inconsistent in-plane dimensions, Part 11 uses
a LOCAL compatibility loader which:
  1. reads the DICOM slices,
  2. sorts them in the same physical order,
  3. center-pads/crops each slice to the maximum H/W in that series,
  4. constructs the image volume,
  5. loads the Part 6 pseudo-mask,
  6. harmonizes the mask to the same native H/W,
  7. performs the final training resize to (64, 96, 96).

No RSNA files are modified.
Part 9 is not weakened or changed.
SPIDER is not used.
"""

from __future__ import annotations

import gc
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR


# ============================================================================
# PATHS
# ============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)
TRAIN_IMAGES_DIR = RSNA_ROOT / "train_images"

PART8_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
)
MANIFEST_DIR = PART8_DIR / "manifests"

TRAIN_MANIFEST = MANIFEST_DIR / "rsna_part8_train_manifest.csv"
VAL_MANIFEST = MANIFEST_DIR / "rsna_part8_validation_manifest.csv"

PART6_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part6_pseudomask_generation"
)
PSEUDOMASK_DIR = PART6_DIR / "pseudo_masks"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part11_controlled_pilot_training"
)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_DIR = OUTPUT_DIR / "reports"


# ============================================================================
# LOCKED PART 10 CONFIGURATION
# ============================================================================

PATCH_SIZE = (64, 96, 96)
IN_CHANNELS = 1
NUM_CLASSES = 6
FEATURE_SIZE = 12
BATCH_SIZE = 1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

USE_AMP = True
GRADIENT_ACCUMULATION = 1

PILOT_TRAIN_CASES = 100
PILOT_VAL_CASES = 30
PILOT_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3

SEED = 42
CPU_THREADS = min(4, max(1, torch.get_num_threads()))


LABELS = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def parameter_summary(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


# ============================================================================
# PART 9 IMPORT
# ============================================================================

def load_part9_module() -> Any:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    import segmentation_rsna_part9_3d_dataset_loader as part9

    if not hasattr(part9, "load_case"):
        raise RuntimeError(
            "Part 9 loader does not expose load_case(row)."
        )

    return part9


# ============================================================================
# MANIFEST SELECTION
# ============================================================================

def select_pilot_rows(
    manifest_path: Path,
    max_cases: int,
    seed: int,
) -> pd.DataFrame:

    df = pd.read_csv(manifest_path)

    if df.empty:
        raise RuntimeError(
            f"Manifest is empty: {manifest_path}"
        )

    if max_cases < len(df):
        df = df.sample(
            n=max_cases,
            random_state=seed,
        )

    return df.reset_index(drop=True)


# ============================================================================
# IDENTIFIER HELPERS
# ============================================================================

def first_value(
    row: pd.Series,
    names: List[str],
    default: Any = None,
) -> Any:

    for name in names:
        if name in row.index:
            value = row[name]

            if pd.isna(value):
                continue

            return value

    return default


def as_id(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def resolve_series_dir(
    row: pd.Series,
) -> Path:

    study_id = as_id(
        first_value(
            row,
            ["study_id", "study"],
        )
    )

    series_id = as_id(
        first_value(
            row,
            ["series_id", "series"],
        )
    )

    if not study_id or not series_id:
        raise RuntimeError(
            "Manifest row does not contain usable study_id/series_id."
        )

    path = (
        TRAIN_IMAGES_DIR
        / study_id
        / series_id
    )

    if not path.exists():
        raise FileNotFoundError(
            f"RSNA DICOM series not found: {path}"
        )

    return path


def resolve_mask_path(
    row: pd.Series,
) -> Path:

    # Prefer explicit path columns from the Part 8 manifest.
    path_candidates = [
        "pseudo_mask_path",
        "mask_path",
        "pseudo_mask",
        "mask",
        "target_path",
        "label_path",
    ]

    for column in path_candidates:
        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if not value:
            continue

        path = Path(value)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if path.exists():
            return path

    # Fallback: identify the pseudo-mask using study/series IDs.
    study_id = as_id(
        first_value(row, ["study_id", "study"])
    )
    series_id = as_id(
        first_value(row, ["series_id", "series"])
    )

    if not study_id or not series_id:
        raise RuntimeError(
            "Cannot resolve pseudo-mask path from manifest row."
        )

    exact_candidates = [
        PSEUDOMASK_DIR / f"{study_id}_{series_id}.npz",
        PSEUDOMASK_DIR / f"{study_id}__{series_id}.npz",
        PSEUDOMASK_DIR / f"{series_id}.npz",
    ]

    for path in exact_candidates:
        if path.exists():
            return path

    # Last-resort recursive lookup by series ID.
    matches = list(
        PSEUDOMASK_DIR.rglob(
            f"*{series_id}*.npz"
        )
    )

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        "Could not resolve pseudo-mask for "
        f"study={study_id}, series={series_id}."
    )


# ============================================================================
# DICOM COMPATIBILITY LOADER
# ============================================================================

def dicom_sort_key(ds: Any) -> Tuple:
    instance = getattr(
        ds,
        "InstanceNumber",
        None,
    )

    try:
        instance_value = int(instance)
    except Exception:
        instance_value = 10**9

    ipp = getattr(
        ds,
        "ImagePositionPatient",
        None,
    )

    if ipp is not None and len(ipp) >= 3:
        try:
            z = float(ipp[2])
        except Exception:
            z = float(instance_value)
    else:
        z = float(instance_value)

    return (
        z,
        instance_value,
        str(getattr(ds, "SOPInstanceUID", "")),
    )


def read_dicom_series_robust(
    series_dir: Path,
) -> Tuple[np.ndarray, List[Any], Dict[str, Any]]:

    files = sorted(
        series_dir.glob("*.dcm")
    )

    if not files:
        raise RuntimeError(
            f"No DICOM files found: {series_dir}"
        )

    datasets = []

    for path in files:
        ds = pydicom.dcmread(
            str(path),
            force=True,
        )

        if not hasattr(ds, "PixelData"):
            continue

        datasets.append(ds)

    if not datasets:
        raise RuntimeError(
            f"No pixel-bearing DICOM files found: {series_dir}"
        )

    datasets.sort(
        key=dicom_sort_key
    )

    arrays = []

    for ds in datasets:
        arr = ds.pixel_array.astype(
            np.float32
        )

        slope = float(
            getattr(
                ds,
                "RescaleSlope",
                1.0,
            )
        )
        intercept = float(
            getattr(
                ds,
                "RescaleIntercept",
                0.0,
            )
        )

        arr = arr * slope + intercept
        arrays.append(arr)

    shapes = [
        tuple(a.shape)
        for a in arrays
    ]

    unique_shapes = sorted(
        set(shapes)
    )

    max_h = max(
        a.shape[0]
        for a in arrays
    )
    max_w = max(
        a.shape[1]
        for a in arrays
    )

    used_harmonization = (
        len(unique_shapes) > 1
    )

    if used_harmonization:
        print(
            "  DICOM compatibility: "
            f"{unique_shapes} -> "
            f"({max_h}, {max_w})"
        )

    volume = np.zeros(
        (
            len(arrays),
            max_h,
            max_w,
        ),
        dtype=np.float32,
    )

    for i, arr in enumerate(arrays):

        h, w = arr.shape

        # Center placement preserves the approximate image geometry
        # without modifying the original DICOM files.
        y0 = (max_h - h) // 2
        x0 = (max_w - w) // 2

        volume[
            i,
            y0:y0 + h,
            x0:x0 + w,
        ] = arr

    metadata = {
        "original_shapes": [
            list(s)
            for s in unique_shapes
        ],
        "native_shape": list(volume.shape),
        "harmonized": used_harmonization,
        "num_slices": len(arrays),
    }

    return (
        volume,
        datasets,
        metadata,
    )


# ============================================================================
# PSEUDO-MASK LOADER
# ============================================================================

def load_pseudomask(
    path: Path,
) -> np.ndarray:

    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        keys = list(data.keys())

        preferred = [
            "mask",
            "pseudo_mask",
            "segmentation",
            "labels",
        ]

        selected = None

        for key in preferred:
            if key in data:
                selected = key
                break

        if selected is None:
            if not keys:
                raise RuntimeError(
                    f"Empty NPZ file: {path}"
                )

            selected = keys[0]

        mask = np.asarray(
            data[selected]
        )

    if mask.ndim != 3:
        raise RuntimeError(
            f"Expected 3D pseudo-mask, got "
            f"shape={mask.shape} from {path}"
        )

    return mask.astype(
        np.int64
    )


# ============================================================================
# NATIVE MASK HARMONIZATION
# ============================================================================

def harmonize_mask_to_image(
    mask: np.ndarray,
    image_shape: Tuple[int, int, int],
) -> np.ndarray:

    target_d, target_h, target_w = image_shape

    mask = np.asarray(
        mask,
        dtype=np.int64,
    )

    # Depth must correspond to DICOM slice count.
    if mask.shape[0] != target_d:
        raise RuntimeError(
            "Pseudo-mask depth does not match DICOM depth: "
            f"mask={mask.shape}, image={image_shape}"
        )

    if (
        mask.shape[1] == target_h
        and mask.shape[2] == target_w
    ):
        return mask

    # Center-pad/crop in-plane, mirroring the image harmonization.
    result = np.zeros(
        image_shape,
        dtype=np.int64,
    )

    src_h, src_w = mask.shape[1:]

    # Source coordinates.
    src_y0 = max(
        0,
        (src_h - target_h) // 2,
    )
    src_x0 = max(
        0,
        (src_w - target_w) // 2,
    )

    # Destination coordinates.
    dst_y0 = max(
        0,
        (target_h - src_h) // 2,
    )
    dst_x0 = max(
        0,
        (target_w - src_w) // 2,
    )

    copy_h = min(
        src_h,
        target_h,
    )
    copy_w = min(
        src_w,
        target_w,
    )

    result[
        :,
        dst_y0:dst_y0 + copy_h,
        dst_x0:dst_x0 + copy_w,
    ] = mask[
        :,
        src_y0:src_y0 + copy_h,
        src_x0:src_x0 + copy_w,
    ]

    return result


# ============================================================================
# ROBUST CASE LOADER
# ============================================================================

def load_case_robust(
    row: pd.Series,
    part9: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

    # First use the validated Part 9 loader.
    try:
        image, mask, info = part9.load_case(
            row
        )

        info = dict(
            info
            if isinstance(info, dict)
            else {}
        )

        info["part11_loader"] = "part9_exact"
        info["dicom_harmonized"] = False

        return (
            np.asarray(image, dtype=np.float32),
            np.asarray(mask, dtype=np.int64),
            info,
        )

    except RuntimeError as exc:

        message = str(exc)

        if "Inconsistent DICOM shapes" not in message:
            raise

        print(
            "  Part 9 strict loader detected mixed DICOM shapes."
        )
        print(
            "  Activating Part 11 local compatibility loader."
        )

    # Local fallback.
    series_dir = resolve_series_dir(
        row
    )

    mask_path = resolve_mask_path(
        row
    )

    image, datasets, dicom_info = (
        read_dicom_series_robust(
            series_dir
        )
    )

    mask = load_pseudomask(
        mask_path
    )

    mask = harmonize_mask_to_image(
        mask,
        tuple(image.shape),
    )

    study_id = as_id(
        first_value(
            row,
            ["study_id", "study"],
        )
    )

    series_id = as_id(
        first_value(
            row,
            ["series_id", "series"],
        )
    )

    series_description = first_value(
        row,
        [
            "series_description",
            "series_type",
        ],
        "",
    )

    info = {
        "study_id": study_id,
        "series_id": series_id,
        "series_description": (
            ""
            if series_description is None
            else str(series_description)
        ),
        "part11_loader": "local_robust_fallback",
        "dicom_harmonized": True,
        "dicom_original_shapes": (
            dicom_info["original_shapes"]
        ),
        "dicom_native_shape": (
            dicom_info["native_shape"]
        ),
        "pseudo_mask_path": str(
            mask_path
        ),
        "num_slices": (
            dicom_info["num_slices"]
        ),
    }

    return (
        image,
        mask,
        info,
    )


# ============================================================================
# FINAL TRAINING PREPROCESSING
# ============================================================================

def resize_3d(
    array: np.ndarray,
    target_shape: Tuple[int, int, int],
    is_mask: bool,
) -> np.ndarray:

    array = np.asarray(
        array,
        dtype=(
            np.int64
            if is_mask
            else np.float32
        ),
    )

    if tuple(array.shape) == tuple(
        target_shape
    ):
        return array.copy()

    tensor = torch.from_numpy(
        array.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)

    if is_mask:
        out = F.interpolate(
            tensor,
            size=target_shape,
            mode="nearest",
        )
    else:
        out = F.interpolate(
            tensor,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )

    result = (
        out
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )

    if is_mask:
        return np.rint(
            result
        ).astype(np.int64)

    return result.astype(
        np.float32
    )


def preprocess_case(
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    mask = np.asarray(
        mask,
        dtype=np.int64,
    )

    image = resize_3d(
        image,
        PATCH_SIZE,
        is_mask=False,
    )

    mask = resize_3d(
        mask,
        PATCH_SIZE,
        is_mask=True,
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    lo = float(
        np.percentile(
            image,
            1.0,
        )
    )
    hi = float(
        np.percentile(
            image,
            99.0,
        )
    )

    if hi > lo:
        image = np.clip(
            image,
            lo,
            hi,
        )
        image = (
            image - lo
        ) / (
            hi - lo
        )
    else:
        image = np.zeros_like(
            image,
            dtype=np.float32,
        )

    # Protect against unexpected pseudo-label values.
    invalid = (
        (mask < 0)
        | (mask >= NUM_CLASSES)
    )

    if np.any(invalid):
        bad_values = sorted(
            np.unique(
                mask[invalid]
            ).tolist()
        )

        raise RuntimeError(
            "Pseudo-mask contains invalid "
            f"class IDs: {bad_values}"
        )

    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    mask_tensor = torch.from_numpy(
        mask
    ).long()

    return (
        image_tensor,
        mask_tensor,
    )


def load_tensor_case(
    row: pd.Series,
    part9: Any,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
]:

    image, mask, info = load_case_robust(
        row,
        part9,
    )

    image_tensor, mask_tensor = (
        preprocess_case(
            image,
            mask,
        )
    )

    return (
        image_tensor,
        mask_tensor,
        info,
    )


# ============================================================================
# MODEL
# ============================================================================

def create_model(
    device: torch.device,
) -> torch.nn.Module:

    model = SwinUNETR(
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        use_checkpoint=False,
        spatial_dims=3,
    )

    return model.to(device)


# ============================================================================
# DICE METRICS
# ============================================================================

def dice_from_prediction(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[float, List[float]]:

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    class_dice = []

    for class_id in range(
        1,
        NUM_CLASSES,
    ):

        pred = (
            prediction
            == class_id
        )

        true = (
            target
            == class_id
        )

        intersection = (
            pred & true
        ).sum().float()

        denominator = (
            pred.sum()
            + true.sum()
        ).float()

        if denominator.item() == 0:
            dice = 1.0
        else:
            dice = (
                2.0
                * intersection
                / denominator
            ).item()

        class_dice.append(
            float(dice)
        )

    return (
        float(
            np.mean(class_dice)
        ),
        class_dice,
    )


# ============================================================================
# TRAIN
# ============================================================================

def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    rows: pd.DataFrame,
    part9: Any,
    device: torch.device,
    scaler: Any,
) -> Tuple[
    float,
    float,
    List[float],
    float,
    int,
]:

    model.train()

    losses = []
    dice_values = []
    class_dice_values = []

    fallback_count = 0

    start = time.time()

    optimizer.zero_grad(
        set_to_none=True
    )

    for step, (_, row) in enumerate(
        rows.iterrows(),
        start=1,
    ):

        image, mask, info = (
            load_tensor_case(
                row,
                part9,
            )
        )

        if info.get(
            "part11_loader"
        ) == "local_robust_fallback":
            fallback_count += 1

        image = image.unsqueeze(
            0
        ).to(
            device,
            non_blocking=True,
        )

        mask = mask.unsqueeze(
            0
        ).to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(
                USE_AMP
                and device.type == "cuda"
            ),
        ):

            logits = model(
                image
            )

            loss = loss_function(
                logits,
                mask.unsqueeze(1),
            )

            backward_loss = (
                loss
                / GRADIENT_ACCUMULATION
            )

        if scaler is not None:
            scaler.scale(
                backward_loss
            ).backward()
        else:
            backward_loss.backward()

        if (
            step
            % GRADIENT_ACCUMULATION
            == 0
            or step == len(rows)
        ):

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

        mean_dice, class_dice = (
            dice_from_prediction(
                logits.detach(),
                mask,
            )
        )

        losses.append(
            float(loss.item())
        )
        dice_values.append(
            mean_dice
        )
        class_dice_values.append(
            class_dice
        )

        del (
            image,
            mask,
            logits,
            loss,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (
            step == 1
            or step % 10 == 0
            or step == len(rows)
        ):

            print(
                f"  Train {step:>3}/{len(rows)} "
                f"Loss={np.mean(losses):.4f} "
                f"Dice={np.mean(dice_values):.4f}"
            )

    class_means = np.mean(
        np.asarray(
            class_dice_values
        ),
        axis=0,
    ).tolist()

    return (
        float(np.mean(losses)),
        float(np.mean(dice_values)),
        class_means,
        float(
            (time.time() - start)
            / 60.0
        ),
        fallback_count,
    )


# ============================================================================
# VALIDATE
# ============================================================================

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    rows: pd.DataFrame,
    part9: Any,
    device: torch.device,
) -> Tuple[
    float,
    float,
    List[float],
    float,
    int,
]:

    model.eval()

    losses = []
    dice_values = []
    class_dice_values = []

    fallback_count = 0

    start = time.time()

    for step, (_, row) in enumerate(
        rows.iterrows(),
        start=1,
    ):

        image, mask, info = (
            load_tensor_case(
                row,
                part9,
            )
        )

        if info.get(
            "part11_loader"
        ) == "local_robust_fallback":
            fallback_count += 1

        image = image.unsqueeze(
            0
        ).to(device)

        mask = mask.unsqueeze(
            0
        ).to(device)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(
                USE_AMP
                and device.type == "cuda"
            ),
        ):

            logits = model(
                image
            )

            loss = loss_function(
                logits,
                mask.unsqueeze(1),
            )

        mean_dice, class_dice = (
            dice_from_prediction(
                logits,
                mask,
            )
        )

        losses.append(
            float(loss.item())
        )
        dice_values.append(
            mean_dice
        )
        class_dice_values.append(
            class_dice
        )

        del (
            image,
            mask,
            logits,
            loss,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (
            step == 1
            or step % 10 == 0
            or step == len(rows)
        ):

            print(
                f"  Val   {step:>3}/{len(rows)} "
                f"Loss={np.mean(losses):.4f} "
                f"Dice={np.mean(dice_values):.4f}"
            )

    class_means = np.mean(
        np.asarray(
            class_dice_values
        ),
        axis=0,
    ).tolist()

    return (
        float(np.mean(losses)),
        float(np.mean(dice_values)),
        class_means,
        float(
            (time.time() - start)
            / 60.0
        ),
        fallback_count,
    )


# ============================================================================
# CHECKPOINTS
# ============================================================================

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_val_dice: float,
    history: List[Dict[str, Any]],
) -> None:

    payload = {
        "epoch": epoch,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "best_val_dice": (
            best_val_dice
        ),
        "history": history,
        "config": {
            "dataset": "RSNA only",
            "spider_used": False,
            "patch_size": list(
                PATCH_SIZE
            ),
            "in_channels": IN_CHANNELS,
            "num_classes": NUM_CLASSES,
            "feature_size": FEATURE_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "amp": USE_AMP,
            "train_cases": (
                PILOT_TRAIN_CASES
            ),
            "validation_cases": (
                PILOT_VAL_CASES
            ),
            "pilot_epochs": PILOT_EPOCHS,
            "seed": SEED,
            "dicom_shape_fallback": True,
        },
    }

    if scaler is not None:
        payload[
            "scaler_state_dict"
        ] = scaler.state_dict()

    torch.save(
        payload,
        path,
    )


# ============================================================================
# REPORTING
# ============================================================================

def save_history(
    history: List[Dict[str, Any]],
) -> None:

    path = (
        OUTPUT_DIR
        / "part11_training_history.csv"
    )

    rows = []

    for item in history:

        row = {
            "epoch": item["epoch"],
            "train_loss": item[
                "train_loss"
            ],
            "train_mean_dice": item[
                "train_mean_dice"
            ],
            "val_loss": item[
                "val_loss"
            ],
            "val_mean_dice": item[
                "val_mean_dice"
            ],
            "epoch_time_min": item[
                "epoch_time_min"
            ],
            "peak_allocated_gb": item[
                "peak_allocated_gb"
            ],
            "peak_reserved_gb": item[
                "peak_reserved_gb"
            ],
            "train_fallback_cases": item[
                "train_fallback_cases"
            ],
            "val_fallback_cases": item[
                "val_fallback_cases"
            ],
        }

        for i, value in enumerate(
            item["val_class_dice"],
            start=1,
        ):
            row[
                f"val_class_{i}_dice"
            ] = value

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )

    print(
        f"Saved: {path}"
    )


def save_reports(
    history: List[Dict[str, Any]],
    best_epoch: int,
    best_val_dice: float,
    total_training_minutes: float,
) -> None:

    summary = {
        "phase": "Phase 4 - Part 11",
        "description": (
            "RSNA-only Swin-UNETR controlled pilot training"
        ),
        "dataset": (
            "RSNA 2024 Lumbar Spine "
            "Degenerative Classification"
        ),
        "spider_used": False,
        "patch_size": list(
            PATCH_SIZE
        ),
        "batch_size": BATCH_SIZE,
        "feature_size": FEATURE_SIZE,
        "classes": NUM_CLASSES,
        "amp": USE_AMP,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "train_cases": (
            PILOT_TRAIN_CASES
        ),
        "validation_cases": (
            PILOT_VAL_CASES
        ),
        "requested_epochs": PILOT_EPOCHS,
        "completed_epochs": len(
            history
        ),
        "best_epoch": best_epoch,
        "best_validation_mean_foreground_dice": (
            best_val_dice
        ),
        "total_training_minutes": (
            total_training_minutes
        ),
        "dicom_shape_fallback_enabled": True,
        "training_completed": bool(
            history
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "phase4_part11_pilot_training_summary.json"
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

    print(
        f"Saved: {json_path}"
    )

    report_path = (
        REPORT_DIR
        / "phase4_part11_pilot_training_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 4 - PART 11\n"
            "RSNA-ONLY SWIN-UNETR "
            "CONTROLLED PILOT TRAINING\n\n"
        )

        f.write(
            "Dataset: RSNA only\n"
            "SPIDER used: NO\n"
            "Test set used for training: NO\n"
            f"Patch size: {PATCH_SIZE}\n"
            f"Batch size: {BATCH_SIZE}\n"
            f"Feature size: {FEATURE_SIZE}\n"
            f"Classes: {NUM_CLASSES}\n"
            f"AMP: {USE_AMP}\n"
            f"Learning rate: {LEARNING_RATE}\n"
            f"Weight decay: {WEIGHT_DECAY}\n"
            f"Train cases: {PILOT_TRAIN_CASES}\n"
            f"Validation cases: {PILOT_VAL_CASES}\n"
            f"Requested epochs: {PILOT_EPOCHS}\n"
            "DICOM mixed-shape compatibility: ENABLED\n\n"
        )

        f.write(
            f"Completed epochs: {len(history)}\n"
            f"Best epoch: {best_epoch}\n"
            f"Best validation mean foreground Dice: "
            f"{best_val_dice:.6f}\n"
            f"Total training time: "
            f"{total_training_minutes:.2f} min\n\n"
        )

        f.write(
            "Epoch history\n"
        )
        f.write(
            "-" * 78
            + "\n"
        )

        for item in history:

            f.write(
                f"Epoch {item['epoch']}: "
                f"train_loss={item['train_loss']:.6f}, "
                f"train_dice={item['train_mean_dice']:.6f}, "
                f"val_loss={item['val_loss']:.6f}, "
                f"val_dice={item['val_mean_dice']:.6f}, "
                f"train_fallback={item['train_fallback_cases']}, "
                f"val_fallback={item['val_fallback_cases']}, "
                f"peak_reserved_gb="
                f"{item['peak_reserved_gb']:.3f}\n"
            )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    header(
        "PHASE 4 - PART 11\n"
        "RSNA-ONLY SWIN-UNETR CONTROLLED PILOT TRAINING"
    )

    set_seed(SEED)
    torch.set_num_threads(
        CPU_THREADS
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("PROJECT ROOT")
    print(PROJECT_ROOT)

    print()
    print("RSNA DATASET")
    print(RSNA_ROOT)

    print()
    print("PART 8 TRAIN MANIFEST")
    print(TRAIN_MANIFEST)

    print()
    print("PART 8 VALIDATION MANIFEST")
    print(VAL_MANIFEST)

    print()
    print("PART 6 PSEUDO-MASKS")
    print(PSEUDOMASK_DIR)

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    # ---------------------------------------------------------------------
    # PATHS
    # ---------------------------------------------------------------------

    header("PATH VALIDATION")

    required = {
        "RSNA root": RSNA_ROOT,
        "train_images": TRAIN_IMAGES_DIR,
        "train manifest": TRAIN_MANIFEST,
        "validation manifest": VAL_MANIFEST,
        "Part 6 pseudo_masks": PSEUDOMASK_DIR,
        "Part 9 loader": (
            SRC_DIR
            / "segmentation_rsna_part9_3d_dataset_loader.py"
        ),
    }

    for name, path in required.items():
        print(
            f"{name:<28}: "
            f"{'FOUND' if path.exists() else 'MISSING'}"
        )

    missing = [
        name
        for name, path in required.items()
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required paths: "
            + ", ".join(missing)
        )

    # ---------------------------------------------------------------------
    # DEVICE
    # ---------------------------------------------------------------------

    header(
        "PYTORCH / GPU ENVIRONMENT"
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"PyTorch version : "
        f"{torch.__version__}"
    )
    print(
        f"CUDA available  : "
        f"{torch.cuda.is_available()}"
    )
    print(
        f"Device          : "
        f"{device}"
    )

    if device.type == "cuda":

        props = (
            torch.cuda
            .get_device_properties(0)
        )

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            f"GPU memory      : "
            f"{props.total_memory / (1024 ** 3):.2f} GB"
        )
        print(
            f"CUDA runtime    : "
            f"{torch.version.cuda}"
        )

    print(
        f"CPU threads     : "
        f"{CPU_THREADS}"
    )

    # ---------------------------------------------------------------------
    # CONFIG
    # ---------------------------------------------------------------------

    header(
        "PART 10 VALIDATED TRAINING CONFIGURATION"
    )

    print(
        f"Patch size       : {PATCH_SIZE}"
    )
    print(
        f"Batch size       : {BATCH_SIZE}"
    )
    print(
        f"Feature size     : {FEATURE_SIZE}"
    )
    print(
        f"Classes          : {NUM_CLASSES}"
    )
    print(
        f"Learning rate    : {LEARNING_RATE}"
    )
    print(
        f"Weight decay     : {WEIGHT_DECAY}"
    )
    print(
        f"AMP              : {USE_AMP}"
    )
    print(
        f"Train cases      : {PILOT_TRAIN_CASES}"
    )
    print(
        f"Validation cases : {PILOT_VAL_CASES}"
    )
    print(
        f"Pilot epochs     : {PILOT_EPOCHS}"
    )
    print(
        f"Seed             : {SEED}"
    )

    print()
    for i in range(
        NUM_CLASSES
    ):
        print(
            f"{i}: {LABELS[i]}"
        )

    # ---------------------------------------------------------------------
    # PART 9
    # ---------------------------------------------------------------------

    header(
        "LOADING VALIDATED PART 9 DATA LOADER"
    )

    part9 = load_part9_module()

    print(
        "✓ Part 9 loader imported."
    )
    print(
        "✓ Strict Part 9 path retained."
    )
    print(
        "✓ Local mixed-shape fallback enabled only in Part 11."
    )

    # ---------------------------------------------------------------------
    # PILOT DATA
    # ---------------------------------------------------------------------

    header(
        "SELECTING CONTROLLED PILOT CASES"
    )

    train_rows = select_pilot_rows(
        TRAIN_MANIFEST,
        PILOT_TRAIN_CASES,
        SEED,
    )

    val_rows = select_pilot_rows(
        VAL_MANIFEST,
        PILOT_VAL_CASES,
        SEED,
    )

    print(
        f"Pilot train cases    : "
        f"{len(train_rows)}"
    )
    print(
        f"Pilot validation     : "
        f"{len(val_rows)}"
    )

    # ---------------------------------------------------------------------
    # SANITY
    # ---------------------------------------------------------------------

    header(
        "PILOT DATA SANITY CHECK"
    )

    sample_image, sample_mask, sample_info = (
        load_tensor_case(
            train_rows.iloc[0],
            part9,
        )
    )

    print(
        f"Study ID          : "
        f"{sample_info.get('study_id', '')}"
    )
    print(
        f"Series ID         : "
        f"{sample_info.get('series_id', '')}"
    )
    print(
        f"Series type       : "
        f"{sample_info.get('series_description', '')}"
    )
    print(
        f"Loader            : "
        f"{sample_info.get('part11_loader', '')}"
    )
    print(
        f"Image tensor      : "
        f"{tuple(sample_image.shape)}"
    )
    print(
        f"Mask tensor       : "
        f"{tuple(sample_mask.shape)}"
    )
    print(
        f"Image range       : "
        f"{float(sample_image.min()):.4f} - "
        f"{float(sample_image.max()):.4f}"
    )
    print(
        f"Mask labels       : "
        f"{sorted(torch.unique(sample_mask).tolist())}"
    )
    print(
        f"Foreground voxels : "
        f"{int((sample_mask > 0).sum())}"
    )

    expected = (
        1,
        *PATCH_SIZE,
    )

    if tuple(sample_image.shape) != expected:
        raise RuntimeError(
            f"Unexpected image tensor shape. "
            f"Expected {expected}, "
            f"got {tuple(sample_image.shape)}"
        )

    # ---------------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------------

    header(
        "CREATING SWIN-UNETR"
    )

    reset_cuda_memory()

    model = create_model(
        device
    )

    total_params, trainable_params = (
        parameter_summary(model)
    )

    print(
        "✓ Swin-UNETR created."
    )
    print(
        f"Total parameters     : "
        f"{total_params:,}"
    )
    print(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )

    # ---------------------------------------------------------------------
    # LOSS
    # ---------------------------------------------------------------------

    header(
        "LOSS / OPTIMIZER"
    )

    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "Loss          : DiceCELoss"
    )
    print(
        "Optimizer     : AdamW"
    )
    print(
        f"Learning rate : {LEARNING_RATE}"
    )
    print(
        f"Weight decay  : {WEIGHT_DECAY}"
    )

    scaler = None

    if (
        USE_AMP
        and device.type == "cuda"
    ):
        scaler = torch.amp.GradScaler(
            "cuda"
        )
        print(
            "AMP           : enabled"
        )
    else:
        print(
            "AMP           : disabled"
        )

    # ---------------------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------------------

    header(
        "STARTING CONTROLLED PILOT TRAINING"
    )

    print(
        "RSNA only."
    )
    print(
        "SPIDER is not used."
    )
    print(
        "Test manifest is not used."
    )
    print(
        "Part 9 remains strict; Part 11 handles mixed DICOM shapes locally."
    )

    history = []

    best_val_dice = -float(
        "inf"
    )
    best_epoch = 0
    patience = 0

    best_checkpoint = (
        CHECKPOINT_DIR
        / "best_model.pth"
    )

    last_checkpoint = (
        CHECKPOINT_DIR
        / "last_model.pth"
    )

    training_start = time.time()

    for epoch in range(
        1,
        PILOT_EPOCHS + 1,
    ):

        header(
            f"EPOCH {epoch}/{PILOT_EPOCHS}"
        )

        reset_cuda_memory()

        (
            train_loss,
            train_dice,
            train_class_dice,
            train_time,
            train_fallback,
        ) = train_one_epoch(
            model,
            optimizer,
            loss_function,
            train_rows,
            part9,
            device,
            scaler,
        )

        reset_cuda_memory()

        (
            val_loss,
            val_dice,
            val_class_dice,
            val_time,
            val_fallback,
        ) = validate(
            model,
            loss_function,
            val_rows,
            part9,
            device,
        )

        if device.type == "cuda":
            peak_allocated_gb = (
                torch.cuda
                .max_memory_allocated()
                / (1024 ** 3)
            )
            peak_reserved_gb = (
                torch.cuda
                .max_memory_reserved()
                / (1024 ** 3)
            )
        else:
            peak_allocated_gb = 0.0
            peak_reserved_gb = 0.0

        item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mean_dice": train_dice,
            "train_class_dice": train_class_dice,
            "val_loss": val_loss,
            "val_mean_dice": val_dice,
            "val_class_dice": val_class_dice,
            "epoch_time_min": (
                train_time
                + val_time
            ),
            "peak_allocated_gb": (
                peak_allocated_gb
            ),
            "peak_reserved_gb": (
                peak_reserved_gb
            ),
            "train_fallback_cases": (
                train_fallback
            ),
            "val_fallback_cases": (
                val_fallback
            ),
        }

        history.append(
            item
        )

        print()
        print(
            f"Train Loss       : "
            f"{train_loss:.6f}"
        )
        print(
            f"Train Mean Dice  : "
            f"{train_dice:.6f}"
        )
        print(
            f"Validation Loss  : "
            f"{val_loss:.6f}"
        )
        print(
            f"Validation Dice  : "
            f"{val_dice:.6f}"
        )
        print(
            f"Train DICOM fallbacks : "
            f"{train_fallback}"
        )
        print(
            f"Val DICOM fallbacks   : "
            f"{val_fallback}"
        )

        for class_id, value in enumerate(
            val_class_dice,
            start=1,
        ):
            print(
                f"{LABELS[class_id]:<34}: "
                f"{value:.6f}"
            )

        print(
            f"Peak GPU allocated : "
            f"{peak_allocated_gb:.3f} GB"
        )
        print(
            f"Peak GPU reserved  : "
            f"{peak_reserved_gb:.3f} GB"
        )

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scaler,
            epoch,
            best_val_dice,
            history,
        )

        print(
            f"Saved: {last_checkpoint}"
        )

        if val_dice > best_val_dice:

            best_val_dice = val_dice
            best_epoch = epoch
            patience = 0

            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scaler,
                epoch,
                best_val_dice,
                history,
            )

            print(
                "✓ New best validation model saved."
            )
            print(
                f"Best validation Dice: "
                f"{best_val_dice:.6f}"
            )

        else:

            patience += 1

            print(
                f"No validation improvement. "
                f"Patience {patience}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if patience >= (
            EARLY_STOPPING_PATIENCE
        ):
            print(
                "Early stopping triggered."
            )
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------------

    total_training_minutes = (
        time.time()
        - training_start
    ) / 60.0

    header(
        "PART 11 FINAL SUMMARY"
    )

    print(
        f"Completed epochs : "
        f"{len(history)}"
    )
    print(
        f"Best epoch       : "
        f"{best_epoch}"
    )
    print(
        f"Best Val Dice    : "
        f"{best_val_dice:.6f}"
    )
    print(
        f"Training time    : "
        f"{total_training_minutes:.2f} min"
    )

    print()
    print(
        "Best checkpoint:"
    )
    print(
        best_checkpoint
    )

    save_history(
        history
    )

    save_reports(
        history,
        best_epoch,
        best_val_dice,
        total_training_minutes,
    )

    result = {
        "completed_epochs": len(
            history
        ),
        "best_epoch": best_epoch,
        "best_validation_dice": (
            best_val_dice
        ),
        "total_training_minutes": (
            total_training_minutes
        ),
        "best_checkpoint": str(
            best_checkpoint
        ),
        "last_checkpoint": str(
            last_checkpoint
        ),
        "patch_size": list(
            PATCH_SIZE
        ),
        "batch_size": BATCH_SIZE,
        "feature_size": FEATURE_SIZE,
        "train_cases": len(
            train_rows
        ),
        "validation_cases": len(
            val_rows
        ),
        "spider_used": False,
        "dicom_shape_fallback_enabled": True,
    }

    result_path = (
        OUTPUT_DIR
        / "part11_training_result.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        f"Saved: {result_path}"
    )

    header(
        "PHASE 4 - PART 11 COMPLETE"
    )

    print(
        "PASS - controlled RSNA-only pilot training completed."
    )
    print(
        "SPIDER used             : NO"
    )
    print(
        "Test set used in train  : NO"
    )
    print(
        "DICOM fallback enabled  : YES"
    )
    print(
        "Model checkpoint saved  : YES"
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "=" * 78
        )
        print(
            "PART 11 INTERRUPTED"
        )
        print(
            "=" * 78
        )
        raise

    except Exception as exc:

        print()
        print(
            "=" * 78
        )
        print(
            "PART 11 ERROR"
        )
        print(
            "=" * 78
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
