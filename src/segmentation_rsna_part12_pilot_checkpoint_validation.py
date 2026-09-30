"""
PHASE 4 - PART 12
RSNA-ONLY SWIN-UNETR PILOT CHECKPOINT VALIDATION

Purpose
-------
Validate the best Part 11 checkpoint on the held-out RSNA validation manifest.

This part:
- Uses RSNA only.
- Uses the Part 11 best checkpoint.
- Does NOT train or modify model weights.
- Does NOT use the RSNA test manifest.
- Computes validation loss/Dice and per-class Dice.
- Computes foreground precision/recall where meaningful.
- Saves case-level metrics, summary JSON/CSV, and representative prediction
  overlays as PNG files.
- Reuses the Part 11-compatible mixed-DICOM loader behavior locally by first
  trying Part 9 and falling back to a robust native-DICOM loader.

Important:
The targets are RSNA point-derived pseudo-masks, not manual segmentation
ground truth. Therefore these metrics describe agreement with the pseudo-targets,
not clinical segmentation accuracy.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    import pydicom
except Exception as exc:
    raise RuntimeError("pydicom is required.") from exc

try:
    from monai.networks.nets import SwinUNETR
    from monai.losses import DiceCELoss
except Exception as exc:
    raise RuntimeError("MONAI is required for Part 12.") from exc


# ---------------------------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES = RSNA_ROOT / "train_images"

PART8_MANIFEST_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
)

VAL_MANIFEST = PART8_MANIFEST_DIR / "rsna_part8_validation_manifest.csv"

PART11_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part11_controlled_pilot_training"
)

CHECKPOINT = PART11_DIR / "checkpoints" / "best_model.pth"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part12_pilot_checkpoint_validation"
)

CASE_DIR = OUTPUT_DIR / "case_visualizations"

PART9_PATH = PROJECT_ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py"

SEED = 42
NUM_CLASSES = 6
FEATURE_SIZE = 12
PATCH_SIZE = (64, 96, 96)
BATCH_SIZE = 1

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def banner(text: str) -> None:
    print("=" * 78)
    print(text)
    print("=" * 78)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_paths() -> None:
    required = {
        "RSNA root": RSNA_ROOT,
        "train_images": TRAIN_IMAGES,
        "validation manifest": VAL_MANIFEST,
        "Part 11 directory": PART11_DIR,
        "best checkpoint": CHECKPOINT,
        "Part 9 loader": PART9_PATH,
    }

    banner("PATH VALIDATION")
    for name, path in required.items():
        print(f"{name:<30}: {'FOUND' if path.exists() else 'MISSING'}")

    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Part 12 input(s) missing:\n" + "\n".join(missing)
        )


def import_part9():
    banner("LOADING PART 9 DATA LOADER")
    spec = importlib.util.spec_from_file_location("segmentation_rsna_part9", PART9_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Could not import Part 9 loader.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["segmentation_rsna_part9"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "load_case"):
        raise AttributeError("Part 9 loader does not expose load_case(row).")

    print("✓ Part 9 loader imported.")
    print("✓ Part 9 strict loader retained.")
    print("✓ Part 12 compatibility fallback enabled only when necessary.")
    return module


# ---------------------------------------------------------------------------
# LOCAL MIXED-SHAPE DICOM FALLBACK
# ---------------------------------------------------------------------------

def _natural_key(path: Path):
    stem = path.stem
    try:
        return (0, int(stem))
    except Exception:
        return (1, stem)


def _read_series(series_path: Path) -> Tuple[np.ndarray, List[object]]:
    files = sorted(series_path.glob("*.dcm"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No DICOM files in {series_path}")

    datasets = []
    for f in files:
        ds = pydicom.dcmread(str(f))
        if hasattr(ds, "pixel_array"):
            datasets.append(ds)

    if not datasets:
        raise RuntimeError(f"No readable DICOM pixel data in {series_path}")

    datasets.sort(
        key=lambda ds: (
            float(getattr(ds, "ImagePositionPatient", [0, 0, float("inf")])[2])
            if hasattr(ds, "ImagePositionPatient")
            else float(getattr(ds, "InstanceNumber", 0))
        )
    )

    arrays = [np.asarray(ds.pixel_array, dtype=np.float32) for ds in datasets]
    shapes = {tuple(a.shape) for a in arrays}

    if len(shapes) == 1:
        vol = np.stack(arrays, axis=0)
        return vol, datasets

    # Compatibility behavior used by Part 11:
    # pad/crop each slice to the largest H,W, centered.
    target_h = max(a.shape[0] for a in arrays)
    target_w = max(a.shape[1] for a in arrays)

    print(f"  DICOM compatibility: {sorted(shapes)} -> ({target_h}, {target_w})")

    out = np.zeros((len(arrays), target_h, target_w), dtype=np.float32)

    for i, arr in enumerate(arrays):
        h, w = arr.shape
        y0 = (target_h - h) // 2
        x0 = (target_w - w) // 2
        out[i, y0:y0 + h, x0:x0 + w] = arr

    return out, datasets


def _normalize_volume(vol: np.ndarray) -> np.ndarray:
    vol = np.nan_to_num(vol.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Robust percentile normalization.
    lo, hi = np.percentile(vol, [1.0, 99.0])
    if hi > lo:
        vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    else:
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol.fill(0.0)

    return vol.astype(np.float32)


def _resize_3d_nearest(mask: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    t, h, w = target
    z_idx = np.linspace(0, mask.shape[0] - 1, t).round().astype(int)
    y_idx = np.linspace(0, mask.shape[1] - 1, h).round().astype(int)
    x_idx = np.linspace(0, mask.shape[2] - 1, w).round().astype(int)
    return mask[np.ix_(z_idx, y_idx, x_idx)]


def _resize_3d_linear(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    # Uses torch interpolation for a dependable CPU implementation.
    x = torch.from_numpy(vol[None, None]).float()
    y = torch.nn.functional.interpolate(
        x, size=target, mode="trilinear", align_corners=False
    )
    return y[0, 0].numpy().astype(np.float32)


def _find_manifest_value(row, names, default=None):
    for n in names:
        if n in row.index and pd.notna(row[n]):
            return row[n]
    return default


def local_compat_load_case(row) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    study_id = str(int(float(_find_manifest_value(row, ["study_id"]))))
    series_id = str(int(float(_find_manifest_value(row, ["series_id"]))))

    series_path = TRAIN_IMAGES / study_id / series_id
    if not series_path.exists():
        raise FileNotFoundError(f"Series directory not found: {series_path}")

    volume, datasets = _read_series(series_path)
    volume = _normalize_volume(volume)

    mask_path_value = _find_manifest_value(
        row,
        ["pseudo_mask_path", "mask_path", "mask", "pseudomask_path"],
    )

    if mask_path_value is None:
        raise RuntimeError(
            f"Validation manifest does not contain a pseudo-mask path for "
            f"{study_id}/{series_id}."
        )

    mask_path = Path(str(mask_path_value))
    if not mask_path.is_absolute():
        mask_path = PROJECT_ROOT / mask_path

    if not mask_path.exists():
        # Try basename under Part 6 pseudo_masks.
        alt = (
            PROJECT_ROOT
            / "outputs"
            / "segmentation"
            / "rsna_part6_pseudomask_generation"
            / "pseudo_masks"
            / mask_path.name
        )
        if alt.exists():
            mask_path = alt

    if not mask_path.exists():
        raise FileNotFoundError(f"Pseudo-mask not found: {mask_path}")

    with np.load(mask_path, allow_pickle=True) as data:
        if "mask" in data:
            mask = data["mask"]
        elif "pseudo_mask" in data:
            mask = data["pseudo_mask"]
        elif "segmentation" in data:
            mask = data["segmentation"]
        elif "labels" in data:
            mask = data["labels"]
        else:
            mask = data[list(data.keys())[0]]

    mask = np.asarray(mask)

    if mask.shape != volume.shape:
        mask = _resize_3d_nearest(mask, volume.shape)

    image = torch.from_numpy(volume).float().unsqueeze(0)
    mask = torch.from_numpy(mask.astype(np.int64))

    info = {
        "study_id": study_id,
        "series_id": series_id,
        "series_description": str(
            _find_manifest_value(row, ["series_description"], "unknown")
        ),
        "loader": "part12_mixed_shape_compatibility",
        "original_shape": tuple(volume.shape),
    }

    return image, mask, info


def load_case_with_fallback(part9, row):
    try:
        image, mask, info = part9.load_case(row)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        return image.float(), mask.long(), dict(info or {}, loader="part9_exact")
    except RuntimeError as exc:
        if "Inconsistent DICOM shapes" not in str(exc):
            raise

        print("  Part 9 strict loader detected mixed DICOM shapes.")
        print("  Activating Part 12 local compatibility loader.")
        return local_compat_load_case(row)


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def create_model(device: torch.device):
    model = SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        use_checkpoint=False,
    )
    model.to(device)
    return model


def load_checkpoint(model: nn.Module, device: torch.device) -> dict:
    banner("LOADING PART 11 BEST CHECKPOINT")
    checkpoint = torch.load(CHECKPOINT, map_location=device)

    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break

    if not isinstance(state, dict):
        raise RuntimeError("Unsupported checkpoint structure.")

    # Remove common DataParallel prefix.
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Missing keys      : {len(missing)}")
    print(f"Unexpected keys   : {len(unexpected)}")

    if missing:
        print("WARNING: checkpoint has missing model keys.")
    if unexpected:
        print("WARNING: checkpoint has unexpected keys.")

    return checkpoint if isinstance(checkpoint, dict) else {}


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------

def center_crop_or_pad_3d(
    arr: torch.Tensor,
    target: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Center crop/pad a [D,H,W] tensor."""
    d, h, w = arr.shape
    td, th, tw = target

    out = torch.full(target, pad_value, dtype=arr.dtype)

    sd = min(d, td)
    sh = min(h, th)
    sw = min(w, tw)

    src_d0 = (d - sd) // 2
    src_h0 = (h - sh) // 2
    src_w0 = (w - sw) // 2

    dst_d0 = (td - sd) // 2
    dst_h0 = (th - sh) // 2
    dst_w0 = (tw - sw) // 2

    out[
        dst_d0:dst_d0 + sd,
        dst_h0:dst_h0 + sh,
        dst_w0:dst_w0 + sw,
    ] = arr[
        src_d0:src_d0 + sd,
        src_h0:src_h0 + sh,
        src_w0:src_w0 + sw,
    ]

    return out


def prepare_patch(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = image.squeeze(0).float()
    mask = mask.long()

    # Resize preserving the full volume first.
    image = torch.nn.functional.interpolate(
        image[None, None],
        size=PATCH_SIZE,
        mode="trilinear",
        align_corners=False,
    )[0, 0]

    mask = torch.nn.functional.interpolate(
        mask[None, None].float(),
        size=PATCH_SIZE,
        mode="nearest",
    )[0, 0].long()

    return image[None], mask


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    include_background: bool = False,
) -> Dict[int, float]:
    result = {}
    start = 0 if include_background else 1

    for c in range(start, num_classes):
        p = pred == c
        t = target == c

        p_sum = int(p.sum().item())
        t_sum = int(t.sum().item())

        if p_sum == 0 and t_sum == 0:
            result[c] = float("nan")
            continue

        inter = int((p & t).sum().item())
        result[c] = (2.0 * inter) / (p_sum + t_sum + 1e-8)

    return result


def precision_recall_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[int, Tuple[float, float]]:
    out = {}

    for c in range(1, NUM_CLASSES):
        p = pred == c
        t = target == c

        tp = int((p & t).sum().item())
        fp = int((p & ~t).sum().item())
        fn = int((~p & t).sum().item())

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        out[c] = (precision, recall)

    return out


def multiclass_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    values = dice_per_class(pred, target, include_background=False)
    vals = [v for v in values.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

def choose_slice(mask: np.ndarray, pred: np.ndarray) -> int:
    combined = (mask > 0) | (pred > 0)
    counts = combined.reshape(combined.shape[0], -1).sum(axis=1)
    if counts.max() > 0:
        return int(np.argmax(counts))
    return int(mask.shape[0] // 2)


def save_case_visualization(
    case_dir: Path,
    image: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    info: Dict,
    dice: float,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)

    z = choose_slice(target, pred)

    fig = plt.figure(figsize=(14, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image[z], cmap="gray")
    ax1.set_title(f"Input\nslice {z}")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(image[z], cmap="gray")
    ax2.imshow(np.ma.masked_where(target[z] == 0, target[z]), alpha=0.45)
    ax2.set_title("RSNA pseudo-target")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(image[z], cmap="gray")
    ax3.imshow(np.ma.masked_where(pred[z] == 0, pred[z]), alpha=0.45)
    ax3.set_title(f"Prediction\nDice={dice:.4f}")
    ax3.axis("off")

    fig.suptitle(
        f"Study {info.get('study_id')} | Series {info.get('series_id')} | "
        f"{info.get('series_description')}"
    )
    fig.tight_layout()

    fig.savefig(case_dir / "checkpoint_validation_visualization.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    part9,
    manifest: pd.DataFrame,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[List[Dict], Dict]:
    model.eval()

    rows = []
    visual_candidates = []

    for idx, row in manifest.iterrows():
        image, mask, info = load_case_with_fallback(part9, row)

        image, mask = prepare_patch(image, mask)

        image_b = image.unsqueeze(0).to(device)
        mask_b = mask.unsqueeze(0).to(device)

        logits = model(image_b)
        loss = loss_fn(logits, mask_b.unsqueeze(1))

        pred = torch.argmax(logits, dim=1)[0].cpu()
        target = mask.cpu()

        class_dice = dice_per_class(pred, target)
        pr = precision_recall_per_class(pred, target)
        overall = multiclass_dice(pred, target)

        row_out = {
            "index": int(idx),
            "study_id": info.get("study_id"),
            "series_id": info.get("series_id"),
            "series_description": info.get("series_description"),
            "loader": info.get("loader"),
            "loss": float(loss.item()),
            "mean_dice": overall,
        }

        for c in range(1, NUM_CLASSES):
            row_out[f"dice_class_{c}"] = class_dice[c]
            row_out[f"precision_class_{c}"] = pr[c][0]
            row_out[f"recall_class_{c}"] = pr[c][1]

        row_out["target_foreground_voxels"] = int((target > 0).sum().item())
        row_out["prediction_foreground_voxels"] = int((pred > 0).sum().item())

        rows.append(row_out)

        visual_candidates.append(
            (
                overall,
                info,
                image[0].cpu().numpy(),
                target.numpy(),
                pred.numpy(),
            )
        )

        print(
            f"  [{len(rows):03d}/{len(manifest)}] "
            f"{info.get('study_id')} | {info.get('series_id')} | "
            f"Loss={loss.item():.4f} Dice={overall:.4f}"
        )

    df = pd.DataFrame(rows)

    summary = {
        "cases": int(len(df)),
        "mean_loss": float(df["loss"].mean()),
        "median_loss": float(df["loss"].median()),
        "mean_dice": float(df["mean_dice"].mean()),
        "median_dice": float(df["mean_dice"].median()),
    }

    for c in range(1, NUM_CLASSES):
        col = f"dice_class_{c}"
        pcol = f"precision_class_{c}"
        rcol = f"recall_class_{c}"

        summary[col] = float(df[col].mean())
        summary[pcol] = float(df[pcol].mean())
        summary[rcol] = float(df[rcol].mean())

    # Representative visual cases:
    # worst, best, median and one mixed-shape/fallback case if available.
    candidates_sorted = sorted(visual_candidates, key=lambda x: x[0])
    selected = []

    if candidates_sorted:
        selected.append(candidates_sorted[0])
        selected.append(candidates_sorted[len(candidates_sorted) // 2])
        selected.append(candidates_sorted[-1])

    fallback_cases = [
        x for x in visual_candidates
        if x[1].get("loader") != "part9_exact"
    ]
    if fallback_cases:
        selected.append(fallback_cases[0])

    # Deduplicate by study/series.
    seen = set()
    unique_selected = []
    for item in selected:
        key = (
            str(item[1].get("study_id")),
            str(item[1].get("series_id")),
        )
        if key not in seen:
            seen.add(key)
            unique_selected.append(item)

    for rank, item in enumerate(unique_selected, start=1):
        dice, info, image, target, pred = item
        name = f"{rank:02d}_{info.get('study_id')}_{info.get('series_id')}"
        save_case_visualization(
            CASE_DIR / name,
            image,
            target,
            pred,
            info,
            dice,
        )

    return rows, summary


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)

    banner("PHASE 4 - PART 12")
    print("RSNA-ONLY SWIN-UNETR PILOT CHECKPOINT VALIDATION")
    print()

    print("PROJECT ROOT")
    print(PROJECT_ROOT)
    print()
    print("RSNA DATASET")
    print(RSNA_ROOT)
    print()
    print("PART 11 BEST CHECKPOINT")
    print(CHECKPOINT)
    print()
    print("VALIDATION MANIFEST")
    print(VAL_MANIFEST)
    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)
    print()

    ensure_paths()

    banner("PYTORCH / GPU ENVIRONMENT")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"Device          : {device}")

    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory      : {props.total_memory / (1024**3):.2f} GB")

    banner("LOADING VALIDATION MANIFEST")
    manifest = pd.read_csv(VAL_MANIFEST)
    print(f"Validation rows : {len(manifest)}")

    # Part 12 intentionally validates the same 30-case controlled validation
    # cohort used in Part 11, if the manifest contains more rows.
    if len(manifest) > 30:
        manifest = manifest.sample(n=30, random_state=SEED).reset_index(drop=True)
        print("Controlled validation cohort : 30 cases")
    else:
        manifest = manifest.reset_index(drop=True)
        print(f"Controlled validation cohort : {len(manifest)} cases")

    banner("LOADING PART 9 DATA LOADER")
    part9 = import_part9()

    banner("CREATING SWIN-UNETR")
    model = create_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("✓ Swin-UNETR created.")
    print(f"Total parameters     : {total_params:,}")
    print(f"Output classes       : {NUM_CLASSES}")

    checkpoint_meta = load_checkpoint(model, device)

    banner("LOSS / EVALUATION CONFIGURATION")
    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )
    print("Loss         : DiceCELoss")
    print("Mode         : evaluation only")
    print("Gradient     : disabled")
    print("Weight update: disabled")

    banner("STARTING CHECKPOINT VALIDATION")
    print("RSNA only.")
    print("SPIDER is not used.")
    print("Test manifest is not used.")
    print("No training is performed.")
    print()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    rows, summary = validate(
        model=model,
        part9=part9,
        manifest=manifest,
        device=device,
        loss_fn=loss_fn,
    )
    elapsed = time.time() - start

    case_df = pd.DataFrame(rows)
    case_csv = OUTPUT_DIR / "part12_validation_case_metrics.csv"
    case_df.to_csv(case_csv, index=False)

    summary["evaluation_time_seconds"] = float(elapsed)
    summary["evaluation_time_minutes"] = float(elapsed / 60.0)

    if torch.cuda.is_available():
        summary["peak_gpu_allocated_gb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024**3)
        )
        summary["peak_gpu_reserved_gb"] = float(
            torch.cuda.max_memory_reserved(device) / (1024**3)
        )

    summary["checkpoint"] = str(CHECKPOINT)
    summary["validation_manifest"] = str(VAL_MANIFEST)
    summary["validation_cases"] = int(len(manifest))
    summary["num_classes"] = NUM_CLASSES
    summary["patch_size"] = list(PATCH_SIZE)
    summary["feature_size"] = FEATURE_SIZE
    summary["spider_used"] = False
    summary["training_performed"] = False
    summary["weights_modified"] = False
    summary["pseudo_masks_are_manual_ground_truth"] = False

    summary_json = OUTPUT_DIR / "phase4_part12_checkpoint_validation_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Compact class summary.
    class_rows = []
    for c in range(1, NUM_CLASSES):
        class_rows.append(
            {
                "class_id": c,
                "class_name": CLASS_NAMES[c],
                "mean_dice": summary[f"dice_class_{c}"],
                "mean_precision": summary[f"precision_class_{c}"],
                "mean_recall": summary[f"recall_class_{c}"],
            }
        )

    class_df = pd.DataFrame(class_rows)
    class_csv = OUTPUT_DIR / "part12_per_class_metrics.csv"
    class_df.to_csv(class_csv, index=False)

    # Text report.
    report_path = OUTPUT_DIR / "reports" / "phase4_part12_checkpoint_validation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "PHASE 4 - PART 12",
        "RSNA-ONLY SWIN-UNETR PILOT CHECKPOINT VALIDATION",
        "",
        f"Validation cases: {len(manifest)}",
        f"Mean loss: {summary['mean_loss']:.6f}",
        f"Median loss: {summary['median_loss']:.6f}",
        f"Mean validation Dice: {summary['mean_dice']:.6f}",
        f"Median validation Dice: {summary['median_dice']:.6f}",
        "",
        "PER-CLASS METRICS",
    ]

    for c in range(1, NUM_CLASSES):
        report_lines.append(
            f"{c}: {CLASS_NAMES[c]} | "
            f"Dice={summary[f'dice_class_{c}']:.6f} | "
            f"Precision={summary[f'precision_class_{c}']:.6f} | "
            f"Recall={summary[f'recall_class_{c}']:.6f}"
        )

    report_lines.extend(
        [
            "",
            "RESOURCE / CONTROL",
            f"Evaluation time: {elapsed / 60.0:.2f} min",
            f"Checkpoint: {CHECKPOINT}",
            "RSNA only: YES",
            "SPIDER used: NO",
            "Test set used: NO",
            "Training performed: NO",
            "Weights modified: NO",
            "",
            "SCIENTIFIC NOTE",
            "Metrics measure agreement with RSNA point-derived pseudo-masks.",
            "They are not equivalent to manually annotated segmentation ground truth.",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    banner("PART 12 FINAL SUMMARY")
    print(f"Validation cases       : {len(manifest)}")
    print(f"Mean validation loss   : {summary['mean_loss']:.6f}")
    print(f"Mean validation Dice   : {summary['mean_dice']:.6f}")
    print(f"Median validation Dice : {summary['median_dice']:.6f}")

    for c in range(1, NUM_CLASSES):
        print(
            f"{CLASS_NAMES[c]:35s}: "
            f"Dice={summary[f'dice_class_{c}']:.6f}"
        )

    if "peak_gpu_allocated_gb" in summary:
        print(f"Peak GPU allocated     : {summary['peak_gpu_allocated_gb']:.3f} GB")
        print(f"Peak GPU reserved      : {summary['peak_gpu_reserved_gb']:.3f} GB")

    print()
    print(f"Saved: {case_csv}")
    print(f"Saved: {class_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_path}")

    banner("PART 12 DECISION")
    print("PASS - Part 11 best checkpoint was successfully evaluated on the")
    print("controlled RSNA validation cohort without training or weight updates.")
    print()
    print("SPIDER used          : NO")
    print("Test set used        : NO")
    print("Training performed   : NO")
    print("Model weights changed: NO")
    print()
    print("IMPORTANT:")
    print("These are pseudo-mask validation metrics, not final clinical")
    print("segmentation performance against manual ground-truth masks.")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print()
    banner("PHASE 4 - PART 12 COMPLETE")


if __name__ == "__main__":
    main()
