"""
Part 78 — C2/C3 Boundary-Aware Supervision
===========================================

Controlled experiment for the MSc Swin-UNETR RSNA segmentation project.

A: R2_FULL_CLASS_BALANCED
B: R2_FULL_CLASS_BALANCED_BOUNDARY

Locked:
- RSNA only
- Exact Part 15 initialization
- R2_FULL pseudo-mask
- Centered crop: (32, 64, 64)
- Full preprocessed case: (64, 96, 96)
- 100 train / 50 validation
- SwinUNETR feature_size=12
- batch_size=1
- lr=1e-4
- weight_decay=1e-5
- seed=42
- epochs=3
- boundary lambda=0.10
- C2/C3 focused boundary loss
- No SPIDER
- No test set
- Does not modify Part 15 checkpoint

The boundary term is derived from the R2 pseudo-mask itself. It is NOT
medical ground-truth segmentation and must be interpreted accordingly.

Run from:
C:\\Saad\\Msc Major Project Swin Unetr Framework\\MSc_SwinUNETR_Project

Example:
python src\\segmentation_rsna_part78_c2_c3_boundary_aware_supervision.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
SRC_DIR = PROJECT_ROOT / "src"

PART9_PATH = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"
PART11_PATH = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training.py"

BASE_DIR = PROJECT_ROOT / "outputs" / "segmentation" / \
    "rsna_part15_extended_controlled_training"

TRAIN_COHORT = BASE_DIR / "part15_train_cohort.csv"
VAL_COHORT = BASE_DIR / "part15_validation_cohort.csv"
INIT_CKPT = BASE_DIR / "checkpoints" / "part15_initialization_from_part11.pth"

OUT_DIR = PROJECT_ROOT / "outputs" / "segmentation" / \
    "rsna_part78_c2_c3_boundary_aware_supervision"
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_DIR = OUT_DIR / "reports"

# ---------------------------------------------------------------------
# Locked experiment parameters
# ---------------------------------------------------------------------

SEED = 42
EPOCHS = 3
TRAIN_N = 100
VAL_N = 50

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)

LR = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1

BOUNDARY_LAMBDA = 0.10
BOUNDARY_CLASSES = (2, 3)

NUM_CLASSES = 6

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

# Class-balanced DiceCE weights locked from Part 71.
CLASS_WEIGHTS = torch.tensor(
    [0.05, 0.722595, 0.935995, 1.017003, 1.178304, 1.146102],
    dtype=torch.float32,
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is preferred for controlled ablation.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module_from_file(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def validate_paths() -> None:
    paths = [
        ("PROJECT_ROOT", PROJECT_ROOT),
        ("PART9_PATH", PART9_PATH),
        ("PART11_PATH", PART11_PATH),
        ("TRAIN_COHORT", TRAIN_COHORT),
        ("VAL_COHORT", VAL_COHORT),
        ("INIT_CKPT", INIT_CKPT),
    ]
    missing = [f"{name}: {p}" for name, p in paths if not p.exists()]

    print("\n[PATH VALIDATION]")
    for name, p in paths:
        print(f"  {name}: {'FOUND' if p.exists() else 'MISSING'} -> {p}")

    if missing:
        raise FileNotFoundError("\n".join(missing))


# ---------------------------------------------------------------------
# Tensor / crop helpers
# ---------------------------------------------------------------------

def as_dhw(x: torch.Tensor) -> torch.Tensor:
    """Normalize image/mask to [D,H,W]."""
    x = x.detach().cpu()

    while x.ndim > 3:
        # Expected image shape is [1,D,H,W] or similar.
        singleton = None
        for i, s in enumerate(x.shape):
            if s == 1:
                singleton = i
                break
        if singleton is None:
            raise ValueError(f"Cannot reduce tensor shape {tuple(x.shape)} to DHW")
        x = x.squeeze(singleton)

    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}")

    return x


def center_crop_3d(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_shape: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    d, h, w = image.shape
    cd, ch, cw = crop_shape

    if d < cd or h < ch or w < cw:
        raise ValueError(
            f"Image {tuple(image.shape)} smaller than crop {crop_shape}"
        )

    z0 = (d - cd) // 2
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2

    return (
        image[z0:z0 + cd, y0:y0 + ch, x0:x0 + cw],
        mask[z0:z0 + cd, y0:y0 + ch, x0:x0 + cw],
    )


def prepare_case(
    row: pd.Series,
    part9,
    part11,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loaded = part11.load_tensor_case(row, part9)

    if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
        raise ValueError("Unexpected load_tensor_case return value")

    image = as_dhw(loaded[0]).float()
    mask = as_dhw(loaded[1]).long()

    # The Part 9/11 loader already performs the established preprocessing.
    # Do NOT call part11.preprocess_case() again.
    image, mask = center_crop_3d(image, mask, CROP_SHAPE)

    if tuple(image.shape) != CROP_SHAPE:
        raise ValueError(f"Unexpected image crop shape: {tuple(image.shape)}")
    if tuple(mask.shape) != CROP_SHAPE:
        raise ValueError(f"Unexpected mask crop shape: {tuple(mask.shape)}")

    return image, mask


# ---------------------------------------------------------------------
# Boundary construction
# ---------------------------------------------------------------------

def binary_boundary(mask: torch.Tensor, class_id: int) -> torch.Tensor:
    """
    Create a 1-voxel boundary band for one class.

    mask: [D,H,W], integer labels.
    Returns float tensor [D,H,W] in {0,1}.

    Boundary is derived from the pseudo-mask using a 6-connected
    neighborhood. It is intentionally simple and deterministic.
    """
    fg = (mask == class_id).float()[None, None]  # 1,1,D,H,W

    # Erosion: voxel remains interior only when all 6 axial neighbors
    # are also foreground.
    center = fg
    xp = F.pad(fg[:, :, 1:, :, :], (0, 0, 0, 0, 0, 1))
    xm = F.pad(fg[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0))
    yp = F.pad(fg[:, :, :, 1:, :], (0, 0, 0, 1, 0, 0))
    ym = F.pad(fg[:, :, :, :-1, :], (0, 1, 0, 0, 0, 0))
    zp = F.pad(fg[:, :, :, :, 1:], (0, 1, 0, 0, 0, 0))
    zm = F.pad(fg[:, :, :, :, :-1], (1, 0, 0, 0, 0, 0))

    interior = center * xp * xm * yp * ym * zp * zm
    boundary = (center - interior).clamp_min(0.0)

    return boundary[0, 0]


def make_boundary_target(mask: torch.Tensor) -> torch.Tensor:
    """
    Returns [2,D,H,W] for C2 and C3 boundary targets.
    """
    targets = [
        binary_boundary(mask, 2),
        binary_boundary(mask, 3),
    ]
    return torch.stack(targets, dim=0)


def boundary_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Boundary BCE-with-logits for C2/C3.

    logits: [B,6,D,H,W]
    mask:   [B,D,H,W]

    To avoid class-imbalance domination, positive pixels receive a
    balanced BCE weight independently for each class.
    """
    losses = []

    for class_id, channel in zip(BOUNDARY_CLASSES, (2, 3)):
        logit = logits[:, channel:channel + 1]
        target_list = []

        for b in range(mask.shape[0]):
            target_list.append(binary_boundary(mask[b], class_id))

        target = torch.stack(target_list, dim=0)[:, None].to(
            device=logit.device,
            dtype=logit.dtype,
        )

        pos = target.sum()
        total = target.numel()
        neg = total - pos

        if pos.item() <= 0:
            # No boundary for this class in the batch.
            continue

        # Balanced BCE: positive/negative contributions are comparable.
        pos_weight = (neg / pos.clamp_min(1.0)).clamp(max=100.0)

        losses.append(
            F.binary_cross_entropy_with_logits(
                logit,
                target,
                pos_weight=pos_weight,
            )
        )

    if not losses:
        return logits.sum() * 0.0

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------
# Model / loss
# ---------------------------------------------------------------------

def create_model(part11, device):
    model = part11.create_model(device)
    return model


def create_seg_loss(device):
    weights = CLASS_WEIGHTS.to(device)

    # MONAI DiceCELoss accepts class weights through the CE component.
    # sigmoid=False / softmax=True corresponds to six mutually exclusive
    # segmentation classes.
    return DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        sigmoid=False,
        squared_pred=False,
        reduction="mean",
        ce_weight=weights,
    )


def dice_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[float, Dict[int, Dict[str, float]]]:
    """
    Conventional foreground Dice from argmax prediction.

    Returns overall foreground Dice and per-class precision/recall/Dice.
    """
    pred = torch.argmax(logits, dim=1)
    target = target.long()

    class_rows: Dict[int, Dict[str, float]] = {}

    foreground_dice = []

    for c in range(1, NUM_CLASSES):
        p = pred == c
        t = target == c

        inter = (p & t).sum().item()
        p_count = p.sum().item()
        t_count = t.sum().item()

        dice = (2.0 * inter) / (p_count + t_count + 1e-8)
        precision = inter / (p_count + 1e-8)
        recall = inter / (t_count + 1e-8)

        class_rows[c] = {
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "pred_voxels": float(p_count),
            "target_voxels": float(t_count),
            "pred_target_ratio": float(p_count / (t_count + 1e-8)),
        }
        foreground_dice.append(dice)

    overall = float(np.mean(foreground_dice))
    return overall, class_rows


def c2_c3_centroid_distance(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Dict[int, float]:
    """
    Centroid distance in crop voxel coordinates for C2/C3.

    Finite only when both target and prediction contain the class.
    """
    pred = torch.argmax(logits, dim=1)
    out = {}

    for c in BOUNDARY_CLASSES:
        distances = []

        for b in range(target.shape[0]):
            t_idx = torch.nonzero(target[b] == c, as_tuple=False)
            p_idx = torch.nonzero(pred[b] == c, as_tuple=False)

            if t_idx.numel() == 0 or p_idx.numel() == 0:
                continue

            tc = t_idx.float().mean(dim=0)
            pc = p_idx.float().mean(dim=0)
            distances.append(torch.linalg.vector_norm(tc - pc).item())

        out[c] = float(np.mean(distances)) if distances else float("nan")

    return out


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------

def select_rows(path: Path, n: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    if len(df) < n:
        raise ValueError(f"{path} contains only {len(df)} rows; need {n}")

    # Preserve the established cohort order and take the first N.
    return df.iloc[:n].reset_index(drop=True)


def load_cases(
    df: pd.DataFrame,
    part9,
    part11,
    split_name: str,
) -> List[Tuple[torch.Tensor, torch.Tensor, str]]:
    cases = []
    failures = []

    print(f"\n[LOADING {split_name.upper()} CASES] requested={len(df)}")

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            image, mask = prepare_case(row, part9, part11)

            case_id = str(
                row.get("study_id",
                       row.get("series_id",
                              row.get("id", i)))
            )

            cases.append((image, mask, case_id))

        except Exception as exc:
            failures.append((i, str(exc)))

        if (i + 1) % 20 == 0 or i + 1 == len(df):
            print(
                f"  processed={i + 1}/{len(df)} "
                f"success={len(cases)} failed={len(failures)}"
            )

    if failures:
        print("\n[LOAD FAILURES]")
        for item in failures[:10]:
            print(f"  row={item[0]} error={item[1]}")

    if len(cases) < len(df):
        raise RuntimeError(
            f"{split_name}: {len(failures)} case(s) failed. "
            "Controlled Part 78 requires the requested cohort to load."
        )

    return cases


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    cases,
    seg_loss,
    device,
    include_boundary: bool = False,
):
    model.eval()

    losses = []
    fg_dices = []
    pred_fg_counts = []
    empty_count = 0

    class_accum = {
        c: {
            "dice": [],
            "precision": [],
            "recall": [],
            "pred_voxels": [],
            "target_voxels": [],
            "pred_target_ratio": [],
        }
        for c in range(1, NUM_CLASSES)
    }

    c2c3_dist = {2: [], 3: []}

    for image, mask, _ in cases:
        x = image[None, None].to(device, non_blocking=True)
        y = mask[None].to(device, non_blocking=True)

        logits = model(x)

        loss = seg_loss(logits, y[:, None])
        if include_boundary:
            loss = loss + BOUNDARY_LAMBDA * boundary_loss(logits, y)

        losses.append(loss.item())

        fg_dice, class_rows = dice_metrics(logits, y)
        fg_dices.append(fg_dice)

        pred = torch.argmax(logits, dim=1)
        pred_fg = (pred > 0).sum().item()
        pred_fg_counts.append(pred_fg)

        if pred_fg == 0:
            empty_count += 1

        for c, row in class_rows.items():
            for k, v in row.items():
                class_accum[c][k].append(v)

        distances = c2_c3_centroid_distance(logits, y)
        for c in BOUNDARY_CLASSES:
            if not math.isnan(distances[c]):
                c2c3_dist[c].append(distances[c])

    class_mean = {}
    for c in range(1, NUM_CLASSES):
        class_mean[c] = {
            k: float(np.mean(v)) if v else float("nan")
            for k, v in class_accum[c].items()
        }

    dist_mean = {
        c: float(np.mean(v)) if v else float("nan")
        for c, v in c2c3_dist.items()
    }

    return {
        "loss": float(np.mean(losses)),
        "fg_dice": float(np.mean(fg_dices)),
        "pred_fg_mean": float(np.mean(pred_fg_counts)),
        "empty_cases": int(empty_count),
        "classwise": class_mean,
        "c2_c3_centroid_distance": dist_mean,
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_condition(
    condition_name: str,
    model,
    train_cases,
    val_cases,
    seg_loss,
    device,
    include_boundary: bool,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Save exact initialization state before any optimizer step.
    init_model_path = CKPT_DIR / f"part78_{condition_name.lower()}_init.pth"
    torch.save(model.state_dict(), init_model_path)

    init_eval = evaluate(
        model,
        val_cases,
        seg_loss,
        device,
        include_boundary=include_boundary,
    )

    trajectory = [{
        "condition": condition_name,
        "epoch": 0,
        "train_total_loss": float("nan"),
        "train_seg_loss": float("nan"),
        "train_boundary_loss": float("nan"),
        "val_loss": init_eval["loss"],
        "fg_dice": init_eval["fg_dice"],
        "pred_fg_mean": init_eval["pred_fg_mean"],
        "empty_cases": init_eval["empty_cases"],
        "c2_dice": init_eval["classwise"][2]["dice"],
        "c3_dice": init_eval["classwise"][3]["dice"],
        "c2_precision": init_eval["classwise"][2]["precision"],
        "c2_recall": init_eval["classwise"][2]["recall"],
        "c3_precision": init_eval["classwise"][3]["precision"],
        "c3_recall": init_eval["classwise"][3]["recall"],
        "c2_centroid_distance": init_eval["c2_c3_centroid_distance"][2],
        "c3_centroid_distance": init_eval["c2_c3_centroid_distance"][3],
    }]

    print(
        f"\n[{condition_name}] INIT "
        f"val_loss={init_eval['loss']:.6f} "
        f"FGDice={init_eval['fg_dice']:.6f} "
        f"PredFG={init_eval['pred_fg_mean']:.1f} "
        f"Empty={init_eval['empty_cases']}/{len(val_cases)}"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_losses = []
        seg_losses = []
        boundary_losses = []

        t0 = time.time()

        for image, mask, _ in train_cases:
            x = image[None, None].to(device)
            y = mask[None].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            seg = seg_loss(logits, y[:, None])

            if include_boundary:
                bnd = boundary_loss(logits, y)
            else:
                bnd = logits.sum() * 0.0

            total = seg + BOUNDARY_LAMBDA * bnd

            total.backward()
            optimizer.step()

            total_losses.append(total.item())
            seg_losses.append(seg.item())
            boundary_losses.append(bnd.item())

        val_eval = evaluate(
            model,
            val_cases,
            seg_loss,
            device,
            include_boundary=include_boundary,
        )

        elapsed = time.time() - t0

        row = {
            "condition": condition_name,
            "epoch": epoch,
            "train_total_loss": float(np.mean(total_losses)),
            "train_seg_loss": float(np.mean(seg_losses)),
            "train_boundary_loss": float(np.mean(boundary_losses)),
            "val_loss": val_eval["loss"],
            "fg_dice": val_eval["fg_dice"],
            "pred_fg_mean": val_eval["pred_fg_mean"],
            "empty_cases": val_eval["empty_cases"],
            "c2_dice": val_eval["classwise"][2]["dice"],
            "c3_dice": val_eval["classwise"][3]["dice"],
            "c2_precision": val_eval["classwise"][2]["precision"],
            "c2_recall": val_eval["classwise"][2]["recall"],
            "c3_precision": val_eval["classwise"][3]["precision"],
            "c3_recall": val_eval["classwise"][3]["recall"],
            "c2_centroid_distance": val_eval["c2_c3_centroid_distance"][2],
            "c3_centroid_distance": val_eval["c2_c3_centroid_distance"][3],
            "elapsed_sec": elapsed,
        }
        trajectory.append(row)

        ckpt_path = CKPT_DIR / (
            f"part78_{condition_name.lower()}_epoch{epoch}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)

        print(
            f"[{condition_name}] E{epoch} "
            f"train_total={row['train_total_loss']:.6f} "
            f"seg={row['train_seg_loss']:.6f} "
            f"boundary={row['train_boundary_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} "
            f"FGDice={row['fg_dice']:.6f} "
            f"PredFG={row['pred_fg_mean']:.1f} "
            f"Empty={row['empty_cases']}/{len(val_cases)} "
            f"C2dist={row['c2_centroid_distance']:.5f} "
            f"C3dist={row['c3_centroid_distance']:.5f} "
            f"time={elapsed:.1f}s"
        )

    return trajectory


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_condition_summary(rows: List[Dict]) -> List[Dict]:
    by_condition = {}

    for r in rows:
        by_condition.setdefault(r["condition"], []).append(r)

    out = []

    for condition, rs in by_condition.items():
        epochs = [r for r in rs if int(r["epoch"]) > 0]

        best = max(
            epochs,
            key=lambda r: (
                -1e9 if math.isnan(float(r["fg_dice"]))
                else float(r["fg_dice"])
            ),
        )

        final = epochs[-1]

        out.append({
            "condition": condition,
            "best_epoch": int(best["epoch"]),
            "best_fg_dice": float(best["fg_dice"]),
            "final_fg_dice": float(final["fg_dice"]),
            "final_c2_dice": float(final["c2_dice"]),
            "final_c3_dice": float(final["c3_dice"]),
            "final_c2_precision": float(final["c2_precision"]),
            "final_c2_recall": float(final["c2_recall"]),
            "final_c3_precision": float(final["c3_precision"]),
            "final_c3_recall": float(final["c3_recall"]),
            "final_c2_centroid_distance": float(final["c2_centroid_distance"]),
            "final_c3_centroid_distance": float(final["c3_centroid_distance"]),
            "final_pred_fg_mean": float(final["pred_fg_mean"]),
            "final_empty_cases": int(final["empty_cases"]),
        })

    return out


def safe_delta(a, b):
    try:
        if math.isnan(float(a)) or math.isnan(float(b)):
            return float("nan")
        return float(b - a)
    except Exception:
        return float("nan")


def write_report(
    trajectory: List[Dict],
    summary: List[Dict],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    init_hash: str,
    device: str,
) -> Dict:

    base = next(x for x in summary if x["condition"] == "BASELINE")
    boundary = next(x for x in summary if x["condition"] == "BOUNDARY")

    delta = {
        "best_fg_dice_delta_boundary_minus_baseline":
            safe_delta(base["best_fg_dice"], boundary["best_fg_dice"]),
        "final_fg_dice_delta_boundary_minus_baseline":
            safe_delta(base["final_fg_dice"], boundary["final_fg_dice"]),
        "final_c2_dice_delta":
            safe_delta(base["final_c2_dice"], boundary["final_c2_dice"]),
        "final_c3_dice_delta":
            safe_delta(base["final_c3_dice"], boundary["final_c3_dice"]),
        "final_c2_centroid_distance_delta":
            safe_delta(
                base["final_c2_centroid_distance"],
                boundary["final_c2_centroid_distance"],
            ),
        "final_c3_centroid_distance_delta":
            safe_delta(
                base["final_c3_centroid_distance"],
                boundary["final_c3_centroid_distance"],
            ),
    }

    # Conservative interpretation:
    # meaningful only if there is a clear overall Dice gain or clear
    # C2/C3 improvement rather than tiny numerical fluctuations.
    overall_gain = delta["best_fg_dice_delta_boundary_minus_baseline"]
    c2_gain = delta["final_c2_dice_delta"]
    c3_gain = delta["final_c3_dice_delta"]

    if (
        overall_gain > 0.01
        or c2_gain > 0.01
        or c3_gain > 0.01
    ):
        diagnosis = "BOUNDARY_AWARE_SUPERVISION_SHOWED_MEANINGFUL_SIGNAL"
    else:
        diagnosis = "NO_MEANINGFUL_C2_C3_BOUNDARY_SUPERVISION_ADVANTAGE"

    report = {
        "part": 78,
        "title": "C2/C3 Boundary-Aware Supervision",
        "diagnosis": diagnosis,
        "device": device,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_n": TRAIN_N,
        "val_n": VAL_N,
        "full_shape": FULL_SHAPE,
        "crop_shape": CROP_SHAPE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "boundary_lambda": BOUNDARY_LAMBDA,
        "boundary_classes": list(BOUNDARY_CLASSES),
        "class_weights": CLASS_WEIGHTS.tolist(),
        "part15_initialization_sha256": init_hash,
        "conditions": summary,
        "deltas": delta,
    }

    with (REPORT_DIR / "part78_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    write_csv(
        REPORT_DIR / "part78_learning_trajectory.csv",
        trajectory,
    )
    write_csv(
        REPORT_DIR / "part78_condition_comparison.csv",
        summary,
    )
    write_csv(
        REPORT_DIR / "part78_c2_c3_delta.csv",
        [delta],
    )

    txt = []
    txt.append("PART 78 — C2/C3 BOUNDARY-AWARE SUPERVISION")
    txt.append("=" * 60)
    txt.append("")
    txt.append(f"Diagnosis: {diagnosis}")
    txt.append(f"Device: {device}")
    txt.append(f"Part 15 init SHA256: {init_hash}")
    txt.append("")
    txt.append("LOCKED SETUP")
    txt.append(f"Train/Val: {TRAIN_N}/{VAL_N}")
    txt.append(f"Full shape: {FULL_SHAPE}")
    txt.append(f"Crop shape: {CROP_SHAPE}")
    txt.append(f"LR: {LR}")
    txt.append(f"Weight decay: {WEIGHT_DECAY}")
    txt.append(f"Boundary lambda: {BOUNDARY_LAMBDA}")
    txt.append(f"Boundary classes: {BOUNDARY_CLASSES}")
    txt.append("")
    txt.append("CONDITION RESULTS")
    txt.append("-" * 60)

    for s in summary:
        txt.append(
            f"{s['condition']}: "
            f"best={s['best_fg_dice']:.6f} "
            f"(E{s['best_epoch']}), "
            f"final={s['final_fg_dice']:.6f}, "
            f"C2={s['final_c2_dice']:.6f}, "
            f"C3={s['final_c3_dice']:.6f}, "
            f"C2dist={s['final_c2_centroid_distance']:.6f}, "
            f"C3dist={s['final_c3_centroid_distance']:.6f}"
        )

    txt.append("")
    txt.append("DELTA: BOUNDARY - BASELINE")
    txt.append("-" * 60)
    for k, v in delta.items():
        txt.append(f"{k}: {v}")

    txt.append("")
    txt.append(
        "Interpretation note: boundary targets are derived from R2 pseudo-masks, "
        "not manual clinical segmentation masks."
    )
    txt.append(
        "Do not treat this experiment as evidence that the RSNA point annotations "
        "constitute pixel/voxel ground truth."
    )

    (REPORT_DIR / "part78_report.txt").write_text(
        "\n".join(txt),
        encoding="utf-8",
    )

    return report


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 78)
    print("PART 78 — C2/C3 BOUNDARY-AWARE SUPERVISION")
    print("=" * 78)

    ensure_dirs()
    validate_paths()
    seed_everything(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    init_hash = sha256_file(INIT_CKPT)
    print(f"Part 15 initialization SHA256: {init_hash}")

    expected_hash = (
        "0900c0e6490fdaddf455763b465d4a607acfb310349c9f9ec82a61117016aec6"
    )

    if init_hash != expected_hash:
        raise RuntimeError(
            "Part 15 initialization SHA256 mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {init_hash}"
        )

    # Import the established Part 9/11 pipeline.
    part9 = load_module_from_file("part9_part78", PART9_PATH)
    part11 = load_module_from_file("part11_part78", PART11_PATH)

    train_df = select_rows(TRAIN_COHORT, TRAIN_N)
    val_df = select_rows(VAL_COHORT, VAL_N)

    print(f"\nTrain cohort: {len(train_df)}")
    print(f"Val cohort:   {len(val_df)}")

    train_cases = load_cases(train_df, part9, part11, "train")
    val_cases = load_cases(val_df, part9, part11, "validation")

    # Save a compact cohort copy for reproducibility.
    train_df.to_csv(
        OUT_DIR / "part78_train_cohort.csv",
        index=False,
    )
    val_df.to_csv(
        OUT_DIR / "part78_validation_cohort.csv",
        index=False,
    )

    seg_loss = create_seg_loss(device)

    # -------------------------------------------------------------
    # Condition A — exact Part 71 class-balanced baseline
    # -------------------------------------------------------------
    seed_everything(SEED)

    baseline_model = create_model(part11, device)
    baseline_model.load_state_dict(
        torch.load(
            INIT_CKPT,
            map_location=device,
            weights_only=True,
        )
    )
    baseline_model.to(device)

    baseline_rows = train_one_condition(
        "BASELINE",
        baseline_model,
        train_cases,
        val_cases,
        seg_loss,
        device,
        include_boundary=False,
    )

    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Condition B — class-balanced + C2/C3 boundary supervision
    # -------------------------------------------------------------
    seed_everything(SEED)

    boundary_model = create_model(part11, device)
    boundary_model.load_state_dict(
        torch.load(
            INIT_CKPT,
            map_location=device,
            weights_only=True,
        )
    )
    boundary_model.to(device)

    boundary_rows = train_one_condition(
        "BOUNDARY",
        boundary_model,
        train_cases,
        val_cases,
        seg_loss,
        device,
        include_boundary=True,
    )

    del boundary_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trajectory = baseline_rows + boundary_rows
    summary = make_condition_summary(trajectory)

    report = write_report(
        trajectory=trajectory,
        summary=summary,
        train_df=train_df,
        val_df=val_df,
        init_hash=init_hash,
        device=str(device),
    )

    print("\n" + "=" * 78)
    print("PART 78 COMPLETE")
    print("=" * 78)
    print(f"Diagnosis: {report['diagnosis']}")
    print("")
    for s in summary:
        print(
            f"{s['condition']}: "
            f"best FGDice={s['best_fg_dice']:.6f} "
            f"(E{s['best_epoch']}), "
            f"final C2={s['final_c2_dice']:.6f}, "
            f"final C3={s['final_c3_dice']:.6f}"
        )

    print("\nReports:")
    for p in sorted(REPORT_DIR.glob("part78_*")):
        print(f"  {p}")

    print("\nCheckpoints:")
    for p in sorted(CKPT_DIR.glob("part78_*")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
