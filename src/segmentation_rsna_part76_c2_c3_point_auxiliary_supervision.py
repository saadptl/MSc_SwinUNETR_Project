"""
PART 76
RSNA-ONLY C2/C3 POINT-CONSISTENT AUXILIARY SUPERVISION PILOT

Purpose
-------
Test whether explicit RSNA point-localization supervision improves the
C2/C3 failure identified in Parts 70-75.

Controlled comparison
---------------------
A) R2_FULL_CLASS_BALANCED:
   - Exact Part 15 initialization
   - R2_FULL pseudo-mask
   - centered (32,64,64) crop
   - class-balanced DiceCELoss

B) R2_FULL_CLASS_BALANCED_POINT_AUX:
   - EXACT same initialization/data/model/optimizer
   - same class-balanced DiceCELoss
   - PLUS an auxiliary point-level cross-entropy loss for C2 and C3 only

The auxiliary loss supervises the model probability at the actual RSNA
annotation point after the exact native -> FULL -> centered-crop coordinate
mapping used in Parts 67/68.1/75.2.

IMPORTANT SCIENTIFIC LIMITATION
-------------------------------
RSNA coordinates are point/localizer annotations, NOT segmentation masks.
The point loss is therefore localization/point-consistency supervision,
not manual segmentation ground truth.

Locked:
- 100 train / 50 validation
- full shape (64,96,96)
- centered R2 crop (32,64,64)
- SwinUNETR feature_size=12
- batch size 1
- lr 1e-4
- weight decay 1e-5
- 3 epochs
- seed 42
- exact Part 15 initialization
- no SPIDER
- no RSNA test set

Auxiliary loss:
- C2 and C3 points only
- point cross-entropy from softmax probabilities
- lambda = 0.20
- no synthetic segmentation mask is created from the points
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from scipy import ndimage


# ============================================================================
# PATHS
# ============================================================================
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

P11 = SRC / "segmentation_rsna_part11_controlled_pilot_training.py"
P9 = SRC / "segmentation_rsna_part9_3d_dataset_loader.py"

P15 = ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
TRAIN_CSV = P15 / "part15_train_cohort.csv"
VAL_CSV = P15 / "part15_validation_cohort.csv"
INIT_CKPT = P15 / "checkpoints" / "part15_initialization_from_part11.pth"

RSNA = ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification"
COORD = RSNA / "train_label_coordinates.csv"

OUT = ROOT / "outputs" / "segmentation" / "rsna_part76_c2_c3_point_auxiliary_supervision"
REPORT = OUT / "reports"
CKPT = OUT / "checkpoints"

REPORT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOCKED CONFIGURATION
# ============================================================================
TRAIN_N = 100
VAL_N = 50

FULL = (64, 96, 96)
CROP = (32, 64, 64)

NUM_CLASSES = 6
FEATURE_SIZE = 12
BATCH_SIZE = 1
EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42

POINT_AUX_LAMBDA = 0.20
FOCUS_CLASSES = (2, 3)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASSES = {
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}

NAME = {v: k for k, v in CLASSES.items()}
NAME.update(
    {
        "Spinal Canal Stenosis": 1,
        "Left Neural Foraminal Narrowing": 2,
        "Right Neural Foraminal Narrowing": 3,
        "Left Subarticular Stenosis": 4,
        "Right Subarticular Stenosis": 5,
    }
)

# Same Part 71 class-balanced weights.
RAW_FG_WEIGHTS = np.array(
    [0.722595, 0.935995, 1.017003, 1.178304, 1.146102],
    dtype=np.float32,
)
FG_WEIGHTS = RAW_FG_WEIGHTS / RAW_FG_WEIGHTS.mean()
CLASS_BALANCED_WEIGHTS = np.concatenate(
    [np.array([0.05], dtype=np.float32), FG_WEIGHTS]
).astype(np.float32)


# ============================================================================
# UTILITIES
# ============================================================================
def banner(text: str) -> None:
    print("\n" + "=" * 92)
    print(text)
    print("=" * 92)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def validate_paths() -> None:
    banner("PART 76 PATH VALIDATION")

    required = {
        "Project root": ROOT,
        "Part 11": P11,
        "Part 9": P9,
        "Part 15 train cohort": TRAIN_CSV,
        "Part 15 validation cohort": VAL_CSV,
        "Part 15 initialization": INIT_CKPT,
        "RSNA coordinate CSV": COORD,
    }

    missing = []
    for name, path in required.items():
        ok = path.exists()
        print(f"{name:<34}: {'FOUND' if ok else 'MISSING'}")
        if not ok:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError("\n".join(missing))


# ============================================================================
# PSEUDO-MASK MORPHOLOGY
# ============================================================================
def dilate_labels(mask: np.ndarray, radius: int) -> np.ndarray:
    """Label-preserving 3D Chebyshev dilation matching Part 71 R2."""
    mask = np.asarray(mask, dtype=np.int64)

    if radius == 0:
        return mask.copy()

    result = mask.copy()

    for c in range(1, NUM_CLASSES):
        binary = mask == c
        if not binary.any():
            continue

        structure = np.ones(
            (2 * radius + 1,) * 3,
            dtype=bool,
        )
        expanded = ndimage.binary_dilation(
            binary,
            structure=structure,
        )
        result[(expanded) & (result == 0)] = c

    return result


def centered_crop(
    mask: np.ndarray,
    center: np.ndarray,
    shape: tuple[int, int, int],
):
    starts = []

    for dim, size, c in zip(mask.shape, shape, center):
        start = int(round(float(c) - size / 2.0))
        start = max(0, min(start, dim - size))
        starts.append(start)

    z0, y0, x0 = starts
    crop = mask[
        z0:z0 + shape[0],
        y0:y0 + shape[1],
        x0:x0 + shape[2],
    ]

    return crop, tuple(starts)


# ============================================================================
# EXACT PART 67/68.1 COORDINATE MAPPING
# ============================================================================
def map_coord(
    x: float,
    y: float,
    z: float,
    native,
) -> tuple[float, float, float]:
    """
    Exact native -> FULL coordinate transform used in Part 67/68.1.
    """
    nz, nh, nw = map(float, native)

    return (
        (z + 0.5) * FULL[0] / nz - 0.5,
        (y + 0.5) * FULL[1] / nh - 0.5,
        (x + 0.5) * FULL[2] / nw - 0.5,
    )


def crop_point(
    z: float,
    y: float,
    x: float,
    starts: tuple[int, int, int],
) -> tuple[float, float, float]:
    return (
        z - starts[0],
        y - starts[1],
        x - starts[2],
    )


def rounded_point_inside(
    p: tuple[float, float, float],
    shape: tuple[int, int, int],
):
    """
    Part 67/68.1 style point inclusion:
    round transformed coordinates, then check crop bounds.
    """
    q = tuple(int(round(v)) for v in p)

    if all(0 <= q[i] < shape[i] for i in range(3)):
        return q

    return None


# ============================================================================
# CASE LOADING + ANNOTATION EXTRACTION
# ============================================================================
def load_case(
    p11,
    p9,
    row: pd.Series,
    coord: pd.DataFrame,
):
    """
    Load one case using Part 11 robust loading.

    Returns:
        image_crop
        r2_crop
        point_records
        crop_starts
    """
    image, raw_mask, _ = p11.load_case_robust(row, p9)

    image = np.asarray(image, dtype=np.float32)
    raw_mask = np.asarray(raw_mask, dtype=np.int64)

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    if raw_mask.ndim == 4 and raw_mask.shape[0] == 1:
        raw_mask = raw_mask[0]

    native = raw_mask.shape

    resized_mask = p11.resize_3d(
        raw_mask,
        FULL,
        is_mask=True,
    )
    r2 = dilate_labels(resized_mask, 2)

    q = np.argwhere(r2 > 0)

    if len(q):
        center = q.mean(axis=0)
    else:
        center = np.array([31.5, 47.5, 47.5])

    _, starts = centered_crop(
        r2,
        center,
        CROP,
    )

    z0, y0, x0 = starts

    image_resized = p11.resize_3d(
        image,
        FULL,
        is_mask=False,
    )

    image_crop = image_resized[
        z0:z0 + CROP[0],
        y0:y0 + CROP[1],
        x0:x0 + CROP[2],
    ]

    mask_crop = r2[
        z0:z0 + CROP[0],
        y0:y0 + CROP[1],
        x0:x0 + CROP[2],
    ]

    # ------------------------------------------------------------------------
    # Exact RSNA annotation extraction.
    # ------------------------------------------------------------------------
    sid = str(row["study_id"])
    ser = str(row["series_id"])

    ann = coord[
        (coord.study_id.astype(str) == sid)
        & (coord.series_id.astype(str) == ser)
    ]

    # Robust Part 11 DICOM ordering / metadata.
    _, ds, _ = p11.read_dicom_series_robust(
        p11.resolve_series_dir(row)
    )

    inst_to_z = {
        int(d.get("InstanceNumber", 0)): i
        for i, d in enumerate(ds)
    }

    points = []

    for _, a in ann.iterrows():
        condition = str(a["condition"]).strip()
        cid = NAME.get(condition)

        # Part 76 intentionally supervises only C2/C3.
        if cid not in FOCUS_CLASSES:
            continue

        try:
            x = float(a["x"])
            y = float(a["y"])
            inst = int(float(a["instance_number"]))
        except Exception:
            continue

        if inst not in inst_to_z:
            continue

        z = inst_to_z[inst]

        if not (
            0 <= x < native[2]
            and 0 <= y < native[1]
            and 0 <= z < native[0]
        ):
            continue

        zz, yy, xx = map_coord(
            x,
            y,
            z,
            native,
        )

        cz, cy, cx = crop_point(
            zz,
            yy,
            xx,
            starts,
        )

        rounded = rounded_point_inside(
            (cz, cy, cx),
            CROP,
        )

        points.append(
            {
                "class_id": int(cid),
                "class_name": CLASSES[int(cid)],
                "full_z": float(zz),
                "full_y": float(yy),
                "full_x": float(xx),
                "crop_z": float(cz),
                "crop_y": float(cy),
                "crop_x": float(cx),
                "z": int(rounded[0]) if rounded else -1,
                "y": int(rounded[1]) if rounded else -1,
                "x": int(rounded[2]) if rounded else -1,
                "inside_crop": bool(rounded is not None),
                "level": str(a.get("level", "")),
                "condition": condition,
                "instance_number": int(inst),
                "native_x": float(x),
                "native_y": float(y),
                "native_z": int(z),
            }
        )

    return (
        torch.as_tensor(image_crop, dtype=torch.float32),
        torch.as_tensor(mask_crop, dtype=torch.long),
        points,
        starts,
    )


# ============================================================================
# MODEL + LOSS
# ============================================================================
def create_model(part11):
    return part11.create_model(DEVICE).to(DEVICE)


def create_loss():
    weights = torch.tensor(
        CLASS_BALANCED_WEIGHTS,
        dtype=torch.float32,
        device=DEVICE,
    )

    return DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        weight=weights,
    ).to(DEVICE)


def load_initial_state() -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        INIT_CKPT,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    return {
        k: v.detach().cpu().clone()
        for k, v in state.items()
    }


# ============================================================================
# POINT AUXILIARY LOSS
# ============================================================================
def point_auxiliary_loss(
    logits: torch.Tensor,
    point_records: list[dict],
) -> tuple[torch.Tensor, int]:
    """
    Point-level cross-entropy for C2/C3 RSNA annotations.

    The loss samples the class logits exactly at the rounded point voxel.
    It does NOT convert the point into a segmentation mask.

    Returns:
        loss
        number_of_valid_points
    """
    valid = [
        p for p in point_records
        if p["inside_crop"] and p["class_id"] in FOCUS_CLASSES
    ]

    if not valid:
        return logits.sum() * 0.0, 0

    point_logits = []
    point_targets = []

    for p in valid:
        z = p["z"]
        y = p["y"]
        x = p["x"]
        c = p["class_id"]

        point_logits.append(
            logits[0, :, z, y, x]
        )
        point_targets.append(c)

    point_logits = torch.stack(point_logits, dim=0)
    point_targets = torch.tensor(
        point_targets,
        dtype=torch.long,
        device=logits.device,
    )

    return (
        F.cross_entropy(
            point_logits,
            point_targets,
            reduction="mean",
        ),
        len(valid),
    )


# ============================================================================
# METRICS
# ============================================================================
def class_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    c: int,
):
    p = pred == c
    t = target == c

    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())

    pred_n = int(p.sum())
    target_n = int(t.sum())

    return {
        "predicted_voxels": pred_n,
        "target_voxels": target_n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": safe_div(2 * tp, pred_n + target_n),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "iou": safe_div(tp, tp + fp + fn),
        "empty_prediction": int(pred_n == 0),
    }


def evaluate_model(
    model,
    loss_fn,
    rows,
    part9,
    part11,
    coord,
    condition: str,
):
    model.eval()

    losses = []
    class_records = []
    case_records = []
    point_records = []

    with torch.no_grad():
        for case_no, (_, row) in enumerate(
            rows.iterrows(),
            start=1,
        ):
            image, target, points, starts = load_case(
                part11,
                part9,
                row,
                coord,
            )

            x = image.unsqueeze(0).unsqueeze(0).to(
                DEVICE,
                non_blocking=True,
            )
            y = target.unsqueeze(0).unsqueeze(0).to(DEVICE)

            logits = model(x)
            loss = loss_fn(logits, y)

            pred = torch.argmax(
                logits,
                dim=1,
            )[0].detach().cpu().numpy()

            tgt = target.numpy()

            losses.append(float(loss.detach().cpu()))

            case_rec = {
                "condition": condition,
                "case_no": case_no,
                "study_id": str(row["study_id"]),
                "target_foreground_voxels": int((tgt > 0).sum()),
                "predicted_foreground_voxels": int((pred > 0).sum()),
                "empty_foreground_prediction": int(
                    (pred > 0).sum() == 0
                ),
                "inside_c2_c3_points": int(
                    sum(
                        p["inside_crop"]
                        and p["class_id"] in FOCUS_CLASSES
                        for p in points
                    )
                ),
                "total_c2_c3_points": int(
                    len(points)
                ),
            }

            for c in range(1, NUM_CLASSES):
                m = class_metrics(
                    pred,
                    tgt,
                    c,
                )

                class_records.append(
                    {
                        "condition": condition,
                        "case_no": case_no,
                        "class_id": c,
                        "class_name": CLASSES[c],
                        **m,
                    }
                )

                for key, value in m.items():
                    case_rec[f"class_{c}_{key}"] = value

            # C2/C3 localization diagnostics.
            for p in points:
                if not p["inside_crop"]:
                    continue

                c = p["class_id"]
                z, y0, x0 = p["z"], p["y"], p["x"]

                target_same = tgt == c
                pred_same = pred == c

                # Point-on-target / point-on-prediction.
                on_target = bool(
                    target_same[z, y0, x0]
                )
                on_prediction = bool(
                    pred_same[z, y0, x0]
                )

                # Direct point -> region distances.
                if target_same.any():
                    dt = ndimage.distance_transform_edt(
                        ~target_same
                    )
                    ann_to_target = float(
                        dt[z, y0, x0]
                    )
                else:
                    ann_to_target = float("nan")

                if pred_same.any():
                    dp = ndimage.distance_transform_edt(
                        ~pred_same
                    )
                    ann_to_prediction = float(
                        dp[z, y0, x0]
                    )
                else:
                    ann_to_prediction = float("nan")

                point_records.append(
                    {
                        "condition": condition,
                        "case_no": case_no,
                        "study_id": str(row["study_id"]),
                        "class_id": c,
                        "class_name": CLASSES[c],
                        "z": z,
                        "y": y0,
                        "x": x0,
                        "point_on_target": int(on_target),
                        "point_on_prediction": int(on_prediction),
                        "annotation_to_target_distance": ann_to_target,
                        "annotation_to_prediction_distance": ann_to_prediction,
                    }
                )

            case_records.append(case_rec)

            if case_no % 10 == 0:
                print(
                    f"  evaluated {case_no}/{len(rows)} cases",
                    flush=True,
                )

    cdf = pd.DataFrame(class_records)
    kdf = pd.DataFrame(case_records)
    pdf = pd.DataFrame(point_records)

    summary = {
        "condition": condition,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "foreground_target_voxels": int(
            kdf.target_foreground_voxels.sum()
        ),
        "foreground_predicted_voxels": int(
            kdf.predicted_foreground_voxels.sum()
        ),
        "foreground_empty_cases": int(
            kdf.empty_foreground_prediction.sum()
        ),
        "cases": len(kdf),
    }

    class_summary = []

    for c in range(1, NUM_CLASSES):
        g = cdf[cdf.class_id == c]

        tp = int(g.tp.sum())
        fp = int(g.fp.sum())
        fn = int(g.fn.sum())

        pred_n = int(g.predicted_voxels.sum())
        target_n = int(g.target_voxels.sum())

        class_summary.append(
            {
                "condition": condition,
                "class_id": c,
                "class_name": CLASSES[c],
                "target_voxels": target_n,
                "predicted_voxels": pred_n,
                "prediction_target_ratio": safe_div(
                    pred_n,
                    target_n,
                ),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "dice": safe_div(
                    2 * tp,
                    pred_n + target_n,
                ),
                "precision": safe_div(
                    tp,
                    tp + fp,
                ),
                "recall": safe_div(
                    tp,
                    tp + fn,
                ),
                "iou": safe_div(
                    tp,
                    tp + fp + fn,
                ),
                "empty_cases": int(
                    g.empty_prediction.sum()
                ),
            }
        )

    total_tp = sum(int(x["tp"]) for x in class_summary)
    total_pred = sum(int(x["predicted_voxels"]) for x in class_summary)
    total_target = sum(int(x["target_voxels"]) for x in class_summary)

    summary["foreground_dice"] = safe_div(
        2 * total_tp,
        total_pred + total_target,
    )

    return (
        summary,
        pd.DataFrame(class_summary),
        cdf,
        kdf,
        pdf,
    )


# ============================================================================
# TRAINING
# ============================================================================
def train_one_epoch(
    model,
    loss_fn,
    optimizer,
    rows,
    part9,
    part11,
    coord,
    use_point_aux: bool,
):
    model.train()

    losses = []
    seg_losses = []
    point_losses = []
    target_fg = []
    point_counts = []
    successful = 0

    for case_no, (_, row) in enumerate(
        rows.iterrows(),
        start=1,
    ):
        image, target, points, _ = load_case(
            part11,
            part9,
            row,
            coord,
        )

        x = image.unsqueeze(0).unsqueeze(0).to(
            DEVICE,
            non_blocking=True,
        )
        y = target.unsqueeze(0).unsqueeze(0).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            enabled=DEVICE.type == "cuda",
        ):
            logits = model(x)

            seg_loss = loss_fn(
                logits,
                y,
            )

            if use_point_aux:
                p_loss, n_points = point_auxiliary_loss(
                    logits,
                    points,
                )
                total_loss = (
                    seg_loss
                    + POINT_AUX_LAMBDA * p_loss
                )
            else:
                p_loss = logits.sum() * 0.0
                n_points = 0
                total_loss = seg_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite total loss: "
                f"{float(total_loss.detach().cpu())}"
            )

        total_loss.backward()
        optimizer.step()

        losses.append(
            float(total_loss.detach().cpu())
        )
        seg_losses.append(
            float(seg_loss.detach().cpu())
        )
        point_losses.append(
            float(p_loss.detach().cpu())
        )
        target_fg.append(
            int((target > 0).sum())
        )
        point_counts.append(
            int(n_points)
        )

        successful += 1

        if case_no % 20 == 0:
            print(
                f"  trained {case_no}/{len(rows)} cases",
                flush=True,
            )

    return {
        "loss": float(np.mean(losses)),
        "seg_loss": float(np.mean(seg_losses)),
        "point_loss": float(np.mean(point_losses)),
        "mean_target_foreground_voxels": float(
            np.mean(target_fg)
        ),
        "mean_valid_c2_c3_points": float(
            np.mean(point_counts)
        ),
        "successful_cases": successful,
    }


# ============================================================================
# CHECKPOINT
# ============================================================================
def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    condition: str,
    val_fg_dice: float,
):
    path = (
        CKPT
        / f"part76_{condition.lower()}_epoch{epoch}.pth"
    )

    torch.save(
        {
            "part": 76,
            "epoch": epoch,
            "condition": condition,
            "val_foreground_dice": float(val_fg_dice),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": SEED,
            "device": str(DEVICE),
            "point_aux_lambda": POINT_AUX_LAMBDA,
            "focus_classes": list(FOCUS_CLASSES),
            "initialization_checkpoint": str(INIT_CKPT),
            "initialization_sha256": sha256_file(
                INIT_CKPT
            ),
        },
        path,
    )

    return path


# ============================================================================
# RUN CONDITION
# ============================================================================
def run_condition(
    condition: str,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    part9,
    part11,
    coord: pd.DataFrame,
    initial_state,
):
    use_point_aux = condition == (
        "R2_FULL_CLASS_BALANCED_POINT_AUX"
    )

    banner(f"PART 76 CONDITION: {condition}")

    model = create_model(part11)
    model.load_state_dict(
        initial_state,
        strict=True,
    )

    loss_fn = create_loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Initial validation.
    t0 = time.time()

    (
        init_summary,
        init_classes,
        init_cases,
        init_case_df,
        init_points,
    ) = evaluate_model(
        model,
        loss_fn,
        val_rows,
        part9,
        part11,
        coord,
        condition,
    )

    print(
        f"INIT | val_loss={init_summary['loss']:.6f} "
        f"FGDice={init_summary['foreground_dice']:.6f} "
        f"PredFG={init_summary['foreground_predicted_voxels']} "
        f"Empty={init_summary['foreground_empty_cases']}/{len(val_rows)} "
        f"C2/C3 points={len(init_points)}"
    )

    trajectory = [
        {
            "condition": condition,
            "epoch": 0,
            "train_loss": np.nan,
            "train_seg_loss": np.nan,
            "train_point_loss": np.nan,
            "train_mean_points": np.nan,
            "train_target_fg": np.nan,
            "val_loss": init_summary["loss"],
            "val_fg_dice": init_summary["foreground_dice"],
            "val_pred_fg": init_summary[
                "foreground_predicted_voxels"
            ],
            "val_empty_fg": init_summary[
                "foreground_empty_cases"
            ],
            "val_c2_c3_points": len(init_points),
            "elapsed_sec": time.time() - t0,
        }
    ]

    all_class = []
    all_cases = []
    all_points = []

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_info = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            train_rows,
            part9,
            part11,
            coord,
            use_point_aux,
        )

        (
            val_summary,
            val_classes,
            val_case_metrics,
            val_case_df,
            val_points,
        ) = evaluate_model(
            model,
            loss_fn,
            val_rows,
            part9,
            part11,
            coord,
            condition,
        )

        elapsed = time.time() - start

        trajectory.append(
            {
                "condition": condition,
                "epoch": epoch,
                "train_loss": train_info["loss"],
                "train_seg_loss": train_info["seg_loss"],
                "train_point_loss": train_info["point_loss"],
                "train_mean_points": train_info[
                    "mean_valid_c2_c3_points"
                ],
                "train_target_fg": train_info[
                    "mean_target_foreground_voxels"
                ],
                "val_loss": val_summary["loss"],
                "val_fg_dice": val_summary[
                    "foreground_dice"
                ],
                "val_pred_fg": val_summary[
                    "foreground_predicted_voxels"
                ],
                "val_empty_fg": val_summary[
                    "foreground_empty_cases"
                ],
                "val_c2_c3_points": len(val_points),
                "elapsed_sec": elapsed,
            }
        )

        all_class.append(val_classes)

        tmp_cases = val_case_df.copy()
        tmp_cases["epoch"] = epoch
        all_cases.append(tmp_cases)

        tmp_points = val_points.copy()
        if len(tmp_points):
            tmp_points["epoch"] = epoch
        all_points.append(tmp_points)

        print(
            f"E{epoch} | "
            f"train_total={train_info['loss']:.6f} "
            f"seg={train_info['seg_loss']:.6f} "
            f"point={train_info['point_loss']:.6f} "
            f"trainPts={train_info['mean_valid_c2_c3_points']:.2f} "
            f"val_loss={val_summary['loss']:.6f} "
            f"FGDice={val_summary['foreground_dice']:.6f} "
            f"PredFG={val_summary['foreground_predicted_voxels']} "
            f"Empty={val_summary['foreground_empty_cases']}/{len(val_rows)} "
            f"time={elapsed/60:.2f} min",
            flush=True,
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            condition,
            val_summary["foreground_dice"],
        )

        reset_cuda()

    trajectory_df = pd.DataFrame(trajectory)

    class_df = (
        pd.concat(all_class, ignore_index=True)
        if all_class
        else pd.DataFrame()
    )

    case_df = (
        pd.concat(all_cases, ignore_index=True)
        if all_cases
        else pd.DataFrame()
    )

    point_df = (
        pd.concat(all_points, ignore_index=True)
        if all_points
        else pd.DataFrame()
    )

    best_idx = trajectory_df[
        "val_fg_dice"
    ].idxmax()

    best = trajectory_df.loc[
        best_idx
    ].to_dict()

    return {
        "trajectory": trajectory_df,
        "class_trajectory": class_df,
        "case_trajectory": case_df,
        "point_trajectory": point_df,
        "best": best,
    }


# ============================================================================
# FINAL LOCALIZATION SUMMARY
# ============================================================================
def localization_summary(
    best_class_df: pd.DataFrame,
    best_point_df: pd.DataFrame,
):
    rows = []

    for c in FOCUS_CLASSES:
        g = best_class_df[
            best_class_df.class_id == c
        ].copy()

        p = best_point_df[
            best_point_df.class_id == c
        ].copy()

        target = int(g.target_voxels.sum())
        pred = int(g.predicted_voxels.sum())
        tp = int(g.tp.sum())

        rows.append(
            {
                "class_id": c,
                "class_name": CLASSES[c],
                "target_voxels": target,
                "predicted_voxels": pred,
                "prediction_target_ratio": safe_div(
                    pred,
                    target,
                ),
                "dice": safe_div(
                    2 * tp,
                    pred + target,
                ),
                "precision": safe_div(
                    tp,
                    tp + int(g.fp.sum()),
                ),
                "recall": safe_div(
                    tp,
                    tp + int(g.fn.sum()),
                ),
                "point_count": len(p),
                "point_on_target_fraction": (
                    float(p.point_on_target.mean())
                    if len(p)
                    else np.nan
                ),
                "point_on_prediction_fraction": (
                    float(p.point_on_prediction.mean())
                    if len(p)
                    else np.nan
                ),
                "annotation_to_target_mean_distance": (
                    float(
                        p.annotation_to_target_distance
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                        .mean()
                    )
                    if len(p)
                    and p.annotation_to_target_distance
                    .notna()
                    .any()
                    else np.nan
                ),
                "annotation_to_prediction_mean_distance": (
                    float(
                        p.annotation_to_prediction_distance
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                        .mean()
                    )
                    if len(p)
                    and p.annotation_to_prediction_distance
                    .notna()
                    .any()
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================
def main():
    banner("PART 76")
    print("RSNA-ONLY C2/C3 POINT-CONSISTENT AUXILIARY SUPERVISION")
    print("")
    print("Purpose:")
    print(
        "Test whether explicit C2/C3 point supervision improves "
        "foreground segmentation and localization."
    )
    print("")
    print("No SPIDER.")
    print("No RSNA test set.")
    print("Both conditions start from the exact same Part 15 initialization.")
    print("RSNA points are localization annotations, not segmentation masks.")

    validate_paths()
    set_seed(SEED)

    part11 = load_module(
        P11,
        "part76_part11",
    )
    part9 = load_module(
        P9,
        "part76_part9",
    )

    coord = pd.read_csv(COORD)

    train_rows = pd.read_csv(
        TRAIN_CSV
    ).head(TRAIN_N).copy()

    val_rows = pd.read_csv(
        VAL_CSV
    ).head(VAL_N).copy()

    if len(train_rows) != TRAIN_N:
        raise RuntimeError(
            f"Unexpected train cohort size: {len(train_rows)}"
        )

    if len(val_rows) != VAL_N:
        raise RuntimeError(
            f"Unexpected validation cohort size: {len(val_rows)}"
        )

    banner("PART 76 LOCKED EXPERIMENT")

    print(f"Device                     : {DEVICE}")
    print(f"Train cases                : {len(train_rows)}")
    print(f"Validation cases           : {len(val_rows)}")
    print(f"Full shape                 : {FULL}")
    print(f"Centered R2 crop           : {CROP}")
    print(f"Epochs                     : {EPOCHS}")
    print(f"Learning rate              : {LR}")
    print(f"Weight decay               : {WEIGHT_DECAY}")
    print(f"Batch size                 : {BATCH_SIZE}")
    print(f"Feature size               : {FEATURE_SIZE}")
    print(f"Focus classes              : C2, C3")
    print(f"Point auxiliary lambda     : {POINT_AUX_LAMBDA}")
    print(
        f"Initialization SHA256      : "
        f"{sha256_file(INIT_CKPT)}"
    )

    print("")
    print("Conditions:")
    print("  A = R2_FULL_CLASS_BALANCED")
    print("  B = R2_FULL_CLASS_BALANCED_POINT_AUX")

    print("")
    print("Class-balanced weights:")
    for c, w in enumerate(
        CLASS_BALANCED_WEIGHTS
    ):
        name = (
            "Background"
            if c == 0
            else CLASSES[c]
        )
        print(
            f"  C{c} {name:<42}: {w:.6f}"
        )

    initial_state = load_initial_state()

    results = {}

    for condition in [
        "R2_FULL_CLASS_BALANCED",
        "R2_FULL_CLASS_BALANCED_POINT_AUX",
    ]:
        set_seed(SEED)

        results[condition] = run_condition(
            condition,
            train_rows,
            val_rows,
            part9,
            part11,
            coord,
            initial_state,
        )

    # ========================================================================
    # SAVE TRAJECTORIES
    # ========================================================================
    all_traj = pd.concat(
        [
            v["trajectory"]
            for v in results.values()
        ],
        ignore_index=True,
    )

    all_class = pd.concat(
        [
            v["class_trajectory"]
            for v in results.values()
        ],
        ignore_index=True,
    )

    all_cases = pd.concat(
        [
            v["case_trajectory"]
            for v in results.values()
        ],
        ignore_index=True,
    )

    all_points = pd.concat(
        [
            v["point_trajectory"]
            for v in results.values()
            if len(v["point_trajectory"])
        ],
        ignore_index=True,
    )

    all_traj.to_csv(
        REPORT / "part76_learning_trajectory.csv",
        index=False,
    )

    all_class.to_csv(
        REPORT / "part76_classwise_trajectory.csv",
        index=False,
    )

    all_cases.to_csv(
        REPORT / "part76_case_trajectory.csv",
        index=False,
    )

    all_points.to_csv(
        REPORT / "part76_point_trajectory.csv",
        index=False,
    )

    # ========================================================================
    # BEST CONDITIONS
    # ========================================================================
    base_best = results[
        "R2_FULL_CLASS_BALANCED"
    ]["best"]

    aux_best = results[
        "R2_FULL_CLASS_BALANCED_POINT_AUX"
    ]["best"]

    base_epoch = int(base_best["epoch"])
    aux_epoch = int(aux_best["epoch"])

    base_class = all_class[
        (all_class.condition == "R2_FULL_CLASS_BALANCED")
        & (all_class.epoch == base_epoch)
    ].copy()

    aux_class = all_class[
        (
            all_class.condition
            == "R2_FULL_CLASS_BALANCED_POINT_AUX"
        )
        & (all_class.epoch == aux_epoch)
    ].copy()

    base_points = all_points[
        (all_points.condition == "R2_FULL_CLASS_BALANCED")
        & (all_points.epoch == base_epoch)
    ].copy()

    aux_points = all_points[
        (
            all_points.condition
            == "R2_FULL_CLASS_BALANCED_POINT_AUX"
        )
        & (all_points.epoch == aux_epoch)
    ].copy()

    base_loc = localization_summary(
        base_class,
        base_points,
    )
    aux_loc = localization_summary(
        aux_class,
        aux_points,
    )

    base_loc["condition"] = (
        "R2_FULL_CLASS_BALANCED"
    )
    aux_loc["condition"] = (
        "R2_FULL_CLASS_BALANCED_POINT_AUX"
    )

    loc_df = pd.concat(
        [base_loc, aux_loc],
        ignore_index=True,
    )

    loc_df.to_csv(
        REPORT / "part76_best_epoch_c2_c3_localization.csv",
        index=False,
    )

    # ========================================================================
    # CONDITION COMPARISON
    # ========================================================================
    comparison = pd.DataFrame(
        [
            {
                "condition": "R2_FULL_CLASS_BALANCED",
                "best_epoch": base_epoch,
                "best_val_fg_dice": float(
                    base_best["val_fg_dice"]
                ),
                "best_val_loss": float(
                    base_best["val_loss"]
                ),
                "best_val_pred_fg": float(
                    base_best["val_pred_fg"]
                ),
                "best_val_empty_fg": int(
                    base_best["val_empty_fg"]
                ),
            },
            {
                "condition": "R2_FULL_CLASS_BALANCED_POINT_AUX",
                "best_epoch": aux_epoch,
                "best_val_fg_dice": float(
                    aux_best["val_fg_dice"]
                ),
                "best_val_loss": float(
                    aux_best["val_loss"]
                ),
                "best_val_pred_fg": float(
                    aux_best["val_pred_fg"]
                ),
                "best_val_empty_fg": int(
                    aux_best["val_empty_fg"]
                ),
            },
        ]
    )

    baseline_dice = float(
        base_best["val_fg_dice"]
    )
    aux_dice = float(
        aux_best["val_fg_dice"]
    )

    comparison["delta_vs_baseline"] = (
        comparison["best_val_fg_dice"]
        - baseline_dice
    )

    comparison.to_csv(
        REPORT / "part76_condition_comparison.csv",
        index=False,
    )

    # ========================================================================
    # C2/C3 DELTA TABLE
    # ========================================================================
    b = base_loc.set_index("class_id")
    a = aux_loc.set_index("class_id")

    delta_rows = []

    for c in FOCUS_CLASSES:
        delta_rows.append(
            {
                "class_id": c,
                "class_name": CLASSES[c],
                "baseline_dice": float(
                    b.loc[c, "dice"]
                ),
                "point_aux_dice": float(
                    a.loc[c, "dice"]
                ),
                "delta_dice": float(
                    a.loc[c, "dice"]
                    - b.loc[c, "dice"]
                ),
                "baseline_precision": float(
                    b.loc[c, "precision"]
                ),
                "point_aux_precision": float(
                    a.loc[c, "precision"]
                ),
                "delta_precision": float(
                    a.loc[c, "precision"]
                    - b.loc[c, "precision"]
                ),
                "baseline_recall": float(
                    b.loc[c, "recall"]
                ),
                "point_aux_recall": float(
                    a.loc[c, "recall"]
                ),
                "delta_recall": float(
                    a.loc[c, "recall"]
                    - b.loc[c, "recall"]
                ),
                "baseline_prediction_target_ratio": float(
                    b.loc[
                        c,
                        "prediction_target_ratio",
                    ]
                ),
                "point_aux_prediction_target_ratio": float(
                    a.loc[
                        c,
                        "prediction_target_ratio",
                    ]
                ),
                "baseline_point_on_prediction": float(
                    b.loc[
                        c,
                        "point_on_prediction_fraction",
                    ]
                ),
                "point_aux_point_on_prediction": float(
                    a.loc[
                        c,
                        "point_on_prediction_fraction",
                    ]
                ),
                "delta_point_on_prediction": float(
                    a.loc[
                        c,
                        "point_on_prediction_fraction",
                    ]
                    - b.loc[
                        c,
                        "point_on_prediction_fraction",
                    ]
                ),
                "baseline_annotation_to_prediction_distance": float(
                    b.loc[
                        c,
                        "annotation_to_prediction_mean_distance",
                    ]
                ),
                "point_aux_annotation_to_prediction_distance": float(
                    a.loc[
                        c,
                        "annotation_to_prediction_mean_distance",
                    ]
                ),
                "delta_annotation_to_prediction_distance": float(
                    a.loc[
                        c,
                        "annotation_to_prediction_mean_distance",
                    ]
                    - b.loc[
                        c,
                        "annotation_to_prediction_mean_distance",
                    ]
                ),
            }
        )

    delta_df = pd.DataFrame(delta_rows)

    delta_df.to_csv(
        REPORT / "part76_c2_c3_delta.csv",
        index=False,
    )

    # ========================================================================
    # DIAGNOSIS
    # ========================================================================
    mean_focus_delta = float(
        delta_df.delta_dice.mean()
    )

    mean_point_hit_delta = float(
        delta_df.delta_point_on_prediction.mean()
    )

    mean_distance_delta = float(
        delta_df
        .delta_annotation_to_prediction_distance
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .mean()
    )

    if (
        mean_focus_delta >= 0.02
        and mean_point_hit_delta >= 0.02
        and mean_distance_delta <= 0
    ):
        diagnosis = (
            "MEANINGFUL_POINT_AUXILIARY_C2_C3_LOCALIZATION_IMPROVEMENT"
        )
    elif mean_focus_delta >= 0.02:
        diagnosis = (
            "POINT_AUXILIARY_IMPROVES_C2_C3_DICE_BUT_LOCALIZATION_GAIN_IS_INCOMPLETE"
        )
    elif mean_point_hit_delta >= 0.02:
        diagnosis = (
            "POINT_AUXILIARY_IMPROVES_POINT_LOCALIZATION_WITHOUT_MEANINGFUL_DICE_GAIN"
        )
    elif mean_distance_delta < 0:
        diagnosis = (
            "POINT_AUXILIARY_REDUCES_ANNOTATION_TO_PREDICTION_DISTANCE_WITHOUT_MEANINGFUL_DICE_GAIN"
        )
    else:
        diagnosis = (
            "NO_MEANINGFUL_C2_C3_POINT_AUXILIARY_ADVANTAGE"
        )

    # ========================================================================
    # FINAL PRINT
    # ========================================================================
    banner("PART 76 FINAL SUMMARY")

    print(
        f"Baseline best Dice       : "
        f"{baseline_dice:.6f} "
        f"(E{base_epoch})"
    )

    print(
        f"Point-aux best Dice      : "
        f"{aux_dice:.6f} "
        f"(E{aux_epoch})"
    )

    print(
        f"Overall Dice delta       : "
        f"{aux_dice - baseline_dice:+.6f}"
    )

    print(
        f"Mean C2/C3 Dice delta    : "
        f"{mean_focus_delta:+.6f}"
    )

    print(
        f"Mean point-hit delta     : "
        f"{mean_point_hit_delta:+.6f}"
    )

    print(
        f"Mean annotation→pred Δ   : "
        f"{mean_distance_delta:+.6f}"
    )

    print("")
    print("C2/C3 best-epoch comparison:")

    for _, r in delta_df.iterrows():
        print(
            f"  C{int(r['class_id'])} "
            f"{r['class_name']:<38} "
            f"Dice {r['baseline_dice']:.6f}"
            f" -> {r['point_aux_dice']:.6f} "
            f"(Δ {r['delta_dice']:+.6f})"
        )

        print(
            f"       point-on-pred "
            f"{r['baseline_point_on_prediction']:.4f}"
            f" -> {r['point_aux_point_on_prediction']:.4f}"
        )

        print(
            f"       ann→pred distance "
            f"{r['baseline_annotation_to_prediction_distance']:.3f}"
            f" -> "
            f"{r['point_aux_annotation_to_prediction_distance']:.3f}"
        )

    print("")
    print(f"Diagnosis                 : {diagnosis}")

    # ========================================================================
    # REPORT
    # ========================================================================
    summary = {
        "part": 76,
        "purpose": (
            "Controlled C2/C3 point-consistent auxiliary "
            "supervision training pilot."
        ),
        "device": str(DEVICE),
        "train_cases": TRAIN_N,
        "validation_cases": VAL_N,
        "full_shape": list(FULL),
        "crop_shape": list(CROP),
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "focus_classes": list(FOCUS_CLASSES),
        "point_aux_lambda": POINT_AUX_LAMBDA,
        "class_balanced_weights": CLASS_BALANCED_WEIGHTS.tolist(),
        "initialization_checkpoint": str(INIT_CKPT),
        "initialization_sha256": sha256_file(INIT_CKPT),
        "baseline_best": base_best,
        "point_aux_best": aux_best,
        "overall_dice_delta": float(
            aux_dice - baseline_dice
        ),
        "mean_c2_c3_dice_delta": mean_focus_delta,
        "mean_point_hit_delta": mean_point_hit_delta,
        "mean_annotation_to_prediction_distance_delta": mean_distance_delta,
        "diagnosis": diagnosis,
        "training_completed": True,
        "spider_used": False,
        "rsna_test_set_used": False,
        "point_annotations_are_segmentation_ground_truth": False,
        "scientific_limitation": (
            "RSNA coordinates are point/localizer annotations, "
            "not manual segmentation masks. The auxiliary loss "
            "therefore provides point-consistency/localization "
            "supervision rather than clinical segmentation ground truth."
        ),
    }

    with open(
        REPORT / "part76_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    report_lines = [
        "PART 76 — C2/C3 POINT-CONSISTENT AUXILIARY SUPERVISION",
        "",
        f"Device: {DEVICE}",
        f"Train cases: {TRAIN_N}",
        f"Validation cases: {VAL_N}",
        f"Full shape: {FULL}",
        f"Crop shape: {CROP}",
        f"Epochs: {EPOCHS}",
        f"Learning rate: {LR}",
        f"Point auxiliary lambda: {POINT_AUX_LAMBDA}",
        "",
        "Baseline:",
        f"  Best epoch: {base_epoch}",
        f"  Best foreground Dice: {baseline_dice:.6f}",
        "",
        "Point auxiliary:",
        f"  Best epoch: {aux_epoch}",
        f"  Best foreground Dice: {aux_dice:.6f}",
        "",
        f"Overall Dice delta: {aux_dice - baseline_dice:+.6f}",
        f"Mean C2/C3 Dice delta: {mean_focus_delta:+.6f}",
        f"Mean point-hit delta: {mean_point_hit_delta:+.6f}",
        f"Mean annotation->prediction distance delta: {mean_distance_delta:+.6f}",
        "",
        f"Diagnosis: {diagnosis}",
        "",
        "Scientific limitation:",
        "RSNA coordinates are point/localizer annotations, not manual",
        "segmentation masks. Point supervision is therefore a",
        "localization/target-consistency intervention, not clinical",
        "segmentation ground truth.",
    ]

    (
        REPORT / "part76_report.txt"
    ).write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    banner("PART 76 COMPLETE")

    print(
        f"Learning trajectory : "
        f"{REPORT / 'part76_learning_trajectory.csv'}"
    )
    print(
        f"Class trajectory    : "
        f"{REPORT / 'part76_classwise_trajectory.csv'}"
    )
    print(
        f"Case trajectory     : "
        f"{REPORT / 'part76_case_trajectory.csv'}"
    )
    print(
        f"Point trajectory    : "
        f"{REPORT / 'part76_point_trajectory.csv'}"
    )
    print(
        f"C2/C3 localization  : "
        f"{REPORT / 'part76_best_epoch_c2_c3_localization.csv'}"
    )
    print(
        f"C2/C3 delta         : "
        f"{REPORT / 'part76_c2_c3_delta.csv'}"
    )
    print(
        f"Condition comparison : "
        f"{REPORT / 'part76_condition_comparison.csv'}"
    )
    print(
        f"Summary JSON         : "
        f"{REPORT / 'part76_summary.json'}"
    )
    print(
        f"Text report          : "
        f"{REPORT / 'part76_report.txt'}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        banner("PART 76 FAILED")
        traceback.print_exc()
        raise
