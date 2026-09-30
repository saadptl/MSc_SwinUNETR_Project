"""
PART 71
RSNA-ONLY CLASS-BALANCED DICECE TRAINING ABLATION

Purpose
-------
Test whether explicit class-balanced DiceCELoss improves the severe
class-selective learning problem identified in Part 70.

Controlled comparison
---------------------
A) R2_FULL baseline:
   - Exact Part 15 initialization
   - Original Part 11 DiceCELoss configuration
B) R2_FULL class-balanced:
   - Exact same initialization
   - Same data/crop/model/optimizer/epochs
   - DiceCELoss with fixed class weights derived from the previously
     established empirical class-frequency weights, normalized to mean 1
     over the five foreground classes and with a reduced background weight.

This is a TRAINING ABLATION.
No SPIDER and no RSNA test set are used.

Locked:
- 100 train / 50 validation
- centered R2 crop (32,64,64)
- full preprocessing shape (64,96,96)
- SwinUNETR feature_size=12
- batch size 1
- lr 1e-4
- weight decay 1e-5
- 3 epochs
- seed 42
- exact Part 15 initialization
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

OUT = ROOT / "outputs" / "segmentation" / "rsna_part71_class_balanced_dicece_training"
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
IN_CHANNELS = 1
FEATURE_SIZE = 12

BATCH_SIZE = 1
EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASSES = {
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}

# Part 49 empirical foreground-frequency weights.
# We retain their relative class ordering, then normalize foreground
# weights to mean 1. Background is deliberately downweighted because
# Part 48 established extreme voxel-level background dominance.
RAW_FG_WEIGHTS = np.array(
    [0.722595, 0.935995, 1.017003, 1.178304, 1.146102],
    dtype=np.float32,
)

FG_WEIGHTS = RAW_FG_WEIGHTS / RAW_FG_WEIGHTS.mean()

# Conservative background weight; this is the intervention component.
CLASS_BALANCED_WEIGHTS = np.concatenate(
    [np.array([0.05], dtype=np.float32), FG_WEIGHTS]
).astype(np.float32)


# ============================================================================
# UTILITIES
# ============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


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


def validate_paths() -> None:
    banner("PART 71 PATH VALIDATION")

    required = {
        "Project root": ROOT,
        "Part 11": P11,
        "Part 9": P9,
        "Part 15 train cohort": TRAIN_CSV,
        "Part 15 validation cohort": VAL_CSV,
        "Part 15 initialization": INIT_CKPT,
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
    """
    Label-preserving 3D dilation using Chebyshev neighborhoods, matching
    the local Part 52/68.1 morphology convention.
    """
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


# ============================================================================
# CENTERED CROP
# ============================================================================

def centered_crop(mask: np.ndarray, center: np.ndarray, shape: tuple[int, int, int]):
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


def make_r2_crop(part11: Any, image: torch.Tensor, raw_mask: torch.Tensor):
    image_np = image.detach().cpu().numpy()
    mask_np = raw_mask.detach().cpu().numpy().astype(np.int64)

    if image_np.ndim == 4 and image_np.shape[0] == 1:
        image_np = image_np[0]
    if mask_np.ndim == 4 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]

    resized_mask = part11.resize_3d(
        mask_np,
        FULL,
        is_mask=True,
    )
    r2 = dilate_labels(resized_mask, 2)

    q = np.argwhere(r2 > 0)
    if len(q):
        center = q.mean(axis=0)
    else:
        center = np.array([31.5, 47.5, 47.5])

    _, starts = centered_crop(r2, center, CROP)

    z0, y0, x0 = starts

    image_resized = part11.resize_3d(
        image_np,
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

    return (
        torch.as_tensor(image_crop, dtype=torch.float32),
        torch.as_tensor(mask_crop, dtype=torch.long),
    )


def load_case(row: pd.Series, part9: Any, part11: Any):
    loaded = part11.load_tensor_case(row, part9)

    if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
        raise RuntimeError("Unexpected Part 11 load_tensor_case() result.")

    image = torch.as_tensor(loaded[0])
    mask = torch.as_tensor(loaded[1])

    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    image, mask = make_r2_crop(part11, image, mask)

    return image, mask


# ============================================================================
# MODEL / LOSS
# ============================================================================

def create_model(part11: Any):
    model = part11.create_model(DEVICE).to(DEVICE)
    return model


def create_loss(condition: str):
    if condition == "R2_FULL_BASELINE":
        return DiceCELoss(
            to_onehot_y=True,
            softmax=True,
        ).to(DEVICE)

    if condition == "R2_FULL_CLASS_BALANCED":
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

    raise ValueError(condition)


def load_initial_state() -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        INIT_CKPT,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    cleaned = {}
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        cleaned[key] = value.detach().cpu().clone()

    # Strict compatibility check.
    model = create_model(P11_MODULE)
    model.load_state_dict(cleaned, strict=True)
    del model
    reset_cuda()

    return cleaned


# ============================================================================
# METRICS
# ============================================================================

def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def class_metrics(pred: np.ndarray, target: np.ndarray, c: int) -> dict:
    p = pred == c
    t = target == c

    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())

    pred_n = int(p.sum())
    target_n = int(t.sum())

    dice = safe_div(2 * tp, pred_n + target_n)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    iou = safe_div(tp, tp + fp + fn)

    return {
        "target_voxels": target_n,
        "predicted_voxels": pred_n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "empty_prediction": int(pred_n == 0),
    }


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    rows: pd.DataFrame,
    part9: Any,
    part11: Any,
):
    model.eval()

    losses = []
    class_records = []
    case_records = []

    with torch.no_grad():
        for case_no, (_, row) in enumerate(rows.iterrows(), start=1):
            image, target = load_case(row, part9, part11)

            x = image.unsqueeze(0).unsqueeze(0).to(
                DEVICE,
                non_blocking=True,
            )
            y = target.unsqueeze(0).unsqueeze(0).to(DEVICE)

            logits = model(x)
            loss = loss_fn(logits, y)

            pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy()
            tgt = target.numpy()

            losses.append(float(loss.detach().cpu()))

            case_fg_target = int((tgt > 0).sum())
            case_fg_pred = int((pred > 0).sum())

            case_rec = {
                "case_no": case_no,
                "target_foreground_voxels": case_fg_target,
                "predicted_foreground_voxels": case_fg_pred,
                "empty_foreground_prediction": int(case_fg_pred == 0),
            }

            for c in range(1, NUM_CLASSES):
                m = class_metrics(pred, tgt, c)
                rec = {
                    "case_no": case_no,
                    "class_id": c,
                    "class_name": CLASSES[c],
                    **m,
                }
                class_records.append(rec)

                for key, value in m.items():
                    case_rec[f"class_{c}_{key}"] = value

            case_records.append(case_rec)

    cdf = pd.DataFrame(class_records)
    kdf = pd.DataFrame(case_records)

    summary = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "foreground_target_voxels": int(kdf.target_foreground_voxels.sum()),
        "foreground_predicted_voxels": int(kdf.predicted_foreground_voxels.sum()),
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
                "class_id": c,
                "class_name": CLASSES[c],
                "target_voxels": target_n,
                "predicted_voxels": pred_n,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "dice": safe_div(2 * tp, pred_n + target_n),
                "precision": safe_div(tp, tp + fp),
                "recall": safe_div(tp, tp + fn),
                "iou": safe_div(tp, tp + fp + fn),
                "empty_cases": int(g.empty_prediction.sum()),
            }
        )

    # Micro foreground Dice.
    total_tp = sum(int(x["tp"]) for x in class_summary)
    total_pred = sum(int(x["predicted_voxels"]) for x in class_summary)
    total_target = sum(int(x["target_voxels"]) for x in class_summary)

    summary["foreground_dice"] = safe_div(
        2 * total_tp,
        total_pred + total_target,
    )

    return summary, pd.DataFrame(class_summary), cdf, kdf


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: pd.DataFrame,
    part9: Any,
    part11: Any,
):
    model.train()

    losses = []
    target_fg = []
    successful = 0

    for _, row in rows.iterrows():
        image, target = load_case(row, part9, part11)

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
            loss = loss_fn(logits, y)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss: {float(loss.detach().cpu())}"
            )

        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        target_fg.append(int((target > 0).sum()))
        successful += 1

    return {
        "loss": float(np.mean(losses)),
        "mean_target_foreground_voxels": float(np.mean(target_fg)),
        "successful_cases": successful,
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    condition: str,
    val_fg_dice: float,
):
    path = CKPT / f"part71_{condition.lower()}_epoch{epoch}.pth"

    torch.save(
        {
            "epoch": epoch,
            "condition": condition,
            "val_foreground_dice": val_fg_dice,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": SEED,
            "device": str(DEVICE),
        },
        path,
    )

    return path


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_condition(
    condition: str,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    part9: Any,
    part11: Any,
    initial_state: dict[str, torch.Tensor],
):
    banner(f"PART 71 CONDITION: {condition}")

    model = create_model(part11)
    model.load_state_dict(initial_state, strict=True)

    loss_fn = create_loss(condition)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Initial validation before any update.
    t0 = time.time()

    init_summary, init_classes, init_case_df, _ = evaluate(
        model,
        loss_fn,
        val_rows,
        part9,
        part11,
    )

    print(
        f"INIT | val_loss={init_summary['loss']:.6f} "
        f"FGDice={init_summary['foreground_dice']:.6f} "
        f"PredFG={init_summary['foreground_predicted_voxels']} "
        f"Empty={init_summary['foreground_empty_cases']}/{len(val_rows)}"
    )

    trajectory = [
        {
            "condition": condition,
            "epoch": 0,
            "train_loss": np.nan,
            "train_target_fg": np.nan,
            "val_loss": init_summary["loss"],
            "val_fg_dice": init_summary["foreground_dice"],
            "val_pred_fg": init_summary["foreground_predicted_voxels"],
            "val_empty_fg": init_summary["foreground_empty_cases"],
            "elapsed_sec": time.time() - t0,
        }
    ]

    class_trajectories = []

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_info = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            train_rows,
            part9,
            part11,
        )

        val_summary, val_classes, _, _ = evaluate(
            model,
            loss_fn,
            val_rows,
            part9,
            part11,
        )

        elapsed = time.time() - start

        trajectory.append(
            {
                "condition": condition,
                "epoch": epoch,
                "train_loss": train_info["loss"],
                "train_target_fg": train_info[
                    "mean_target_foreground_voxels"
                ],
                "val_loss": val_summary["loss"],
                "val_fg_dice": val_summary["foreground_dice"],
                "val_pred_fg": val_summary[
                    "foreground_predicted_voxels"
                ],
                "val_empty_fg": val_summary[
                    "foreground_empty_cases"
                ],
                "elapsed_sec": elapsed,
            }
        )

        for _, r in val_classes.iterrows():
            class_trajectories.append(
                {
                    "condition": condition,
                    "epoch": epoch,
                    **r.to_dict(),
                }
            )

        print(
            f"E{epoch} | "
            f"train_loss={train_info['loss']:.6f} "
            f"trainFG={train_info['mean_target_foreground_voxels']:.1f} "
            f"val_loss={val_summary['loss']:.6f} "
            f"FGDice={val_summary['foreground_dice']:.6f} "
            f"PredFG={val_summary['foreground_predicted_voxels']} "
            f"Empty={val_summary['foreground_empty_cases']}/{len(val_rows)} "
            f"time={elapsed/60:.2f} min"
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
    class_traj_df = pd.DataFrame(class_trajectories)

    best_idx = trajectory_df["val_fg_dice"].idxmax()
    best = trajectory_df.loc[best_idx]

    return {
        "trajectory": trajectory_df,
        "class_trajectory": class_traj_df,
        "best": best.to_dict(),
    }


def main():
    global P11_MODULE

    banner("PART 71")
    print("RSNA-ONLY CLASS-BALANCED DICECELoss TRAINING ABLATION")
    print("")
    print("Purpose:")
    print(
        "Test whether explicit class-balanced DiceCELoss improves "
        "the class-selective failure identified in Part 70."
    )
    print("")
    print("No SPIDER.")
    print("No RSNA test set.")
    print("Both conditions start from the exact same Part 15 initialization.")

    validate_paths()
    set_seed(SEED)

    P11_MODULE = load_module(P11, "part71_part11")
    P9_MODULE = load_module(P9, "part71_part9")

    train_rows = pd.read_csv(TRAIN_CSV).head(TRAIN_N).copy()
    val_rows = pd.read_csv(VAL_CSV).head(VAL_N).copy()

    if len(train_rows) != TRAIN_N or len(val_rows) != VAL_N:
        raise RuntimeError(
            f"Unexpected cohort sizes: train={len(train_rows)}, "
            f"val={len(val_rows)}"
        )

    banner("PART 71 LOCKED EXPERIMENT")
    print(f"Device                     : {DEVICE}")
    print(f"Train cases                : {len(train_rows)}")
    print(f"Validation cases           : {len(val_rows)}")
    print(f"Full shape                 : {FULL}")
    print(f"Centered crop              : {CROP}")
    print(f"Epochs                     : {EPOCHS}")
    print(f"Learning rate              : {LR}")
    print(f"Batch size                 : {BATCH_SIZE}")
    print(f"R2 supervision             : R2_FULL")
    print(f"Initialization SHA256      : {sha256_file(INIT_CKPT)}")
    print("")
    print("Conditions:")
    print("  A = R2_FULL_BASELINE")
    print("  B = R2_FULL_CLASS_BALANCED")
    print("")
    print("Class-balanced weights:")
    for c, w in enumerate(CLASS_BALANCED_WEIGHTS):
        name = "Background" if c == 0 else CLASSES[c]
        print(f"  C{c} {name:<42}: {w:.6f}")

    initial_state = load_initial_state()

    # Same initial state and seed for each condition.
    results = {}

    for condition in [
        "R2_FULL_BASELINE",
        "R2_FULL_CLASS_BALANCED",
    ]:
        set_seed(SEED)

        results[condition] = run_condition(
            condition,
            train_rows,
            val_rows,
            P9_MODULE,
            P11_MODULE,
            initial_state,
        )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    all_traj = pd.concat(
        [v["trajectory"] for v in results.values()],
        ignore_index=True,
    )

    all_class = pd.concat(
        [v["class_trajectory"] for v in results.values()],
        ignore_index=True,
    )

    baseline_best = results["R2_FULL_BASELINE"]["best"]
    balanced_best = results["R2_FULL_CLASS_BALANCED"]["best"]

    delta = (
        float(balanced_best["val_fg_dice"])
        - float(baseline_best["val_fg_dice"])
    )

    banner("PART 71 FINAL SUMMARY")
    print(
        f"Baseline best Dice        : "
        f"{float(baseline_best['val_fg_dice']):.6f} "
        f"(E{int(baseline_best['epoch'])})"
    )
    print(
        f"Class-balanced best Dice  : "
        f"{float(balanced_best['val_fg_dice']):.6f} "
        f"(E{int(balanced_best['epoch'])})"
    )
    print(f"Best Dice delta (B-A)     : {delta:+.6f}")

    if delta >= 0.02:
        diagnosis = "MEANINGFUL_CLASS_BALANCED_DICECE_IMPROVEMENT"
    elif delta >= 0.005:
        diagnosis = "SMALL_CLASS_BALANCED_DICECE_IMPROVEMENT"
    elif delta <= -0.005:
        diagnosis = "CLASS_BALANCED_DICECE_HURTS_PERFORMANCE"
    else:
        diagnosis = "NO_MEANINGFUL_CLASS_BALANCED_DICECE_ADVANTAGE"

    print(f"Diagnosis                  : {diagnosis}")

    # Best class-wise snapshot for each condition.
    best_class_rows = []
    for condition, result in results.items():
        best_epoch = int(result["best"]["epoch"])

        if best_epoch == 0:
            # INIT class trajectory is not stored separately; use the first
            # trained epoch as the class-wise analysis baseline if needed.
            continue

        g = all_class[
            (all_class.condition == condition)
            & (all_class.epoch == best_epoch)
        ].copy()

        best_class_rows.append(g)

    if best_class_rows:
        best_class_df = pd.concat(best_class_rows, ignore_index=True)
    else:
        best_class_df = pd.DataFrame()

    # Save outputs.
    all_traj.to_csv(
        REPORT / "part71_learning_trajectory.csv",
        index=False,
    )

    all_class.to_csv(
        REPORT / "part71_classwise_trajectory.csv",
        index=False,
    )

    best_class_df.to_csv(
        REPORT / "part71_best_epoch_classwise_metrics.csv",
        index=False,
    )

    comparison = pd.DataFrame(
        [
            {
                "condition": "R2_FULL_BASELINE",
                "best_epoch": int(baseline_best["epoch"]),
                "best_val_fg_dice": float(
                    baseline_best["val_fg_dice"]
                ),
                "best_val_loss": float(
                    baseline_best["val_loss"]
                ),
                "best_val_pred_fg": float(
                    baseline_best["val_pred_fg"]
                ),
                "best_val_empty_fg": int(
                    baseline_best["val_empty_fg"]
                ),
            },
            {
                "condition": "R2_FULL_CLASS_BALANCED",
                "best_epoch": int(balanced_best["epoch"]),
                "best_val_fg_dice": float(
                    balanced_best["val_fg_dice"]
                ),
                "best_val_loss": float(
                    balanced_best["val_loss"]
                ),
                "best_val_pred_fg": float(
                    balanced_best["val_pred_fg"]
                ),
                "best_val_empty_fg": int(
                    balanced_best["val_empty_fg"]
                ),
            },
        ]
    )

    comparison["delta_vs_baseline"] = (
        comparison["best_val_fg_dice"]
        - float(baseline_best["val_fg_dice"])
    )

    comparison.to_csv(
        REPORT / "part71_condition_comparison.csv",
        index=False,
    )

    summary = {
        "part": 71,
        "device": str(DEVICE),
        "train_cases": TRAIN_N,
        "validation_cases": VAL_N,
        "full_shape": FULL,
        "crop_shape": CROP,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "initialization_checkpoint": str(INIT_CKPT),
        "initialization_sha256": sha256_file(INIT_CKPT),
        "class_balanced_weights": CLASS_BALANCED_WEIGHTS.tolist(),
        "baseline_best": baseline_best,
        "class_balanced_best": balanced_best,
        "best_dice_delta": delta,
        "diagnosis": diagnosis,
        "training_completed": True,
    }

    (REPORT / "part71_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    report_lines = [
        "PART 71 - RSNA-ONLY CLASS-BALANCED DICECE TRAINING ABLATION",
        "",
        "Controlled comparison:",
        "A) R2_FULL_BASELINE - original Part 11 DiceCELoss",
        "B) R2_FULL_CLASS_BALANCED - explicit class-balanced DiceCELoss",
        "",
        "No SPIDER.",
        "No RSNA test set.",
        "Same Part 15 initialization for both conditions.",
        "",
        "CLASS-BALANCED WEIGHTS:",
        json.dumps(CLASS_BALANCED_WEIGHTS.tolist(), indent=2),
        "",
        "FINAL RESULTS:",
        json.dumps(
            {
                "baseline_best": baseline_best,
                "class_balanced_best": balanced_best,
                "best_dice_delta": delta,
                "diagnosis": diagnosis,
            },
            indent=2,
            default=str,
        ),
    ]

    (REPORT / "part71_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    banner("PART 71 OUTPUTS")
    print(REPORT / "part71_learning_trajectory.csv")
    print(REPORT / "part71_classwise_trajectory.csv")
    print(REPORT / "part71_best_epoch_classwise_metrics.csv")
    print(REPORT / "part71_condition_comparison.csv")
    print(REPORT / "part71_summary.json")
    print(REPORT / "part71_report.txt")
    print("")
    print("Checkpoints:")
    print(CKPT)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        banner("PART 71 FAILED")
        traceback.print_exc()
        raise
