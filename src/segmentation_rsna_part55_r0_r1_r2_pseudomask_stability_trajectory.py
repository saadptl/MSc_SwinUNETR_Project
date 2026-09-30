"""
PART 55 — R0 vs R1 vs R2 PSEUDO-MASK STABILITY / TRAJECTORY

Purpose:
    Determine whether the Part 54 R0/R1/R2 foreground-Dice ordering remains
    stable beyond 3 epochs, without introducing another loss or sampling
    variable.

Conditions:
    R0 = original Part 15 pseudo-mask
    R1 = 1-voxel 3-D 6-connected dilation
    R2 = 2-voxel 3-D 6-connected dilation

Controlled variables:
    - exact Part 15 original initialization
    - first 100 Part 15 training cases
    - first 50 Part 15 validation cases
    - foreground-centered 32x64x64 crops
    - original DiceCELoss
    - AdamW, LR=1e-4, WD=1e-5
    - 5 epochs
    - same cohort/order
    - same crop strategy
    - no SPIDER
    - no test set
    - Part 15 untouched

R3 is excluded based on Part 53 visual/spatial assessment.

Important:
    This is a stability/trajectory experiment, not a medical ground-truth
    segmentation validation. Pseudo-mask dilation is an experimental target
    representation.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss


# ================================================================
# CONFIGURATION
# ================================================================

ROOT = Path(
    r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"
)

PART11_PATH = ROOT / "src" / "segmentation_rsna_part11_controlled_pilot_training.py"
PART9_PATH = ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py"

P15 = ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
INIT = P15 / "checkpoints" / "part15_initialization_from_part11.pth"
TRCSV = P15 / "part15_train_cohort.csv"
VACSV = P15 / "part15_validation_cohort.csv"

OUT = ROOT / "outputs" / "segmentation" / "rsna_part55_r0_r1_r2_pseudomask_stability_trajectory"
REPORT = OUT / "reports"

CONDITIONS = {
    "R0_original": 0,
    "R1_dilation": 1,
    "R2_dilation": 2,
}

TRAIN_N = 100
VAL_N = 50
EPOCHS = 5
SEED = 155

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)
NUM_CLASSES = 6

LR = 1e-4
WD = 1e-5

CLASS_NAMES = {
    0: "Background",
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}


# ================================================================
# UTILITIES
# ================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clamp_start(center: int, crop: int, total: int):
    return max(0, min(center - crop // 2, total - crop))


def foreground_centered_crop(image, mask):
    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        center = np.array([s // 2 for s in image.shape])
    else:
        center = np.round(coords.mean(axis=0)).astype(int)

    starts = [
        clamp_start(int(center[i]), CROP_SHAPE[i], image.shape[i])
        for i in range(3)
    ]

    z, y, x = starts
    dz, dy, dx = CROP_SHAPE

    return (
        image[z:z + dz, y:y + dy, x:x + dx].astype(np.float32),
        mask[z:z + dz, y:y + dy, x:x + dx].astype(np.int64),
    )


# ================================================================
# 3-D 6-CONNECTED DILATION
# ================================================================

def dilate_binary_6_connected(binary, iterations):
    result = binary.astype(bool).copy()

    for _ in range(iterations):
        expanded = result.copy()

        expanded[1:, :, :] |= result[:-1, :, :]
        expanded[:-1, :, :] |= result[1:, :, :]

        expanded[:, 1:, :] |= result[:, :-1, :]
        expanded[:, :-1, :] |= result[:, 1:, :]

        expanded[:, :, 1:] |= result[:, :, :-1]
        expanded[:, :, :-1] |= result[:, :, 1:]

        result = expanded

    return result


def dilate_multiclass_mask(mask, radius):
    if radius == 0:
        return mask.copy()

    candidates = []

    for class_id in range(1, NUM_CLASSES):
        original = mask == class_id

        if original.any():
            candidates.append(
                (
                    int(original.sum()),
                    class_id,
                    dilate_binary_6_connected(original, radius),
                )
            )

    # Deterministic overlap resolution:
    # larger original class regions get priority, then lower class id.
    candidates.sort(key=lambda x: (-x[0], x[1]))

    result = np.zeros_like(mask, dtype=np.int64)
    occupied = np.zeros_like(mask, dtype=bool)

    for _, class_id, region in candidates:
        assignable = region & (~occupied)
        result[assignable] = class_id
        occupied |= assignable

    return result


# ================================================================
# DATA LOADING
# ================================================================

def load_case(part11, part9, row):
    loaded = part11.load_tensor_case(row, part9)

    image = loaded[0]
    mask = loaded[1]

    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    image = np.asarray(image)
    mask = np.asarray(mask)

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]

    if tuple(image.shape) != FULL_SHAPE:
        raise RuntimeError(f"Unexpected image shape: {image.shape}")

    if tuple(mask.shape) != FULL_SHAPE:
        raise RuntimeError(f"Unexpected mask shape: {mask.shape}")

    return foreground_centered_crop(
        image.astype(np.float32),
        mask.astype(np.int64),
    )


def preload_cases(part11, part9, df, label):
    cases = []

    print()
    print("=" * 82)
    print(f"PART 55 {label.upper()} PRELOAD")
    print("=" * 82)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        image, mask = load_case(part11, part9, row)

        cases.append(
            {
                "image": image,
                "mask": mask,
                "index": i,
            }
        )

        if i in (1, 25, 50, 75, 100) or i == len(df):
            print(
                f"{label.upper()} {i:03d}/{len(df)} "
                f"FG={int((mask > 0).sum())}"
            )

    return cases


# ================================================================
# METRICS
# ================================================================

def binary_fg_dice(target, prediction):
    intersection = np.logical_and(target, prediction).sum()
    denom = target.sum() + prediction.sum()

    if denom == 0:
        return 1.0

    return float(2.0 * intersection / denom)


@torch.no_grad()
def evaluate_model(model, cases, device, loss_fn):
    model.eval()

    losses = []
    dice_values = []
    pred_fg_values = []
    target_fg_values = []
    fg_prob_values = []
    bg_prob_values = []
    empty_cases = 0

    per_class_dice = {i: [] for i in range(NUM_CLASSES)}
    per_class_pred = {i: [] for i in range(NUM_CLASSES)}
    per_class_target = {i: [] for i in range(NUM_CLASSES)}
    per_class_prob = {i: [] for i in range(NUM_CLASSES)}

    for case in cases:
        x = (
            torch.from_numpy(case["image"])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        y = (
            torch.from_numpy(case["mask"])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        logits = model(x)
        loss = loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        target_np = y[:, 0].cpu().numpy()[0]
        pred_np = pred.cpu().numpy()[0]

        fg_target = target_np > 0
        fg_pred = pred_np > 0

        dice_values.append(binary_fg_dice(fg_target, fg_pred))
        losses.append(float(loss.item()))

        pred_fg_values.append(int(fg_pred.sum()))
        target_fg_values.append(int(fg_target.sum()))

        fg_prob_values.append(
            float(probs[:, 1:, ...].sum(dim=1).mean().item())
        )

        bg_prob_values.append(
            float(probs[:, 0, ...].mean().item())
        )

        if fg_pred.sum() == 0:
            empty_cases += 1

        for class_id in range(NUM_CLASSES):
            t = target_np == class_id
            p = pred_np == class_id

            per_class_target[class_id].append(int(t.sum()))
            per_class_pred[class_id].append(int(p.sum()))
            per_class_prob[class_id].append(
                float(probs[:, class_id, ...].mean().item())
            )

            d = t.sum() + p.sum()

            if d == 0:
                # Explicitly record absent/absent as NaN for the aggregate.
                # This avoids treating a missing class as a perfect prediction.
                per_class_dice[class_id].append(np.nan)
            else:
                per_class_dice[class_id].append(
                    float(2.0 * np.logical_and(t, p).sum() / d)
                )

        del x, y, logits, probs, pred

    class_dice_summary = {}
    class_pred_summary = {}
    class_target_summary = {}
    class_prob_summary = {}

    for class_id in range(NUM_CLASSES):
        vals = np.asarray(per_class_dice[class_id], dtype=np.float64)

        class_dice_summary[str(class_id)] = (
            float(np.nanmean(vals))
            if np.isfinite(vals).any()
            else None
        )

        class_pred_summary[str(class_id)] = float(
            np.mean(per_class_pred[class_id])
        )

        class_target_summary[str(class_id)] = float(
            np.mean(per_class_target[class_id])
        )

        class_prob_summary[str(class_id)] = float(
            np.mean(per_class_prob[class_id])
        )

    return {
        "loss": float(np.mean(losses)),
        "foreground_dice": float(np.mean(dice_values)),
        "predicted_foreground_voxels": float(np.mean(pred_fg_values)),
        "target_foreground_voxels": float(np.mean(target_fg_values)),
        "foreground_probability": float(np.mean(fg_prob_values)),
        "background_probability": float(np.mean(bg_prob_values)),
        "empty_prediction_cases": int(empty_cases),
        "total_cases": len(cases),
        "class_dice": class_dice_summary,
        "class_predicted_voxels": class_pred_summary,
        "class_target_voxels": class_target_summary,
        "class_probability": class_prob_summary,
    }


def train_one_epoch(model, cases, device, loss_fn, optimizer, scaler):
    model.train()

    losses = []
    dice_values = []

    for case in cases:
        x = (
            torch.from_numpy(case["image"])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        y = (
            torch.from_numpy(case["mask"])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.amp.autocast(
                device_type="cuda",
                enabled=True,
            ):
                logits = model(x)
                loss = loss_fn(logits, y)
        else:
            logits = model(x)
            loss = loss_fn(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)

            t = y[:, 0, ...].cpu().numpy()[0] > 0
            p = pred.cpu().numpy()[0] > 0

            dice = binary_fg_dice(t, p)

        losses.append(float(loss.item()))
        dice_values.append(float(dice))

        del x, y, logits, loss, pred

    return {
        "loss": float(np.mean(losses)),
        "foreground_dice": float(np.mean(dice_values)),
    }


# ================================================================
# CONDITION
# ================================================================

def run_condition(
    name,
    radius,
    base_train,
    base_val,
    part11,
    device,
):
    print()
    print("=" * 82)
    print(f"PART 55 CONDITION: {name} | RADIUS {radius}")
    print("=" * 82)

    train_cases = [
        {
            "image": c["image"],
            "mask": dilate_multiclass_mask(c["mask"], radius),
            "index": c["index"],
        }
        for c in base_train
    ]

    val_cases = [
        {
            "image": c["image"],
            "mask": dilate_multiclass_mask(c["mask"], radius),
            "index": c["index"],
        }
        for c in base_val
    ]

    train_fg = [int((c["mask"] > 0).sum()) for c in train_cases]
    val_fg = [int((c["mask"] > 0).sum()) for c in val_cases]

    print(f"Train mean target FG : {np.mean(train_fg):.2f}")
    print(f"Train zero-FG cases  : {sum(v == 0 for v in train_fg)}/{len(train_fg)}")
    print(f"Val mean target FG   : {np.mean(val_fg):.2f}")
    print(f"Val zero-FG cases    : {sum(v == 0 for v in val_fg)}/{len(val_fg)}")

    model = part11.create_model(device)

    state = torch.load(
        INIT,
        map_location=device,
    )

    model.load_state_dict(
        state["model_state_dict"],
        strict=True,
    )

    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
    )

    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if device.type == "cuda"
        else None
    )

    condition_dir = OUT / name
    checkpoint_dir = condition_dir / "checkpoints"

    condition_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    initial = evaluate_model(
        model,
        val_cases,
        device,
        loss_fn,
    )

    initial_record = {
        **initial,
        "epoch": 0,
        "train_loss": None,
        "train_foreground_dice": None,
        "elapsed_seconds": 0.0,
    }

    history.append(initial_record)

    print(
        f"INIT | "
        f"val_loss={initial['loss']:.6f} "
        f"val_FGDice={initial['foreground_dice']:.6f} "
        f"PredFG={initial['predicted_foreground_voxels']:.1f} "
        f"TargetFG={initial['target_foreground_voxels']:.1f} "
        f"FGProb={initial['foreground_probability']:.6f} "
        f"Empty={initial['empty_prediction_cases']}/{VAL_N}"
    )

    best_dice = initial["foreground_dice"]
    best_epoch = 0
    best_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_metrics = train_one_epoch(
            model,
            train_cases,
            device,
            loss_fn,
            optimizer,
            scaler,
        )

        val_metrics = evaluate_model(
            model,
            val_cases,
            device,
            loss_fn,
        )

        elapsed = float(time.time() - start)

        record = {
            **val_metrics,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_foreground_dice": train_metrics["foreground_dice"],
            "elapsed_seconds": elapsed,
        }

        history.append(record)

        if val_metrics["foreground_dice"] > best_dice:
            best_dice = val_metrics["foreground_dice"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        # Save every epoch.
        torch.save(
            {
                "part": 55,
                "condition": name,
                "radius": radius,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            checkpoint_dir / f"epoch_{epoch:02d}.pth",
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_FGDice={train_metrics['foreground_dice']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_FGDice={val_metrics['foreground_dice']:.6f} | "
            f"PredFG={val_metrics['predicted_foreground_voxels']:.1f} "
            f"TargetFG={val_metrics['target_foreground_voxels']:.1f} "
            f"FGProb={val_metrics['foreground_probability']:.6f} "
            f"Empty={val_metrics['empty_prediction_cases']}/{VAL_N} | "
            f"{elapsed:.1f}s"
        )

    # Save the actual best model state, not merely the final state.
    torch.save(
        {
            "part": 55,
            "condition": name,
            "radius": radius,
            "best_epoch": best_epoch,
            "best_foreground_dice": float(best_dice),
            "model_state_dict": best_state,
        },
        checkpoint_dir / "best_model.pth",
    )

    history_path = condition_dir / "history.csv"

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_foreground_dice",
                "val_loss",
                "val_foreground_dice",
                "predicted_foreground_voxels",
                "target_foreground_voxels",
                "foreground_probability",
                "background_probability",
                "empty_prediction_cases",
                "elapsed_seconds",
            ]
        )

        for r in history:
            writer.writerow(
                [
                    r["epoch"],
                    r["train_loss"],
                    r["train_foreground_dice"],
                    r["loss"],
                    r["foreground_dice"],
                    r["predicted_foreground_voxels"],
                    r["target_foreground_voxels"],
                    r["foreground_probability"],
                    r["background_probability"],
                    r["empty_prediction_cases"],
                    r["elapsed_seconds"],
                ]
            )

    with open(
        condition_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "condition": name,
                "radius": radius,
                "train_mean_target_fg": float(np.mean(train_fg)),
                "validation_mean_target_fg": float(np.mean(val_fg)),
                "best_epoch": int(best_epoch),
                "best_foreground_dice": float(best_dice),
                "history": history,
            },
            f,
            indent=2,
        )

    result = {
        "condition": name,
        "radius": radius,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_foreground_dice": float(best_dice),
        "train_mean_target_fg": float(np.mean(train_fg)),
        "validation_mean_target_fg": float(np.mean(val_fg)),
    }

    del model, optimizer, scaler, train_cases, val_cases

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ================================================================
# MAIN
# ================================================================

def main():
    seed_everything(SEED)

    print("=" * 82)
    print("PART 55 PATH VALIDATION")
    print("=" * 82)

    required = [
        ("Project root", ROOT),
        ("Part 11", PART11_PATH),
        ("Part 9", PART9_PATH),
        ("Part 15 initialization", INIT),
        ("Part 15 train cohort", TRCSV),
        ("Part 15 validation cohort", VACSV),
    ]

    for label, path in required:
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{label:<40}: {status}")

        if not path.exists():
            raise FileNotFoundError(path)

    print()
    print("=" * 82)
    print("PART 55 — R0 vs R1 vs R2 PSEUDO-MASK STABILITY / TRAJECTORY")
    print("=" * 82)
    print(f"Train subset : {TRAIN_N}")
    print(f"Validation subset : {VAL_N}")
    print(f"Epochs : {EPOCHS}")
    print(f"Full volume : {FULL_SHAPE}")
    print(f"Crop : {CROP_SHAPE}")
    print(f"LR : {LR}")
    print(f"WD : {WD}")
    print("Crop strategy : foreground-centered")
    print("Loss : original DiceCELoss")
    print("Radii : [0, 1, 2]")
    print("R3 trained : NO")
    print("SPIDER : NO")
    print("Test set : NO")
    print("Part 15 overwritten : NO")
    print(f"Initialization SHA256 : {sha256(INIT)}")

    part11 = load_module(
        PART11_PATH,
        "segmentation_rsna_part11_part55_runtime",
    )

    part9 = load_module(
        PART9_PATH,
        "segmentation_rsna_part9_part55_runtime",
    )

    train_df = pd.read_csv(TRCSV).head(TRAIN_N)
    val_df = pd.read_csv(VACSV).head(VAL_N)

    base_train = preload_cases(
        part11,
        part9,
        train_df,
        "train",
    )

    base_val = preload_cases(
        part11,
        part9,
        val_df,
        "validation",
    )

    print()
    print("=" * 82)
    print("PART 55 SHAPE / LABEL SMOKE TEST")
    print("=" * 82)

    assert base_train[0]["image"].shape == CROP_SHAPE
    assert base_train[0]["mask"].shape == CROP_SHAPE
    assert base_train[0]["mask"].min() >= 0
    assert base_train[0]["mask"].max() < NUM_CLASSES

    print(f"Image : {base_train[0]['image'].shape}")
    print(f"Mask : {base_train[0]['mask'].shape}")
    print(f"Labels : {sorted(np.unique(base_train[0]['mask']).tolist())}")
    print("✓ Shape / label smoke test PASSED.")

    print()
    print("=" * 82)
    print("PART 55 TARGET DENSITY PREVIEW")
    print("=" * 82)

    density = {}

    for name, radius in CONDITIONS.items():
        train_fg = [
            int(
                (dilate_multiclass_mask(c["mask"], radius) > 0).sum()
            )
            for c in base_train
        ]

        val_fg = [
            int(
                (dilate_multiclass_mask(c["mask"], radius) > 0).sum()
            )
            for c in base_val
        ]

        density[name] = {
            "radius": radius,
            "train_mean_fg": float(np.mean(train_fg)),
            "train_occupancy": float(
                np.mean(train_fg) / np.prod(CROP_SHAPE)
            ),
            "validation_mean_fg": float(np.mean(val_fg)),
            "validation_occupancy": float(
                np.mean(val_fg) / np.prod(CROP_SHAPE)
            ),
        }

        print(
            f"{name:<18} | "
            f"train FG={np.mean(train_fg):.2f} "
            f"occ={np.mean(train_fg)/np.prod(CROP_SHAPE):.6f} | "
            f"val FG={np.mean(val_fg):.2f} "
            f"occ={np.mean(val_fg)/np.prod(CROP_SHAPE):.6f}"
        )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print()
    print(f"PyTorch : {torch.__version__}")
    print(f"Device : {device}")

    results = {}

    for name, radius in CONDITIONS.items():
        # Same deterministic starting point for every condition.
        seed_everything(SEED)

        results[name] = run_condition(
            name=name,
            radius=radius,
            base_train=base_train,
            base_val=base_val,
            part11=part11,
            device=device,
        )

    # ============================================================
    # TRAJECTORY COMPARISON
    # ============================================================

    print()
    print("=" * 82)
    print("PART 55 TRAJECTORY COMPARISON")
    print("=" * 82)

    trajectory_rows = []

    for epoch in range(0, EPOCHS + 1):
        values = {}

        for name in CONDITIONS:
            rec = results[name]["history"][epoch]

            values[name] = {
                "foreground_dice": rec["foreground_dice"],
                "val_loss": rec["loss"],
                "predicted_fg": rec["predicted_foreground_voxels"],
                "fg_probability": rec["foreground_probability"],
                "empty": rec["empty_prediction_cases"],
            }

        print()
        print(f"Epoch {epoch:02d}")

        for name in CONDITIONS:
            v = values[name]

            print(
                f"  {name:<18} | "
                f"FGDice={v['foreground_dice']:.6f} | "
                f"val_loss={v['val_loss']:.6f} | "
                f"PredFG={v['predicted_fg']:.1f} | "
                f"FGProb={v['fg_probability']:.6f} | "
                f"Empty={v['empty']}/{VAL_N}"
            )

        trajectory_rows.append(
            {
                "epoch": epoch,
                "R0_FGDice": values["R0_original"]["foreground_dice"],
                "R1_FGDice": values["R1_dilation"]["foreground_dice"],
                "R2_FGDice": values["R2_dilation"]["foreground_dice"],
                "R0_val_loss": values["R0_original"]["val_loss"],
                "R1_val_loss": values["R1_dilation"]["val_loss"],
                "R2_val_loss": values["R2_dilation"]["val_loss"],
                "R0_pred_fg": values["R0_original"]["predicted_fg"],
                "R1_pred_fg": values["R1_dilation"]["predicted_fg"],
                "R2_pred_fg": values["R2_dilation"]["predicted_fg"],
                "R0_fg_prob": values["R0_original"]["fg_probability"],
                "R1_fg_prob": values["R1_dilation"]["fg_probability"],
                "R2_fg_prob": values["R2_dilation"]["fg_probability"],
                "R0_empty": values["R0_original"]["empty"],
                "R1_empty": values["R1_dilation"]["empty"],
                "R2_empty": values["R2_dilation"]["empty"],
            }
        )

    # ============================================================
    # STABILITY DIAGNOSTICS
    # ============================================================

    final_dice = {
        name: results[name]["history"][-1]["foreground_dice"]
        for name in CONDITIONS
    }

    best_dice = {
        name: results[name]["best_foreground_dice"]
        for name in CONDITIONS
    }

    best_epochs = {
        name: results[name]["best_epoch"]
        for name in CONDITIONS
    }

    final_pred_fg = {
        name: results[name]["history"][-1]["predicted_foreground_voxels"]
        for name in CONDITIONS
    }

    first_zero_epoch = {}

    for name in CONDITIONS:
        first_zero_epoch[name] = None

        for rec in results[name]["history"]:
            if rec["empty_prediction_cases"] == VAL_N:
                first_zero_epoch[name] = rec["epoch"]
                break

    # Determine the winner at each epoch.
    epoch_winners = []

    for row in trajectory_rows:
        scores = {
            "R0": row["R0_FGDice"],
            "R1": row["R1_FGDice"],
            "R2": row["R2_FGDice"],
        }

        winner = max(scores.items(), key=lambda x: x[1])[0]

        epoch_winners.append(
            {
                "epoch": row["epoch"],
                "winner": winner,
                "scores": scores,
            }
        )

    # Winner from epoch 1 onward.
    post_init_winners = [
        x["winner"]
        for x in epoch_winners
        if x["epoch"] >= 1
    ]

    r2_post_init_wins = sum(
        winner == "R2"
        for winner in post_init_winners
    )

    if r2_post_init_wins == EPOCHS:
        trajectory_diagnosis = "R2_WIN_STABLE_ACROSS_ALL_TRAINING_EPOCHS"
    elif r2_post_init_wins >= 3:
        trajectory_diagnosis = "R2_GENERALLY_STRONGEST_BUT_NOT_UNIVERSALLY_STABLE"
    else:
        trajectory_diagnosis = "R2_ADVANTAGE_NOT_STABLE"

    if all(
        first_zero_epoch[name] is None
        for name in CONDITIONS
    ):
        collapse_diagnosis = "NO_COMPLETE_BACKGROUND_COLLAPSE_WITHIN_5_EPOCHS"
    else:
        collapse_diagnosis = "AT_LEAST_ONE_CONDITION_REACHED_COMPLETE_BACKGROUND_COLLAPSE"

    final_best = max(
        final_dice.items(),
        key=lambda x: x[1],
    )

    print()
    print("=" * 82)
    print("PART 55 STABILITY DIAGNOSTICS")
    print("=" * 82)

    for name in CONDITIONS:
        print(
            f"{name:<18} | "
            f"best FGDice={best_dice[name]:.6f} "
            f"(epoch {best_epochs[name]}) | "
            f"final FGDice={final_dice[name]:.6f} | "
            f"first complete collapse epoch={first_zero_epoch[name]}"
        )

    print()
    print(f"Final best radius : {final_best[0]}")
    print(f"Final best FGDice : {final_best[1]:.6f}")
    print(f"R2 wins epochs 1-5 : {r2_post_init_wins}/{EPOCHS}")
    print(f"Trajectory diagnosis : {trajectory_diagnosis}")
    print(f"Collapse diagnosis : {collapse_diagnosis}")

    # ============================================================
    # SAVE REPORTS
    # ============================================================

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_csv = REPORT / "part55_trajectory.csv"

    with open(
        trajectory_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(trajectory_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(trajectory_rows)

    summary = {
        "part": 55,
        "purpose": (
            "Five-epoch stability/trajectory validation of the Part 54 "
            "R0/R1/R2 pseudo-mask ablation."
        ),
        "configuration": {
            "train_cases": TRAIN_N,
            "validation_cases": VAL_N,
            "epochs": EPOCHS,
            "full_shape": FULL_SHAPE,
            "crop_shape": CROP_SHAPE,
            "crop_strategy": "foreground-centered",
            "loss": "DiceCELoss(to_onehot_y=True, softmax=True)",
            "optimizer": "AdamW",
            "learning_rate": LR,
            "weight_decay": WD,
            "radii": [0, 1, 2],
            "r3_trained": False,
            "spider": False,
            "test_set": False,
            "part15_overwritten": False,
            "initialization_sha256": sha256(INIT),
        },
        "target_density": density,
        "results": results,
        "trajectory": trajectory_rows,
        "epoch_winners": epoch_winners,
        "stability": {
            "best_foreground_dice": best_dice,
            "best_epoch": best_epochs,
            "final_foreground_dice": final_dice,
            "final_predicted_foreground": final_pred_fg,
            "first_complete_collapse_epoch": first_zero_epoch,
            "r2_wins_epochs_1_to_5": r2_post_init_wins,
            "trajectory_diagnosis": trajectory_diagnosis,
            "collapse_diagnosis": collapse_diagnosis,
        },
    }

    summary_path = REPORT / "part55_summary.json"

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

    print()
    print("=" * 82)
    print("PART 55 COMPLETE")
    print("=" * 82)
    print(f"Output directory : {OUT}")
    print(f"Trajectory CSV : {trajectory_csv}")
    print(f"Summary JSON : {summary_path}")
    print()
    print(
        "Scientific note: pseudo-mask dilation changes the experimental "
        "target representation; these metrics do not establish medical "
        "ground-truth segmentation accuracy."
    )


if __name__ == "__main__":
    main()
