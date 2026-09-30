"""
PART 54 — R0 vs R1 vs R2 PSEUDO-MASK TRAINING ABLATION

Controlled training ablation following Parts 51–53.

Conditions:
    R0 = original Part 15 pseudo-mask
    R1 = 1-voxel 3-D 6-connected dilation
    R2 = 2-voxel 3-D 6-connected dilation

Everything else is held constant:
    - exact Part 15 original initialization
    - first 100 Part 15 training cases
    - first 50 Part 15 validation cases
    - foreground-centered 32x64x64 crops
    - original Dice + CE loss
    - AdamW, LR=1e-4, WD=1e-5
    - 3 epochs
    - identical data order
    - no SPIDER
    - no test set
    - Part 15 is never overwritten

R3 is deliberately excluded because Part 53 visual inspection showed
that it is an aggressive spatial expansion and is not required to answer
the controlled R0/R1/R2 question.

This experiment does NOT claim that a dilated pseudo-mask is medical
ground truth. It tests whether controlled target-density changes alter
training behavior.
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
# CONFIG
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

OUT = ROOT / "outputs" / "segmentation" / "rsna_part54_r0_r1_r2_pseudomask_training_ablation"
REPORT = OUT / "reports"

CONDITIONS = {
    "R0_original": 0,
    "R1_dilation": 1,
    "R2_dilation": 2,
}

TRAIN_N = 100
VAL_N = 50
EPOCHS = 3
SEED = 154

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
        image[z:z+dz, y:y+dy, x:x+dx].astype(np.float32),
        mask[z:z+dz, y:y+dy, x:x+dx].astype(np.int64),
    )


# ================================================================
# DILATION — same 3-D 6-connected construction used in Part 52/53
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
                (int(original.sum()), class_id,
                 dilate_binary_6_connected(original, radius))
            )

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
    print(f"PART 54 {label.upper()} PRELOAD")
    print("=" * 82)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        image, mask = load_case(part11, part9, row)

        cases.append({
            "image": image,
            "mask": mask,
            "index": i,
        })

        if i in (1, 25, 50, 75, 100) or i == len(df):
            print(
                f"{label.upper()} {i:03d}/{len(df)} "
                f"FG={int((mask > 0).sum())}"
            )

    return cases


# ================================================================
# METRICS
# ================================================================

@torch.no_grad()
def evaluate_model(model, cases, device, loss_fn):
    model.eval()

    losses = []
    dice_values = []
    pred_fg = []
    target_fg = []
    fg_probs = []
    bg_probs = []
    empty = 0
    class_dice_accum = [[] for _ in range(NUM_CLASSES)]

    for case in cases:
        x = torch.from_numpy(case["image"]).unsqueeze(0).unsqueeze(0).to(device)
        y = torch.from_numpy(case["mask"]).unsqueeze(0).unsqueeze(0).to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        fg_target = (y > 0)
        fg_pred = (pred > 0)

        intersection = (fg_target & fg_pred).sum().item()
        denom = fg_target.sum().item() + fg_pred.sum().item()

        dice = (
            (2.0 * intersection / denom)
            if denom > 0
            else 1.0
        )

        if fg_pred.sum().item() == 0:
            empty += 1

        losses.append(float(loss.item()))
        dice_values.append(float(dice))
        pred_fg.append(int(fg_pred.sum().item()))
        target_fg.append(int(fg_target.sum().item()))
        fg_probs.append(float(probs[:, 1:, ...].sum(dim=1).mean().item()))
        bg_probs.append(float(probs[:, 0, ...].mean().item()))

        for class_id in range(NUM_CLASSES):
            t = y[:, 0, ...] == class_id
            p = pred == class_id
            inter = (t & p).sum().item()
            d = t.sum().item() + p.sum().item()
            class_dice_accum[class_id].append(
                float(2.0 * inter / d) if d > 0 else 1.0
            )

    return {
        "loss": float(np.mean(losses)),
        "foreground_dice": float(np.mean(dice_values)),
        "predicted_foreground_voxels": float(np.mean(pred_fg)),
        "target_foreground_voxels": float(np.mean(target_fg)),
        "foreground_probability": float(np.mean(fg_probs)),
        "background_probability": float(np.mean(bg_probs)),
        "empty_prediction_cases": int(empty),
        "total_cases": len(cases),
        "class_dice": {
            str(i): float(np.mean(class_dice_accum[i]))
            for i in range(NUM_CLASSES)
        },
    }


def train_one_epoch(model, cases, device, loss_fn, optimizer, scaler):
    model.train()

    losses = []
    dice_values = []

    for case in cases:
        x = torch.from_numpy(case["image"]).unsqueeze(0).unsqueeze(0).to(device)
        y = torch.from_numpy(case["mask"]).unsqueeze(0).unsqueeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=(device.type == "cuda"),
        ):
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
            t = y > 0
            p = pred > 0
            inter = (t & p).sum().item()
            denom = t.sum().item() + p.sum().item()
            dice = (2.0 * inter / denom) if denom > 0 else 1.0

        losses.append(float(loss.item()))
        dice_values.append(float(dice))

        del x, y, logits, loss

    return {
        "loss": float(np.mean(losses)),
        "foreground_dice": float(np.mean(dice_values)),
    }


# ================================================================
# CONDITION RUN
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
    print(f"PART 54 CONDITION: {name} | RADIUS {radius}")
    print("=" * 82)

    train_cases = []
    val_cases = []

    for case in base_train:
        train_cases.append({
            "image": case["image"],
            "mask": dilate_multiclass_mask(case["mask"], radius),
            "index": case["index"],
        })

    for case in base_val:
        val_cases.append({
            "image": case["image"],
            "mask": dilate_multiclass_mask(case["mask"], radius),
            "index": case["index"],
        })

    train_fg = [int((c["mask"] > 0).sum()) for c in train_cases]
    val_fg = [int((c["mask"] > 0).sum()) for c in val_cases]

    print(f"Train mean target FG : {np.mean(train_fg):.2f}")
    print(f"Train zero-FG cases  : {sum(v == 0 for v in train_fg)}/{len(train_fg)}")
    print(f"Val mean target FG   : {np.mean(val_fg):.2f}")
    print(f"Val zero-FG cases    : {sum(v == 0 for v in val_fg)}/{len(val_fg)}")

    # Every condition starts from exactly the same stored Part 15 state.
    model = part11.create_model(device)
    state = torch.load(INIT, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)

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

    history = []

    initial = evaluate_model(
        model,
        val_cases,
        device,
        loss_fn,
    )

    initial["epoch"] = 0
    initial["train_loss"] = None
    initial["train_foreground_dice"] = None
    history.append(initial)

    print(
        f"INIT | val_loss={initial['loss']:.6f} "
        f"val_FGDice={initial['foreground_dice']:.6f} "
        f"PredFG={initial['predicted_foreground_voxels']:.1f} "
        f"FGProb={initial['foreground_probability']:.6f}"
    )

    condition_dir = OUT / name
    checkpoint_dir = condition_dir / "checkpoints"
    condition_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

        record = {
            **val_metrics,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_foreground_dice": train_metrics["foreground_dice"],
            "elapsed_seconds": float(time.time() - start),
        }

        history.append(record)

        ckpt = {
            "part": 54,
            "condition": name,
            "radius": radius,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }

        torch.save(
            ckpt,
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
            f"Empty={val_metrics['empty_prediction_cases']}/{val_metrics['total_cases']} | "
            f"{record['elapsed_seconds']:.1f}s"
        )

    # Best by validation foreground Dice.
    best = max(
        history,
        key=lambda r: r["foreground_dice"],
    )

    torch.save(
        {
            "part": 54,
            "condition": name,
            "radius": radius,
            "best_epoch": best["epoch"],
            "model_state_dict": model.state_dict(),
            "best_metric": best["foreground_dice"],
        },
        checkpoint_dir / "best_model.pth",
    )

    history_path = condition_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
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
        ])

        for r in history:
            writer.writerow([
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
                r.get("elapsed_seconds"),
            ])

    with open(condition_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "condition": name,
                "radius": radius,
                "train_mean_target_fg": float(np.mean(train_fg)),
                "validation_mean_target_fg": float(np.mean(val_fg)),
                "history": history,
                "best_epoch": int(best["epoch"]),
                "best_foreground_dice": float(best["foreground_dice"]),
            },
            f,
            indent=2,
        )

    del model, optimizer, scaler, train_cases, val_cases
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history, best


# ================================================================
# MAIN
# ================================================================

def main():
    seed_everything(SEED)

    print("=" * 82)
    print("PART 54 PATH VALIDATION")
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
    print("PART 54 — R0 vs R1 vs R2 PSEUDO-MASK TRAINING ABLATION")
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
        "segmentation_rsna_part11_part54_runtime",
    )
    part9 = load_module(
        PART9_PATH,
        "segmentation_rsna_part9_part54_runtime",
    )

    train_df = pd.read_csv(TRCSV).head(TRAIN_N)
    val_df = pd.read_csv(VACSV).head(VAL_N)

    # Ensure deterministic row order.
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
    print("PART 54 SHAPE / LABEL SMOKE TEST")
    print("=" * 82)

    assert base_train[0]["image"].shape == CROP_SHAPE
    assert base_train[0]["mask"].shape == CROP_SHAPE
    assert base_train[0]["mask"].min() >= 0
    assert base_train[0]["mask"].max() < NUM_CLASSES

    print(
        f"Image : {base_train[0]['image'].shape}"
    )
    print(
        f"Mask : {base_train[0]['mask'].shape}"
    )
    print(
        f"Labels : "
        f"{sorted(np.unique(base_train[0]['mask']).tolist())}"
    )
    print("✓ Shape / label smoke test PASSED.")

    # Validate that all three conditions derive from identical R0 data.
    print()
    print("=" * 82)
    print("PART 54 TARGET DENSITY PREVIEW")
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
            "train_occupancy": float(np.mean(train_fg) / np.prod(CROP_SHAPE)),
            "val_mean_fg": float(np.mean(val_fg)),
            "val_occupancy": float(np.mean(val_fg) / np.prod(CROP_SHAPE)),
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

    all_results = {}

    # IMPORTANT: reset RNG before every condition so data/model execution
    # starts from the same deterministic state.
    for name, radius in CONDITIONS.items():
        seed_everything(SEED)

        history, best = run_condition(
            name,
            radius,
            base_train,
            base_val,
            part11,
            device,
        )

        all_results[name] = {
            "radius": radius,
            "history": history,
            "best_epoch": int(best["epoch"]),
            "best_foreground_dice": float(best["foreground_dice"]),
        }

    # ------------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------------

    baseline = all_results["R0_original"]["history"][-1]

    comparison_rows = []

    for name in CONDITIONS:
        final = all_results[name]["history"][-1]

        comparison_rows.append({
            "condition": name,
            "radius": CONDITIONS[name],
            "final_val_loss": final["loss"],
            "final_val_foreground_dice": final["foreground_dice"],
            "final_predicted_fg": final["predicted_foreground_voxels"],
            "final_target_fg": final["target_foreground_voxels"],
            "final_fg_probability": final["foreground_probability"],
            "final_background_probability": final["background_probability"],
            "empty_prediction_cases": final["empty_prediction_cases"],
            "dice_delta_vs_R0": (
                final["foreground_dice"]
                - baseline["foreground_dice"]
            ),
        })

    comparison_path = REPORT / "part54_comparison.json"
    REPORT.mkdir(parents=True, exist_ok=True)

    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "part": 54,
                "configuration": {
                    "train_cases": TRAIN_N,
                    "validation_cases": VAL_N,
                    "epochs": EPOCHS,
                    "crop": CROP_SHAPE,
                    "loss": "DiceCELoss(to_onehot_y=True, softmax=True)",
                    "lr": LR,
                    "weight_decay": WD,
                    "init_sha256": sha256(INIT),
                    "r3_excluded": True,
                },
                "density_preview": density,
                "results": all_results,
                "comparison": comparison_rows,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 82)
    print("PART 54 FINAL COMPARISON")
    print("=" * 82)

    for row in comparison_rows:
        print(
            f"{row['condition']:<18} | "
            f"val_loss={row['final_val_loss']:.6f} | "
            f"FGDice={row['final_val_foreground_dice']:.6f} | "
            f"PredFG={row['final_predicted_fg']:.1f} | "
            f"TargetFG={row['final_target_fg']:.1f} | "
            f"FGProb={row['final_fg_probability']:.6f} | "
            f"Empty={row['empty_prediction_cases']}/{VAL_N} | "
            f"ΔDice={row['dice_delta_vs_R0']:+.6f}"
        )

    r0 = all_results["R0_original"]["history"][-1]["foreground_dice"]
    r1 = all_results["R1_dilation"]["history"][-1]["foreground_dice"]
    r2 = all_results["R2_dilation"]["history"][-1]["foreground_dice"]

    best_radius = max(
        [(0, r0), (1, r1), (2, r2)],
        key=lambda x: x[1],
    )

    max_delta = best_radius[1] - r0

    if max_delta >= 0.05:
        diagnosis = "DILATION_PRODUCED_MEANINGFUL_FOREGROUND_DICE_GAIN"
    elif max_delta >= 0.01:
        diagnosis = "DILATION_PRODUCED_SMALL_FOREGROUND_DICE_GAIN"
    else:
        diagnosis = "DILATION_DID_NOT_PRODUCE_MEANINGFUL_FOREGROUND_DICE_GAIN"

    print()
    print("=" * 82)
    print("PART 54 DIAGNOSTIC INTERPRETATION")
    print("=" * 82)
    print(f"Best radius by final FG Dice : R{best_radius[0]}")
    print(f"Best final FG Dice : {best_radius[1]:.6f}")
    print(f"Best ΔDice vs R0 : {max_delta:+.6f}")
    print(f"Diagnosis : {diagnosis}")

    print()
    print("=" * 82)
    print("PART 54 COMPLETE")
    print("=" * 82)
    print(f"Output directory : {OUT}")
    print(f"Comparison JSON : {comparison_path}")
    print()
    print(
        "Scientific note: this is a pseudo-mask construction ablation. "
        "It does not establish medical ground-truth segmentation accuracy."
    )


if __name__ == "__main__":
    main()
