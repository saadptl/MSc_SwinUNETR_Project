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


ROOT = Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
PART11_PATH = ROOT / "src" / "segmentation_rsna_part11_controlled_pilot_training.py"
PART9_PATH = ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py"

P15 = ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
INIT = P15 / "checkpoints" / "part15_initialization_from_part11.pth"
TRCSV = P15 / "part15_train_cohort.csv"
VACSV = P15 / "part15_validation_cohort.csv"

OUT = ROOT / "outputs" / "segmentation" / "rsna_part57_learning_rate_early_checkpoint_stability"
REPORT = OUT / "reports"

TRAIN_N = 100
VAL_N = 50
EPOCHS = 5
FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)
NUM_CLASSES = 6
RADIUS = 2
WD = 1e-5
SEED = 157

CONDITIONS = {
    "A_constant_1e4": {"initial_lr": 1e-4, "schedule": "constant"},
    "B_constant_5e5": {"initial_lr": 5e-5, "schedule": "constant"},
    "C_step_1e4_to_5e5": {"initial_lr": 1e-4, "schedule": "step"},
}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clamp_start(center, crop, total):
    return max(0, min(int(center) - crop // 2, total - crop))


def centered_crop(image, mask):
    coords = np.argwhere(mask > 0)
    center = (
        np.round(coords.mean(axis=0)).astype(int)
        if coords.size else np.array([s // 2 for s in image.shape])
    )
    starts = [
        clamp_start(center[i], CROP_SHAPE[i], image.shape[i])
        for i in range(3)
    ]
    z, y, x = starts
    dz, dy, dx = CROP_SHAPE
    return (
        image[z:z+dz, y:y+dy, x:x+dx].astype(np.float32),
        mask[z:z+dz, y:y+dy, x:x+dx].astype(np.int64),
    )


def dilate_binary_6(binary, iterations):
    result = binary.astype(bool).copy()
    for _ in range(iterations):
        expanded = result.copy()
        expanded[1:] |= result[:-1]
        expanded[:-1] |= result[1:]
        expanded[:, 1:] |= result[:, :-1]
        expanded[:, :-1] |= result[:, 1:]
        expanded[:, :, 1:] |= result[:, :, :-1]
        expanded[:, :, :-1] |= result[:, :, 1:]
        result = expanded
    return result


def dilate_multiclass(mask, radius=RADIUS):
    if radius == 0:
        return mask.copy()

    candidates = []
    for cid in range(1, NUM_CLASSES):
        region = mask == cid
        if region.any():
            candidates.append((int(region.sum()), cid,
                               dilate_binary_6(region, radius)))
    candidates.sort(key=lambda x: (-x[0], x[1]))

    out = np.zeros_like(mask, dtype=np.int64)
    occupied = np.zeros_like(mask, dtype=bool)
    for _, cid, region in candidates:
        assignable = region & ~occupied
        out[assignable] = cid
        occupied |= assignable
    return out


def load_case(part11, part9, row):
    loaded = part11.load_tensor_case(row, part9)
    image, mask = loaded[0], loaded[1]

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

    if tuple(image.shape) != FULL_SHAPE or tuple(mask.shape) != FULL_SHAPE:
        raise RuntimeError(f"Unexpected shapes: image={image.shape}, mask={mask.shape}")

    mask = dilate_multiclass(mask.astype(np.int64), RADIUS)
    return centered_crop(image.astype(np.float32), mask)


def preload(part11, part9, df, label):
    cases = []
    print()
    print("=" * 82)
    print(f"PART 57 {label.upper()} PRELOAD")
    print("=" * 82)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        image, mask = load_case(part11, part9, row)
        cases.append({"image": image, "mask": mask, "index": i})
        if i in (1, 25, 50, 75, 100) or (label == "validation" and i in (10, 20, 40)):
            print(f"{label.upper()} {i:03d}/{len(df)} FG={int((mask > 0).sum())}")
    return cases


def evaluate(model, cases, device, loss_fn):
    model.eval()
    losses, dices, pred_fg, target_fg, fg_prob, bg_prob = [], [], [], [], [], []
    empty = 0

    for case in cases:
        x = torch.from_numpy(case["image"]).unsqueeze(0).unsqueeze(0).to(device)
        y = torch.from_numpy(case["mask"]).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits, y)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)

        t = y[:, 0].cpu().numpy()[0]
        p = pred.cpu().numpy()[0]
        tfg, pfg = t > 0, p > 0
        den = tfg.sum() + pfg.sum()
        dice = float(2 * np.logical_and(tfg, pfg).sum() / den) if den else 1.0

        if pfg.sum() == 0:
            empty += 1

        losses.append(float(loss.item()))
        dices.append(dice)
        pred_fg.append(int(pfg.sum()))
        target_fg.append(int(tfg.sum()))
        fg_prob.append(float(probs[:, 1:].sum(1).mean().item()))
        bg_prob.append(float(probs[:, 0].mean().item()))

        del x, y, logits, loss, probs, pred

    mp, mt = float(np.mean(pred_fg)), float(np.mean(target_fg))
    return {
        "loss": float(np.mean(losses)),
        "foreground_dice": float(np.mean(dices)),
        "predicted_foreground_voxels": mp,
        "target_foreground_voxels": mt,
        "foreground_probability": float(np.mean(fg_prob)),
        "background_probability": float(np.mean(bg_prob)),
        "probability_gap": float(np.mean(fg_prob) - np.mean(bg_prob)),
        "prediction_target_ratio": float(mp / max(mt, 1e-12)),
        "empty_cases": int(empty),
        "total_cases": len(cases),
    }


def train_epoch(model, optimizer, loss_fn, cases, device):
    model.train()
    losses, dices = [], []

    for case in cases:
        x = torch.from_numpy(case["image"]).unsqueeze(0).unsqueeze(0).to(device)
        y = torch.from_numpy(case["mask"]).unsqueeze(0).unsqueeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            t = y[:, 0].cpu().numpy()[0]
            p = pred.cpu().numpy()[0]
            tfg, pfg = t > 0, p > 0
            den = tfg.sum() + pfg.sum()
            dice = float(2 * np.logical_and(tfg, pfg).sum() / den) if den else 1.0

        losses.append(float(loss.item()))
        dices.append(dice)
        del x, y, logits, loss, pred

    return {"loss": float(np.mean(losses)), "foreground_dice": float(np.mean(dices))}


def save_checkpoint(model, optimizer, epoch, condition, lr):
    path = OUT / condition / "checkpoints" / f"epoch_{epoch:02d}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "condition": condition,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "learning_rate": lr,
    }, path)
    return path


def main():
    seed_everything()

    print("=" * 82)
    print("PART 57 PATH VALIDATION")
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
    print("PART 57 — LEARNING-RATE / EARLY-CHECKPOINT STABILITY ABLATION")
    print("=" * 82)
    print(f"Train subset : {TRAIN_N}")
    print(f"Validation subset : {VAL_N}")
    print(f"Epochs : {EPOCHS}")
    print(f"Full volume : {FULL_SHAPE}")
    print(f"Crop : {CROP_SHAPE}")
    print(f"Pseudo-mask radius : R{RADIUS}")
    print("Crop strategy : foreground-centered")
    print("Loss : original Part 15 DiceCELoss")
    print(f"Weight decay : {WD}")
    print("A : constant LR 1e-4")
    print("B : constant LR 5e-5")
    print("C : 1e-4 -> 5e-5 after epoch 2")
    print("SPIDER : NO")
    print("Test set : NO")
    print("Part 15 overwritten : NO")
    print(f"Initialization SHA256 : {sha256(INIT)}")

    part11 = load_module(PART11_PATH, "part11_part57_runtime")
    part9 = load_module(PART9_PATH, "part9_part57_runtime")

    train_df = pd.read_csv(TRCSV).head(TRAIN_N)
    val_df = pd.read_csv(VACSV).head(VAL_N)

    train_cases = preload(part11, part9, train_df, "train")
    val_cases = preload(part11, part9, val_df, "validation")

    print()
    print("=" * 82)
    print("PART 57 SHAPE / LABEL SMOKE TEST")
    print("=" * 82)
    print(f"Image : {train_cases[0]['image'].shape}")
    print(f"Mask : {train_cases[0]['mask'].shape}")
    print(f"Labels : {sorted(np.unique(train_cases[0]['mask']).tolist())}")
    assert train_cases[0]["image"].shape == CROP_SHAPE
    assert train_cases[0]["mask"].shape == CROP_SHAPE
    assert train_cases[0]["mask"].min() >= 0
    assert train_cases[0]["mask"].max() < NUM_CLASSES
    print("✓ Shape / label smoke test PASSED.")

    train_fg = np.mean([(c["mask"] > 0).sum() for c in train_cases])
    val_fg = np.mean([(c["mask"] > 0).sum() for c in val_cases])

    print()
    print("=" * 82)
    print("PART 57 R2 TARGET DENSITY")
    print("=" * 82)
    print(f"Train mean FG : {train_fg:.2f}")
    print(f"Train occupancy : {train_fg / np.prod(CROP_SHAPE):.6f}")
    print(f"Validation mean FG : {val_fg:.2f}")
    print(f"Validation occupancy : {val_fg / np.prod(CROP_SHAPE):.6f}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print()
    print(f"PyTorch : {torch.__version__}")
    print(f"Device : {device}")

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    records = []

    for condition, cfg in CONDITIONS.items():
        print()
        print("=" * 82)
        print(f"PART 57 CONDITION: {condition}")
        print("=" * 82)

        model = part11.create_model(device)
        state = torch.load(INIT, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["initial_lr"],
            weight_decay=WD,
        )

        init = evaluate(model, val_cases, device, loss_fn)
        print(
            f"INIT | val_loss={init['loss']:.6f} "
            f"val_FGDice={init['foreground_dice']:.6f} "
            f"PredFG={init['predicted_foreground_voxels']:.1f} "
            f"FGProb={init['foreground_probability']:.6f}"
        )

        records.append({
            "condition": condition,
            "epoch": 0,
            "learning_rate": cfg["initial_lr"],
            "train_loss": None,
            "train_FGDice": None,
            "val_loss": init["loss"],
            "val_FGDice": init["foreground_dice"],
            "predicted_foreground_voxels": init["predicted_foreground_voxels"],
            "target_foreground_voxels": init["target_foreground_voxels"],
            "foreground_probability": init["foreground_probability"],
            "background_probability": init["background_probability"],
            "probability_gap": init["probability_gap"],
            "prediction_target_ratio": init["prediction_target_ratio"],
            "empty_cases": init["empty_cases"],
            "total_cases": init["total_cases"],
            "checkpoint": str(INIT),
        })

        for epoch in range(1, EPOCHS + 1):
            start = time.time()

            if cfg["schedule"] == "step" and epoch == 3:
                for group in optimizer.param_groups:
                    group["lr"] = 5e-5

            lr = float(optimizer.param_groups[0]["lr"])

            tr = train_epoch(
                model, optimizer, loss_fn, train_cases, device
            )
            va = evaluate(
                model, val_cases, device, loss_fn
            )

            checkpoint = save_checkpoint(
                model, optimizer, epoch, condition, lr
            )

            elapsed = time.time() - start

            print(
                f"Epoch {epoch:02d} | LR={lr:.2e} | "
                f"train_loss={tr['loss']:.6f} "
                f"train_FGDice={tr['foreground_dice']:.6f} | "
                f"val_loss={va['loss']:.6f} "
                f"val_FGDice={va['foreground_dice']:.6f} | "
                f"PredFG={va['predicted_foreground_voxels']:.1f} "
                f"TargetFG={va['target_foreground_voxels']:.1f} "
                f"FGProb={va['foreground_probability']:.6f} "
                f"Empty={va['empty_cases']}/{VAL_N} | "
                f"{elapsed:.1f}s"
            )

            records.append({
                "condition": condition,
                "epoch": epoch,
                "learning_rate": lr,
                "train_loss": tr["loss"],
                "train_FGDice": tr["foreground_dice"],
                "val_loss": va["loss"],
                "val_FGDice": va["foreground_dice"],
                "predicted_foreground_voxels": va["predicted_foreground_voxels"],
                "target_foreground_voxels": va["target_foreground_voxels"],
                "foreground_probability": va["foreground_probability"],
                "background_probability": va["background_probability"],
                "probability_gap": va["probability_gap"],
                "prediction_target_ratio": va["prediction_target_ratio"],
                "empty_cases": va["empty_cases"],
                "total_cases": va["total_cases"],
                "checkpoint": str(checkpoint),
            })

        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 82)
    print("PART 57 BEST-CHECKPOINT ANALYSIS")
    print("=" * 82)

    summaries = []

    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        post_init = [r for r in rows if r["epoch"] >= 1]
        best = max(post_init, key=lambda r: r["val_FGDice"])
        final = max(rows, key=lambda r: r["epoch"])

        summaries.append({
            "condition": condition,
            "best_epoch": best["epoch"],
            "best_val_FGDice": best["val_FGDice"],
            "best_val_loss": best["val_loss"],
            "best_predicted_foreground_voxels": best["predicted_foreground_voxels"],
            "final_epoch": final["epoch"],
            "final_val_FGDice": final["val_FGDice"],
            "final_val_loss": final["val_loss"],
            "final_predicted_foreground_voxels": final["predicted_foreground_voxels"],
            "final_empty_cases": final["empty_cases"],
            "best_checkpoint": best["checkpoint"],
            "final_checkpoint": final["checkpoint"],
        })

        print(
            f"{condition:<24} | "
            f"Best E{best['epoch']} Dice={best['val_FGDice']:.6f} "
            f"PredFG={best['predicted_foreground_voxels']:.1f} | "
            f"Final E{final['epoch']} Dice={final['val_FGDice']:.6f} "
            f"PredFG={final['predicted_foreground_voxels']:.1f} "
            f"Empty={final['empty_cases']}/{VAL_N}"
        )

    overall_best = max(
        summaries,
        key=lambda x: x["best_val_FGDice"]
    )

    baseline = next(
        x for x in summaries
        if x["condition"] == "A_constant_1e4"
    )

    if any(
        x["final_predicted_foreground_voxels"]
        > baseline["final_predicted_foreground_voxels"]
        for x in summaries
        if x["condition"] != baseline["condition"]
    ):
        diagnosis = "LEARNING_RATE_SCHEDULE_SHOWS_POTENTIAL_COLLAPSE_DELAY"
    elif overall_best["best_val_FGDice"] > baseline["best_val_FGDice"]:
        diagnosis = "LEARNING_RATE_CHANGES_IMPROVED_PEAK_FOREGROUND_DICE"
    else:
        diagnosis = "NO_MEANINGFUL_LEARNING_RATE_STABILITY_GAIN"

    print()
    print(f"Overall best condition : {overall_best['condition']}")
    print(f"Overall best epoch : {overall_best['best_epoch']}")
    print(f"Overall best FGDice : {overall_best['best_val_FGDice']:.6f}")
    print(f"Diagnosis : {diagnosis}")

    REPORT.mkdir(parents=True, exist_ok=True)

    trajectory_csv = REPORT / "part57_learning_rate_trajectory.csv"
    fields = [
        "condition", "epoch", "learning_rate", "train_loss",
        "train_FGDice", "val_loss", "val_FGDice",
        "predicted_foreground_voxels", "target_foreground_voxels",
        "foreground_probability", "background_probability",
        "probability_gap", "prediction_target_ratio",
        "empty_cases", "total_cases", "checkpoint"
    ]

    with open(trajectory_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    summary_csv = REPORT / "part57_best_checkpoint_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fields2 = list(summaries[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields2)
        writer.writeheader()
        writer.writerows(summaries)

    summary_json = REPORT / "part57_summary.json"
    payload = {
        "part": 57,
        "purpose": "Learning-rate and early-checkpoint stability ablation after Part 56.",
        "configuration": {
            "train_cases": TRAIN_N,
            "validation_cases": VAL_N,
            "epochs": EPOCHS,
            "full_shape": FULL_SHAPE,
            "crop_shape": CROP_SHAPE,
            "pseudo_mask_radius": RADIUS,
            "crop_strategy": "foreground-centered",
            "loss": "original Part 15 DiceCELoss",
            "weight_decay": WD,
            "spider": False,
            "test_set": False,
            "part15_modified": False,
            "initialization_sha256": sha256(INIT),
            "conditions": CONDITIONS,
        },
        "trajectory": records,
        "best_checkpoint_summaries": summaries,
        "overall_best": overall_best,
        "diagnosis": diagnosis,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print()
    print("=" * 82)
    print("PART 57 COMPLETE")
    print("=" * 82)
    print(f"Output directory : {OUT}")
    print(f"Trajectory CSV : {trajectory_csv}")
    print(f"Best-checkpoint CSV : {summary_csv}")
    print(f"Summary JSON : {summary_json}")


if __name__ == "__main__":
    main()
