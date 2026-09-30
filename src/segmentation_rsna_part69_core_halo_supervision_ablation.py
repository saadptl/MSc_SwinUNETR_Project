from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# PART 69
# Controlled pseudo-mask supervision ablation: R2 FULL vs R2 CORE vs R2 HALO
# ============================================================================
# Purpose:
#   Test whether the boundary/halo portion of the R2 pseudo-mask is useful or
#   harmful as a training target, under otherwise identical conditions.
#
# This is a methodological sensitivity experiment. It does NOT establish that
# any region is medically correct/incorrect because RSNA annotations are point
# annotations rather than manual segmentation masks.
#
# Locked controls:
#   - RSNA 2024 Lumbar Spine Degenerative Classification only
#   - first 100 Part 15 training cases / first 50 validation cases
#   - exact Part 15 initialization checkpoint
#   - SwinUNETR 3D, 1 input channel, 6 output classes, feature_size=12
#   - full preprocessing shape (64,96,96)
#   - centered crop (32,64,64)
#   - original DiceCELoss
#   - batch size 1
#   - learning rate 1e-4
#   - 3 epochs
#   - same optimizer/seed/configuration for every condition
#
# Conditions:
#   R2_FULL : original R2 pseudo-mask
#   R2_CORE : one-voxel 6-connected erosion of R2 foreground
#   R2_HALO : R2 foreground minus R2_CORE
#
# IMPORTANT:
#   R2_HALO is included as a diagnostic supervision condition, not as a
#   proposed final segmentation target.
# ============================================================================

ROOT = Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
SRC = ROOT / "src"
P11 = SRC / "segmentation_rsna_part11_controlled_pilot_training.py"
P9 = SRC / "segmentation_rsna_part9_3d_dataset_loader.py"

P15 = ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
TRAIN_CSV = P15 / "part15_train_cohort.csv"
VAL_CSV = P15 / "part15_validation_cohort.csv"
INIT_CKPT = P15 / "checkpoints" / "part15_initialization_from_part11.pth"

OUT = ROOT / "outputs" / "segmentation" / "rsna_part69_core_halo_supervision_ablation"
REPORT = OUT / "reports"
CKPT_DIR = OUT / "checkpoints"

N_TRAIN = 100
N_VAL = 50
FULL = (64, 96, 96)
CROP = (32, 64, 64)
EPOCHS = 3
BATCH_SIZE = 1
LR = 1e-4
SEED = 42

CLASSES = {
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def banner(text: str):
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def validate_paths():
    required = {
        "Project root": ROOT,
        "Part 11": P11,
        "Part 9": P9,
        "Part 15 train cohort": TRAIN_CSV,
        "Part 15 validation cohort": VAL_CSV,
        "Part 15 initialization": INIT_CKPT,
    }
    banner("PART 69 PATH VALIDATION")
    missing = []
    for name, path in required.items():
        ok = path.exists()
        print(f"{name:<34}: {'FOUND' if ok else 'MISSING'}")
        if not ok:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Required input(s) missing:\n" + "\n".join(missing))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_tensor_image(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.float()
    while x.ndim > 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D image after squeeze, got {tuple(x.shape)}")
    return x


def as_mask(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    while x.ndim > 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D mask after squeeze, got {x.shape}")
    return x.astype(np.int64)


def centered_crop(arr: np.ndarray, center: np.ndarray, crop_shape=CROP):
    arr = np.asarray(arr)
    starts = []
    for dim, c, size in zip(arr.shape, center, crop_shape):
        start = int(round(float(c) - size / 2.0))
        start = max(0, min(start, dim - size))
        starts.append(start)
    z0, y0, x0 = starts
    crop = arr[z0:z0+crop_shape[0], y0:y0+crop_shape[1], x0:x0+crop_shape[2]]
    if crop.shape != crop_shape:
        raise ValueError(f"Crop shape {crop.shape} != {crop_shape}")
    return crop, tuple(starts)


def dilate_labels(mask: np.ndarray, radius: int) -> np.ndarray:
    """Label-preserving Chebyshev dilation, matching the established Part 52/68 logic."""
    mask = np.asarray(mask, dtype=np.int64)
    if radius == 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=np.int64)
    for c in range(1, 6):
        binary = mask == c
        if not binary.any():
            continue
        d = ndimage.binary_dilation(binary, iterations=radius, structure=np.ones((3, 3, 3), dtype=bool))
        # Preserve existing earlier classes if overlap occurs; process by class order.
        out[(d) & (out == 0)] = c
    # Original labelled foreground is always retained.
    out[mask > 0] = mask[mask > 0]
    return out


def make_r2_regions(raw_mask: np.ndarray):
    resized = p11.resize_3d(raw_mask, FULL, is_mask=True)
    r2 = dilate_labels(resized, 2)
    binary = r2 > 0
    structure = ndimage.generate_binary_structure(3, 1)
    core_binary = ndimage.binary_erosion(binary, structure=structure, iterations=1, border_value=0)
    core = np.where(core_binary, r2, 0).astype(np.int64)
    halo_binary = binary & (~core_binary)
    halo = np.where(halo_binary, r2, 0).astype(np.int64)
    return {"R2_FULL": r2.astype(np.int64), "R2_CORE": core, "R2_HALO": halo}


def load_initial_state():
    obj = torch.load(INIT_CKPT, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if isinstance(obj, dict):
        # Raw state_dict is normally a dict of parameter tensors.
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Could not locate a model state_dict in the Part 15 initialization checkpoint.")


def normalize_state_keys(state):
    if not state:
        return state
    keys = list(state.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def create_model():
    model = p11.create_model(DEVICE)
    state = normalize_state_keys(load_initial_state())
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Part 15 initialization did not load cleanly. "
            f"Missing={missing[:10]} Unexpected={unexpected[:10]}"
        )
    model.to(DEVICE)
    return model


def dice_foreground(logits: torch.Tensor, target: torch.Tensor):
    pred = torch.argmax(logits, dim=1)
    target = target[:, 0]
    dices = []
    pred_counts = []
    target_counts = []
    for c in range(1, 6):
        p = pred == c
        t = target == c
        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        dices.append((2.0 * inter / denom) if denom else 1.0)
        pred_counts.append(int(p.sum().item()))
        target_counts.append(int(t.sum().item()))
    return float(np.mean(dices)), dices, pred_counts, target_counts


class CaseDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, condition: str, split: str):
        self.rows = rows.reset_index(drop=True)
        self.condition = condition
        self.split = split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        loaded = p11.load_tensor_case(row, p9)
        image = as_tensor_image(loaded[0])
        raw_mask = as_mask(loaded[1])
        regions = make_r2_regions(raw_mask)
        r2 = regions["R2_FULL"]
        q = np.argwhere(r2 > 0)
        center = q.mean(axis=0) if len(q) else np.array([31.5, 47.5, 47.5])
        image_crop, starts = centered_crop(image.numpy(), center)
        target_crop, _ = centered_crop(regions[self.condition], center)

        image_crop = torch.from_numpy(image_crop.astype(np.float32)).unsqueeze(0)
        target_crop = torch.from_numpy(target_crop.astype(np.int64)).unsqueeze(0)

        return {
            "image": image_crop,
            "target": target_crop,
            "case_no": int(idx + 1),
            "condition": self.condition,
            "fg_voxels": int((target_crop > 0).sum().item()),
        }


def make_loader(rows, condition, split, shuffle):
    ds = CaseDataset(rows, condition, split)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=False)


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total_fg = 0
    n = 0
    for batch in loader:
        image = batch["image"].to(DEVICE, non_blocking=False)
        target = batch["target"].to(DEVICE, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = loss_fn(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_fg += int(batch["fg_voxels"][0])
        n += 1
        del logits, loss, image, target
    return total_loss / max(n, 1), total_fg / max(n, 1)


def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    dice_values = []
    class_dices = [[] for _ in range(5)]
    pred_fg = 0
    target_fg = 0
    empty = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            logits = model(image)
            loss = loss_fn(logits, target)
            d, ds, pcs, tcs = dice_foreground(logits, target)
            total_loss += float(loss.item())
            dice_values.append(d)
            for c in range(5):
                class_dices[c].append(ds[c])
            pf = sum(pcs)
            pred_fg += pf
            target_fg += sum(tcs)
            if pf == 0:
                empty += 1
            n += 1
            del logits, loss, image, target
    return {
        "val_loss": total_loss / max(n, 1),
        "val_foreground_dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "val_class_dice": [float(np.mean(x)) if x else 0.0 for x in class_dices],
        "val_pred_fg_mean_per_case": pred_fg / max(n, 1),
        "val_target_fg_mean_per_case": target_fg / max(n, 1),
        "val_empty_prediction_cases": empty,
        "val_cases": n,
    }


def save_checkpoint(model, optimizer, condition, epoch, metrics):
    path = CKPT_DIR / f"part69_{condition.lower()}_epoch{epoch}.pth"
    torch.save(
        {
            "part": 69,
            "condition": condition,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "seed": SEED,
            "learning_rate": LR,
            "train_cases": N_TRAIN,
            "val_cases": N_VAL,
            "initialization_checkpoint": str(INIT_CKPT),
            "initialization_sha256": sha256_file(INIT_CKPT),
        },
        path,
    )
    return path


def main():
    global p11, p9
    validate_paths()
    REPORT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    p11 = load_module(P11, "part11_part69")
    p9 = load_module(P9, "part9_part69")

    train_rows = pd.read_csv(TRAIN_CSV).head(N_TRAIN).copy()
    val_rows = pd.read_csv(VAL_CSV).head(N_VAL).copy()

    banner("PART 69 LOCKED EXPERIMENT")
    print(f"Device                     : {DEVICE}")
    print(f"Train cases                : {len(train_rows)}")
    print(f"Validation cases           : {len(val_rows)}")
    print(f"Full shape                 : {FULL}")
    print(f"Centered crop              : {CROP}")
    print(f"Epochs                     : {EPOCHS}")
    print(f"Learning rate              : {LR}")
    print(f"Batch size                 : {BATCH_SIZE}")
    print(f"Loss                       : MONAI DiceCELoss (Part 11 configuration)")
    print(f"Initialization             : {INIT_CKPT}")
    print(f"Initialization SHA256      : {sha256_file(INIT_CKPT)}")
    print("Conditions                 : R2_FULL / R2_CORE / R2_HALO")

    # Part 11 uses its established DiceCELoss configuration internally in the
    # training implementation. Reuse the exact loss object/configuration when
    # available; otherwise fail loudly rather than silently changing the loss.
    try:
        from monai.losses import DiceCELoss
    except Exception as e:
        raise ImportError("MONAI DiceCELoss is required for Part 69.") from e
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)

    all_history = []
    best_rows = []
    condition_summary = []

    for condition in ["R2_FULL", "R2_CORE", "R2_HALO"]:
        banner(f"PART 69 CONDITION: {condition}")
        set_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        # Fresh loaders for each condition, same rows and same crop geometry.
        train_loader = make_loader(train_rows, condition, "train", shuffle=True)
        val_loader = make_loader(val_rows, condition, "validation", shuffle=False)

        # Record initial validation state before any update.
        init_metrics = validate(model, val_loader, loss_fn)
        init_record = {
            "condition": condition,
            "epoch": 0,
            "train_loss": np.nan,
            "train_target_fg_mean_per_case": np.nan,
            **init_metrics,
        }
        all_history.append(init_record)
        print(
            f"INIT | val_loss={init_metrics['val_loss']:.6f} "
            f"FGDice={init_metrics['val_foreground_dice']:.6f} "
            f"PredFG={init_metrics['val_pred_fg_mean_per_case']:.1f} "
            f"Empty={init_metrics['val_empty_prediction_cases']}/{N_VAL}"
        )

        best_dice = init_metrics["val_foreground_dice"]
        best_epoch = 0
        best_ckpt = None

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_fg = train_epoch(model, train_loader, optimizer, loss_fn)
            val_metrics = validate(model, val_loader, loss_fn)
            record = {
                "condition": condition,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_target_fg_mean_per_case": train_fg,
                **val_metrics,
            }
            all_history.append(record)

            ckpt = save_checkpoint(model, optimizer, condition, epoch, record)
            if val_metrics["val_foreground_dice"] > best_dice:
                best_dice = val_metrics["val_foreground_dice"]
                best_epoch = epoch
                best_ckpt = str(ckpt)

            print(
                f"E{epoch} | train_loss={train_loss:.6f} "
                f"trainFG={train_fg:.1f} "
                f"val_loss={val_metrics['val_loss']:.6f} "
                f"FGDice={val_metrics['val_foreground_dice']:.6f} "
                f"PredFG={val_metrics['val_pred_fg_mean_per_case']:.1f} "
                f"Empty={val_metrics['val_empty_prediction_cases']}/{N_VAL}"
            )

        # Save final model and best model separately.
        final_path = CKPT_DIR / f"part69_{condition.lower()}_final.pth"
        torch.save(
            {
                "part": 69,
                "condition": condition,
                "epoch": EPOCHS,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "seed": SEED,
                "learning_rate": LR,
                "initialization_checkpoint": str(INIT_CKPT),
                "initialization_sha256": sha256_file(INIT_CKPT),
            },
            final_path,
        )

        if best_ckpt is None:
            best_ckpt = str(final_path) if EPOCHS else None

        best_rows.append(
            {
                "condition": condition,
                "best_epoch": best_epoch,
                "best_val_foreground_dice": best_dice,
                "best_checkpoint": best_ckpt,
                "final_checkpoint": str(final_path),
            }
        )

        # Mean target foreground across the actual training dataset.
        train_fg_values = []
        for i in range(min(len(train_rows), N_TRAIN)):
            # Dataset construction is deterministic but we do not need to run
            # inference again; collect from one non-shuffled pass.
            pass
        condition_summary.append(
            {
                "condition": condition,
                "best_epoch": best_epoch,
                "best_val_foreground_dice": best_dice,
                "final_val_foreground_dice": all_history[-1]["val_foreground_dice"],
                "final_val_pred_fg_mean_per_case": all_history[-1]["val_pred_fg_mean_per_case"],
                "final_val_empty_prediction_cases": all_history[-1]["val_empty_prediction_cases"],
            }
        )

        del train_loader, val_loader, optimizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    history_df = pd.DataFrame(all_history)
    best_df = pd.DataFrame(best_rows)
    summary_df = pd.DataFrame(condition_summary)

    history_df.to_csv(REPORT / "part69_learning_trajectory.csv", index=False)
    best_df.to_csv(REPORT / "part69_best_checkpoint_summary.csv", index=False)
    summary_df.to_csv(REPORT / "part69_condition_summary.csv", index=False)

    # Direct full-vs-core and full-vs-halo comparisons at each common epoch.
    pivot = history_df.pivot(index="epoch", columns="condition", values="val_foreground_dice").reset_index()
    for c in ["R2_CORE", "R2_HALO"]:
        if "R2_FULL" in pivot.columns and c in pivot.columns:
            pivot[f"FULL_minus_{c}_dice"] = pivot["R2_FULL"] - pivot[c]
    pivot.to_csv(REPORT / "part69_full_core_halo_dice_comparison.csv", index=False)

    # Identify the best condition by best checkpoint, not by a single arbitrary
    # final epoch.
    best_condition_row = best_df.loc[best_df.best_val_foreground_dice.idxmax()]
    full_best = float(best_df.loc[best_df.condition == "R2_FULL", "best_val_foreground_dice"].iloc[0])
    core_best = float(best_df.loc[best_df.condition == "R2_CORE", "best_val_foreground_dice"].iloc[0])
    halo_best = float(best_df.loc[best_df.condition == "R2_HALO", "best_val_foreground_dice"].iloc[0])

    # Conservative interpretation rule: call a condition a meaningful winner
    # only if the absolute best-Dice gain over R2_FULL is >= 0.02. Otherwise the
    # result is treated as no large supervision advantage in this small pilot.
    core_delta = core_best - full_best
    halo_delta = halo_best - full_best
    if core_delta >= 0.02:
        diagnosis = "R2_CORE_SHOWED_MEANINGFUL_GAIN_OVER_FULL_R2"
    elif halo_delta >= 0.02:
        diagnosis = "R2_HALO_SHOWED_MEANINGFUL_GAIN_OVER_FULL_R2"
    elif abs(core_delta) < 0.02 and abs(halo_delta) < 0.02:
        diagnosis = "NO_LARGE_CORE_OR_HALO_SUPERVISION_ADVANTAGE"
    elif core_delta <= -0.02 and halo_delta <= -0.02:
        diagnosis = "FULL_R2_OUTPERFORMED_BOTH_CORE_AND_HALO"
    else:
        diagnosis = "SUPERVISION_ABLATION_SHOWED_MIXED_EFFECTS"

    summary = {
        "part": 69,
        "purpose": "Controlled pseudo-mask supervision ablation: R2 full vs R2 core vs R2 halo",
        "train_cases": len(train_rows),
        "validation_cases": len(val_rows),
        "full_shape": list(FULL),
        "crop_shape": list(CROP),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "seed": SEED,
        "device": str(DEVICE),
        "initialization_checkpoint": str(INIT_CKPT),
        "initialization_sha256": sha256_file(INIT_CKPT),
        "conditions": ["R2_FULL", "R2_CORE", "R2_HALO"],
        "best_val_foreground_dice": {
            "R2_FULL": full_best,
            "R2_CORE": core_best,
            "R2_HALO": halo_best,
        },
        "core_minus_full_best_dice": core_delta,
        "halo_minus_full_best_dice": halo_delta,
        "best_condition": str(best_condition_row["condition"]),
        "diagnosis": diagnosis,
        "medical_interpretation_limit": "RSNA coordinate annotations are point annotations, not manual segmentation masks; this is a pseudo-mask sensitivity experiment.",
    }
    (REPORT / "part69_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "PART 69 — CORE/HALO PSEUDO-MASK SUPERVISION ABLATION",
        "",
        f"Device: {DEVICE}",
        f"Train cases: {len(train_rows)}",
        f"Validation cases: {len(val_rows)}",
        f"Initialization SHA256: {sha256_file(INIT_CKPT)}",
        f"Crop: {CROP}",
        f"Epochs: {EPOCHS}",
        f"Learning rate: {LR}",
        "",
        "BEST VALIDATION FOREGROUND DICE",
        f"R2_FULL : {full_best:.6f}",
        f"R2_CORE : {core_best:.6f}  (delta vs full = {core_delta:+.6f})",
        f"R2_HALO : {halo_best:.6f}  (delta vs full = {halo_delta:+.6f})",
        "",
        f"Diagnosis: {diagnosis}",
        "",
        "Interpretation limit: R2_CORE/R2_HALO are morphological partitions of the pseudo-mask. Low or high performance does not prove medical correctness or incorrectness of those regions.",
    ]
    (REPORT / "part69_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    banner("PART 69 FINAL SUMMARY")
    print(f"R2_FULL best Dice : {full_best:.6f}")
    print(f"R2_CORE best Dice : {core_best:.6f}  delta={core_delta:+.6f}")
    print(f"R2_HALO best Dice : {halo_best:.6f}  delta={halo_delta:+.6f}")
    print(f"Best condition    : {best_condition_row['condition']}")
    print(f"Diagnosis         : {diagnosis}")
    print("\nReports:")
    print(REPORT / "part69_learning_trajectory.csv")
    print(REPORT / "part69_best_checkpoint_summary.csv")
    print(REPORT / "part69_condition_summary.csv")
    print(REPORT / "part69_full_core_halo_dice_comparison.csv")
    print(REPORT / "part69_summary.json")
    print(REPORT / "part69_report.txt")
    print("\nCheckpoints:")
    print(CKPT_DIR)


if __name__ == "__main__":
    main()
