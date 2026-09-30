"""
PART 77
RSNA-ONLY C2/C3 SPATIAL-CONSISTENCY SUPERVISION PILOT

Purpose
-------
Part 75/74 identified weak or spatially misplaced C2/C3 activation.
Part 76 showed that direct RSNA point auxiliary supervision changed
localization behavior but did not improve overall segmentation Dice.

Part 77 therefore tests a different hypothesis:

    Can a differentiable C2/C3 spatial-consistency loss, based on the
    existing R2_FULL pseudo-mask geometry, improve C2/C3 localization
    while preserving the normal segmentation objective?

Controlled comparison
---------------------
A) R2_FULL_CLASS_BALANCED:
   - exact Part 15 initialization
   - R2_FULL pseudo-mask
   - centered (32,64,64) crop
   - Part 71 class-balanced DiceCE

B) R2_FULL_CLASS_BALANCED_SPATIAL:
   - exact same initialization/data/model/optimizer
   - same class-balanced DiceCE
   - PLUS differentiable C2/C3 soft-centroid consistency loss

IMPORTANT
---------
The spatial target is derived from the existing R2 pseudo-mask.
RSNA coordinate annotations are NOT treated as segmentation ground truth.

Locked:
- 100 train / 50 validation
- full shape (64,96,96)
- centered R2 crop (32,64,64)
- SwinUNETR feature_size=12 via Part 11
- batch size 1
- lr 1e-4
- weight decay 1e-5
- 3 epochs
- seed 42
- exact Part 15 initialization
- no SPIDER
- no RSNA test set

Spatial loss
------------
For C2 and C3 independently:
1. Take the softmax probability volume for the class.
2. Compute its differentiable probability-weighted centroid.
3. Compute the centroid of the corresponding R2_FULL target region.
4. Penalize normalized squared centroid distance.

This is a localization regularizer, not a replacement segmentation loss.

The default lambda is deliberately conservative (0.10) so that the
experiment tests spatial guidance without overwhelming DiceCE.
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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ============================================================================
# PATHS
# ============================================================================
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

P11 = SRC / "segmentation_rsna_part11_controlled_pilot_training.py"
P9 = SRC / "segmentation_rsna_part9_3d_dataset_loader.py"

# Part 15 locked cohort + initialization.
P15 = ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
TRAIN_CSV = P15 / "part15_train_cohort.csv"
VAL_CSV = P15 / "part15_validation_cohort.csv"
INIT_CKPT = P15 / "checkpoints" / "part15_initialization_from_part11.pth"

# RSNA coordinate file is loaded only for localization reporting.
RSNA = ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification"
COORD = RSNA / "train_label_coordinates.csv"

# Reuse the validated Part 76 implementation for exact data/mask loading.
PART76_CANDIDATES = [
    SRC / "segmentation_rsna_part76_c2_c3_point_auxiliary_supervision.py",
    SRC / "segmentation_rsna_part76_c2_c3_point_auxiliary_supervision_FIXED_v2.py",
    SRC / "segmentation_rsna_part76_c2_c3_point_auxiliary_supervision_REPORTING_FIXED.py",
]

OUT = ROOT / "outputs" / "segmentation" / "rsna_part77_c2_c3_spatial_consistency_supervision"
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
BATCH_SIZE = 1
EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42

FOCUS_CLASSES = (2, 3)

# Conservative localization regularization.
SPATIAL_LAMBDA = 0.10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASSES = {
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}

# Same Part 71/76 class-balanced foreground weights.
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


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_paths(part76) -> None:
    banner("PART 77 PATH VALIDATION")

    required = {
        "Project root": ROOT,
        "Part 11": P11,
        "Part 9": P9,
        "Part 15 train cohort": TRAIN_CSV,
        "Part 15 validation cohort": VAL_CSV,
        "Part 15 initialization": INIT_CKPT,
        "RSNA coordinate CSV": COORD,
        "Part 76 source": Path(part76.__file__),
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
# TARGET SPATIAL CENTROIDS
# ============================================================================
def target_centroid(target: torch.Tensor, class_id: int):
    """
    Returns normalized centroid of a target class.

    target: [D,H,W], integer labels.
    normalized coordinates are in [0,1].
    """
    coords = torch.nonzero(target == class_id, as_tuple=False)

    if coords.numel() == 0:
        return None

    coords = coords.float()
    denom = torch.tensor(
        [max(target.shape[0] - 1, 1),
         max(target.shape[1] - 1, 1),
         max(target.shape[2] - 1, 1)],
        dtype=torch.float32,
        device=target.device,
    )

    return (coords.mean(dim=0) / denom).to(target.device)


def prediction_soft_centroid(
    probabilities: torch.Tensor,
    class_id: int,
):
    """
    Differentiable probability-weighted centroid.

    probabilities: [D,H,W] softmax probability for one class.
    """
    d, h, w = probabilities.shape
    dtype = probabilities.dtype
    device = probabilities.device

    z = torch.linspace(0.0, 1.0, d, device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    mass = probabilities.sum()
    denom = mass + 1e-6

    return torch.stack(
        [
            (probabilities * zz).sum() / denom,
            (probabilities * yy).sum() / denom,
            (probabilities * xx).sum() / denom,
        ]
    )


def spatial_consistency_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    C2/C3 soft-centroid consistency.

    The loss is differentiable with respect to model probabilities and
    compares predicted probability centroids to the corresponding R2_FULL
    target centroids.

    Returns:
        loss
        diagnostics dict
    """
    probs = torch.softmax(logits, dim=1)[0]

    losses = []
    distances = {}
    valid_classes = []

    target_cpu = target.detach().cpu().numpy()

    for c in FOCUS_CLASSES:
        if not np.any(target_cpu == c):
            continue

        tc = target_centroid(target, c)
        pc = prediction_soft_centroid(probs[c], c)

        diff = pc - tc
        sq = (diff * diff).mean()
        losses.append(sq)

        distances[c] = float(torch.sqrt((diff * diff).sum()).detach().cpu())
        valid_classes.append(c)

    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "valid_classes": 0,
            "c2_centroid_distance": np.nan,
            "c3_centroid_distance": np.nan,
        }

    loss = torch.stack(losses).mean()

    return loss, {
        "valid_classes": len(valid_classes),
        "c2_centroid_distance": distances.get(2, np.nan),
        "c3_centroid_distance": distances.get(3, np.nan),
    }


# ============================================================================
# METRICS
# ============================================================================
def class_metrics(pred: np.ndarray, target: np.ndarray, c: int):
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


# ============================================================================
# TRAINING / EVALUATION
# ============================================================================
def create_loss():
    from monai.losses import DiceCELoss

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


def evaluate_model(model, loss_fn, rows, part9, part11, condition):
    model.eval()

    losses = []
    class_rows = []
    case_rows = []
    spatial_rows = []

    with torch.no_grad():
        for case_no, (_, row) in enumerate(rows.iterrows(), start=1):
            image, target, _, _ = part76.load_case(
                part11,
                part9,
                row,
                pd.read_csv(COORD),
            )

            x = image.unsqueeze(0).unsqueeze(0).to(DEVICE)
            y = target.unsqueeze(0).unsqueeze(0).to(DEVICE)

            logits = model(x)
            loss = loss_fn(logits, y)

            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
            tgt = target.numpy()

            losses.append(float(loss.cpu()))

            case_rec = {
                "condition": condition,
                "case_no": case_no,
                "study_id": str(row["study_id"]),
                "target_foreground_voxels": int((tgt > 0).sum()),
                "predicted_foreground_voxels": int((pred > 0).sum()),
                "empty_foreground_prediction": int((pred > 0).sum() == 0),
            }

            for c in range(1, NUM_CLASSES):
                m = class_metrics(pred, tgt, c)
                class_rows.append({
                    "condition": condition,
                    "case_no": case_no,
                    "class_id": c,
                    "class_name": CLASSES[c],
                    **m,
                })
                for k, v in m.items():
                    case_rec[f"class_{c}_{k}"] = v

            # Evaluation-only spatial distances using argmax probabilities.
            probs = torch.softmax(logits, dim=1)[0]
            for c in FOCUS_CLASSES:
                tc = target_centroid(target.to(DEVICE), c)
                if tc is None:
                    continue

                pc = prediction_soft_centroid(probs[c], c)
                dist = float(torch.sqrt(((pc - tc) ** 2).sum()).cpu())

                spatial_rows.append({
                    "condition": condition,
                    "case_no": case_no,
                    "study_id": str(row["study_id"]),
                    "class_id": c,
                    "class_name": CLASSES[c],
                    "target_centroid_z": float(tc[0].cpu()),
                    "target_centroid_y": float(tc[1].cpu()),
                    "target_centroid_x": float(tc[2].cpu()),
                    "prediction_centroid_z": float(pc[0].cpu()),
                    "prediction_centroid_y": float(pc[1].cpu()),
                    "prediction_centroid_x": float(pc[2].cpu()),
                    "normalized_centroid_distance": dist,
                })

            case_rows.append(case_rec)

            if case_no % 10 == 0:
                print(f"  evaluated {case_no}/{len(rows)} cases", flush=True)

    class_df = pd.DataFrame(class_rows)
    case_df = pd.DataFrame(case_rows)
    spatial_df = pd.DataFrame(spatial_rows)

    class_summary = []
    for c in range(1, NUM_CLASSES):
        g = class_df[class_df.class_id == c]
        tp = int(g.tp.sum())
        fp = int(g.fp.sum())
        fn = int(g.fn.sum())
        pred_n = int(g.predicted_voxels.sum())
        target_n = int(g.target_voxels.sum())

        class_summary.append({
            "condition": condition,
            "class_id": c,
            "class_name": CLASSES[c],
            "target_voxels": target_n,
            "predicted_voxels": pred_n,
            "prediction_target_ratio": safe_div(pred_n, target_n),
            "dice": safe_div(2 * tp, pred_n + target_n),
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
            "iou": safe_div(tp, tp + fp + fn),
            "empty_cases": int(g.empty_prediction.sum()),
        })

    total_tp = sum(int(x["dice"] * (x["predicted_voxels"] + x["target_voxels"]) / 2)
                   for x in class_summary)
    total_pred = sum(x["predicted_voxels"] for x in class_summary)
    total_target = sum(x["target_voxels"] for x in class_summary)

    # Recompute exact global TP directly from class records.
    total_tp = int(class_df.tp.sum())

    summary = {
        "condition": condition,
        "loss": float(np.mean(losses)),
        "foreground_target_voxels": int(case_df.target_foreground_voxels.sum()),
        "foreground_predicted_voxels": int(case_df.predicted_foreground_voxels.sum()),
        "foreground_empty_cases": int(case_df.empty_foreground_prediction.sum()),
        "foreground_dice": safe_div(2 * total_tp, total_pred + total_target),
        "cases": len(case_df),
    }

    for c in FOCUS_CLASSES:
        g = spatial_df[spatial_df.class_id == c]
        summary[f"c{c}_mean_centroid_distance"] = (
            float(g.normalized_centroid_distance.mean())
            if len(g) else np.nan
        )

    return summary, pd.DataFrame(class_summary), class_df, case_df, spatial_df


def train_one_epoch(
    model,
    loss_fn,
    optimizer,
    rows,
    part9,
    part11,
    coord,
    use_spatial,
):
    model.train()

    total_losses = []
    seg_losses = []
    spatial_losses = []
    c2_distances = []
    c3_distances = []

    for case_no, (_, row) in enumerate(rows.iterrows(), start=1):
        image, target, _, _ = part76.load_case(
            part11,
            part9,
            row,
            coord,
        )

        x = image.unsqueeze(0).unsqueeze(0).to(DEVICE)
        y = target.unsqueeze(0).unsqueeze(0).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            enabled=DEVICE.type == "cuda",
        ):
            logits = model(x)
            seg_loss = loss_fn(logits, y)

            if use_spatial:
                loc_loss, diag = spatial_consistency_loss(
                    logits,
                    target.to(DEVICE),
                )
                total_loss = seg_loss + SPATIAL_LAMBDA * loc_loss
            else:
                loc_loss = logits.sum() * 0.0
                diag = {
                    "c2_centroid_distance": np.nan,
                    "c3_centroid_distance": np.nan,
                }
                total_loss = seg_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite total loss: {float(total_loss.detach().cpu())}"
            )

        total_loss.backward()
        optimizer.step()

        total_losses.append(float(total_loss.detach().cpu()))
        seg_losses.append(float(seg_loss.detach().cpu()))
        spatial_losses.append(float(loc_loss.detach().cpu()))

        if np.isfinite(diag["c2_centroid_distance"]):
            c2_distances.append(diag["c2_centroid_distance"])
        if np.isfinite(diag["c3_centroid_distance"]):
            c3_distances.append(diag["c3_centroid_distance"])

        if case_no % 20 == 0:
            print(f"  trained {case_no}/{len(rows)} cases", flush=True)

    return {
        "loss": float(np.mean(total_losses)),
        "seg_loss": float(np.mean(seg_losses)),
        "spatial_loss": float(np.mean(spatial_losses)),
        "mean_c2_centroid_distance": (
            float(np.mean(c2_distances)) if c2_distances else np.nan
        ),
        "mean_c3_centroid_distance": (
            float(np.mean(c3_distances)) if c3_distances else np.nan
        ),
    }


# ============================================================================
# CHECKPOINT
# ============================================================================
def save_checkpoint(model, optimizer, epoch, condition, val_fg_dice):
    path = CKPT / f"part77_{condition.lower()}_epoch{epoch}.pth"

    torch.save(
        {
            "part": 77,
            "epoch": epoch,
            "condition": condition,
            "val_foreground_dice": float(val_fg_dice),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": SEED,
            "device": str(DEVICE),
            "spatial_lambda": SPATIAL_LAMBDA,
            "focus_classes": list(FOCUS_CLASSES),
            "initialization_checkpoint": str(INIT_CKPT),
            "initialization_sha256": sha256_file(INIT_CKPT),
        },
        path,
    )

    return path


# ============================================================================
# RUN ONE CONDITION
# ============================================================================
def run_condition(
    condition,
    train_rows,
    val_rows,
    part9,
    part11,
    coord,
    initial_state,
):
    use_spatial = condition == "R2_FULL_CLASS_BALANCED_SPATIAL"

    banner(f"PART 77 CONDITION: {condition}")

    model = part76.create_model(part11)
    model.load_state_dict(initial_state, strict=True)

    loss_fn = create_loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    trajectory = []
    class_all = []
    case_all = []
    spatial_all = []

    t0 = time.time()
    init_summary, init_classes, _, init_cases, init_spatial = evaluate_model(
        model, loss_fn, val_rows, part9, part11, condition
    )

    trajectory.append({
        "condition": condition,
        "epoch": 0,
        "train_loss": np.nan,
        "train_seg_loss": np.nan,
        "train_spatial_loss": np.nan,
        "train_c2_centroid_distance": np.nan,
        "train_c3_centroid_distance": np.nan,
        "val_loss": init_summary["loss"],
        "val_fg_dice": init_summary["foreground_dice"],
        "val_pred_fg": init_summary["foreground_predicted_voxels"],
        "val_empty_fg": init_summary["foreground_empty_cases"],
        "val_c2_centroid_distance": init_summary.get("c2_mean_centroid_distance", np.nan),
        "val_c3_centroid_distance": init_summary.get("c3_mean_centroid_distance", np.nan),
        "elapsed_sec": time.time() - t0,
    })

    print(
        f"INIT | val_loss={init_summary['loss']:.6f} "
        f"FGDice={init_summary['foreground_dice']:.6f} "
        f"PredFG={init_summary['foreground_predicted_voxels']} "
        f"Empty={init_summary['foreground_empty_cases']}/{len(val_rows)}"
    )

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
            use_spatial,
        )

        val_summary, val_classes, _, val_cases, val_spatial = evaluate_model(
            model,
            loss_fn,
            val_rows,
            part9,
            part11,
            condition,
        )

        trajectory.append({
            "condition": condition,
            "epoch": epoch,
            "train_loss": train_info["loss"],
            "train_seg_loss": train_info["seg_loss"],
            "train_spatial_loss": train_info["spatial_loss"],
            "train_c2_centroid_distance": train_info["mean_c2_centroid_distance"],
            "train_c3_centroid_distance": train_info["mean_c3_centroid_distance"],
            "val_loss": val_summary["loss"],
            "val_fg_dice": val_summary["foreground_dice"],
            "val_pred_fg": val_summary["foreground_predicted_voxels"],
            "val_empty_fg": val_summary["foreground_empty_cases"],
            "val_c2_centroid_distance": val_summary.get("c2_mean_centroid_distance", np.nan),
            "val_c3_centroid_distance": val_summary.get("c3_mean_centroid_distance", np.nan),
            "elapsed_sec": time.time() - start,
        })

        vc = val_classes.copy()
        vc["epoch"] = epoch
        class_all.append(vc)

        vk = val_cases.copy()
        vk["epoch"] = epoch
        case_all.append(vk)

        vs = val_spatial.copy()
        vs["epoch"] = epoch
        spatial_all.append(vs)

        print(
            f"E{epoch} | "
            f"train_total={train_info['loss']:.6f} "
            f"seg={train_info['seg_loss']:.6f} "
            f"spatial={train_info['spatial_loss']:.6f} "
            f"val_loss={val_summary['loss']:.6f} "
            f"FGDice={val_summary['foreground_dice']:.6f} "
            f"PredFG={val_summary['foreground_predicted_voxels']} "
            f"Empty={val_summary['foreground_empty_cases']}/{len(val_rows)} "
            f"C2dist={val_summary.get('c2_mean_centroid_distance', np.nan):.5f} "
            f"C3dist={val_summary.get('c3_mean_centroid_distance', np.nan):.5f} "
            f"time={(time.time()-start)/60:.2f} min",
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
    class_df = pd.concat(class_all, ignore_index=True)
    case_df = pd.concat(case_all, ignore_index=True)
    spatial_df = pd.concat(spatial_all, ignore_index=True)

    best_idx = trajectory_df["val_fg_dice"].idxmax()
    best = trajectory_df.loc[best_idx].to_dict()

    return {
        "trajectory": trajectory_df,
        "class_trajectory": class_df,
        "case_trajectory": case_df,
        "spatial_trajectory": spatial_df,
        "best": best,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    global part76

    banner("PART 77")
    print("RSNA-ONLY C2/C3 SPATIAL-CONSISTENCY SUPERVISION")
    print("")
    print("NO SPIDER")
    print("NO RSNA TEST SET")
    print("Both conditions start from the exact same Part 15 initialization.")
    print("RSNA points are not used as segmentation ground truth.")
    print("Part 77 adds only a differentiable R2 target-centroid consistency loss.")

    # Locate validated Part 76 implementation.
    p76_path = next((p for p in PART76_CANDIDATES if p.exists()), None)
    if p76_path is None:
        raise FileNotFoundError(
            "No validated Part 76 source found. Expected one of:\n"
            + "\n".join(str(p) for p in PART76_CANDIDATES)
        )

    part76 = load_module(p76_path, "part77_part76")
    validate_paths(part76)
    set_seed(SEED)

    part11 = load_module(P11, "part77_part11")
    part9 = load_module(P9, "part77_part9")

    coord = pd.read_csv(COORD)
    train_rows = pd.read_csv(TRAIN_CSV).head(TRAIN_N).copy()
    val_rows = pd.read_csv(VAL_CSV).head(VAL_N).copy()

    if len(train_rows) != TRAIN_N:
        raise RuntimeError(f"Unexpected train cohort size: {len(train_rows)}")
    if len(val_rows) != VAL_N:
        raise RuntimeError(f"Unexpected validation cohort size: {len(val_rows)}")

    banner("PART 77 LOCKED EXPERIMENT")

    print(f"Device                     : {DEVICE}")
    print(f"Train cases                : {TRAIN_N}")
    print(f"Validation cases           : {VAL_N}")
    print(f"Full shape                 : {FULL}")
    print(f"Centered R2 crop           : {CROP}")
    print(f"Epochs                     : {EPOCHS}")
    print(f"Learning rate              : {LR}")
    print(f"Weight decay               : {WEIGHT_DECAY}")
    print(f"Batch size                 : {BATCH_SIZE}")
    print(f"Focus classes              : C2, C3")
    print(f"Spatial lambda             : {SPATIAL_LAMBDA}")
    print(f"Initialization SHA256      : {sha256_file(INIT_CKPT)}")
    print(f"Part 76 source             : {p76_path}")

    print("")
    print("Conditions:")
    print("  A = R2_FULL_CLASS_BALANCED")
    print("  B = R2_FULL_CLASS_BALANCED_SPATIAL")

    print("")
    print("Class-balanced weights:")
    for c, w in enumerate(CLASS_BALANCED_WEIGHTS):
        name = "Background" if c == 0 else CLASSES[c]
        print(f"  C{c} {name:<42}: {w:.6f}")

    initial_state = part76.load_initial_state()
    results = {}

    for condition in [
        "R2_FULL_CLASS_BALANCED",
        "R2_FULL_CLASS_BALANCED_SPATIAL",
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
    # REPORT TABLES
    # ========================================================================
    all_traj = pd.concat(
        [v["trajectory"] for v in results.values()],
        ignore_index=True,
    )
    all_class = pd.concat(
        [v["class_trajectory"] for v in results.values()],
        ignore_index=True,
    )
    all_cases = pd.concat(
        [v["case_trajectory"] for v in results.values()],
        ignore_index=True,
    )
    all_spatial = pd.concat(
        [v["spatial_trajectory"] for v in results.values()],
        ignore_index=True,
    )

    all_traj.to_csv(REPORT / "part77_learning_trajectory.csv", index=False)
    all_class.to_csv(REPORT / "part77_classwise_trajectory.csv", index=False)
    all_cases.to_csv(REPORT / "part77_case_trajectory.csv", index=False)
    all_spatial.to_csv(REPORT / "part77_spatial_trajectory.csv", index=False)

    baseline_best = results["R2_FULL_CLASS_BALANCED"]["best"]
    spatial_best = results["R2_FULL_CLASS_BALANCED_SPATIAL"]["best"]

    comparison = pd.DataFrame([
        {
            "condition": "R2_FULL_CLASS_BALANCED",
            "best_epoch": int(baseline_best["epoch"]),
            "best_val_fg_dice": float(baseline_best["val_fg_dice"]),
            "best_val_loss": float(baseline_best["val_loss"]),
            "best_val_pred_fg": int(baseline_best["val_pred_fg"]),
            "best_val_empty_fg": int(baseline_best["val_empty_fg"]),
            "best_c2_centroid_distance": float(
                baseline_best["val_c2_centroid_distance"]
            ),
            "best_c3_centroid_distance": float(
                baseline_best["val_c3_centroid_distance"]
            ),
        },
        {
            "condition": "R2_FULL_CLASS_BALANCED_SPATIAL",
            "best_epoch": int(spatial_best["epoch"]),
            "best_val_fg_dice": float(spatial_best["val_fg_dice"]),
            "best_val_loss": float(spatial_best["val_loss"]),
            "best_val_pred_fg": int(spatial_best["val_pred_fg"]),
            "best_val_empty_fg": int(spatial_best["val_empty_fg"]),
            "best_c2_centroid_distance": float(
                spatial_best["val_c2_centroid_distance"]
            ),
            "best_c3_centroid_distance": float(
                spatial_best["val_c3_centroid_distance"]
            ),
        },
    ])

    baseline_dice = float(baseline_best["val_fg_dice"])
    spatial_dice = float(spatial_best["val_fg_dice"])
    dice_delta = spatial_dice - baseline_dice

    # Compare spatial distance at each condition's own best Dice epoch.
    b_epoch = int(baseline_best["epoch"])
    s_epoch = int(spatial_best["epoch"])

    b_sp = all_spatial[
        (all_spatial.condition == "R2_FULL_CLASS_BALANCED")
        & (all_spatial.epoch == b_epoch)
    ]
    s_sp = all_spatial[
        (all_spatial.condition == "R2_FULL_CLASS_BALANCED_SPATIAL")
        & (all_spatial.epoch == s_epoch)
    ]

    spatial_rows = []
    for c in FOCUS_CLASSES:
        bg = b_sp[b_sp.class_id == c]
        sg = s_sp[s_sp.class_id == c]

        bdist = float(bg.normalized_centroid_distance.mean())
        sdist = float(sg.normalized_centroid_distance.mean())

        spatial_rows.append({
            "class_id": c,
            "class_name": CLASSES[c],
            "baseline_best_epoch": b_epoch,
            "spatial_best_epoch": s_epoch,
            "baseline_centroid_distance": bdist,
            "spatial_centroid_distance": sdist,
            "delta_centroid_distance": sdist - bdist,
        })

    spatial_comparison = pd.DataFrame(spatial_rows)
    spatial_comparison.to_csv(
        REPORT / "part77_spatial_comparison.csv",
        index=False,
    )

    comparison["delta_vs_baseline"] = (
        comparison["best_val_fg_dice"] - baseline_dice
    )
    comparison.to_csv(
        REPORT / "part77_condition_comparison.csv",
        index=False,
    )

    # C2/C3 classwise best-epoch comparison.
    bclass = all_class[
        (all_class.condition == "R2_FULL_CLASS_BALANCED")
        & (all_class.epoch == b_epoch)
        & (all_class.class_id.isin(FOCUS_CLASSES))
    ].copy()

    sclass = all_class[
        (all_class.condition == "R2_FULL_CLASS_BALANCED_SPATIAL")
        & (all_class.epoch == s_epoch)
        & (all_class.class_id.isin(FOCUS_CLASSES))
    ].copy()

    merged = bclass.merge(
        sclass,
        on=["class_id", "class_name"],
        suffixes=("_baseline", "_spatial"),
    )

    merged["delta_dice"] = (
        merged["dice_spatial"] - merged["dice_baseline"]
    )
    merged["delta_precision"] = (
        merged["precision_spatial"] - merged["precision_baseline"]
    )
    merged["delta_recall"] = (
        merged["recall_spatial"] - merged["recall_baseline"]
    )

    merged.to_csv(
        REPORT / "part77_c2_c3_delta.csv",
        index=False,
    )

    # ========================================================================
    # DIAGNOSIS
    # ========================================================================
    mean_focus_dice_delta = float(merged.delta_dice.mean())
    mean_distance_delta = float(
        spatial_comparison.delta_centroid_distance.mean()
    )

    # Conservative interpretation:
    # meaningful Dice improvement requires >= 0.02, matching the project's
    # previous supervision-ablation decision threshold.
    if dice_delta >= 0.02 and mean_distance_delta < 0:
        diagnosis = "MEANINGFUL_C2_C3_SPATIAL_CONSISTENCY_IMPROVEMENT"
    elif mean_focus_dice_delta >= 0.02:
        diagnosis = "C2_C3_DICE_IMPROVES_WITH_SPATIAL_CONSISTENCY"
    elif mean_distance_delta < -0.01:
        diagnosis = "SPATIAL_LOCALIZATION_IMPROVES_WITHOUT_MEANINGFUL_DICE_GAIN"
    else:
        diagnosis = "NO_MEANINGFUL_C2_C3_SPATIAL_CONSISTENCY_ADVANTAGE"

    banner("PART 77 FINAL SUMMARY")

    print(f"Baseline best Dice       : {baseline_dice:.6f} (E{b_epoch})")
    print(f"Spatial best Dice        : {spatial_dice:.6f} (E{s_epoch})")
    print(f"Overall Dice delta       : {dice_delta:+.6f}")
    print(f"Mean C2/C3 Dice delta    : {mean_focus_dice_delta:+.6f}")
    print(f"Mean centroid Δ distance : {mean_distance_delta:+.6f}")

    print("")
    print("C2/C3 comparison:")
    for _, r in merged.iterrows():
        print(
            f"  C{int(r['class_id'])} {r['class_name']:<38} "
            f"Dice {r['dice_baseline']:.6f} -> "
            f"{r['dice_spatial']:.6f} "
            f"(Δ {r['delta_dice']:+.6f})"
        )

    print("")
    print("Centroid localization:")
    for _, r in spatial_comparison.iterrows():
        print(
            f"  C{int(r['class_id'])} "
            f"{r['baseline_centroid_distance']:.6f} -> "
            f"{r['spatial_centroid_distance']:.6f} "
            f"(Δ {r['delta_centroid_distance']:+.6f})"
        )

    print("")
    print(f"Diagnosis                 : {diagnosis}")

    # ========================================================================
    # JSON + TEXT REPORT
    # ========================================================================
    summary = {
        "part": 77,
        "purpose": (
            "Controlled C2/C3 spatial-consistency supervision pilot "
            "using differentiable soft-centroid alignment to R2_FULL."
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
        "spatial_lambda": SPATIAL_LAMBDA,
        "class_balanced_weights": CLASS_BALANCED_WEIGHTS.tolist(),
        "initialization_checkpoint": str(INIT_CKPT),
        "initialization_sha256": sha256_file(INIT_CKPT),
        "part76_source": str(p76_path),
        "baseline_best": baseline_best,
        "spatial_best": spatial_best,
        "overall_dice_delta": dice_delta,
        "mean_c2_c3_dice_delta": mean_focus_dice_delta,
        "mean_centroid_distance_delta": mean_distance_delta,
        "diagnosis": diagnosis,
        "training_completed": True,
        "spider_used": False,
        "rsna_test_set_used": False,
        "rsna_points_are_segmentation_ground_truth": False,
        "scientific_limitation": (
            "The spatial target is derived from R2 pseudo-masks. "
            "RSNA coordinate annotations are point/localizer annotations, "
            "not manual segmentation masks."
        ),
    }

    (REPORT / "part77_summary.json").write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )

    report_lines = [
        "PART 77 — C2/C3 SPATIAL-CONSISTENCY SUPERVISION",
        "",
        f"Device: {DEVICE}",
        f"Train cases: {TRAIN_N}",
        f"Validation cases: {VAL_N}",
        f"Full shape: {FULL}",
        f"Crop shape: {CROP}",
        f"Epochs: {EPOCHS}",
        f"Learning rate: {LR}",
        f"Spatial lambda: {SPATIAL_LAMBDA}",
        f"Initialization SHA256: {sha256_file(INIT_CKPT)}",
        "",
        "Baseline:",
        f"  Best epoch: {b_epoch}",
        f"  Best foreground Dice: {baseline_dice:.6f}",
        "",
        "Spatial consistency:",
        f"  Best epoch: {s_epoch}",
        f"  Best foreground Dice: {spatial_dice:.6f}",
        "",
        f"Overall Dice delta: {dice_delta:+.6f}",
        f"Mean C2/C3 Dice delta: {mean_focus_dice_delta:+.6f}",
        f"Mean centroid distance delta: {mean_distance_delta:+.6f}",
        "",
        f"Diagnosis: {diagnosis}",
        "",
        "Scientific limitation:",
        "R2_FULL is a pseudo-mask derived from RSNA point/localizer",
        "annotations. It is not manual clinical segmentation ground truth.",
    ]

    (REPORT / "part77_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    banner("PART 77 COMPLETE")
    print(f"Learning trajectory : {REPORT / 'part77_learning_trajectory.csv'}")
    print(f"Class trajectory    : {REPORT / 'part77_classwise_trajectory.csv'}")
    print(f"Case trajectory     : {REPORT / 'part77_case_trajectory.csv'}")
    print(f"Spatial trajectory  : {REPORT / 'part77_spatial_trajectory.csv'}")
    print(f"Spatial comparison  : {REPORT / 'part77_spatial_comparison.csv'}")
    print(f"C2/C3 delta         : {REPORT / 'part77_c2_c3_delta.csv'}")
    print(f"Condition comparison : {REPORT / 'part77_condition_comparison.csv'}")
    print(f"Summary JSON         : {REPORT / 'part77_summary.json'}")
    print(f"Text report          : {REPORT / 'part77_report.txt'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        banner("PART 77 FAILED")
        traceback.print_exc()
        raise
