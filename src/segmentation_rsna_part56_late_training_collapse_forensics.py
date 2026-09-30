"""
PART 56 — LATE-TRAINING COLLAPSE FORENSICS

Goal
----
Part 55 showed that R0, R1 and R2 pseudo-mask conditions can produce
foreground predictions early in training but move rapidly toward
background-only predictions by epoch 5.

Part 56 investigates THAT transition directly.

This is a NO-TRAINING diagnostic:
    - loads the already-saved Part 55 checkpoints
    - evaluates epochs 0-5 on the exact Part 55 validation cohort
    - measures per-class probabilities/predictions
    - measures foreground-vs-background probability gap
    - measures prediction occupancy relative to target occupancy
    - measures foreground/background gradient pressure at each saved epoch
      on representative validation cases
    - does not update weights

No optimizer.step(), backward(), training loop, or checkpoint modification
is performed.

Conditions:
    R0 = original pseudo-mask
    R1 = 1-voxel 3-D 6-connected dilation
    R2 = 2-voxel 3-D 6-connected dilation

The Part 55 checkpoints are the evidence source. Part 15 remains untouched.
"""

from __future__ import annotations

import csv
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

P55 = ROOT / "outputs" / "segmentation" / "rsna_part55_r0_r1_r2_pseudomask_stability_trajectory"

OUT = ROOT / "outputs" / "segmentation" / "rsna_part56_late_training_collapse_forensics"
REPORT = OUT / "reports"

CONDITIONS = {
    "R0_original": 0,
    "R1_dilation": 1,
    "R2_dilation": 2,
}

EPOCHS = [0, 1, 2, 3, 4, 5]

TRAIN_N = 100
VAL_N = 50

# Full Part 55 validation cohort is evaluated for prediction behavior.
# Gradient forensics are deliberately limited to these representative cases
# to keep this diagnostic lightweight.
GRADIENT_CASES = [1, 10, 20, 25, 40]

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)
NUM_CLASSES = 6

LR = 1e-4
WD = 1e-5

SEED = 156

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
        clamp_start(
            int(center[i]),
            CROP_SHAPE[i],
            image.shape[i],
        )
        for i in range(3)
    ]

    z, y, x = starts
    dz, dy, dx = CROP_SHAPE

    return (
        image[z:z + dz, y:y + dy, x:x + dx].astype(np.float32),
        mask[z:z + dz, y:y + dy, x:x + dx].astype(np.int64),
    )


# ================================================================
# SAME DILATION CONSTRUCTION AS PARTS 52–55
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
                    dilate_binary_6_connected(
                        original,
                        radius,
                    ),
                )
            )

    candidates.sort(
        key=lambda x: (-x[0], x[1])
    )

    result = np.zeros_like(
        mask,
        dtype=np.int64,
    )

    occupied = np.zeros_like(
        mask,
        dtype=bool,
    )

    for _, class_id, region in candidates:
        assignable = region & (~occupied)
        result[assignable] = class_id
        occupied |= assignable

    return result


# ================================================================
# DATA
# ================================================================

def load_case(part11, part9, row):
    loaded = part11.load_tensor_case(
        row,
        part9,
    )

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
        raise RuntimeError(
            f"Unexpected image shape: {image.shape}"
        )

    if tuple(mask.shape) != FULL_SHAPE:
        raise RuntimeError(
            f"Unexpected mask shape: {mask.shape}"
        )

    return foreground_centered_crop(
        image.astype(np.float32),
        mask.astype(np.int64),
    )


def preload_validation(part11, part9, df):
    cases = []

    print()
    print("=" * 82)
    print("PART 56 VALIDATION PRELOAD")
    print("=" * 82)

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        image, mask = load_case(
            part11,
            part9,
            row,
        )

        cases.append(
            {
                "image": image,
                "mask": mask,
                "index": i,
            }
        )

        if i in (1, 10, 20, 25, 40, 50):
            print(
                f"VALIDATION {i:03d}/{len(df)} "
                f"FG={int((mask > 0).sum())}"
            )

    return cases


# ================================================================
# CHECKPOINTS
# ================================================================

def checkpoint_path(condition, epoch):
    condition_dir = P55 / condition / "checkpoints"

    if epoch == 0:
        return INIT

    return condition_dir / f"epoch_{epoch:02d}.pth"


def load_checkpoint_into_model(model, condition, epoch, device):
    path = checkpoint_path(
        condition,
        epoch,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}"
        )

    state = torch.load(
        path,
        map_location=device,
    )

    if "model_state_dict" not in state:
        raise KeyError(
            f"model_state_dict missing from {path}"
        )

    model.load_state_dict(
        state["model_state_dict"],
        strict=True,
    )

    return path


# ================================================================
# PREDICTION FORENSICS
# ================================================================

@torch.no_grad()
def prediction_metrics(
    model,
    cases,
    device,
    loss_fn,
):
    model.eval()

    losses = []
    fg_dice = []

    pred_fg = []
    target_fg = []

    fg_prob = []
    bg_prob = []

    mean_fg_logit = []
    mean_bg_logit = []

    class_pred = {
        i: []
        for i in range(NUM_CLASSES)
    }

    class_target = {
        i: []
        for i in range(NUM_CLASSES)
    }

    class_prob = {
        i: []
        for i in range(NUM_CLASSES)
    }

    class_dice = {
        i: []
        for i in range(NUM_CLASSES)
    }

    empty = 0

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

        loss = loss_fn(
            logits,
            y,
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )

        pred = torch.argmax(
            probs,
            dim=1,
        )

        target_np = y[:, 0].detach().cpu().numpy()[0]
        pred_np = pred.detach().cpu().numpy()[0]

        target_fg_mask = target_np > 0
        prediction_fg_mask = pred_np > 0

        intersection = np.logical_and(
            target_fg_mask,
            prediction_fg_mask,
        ).sum()

        denominator = (
            target_fg_mask.sum()
            + prediction_fg_mask.sum()
        )

        dice = (
            float(2.0 * intersection / denominator)
            if denominator > 0
            else 1.0
        )

        if prediction_fg_mask.sum() == 0:
            empty += 1

        losses.append(
            float(loss.item())
        )

        fg_dice.append(dice)

        pred_fg.append(
            int(prediction_fg_mask.sum())
        )

        target_fg.append(
            int(target_fg_mask.sum())
        )

        fg_prob.append(
            float(
                probs[:, 1:, ...]
                .sum(dim=1)
                .mean()
                .item()
            )
        )

        bg_prob.append(
            float(
                probs[:, 0, ...]
                .mean()
                .item()
            )
        )

        fg_mask = target_fg_mask

        if fg_mask.any():
            mean_fg_logit.append(
                float(
                    logits[
                        0,
                        1:,
                        ...
                    ][:, fg_mask]
                    .mean()
                    .item()
                )
            )

            mean_bg_logit.append(
                float(
                    logits[
                        0,
                        0,
                        ...
                    ][fg_mask]
                    .mean()
                    .item()
                )
            )

        for class_id in range(NUM_CLASSES):
            t = target_np == class_id
            p = pred_np == class_id

            class_target[class_id].append(
                int(t.sum())
            )

            class_pred[class_id].append(
                int(p.sum())
            )

            class_prob[class_id].append(
                float(
                    probs[
                        0,
                        class_id,
                        ...
                    ].mean().item()
                )
            )

            d = t.sum() + p.sum()

            if d == 0:
                class_dice[class_id].append(
                    np.nan
                )
            else:
                class_dice[class_id].append(
                    float(
                        2.0
                        * np.logical_and(t, p).sum()
                        / d
                    )
                )

        del x, y, logits, loss, probs, pred

    return {
        "loss": float(np.mean(losses)),
        "foreground_dice": float(np.mean(fg_dice)),
        "predicted_foreground_voxels": float(np.mean(pred_fg)),
        "target_foreground_voxels": float(np.mean(target_fg)),
        "foreground_probability": float(np.mean(fg_prob)),
        "background_probability": float(np.mean(bg_prob)),
        "foreground_background_probability_gap": float(
            np.mean(fg_prob)
            - np.mean(bg_prob)
        ),
        "mean_foreground_logit": (
            float(np.mean(mean_fg_logit))
            if mean_fg_logit
            else None
        ),
        "mean_background_logit": (
            float(np.mean(mean_bg_logit))
            if mean_bg_logit
            else None
        ),
        "empty_prediction_cases": int(empty),
        "total_cases": len(cases),
        "prediction_to_target_fg_ratio": float(
            np.mean(pred_fg)
            / max(np.mean(target_fg), 1e-12)
        ),
        "class_predicted_voxels": {
            str(i): float(np.mean(class_pred[i]))
            for i in range(NUM_CLASSES)
        },
        "class_target_voxels": {
            str(i): float(np.mean(class_target[i]))
            for i in range(NUM_CLASSES)
        },
        "class_probability": {
            str(i): float(np.mean(class_prob[i]))
            for i in range(NUM_CLASSES)
        },
        "class_dice": {
            str(i): (
                float(np.nanmean(class_dice[i]))
                if np.isfinite(
                    np.asarray(class_dice[i])
                ).any()
                else None
            )
            for i in range(NUM_CLASSES)
        },
    }


# ================================================================
# GRADIENT PRESSURE
# ================================================================

def gradient_forensics(
    model,
    cases,
    device,
    condition,
    epoch,
):
    """
    Evaluate the loss gradient on representative validation cases.

    No optimizer step occurs.

    We compute:
        - mean signed gradient on foreground-class logits at target-FG voxels
        - mean signed background-logit gradient at target-FG voxels
        - absolute gradient magnitudes
        - ratio |FG| / |BG|

    Positive/negative interpretation follows gradient descent:
        negative dL/dz_fg -> increasing FG logit lowers loss
        positive dL/dz_bg -> decreasing BG logit lowers loss
    """

    model.eval()

    dice_loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    records = []

    # Only representative cases.
    selected = [
        c
        for c in cases
        if c["index"] in GRADIENT_CASES
    ]

    for case in selected:
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

        model.zero_grad(set_to_none=True)

        logits = model(x)

        logits.retain_grad()

        loss = dice_loss_fn(
            logits,
            y,
        )

        loss.backward()

        with torch.no_grad():
            target = y[:, 0, ...]
            fg_mask = target > 0

            if not fg_mask.any():
                continue

            # Aggregate gradients of the five foreground channels
            # on target foreground voxels.
            # logits.grad has shape [B, C, D, H, W].
            # Flatten the spatial dimensions before applying the
            # 3-D Boolean foreground mask.
            flat_fg_mask = fg_mask.reshape(-1)

            fg_gradient = (
                logits.grad[
                    0,
                    1:,
                    ...
                ]
                .reshape(NUM_CLASSES - 1, -1)[:, flat_fg_mask]
            )

            bg_gradient = (
                logits.grad[
                    0,
                    0,
                    ...
                ]
                .reshape(-1)[flat_fg_mask]
            )

            signed_fg = float(
                fg_gradient.mean().item()
            )

            signed_bg = float(
                bg_gradient.mean().item()
            )

            abs_fg = float(
                fg_gradient.abs().mean().item()
            )

            abs_bg = float(
                bg_gradient.abs().mean().item()
            )

            ratio = (
                abs_fg / abs_bg
                if abs_bg > 0
                else None
            )

            records.append(
                {
                    "condition": condition,
                    "epoch": epoch,
                    "case": case["index"],
                    "loss": float(loss.item()),
                    "signed_foreground_gradient": signed_fg,
                    "signed_background_gradient": signed_bg,
                    "absolute_foreground_gradient": abs_fg,
                    "absolute_background_gradient": abs_bg,
                    "foreground_background_gradient_ratio": ratio,
                }
            )

        model.zero_grad(set_to_none=True)

        del x, y, logits, loss

    return records


# ================================================================
# MAIN
# ================================================================

def main():
    seed_everything(SEED)

    print("=" * 82)
    print("PART 56 PATH VALIDATION")
    print("=" * 82)

    required = [
        ("Project root", ROOT),
        ("Part 11", PART11_PATH),
        ("Part 9", PART9_PATH),
        ("Part 15 initialization", INIT),
        ("Part 15 train cohort", TRCSV),
        ("Part 15 validation cohort", VACSV),
        ("Part 55 output", P55),
    ]

    for label, path in required:
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{label:<40}: {status}")

        if not path.exists():
            raise FileNotFoundError(path)

    print()
    print("=" * 82)
    print("PART 56 — LATE-TRAINING COLLAPSE FORENSICS")
    print("=" * 82)
    print(f"Validation cohort : {VAL_N}")
    print(f"Epochs evaluated : {EPOCHS}")
    print(f"Gradient cases : {GRADIENT_CASES}")
    print(f"Full volume : {FULL_SHAPE}")
    print(f"Crop : {CROP_SHAPE}")
    print(f"Radii : {list(CONDITIONS.values())}")
    print("Training performed : NO")
    print("Optimizer used : NO")
    print("Backward for diagnostics only : YES")
    print("Optimizer step : NO")
    print("SPIDER : NO")
    print("Test set : NO")
    print("Part 15 modified : NO")
    print(f"Part 15 initialization SHA256 : {sha256(INIT)}")

    part11 = load_module(
        PART11_PATH,
        "segmentation_rsna_part11_part56_runtime",
    )

    part9 = load_module(
        PART9_PATH,
        "segmentation_rsna_part9_part56_runtime",
    )

    val_df = pd.read_csv(
        VACSV
    ).head(VAL_N)

    val_cases_base = preload_validation(
        part11,
        part9,
        val_df,
    )

    print()
    print("=" * 82)
    print("PART 56 SHAPE / LABEL SMOKE TEST")
    print("=" * 82)

    assert (
        val_cases_base[0]["image"].shape
        == CROP_SHAPE
    )

    assert (
        val_cases_base[0]["mask"].shape
        == CROP_SHAPE
    )

    assert (
        val_cases_base[0]["mask"].min()
        >= 0
    )

    assert (
        val_cases_base[0]["mask"].max()
        < NUM_CLASSES
    )

    print(
        f"Image : {val_cases_base[0]['image'].shape}"
    )

    print(
        f"Mask : {val_cases_base[0]['mask'].shape}"
    )

    print(
        f"Labels : "
        f"{sorted(np.unique(val_cases_base[0]['mask']).tolist())}"
    )

    print("✓ Shape / label smoke test PASSED.")

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"PyTorch : {torch.__version__}")
    print(f"Device : {device}")

    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    prediction_records = []
    gradient_records = []

    # ============================================================
    # CONDITION / EPOCH FORENSICS
    # ============================================================

    for condition, radius in CONDITIONS.items():

        print()
        print("=" * 82)
        print(
            f"PART 56 CONDITION: {condition} | RADIUS {radius}"
        )
        print("=" * 82)

        condition_cases = [
            {
                "image": c["image"],
                "mask": dilate_multiclass_mask(
                    c["mask"],
                    radius,
                ),
                "index": c["index"],
            }
            for c in val_cases_base
        ]

        model = part11.create_model(
            device
        )

        for epoch in EPOCHS:

            ckpt = checkpoint_path(
                condition,
                epoch,
            )

            model_path = load_checkpoint_into_model(
                model,
                condition,
                epoch,
                device,
            )

            metrics = prediction_metrics(
                model,
                condition_cases,
                device,
                loss_fn,
            )

            record = {
                "condition": condition,
                "radius": radius,
                "epoch": epoch,
                "checkpoint": str(model_path),
                **metrics,
            }

            prediction_records.append(
                record
            )

            print(
                f"Epoch {epoch:02d} | "
                f"val_loss={metrics['loss']:.6f} | "
                f"FGDice={metrics['foreground_dice']:.6f} | "
                f"PredFG={metrics['predicted_foreground_voxels']:.1f} | "
                f"TargetFG={metrics['target_foreground_voxels']:.1f} | "
                f"FGProb={metrics['foreground_probability']:.6f} | "
                f"BGProb={metrics['background_probability']:.6f} | "
                f"Gap={metrics['foreground_background_probability_gap']:.6f} | "
                f"Ratio={metrics['prediction_to_target_fg_ratio']:.3f} | "
                f"Empty={metrics['empty_prediction_cases']}/{VAL_N}"
            )

            # Gradient diagnostics at each saved epoch.
            grads = gradient_forensics(
                model,
                condition_cases,
                device,
                condition,
                epoch,
            )

            gradient_records.extend(
                grads
            )

            if grads:
                fg_abs = float(
                    np.mean(
                        [
                            r["absolute_foreground_gradient"]
                            for r in grads
                        ]
                    )
                )

                bg_abs = float(
                    np.mean(
                        [
                            r["absolute_background_gradient"]
                            for r in grads
                        ]
                    )
                )

                fg_signed = float(
                    np.mean(
                        [
                            r["signed_foreground_gradient"]
                            for r in grads
                        ]
                    )
                )

                bg_signed = float(
                    np.mean(
                        [
                            r["signed_background_gradient"]
                            for r in grads
                        ]
                    )
                )

                ratio = (
                    fg_abs / bg_abs
                    if bg_abs > 0
                    else None
                )

                print(
                    f"           GRAD | "
                    f"signedFG={fg_signed:.3e} "
                    f"signedBG={bg_signed:.3e} "
                    f"|absFG|={fg_abs:.3e} "
                    f"|absBG|={bg_abs:.3e} "
                    f"|ratio={ratio:.4f}"
                )

        del model, condition_cases

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # COLLAPSE-TRANSITION ANALYSIS
    # ============================================================

    print()
    print("=" * 82)
    print("PART 56 COLLAPSE-TRANSITION ANALYSIS")
    print("=" * 82)

    transition_rows = []

    for condition in CONDITIONS:
        records = [
            r
            for r in prediction_records
            if r["condition"] == condition
        ]

        records.sort(
            key=lambda r: r["epoch"]
        )

        first_epoch_with_major_drop = None

        for prev, current in zip(
            records[:-1],
            records[1:],
        ):
            previous_pred = prev[
                "predicted_foreground_voxels"
            ]

            current_pred = current[
                "predicted_foreground_voxels"
            ]

            if (
                previous_pred > 0
                and current_pred
                <= 0.25 * previous_pred
            ):
                first_epoch_with_major_drop = current[
                    "epoch"
                ]
                break

        epoch3 = next(
            r for r in records
            if r["epoch"] == 3
        )

        epoch4 = next(
            r for r in records
            if r["epoch"] == 4
        )

        epoch5 = next(
            r for r in records
            if r["epoch"] == 5
        )

        transition_rows.append(
            {
                "condition": condition,
                "major_prediction_drop_epoch": first_epoch_with_major_drop,
                "epoch3_pred_fg": epoch3[
                    "predicted_foreground_voxels"
                ],
                "epoch4_pred_fg": epoch4[
                    "predicted_foreground_voxels"
                ],
                "epoch5_pred_fg": epoch5[
                    "predicted_foreground_voxels"
                ],
                "epoch3_to_epoch5_pred_ratio": (
                    epoch5[
                        "predicted_foreground_voxels"
                    ]
                    / max(
                        epoch3[
                            "predicted_foreground_voxels"
                        ],
                        1e-12,
                    )
                ),
                "epoch3_fg_dice": epoch3[
                    "foreground_dice"
                ],
                "epoch4_fg_dice": epoch4[
                    "foreground_dice"
                ],
                "epoch5_fg_dice": epoch5[
                    "foreground_dice"
                ],
            }
        )

        print(
            f"{condition:<18} | "
            f"major_drop_epoch={first_epoch_with_major_drop} | "
            f"PredFG E3={epoch3['predicted_foreground_voxels']:.1f} "
            f"E4={epoch4['predicted_foreground_voxels']:.1f} "
            f"E5={epoch5['predicted_foreground_voxels']:.1f} | "
            f"E3→E5 ratio="
            f"{epoch5['predicted_foreground_voxels'] / max(epoch3['predicted_foreground_voxels'], 1e-12):.6f}"
        )

    # ============================================================
    # GRADIENT SUMMARY
    # ============================================================

    print()
    print("=" * 82)
    print("PART 56 GRADIENT SUMMARY")
    print("=" * 82)

    gradient_summary = []

    for condition in CONDITIONS:
        for epoch in EPOCHS:

            rows = [
                r
                for r in gradient_records
                if r["condition"] == condition
                and r["epoch"] == epoch
            ]

            if not rows:
                continue

            summary_row = {
                "condition": condition,
                "epoch": epoch,
                "mean_signed_fg_gradient": float(
                    np.mean(
                        [
                            r[
                                "signed_foreground_gradient"
                            ]
                            for r in rows
                        ]
                    )
                ),
                "mean_signed_bg_gradient": float(
                    np.mean(
                        [
                            r[
                                "signed_background_gradient"
                            ]
                            for r in rows
                        ]
                    )
                ),
                "mean_abs_fg_gradient": float(
                    np.mean(
                        [
                            r[
                                "absolute_foreground_gradient"
                            ]
                            for r in rows
                        ]
                    )
                ),
                "mean_abs_bg_gradient": float(
                    np.mean(
                        [
                            r[
                                "absolute_background_gradient"
                            ]
                            for r in rows
                        ]
                    )
                ),
            }

            summary_row[
                "fg_bg_gradient_ratio"
            ] = (
                summary_row[
                    "mean_abs_fg_gradient"
                ]
                / max(
                    summary_row[
                        "mean_abs_bg_gradient"
                    ],
                    1e-30,
                )
            )

            gradient_summary.append(
                summary_row
            )

            print(
                f"{condition:<18} E{epoch} | "
                f"sFG={summary_row['mean_signed_fg_gradient']:.3e} "
                f"sBG={summary_row['mean_signed_bg_gradient']:.3e} | "
                f"|FG|={summary_row['mean_abs_fg_gradient']:.3e} "
                f"|BG|={summary_row['mean_abs_bg_gradient']:.3e} "
                f"ratio={summary_row['fg_bg_gradient_ratio']:.4f}"
            )

    # ============================================================
    # DIAGNOSIS
    # ============================================================

    all_final_empty = all(
        r["empty_prediction_cases"] > 0
        for r in prediction_records
        if r["epoch"] == 5
    )

    all_major_drop = all(
        r["major_prediction_drop_epoch"] is not None
        for r in transition_rows
    )

    all_fg_gradient_negative = all(
        r["mean_signed_fg_gradient"] < 0
        for r in gradient_summary
    )

    all_bg_gradient_positive = all(
        r["mean_signed_bg_gradient"] > 0
        for r in gradient_summary
    )

    if (
        all_major_drop
        and all_final_empty
        and all_fg_gradient_negative
        and all_bg_gradient_positive
    ):
        diagnosis = (
            "LATE_TRAINING_FOREGROUND_COLLAPSE_WITH_CORRECT_GRADIENT_DIRECTION"
        )
    elif all_major_drop and all_final_empty:
        diagnosis = (
            "LATE_TRAINING_FOREGROUND_COLLAPSE_CONFIRMED"
        )
    else:
        diagnosis = (
            "NO_UNIVERSAL_LATE_TRAINING_COLLAPSE_PATTERN"
        )

    # ============================================================
    # SAVE CSV REPORTS
    # ============================================================

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_csv = (
        REPORT
        / "part56_prediction_forensics.csv"
    )

    with open(
        prediction_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "condition",
            "radius",
            "epoch",
            "checkpoint",
            "loss",
            "foreground_dice",
            "predicted_foreground_voxels",
            "target_foreground_voxels",
            "foreground_probability",
            "background_probability",
            "foreground_background_probability_gap",
            "mean_foreground_logit",
            "mean_background_logit",
            "empty_prediction_cases",
            "total_cases",
            "prediction_to_target_fg_ratio",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in prediction_records:
            writer.writerow(
                {
                    k: row.get(k)
                    for k in fieldnames
                }
            )

    gradient_csv = (
        REPORT
        / "part56_gradient_forensics.csv"
    )

    with open(
        gradient_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "condition",
            "epoch",
            "case",
            "loss",
            "signed_foreground_gradient",
            "signed_background_gradient",
            "absolute_foreground_gradient",
            "absolute_background_gradient",
            "foreground_background_gradient_ratio",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in gradient_records:
            writer.writerow(
                {
                    k: row.get(k)
                    for k in fieldnames
                }
            )

    gradient_summary_csv = (
        REPORT
        / "part56_gradient_summary.csv"
    )

    with open(
        gradient_summary_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = list(
            gradient_summary[0].keys()
        )

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            gradient_summary
        )

    transition_csv = (
        REPORT
        / "part56_transition_summary.csv"
    )

    with open(
        transition_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = list(
            transition_rows[0].keys()
        )

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            transition_rows
        )

    summary = {
        "part": 56,
        "purpose": (
            "Forensic analysis of the late-training foreground collapse "
            "observed in Part 55."
        ),
        "configuration": {
            "validation_cases": VAL_N,
            "epochs": EPOCHS,
            "gradient_cases": GRADIENT_CASES,
            "full_shape": FULL_SHAPE,
            "crop_shape": CROP_SHAPE,
            "radii": list(CONDITIONS.values()),
            "training_performed": False,
            "optimizer_used": False,
            "backward_for_diagnostic_gradient": True,
            "optimizer_step": False,
            "spider": False,
            "test_set": False,
            "part15_modified": False,
            "part15_initialization_sha256": sha256(INIT),
        },
        "prediction_records": prediction_records,
        "gradient_records": gradient_records,
        "gradient_summary": gradient_summary,
        "transition_summary": transition_rows,
        "diagnosis": diagnosis,
        "interpretation": (
            "The diagnostic determines whether the late collapse is "
            "associated with a rapid reduction in predicted foreground "
            "while the signed gradient direction on target foreground "
            "voxels remains directionally appropriate. This does not "
            "identify a single causal mechanism by itself."
        ),
    }

    summary_path = (
        REPORT
        / "part56_summary.json"
    )

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
    print("PART 56 DIAGNOSTIC INTERPRETATION")
    print("=" * 82)
    print(f"Diagnosis : {diagnosis}")
    print(
        "Gradient interpretation : "
        "negative signed foreground gradient and positive signed "
        "background gradient on target-FG voxels indicate the loss "
        "continues to favor increasing foreground logits and decreasing "
        "background logits at those voxels."
    )
    print(
        "Important : this is a checkpoint-forensics diagnostic; "
        "no weights were updated."
    )

    print()
    print("=" * 82)
    print("PART 56 COMPLETE")
    print("=" * 82)
    print(f"Output directory : {OUT}")
    print(f"Prediction CSV : {prediction_csv}")
    print(f"Gradient CSV : {gradient_csv}")
    print(f"Gradient summary : {gradient_summary_csv}")
    print(f"Transition summary : {transition_csv}")
    print(f"Summary JSON : {summary_path}")


if __name__ == "__main__":
    main()
