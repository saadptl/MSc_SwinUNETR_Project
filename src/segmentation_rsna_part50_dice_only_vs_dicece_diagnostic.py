"""
PART 50 — DICE-ONLY VS DICE+CE TRAINING-DYNAMICS DIAGNOSTIC

Controlled diagnostic:
    A = original Dice + CE
    B = Dice-only

Both conditions use:
    - exact Part 15 initialization
    - first 100 train cases
    - first 50 validation cases
    - foreground-centered 32x64x64 crops
    - 3 epochs
    - AdamW, LR=1e-4, WD=1e-5
    - same data and initialization
    - no SPIDER
    - no test set
    - Part 15 untouched

The purpose is to isolate whether removing CE materially changes
foreground behavior / collapse dynamics.
"""

from __future__ import annotations

import csv
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

PART11_PATH = (
    ROOT / "src" / "segmentation_rsna_part11_controlled_pilot_training.py"
)
PART9_PATH = (
    ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py"
)

P15 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

INIT = P15 / "checkpoints" / "part15_initialization_from_part11.pth"
TRCSV = P15 / "part15_train_cohort.csv"
VACSV = P15 / "part15_validation_cohort.csv"

OUT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part50_dice_only_vs_dicece_diagnostic"
)

BASE = OUT / "original_dicece"
DICE_ONLY = OUT / "dice_only"
REPORT = OUT / "reports"

SEED = 150
TRAIN_N = 100
VAL_N = 50
EPOCHS = 3

LR = 1e-4
WD = 1e-5
GRAD_CLIP = 1.0

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)

NUM_CLASSES = 6

CLASS_NAMES = [
    "Background",
    "Spinal_Canal_Stenosis",
    "Left_Neural_Foraminal_Narrowing",
    "Right_Neural_Foraminal_Narrowing",
    "Left_Subarticular_Stenosis",
    "Right_Subarticular_Stenosis",
]


# ================================================================
# REPRODUCIBILITY
# ================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# ================================================================
# MODULE LOADING
# ================================================================

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


# ================================================================
# PATH VALIDATION
# ================================================================

def validate_paths() -> None:
    print("=" * 82)
    print("PART 50 PATH VALIDATION")
    print("=" * 82)

    checks = [
        ("Project root", ROOT),
        ("Part 11", PART11_PATH),
        ("Part 9", PART9_PATH),
        ("Part 15 initialization", INIT),
        ("Part 15 train cohort", TRCSV),
        ("Part 15 validation cohort", VACSV),
    ]

    for label, path in checks:
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{label:<40}: {status}")

        if not path.exists():
            raise FileNotFoundError(path)


# ================================================================
# SPATIAL CROP
# ================================================================

def clamp_start(center: int, crop: int, total: int) -> int:
    start = center - crop // 2
    return max(0, min(start, total - crop))


def foreground_centered_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:

    fg = np.argwhere(mask > 0)

    if fg.size == 0:
        starts = [
            max(0, (total - crop) // 2)
            for total, crop in zip(image.shape, crop_shape)
        ]

    else:
        center = np.round(fg.mean(axis=0)).astype(int)

        starts = [
            clamp_start(
                int(center[i]),
                crop_shape[i],
                image.shape[i],
            )
            for i in range(3)
        ]

    z, y, x = starts
    dz, dy, dx = crop_shape

    cropped_image = image[
        z:z + dz,
        y:y + dy,
        x:x + dx,
    ].astype(np.float32)

    cropped_mask = mask[
        z:z + dz,
        y:y + dy,
        x:x + dx,
    ].astype(np.int64)

    return cropped_image, cropped_mask


# ================================================================
# DATA PRELOAD
# ================================================================

def load_cases(
    part11,
    part9,
    csv_path: Path,
    n: int,
    label: str,
):
    df = pd.read_csv(csv_path).head(n).copy()

    if len(df) != n:
        raise RuntimeError(
            f"{label}: expected {n} rows, found {len(df)}"
        )

    cases = []

    print()
    print("=" * 82)
    print(f"PART 50 {label.upper()} DATA PRELOAD")
    print("-" * 82)

    for i, (_, row) in enumerate(df.iterrows(), start=1):

        loaded = part11.load_tensor_case(row, part9)

        if not isinstance(loaded, (tuple, list)):
            raise RuntimeError(
                f"Unexpected loader return type: {type(loaded)}"
            )

        if len(loaded) < 2:
            raise RuntimeError(
                f"Loader returned fewer than two values for case {i}"
            )

        image = loaded[0]
        mask = loaded[1]

        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()

        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()

        image = np.asarray(image)
        mask = np.asarray(mask)

        # Normalize [1,D,H,W] -> [D,H,W].
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]

        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask[0]

        image = image.astype(np.float32)
        mask = mask.astype(np.int64)

        if tuple(image.shape) != FULL_SHAPE:
            raise RuntimeError(
                f"{label} case {i}: image shape {image.shape}; "
                f"expected {FULL_SHAPE}"
            )

        if tuple(mask.shape) != FULL_SHAPE:
            raise RuntimeError(
                f"{label} case {i}: mask shape {mask.shape}; "
                f"expected {FULL_SHAPE}"
            )

        crop_image, crop_mask = foreground_centered_crop(
            image,
            mask,
            CROP_SHAPE,
        )

        fg = int((crop_mask > 0).sum())

        if i == 1 or i == n or i % 25 == 0:
            print(
                f"{label.upper()} {i:03d}/{n} FG={fg}"
            )

        cases.append((crop_image, crop_mask))

    foreground_counts = [
        int((mask > 0).sum())
        for _, mask in cases
    ]

    mean_fg = float(np.mean(foreground_counts))
    zero_fg = int(sum(v == 0 for v in foreground_counts))

    print(
        f"Mean {label.lower()} crop foreground voxels : "
        f"{mean_fg:.2f}"
    )

    print(
        f"{label.capitalize()} zero-FG crops : "
        f"{zero_fg}/{n}"
    )

    return cases


# ================================================================
# MODEL
# ================================================================

def create_model(part11, device):
    return part11.create_model(device)


def load_initialization(model, device) -> None:
    checkpoint = torch.load(
        INIT,
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )


# ================================================================
# LOSSES
# ================================================================

def create_losses():
    original_dicece = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    dice_only = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        lambda_dice=1.0,
        lambda_ce=0.0,
    )

    return original_dicece, dice_only


# ================================================================
# PREDICTION METRICS
# ================================================================

@torch.no_grad()
def prediction_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
):
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    prediction = torch.argmax(
        probabilities,
        dim=1,
    )

    target_fg = target > 0
    prediction_fg = prediction > 0

    target_fg_count = int(
        target_fg.sum().item()
    )

    prediction_fg_count = int(
        prediction_fg.sum().item()
    )

    intersection = int(
        (target_fg & prediction_fg).sum().item()
    )

    denominator = (
        target_fg_count
        + prediction_fg_count
    )

    if denominator > 0:
        fg_dice = (
            2.0 * intersection / denominator
        )
    else:
        fg_dice = 1.0

    fg_probability = float(
        probabilities[:, 1:, ...]
        .sum(dim=1)
        .mean()
        .item()
    )

    bg_probability = float(
        probabilities[:, 0, ...]
        .mean()
        .item()
    )

    return {
        "fg_dice": float(fg_dice),
        "pred_fg": prediction_fg_count,
        "target_fg": target_fg_count,
        "fg_prob": fg_probability,
        "bg_prob": bg_probability,
    }


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate(
    model,
    cases,
    loss_fn,
    device,
):
    model.eval()

    losses = []
    dices = []
    predicted_fg = []
    target_fg = []
    fg_probabilities = []
    bg_probabilities = []

    empty_cases = 0

    for image_np, mask_np in cases:

        x = (
            torch.from_numpy(image_np)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        y = (
            torch.from_numpy(mask_np)
            .unsqueeze(0)
            .to(device)
        )

        logits = model(x)

        loss = loss_fn(
            logits,
            y.unsqueeze(1),
        )

        stats = prediction_stats(
            logits,
            y,
        )

        losses.append(float(loss.item()))
        dices.append(stats["fg_dice"])
        predicted_fg.append(stats["pred_fg"])
        target_fg.append(stats["target_fg"])
        fg_probabilities.append(stats["fg_prob"])
        bg_probabilities.append(stats["bg_prob"])

        if stats["pred_fg"] == 0:
            empty_cases += 1

    return {
        "loss": float(np.mean(losses)),
        "fg_dice": float(np.mean(dices)),
        "pred_fg": float(np.mean(predicted_fg)),
        "target_fg": float(np.mean(target_fg)),
        "fg_prob": float(np.mean(fg_probabilities)),
        "bg_prob": float(np.mean(bg_probabilities)),
        "empty": int(empty_cases),
        "n": len(cases),
    }


# ================================================================
# TRAIN ONE CONDITION
# ================================================================

def train_condition(
    condition_name: str,
    output_dir: Path,
    train_cases,
    val_cases,
    part11,
    device,
    loss_fn,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = create_model(
        part11,
        device,
    )

    load_initialization(
        model,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
    )

    history = []

    # ------------------------------------------------------------
    # INITIALIZATION EVALUATION
    # ------------------------------------------------------------

    initial_eval = evaluate(
        model,
        val_cases,
        loss_fn,
        device,
    )

    print(
        f"Initialization: "
        f"loss={initial_eval['loss']:.6f} "
        f"FGDice={initial_eval['fg_dice']:.6f} "
        f"PredFG={initial_eval['pred_fg']:.1f} "
        f"FGProb={initial_eval['fg_prob']:.6f} "
        f"empty={initial_eval['empty']}/{initial_eval['n']}"
    )

    history.append(
        {
            "epoch": 0,
            "phase": "initialization",
            "train_loss": "",
            "train_fg_dice": "",
            "val_loss": initial_eval["loss"],
            "val_fg_dice": initial_eval["fg_dice"],
            "val_pred_fg": initial_eval["pred_fg"],
            "val_target_fg": initial_eval["target_fg"],
            "val_fg_prob": initial_eval["fg_prob"],
            "val_bg_prob": initial_eval["bg_prob"],
            "val_empty": initial_eval["empty"],
            "elapsed_seconds": 0.0,
        }
    )

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        start_time = time.time()

        model.train()

        train_losses = []
        train_dices = []

        for batch_idx, (image_np, mask_np) in enumerate(
            train_cases,
            start=1,
        ):

            x = (
                torch.from_numpy(image_np)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            y = (
                torch.from_numpy(mask_np)
                .unsqueeze(0)
                .to(device)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = loss_fn(
                logits,
                y.unsqueeze(1),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            stats = prediction_stats(
                logits.detach(),
                y,
            )

            train_losses.append(
                float(loss.item())
            )

            train_dices.append(
                stats["fg_dice"]
            )

            if (
                batch_idx == 1
                or batch_idx == len(train_cases)
            ):
                print(
                    f"  batch {batch_idx:03d}/"
                    f"{len(train_cases)} "
                    f"loss={loss.item():.6f} "
                    f"FGDice={stats['fg_dice']:.6f} "
                    f"PredFG={stats['pred_fg']}"
                )

        train_loss = float(
            np.mean(train_losses)
        )

        train_dice = float(
            np.mean(train_dices)
        )

        validation = evaluate(
            model,
            val_cases,
            loss_fn,
            device,
        )

        elapsed = (
            time.time() - start_time
        )

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}: "
            f"train_loss={train_loss:.6f}, "
            f"train_FGDice={train_dice:.6f}, "
            f"val_loss={validation['loss']:.6f}, "
            f"val_FGDice={validation['fg_dice']:.6f}, "
            f"PredFG={validation['pred_fg']:.1f}, "
            f"FGProb={validation['fg_prob']:.6f}, "
            f"empty={validation['empty']}/"
            f"{validation['n']}, "
            f"time={elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "condition": condition_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_description": (
                "Dice + CE"
                if condition_name == "original_dicece"
                else "Dice-only"
            ),
        }

        torch.save(
            checkpoint,
            output_dir / f"epoch_{epoch:02d}.pth",
        )

        history.append(
            {
                "epoch": epoch,
                "phase": "training",
                "train_loss": train_loss,
                "train_fg_dice": train_dice,
                "val_loss": validation["loss"],
                "val_fg_dice": validation["fg_dice"],
                "val_pred_fg": validation["pred_fg"],
                "val_target_fg": validation["target_fg"],
                "val_fg_prob": validation["fg_prob"],
                "val_bg_prob": validation["bg_prob"],
                "val_empty": validation["empty"],
                "elapsed_seconds": elapsed,
            }
        )

    # ------------------------------------------------------------
    # SAVE HISTORY
    # ------------------------------------------------------------

    with open(
        output_dir / "history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    # Use the union of all dictionary keys.
    # This prevents the initialization/training row mismatch
    # that caused the previous Part 50 failure.
    all_fields = []

    for row in history:
        for key in row.keys():
            if key not in all_fields:
                all_fields.append(key)

    with open(
        output_dir / "history.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(history)

    return history


# ================================================================
# MAIN
# ================================================================

def main():

    seed_everything(SEED)

    validate_paths()

    print()
    print("=" * 82)
    print(
        "PART 50 — DICE-ONLY VS DICE+CE "
        "TRAINING-DYNAMICS DIAGNOSTIC"
    )
    print("=" * 82)

    print(f"PyTorch : {torch.__version__}")

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device : {device}")
    print(f"Train subset : {TRAIN_N}")
    print(f"Validation subset : {VAL_N}")
    print(f"Epochs : {EPOCHS}")
    print(f"Full volume : {FULL_SHAPE}")
    print(f"Crop : {CROP_SHAPE}")
    print(f"Learning rate : {LR}")
    print(f"Weight decay : {WD}")
    print("Crop policy : foreground-centered")
    print("Condition A : original Dice + unweighted CE")
    print("Condition B : Dice-only (CE removed)")
    print("Part 15 overwritten : NO")
    print("SPIDER used : NO")
    print("Test set used : NO")
    print(
        f"Initialization SHA256 : "
        f"{sha256_file(INIT)}"
    )

    # ------------------------------------------------------------
    # LOAD PROJECT MODULES
    # ------------------------------------------------------------

    part11 = load_module(
        PART11_PATH,
        "segmentation_rsna_part11_part50_clean_runtime",
    )

    part9 = load_module(
        PART9_PATH,
        "segmentation_rsna_part9_part50_clean_runtime",
    )

    # ------------------------------------------------------------
    # PRELOAD DATA
    # ------------------------------------------------------------

    train_cases = load_cases(
        part11,
        part9,
        TRCSV,
        TRAIN_N,
        "train",
    )

    val_cases = load_cases(
        part11,
        part9,
        VACSV,
        VAL_N,
        "validation",
    )

    # ------------------------------------------------------------
    # SHAPE SMOKE TEST
    # ------------------------------------------------------------

    print()
    print("=" * 82)
    print("PART 50 SHAPE SMOKE TEST")
    print("-" * 82)

    for i, (image, mask) in enumerate(
        train_cases[:3],
        start=1,
    ):
        print(
            f"Case {i}: "
            f"image={image.shape} "
            f"mask={mask.shape} "
            f"FG={(mask > 0).sum()}"
        )

        assert tuple(image.shape) == CROP_SHAPE
        assert tuple(mask.shape) == CROP_SHAPE

    print("✓ Shape smoke test PASSED.")

    # ------------------------------------------------------------
    # LOSS DEFINITIONS
    # ------------------------------------------------------------

    original_loss, dice_only_loss = create_losses()

    # ------------------------------------------------------------
    # IDENTICAL INITIALIZATION CHECK
    # ------------------------------------------------------------

    print()
    print("=" * 82)
    print("PART 50 CONTROL CHECK")
    print("-" * 82)

    model_a = create_model(
        part11,
        device,
    )

    model_b = create_model(
        part11,
        device,
    )

    load_initialization(
        model_a,
        device,
    )

    load_initialization(
        model_b,
        device,
    )

    identical = all(
        torch.equal(
            parameter_a.detach().cpu(),
            parameter_b.detach().cpu(),
        )
        for parameter_a, parameter_b in zip(
            model_a.parameters(),
            model_b.parameters(),
        )
    )

    print(
        f"Identical model initialization : "
        f"{identical}"
    )

    del model_a
    del model_b

    # ------------------------------------------------------------
    # OUTPUT DIRECTORIES
    # ------------------------------------------------------------

    BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    DICE_ONLY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # CONDITION A
    # ------------------------------------------------------------

    print()
    print("=" * 82)
    print("CONDITION A — ORIGINAL DICECE")
    print("=" * 82)

    seed_everything(SEED)

    baseline_history = train_condition(
        "original_dicece",
        BASE,
        train_cases,
        val_cases,
        part11,
        device,
        original_loss,
    )

    # ------------------------------------------------------------
    # CONDITION B
    # ------------------------------------------------------------

    print()
    print("=" * 82)
    print("CONDITION B — DICE-ONLY")
    print("=" * 82)

    seed_everything(SEED)

    dice_only_history = train_condition(
        "dice_only",
        DICE_ONLY,
        train_cases,
        val_cases,
        part11,
        device,
        dice_only_loss,
    )

    # ------------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------------

    baseline_final = baseline_history[-1]
    dice_only_final = dice_only_history[-1]

    dice_delta = (
        dice_only_final["val_fg_dice"]
        - baseline_final["val_fg_dice"]
    )

    empty_delta = (
        baseline_final["val_empty"]
        - dice_only_final["val_empty"]
    )

    if (
        dice_delta > 0.02
        and dice_only_final["val_empty"]
        < baseline_final["val_empty"]
    ):
        interpretation = (
            "REMOVING_CE_APPEARS_TO_REDUCE_COLLAPSE"
        )

    elif abs(dice_delta) < 0.01:
        interpretation = (
            "DICE_ONLY_DID_NOT_MEANINGFULLY_CHANGE_FOREGROUND_DICE"
        )

    else:
        interpretation = (
            "DICE_ONLY_CHANGED_BEHAVIOR_BUT_NOT_DECISIVELY"
        )

    comparison = {
        "part": 50,
        "initialization_sha256": sha256_file(INIT),
        "train_subset": TRAIN_N,
        "validation_subset": VAL_N,
        "epochs": EPOCHS,
        "full_shape": FULL_SHAPE,
        "crop_shape": CROP_SHAPE,
        "crop_policy": "foreground-centered",
        "learning_rate": LR,
        "weight_decay": WD,
        "baseline_final_val_loss": baseline_final["val_loss"],
        "dice_only_final_val_loss": dice_only_final["val_loss"],
        "baseline_final_fg_dice": baseline_final["val_fg_dice"],
        "dice_only_final_fg_dice": dice_only_final["val_fg_dice"],
        "fg_dice_delta": dice_delta,
        "baseline_pred_fg": baseline_final["val_pred_fg"],
        "dice_only_pred_fg": dice_only_final["val_pred_fg"],
        "baseline_target_fg": baseline_final["val_target_fg"],
        "dice_only_target_fg": dice_only_final["val_target_fg"],
        "baseline_fg_prob": baseline_final["val_fg_prob"],
        "dice_only_fg_prob": dice_only_final["val_fg_prob"],
        "baseline_bg_prob": baseline_final["val_bg_prob"],
        "dice_only_bg_prob": dice_only_final["val_bg_prob"],
        "baseline_empty": baseline_final["val_empty"],
        "dice_only_empty": dice_only_final["val_empty"],
        "empty_case_delta": empty_delta,
        "interpretation": interpretation,
    }

    with open(
        REPORT / "part50_comparison.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            comparison,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------

    print()
    print("=" * 82)
    print("PART 50 COMPARISON")
    print("=" * 82)

    print(
        f"Baseline final val loss : "
        f"{baseline_final['val_loss']:.6f}"
    )

    print(
        f"Dice-only final val loss : "
        f"{dice_only_final['val_loss']:.6f}"
    )

    print(
        f"Baseline final FG Dice : "
        f"{baseline_final['val_fg_dice']:.6f}"
    )

    print(
        f"Dice-only final FG Dice : "
        f"{dice_only_final['val_fg_dice']:.6f}"
    )

    print(
        f"FG Dice delta : "
        f"{dice_delta:+.6f}"
    )

    print(
        f"Baseline PredFG : "
        f"{baseline_final['val_pred_fg']:.1f}"
    )

    print(
        f"Dice-only PredFG : "
        f"{dice_only_final['val_pred_fg']:.1f}"
    )

    print(
        f"Baseline FG probability : "
        f"{baseline_final['val_fg_prob']:.6f}"
    )

    print(
        f"Dice-only FG probability : "
        f"{dice_only_final['val_fg_prob']:.6f}"
    )

    print(
        f"Baseline empty cases : "
        f"{baseline_final['val_empty']}/{VAL_N}"
    )

    print(
        f"Dice-only empty cases : "
        f"{dice_only_final['val_empty']}/{VAL_N}"
    )

    print(
        f"Interpretation : "
        f"{interpretation}"
    )

    print()
    print("=" * 82)
    print("PART 50 COMPLETE")
    print("=" * 82)

    print(
        f"Comparison : "
        f"{REPORT / 'part50_comparison.json'}"
    )

    print(
        f"Baseline history : "
        f"{BASE / 'history.csv'}"
    )

    print(
        f"Dice-only history : "
        f"{DICE_ONLY / 'history.csv'}"
    )


if __name__ == "__main__":
    main()
