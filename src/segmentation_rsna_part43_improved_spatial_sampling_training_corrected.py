"""
PART 43 — RSNA-ONLY CONTROLLED SPATIAL-SAMPLING TRAINING

Compares:
1. Random spatial cropping
2. Foreground-centered spatial cropping

Locked design:
- Original Part 15 initialization checkpoint
- 200 training cases
- 100 validation cases
- 5 epochs
- crop size (32, 64, 64)
- DiceCELoss(to_onehot_y=True, softmax=True)
- AdamW, LR=1e-4, WD=1e-5
- Same foreground-centered validation crop for both conditions
- Part 15 is never overwritten
- No SPIDER
- No test set
"""

from __future__ import annotations

import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR


# ============================================================================
# PATHS
# ============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

RSNA_ROOT = PROJECT_ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification"
TRAIN_IMAGES_DIR = RSNA_ROOT / "train_images"

PART8_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part8_dataset_construction"
MANIFEST_DIR = PART8_DIR / "manifests"
TRAIN_MANIFEST = MANIFEST_DIR / "rsna_part8_train_manifest.csv"
VAL_MANIFEST = MANIFEST_DIR / "rsna_part8_validation_manifest.csv"

PART6_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part6_pseudomask_generation"
PSEUDOMASK_DIR = PART6_DIR / "pseudo_masks"

PART15_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
INITIALIZATION_CHECKPOINT = PART15_DIR / "checkpoints" / "part15_initialization_from_part11.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part43_improved_spatial_sampling_training"
RANDOM_OUTPUT_DIR = OUTPUT_DIR / "random_spatial_crop"
CENTERED_OUTPUT_DIR = OUTPUT_DIR / "foreground_centered_spatial_crop"
REPORT_DIR = OUTPUT_DIR / "reports"


# ============================================================================
# LOCKED CONFIGURATION
# ============================================================================

FULL_VOLUME = (64, 96, 96)
CROP_SIZE = (32, 64, 64)

IN_CHANNELS = 1
NUM_CLASSES = 6
FEATURE_SIZE = 12

TRAIN_CASES = 200
VAL_CASES = 100
EPOCHS = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

SEED = 42

EXPECTED_INITIALIZATION_SHA256 = (
    "0900c0e6490fdaddf455763b465d4a607acfb310349c9f9ec82a61117016aec6"
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cpu":
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


# ============================================================================
# UTILITIES
# ============================================================================

def header(title: str) -> None:
    print()
    print("=" * 82)
    print(title)
    print("=" * 82)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ============================================================================
# MODULE IMPORTS
# ============================================================================

def load_modules() -> Tuple[Any, Any]:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    import segmentation_rsna_part9_3d_dataset_loader as part9
    import segmentation_rsna_part11_controlled_pilot_training as part11

    if not hasattr(part9, "load_case"):
        raise RuntimeError("Part 9 does not expose load_case(row).")

    for name in ("create_model", "load_tensor_case", "dice_from_prediction"):
        if not hasattr(part11, name):
            raise RuntimeError(f"Part 11 is missing required API: {name}")

    return part9, part11


# ============================================================================
# MODEL / LOSS
# ============================================================================

def create_model(part11: Any) -> torch.nn.Module:
    return part11.create_model(DEVICE).to(DEVICE)


def create_loss() -> torch.nn.Module:
    # Exact Part 15/Part 11 DiceCELoss configuration.
    # Part 11 imports DiceCELoss but does not expose part11.loss_function.
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    ).to(DEVICE)


def load_initial_state(part11: Any) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(
        INITIALIZATION_CHECKPOINT,
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

    model = create_model(part11)
    model.load_state_dict(cleaned, strict=True)
    del model
    reset_cuda_memory()

    return cleaned


# ============================================================================
# CASE LOADING
# ============================================================================

def load_case(
    row: pd.Series,
    part9: Any,
    part11: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:

    loaded = part11.load_tensor_case(row, part9)

    if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
        raise RuntimeError("Unexpected Part 11 load_tensor_case() result.")

    image = torch.as_tensor(loaded[0])
    mask = torch.as_tensor(loaded[1])

    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    if tuple(image.shape) != FULL_VOLUME:
        raise RuntimeError(
            f"Expected image {FULL_VOLUME}, got {tuple(image.shape)}"
        )
    if tuple(mask.shape) != FULL_VOLUME:
        raise RuntimeError(
            f"Expected mask {FULL_VOLUME}, got {tuple(mask.shape)}"
        )

    return image.float(), mask.long()


# ============================================================================
# CROPPING
# ============================================================================

def random_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    d, h, w = image.shape
    cd, ch, cw = CROP_SIZE

    sd = random.randint(0, d - cd)
    sh = random.randint(0, h - ch)
    sw = random.randint(0, w - cw)

    sl = (
        slice(sd, sd + cd),
        slice(sh, sh + ch),
        slice(sw, sw + cw),
    )

    return image[sl], mask[sl]


def centered_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    coords = torch.nonzero(mask > 0, as_tuple=False)

    if coords.numel() == 0:
        center = (
            image.shape[0] // 2,
            image.shape[1] // 2,
            image.shape[2] // 2,
        )
    else:
        center = tuple(
            int(v)
            for v in coords.float().mean(dim=0).tolist()
        )

    starts = []
    for dim, crop, c in zip(image.shape, CROP_SIZE, center):
        start = int(round(c - crop / 2))
        start = max(0, min(start, dim - crop))
        starts.append(start)

    sd, sh, sw = starts

    sl = (
        slice(sd, sd + CROP_SIZE[0]),
        slice(sh, sh + CROP_SIZE[1]),
        slice(sw, sw + CROP_SIZE[2]),
    )

    return image[sl], mask[sl]


def prepare_case(
    row: pd.Series,
    strategy: str,
    part9: Any,
    part11: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:

    image, mask = load_case(row, part9, part11)

    if strategy == "random":
        image, mask = random_crop(image, mask)
    elif strategy == "centered":
        image, mask = centered_crop(image, mask)
    else:
        raise ValueError(f"Unknown crop strategy: {strategy}")

    # image -> [B,C,D,H,W]
    image = image.unsqueeze(0).unsqueeze(0)

    # mask -> [B,1,D,H,W]
    mask = mask.unsqueeze(0).unsqueeze(0)

    return image, mask


# ============================================================================
# DICE
# ============================================================================

def calculate_dice(
    part11: Any,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[float, List[float], int, int]:

    result = part11.dice_from_prediction(logits, target)

    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(
            "Unexpected Part 11 dice_from_prediction() result."
        )

    dice = float(result[0])
    class_dice = list(result[1])

    pred = torch.argmax(logits, dim=1)
    target_classes = target.squeeze(1)

    pred_fg = int((pred > 0).sum().item())
    target_fg = int((target_classes > 0).sum().item())

    return dice, class_dice, pred_fg, target_fg


# ============================================================================
# COVERAGE AUDIT
# ============================================================================

def coverage_audit(
    rows: pd.DataFrame,
    part9: Any,
    part11: Any,
) -> Dict[str, Any]:

    source_values = []
    random_values = []
    centered_values = []

    random_zero = 0
    centered_zero = 0

    for i, (_, row) in enumerate(rows.iterrows(), start=1):

        image, mask = load_case(row, part9, part11)

        source_fg = int((mask > 0).sum().item())

        _, rm = random_crop(image, mask)
        _, cm = centered_crop(image, mask)

        random_fg = int((rm > 0).sum().item())
        centered_fg = int((cm > 0).sum().item())

        source_values.append(source_fg)
        random_values.append(random_fg)
        centered_values.append(centered_fg)

        random_zero += int(random_fg == 0)
        centered_zero += int(centered_fg == 0)

        if i == 1 or i % 50 == 0 or i == len(rows):
            print(
                f"  COVERAGE {i:03d}/{len(rows)} "
                f"sourceFG={source_fg} "
                f"randomFG={random_fg} "
                f"centeredFG={centered_fg}"
            )

    return {
        "mean_source_fg": float(np.mean(source_values)),
        "mean_random_fg": float(np.mean(random_values)),
        "mean_centered_fg": float(np.mean(centered_values)),
        "random_zero_fg": random_zero,
        "centered_zero_fg": centered_zero,
    }


# ============================================================================
# SHAPE TEST
# ============================================================================

def shape_smoke_test(
    rows: pd.DataFrame,
    part9: Any,
    part11: Any,
) -> None:

    header("PART 43 SHAPE SMOKE TEST")

    for i in range(min(3, len(rows))):

        _, row = next(rows.iloc[[i]].iterrows())

        image, mask = load_case(row, part9, part11)
        ri, rm = random_crop(image, mask)
        ci, cm = centered_crop(image, mask)

        print(
            f"Case {i + 1}: "
            f"full={tuple(image.shape)} "
            f"random={tuple(ri.shape)} "
            f"randomFG={int((rm > 0).sum().item())} "
            f"centered={tuple(ci.shape)} "
            f"centeredFG={int((cm > 0).sum().item())}"
        )

        assert tuple(image.shape) == FULL_VOLUME
        assert tuple(mask.shape) == FULL_VOLUME
        assert tuple(ri.shape) == CROP_SIZE
        assert tuple(rm.shape) == CROP_SIZE
        assert tuple(ci.shape) == CROP_SIZE
        assert tuple(cm.shape) == CROP_SIZE

    print("✓ Shape smoke test PASSED.")


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    rows: pd.DataFrame,
    part9: Any,
    part11: Any,
    loss_fn: torch.nn.Module,
) -> Dict[str, Any]:

    model.eval()

    losses = []
    dices = []
    pred_fg = []
    target_fg = []

    empty = 0

    for _, row in rows.iterrows():

        # SAME validation policy for both conditions.
        x, y = prepare_case(
            row,
            "centered",
            part9,
            part11,
        )

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = loss_fn(logits, y)

        dice, _, pfg, tfg = calculate_dice(
            part11,
            logits,
            y,
        )

        losses.append(float(loss.item()))
        dices.append(dice)
        pred_fg.append(pfg)
        target_fg.append(tfg)

        if pfg == 0:
            empty += 1

        del x, y, logits

    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "pred_fg": float(np.mean(pred_fg)),
        "target_fg": float(np.mean(target_fg)),
        "empty": empty,
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    model: torch.nn.Module,
    rows: pd.DataFrame,
    strategy: str,
    part9: Any,
    part11: Any,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:

    model.train()

    losses = []
    dices = []

    for _, row in rows.iterrows():

        x, y = prepare_case(
            row,
            strategy,
            part9,
            part11,
        )

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dice, _, _, _ = calculate_dice(
                part11,
                logits,
                y,
            )

        losses.append(float(loss.item()))
        dices.append(dice)

        del x, y, logits

    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
    }


# ============================================================================
# CONDITION
# ============================================================================

def run_condition(
    name: str,
    strategy: str,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    part9: Any,
    part11: Any,
    initial_state: Dict[str, torch.Tensor],
    output_dir: Path,
) -> Dict[str, Any]:

    header(name)

    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(part11)
    model.load_state_dict(initial_state, strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = create_loss()

    initial = evaluate(
        model,
        val_rows,
        part9,
        part11,
        loss_fn,
    )

    print(
        f"Initialization: "
        f"loss={initial['loss']:.6f}, "
        f"Dice={initial['dice']:.6f}, "
        f"PredFG={initial['pred_fg']:.1f}, "
        f"empty={initial['empty']}/{len(val_rows)}"
    )

    history = [{
        "epoch": 0,
        "train_loss": None,
        "train_dice": None,
        "val_loss": initial["loss"],
        "val_dice": initial["dice"],
        "pred_fg": initial["pred_fg"],
        "target_fg": initial["target_fg"],
        "empty": initial["empty"],
    }]

    best_dice = initial["dice"]

    for epoch in range(1, EPOCHS + 1):

        start = time.time()

        train_metrics = train_epoch(
            model,
            train_rows,
            strategy,
            part9,
            part11,
            loss_fn,
            optimizer,
        )

        val_metrics = evaluate(
            model,
            val_rows,
            part9,
            part11,
            loss_fn,
        )

        elapsed = time.time() - start

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "pred_fg": val_metrics["pred_fg"],
            "target_fg": val_metrics["target_fg"],
            "empty": val_metrics["empty"],
            "elapsed_seconds": elapsed,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS}: "
            f"train_loss={train_metrics['loss']:.6f}, "
            f"train_Dice={train_metrics['dice']:.6f}, "
            f"val_loss={val_metrics['loss']:.6f}, "
            f"val_Dice={val_metrics['dice']:.6f}, "
            f"PredFG={val_metrics['pred_fg']:.1f}, "
            f"empty={val_metrics['empty']}/{len(val_rows)}, "
            f"time={elapsed:.1f}s"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "condition": name,
                "strategy": strategy,
                "initialization_sha256": EXPECTED_INITIALIZATION_SHA256,
            },
            output_dir / f"epoch_{epoch:02d}.pth",
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "condition": name,
                    "strategy": strategy,
                    "initialization_sha256": EXPECTED_INITIALIZATION_SHA256,
                },
                output_dir / "best_model.pth",
            )

        reset_cuda_memory()

    result = {
        "condition": name,
        "strategy": strategy,
        "final": history[-1],
        "best_val_dice": best_dice,
        "history": history,
        "output_dir": str(output_dir),
    }

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    del model, optimizer, loss_fn
    reset_cuda_memory()

    return result


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    set_seed(SEED)

    header("PART 43 PATH VALIDATION")

    required = {
        "Project root": PROJECT_ROOT,
        "Part 11": SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training.py",
        "Part 9": SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py",
        "Part 15 initialization checkpoint": INITIALIZATION_CHECKPOINT,
        "Part 15 train cohort": TRAIN_MANIFEST,
        "Part 15 validation cohort": VAL_MANIFEST,
    }

    for name, path in required.items():
        print(f"{name:<42}: {'FOUND' if path.exists() else 'MISSING'}")

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required paths: " + ", ".join(missing)
        )

    actual_sha = sha256_file(INITIALIZATION_CHECKPOINT)

    if actual_sha != EXPECTED_INITIALIZATION_SHA256:
        raise RuntimeError(
            "Initialization SHA256 mismatch.\n"
            f"Expected: {EXPECTED_INITIALIZATION_SHA256}\n"
            f"Actual:   {actual_sha}"
        )

    header("PART 43 — IMPROVED CONTROLLED SPATIAL-SAMPLING TRAINING")

    print(f"PyTorch                                    : {torch.__version__}")
    print(f"Device                                     : {DEVICE}")
    print(f"Full volume                                : {FULL_VOLUME}")
    print(f"Training crop                              : {CROP_SIZE}")
    print(f"Train subset                               : {TRAIN_CASES}")
    print(f"Validation subset                          : {VAL_CASES}")
    print(f"Epochs                                     : {EPOCHS}")
    print(f"Learning rate                              : {LEARNING_RATE}")
    print(f"Weight decay                               : {WEIGHT_DECAY}")
    print("Validation crop policy                     : same foreground-centered crop for both")
    print("Part 15 overwritten                        : NO")
    print("SPIDER used                                : NO")
    print("Test set used                              : NO")
    print(f"Initialization SHA256                     : {actual_sha}")
    print("Loss                                       : DiceCELoss(to_onehot_y=True, softmax=True)")

    part9, part11 = load_modules()

    full_train = pd.read_csv(TRAIN_MANIFEST)
    full_val = pd.read_csv(VAL_MANIFEST)

    print(f"Exact Part 15 train cohort rows           : {len(full_train)}")
    print(f"Exact Part 15 validation cohort rows      : {len(full_val)}")

    train_rows = full_train.iloc[:TRAIN_CASES].copy().reset_index(drop=True)
    val_rows = full_val.iloc[:VAL_CASES].copy().reset_index(drop=True)

    print(f"Controlled train subset                   : {len(train_rows)}")
    print(f"Controlled validation subset              : {len(val_rows)}")

    header("PART 43 CROP COVERAGE AUDIT")

    coverage = coverage_audit(
        train_rows,
        part9,
        part11,
    )

    print(f"Mean source foreground voxels       : {coverage['mean_source_fg']:.2f}")
    print(f"Mean random-crop foreground voxels : {coverage['mean_random_fg']:.2f}")
    print(f"Mean centered-crop foreground      : {coverage['mean_centered_fg']:.2f}")
    print(f"Random crops with zero FG           : {coverage['random_zero_fg']}/{TRAIN_CASES}")
    print(f"Centered crops with zero FG         : {coverage['centered_zero_fg']}/{TRAIN_CASES}")

    shape_smoke_test(
        train_rows,
        part9,
        part11,
    )

    initial_state = load_initial_state(part11)

    # Both conditions receive the exact same model initialization.
    random_result = run_condition(
        "RANDOM SPATIAL CROPPING",
        "random",
        train_rows,
        val_rows,
        part9,
        part11,
        initial_state,
        RANDOM_OUTPUT_DIR,
    )

    # Reset all random generators before the second controlled condition.
    set_seed(SEED)

    centered_result = run_condition(
        "FOREGROUND-CENTERED SPATIAL CROPPING",
        "centered",
        train_rows,
        val_rows,
        part9,
        part11,
        initial_state,
        CENTERED_OUTPUT_DIR,
    )

    header("PART 43 CONTROLLED COMPARISON")

    rd = float(random_result["final"]["val_dice"])
    cd = float(centered_result["final"]["val_dice"])

    rl = float(random_result["final"]["val_loss"])
    cl = float(centered_result["final"]["val_loss"])

    dice_delta = cd - rd
    loss_delta = cl - rl

    print(f"Random final validation loss       : {rl:.7f}")
    print(f"Centered final validation loss     : {cl:.7f}")
    print(f"Random final validation Dice       : {rd:.7f}")
    print(f"Centered final validation Dice     : {cd:.7f}")
    print(f"Centered minus random Dice delta   : {dice_delta:+.7f}")
    print(f"Random final validation PredFG     : {random_result['final']['pred_fg']:.2f}")
    print(f"Centered final validation PredFG   : {centered_result['final']['pred_fg']:.2f}")
    print(f"Random empty validation cases      : {random_result['final']['empty']}/{VAL_CASES}")
    print(f"Centered empty validation cases    : {centered_result['final']['empty']}/{VAL_CASES}")
    print(f"Mean random crop FG                : {coverage['mean_random_fg']:.2f}")
    print(f"Mean centered crop FG              : {coverage['mean_centered_fg']:.2f}")
    print("Initialization identical           : TRUE")

    if abs(dice_delta) >= 0.01:
        interpretation = "FOREGROUND_CENTERED_SPATIAL_SAMPLING_CHANGED_BEHAVIOR"
    else:
        interpretation = "NO_LARGE_DICE_DIFFERENCE_IN_THIS_CONTROLLED_PILOT"

    print(f"Interpretation                      : {interpretation}")

    summary = {
        "part": 43,
        "device": str(DEVICE),
        "full_volume": FULL_VOLUME,
        "crop_size": CROP_SIZE,
        "train_cases": TRAIN_CASES,
        "validation_cases": VAL_CASES,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": {
            "name": "DiceCELoss",
            "to_onehot_y": True,
            "softmax": True,
        },
        "initialization_sha256": actual_sha,
        "coverage": coverage,
        "random": random_result,
        "centered": centered_result,
        "comparison": {
            "dice_delta_centered_minus_random": dice_delta,
            "loss_delta_centered_minus_random": loss_delta,
            "interpretation": interpretation,
        },
        "part15_overwritten": False,
        "spider_used": False,
        "test_set_used": False,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with (REPORT_DIR / "part43_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {REPORT_DIR / 'part43_summary.json'}")

    header("PART 43 COMPLETE")


if __name__ == "__main__":
    main()
