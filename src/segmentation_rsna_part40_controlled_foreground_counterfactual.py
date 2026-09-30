"""
PART 40
Controlled Foreground-Aware Counterfactual Training Experiment

Purpose
-------
Test whether foreground-aware optimization changes the rapid foreground
suppression observed in Part 39, while preserving the validated RSNA
preprocessing, Swin-UNETR architecture, patch size, cohort source, and
optimizer settings.

This is a SMALL CONTROLLED PILOT, not final training.

Conditions
----------
A. BASELINE:
   Original Part 15 DiceCELoss configuration.

B. FOREGROUND-AWARE:
   Same DiceCELoss, but with class-weighted CE inside DiceCELoss.
   Background receives weight 0.25; each foreground class receives 1.0.
   The relative foreground/background CE emphasis is therefore 4x.

Both conditions:
- start from the SAME Part 11 initialization checkpoint
- use the SAME exact Part 15 cohort files
- use the SAME deterministic subset of those cohorts
- use the SAME architecture, optimizer, LR, WD, patch and AMP
- train independently into separate output folders
- never overwrite Part 15 checkpoints

The experiment measures whether foreground prediction survives training.
It does NOT establish clinical segmentation performance.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
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
from torch.cuda.amp import GradScaler, autocast


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
SRC_DIR = PROJECT_ROOT / "src"

PART11_SOURCE = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training.py"
PART15_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
PART15_INIT = PART15_DIR / "checkpoints" / "part15_initialization_from_part11.pth"
PART15_TRAIN_COHORT = PART15_DIR / "part15_train_cohort.csv"
PART15_VAL_COHORT = PART15_DIR / "part15_validation_cohort.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part40_controlled_foreground_counterfactual"
)

BASELINE_DIR = OUTPUT_DIR / "baseline_original_dicece"
FOREGROUND_DIR = OUTPUT_DIR / "foreground_aware_weighted_ce"


# =============================================================================
# CONTROLLED PILOT CONFIGURATION
# =============================================================================

SEED = 42
TRAIN_CASES = 100
VAL_CASES = 50
EPOCHS = 3
BATCH_SIZE = 1
PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12
NUM_CLASSES = 6
LR = 1e-4
WEIGHT_DECAY = 1e-5
AMP_ENABLED = True
GRAD_CLIP = 1.0

# Fixed class weights for the counterfactual condition.
# Background is deliberately reduced rather than foreground weights being
# made extremely large, keeping the experiment numerically conservative.
BASELINE_CLASS_WEIGHTS = None
FOREGROUND_AWARE_CLASS_WEIGHTS = [0.25, 1.0, 1.0, 1.0, 1.0, 1.0]

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# =============================================================================
# HELPERS
# =============================================================================

def banner(title: str) -> None:
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def status(label: str, value: Any) -> None:
    print(f"{label:<42}: {value}")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def require_paths() -> None:
    banner("PART 40 PATH VALIDATION")
    required = {
        "Project root": PROJECT_ROOT,
        "Part 11 source": PART11_SOURCE,
        "Part 15 initialization checkpoint": PART15_INIT,
        "Part 15 train cohort": PART15_TRAIN_COHORT,
        "Part 15 validation cohort": PART15_VAL_COHORT,
    }
    missing = []
    for name, path in required.items():
        ok = path.exists()
        status(name, "FOUND" if ok else "MISSING")
        if not ok:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required Part 40 input(s):\n" + "\n".join(missing))

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    FOREGROUND_DIR.mkdir(parents=True, exist_ok=True)


def import_part11():
    banner("IMPORTING VALIDATED PART 11")
    spec = importlib.util.spec_from_file_location("part11_for_part40", PART11_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import Part 11: {PART11_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["part11_for_part40"] = module
    spec.loader.exec_module(module)

    required = [
        "select_pilot_rows",
        "load_part9_module",
        "load_tensor_case",
        "dice_from_prediction",
        "create_model",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError("Part 11 missing required API: " + ", ".join(missing))

    for name in required:
        print(f"✓ {name}")
    return module


def load_part9(part11):
    banner("LOADING PART 9 THROUGH PART 11")
    part9 = part11.load_part9_module()
    print("✓ Part 9 loader imported")
    return part9


def load_cohorts() -> Tuple[pd.DataFrame, pd.DataFrame]:
    banner("LOADING EXACT PART 15 COHORT FILES")
    train = pd.read_csv(PART15_TRAIN_COHORT)
    val = pd.read_csv(PART15_VAL_COHORT)

    status("Part 15 train cohort rows", len(train))
    status("Part 15 validation cohort rows", len(val))

    if len(train) != 500 or len(val) != 100:
        raise RuntimeError(
            f"Unexpected Part 15 cohort sizes: train={len(train)}, val={len(val)}"
        )

    # Deterministic subset of the exact Part 15 cohorts.
    train_subset = train.iloc[:TRAIN_CASES].copy().reset_index(drop=True)
    val_subset = val.iloc[:VAL_CASES].copy().reset_index(drop=True)

    status("Controlled train subset", len(train_subset))
    status("Controlled validation subset", len(val_subset))
    return train_subset, val_subset


def normalize_volume(t: torch.Tensor, name: str) -> torch.Tensor:
    t = torch.as_tensor(t)
    t = t.detach().cpu()

    # Valid Part 11 image can arrive as [1,D,H,W].
    if name == "image" and t.ndim == 4 and t.shape[0] == 1:
        t = t.squeeze(0)

    # Mask should be [D,H,W].
    if name == "mask" and t.ndim == 4 and t.shape[0] == 1:
        t = t.squeeze(0)

    if t.ndim != 3:
        raise RuntimeError(f"{name} has invalid shape after normalization: {tuple(t.shape)}")

    if tuple(t.shape) != PATCH_SIZE:
        raise RuntimeError(
            f"{name} shape {tuple(t.shape)} does not match expected {PATCH_SIZE}"
        )
    return t


def load_case(part11, part9, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    image, mask, info = part11.load_tensor_case(row, part9)
    image = normalize_volume(image, "image").float()
    mask = normalize_volume(mask, "mask").long()
    return image, mask, info


def make_model(part11, device: torch.device) -> torch.nn.Module:
    model = part11.create_model(device)
    model.to(device)
    return model


def load_initialization(model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(PART15_INIT, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unexpected Part 15 initialization checkpoint format.")

    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise KeyError("Part 15 initialization checkpoint has no model state dict.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Initialization checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    model.to(device)
    return checkpoint


def make_loss(condition: str, device: torch.device):
    if condition == "baseline":
        return DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
        )

    if condition == "foreground_aware":
        weight = torch.tensor(FOREGROUND_AWARE_CLASS_WEIGHTS, dtype=torch.float32, device=device)
        return DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            weight=weight,
        )

    raise ValueError(condition)


def safe_dice(part11, logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, List[float]]:
    result = part11.dice_from_prediction(logits.detach(), target.detach())
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"Unexpected dice_from_prediction result: {type(result)}")
    mean_dice, class_dice = result
    return float(mean_dice), [float(x) for x in class_dice]


def foreground_statistics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
    pred = torch.argmax(logits.detach(), dim=1)
    target = target.detach()

    target_fg = int((target > 0).sum().item())
    pred_fg = int((pred > 0).sum().item())
    total = int(target.numel())

    probabilities = torch.softmax(logits.detach().float(), dim=1)
    bg_prob = float(probabilities[:, 0].mean().item())
    best_fg_prob = float(probabilities[:, 1:].max(dim=1).values.mean().item())

    return {
        "target_fg_voxels": target_fg,
        "predicted_fg_voxels": pred_fg,
        "target_fg_fraction": target_fg / total,
        "predicted_fg_fraction": pred_fg / total,
        "predicted_to_target_fg_ratio": (pred_fg / target_fg) if target_fg else None,
        "mean_background_probability": bg_prob,
        "mean_best_foreground_probability": best_fg_prob,
    }


def run_epoch(
    model,
    rows: pd.DataFrame,
    part11,
    part9,
    loss_fn,
    optimizer,
    scaler,
    device,
    training: bool,
    epoch: int,
    condition: str,
) -> Dict[str, Any]:
    model.train(training)

    losses: List[float] = []
    dices: List[float] = []
    fg_counts: List[int] = []
    target_fg_counts: List[int] = []
    bg_probs: List[float] = []
    fg_probs: List[float] = []
    fallback_count = 0

    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        image_d, mask_d, info = load_case(part11, part9, row)
        if info.get("part11_loader") == "local_robust_fallback":
            fallback_count += 1

        # [D,H,W] -> [1,1,D,H,W]
        image = image_d.unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
        # [D,H,W] -> [1,D,H,W]
        target = mask_d.unsqueeze(0).to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=AMP_ENABLED and device.type == "cuda"):
            logits = model(image)
            loss = loss_fn(logits, target.unsqueeze(1))

        if training:
            scaler.scale(loss).backward()
            if GRAD_CLIP is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            mean_dice, _ = safe_dice(part11, logits, target)
            stats = foreground_statistics(logits, target)

        losses.append(float(loss.detach().cpu()))
        dices.append(mean_dice)
        fg_counts.append(stats["predicted_fg_voxels"])
        target_fg_counts.append(stats["target_fg_voxels"])
        bg_probs.append(stats["mean_background_probability"])
        fg_probs.append(stats["mean_best_foreground_probability"])

        if i == 1 or i % 25 == 0 or i == len(rows):
            mode = "TRAIN" if training else "VAL"
            print(
                f"  {mode} {i:03d}/{len(rows)} "
                f"Loss={losses[-1]:.5f} Dice={dices[-1]:.6f} "
                f"PredFG={fg_counts[-1]} TargetFG={target_fg_counts[-1]}"
            )

        del image, target, logits, loss, image_d, mask_d
        if device.type == "cuda" and i % 10 == 0:
            torch.cuda.empty_cache()

    return {
        "condition": condition,
        "epoch": epoch,
        "training": training,
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "mean_predicted_fg_voxels": float(np.mean(fg_counts)),
        "mean_target_fg_voxels": float(np.mean(target_fg_counts)),
        "mean_predicted_fg_fraction": float(np.mean(np.array(fg_counts) / np.prod(PATCH_SIZE))),
        "mean_target_fg_fraction": float(np.mean(np.array(target_fg_counts) / np.prod(PATCH_SIZE))),
        "mean_background_probability": float(np.mean(bg_probs)),
        "mean_best_foreground_probability": float(np.mean(fg_probs)),
        "empty_prediction_cases": int(sum(x == 0 for x in fg_counts)),
        "cases": len(rows),
        "part11_local_fallback_cases": fallback_count,
    }


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, history: List[Dict[str, Any]], condition: str):
    torch.save(
        {
            "part": 40,
            "condition": condition,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "seed": SEED,
            "patch_size": PATCH_SIZE,
            "feature_size": FEATURE_SIZE,
            "num_classes": NUM_CLASSES,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "train_cases": TRAIN_CASES,
            "validation_cases": VAL_CASES,
        },
        path,
    )


def run_condition(condition: str, train_rows: pd.DataFrame, val_rows: pd.DataFrame, part11, part9, device: torch.device) -> Dict[str, Any]:
    condition_dir = BASELINE_DIR if condition == "baseline" else FOREGROUND_DIR
    checkpoint_dir = condition_dir / "checkpoints"
    report_dir = condition_dir / "reports"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    banner(f"PART 40 CONDITION: {condition.upper()}")

    # Fresh model from the SAME Part 11 initialization for each condition.
    seed_everything(SEED)
    model = make_model(part11, device)
    init_meta = load_initialization(model, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler(enabled=AMP_ENABLED and device.type == "cuda")
    loss_fn = make_loss(condition, device)

    status("Condition", condition)
    status("Initialization checkpoint SHA256", sha256_file(PART15_INIT))
    status("Optimizer", "AdamW")
    status("Learning rate", LR)
    status("Weight decay", WEIGHT_DECAY)
    status("AMP", AMP_ENABLED and device.type == "cuda")
    status("Loss", "DiceCELoss")
    status(
        "Class weights",
        "original" if condition == "baseline" else FOREGROUND_AWARE_CLASS_WEIGHTS,
    )

    history: List[Dict[str, Any]] = []

    # Pre-training validation establishes the identical starting point.
    banner(f"{condition.upper()} — INITIALIZATION VALIDATION")
    with torch.no_grad():
        init_val = run_epoch(
            model, val_rows, part11, part9, loss_fn, optimizer, scaler,
            device, False, 0, condition
        )
    init_val["stage"] = "initialization"
    history.append(init_val)
    print(
        f"Initialization: loss={init_val['loss']:.6f}, "
        f"Dice={init_val['dice']:.6f}, "
        f"PredFG={init_val['mean_predicted_fg_voxels']:.1f}, "
        f"empty={init_val['empty_prediction_cases']}/{VAL_CASES}"
    )

    best_dice = init_val["dice"]
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        banner(f"{condition.upper()} — EPOCH {epoch}/{EPOCHS}")
        start = time.time()

        train_result = run_epoch(
            model, train_rows, part11, part9, loss_fn, optimizer, scaler,
            device, True, epoch, condition
        )
        train_result["stage"] = "train"
        history.append(train_result)

        with torch.no_grad():
            val_result = run_epoch(
                model, val_rows, part11, part9, loss_fn, optimizer, scaler,
                device, False, epoch, condition
            )
        val_result["stage"] = "validation"
        history.append(val_result)

        elapsed = time.time() - start
        print()
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_result['loss']:.6f}, "
            f"train_Dice={train_result['dice']:.6f}, "
            f"val_loss={val_result['loss']:.6f}, "
            f"val_Dice={val_result['dice']:.6f}, "
            f"val_PredFG={val_result['mean_predicted_fg_voxels']:.1f}, "
            f"time={elapsed/60:.2f} min"
        )

        save_checkpoint(
            checkpoint_dir / f"epoch_{epoch:02d}.pth",
            model, optimizer, scaler, epoch, history, condition
        )

        if val_result["dice"] > best_dice:
            best_dice = val_result["dice"]
            best_epoch = epoch
            save_checkpoint(
                checkpoint_dir / "best_model.pth",
                model, optimizer, scaler, epoch, history, condition
            )
            print("✓ New best validation Dice checkpoint saved.")

        # Explicit collapse signal.
        if val_result["empty_prediction_cases"] == VAL_CASES:
            print("⚠ VALIDATION: complete foreground prediction collapse.")
        elif val_result["mean_predicted_fg_voxels"] < 1.0:
            print("⚠ VALIDATION: near-complete foreground prediction collapse.")
        else:
            print("✓ VALIDATION: foreground predictions remain non-empty.")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    summary = {
        "condition": condition,
        "train_cases": TRAIN_CASES,
        "validation_cases": VAL_CASES,
        "epochs": EPOCHS,
        "patch_size": list(PATCH_SIZE),
        "feature_size": FEATURE_SIZE,
        "num_classes": NUM_CLASSES,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss": "DiceCELoss",
        "class_weights": (
            None if condition == "baseline" else FOREGROUND_AWARE_CLASS_WEIGHTS
        ),
        "initialization_checkpoint": str(PART15_INIT),
        "initialization_sha256": sha256_file(PART15_INIT),
        "initialization_checkpoint_epoch": init_meta.get("epoch"),
        "best_validation_dice": best_dice,
        "best_epoch": best_epoch,
        "history": history,
    }

    with (report_dir / "part40_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(history).to_csv(report_dir / "part40_history.csv", index=False)

    return summary


def write_comparison(baseline: Dict[str, Any], aware: Dict[str, Any]) -> None:
    banner("PART 40 CONTROLLED COMPARISON")

    def final_validation(summary):
        vals = [x for x in summary["history"] if x.get("stage") == "validation"]
        return vals[-1]

    b = final_validation(baseline)
    a = final_validation(aware)

    comparison = {
        "baseline_final_val_loss": b["loss"],
        "foreground_aware_final_val_loss": a["loss"],
        "baseline_final_val_dice": b["dice"],
        "foreground_aware_final_val_dice": a["dice"],
        "baseline_final_predicted_fg_voxels": b["mean_predicted_fg_voxels"],
        "foreground_aware_final_predicted_fg_voxels": a["mean_predicted_fg_voxels"],
        "baseline_final_empty_cases": b["empty_prediction_cases"],
        "foreground_aware_final_empty_cases": a["empty_prediction_cases"],
        "dice_delta_aware_minus_baseline": a["dice"] - b["dice"],
        "predicted_fg_delta_aware_minus_baseline": a["mean_predicted_fg_voxels"] - b["mean_predicted_fg_voxels"],
        "same_initialization_sha256": baseline["initialization_sha256"] == aware["initialization_sha256"],
        "interpretation": (
            "FOREGROUND_AWARE_CONDITION_CHANGED_TRAINING_BEHAVIOR"
            if (
                a["mean_predicted_fg_voxels"] != b["mean_predicted_fg_voxels"]
                or a["empty_prediction_cases"] != b["empty_prediction_cases"]
            )
            else "NO_MEASURABLE_FOREGROUND_BEHAVIOR_CHANGE_IN_PILOT"
        ),
        "caution": "Pilot evidence only; not proof of clinical segmentation performance or sole causality.",
    }

    print(json.dumps(comparison, indent=2))

    report_dir = OUTPUT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "part40_controlled_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    pd.DataFrame([comparison]).to_csv(
        report_dir / "part40_controlled_comparison.csv", index=False
    )

    print()
    status("Same initialization checkpoint", comparison["same_initialization_sha256"])
    status("Baseline final validation Dice", f"{b['dice']:.6f}")
    status("Foreground-aware final validation Dice", f"{a['dice']:.6f}")
    status("Baseline final predicted FG", f"{b['mean_predicted_fg_voxels']:.1f}")
    status("Foreground-aware final predicted FG", f"{a['mean_predicted_fg_voxels']:.1f}")
    status("Baseline empty validation cases", f"{b['empty_prediction_cases']}/{VAL_CASES}")
    status("Foreground-aware empty validation cases", f"{a['empty_prediction_cases']}/{VAL_CASES}")
    status("Pilot interpretation", comparison["interpretation"])


def main() -> None:
    seed_everything(SEED)
    require_paths()

    banner("PART 40 — CONTROLLED FOREGROUND-AWARE COUNTERFACTUAL")
    status("PyTorch", torch.__version__)
    status("CUDA available", torch.cuda.is_available())
    if torch.cuda.is_available():
        status("GPU", torch.cuda.get_device_name(0))
        status("GPU memory", f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    status("Patch size", PATCH_SIZE)
    status("Feature size", FEATURE_SIZE)
    status("Classes", NUM_CLASSES)
    status("Train subset", TRAIN_CASES)
    status("Validation subset", VAL_CASES)
    status("Epochs per condition", EPOCHS)
    status("Part 15 overwritten", "NO")
    status("SPIDER used", "NO")
    status("Test set used", "NO")

    part11 = import_part11()
    part9 = load_part9(part11)
    train_rows, val_rows = load_cohorts()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    baseline = run_condition("baseline", train_rows, val_rows, part11, part9, device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    aware = run_condition("foreground_aware", train_rows, val_rows, part11, part9, device)

    write_comparison(baseline, aware)

    banner("PART 40 COMPLETE")
    status("Output directory", OUTPUT_DIR)
    status("Baseline directory", BASELINE_DIR)
    status("Foreground-aware directory", FOREGROUND_DIR)
    print("\nPart 40 is a controlled pilot. Do not interpret it as final clinical performance.")


if __name__ == "__main__":
    main()
