"""
Part 84 — Final Reproducible Segmentation Training
===================================================

Purpose
-------
Move from forensic ablation work to the FINAL segmentation training pipeline.

This script:
- DOES train the segmentation model.
- Uses the verified Part15 initialization.
- Uses the R2_FULL pseudo-mask geometry.
- Uses centered 32x64x64 crops.
- Uses the same RSNA-only train/validation cohorts used in the controlled work.
- Uses class-balanced Dice + weighted Cross Entropy.
- Uses the validated Part11 SwinUNETR model factory.
- Evaluates validation performance with an explicit, reproducible metric:
    * micro/global foreground Dice
    * mean-case foreground Dice
    * classwise Dice / precision / recall
- Saves every epoch checkpoint.
- Saves the best checkpoint according to GLOBAL VOXEL-POOLED FOREGROUND DICE.
- Does NOT use the historical/non-reproducible 0.044026 value for model selection.

Recommended execution:
    python "segmentation_rsna_part84_final_reproducible_segmentation_training.py"

Project root:
    C:\\Saad\\Msc Major Project Swin Unetr Framework\\MSc_SwinUNETR_Project

IMPORTANT
---------
This is the first final-training part after the Part83 forensic closure.
No old historical metric is used to select the model.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "outputs" / "segmentation" / "rsna_part84_final_reproducible_training"
CKPT_DIR = OUT / "checkpoints"
REPORT_DIR = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PART11_PATH = SRC / "segmentation_rsna_part11_controlled_pilot_training.py"
PART9_PATH = SRC / "segmentation_rsna_part9_3d_dataset_loader.py"

INIT_CKPT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "checkpoints"
    / "part15_initialization_from_part11.pth"
)

TRAIN_CSV = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_train_cohort.csv"
)

VAL_CSV = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_validation_cohort.csv"
)

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 6
FEATURE_SIZE = 12

# Final controlled cohort sizes. These preserve the verified Part15 split
# while keeping the final experiment feasible on a 4 GB RTX 2050.
TRAIN_CASES = 100
VAL_CASES = 50

EPOCHS = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5

# Centered crop used by the R2_FULL controlled pipeline.
CROP_D = 32
CROP_H = 64
CROP_W = 64

# R2_FULL mask geometry: 2-voxel radius dilation from the point-derived
# pseudo-mask. The actual pseudo-mask is already present in the cohort.
# Therefore the crop itself is the final preprocessing operation here.

# Loss weights previously established by the project's empirical
# class-weight calculation. Background is deliberately down-weighted.
CLASS_WEIGHTS = torch.tensor(
    [
        0.015069,
        0.722595,
        0.935995,
        1.017003,
        1.178304,
        1.146102,
    ],
    dtype=torch.float32,
)

# Training safety.
MAX_CONSECUTIVE_BAD_EPOCHS = 3
MIN_IMPROVEMENT = 1e-5


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic mode improves reproducibility. On CUDA this may be
    # slower, but final experiments should prioritize reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_div(a: float, b: float, eps: float = 1e-8) -> float:
    return float(a / (b + eps))


def safe_float(x: Any) -> Optional[float]:
    try:
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def binary_dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)

    p = int(pred.sum())
    t = int(target.sum())
    tp = int(np.logical_and(pred, target).sum())

    if p == 0 and t == 0:
        return 1.0
    return safe_div(2 * tp, p + t)


def class_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    class_id: int,
) -> Dict[str, float]:
    p = pred == class_id
    t = target == class_id

    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())

    pred_n = int(p.sum())
    target_n = int(t.sum())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_voxels": pred_n,
        "target_voxels": target_n,
        "dice": safe_div(2 * tp, pred_n + target_n),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
    }


# ---------------------------------------------------------------------
# Center crop
# ---------------------------------------------------------------------

def centered_crop_3d(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Center crop image and mask using the exact same origin.

    image: [D,H,W]
    mask:  [D,H,W]
    """
    cd, ch, cw = crop_size

    d, h, w = image.shape[-3:]

    if d < cd or h < ch or w < cw:
        raise ValueError(
            f"Input volume {tuple(image.shape)} is smaller than "
            f"crop {crop_size}"
        )

    sd = (d - cd) // 2
    sh = (h - ch) // 2
    sw = (w - cw) // 2

    return (
        image[sd:sd + cd, sh:sh + ch, sw:sw + cw],
        mask[sd:sd + cd, sh:sh + ch, sw:sw + cw],
    )


# ---------------------------------------------------------------------
# Model and checkpoint
# ---------------------------------------------------------------------

def create_model(part11: Any) -> nn.Module:
    fn = getattr(part11, "create_model", None)
    if not callable(fn):
        raise RuntimeError("Part11 create_model() not found.")

    try:
        model = fn(DEVICE)
    except TypeError:
        model = fn()

    return model.to(DEVICE)


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "network_state_dict",
        ]:
            obj = ckpt.get(key)
            if isinstance(obj, dict) and obj:
                if all(torch.is_tensor(v) for v in obj.values()):
                    return obj

        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt

    raise RuntimeError("Could not find model state_dict in checkpoint.")


def load_initialization(model: nn.Module) -> Dict[str, Any]:
    if not INIT_CKPT.exists():
        raise FileNotFoundError(f"Part15 initialization missing: {INIT_CKPT}")

    try:
        ckpt = torch.load(
            INIT_CKPT,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(INIT_CKPT, map_location="cpu")

    state = extract_state_dict(ckpt)

    result = model.load_state_dict(state, strict=True)

    return {
        "checkpoint": str(INIT_CKPT),
        "sha256": sha256_file(INIT_CKPT),
        "metadata": {
            k: ckpt.get(k)
            for k in [
                "epoch",
                "condition",
                "val_foreground_dice",
                "seed",
                "device",
            ]
            if isinstance(ckpt, dict) and k in ckpt
        },
        "load_result": str(result),
    }


# ---------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------

def load_case(
    row: pd.Series,
    part9: Any,
    part11: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use the validated Part11 compatibility loader.

    The historical Part11 API has been verified to return:
        image, mask, ...
    """
    fn = getattr(part11, "load_tensor_case", None)
    if not callable(fn):
        raise RuntimeError("Part11 load_tensor_case() not found.")

    result = fn(row, part9)

    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(
            "Unexpected Part11 load_tensor_case() return value."
        )

    image, mask = result[0], result[1]

    image = torch.as_tensor(image).float()
    mask = torch.as_tensor(mask).long()

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    elif image.ndim == 5 and image.shape[0] == 1 and image.shape[1] == 1:
        image = image[0, 0]

    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 5 and mask.shape[0] == 1 and mask.shape[1] == 1:
        mask = mask[0, 0]

    if image.ndim != 3 or mask.ndim != 3:
        raise RuntimeError(
            f"Expected 3D image/mask; got {tuple(image.shape)}, "
            f"{tuple(mask.shape)}"
        )

    # Final preprocessing: same centered crop for image and pseudo-mask.
    image, mask = centered_crop_3d(
        image,
        mask,
        (CROP_D, CROP_H, CROP_W),
    )

    # Basic numerical safety.
    image = torch.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return image, mask


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

class FinalClassBalancedDiceCE(nn.Module):
    """
    Explicit implementation of:
        MONAI DiceLoss + weighted PyTorch CrossEntropyLoss

    This avoids version-specific DiceCELoss constructor differences.
    """

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()

        self.register_buffer("class_weights", class_weights)

        self.dice = DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # logits: [B,C,D,H,W]
        # target: [B,1,D,H,W]
        dice_loss = self.dice(logits, target)

        target_ce = target[:, 0].long()

        ce_loss = F.cross_entropy(
            logits,
            target_ce,
            weight=self.class_weights.to(logits.device),
        )

        return dice_loss + ce_loss


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: pd.DataFrame,
    part9: Any,
    part11: Any,
) -> Dict[str, Any]:
    model.train()

    losses: List[float] = []
    target_fg: List[int] = []
    successful = 0

    start = time.time()

    for case_idx, (_, row) in enumerate(rows.iterrows(), start=1):
        image, target = load_case(row, part9, part11)

        x = image.unsqueeze(0).unsqueeze(0).to(
            DEVICE,
            non_blocking=True,
        )
        y = target.unsqueeze(0).unsqueeze(0).to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        if logits.shape[-3:] != y.shape[-3:]:
            raise RuntimeError(
                f"Output/target shape mismatch: "
                f"{tuple(logits.shape)} vs {tuple(y.shape)}"
            )

        loss = loss_fn(logits, y)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at case {case_idx}: "
                f"{float(loss.detach().cpu())}"
            )

        loss.backward()

        # Prevent a single case from destabilizing the final run.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        target_fg.append(int((target > 0).sum()))
        successful += 1

        if case_idx % 10 == 0 or case_idx == len(rows):
            print(
                f"    Train case {case_idx:03d}/{len(rows)} | "
                f"loss={losses[-1]:.5f}"
            )

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mean_target_foreground_voxels": (
            float(np.mean(target_fg)) if target_fg else float("nan")
        ),
        "successful_cases": successful,
        "seconds": time.time() - start,
    }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    rows: pd.DataFrame,
    part9: Any,
    part11: Any,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    model.eval()

    losses: List[float] = []
    class_records: List[Dict[str, Any]] = []
    case_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for case_no, (_, row) in enumerate(rows.iterrows(), start=1):
            image, target = load_case(row, part9, part11)

            x = image.unsqueeze(0).unsqueeze(0).to(
                DEVICE,
                non_blocking=True,
            )
            y = target.unsqueeze(0).unsqueeze(0).to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(x)
            loss = loss_fn(logits, y)

            pred = (
                torch.argmax(logits, dim=1)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

            tgt = target.cpu().numpy().astype(np.int64)

            losses.append(float(loss.detach().cpu()))

            case_fg_target = int((tgt > 0).sum())
            case_fg_pred = int((pred > 0).sum())

            case_rec = {
                "case_no": case_no,
                "target_foreground_voxels": case_fg_target,
                "predicted_foreground_voxels": case_fg_pred,
                "empty_foreground_prediction": int(case_fg_pred == 0),
                "foreground_dice": binary_dice(pred > 0, tgt > 0),
            }

            for c in range(1, NUM_CLASSES):
                m = class_metrics(pred, tgt, c)

                rec = {
                    "case_no": case_no,
                    "class_id": c,
                    **m,
                }

                class_records.append(rec)

                for key, value in m.items():
                    case_rec[f"class_{c}_{key}"] = value

            case_records.append(case_rec)

            print(
                f"    Val case {case_no:03d}/{len(rows)} | "
                f"targetFG={case_fg_target} "
                f"predFG={case_fg_pred} "
                f"FGDice={case_rec['foreground_dice']:.6f}"
            )

    cdf = pd.DataFrame(class_records)
    kdf = pd.DataFrame(case_records)

    # Global micro foreground Dice.
    total_tp = 0
    total_pred = 0
    total_target = 0

    class_summary = []

    for c in range(1, NUM_CLASSES):
        g = cdf[cdf.class_id == c]

        tp = int(g.tp.sum())
        fp = int(g.fp.sum())
        fn = int(g.fn.sum())
        pred_n = int(g.predicted_voxels.sum())
        target_n = int(g.target_voxels.sum())

        total_tp += tp
        total_pred += pred_n
        total_target += target_n

        class_summary.append(
            {
                "class_id": c,
                "target_voxels": target_n,
                "predicted_voxels": pred_n,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "dice": safe_div(2 * tp, pred_n + target_n),
                "precision": safe_div(tp, tp + fp),
                "recall": safe_div(tp, tp + fn),
                "iou": safe_div(tp, tp + fp + fn),
                "empty_cases": int(g.empty_prediction.sum())
                if "empty_prediction" in g.columns
                else int((g.predicted_voxels == 0).sum()),
            }
        )

    micro_fg_dice = safe_div(
        2 * total_tp,
        total_pred + total_target,
    )

    mean_case_fg_dice = (
        float(kdf.foreground_dice.mean())
        if len(kdf)
        else float("nan")
    )

    class_dices = [float(x["dice"]) for x in class_summary]
    macro_fg_dice = float(np.mean(class_dices)) if class_dices else 0.0

    summary = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "foreground_target_voxels": int(
            kdf.target_foreground_voxels.sum()
        ),
        "foreground_predicted_voxels": int(
            kdf.predicted_foreground_voxels.sum()
        ),
        "foreground_empty_cases": int(
            kdf.empty_foreground_prediction.sum()
        ),
        "cases": len(kdf),
        "foreground_dice": micro_fg_dice,
        "mean_case_foreground_dice": mean_case_fg_dice,
        "macro_foreground_dice": macro_fg_dice,
    }

    return summary, pd.DataFrame(class_summary), kdf


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    report_lines: List[str] = []

    def log(text: str = "") -> None:
        print(text)
        report_lines.append(text)

    log("=" * 78)
    log("PART 84 — FINAL REPRODUCIBLE SEGMENTATION TRAINING")
    log("=" * 78)
    log("Training is ENABLED.")
    log(f"Device: {DEVICE}")
    log(f"Seed: {SEED}")
    log(f"Train cases: {TRAIN_CASES}")
    log(f"Validation cases: {VAL_CASES}")
    log(f"Epochs: {EPOCHS}")
    log(f"Learning rate: {LEARNING_RATE}")
    log(f"Weight decay: {WEIGHT_DECAY}")
    log(f"Crop: {(CROP_D, CROP_H, CROP_W)}")
    log("Historical Part71 value 0.044026 is NOT used for selection.")
    log("")

    if not torch.cuda.is_available():
        log("WARNING: CUDA is unavailable. Training will run on CPU and may be slow.")

    # -------------------------------------------------------------
    # Verify source files
    # -------------------------------------------------------------
    for path in [PART11_PATH, PART9_PATH, INIT_CKPT, TRAIN_CSV, VAL_CSV]:
        if not path.exists():
            raise FileNotFoundError(f"Required project file missing: {path}")

    log("Part15 initialization SHA256:")
    log(f"  {sha256_file(INIT_CKPT)}")

    expected_part15_hash = (
        "0900c0e6490fdaddf455763b465d4a607acfb310349c9f9ec82a61117016aec6"
    )

    if sha256_file(INIT_CKPT) != expected_part15_hash:
        raise RuntimeError(
            "Part15 initialization SHA256 does not match the verified "
            "historical initialization."
        )

    # -------------------------------------------------------------
    # Import validated modules
    # -------------------------------------------------------------
    part9 = import_module(PART9_PATH, "part84_part9")
    part11 = import_module(PART11_PATH, "part84_part11")

    log("Part9 import: OK")
    log("Part11 import: OK")

    # -------------------------------------------------------------
    # Load cohorts
    # -------------------------------------------------------------
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_rows = train_df.iloc[:TRAIN_CASES].copy()
    val_rows = val_df.iloc[:VAL_CASES].copy()

    log(f"Loaded train cohort: {len(train_df)} rows")
    log(f"Using first {len(train_rows)} training rows")
    log(f"Loaded validation cohort: {len(val_df)} rows")
    log(f"Using first {len(val_rows)} validation rows")

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    model = create_model(part11)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    log(f"Model total parameters: {total_params:,}")
    log(f"Model trainable parameters: {trainable_params:,}")

    init_info = load_initialization(model)

    log("Part15 initialization load: OK")
    log(f"  {init_info['load_result']}")

    # -------------------------------------------------------------
    # Loss and optimizer
    # -------------------------------------------------------------
    loss_fn = FinalClassBalancedDiceCE(CLASS_WEIGHTS).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    log("")
    log("Class weights:")
    for i, w in enumerate(CLASS_WEIGHTS.tolist()):
        log(f"  class {i}: {w:.6f}")

    # -------------------------------------------------------------
    # Initial validation
    # -------------------------------------------------------------
    log("")
    log("=" * 78)
    log("INITIALIZATION EVALUATION")
    log("=" * 78)

    init_summary, init_classes, init_cases = evaluate(
        model,
        loss_fn,
        val_rows,
        part9,
        part11,
    )

    log("")
    log(
        f"INIT | loss={init_summary['loss']:.6f} | "
        f"globalFGDice={init_summary['foreground_dice']:.6f} | "
        f"meanCaseFGDice={init_summary['mean_case_foreground_dice']:.6f} | "
        f"PredFG={init_summary['foreground_predicted_voxels']}"
    )

    best_metric = float(init_summary["foreground_dice"])
    best_epoch = 0

    best_ckpt_path = CKPT_DIR / "part84_best_model.pth"

    # Save initialization as the current best candidate.
    torch.save(
        {
            "part": 84,
            "epoch": 0,
            "condition": "final_class_balanced_dice_ce",
            "seed": SEED,
            "device": str(DEVICE),
            "train_cases": len(train_rows),
            "validation_cases": len(val_rows),
            "feature_size": FEATURE_SIZE,
            "num_classes": NUM_CLASSES,
            "crop_size": [CROP_D, CROP_H, CROP_W],
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "class_weights": CLASS_WEIGHTS.tolist(),
            "source_initialization": str(INIT_CKPT),
            "source_initialization_sha256": sha256_file(INIT_CKPT),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_summary": init_summary,
            "val_class_summary": init_classes.to_dict(orient="records"),
        },
        best_ckpt_path,
    )

    history: List[Dict[str, Any]] = [
        {
            "epoch": 0,
            "train_loss": None,
            "val_loss": init_summary["loss"],
            "global_fg_dice": init_summary["foreground_dice"],
            "mean_case_fg_dice": init_summary[
                "mean_case_foreground_dice"
            ],
            "macro_fg_dice": init_summary["macro_foreground_dice"],
            "pred_fg_voxels": init_summary[
                "foreground_predicted_voxels"
            ],
            "empty_cases": init_summary["foreground_empty_cases"],
        }
    ]

    bad_epochs = 0

    # -------------------------------------------------------------
    # Final training
    # -------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        log("")
        log("=" * 78)
        log(f"EPOCH {epoch}/{EPOCHS}")
        log("=" * 78)

        epoch_start = time.time()

        train_summary = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            train_rows,
            part9,
            part11,
        )

        log(
            f"TRAIN | loss={train_summary['loss']:.6f} | "
            f"meanTargetFG={train_summary['mean_target_foreground_voxels']:.2f} | "
            f"time={train_summary['seconds']:.1f}s"
        )

        val_summary, val_classes, val_cases = evaluate(
            model,
            loss_fn,
            val_rows,
            part9,
            part11,
        )

        metric = float(val_summary["foreground_dice"])

        log("")
        log(
            f"VAL   | loss={val_summary['loss']:.6f} | "
            f"globalFGDice={metric:.6f} | "
            f"meanCaseFGDice={val_summary['mean_case_foreground_dice']:.6f} | "
            f"macroFGDice={val_summary['macro_foreground_dice']:.6f} | "
            f"PredFG={val_summary['foreground_predicted_voxels']} | "
            f"Empty={val_summary['foreground_empty_cases']}/{VAL_CASES}"
        )

        log("Classwise validation Dice:")
        for _, rec in val_classes.iterrows():
            log(
                f"  C{int(rec['class_id'])}: "
                f"Dice={float(rec['dice']):.6f} "
                f"P={float(rec['precision']):.6f} "
                f"R={float(rec['recall']):.6f} "
                f"Pred={int(rec['predicted_voxels'])} "
                f"Target={int(rec['target_voxels'])}"
            )

        epoch_path = CKPT_DIR / f"part84_epoch{epoch}.pth"

        checkpoint_payload = {
            "part": 84,
            "epoch": epoch,
            "condition": "final_class_balanced_dice_ce",
            "seed": SEED,
            "device": str(DEVICE),
            "train_cases": len(train_rows),
            "validation_cases": len(val_rows),
            "feature_size": FEATURE_SIZE,
            "num_classes": NUM_CLASSES,
            "crop_size": [CROP_D, CROP_H, CROP_W],
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "class_weights": CLASS_WEIGHTS.tolist(),
            "source_initialization": str(INIT_CKPT),
            "source_initialization_sha256": sha256_file(INIT_CKPT),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_summary": train_summary,
            "val_summary": val_summary,
            "val_class_summary": val_classes.to_dict(
                orient="records"
            ),
        }

        torch.save(checkpoint_payload, epoch_path)

        improved = metric > best_metric + MIN_IMPROVEMENT

        if improved:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0

            torch.save(checkpoint_payload, best_ckpt_path)

            log(
                f"*** NEW BEST CHECKPOINT *** "
                f"epoch={epoch} globalFGDice={metric:.6f}"
            )
        else:
            bad_epochs += 1
            log(
                f"No improvement over best={best_metric:.6f}; "
                f"bad_epochs={bad_epochs}"
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_summary["loss"],
                "val_loss": val_summary["loss"],
                "global_fg_dice": metric,
                "mean_case_fg_dice": val_summary[
                    "mean_case_foreground_dice"
                ],
                "macro_fg_dice": val_summary[
                    "macro_foreground_dice"
                ],
                "pred_fg_voxels": val_summary[
                    "foreground_predicted_voxels"
                ],
                "empty_cases": val_summary[
                    "foreground_empty_cases"
                ],
                "seconds": time.time() - epoch_start,
            }
        )

        # Safety stop if validation collapses to zero foreground and remains
        # there for several epochs after having no improvement.
        if (
            metric <= 1e-8
            and epoch >= 2
            and bad_epochs >= MAX_CONSECUTIVE_BAD_EPOCHS
        ):
            log(
                "EARLY STOP: validation foreground prediction has collapsed "
                "for multiple consecutive epochs."
            )
            break

    # -------------------------------------------------------------
    # Save history
    # -------------------------------------------------------------
    history_df = pd.DataFrame(history)
    history_csv = OUT / "part84_training_history.csv"
    history_df.to_csv(history_csv, index=False)

    # -------------------------------------------------------------
    # Final best checkpoint verification
    # -------------------------------------------------------------
    log("")
    log("=" * 78)
    log("BEST CHECKPOINT RE-EVALUATION")
    log("=" * 78)

    try:
        best_ckpt = torch.load(
            best_ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu")

    best_state = extract_state_dict(best_ckpt)
    model.load_state_dict(best_state, strict=True)
    model.to(DEVICE)

    final_summary, final_classes, final_cases = evaluate(
        model,
        loss_fn,
        val_rows,
        part9,
        part11,
    )

    log("")
    log(
        f"BEST VERIFY | epoch={best_ckpt.get('epoch')} | "
        f"globalFGDice={final_summary['foreground_dice']:.6f} | "
        f"meanCaseFGDice={final_summary['mean_case_foreground_dice']:.6f} | "
        f"macroFGDice={final_summary['macro_foreground_dice']:.6f} | "
        f"PredFG={final_summary['foreground_predicted_voxels']}"
    )

    # -------------------------------------------------------------
    # Save final report
    # -------------------------------------------------------------
    summary = {
        "part": 84,
        "title": "Final Reproducible Segmentation Training",
        "training_performed": True,
        "seed": SEED,
        "device": str(DEVICE),
        "train_cases": len(train_rows),
        "validation_cases": len(val_rows),
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history) - 1,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "crop_size": [CROP_D, CROP_H, CROP_W],
        "num_classes": NUM_CLASSES,
        "feature_size": FEATURE_SIZE,
        "class_weights": CLASS_WEIGHTS.tolist(),
        "initialization": init_info,
        "best_epoch": int(best_ckpt.get("epoch", best_epoch)),
        "best_global_foreground_dice_training": best_metric,
        "best_checkpoint": str(best_ckpt_path),
        "best_checkpoint_sha256": sha256_file(best_ckpt_path),
        "final_verified_summary": final_summary,
        "final_verified_classwise": final_classes.to_dict(
            orient="records"
        ),
        "historical_part71_metric_0_044026_used": False,
        "status": (
            "FINAL_SEGMENTATION_TRAINING_COMPLETE"
            if final_summary["foreground_dice"] > 0
            else "FINAL_TRAINING_COMPLETED_BUT_FOREGROUND_DICE_ZERO"
        ),
    }

    summary_path = REPORT_DIR / "part84_final_segmentation_summary.json"
    report_path = REPORT_DIR / "part84_final_segmentation_report.txt"

    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    report_lines.append("")
    report_lines.append("=" * 78)
    report_lines.append("FINAL PART84 STATUS")
    report_lines.append("=" * 78)
    report_lines.append(summary["status"])
    report_lines.append(f"Best epoch: {summary['best_epoch']}")
    report_lines.append(
        f"Verified global foreground Dice: "
        f"{final_summary['foreground_dice']:.6f}"
    )
    report_lines.append(
        f"Verified mean-case foreground Dice: "
        f"{final_summary['mean_case_foreground_dice']:.6f}"
    )
    report_lines.append(f"Best checkpoint: {best_ckpt_path}")
    report_lines.append(f"Best checkpoint SHA256: {sha256_file(best_ckpt_path)}")

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("PART 84 COMPLETE")
    print("=" * 78)
    print(f"Status: {summary['status']}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(
        f"Verified global foreground Dice: "
        f"{final_summary['foreground_dice']:.6f}"
    )
    print(
        f"Verified mean-case foreground Dice: "
        f"{final_summary['mean_case_foreground_dice']:.6f}"
    )
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"History: {history_csv}")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
