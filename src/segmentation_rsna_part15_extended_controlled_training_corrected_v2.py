"""
PHASE 4 - PART 15
RSNA-ONLY EXTENDED CONTROLLED SWIN-UNETR TRAINING

Purpose:
Continue from the reproducible Part 11 best checkpoint using the validated
Part 11 preprocessing/loader/metric pipeline, while expanding the controlled
training cohort. This is NOT final clinical evaluation.

Design:
- RSNA only
- SPIDER never used
- No test set during training
- Batch size 1
- Patch size (64, 96, 96)
- Feature size 12
- 6 classes
- AMP enabled on CUDA
- Train cohort: 500 series
- Validation cohort: 100 series
- 10 additional epochs
- Best checkpoint selected by mean foreground Dice
- Validation cohort is fixed and deterministic
- Existing Part 11 best checkpoint is used as initialization
"""

from __future__ import annotations

import gc
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from monai.losses import DiceCELoss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RSNA_DIR = PROJECT_ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification"

TRAIN_MANIFEST = (
    PROJECT_ROOT / "outputs" / "segmentation" /
    "rsna_part8_dataset_construction" / "manifests" /
    "rsna_part8_train_manifest.csv"
)
VAL_MANIFEST = (
    PROJECT_ROOT / "outputs" / "segmentation" /
    "rsna_part8_dataset_construction" / "manifests" /
    "rsna_part8_validation_manifest.csv"
)

PART11_SOURCE = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
PART11_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "segmentation" /
    "rsna_part11_controlled_pilot_training" / "checkpoints" /
    "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "segmentation" /
    "rsna_part15_extended_controlled_training"
)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_DIR = OUTPUT_DIR / "reports"

SEED = 42
TRAIN_CASES = 500
VAL_CASES = 100
EPOCHS = 10
BATCH_SIZE = 1
PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12
NUM_CLASSES = 6
LR = 1e-4
WEIGHT_DECAY = 1e-5
AMP_ENABLED = True
NUM_WORKERS = 0
GRAD_CLIP = 1.0

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


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


def banner(text: str) -> None:
    print("=" * 78)
    print(text)
    print("=" * 78)


def require_paths() -> None:
    paths = {
        "RSNA root": RSNA_DIR,
        "train_images": RSNA_DIR / "train_images",
        "train manifest": TRAIN_MANIFEST,
        "validation manifest": VAL_MANIFEST,
        "Part 11 source": PART11_SOURCE,
        "Part 11 checkpoint": PART11_CHECKPOINT,
    }
    missing = [f"{k}: {v}" for k, v in paths.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required Part 15 input(s):\n" + "\n".join(missing)
        )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def import_part11():
    spec = importlib.util.spec_from_file_location(
        "part11_for_part15", PART11_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Part 11: {PART11_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["part11_for_part15"] = module
    spec.loader.exec_module(module)

    required = [
        "select_pilot_rows",
        "load_part9_module",
        "load_tensor_case",
        "create_model",
        "dice_from_prediction",
    ]
    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise AttributeError(
            "Part 11 implementation missing required API: " + ", ".join(missing)
        )
    return module


def load_checkpoint(model, path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)

    state = None
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(ckpt, dict) and isinstance(ckpt.get(key), dict):
            state = ckpt[key]
            break

    if state is None:
        if isinstance(ckpt, dict) and all(
            isinstance(v, torch.Tensor) for v in ckpt.values()
        ):
            state = ckpt
        else:
            raise RuntimeError(
                "Could not locate model state dictionary in Part 11 checkpoint."
            )

    # Remove DataParallel prefix if present.
    cleaned = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    result = model.load_state_dict(cleaned, strict=False)

    print(f"Missing keys     : {len(result.missing_keys)}")
    print(f"Unexpected keys  : {len(result.unexpected_keys)}")
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Checkpoint architecture mismatch.")

    meta = {
        "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "checkpoint_best_val_dice": (
            ckpt.get("best_val_dice") if isinstance(ckpt, dict) else None
        ),
    }
    return meta


def _normalize_3d_case(image, mask):
    """
    Enforce the Part 15 model contract:
      image -> [D, H, W] before batching
      mask  -> [D, H, W]

    Part 9/Part 11 may return a singleton channel dimension for some
    cases.  Remove ONLY singleton dimensions; never reinterpret D as C.
    """
    image = torch.as_tensor(image)
    mask = torch.as_tensor(mask)

    # Remove only singleton dimensions.
    while image.ndim > 3:
        singleton = [i for i, s in enumerate(image.shape) if s == 1]
        if not singleton:
            raise RuntimeError(
                f"Part 15 expected a 3-D image [D,H,W], got shape {tuple(image.shape)}"
            )
        image = image.squeeze(singleton[0])

    while mask.ndim > 3:
        singleton = [i for i, s in enumerate(mask.shape) if s == 1]
        if not singleton:
            raise RuntimeError(
                f"Part 15 expected a 3-D mask [D,H,W], got shape {tuple(mask.shape)}"
            )
        mask = mask.squeeze(singleton[0])

    if image.ndim != 3:
        raise RuntimeError(
            f"Part 15 image must be 3-D [D,H,W], got {tuple(image.shape)}"
        )
    if mask.ndim != 3:
        raise RuntimeError(
            f"Part 15 mask must be 3-D [D,H,W], got {tuple(mask.shape)}"
        )

    if tuple(image.shape) != tuple(mask.shape):
        raise RuntimeError(
            "Part 15 image/mask spatial mismatch: "
            f"image={tuple(image.shape)}, mask={tuple(mask.shape)}"
        )

    return image.float().contiguous(), mask.long().contiguous()


def safe_case(part11, part9, row):
    image, mask, info = part11.load_tensor_case(row, part9)
    image, mask = _normalize_3d_case(image, mask)

    # The model expects [B, C, D, H, W].  At this stage C=1.
    image = image.unsqueeze(0)
    mask = mask.contiguous()

    return image, mask, info


def evaluate(
    model,
    rows: pd.DataFrame,
    part11,
    part9,
    loss_fn,
    device: torch.device,
    amp_enabled: bool,
):
    model.eval()
    losses = []
    dices = []
    class_acc = {c: [] for c in range(1, NUM_CLASSES)}
    fallback_count = 0

    with torch.no_grad():
        for i, (_, row) in enumerate(rows.iterrows(), start=1):
            image, mask, info = safe_case(part11, part9, row)
            if info.get("part11_loader") == "local_robust_fallback":
                fallback_count += 1

            image = image.unsqueeze(0).to(device, non_blocking=True)
            mask_d = mask.unsqueeze(0).to(device, non_blocking=True)

            if image.ndim != 5 or image.shape[1] != 1:
                raise RuntimeError(
                    f"Part 15 model input must be [B,1,D,H,W], got {tuple(image.shape)}"
                )
            if mask_d.ndim != 4:
                raise RuntimeError(
                    f"Part 15 mask must be [B,D,H,W], got {tuple(mask_d.shape)}"
                )

            with autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
                logits = model(image)
                loss = loss_fn(logits, mask_d.unsqueeze(1))

            mean_dice, class_dice = part11.dice_from_prediction(
                logits, mask_d
            )
            losses.append(float(loss.detach().cpu()))
            dices.append(float(mean_dice))
            for c in range(1, NUM_CLASSES):
                class_acc[c].append(float(class_dice[c - 1]))

            if i % 25 == 0 or i == len(rows):
                print(
                    f"  VAL {i:03d}/{len(rows)} "
                    f"Loss={np.mean(losses):.4f} "
                    f"Dice={np.mean(dices):.6f}"
                )

    summary = {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "median_dice": float(np.median(dices)),
        "class_dice": {
            str(c): float(np.mean(class_acc[c])) for c in range(1, NUM_CLASSES)
        },
        "fallback_count": fallback_count,
    }
    return summary


def train_epoch(
    model,
    rows: pd.DataFrame,
    part11,
    part9,
    loss_fn,
    optimizer,
    scaler,
    device,
    amp_enabled,
    epoch,
):
    model.train()
    losses = []
    dices = []
    fallback_count = 0

    optimizer.zero_grad(set_to_none=True)

    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        image, mask, info = safe_case(part11, part9, row)
        if info.get("part11_loader") == "local_robust_fallback":
            fallback_count += 1

        image = image.unsqueeze(0).to(device, non_blocking=True)
        mask_d = mask.unsqueeze(0).to(device, non_blocking=True)

        if image.ndim != 5 or image.shape[1] != 1:
            raise RuntimeError(
                f"Part 15 model input must be [B,1,D,H,W], got {tuple(image.shape)}"
            )
        if mask_d.ndim != 4:
            raise RuntimeError(
                f"Part 15 mask must be [B,D,H,W], got {tuple(mask_d.shape)}"
            )

        with autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
            logits = model(image)
            loss = loss_fn(logits, mask_d.unsqueeze(1))

        scaler.scale(loss).backward()

        if GRAD_CLIP is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            mean_dice, _ = part11.dice_from_prediction(
                logits.detach(), mask_d
            )

        losses.append(float(loss.detach().cpu()))
        dices.append(float(mean_dice))

        if i == 1 or i % 25 == 0 or i == len(rows):
            print(
                f"  TRAIN {i:03d}/{len(rows)} "
                f"Loss={np.mean(losses):.4f} "
                f"Dice={np.mean(dices):.6f}"
            )

    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "fallback_count": fallback_count,
    }


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    best_val_dice,
    history,
    path,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": best_val_dice,
            "history": history,
            "seed": SEED,
            "patch_size": PATCH_SIZE,
            "feature_size": FEATURE_SIZE,
            "num_classes": NUM_CLASSES,
            "train_cases": TRAIN_CASES,
            "validation_cases": VAL_CASES,
            "source_checkpoint": str(PART11_CHECKPOINT),
        },
        path,
    )


def main():
    seed_everything()

    banner("PHASE 4 - PART 15")
    print("RSNA-ONLY EXTENDED CONTROLLED SWIN-UNETR TRAINING")
    print()
    print("This stage continues from the reproducible Part 11 checkpoint.")
    print("RSNA only. SPIDER is not used. Test set is not used.")
    print()

    print("PROJECT ROOT")
    print(PROJECT_ROOT)
    print()
    print("RSNA DATASET")
    print(RSNA_DIR)
    print()
    print("PART 11 CHECKPOINT")
    print(PART11_CHECKPOINT)
    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    banner("PATH VALIDATION")
    require_paths()
    print("All required paths found.")

    banner("PYTORCH / GPU ENVIRONMENT")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"Device          : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU             : {props.name}")
        print(f"GPU memory      : {props.total_memory / 1024**3:.2f} GB")
    print(f"Patch size      : {PATCH_SIZE}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Feature size    : {FEATURE_SIZE}")
    print(f"Classes         : {NUM_CLASSES}")
    print(f"Train cases     : {TRAIN_CASES}")
    print(f"Validation      : {VAL_CASES}")
    print(f"Epochs          : {EPOCHS}")
    print(f"Learning rate   : {LR}")
    print(f"AMP             : {AMP_ENABLED and device.type == 'cuda'}")

    banner("IMPORTING VALIDATED PART 11 IMPLEMENTATION")
    part11 = import_part11()
    part9 = part11.load_part9_module()
    print("✓ Part 11 APIs imported.")
    print("✓ Part 9 loader imported through Part 11.")
    print("✓ Part 11 Dice API: dice_from_prediction(logits, target).")

    banner("SELECTING EXPANDED CONTROLLED COHORT")
    train_manifest = pd.read_csv(TRAIN_MANIFEST)
    val_manifest = pd.read_csv(VAL_MANIFEST)

    train_rows = part11.select_pilot_rows(
        TRAIN_MANIFEST, TRAIN_CASES, SEED
    ).reset_index(drop=True)
    val_rows = part11.select_pilot_rows(
        VAL_MANIFEST, VAL_CASES, SEED
    ).reset_index(drop=True)

    print(f"Train manifest total : {len(train_manifest)}")
    print(f"Validation total     : {len(val_manifest)}")
    print(f"Selected train cases : {len(train_rows)}")
    print(f"Selected val cases   : {len(val_rows)}")

    # Save exact cohorts for reproducibility.
    train_rows.to_csv(OUTPUT_DIR / "part15_train_cohort.csv", index=False)
    val_rows.to_csv(OUTPUT_DIR / "part15_validation_cohort.csv", index=False)

    overlap = set(
        zip(train_rows["study_id"].astype(str), train_rows["series_id"].astype(str))
    ) & set(
        zip(val_rows["study_id"].astype(str), val_rows["series_id"].astype(str))
    )
    print(f"Train/validation overlap: {len(overlap)}")
    if overlap:
        raise RuntimeError("Train/validation leakage detected.")

    banner("CREATING SWIN-UNETR AND LOADING PART 11 CHECKPOINT")
    model = part11.create_model(device)
    model = model.to(device)
    print(f"Total parameters : {sum(p.numel() for p in model.parameters()):,}")

    ckpt_meta = load_checkpoint(model, PART11_CHECKPOINT, device)
    print(f"Source epoch     : {ckpt_meta['checkpoint_epoch']}")
    print(f"Source best Dice : {ckpt_meta['checkpoint_best_val_dice']}")

    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler("cuda", enabled=AMP_ENABLED and device.type == "cuda")

    # Establish a baseline before additional training.
    banner("PRE-TRAINING BASELINE ON EXPANDED VALIDATION COHORT")
    print("Input contract: image [B,1,D,H,W], mask [B,D,H,W]")

    # Validate the complete expanded validation cohort's tensor contract before
    # the first model evaluation. This prevents a malformed case from entering
    # Swin-UNETR and producing an obscure LayerNorm error.
    for check_i, (_, check_row) in enumerate(val_rows.iterrows(), start=1):
        check_image, check_mask, _ = safe_case(part11, part9, check_row)
        expected = (1, *PATCH_SIZE)
        if check_image.ndim != 4 or check_image.shape[0] != 1:
            raise RuntimeError(
                f"Validation case {check_i} has invalid pre-batch image shape "
                f"{tuple(check_image.shape)}; expected [1,D,H,W]."
            )
        if check_mask.ndim != 3:
            raise RuntimeError(
                f"Validation case {check_i} has invalid mask shape "
                f"{tuple(check_mask.shape)}; expected [D,H,W]."
            )
        # Do not require native dimensions to equal PATCH_SIZE here. The
        # validated Part 11 preprocessing is responsible for producing the
        # final model patch in load_tensor_case/preprocess_case.

    print(f"✓ Tensor contract validated for all {len(val_rows)} validation cases.")
    baseline = evaluate(
        model, val_rows, part11, part9, loss_fn, device,
        AMP_ENABLED and device.type == "cuda"
    )
    print(f"Baseline validation Dice : {baseline['dice']:.6f}")
    print(f"Baseline validation loss : {baseline['loss']:.6f}")

    history = []
    best_val_dice = baseline["dice"]

    baseline_path = CHECKPOINT_DIR / "part15_initialization_from_part11.pth"
    save_checkpoint(
        model, optimizer, scaler, 0, best_val_dice, history, baseline_path
    )

    banner("STARTING EXTENDED CONTROLLED TRAINING")
    print("No test data will be used.")
    print("The Part 11 checkpoint is initialization only; Part 15 creates new checkpoints.")
    print()

    for epoch in range(1, EPOCHS + 1):
        print(f"EPOCH {epoch}/{EPOCHS}")
        start = time.time()

        train_result = train_epoch(
            model, train_rows, part11, part9, loss_fn,
            optimizer, scaler, device,
            AMP_ENABLED and device.type == "cuda", epoch
        )

        val_result = evaluate(
            model, val_rows, part11, part9, loss_fn, device,
            AMP_ENABLED and device.type == "cuda"
        )

        elapsed = time.time() - start
        record = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_dice": train_result["dice"],
            "val_loss": val_result["loss"],
            "val_dice": val_result["dice"],
            "val_median_dice": val_result["median_dice"],
            "epoch_seconds": elapsed,
            "train_fallback_count": train_result["fallback_count"],
            "val_fallback_count": val_result["fallback_count"],
        }
        for c in range(1, NUM_CLASSES):
            record[f"val_dice_class_{c}"] = val_result["class_dice"][str(c)]
        history.append(record)

        print()
        print(
            f"Epoch {epoch}: "
            f"train loss={record['train_loss']:.6f}, "
            f"train Dice={record['train_dice']:.6f}, "
            f"val loss={record['val_loss']:.6f}, "
            f"val Dice={record['val_dice']:.6f}"
        )

        epoch_path = CHECKPOINT_DIR / f"epoch_{epoch:02d}.pth"
        save_checkpoint(
            model, optimizer, scaler, epoch,
            max(best_val_dice, val_result["dice"]),
            history, epoch_path
        )

        if val_result["dice"] > best_val_dice:
            best_val_dice = val_result["dice"]
            best_path = CHECKPOINT_DIR / "best_model.pth"
            save_checkpoint(
                model, optimizer, scaler, epoch,
                best_val_dice, history, best_path
            )
            print(f"✓ New best checkpoint: Dice={best_val_dice:.6f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        OUTPUT_DIR / "part15_training_history.csv", index=False
    )

    summary = {
        "phase": "4",
        "part": 15,
        "dataset": "RSNA",
        "spider_used": False,
        "test_set_used": False,
        "source_checkpoint": str(PART11_CHECKPOINT),
        "train_cases": len(train_rows),
        "validation_cases": len(val_rows),
        "epochs": EPOCHS,
        "patch_size": list(PATCH_SIZE),
        "feature_size": FEATURE_SIZE,
        "num_classes": NUM_CLASSES,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "amp": AMP_ENABLED and device.type == "cuda",
        "baseline_val_dice": baseline["dice"],
        "best_val_dice": best_val_dice,
        "baseline_val_loss": baseline["loss"],
        "final_val_dice": history[-1]["val_dice"] if history else baseline["dice"],
        "final_val_loss": history[-1]["val_loss"] if history else baseline["loss"],
        "training_performed": True,
        "model_weights_modified": True,
        "clinical_ground_truth_available": False,
        "evaluation_target": "RSNA point-derived pseudo-masks",
    }

    with open(
        OUTPUT_DIR / "phase4_part15_extended_training_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    report = OUTPUT_DIR / "reports" / "phase4_part15_extended_training_report.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write("PHASE 4 - PART 15\n")
        f.write("RSNA-ONLY EXTENDED CONTROLLED SWIN-UNETR TRAINING\n\n")
        f.write(f"Source checkpoint: {PART11_CHECKPOINT}\n")
        f.write(f"Train cases: {len(train_rows)}\n")
        f.write(f"Validation cases: {len(val_rows)}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Patch size: {PATCH_SIZE}\n")
        f.write(f"Baseline validation Dice: {baseline['dice']:.6f}\n")
        f.write(f"Best validation Dice: {best_val_dice:.6f}\n")
        f.write(f"Final validation Dice: {summary['final_val_dice']:.6f}\n\n")
        f.write("IMPORTANT: Dice is measured against RSNA point-derived ")
        f.write("pseudo-masks, not manually delineated clinical ground truth.\n")
        f.write("SPIDER was not used. The test set was not used.\n")

    banner("PART 15 FINAL SUMMARY")
    print(f"Expanded train cases      : {len(train_rows)}")
    print(f"Expanded validation cases : {len(val_rows)}")
    print(f"Additional epochs         : {EPOCHS}")
    print(f"Baseline validation Dice  : {baseline['dice']:.6f}")
    print(f"Best validation Dice      : {best_val_dice:.6f}")
    print(f"Final validation Dice     : {summary['final_val_dice']:.6f}")
    print()
    print("SPIDER used               : NO")
    print("Test set used             : NO")
    print("Training performed        : YES")
    print("Model weights modified    : YES")
    print()
    print("IMPORTANT:")
    print("These are pseudo-mask metrics, not final clinical segmentation")
    print("performance against manual segmentation ground truth.")
    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)
    banner("PHASE 4 - PART 15 COMPLETE")


if __name__ == "__main__":
    main()
