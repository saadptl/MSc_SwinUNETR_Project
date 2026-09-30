"""
PART 72
RSNA-ONLY CLASS-WISE FORENSICS OF PART 71 BEST CHECKPOINTS

Purpose
-------
Determine whether the meaningful Part 71 class-balanced DiceCE improvement
(+0.029881 FG Dice) actually improves the suppressed C2/C3/C4 classes,
or mainly increases foreground prediction in C1/C5.

Evaluation-only:
- No training
- No optimizer updates
- No SPIDER
- No RSNA test set
- First 50 Part 15 validation cases
- Same R2_FULL centered crop used in Part 70/71
- Exact Part 71 E3 checkpoints
"""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss
from scipy import ndimage


# ============================================================================
# PATHS
# ============================================================================
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

P11 = SRC / "segmentation_rsna_part11_controlled_pilot_training.py"
P9 = SRC / "segmentation_rsna_part9_3d_dataset_loader.py"

P15 = ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
VAL_CSV = P15 / "part15_validation_cohort.csv"

P71 = ROOT / "outputs" / "segmentation" / "rsna_part71_class_balanced_dicece_training"
CKPT_DIR = P71 / "checkpoints"

OUT = ROOT / "outputs" / "segmentation" / "rsna_part72_part71_classwise_forensics"
REPORT = OUT / "reports"
REPORT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOCKED CONFIGURATION
# ============================================================================
VAL_N = 50
FULL = (64, 96, 96)
CROP = (32, 64, 64)
NUM_CLASSES = 6

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASSES = {
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}

CHECKPOINTS = {
    "R2_FULL_BASELINE_E3": CKPT_DIR / "part71_r2_full_baseline_epoch3.pth",
    "R2_FULL_CLASS_BALANCED_E3": CKPT_DIR / "part71_r2_full_class_balanced_epoch3.pth",
}


def banner(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


def reset_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================================
# R2 PSEUDO-MASK / CROP
# ============================================================================
def dilate_labels(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    if radius == 0:
        return mask.copy()

    result = mask.copy()

    for c in range(1, NUM_CLASSES):
        binary = mask == c
        if not binary.any():
            continue

        structure = np.ones((2 * radius + 1,) * 3, dtype=bool)
        expanded = ndimage.binary_dilation(binary, structure=structure)
        result[expanded & (result == 0)] = c

    return result


def centered_crop(
    mask: np.ndarray,
    center: np.ndarray,
    shape: tuple[int, int, int],
):
    starts = []

    for dim, size, c in zip(mask.shape, shape, center):
        start = int(round(float(c) - size / 2.0))
        start = max(0, min(start, dim - size))
        starts.append(start)

    z0, y0, x0 = starts

    crop = mask[
        z0:z0 + shape[0],
        y0:y0 + shape[1],
        x0:x0 + shape[2],
    ]

    return crop, tuple(starts)


def load_case(row: pd.Series, part9: Any, part11: Any):
    loaded = part11.load_tensor_case(row, part9)

    if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
        raise RuntimeError("Unexpected Part 11 load_tensor_case() result.")

    image = torch.as_tensor(loaded[0])
    mask = torch.as_tensor(loaded[1])

    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)

    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    image_np = image.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(np.int64)

    resized_mask = part11.resize_3d(mask_np, FULL, is_mask=True)
    r2 = dilate_labels(resized_mask, 2)

    q = np.argwhere(r2 > 0)
    if len(q):
        center = q.mean(axis=0)
    else:
        center = np.array([31.5, 47.5, 47.5])

    _, starts = centered_crop(r2, center, CROP)
    z0, y0, x0 = starts

    image_resized = part11.resize_3d(
        image_np,
        FULL,
        is_mask=False,
    )

    image_crop = image_resized[
        z0:z0 + CROP[0],
        y0:y0 + CROP[1],
        x0:x0 + CROP[2],
    ]

    mask_crop = r2[
        z0:z0 + CROP[0],
        y0:y0 + CROP[1],
        x0:x0 + CROP[2],
    ]

    return (
        torch.as_tensor(image_crop, dtype=torch.float32),
        torch.as_tensor(mask_crop, dtype=torch.long),
    )


# ============================================================================
# MODEL / METRICS
# ============================================================================
def create_model(part11: Any):
    return part11.create_model(DEVICE).to(DEVICE)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def class_metrics(pred: np.ndarray, target: np.ndarray, c: int) -> dict:
    p = pred == c
    t = target == c

    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())

    pred_n = int(p.sum())
    target_n = int(t.sum())

    return {
        "target_voxels": target_n,
        "predicted_voxels": pred_n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": safe_div(2 * tp, pred_n + target_n),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "iou": safe_div(tp, tp + fp + fn),
        "empty_cases": int(pred_n == 0),
        "prediction_target_ratio": safe_div(pred_n, target_n),
    }


def evaluate_condition(
    condition: str,
    checkpoint_path: Path,
    rows: pd.DataFrame,
    part9: Any,
    part11: Any,
):
    banner(f"PART 72 CONDITION: {condition}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state = checkpoint.get("model_state_dict", checkpoint)

    model = create_model(part11)
    model.load_state_dict(state, strict=True)
    model.eval()

    # The checkpoint's loss is reported for reference only.
    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    ).to(DEVICE)

    class_records = []
    case_records = []
    losses = []

    with torch.no_grad():
        for case_no, (_, row) in enumerate(rows.iterrows(), start=1):
            image, target = load_case(row, part9, part11)

            x = image.unsqueeze(0).unsqueeze(0).to(
                DEVICE,
                non_blocking=True,
            )
            y = target.unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.autocast(
                device_type="cuda",
                enabled=DEVICE.type == "cuda",
            ):
                logits = model(x)
                loss = loss_fn(logits, y)

            pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy()
            tgt = target.numpy()

            losses.append(float(loss.detach().cpu()))

            case_records.append({
                "condition": condition,
                "case_no": case_no,
                "target_foreground_voxels": int((tgt > 0).sum()),
                "predicted_foreground_voxels": int((pred > 0).sum()),
                "empty_foreground_prediction": int((pred > 0).sum() == 0),
            })

            for c in range(1, NUM_CLASSES):
                m = class_metrics(pred, tgt, c)
                class_records.append({
                    "condition": condition,
                    "case_no": case_no,
                    "class_id": c,
                    "class_name": CLASSES[c],
                    **m,
                })

    cdf = pd.DataFrame(class_records)
    kdf = pd.DataFrame(case_records)

    summary_rows = []

    for c in range(1, NUM_CLASSES):
        g = cdf[cdf["class_id"] == c]

        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        fn = int(g["fn"].sum())
        pred_n = int(g["predicted_voxels"].sum())
        target_n = int(g["target_voxels"].sum())

        summary_rows.append({
            "condition": condition,
            "class_id": c,
            "class_name": CLASSES[c],
            "target_voxels": target_n,
            "predicted_voxels": pred_n,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "dice": safe_div(2 * tp, pred_n + target_n),
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
            "iou": safe_div(tp, tp + fp + fn),
            "empty_cases": int(g["empty_cases"].sum()),
            "prediction_target_ratio": safe_div(pred_n, target_n),
        })

    sdf = pd.DataFrame(summary_rows)

    total_tp = int(sdf["tp"].sum())
    total_pred = int(sdf["predicted_voxels"].sum())
    total_target = int(sdf["target_voxels"].sum())

    fg_dice = safe_div(
        2 * total_tp,
        total_pred + total_target,
    )

    print(
        f"FGDice={fg_dice:.6f} | "
        f"Loss={np.mean(losses):.6f} | "
        f"PredFG={int(kdf['predicted_foreground_voxels'].sum())} | "
        f"Empty={int(kdf['empty_foreground_prediction'].sum())}/{len(kdf)}"
    )

    print("\nClass-wise:")
    print(
        f"{'Class':<42} {'Dice':>10} {'Precision':>11} "
        f"{'Recall':>10} {'PredFG':>12} {'Target':>12} {'Ratio':>10}"
    )

    for _, r in sdf.iterrows():
        print(
            f"C{int(r['class_id'])} {r['class_name']:<36} "
            f"{r['dice']:>10.6f} "
            f"{r['precision']:>11.6f} "
            f"{r['recall']:>10.6f} "
            f"{int(r['predicted_voxels']):>12} "
            f"{int(r['target_voxels']):>12} "
            f"{r['prediction_target_ratio']:>10.3f}"
        )

    del model, loss_fn
    reset_cuda()

    return sdf, cdf, kdf, {
        "condition": condition,
        "loss": float(np.mean(losses)),
        "foreground_dice": fg_dice,
        "predicted_foreground_voxels": int(
            kdf["predicted_foreground_voxels"].sum()
        ),
        "empty_foreground_cases": int(
            kdf["empty_foreground_prediction"].sum()
        ),
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    banner("PART 72")
    print("RSNA-ONLY PART 71 BEST CHECKPOINT CLASS-WISE FORENSICS")
    print("")
    print("Purpose:")
    print(
        "Determine whether the Part 71 class-balanced improvement "
        "rescues suppressed classes or mainly increases C1/C5 foreground."
    )
    print("")
    print("Evaluation only.")
    print("No training.")
    print("No SPIDER.")
    print("No RSNA test set.")

    banner("PART 72 PATH VALIDATION")

    required = {
        "Project root": ROOT,
        "Part 11": P11,
        "Part 9": P9,
        "Part 15 validation cohort": VAL_CSV,
        "Part 71 output": P71,
    }

    missing = []
    for name, path in required.items():
        ok = path.exists()
        print(f"{name:<34}: {'FOUND' if ok else 'MISSING'}")
        if not ok:
            missing.append(str(path))

    for name, path in CHECKPOINTS.items():
        ok = path.exists()
        print(f"{name:<34}: {'FOUND' if ok else 'MISSING'}")
        if not ok:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "\n".join(missing)
        )

    val_rows = pd.read_csv(VAL_CSV).head(VAL_N).copy()

    if len(val_rows) != VAL_N:
        raise RuntimeError(
            f"Expected {VAL_N} validation cases, found {len(val_rows)}."
        )

    banner("PART 72 LOCKED EVALUATION")
    print(f"Device                     : {DEVICE}")
    print(f"Validation cases           : {len(val_rows)}")
    print(f"Full shape                 : {FULL}")
    print(f"Centered crop              : {CROP}")
    print("Supervision                : R2_FULL")
    print("Checkpoint epoch           : E3")
    print("Part 71 baseline Dice      : 0.014145")
    print("Part 71 class-balanced Dice: 0.044026")
    print("Part 71 Dice delta         : +0.029881")

    part11 = load_module(P11, "part72_part11")
    part9 = load_module(P9, "part72_part9")

    all_class = []
    all_case = []
    summaries = {}

    for condition, checkpoint_path in CHECKPOINTS.items():
        sdf, cdf, kdf, summary = evaluate_condition(
            condition,
            checkpoint_path,
            val_rows,
            part9,
            part11,
        )

        all_class.append(sdf)
        all_case.append(kdf)
        summaries[condition] = summary

    class_df = pd.concat(all_class, ignore_index=True)
    case_df = pd.concat(all_case, ignore_index=True)

    # ------------------------------------------------------------------------
    # DELTA ANALYSIS
    # ------------------------------------------------------------------------
    banner("PART 72 CLASS-WISE DELTA: CLASS-BALANCED MINUS BASELINE")

    pivot = class_df.pivot(
        index=["class_id", "class_name"],
        columns="condition",
        values=[
            "dice",
            "precision",
            "recall",
            "predicted_voxels",
            "target_voxels",
            "prediction_target_ratio",
        ],
    )

    comparison_rows = []

    for c in range(1, NUM_CLASSES):
        base = class_df[
            (class_df.condition == "R2_FULL_BASELINE_E3")
            & (class_df.class_id == c)
        ].iloc[0]

        bal = class_df[
            (class_df.condition == "R2_FULL_CLASS_BALANCED_E3")
            & (class_df.class_id == c)
        ].iloc[0]

        comparison_rows.append({
            "class_id": c,
            "class_name": CLASSES[c],
            "baseline_dice": base["dice"],
            "balanced_dice": bal["dice"],
            "delta_dice": bal["dice"] - base["dice"],
            "baseline_precision": base["precision"],
            "balanced_precision": bal["precision"],
            "delta_precision": bal["precision"] - base["precision"],
            "baseline_recall": base["recall"],
            "balanced_recall": bal["recall"],
            "delta_recall": bal["recall"] - base["recall"],
            "baseline_predicted_voxels": base["predicted_voxels"],
            "balanced_predicted_voxels": bal["predicted_voxels"],
            "delta_predicted_voxels": (
                bal["predicted_voxels"] - base["predicted_voxels"]
            ),
            "target_voxels": base["target_voxels"],
            "baseline_prediction_target_ratio": base[
                "prediction_target_ratio"
            ],
            "balanced_prediction_target_ratio": bal[
                "prediction_target_ratio"
            ],
        })

    comparison = pd.DataFrame(comparison_rows)

    for _, r in comparison.iterrows():
        print(
            f"C{int(r['class_id'])} {r['class_name']:<36} "
            f"Dice {r['baseline_dice']:.6f} -> "
            f"{r['balanced_dice']:.6f} "
            f"({r['delta_dice']:+.6f}) | "
            f"Recall {r['baseline_recall']:.6f} -> "
            f"{r['balanced_recall']:.6f} "
            f"({r['delta_recall']:+.6f})"
        )

    # ------------------------------------------------------------------------
    # DIAGNOSIS
    # ------------------------------------------------------------------------
    c2_c3 = comparison[
        comparison["class_id"].isin([2, 3])
    ]

    all_dice_delta = comparison["delta_dice"].values
    c2c3_gain = float(c2_c3["delta_dice"].mean())

    # Conservative interpretation:
    # - If C2/C3 improve materially, class balancing helps suppressed classes.
    # - If overall improvement exists but C2/C3 remain essentially suppressed,
    #   the gain is mostly from other classes.
    if c2c3_gain >= 0.01:
        diagnosis = "CLASS_BALANCING_IMPROVES_SUPPRESSED_C2_C3_CLASSES"
    elif float(np.mean(all_dice_delta)) >= 0.01:
        diagnosis = "CLASS_BALANCING_IMPROVES_FOREGROUND_BUT_C2_C3_REMAIN_LIMITED"
    elif float(np.max(all_dice_delta)) > 0:
        diagnosis = "SMALL_CLASS_SPECIFIC_CLASS_BALANCING_EFFECT"
    else:
        diagnosis = "NO_CLASSWISE_CLASS_BALANCING_IMPROVEMENT"

    banner("PART 72 FINAL SUMMARY")

    base = summaries["R2_FULL_BASELINE_E3"]
    bal = summaries["R2_FULL_CLASS_BALANCED_E3"]

    print(f"Baseline E3 FG Dice       : {base['foreground_dice']:.6f}")
    print(f"Balanced E3 FG Dice       : {bal['foreground_dice']:.6f}")
    print(
        f"Overall Dice delta        : "
        f"{bal['foreground_dice'] - base['foreground_dice']:+.6f}"
    )
    print(
        f"Baseline PredFG           : "
        f"{base['predicted_foreground_voxels']}"
    )
    print(
        f"Balanced PredFG           : "
        f"{bal['predicted_foreground_voxels']}"
    )
    print(
        f"C2/C3 mean Dice delta     : "
        f"{c2c3_gain:+.6f}"
    )
    print(f"Diagnosis                  : {diagnosis}")

    # ------------------------------------------------------------------------
    # SAVE REPORTS
    # ------------------------------------------------------------------------
    class_df.to_csv(
        REPORT / "part72_classwise_metrics.csv",
        index=False,
    )

    case_df.to_csv(
        REPORT / "part72_case_metrics.csv",
        index=False,
    )

    comparison.to_csv(
        REPORT / "part72_classwise_comparison.csv",
        index=False,
    )

    summary_payload = {
        "part": 72,
        "validation_cases": VAL_N,
        "full_shape": FULL,
        "crop": CROP,
        "checkpoint_epoch": 3,
        "baseline": base,
        "class_balanced": bal,
        "overall_dice_delta": (
            bal["foreground_dice"] - base["foreground_dice"]
        ),
        "c2_c3_mean_dice_delta": c2c3_gain,
        "diagnosis": diagnosis,
    }

    with (REPORT / "part72_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary_payload, f, indent=2)

    with (REPORT / "part72_report.txt").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("PART 72\n")
        f.write("RSNA-ONLY PART 71 BEST CHECKPOINT CLASS-WISE FORENSICS\n\n")
        f.write(f"Baseline E3 FG Dice: {base['foreground_dice']:.6f}\n")
        f.write(f"Balanced E3 FG Dice: {bal['foreground_dice']:.6f}\n")
        f.write(
            f"Overall Dice delta: "
            f"{bal['foreground_dice'] - base['foreground_dice']:+.6f}\n"
        )
        f.write(f"C2/C3 mean Dice delta: {c2c3_gain:+.6f}\n")
        f.write(f"Diagnosis: {diagnosis}\n\n")
        f.write(comparison.to_string(index=False))

    banner("PART 72 OUTPUTS")
    print(REPORT / "part72_classwise_metrics.csv")
    print(REPORT / "part72_case_metrics.csv")
    print(REPORT / "part72_classwise_comparison.csv")
    print(REPORT / "part72_summary.json")
    print(REPORT / "part72_report.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        banner("PART 72 FAILED")
        print(type(exc).__name__, str(exc))
        raise
