"""
PART 73
RSNA-ONLY CLASS-WISE PROBABILITY / THRESHOLD FORENSICS

Purpose
-------
Determine whether the Part 71 class-balanced E3 model contains useful probability
signal for the severely suppressed C2/C3 classes that is being lost by argmax,
or whether those classes are genuinely not learned.

Evaluation only:
- No training
- No optimizer updates
- No SPIDER
- No RSNA test set
- First 50 Part 15 validation cases
- Same R2_FULL centered crop as Parts 70-72
- Part 71 E3 baseline and class-balanced checkpoints
- Uses softmax probabilities rather than only argmax labels

For each foreground class independently:
  * Sweep probability thresholds from 0.05 to 0.50
  * Compute Dice, precision, recall, IoU
  * Report best threshold
  * Compare with standard argmax performance
  * Measure probability statistics inside/outside the target class

This is NOT a replacement for argmax multi-class segmentation.
It is a diagnostic to determine whether suppressed classes have hidden
probability signal.
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

OUT = ROOT / "outputs" / "segmentation" / "rsna_part73_classwise_probability_threshold_forensics"
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
    "R2_FULL_BASELINE_E3":
        CKPT_DIR / "part71_r2_full_baseline_epoch3.pth",
    "R2_FULL_CLASS_BALANCED_E3":
        CKPT_DIR / "part71_r2_full_class_balanced_epoch3.pth",
}

# Probability thresholds.
# 0.50 is included as the conventional reference.
THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]


# ============================================================================
# UTILITIES
# ============================================================================
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


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


# ============================================================================
# R2 PSEUDO-MASK / CENTERED CROP
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

        structure = np.ones(
            (2 * radius + 1,) * 3,
            dtype=bool,
        )

        expanded = ndimage.binary_dilation(
            binary,
            structure=structure,
        )

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


def load_case(
    row: pd.Series,
    part9: Any,
    part11: Any,
):
    loaded = part11.load_tensor_case(row, part9)

    if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
        raise RuntimeError(
            "Unexpected Part 11 load_tensor_case() result."
        )

    image = torch.as_tensor(loaded[0])
    mask = torch.as_tensor(loaded[1])

    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)

    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    image_np = image.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(np.int64)

    resized_mask = part11.resize_3d(
        mask_np,
        FULL,
        is_mask=True,
    )

    r2 = dilate_labels(resized_mask, 2)

    q = np.argwhere(r2 > 0)

    if len(q):
        center = q.mean(axis=0)
    else:
        center = np.array([31.5, 47.5, 47.5])

    _, starts = centered_crop(
        r2,
        center,
        CROP,
    )

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
        torch.as_tensor(
            image_crop,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            mask_crop,
            dtype=torch.long,
        ),
    )


# ============================================================================
# METRICS
# ============================================================================
def binary_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
    class_id: int,
) -> dict:
    pred = probability >= threshold
    truth = target == class_id

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())

    pred_n = int(pred.sum())
    target_n = int(truth.sum())

    return {
        "threshold": threshold,
        "class_id": class_id,
        "class_name": CLASSES[class_id],
        "target_voxels": target_n,
        "predicted_voxels": pred_n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": safe_div(
            2 * tp,
            pred_n + target_n,
        ),
        "precision": safe_div(
            tp,
            tp + fp,
        ),
        "recall": safe_div(
            tp,
            tp + fn,
        ),
        "iou": safe_div(
            tp,
            tp + fp + fn,
        ),
        "prediction_target_ratio": safe_div(
            pred_n,
            target_n,
        ),
    }


def probability_statistics(
    probability: np.ndarray,
    target: np.ndarray,
    class_id: int,
) -> dict:
    truth = target == class_id
    nontruth = ~truth

    inside = probability[truth]
    outside = probability[nontruth]

    if inside.size == 0:
        return {
            "target_voxels": 0,
            "mean_probability_target": 0.0,
            "median_probability_target": 0.0,
            "max_probability_target": 0.0,
            "p90_probability_target": 0.0,
            "mean_probability_non_target": 0.0,
            "median_probability_non_target": 0.0,
            "max_probability_non_target": 0.0,
            "target_mean_minus_non_target_mean": 0.0,
        }

    return {
        "target_voxels": int(inside.size),
        "mean_probability_target": float(np.mean(inside)),
        "median_probability_target": float(np.median(inside)),
        "max_probability_target": float(np.max(inside)),
        "p90_probability_target": float(np.percentile(inside, 90)),
        "mean_probability_non_target": float(np.mean(outside)),
        "median_probability_non_target": float(np.median(outside)),
        "max_probability_non_target": float(np.max(outside)),
        "target_mean_minus_non_target_mean": float(
            np.mean(inside) - np.mean(outside)
        ),
    }


# ============================================================================
# MODEL EVALUATION
# ============================================================================
def create_model(part11: Any):
    return part11.create_model(DEVICE).to(DEVICE)


def evaluate_condition(
    condition: str,
    checkpoint_path: Path,
    rows: pd.DataFrame,
    part9: Any,
    part11: Any,
):
    banner(f"PART 73 CONDITION: {condition}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model = create_model(part11)
    model.load_state_dict(
        state,
        strict=True,
    )
    model.eval()

    threshold_records = []
    probability_records = []
    argmax_records = []

    with torch.no_grad():
        for case_no, (_, row) in enumerate(
            rows.iterrows(),
            start=1,
        ):
            image, target = load_case(
                row,
                part9,
                part11,
            )

            x = image.unsqueeze(0).unsqueeze(0).to(
                DEVICE,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                enabled=DEVICE.type == "cuda",
            ):
                logits = model(x)
                probabilities = torch.softmax(
                    logits.float(),
                    dim=1,
                )[0].detach().cpu().numpy()

            pred_argmax = np.argmax(
                probabilities,
                axis=0,
            )

            target_np = target.numpy()

            # Standard argmax class metrics.
            for c in range(1, NUM_CLASSES):
                m = binary_metrics(
                    probabilities[c],
                    target_np,
                    0.5,
                    c,
                )

                # The threshold=0.5 binary result is intentionally reported
                # separately from argmax. It is not the same segmentation.
                argmax_truth = pred_argmax == c
                target_truth = target_np == c

                tp = int(
                    np.logical_and(
                        argmax_truth,
                        target_truth,
                    ).sum()
                )
                fp = int(
                    np.logical_and(
                        argmax_truth,
                        ~target_truth,
                    ).sum()
                )
                fn = int(
                    np.logical_and(
                        ~argmax_truth,
                        target_truth,
                    ).sum()
                )

                pred_n = int(argmax_truth.sum())
                target_n = int(target_truth.sum())

                argmax_records.append({
                    "condition": condition,
                    "case_no": case_no,
                    "class_id": c,
                    "class_name": CLASSES[c],
                    "target_voxels": target_n,
                    "predicted_voxels": pred_n,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "dice": safe_div(
                        2 * tp,
                        pred_n + target_n,
                    ),
                    "precision": safe_div(
                        tp,
                        tp + fp,
                    ),
                    "recall": safe_div(
                        tp,
                        tp + fn,
                    ),
                    "iou": safe_div(
                        tp,
                        tp + fp + fn,
                    ),
                })

                stats = probability_statistics(
                    probabilities[c],
                    target_np,
                    c,
                )

                probability_records.append({
                    "condition": condition,
                    "case_no": case_no,
                    "class_id": c,
                    "class_name": CLASSES[c],
                    **stats,
                })

                for threshold in THRESHOLDS:
                    tm = binary_metrics(
                        probabilities[c],
                        target_np,
                        threshold,
                        c,
                    )

                    threshold_records.append({
                        "condition": condition,
                        "case_no": case_no,
                        **tm,
                    })

    threshold_df = pd.DataFrame(
        threshold_records
    )

    probability_df = pd.DataFrame(
        probability_records
    )

    argmax_df = pd.DataFrame(
        argmax_records
    )

    # Aggregate threshold performance across all validation cases.
    threshold_summary = []

    for c in range(1, NUM_CLASSES):
        for threshold in THRESHOLDS:
            g = threshold_df[
                (threshold_df["class_id"] == c)
                & (
                    threshold_df["threshold"]
                    == threshold
                )
            ]

            tp = int(g["tp"].sum())
            fp = int(g["fp"].sum())
            fn = int(g["fn"].sum())
            pred_n = int(g["predicted_voxels"].sum())
            target_n = int(g["target_voxels"].sum())

            threshold_summary.append({
                "condition": condition,
                "class_id": c,
                "class_name": CLASSES[c],
                "threshold": threshold,
                "target_voxels": target_n,
                "predicted_voxels": pred_n,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "dice": safe_div(
                    2 * tp,
                    pred_n + target_n,
                ),
                "precision": safe_div(
                    tp,
                    tp + fp,
                ),
                "recall": safe_div(
                    tp,
                    tp + fn,
                ),
                "iou": safe_div(
                    tp,
                    tp + fp + fn,
                ),
                "prediction_target_ratio": safe_div(
                    pred_n,
                    target_n,
                ),
            })

    threshold_summary_df = pd.DataFrame(
        threshold_summary
    )

    best_rows = []

    for c in range(1, NUM_CLASSES):
        g = threshold_summary_df[
            threshold_summary_df["class_id"] == c
        ].copy()

        # Highest Dice. If tied, prefer higher precision.
        g = g.sort_values(
            ["dice", "precision", "recall"],
            ascending=False,
        )

        best = g.iloc[0].to_dict()

        a = argmax_df[
            argmax_df["class_id"] == c
        ].iloc[0]

        p = probability_df[
            probability_df["class_id"] == c
        ]

        best_rows.append({
            "condition": condition,
            "class_id": c,
            "class_name": CLASSES[c],
            "argmax_dice": float(a["dice"]),
            "argmax_precision": float(a["precision"]),
            "argmax_recall": float(a["recall"]),
            "best_threshold": float(best["threshold"]),
            "best_threshold_dice": float(best["dice"]),
            "best_threshold_precision": float(
                best["precision"]
            ),
            "best_threshold_recall": float(
                best["recall"]
            ),
            "threshold_dice_gain_over_argmax": (
                float(best["dice"]) - float(a["dice"])
            ),
            "mean_target_probability": float(
                p["mean_probability_target"].mean()
            ),
            "median_target_probability": float(
                p["median_probability_target"].mean()
            ),
            "max_target_probability": float(
                p["max_probability_target"].max()
            ),
            "mean_non_target_probability": float(
                p["mean_probability_non_target"].mean()
            ),
            "max_non_target_probability": float(
                p["max_probability_non_target"].max()
            ),
            "target_mean_minus_non_target_mean": float(
                p[
                    "target_mean_minus_non_target_mean"
                ].mean()
            ),
        })

    best_df = pd.DataFrame(best_rows)

    print("\nClass-wise probability/threshold summary:")
    print(
        f"{'Class':<42} "
        f"{'Argmax':>10} "
        f"{'BestT':>8} "
        f"{'BestDice':>10} "
        f"{'Gain':>10} "
        f"{'TgtProb':>10} "
        f"{'NonTgtProb':>11}"
    )

    for _, r in best_df.iterrows():
        print(
            f"C{int(r['class_id'])} "
            f"{r['class_name']:<36} "
            f"{r['argmax_dice']:>10.6f} "
            f"{r['best_threshold']:>8.2f} "
            f"{r['best_threshold_dice']:>10.6f} "
            f"{r['threshold_dice_gain_over_argmax']:>+10.6f} "
            f"{r['mean_target_probability']:>10.6f} "
            f"{r['mean_non_target_probability']:>11.6f}"
        )

    del model
    reset_cuda()

    return (
        threshold_df,
        threshold_summary_df,
        probability_df,
        argmax_df,
        best_df,
    )


# ============================================================================
# MAIN
# ============================================================================
def main():
    banner("PART 73")
    print("RSNA-ONLY CLASS-WISE PROBABILITY / THRESHOLD FORENSICS")
    print("")
    print(
        "Purpose:"
    )
    print(
        "Determine whether C2/C3 contain hidden probability signal "
        "that is lost by argmax."
    )
    print("")
    print("Evaluation only.")
    print("No training.")
    print("No optimizer.")
    print("No SPIDER.")
    print("No RSNA test set.")

    banner("PART 73 PATH VALIDATION")

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
        print(
            f"{name:<34}: "
            f"{'FOUND' if ok else 'MISSING'}"
        )

        if not ok:
            missing.append(str(path))

    for name, path in CHECKPOINTS.items():
        ok = path.exists()
        print(
            f"{name:<34}: "
            f"{'FOUND' if ok else 'MISSING'}"
        )

        if not ok:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "\n".join(missing)
        )

    val_rows = pd.read_csv(
        VAL_CSV
    ).head(VAL_N).copy()

    if len(val_rows) != VAL_N:
        raise RuntimeError(
            f"Expected {VAL_N} validation cases, "
            f"found {len(val_rows)}."
        )

    banner("PART 73 LOCKED EVALUATION")

    print(f"Device                     : {DEVICE}")
    print(f"Validation cases           : {VAL_N}")
    print(f"Full shape                 : {FULL}")
    print(f"Centered crop              : {CROP}")
    print("Supervision                : R2_FULL")
    print("Checkpoint epoch           : E3")
    print("Probability thresholds     : 0.05 - 0.50")
    print("Part 71 baseline Dice      : 0.014145")
    print("Part 71 class-balanced Dice: 0.044026")
    print("Part 72 balanced Dice      : 0.044049")

    part11 = load_module(
        P11,
        "part73_part11",
    )

    part9 = load_module(
        P9,
        "part73_part9",
    )

    all_threshold = []
    all_threshold_summary = []
    all_probability = []
    all_argmax = []
    all_best = []

    for condition, checkpoint_path in CHECKPOINTS.items():
        (
            threshold_df,
            threshold_summary_df,
            probability_df,
            argmax_df,
            best_df,
        ) = evaluate_condition(
            condition,
            checkpoint_path,
            val_rows,
            part9,
            part11,
        )

        all_threshold.append(threshold_df)
        all_threshold_summary.append(
            threshold_summary_df
        )
        all_probability.append(
            probability_df
        )
        all_argmax.append(
            argmax_df
        )
        all_best.append(
            best_df
        )

    threshold_df = pd.concat(
        all_threshold,
        ignore_index=True,
    )

    threshold_summary_df = pd.concat(
        all_threshold_summary,
        ignore_index=True,
    )

    probability_df = pd.concat(
        all_probability,
        ignore_index=True,
    )

    argmax_df = pd.concat(
        all_argmax,
        ignore_index=True,
    )

    best_df = pd.concat(
        all_best,
        ignore_index=True,
    )

    # ========================================================================
    # CROSS-CONDITION COMPARISON
    # ========================================================================
    banner(
        "PART 73 CROSS-CONDITION C2/C3 COMPARISON"
    )

    comparison_rows = []

    for c in range(1, NUM_CLASSES):
        base = best_df[
            (best_df.condition == "R2_FULL_BASELINE_E3")
            & (best_df.class_id == c)
        ].iloc[0]

        balanced = best_df[
            (
                best_df.condition
                == "R2_FULL_CLASS_BALANCED_E3"
            )
            & (best_df.class_id == c)
        ].iloc[0]

        comparison_rows.append({
            "class_id": c,
            "class_name": CLASSES[c],
            "baseline_argmax_dice": base[
                "argmax_dice"
            ],
            "baseline_best_threshold": base[
                "best_threshold"
            ],
            "baseline_best_threshold_dice": base[
                "best_threshold_dice"
            ],
            "baseline_threshold_gain": base[
                "threshold_dice_gain_over_argmax"
            ],
            "balanced_argmax_dice": balanced[
                "argmax_dice"
            ],
            "balanced_best_threshold": balanced[
                "best_threshold"
            ],
            "balanced_best_threshold_dice": balanced[
                "best_threshold_dice"
            ],
            "balanced_threshold_gain": balanced[
                "threshold_dice_gain_over_argmax"
            ],
            "balanced_minus_baseline_best_dice": (
                balanced["best_threshold_dice"]
                - base["best_threshold_dice"]
            ),
            "balanced_target_probability": balanced[
                "mean_target_probability"
            ],
            "balanced_non_target_probability": balanced[
                "mean_non_target_probability"
            ],
        })

    comparison = pd.DataFrame(
        comparison_rows
    )

    for _, r in comparison.iterrows():
        print(
            f"C{int(r['class_id'])} "
            f"{r['class_name']:<36} "
            f"Balanced best Dice="
            f"{r['balanced_best_threshold_dice']:.6f} "
            f"BestT="
            f"{r['balanced_best_threshold']:.2f} "
            f"Threshold gain="
            f"{r['balanced_threshold_gain']:+.6f}"
        )

    # ========================================================================
    # DIAGNOSIS
    # ========================================================================
    c2c3 = best_df[
        (best_df.condition == "R2_FULL_CLASS_BALANCED_E3")
        & (best_df.class_id.isin([2, 3]))
    ]

    c2c3_best_dice = float(
        c2c3["best_threshold_dice"].mean()
    )

    c2c3_gain = float(
        c2c3["threshold_dice_gain_over_argmax"].mean()
    )

    c2c3_prob_gap = float(
        c2c3[
            "target_mean_minus_non_target_mean"
        ].mean()
    )

    # Diagnostic interpretation:
    # Large threshold gain means probability signal exists below argmax.
    # Very small C2/C3 best Dice plus tiny probability separation indicates
    # genuinely weak learned signal.
    if c2c3_best_dice >= 0.05 and c2c3_gain >= 0.02:
        diagnosis = (
            "C2_C3_HAVE_HIDDEN_PROBABILITY_SIGNAL_REVEALED_BY_THRESHOLDING"
        )
    elif c2c3_gain >= 0.01:
        diagnosis = (
            "C2_C3_SHOW_SOME_HIDDEN_PROBABILITY_SIGNAL_BUT_REMAIN_WEAK"
        )
    elif c2c3_prob_gap > 0.01:
        diagnosis = (
            "C2_C3_HAVE_WEAK_PROBABILITY_SEPARATION_WITHOUT_STRONG_DICE_GAIN"
        )
    else:
        diagnosis = (
            "C2_C3_PROBABILITY_SIGNAL_REMAINS_SEVERELY_SUPPRESSED"
        )

    banner("PART 73 FINAL SUMMARY")

    print(
        f"C2/C3 mean best-threshold Dice : "
        f"{c2c3_best_dice:.6f}"
    )
    print(
        f"C2/C3 mean threshold gain      : "
        f"{c2c3_gain:+.6f}"
    )
    print(
        f"C2/C3 mean probability gap     : "
        f"{c2c3_prob_gap:+.6f}"
    )
    print(
        f"Diagnosis                       : "
        f"{diagnosis}"
    )

    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================
    threshold_df.to_csv(
        REPORT / "part73_case_threshold_metrics.csv",
        index=False,
    )

    threshold_summary_df.to_csv(
        REPORT / "part73_threshold_summary.csv",
        index=False,
    )

    probability_df.to_csv(
        REPORT / "part73_probability_statistics.csv",
        index=False,
    )

    argmax_df.to_csv(
        REPORT / "part73_argmax_classwise_metrics.csv",
        index=False,
    )

    best_df.to_csv(
        REPORT / "part73_best_threshold_classwise.csv",
        index=False,
    )

    comparison.to_csv(
        REPORT / "part73_cross_condition_comparison.csv",
        index=False,
    )

    summary = {
        "part": 73,
        "validation_cases": VAL_N,
        "full_shape": FULL,
        "crop": CROP,
        "checkpoint_epoch": 3,
        "thresholds": THRESHOLDS,
        "c2_c3_mean_best_threshold_dice": c2c3_best_dice,
        "c2_c3_mean_threshold_gain": c2c3_gain,
        "c2_c3_mean_probability_gap": c2c3_prob_gap,
        "diagnosis": diagnosis,
    }

    with (
        REPORT / "part73_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    with (
        REPORT / "part73_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "PART 73\n"
            "RSNA-ONLY CLASS-WISE PROBABILITY / "
            "THRESHOLD FORENSICS\n\n"
        )
        f.write(
            f"C2/C3 mean best-threshold Dice: "
            f"{c2c3_best_dice:.6f}\n"
        )
        f.write(
            f"C2/C3 mean threshold gain: "
            f"{c2c3_gain:+.6f}\n"
        )
        f.write(
            f"C2/C3 mean probability gap: "
            f"{c2c3_prob_gap:+.6f}\n"
        )
        f.write(
            f"Diagnosis: {diagnosis}\n\n"
        )
        f.write(
            comparison.to_string(index=False)
        )

    banner("PART 73 OUTPUTS")

    print(
        REPORT / "part73_case_threshold_metrics.csv"
    )
    print(
        REPORT / "part73_threshold_summary.csv"
    )
    print(
        REPORT / "part73_probability_statistics.csv"
    )
    print(
        REPORT / "part73_argmax_classwise_metrics.csv"
    )
    print(
        REPORT / "part73_best_threshold_classwise.csv"
    )
    print(
        REPORT / "part73_cross_condition_comparison.csv"
    )
    print(
        REPORT / "part73_summary.json"
    )
    print(
        REPORT / "part73_report.txt"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        banner("PART 73 FAILED")
        print(
            type(exc).__name__,
            str(exc),
        )
        raise
