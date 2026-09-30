"""
PART 74
RSNA-ONLY C2/C3 SPATIAL LOCALIZATION + CLASS-CONFUSION FORENSICS

Purpose
-------
Determine whether C2/C3 probability activation is:
1. correctly localized but weak,
2. diffuse / excessive,
3. confused with each other, or
4. spatially misplaced.

Evaluation only:
- No training
- No optimizer
- No model modification
- No SPIDER
- No RSNA test set
- First 50 Part 15 validation cases
- R2_FULL centered crop
- Part 71 R2_FULL_CLASS_BALANCED_E3 checkpoint

Outputs:
- C2/C3 thresholded spatial overlap
- predicted foreground size ratios
- prediction-to-target distance
- target-to-prediction distance
- symmetric distance
- high-probability localization
- C2 <-> C3 confusion
- probability statistics
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
CKPT = (
    P71
    / "checkpoints"
    / "part71_r2_full_class_balanced_epoch3.pth"
)

OUT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part74_c2_c3_spatial_localization_forensics"
)
REPORT = OUT / "reports"
REPORT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOCKED CONFIGURATION
# ============================================================================
VAL_N = 50
FULL = (64, 96, 96)
CROP = (32, 64, 64)
NUM_CLASSES = 6

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

CLASSES = {
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}

FOCUS_CLASSES = [2, 3]

# Thresholds selected from Part 73 for the class-balanced model.
# Additional thresholds are retained to characterize spatial behavior.
THRESHOLDS = {
    2: [0.10, 0.15, 0.20, 0.25, 0.30],
    3: [0.15, 0.20, 0.25, 0.30, 0.35],
}

# High-probability top-k fractions.
TOP_FRACTIONS = [0.001, 0.005, 0.01, 0.05]

# Connectivity for distance / component analysis.
STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


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


def dice_from_masks(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_n = int(pred.sum())
    truth_n = int(truth.sum())

    if pred_n + truth_n == 0:
        return 1.0

    tp = int(np.logical_and(pred, truth).sum())

    return safe_div(
        2 * tp,
        pred_n + truth_n,
    )


def binary_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
) -> dict:
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())

    pred_n = int(pred.sum())
    truth_n = int(truth.sum())

    return {
        "predicted_voxels": pred_n,
        "target_voxels": truth_n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": safe_div(
            2 * tp,
            pred_n + truth_n,
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
            truth_n,
        ),
    }


# ============================================================================
# R2 PSEUDO-MASK + CENTERED CROP
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

        result[
            expanded & (result == 0)
        ] = c

    return result


def centered_crop(
    mask: np.ndarray,
    center: np.ndarray,
    shape: tuple[int, int, int],
):
    starts = []

    for dim, size, c in zip(
        mask.shape,
        shape,
        center,
    ):
        start = int(
            round(
                float(c) - size / 2.0
            )
        )

        start = max(
            0,
            min(
                start,
                dim - size,
            ),
        )

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
    loaded = part11.load_tensor_case(
        row,
        part9,
    )

    if not isinstance(
        loaded,
        (tuple, list),
    ) or len(loaded) < 2:
        raise RuntimeError(
            "Unexpected Part 11 load_tensor_case() result."
        )

    image = torch.as_tensor(
        loaded[0]
    )

    mask = torch.as_tensor(
        loaded[1]
    )

    if (
        image.ndim == 4
        and image.shape[0] == 1
    ):
        image = image.squeeze(0)

    if (
        mask.ndim == 4
        and mask.shape[0] == 1
    ):
        mask = mask.squeeze(0)

    image_np = (
        image
        .detach()
        .cpu()
        .numpy()
    )

    mask_np = (
        mask
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    resized_mask = part11.resize_3d(
        mask_np,
        FULL,
        is_mask=True,
    )

    r2 = dilate_labels(
        resized_mask,
        2,
    )

    q = np.argwhere(
        r2 > 0
    )

    if len(q):
        center = q.mean(
            axis=0
        )
    else:
        center = np.array(
            [31.5, 47.5, 47.5]
        )

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
        starts,
    )


# ============================================================================
# SPATIAL FORENSICS
# ============================================================================
def distance_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
) -> dict:
    pred_n = int(pred.sum())
    truth_n = int(truth.sum())

    if pred_n == 0 or truth_n == 0:
        return {
            "pred_to_target_mean_distance": -1.0,
            "pred_to_target_median_distance": -1.0,
            "pred_to_target_p90_distance": -1.0,
            "target_to_pred_mean_distance": -1.0,
            "target_to_pred_median_distance": -1.0,
            "target_to_pred_p90_distance": -1.0,
            "symmetric_mean_distance": -1.0,
        }

    # Distance from every voxel to nearest target.
    target_distance = ndimage.distance_transform_edt(
        ~truth
    )

    # Distance from every voxel to nearest prediction.
    pred_distance = ndimage.distance_transform_edt(
        ~pred
    )

    p2t = target_distance[pred]
    t2p = pred_distance[truth]

    p2t_mean = float(
        np.mean(p2t)
    )
    t2p_mean = float(
        np.mean(t2p)
    )

    return {
        "pred_to_target_mean_distance": p2t_mean,
        "pred_to_target_median_distance": float(
            np.median(p2t)
        ),
        "pred_to_target_p90_distance": float(
            np.percentile(
                p2t,
                90,
            )
        ),
        "target_to_pred_mean_distance": t2p_mean,
        "target_to_pred_median_distance": float(
            np.median(t2p)
        ),
        "target_to_pred_p90_distance": float(
            np.percentile(
                t2p,
                90,
            )
        ),
        "symmetric_mean_distance": (
            p2t_mean + t2p_mean
        ) / 2.0,
    }


def component_metrics(
    binary: np.ndarray,
) -> dict:
    labeled, n = ndimage.label(
        binary,
        structure=STRUCTURE_6,
    )

    if n == 0:
        return {
            "components": 0,
            "largest_component": 0,
            "largest_component_fraction": 0.0,
        }

    sizes = np.bincount(
        labeled.ravel()
    )[1:]

    largest = int(
        sizes.max()
    )

    return {
        "components": int(n),
        "largest_component": largest,
        "largest_component_fraction": safe_div(
            largest,
            int(binary.sum()),
        ),
    }


def top_probability_metrics(
    probability: np.ndarray,
    truth: np.ndarray,
    fraction: float,
) -> dict:
    flat = probability.ravel()

    n = max(
        1,
        int(
            round(
                flat.size * fraction
            )
        ),
    )

    indices = np.argpartition(
        flat,
        -n,
    )[-n:]

    top = np.zeros_like(
        flat,
        dtype=bool,
    )

    top[indices] = True
    top = top.reshape(
        probability.shape
    )

    truth_n = int(
        truth.sum()
    )

    hit = int(
        np.logical_and(
            top,
            truth,
        ).sum()
    )

    return {
        "top_fraction": fraction,
        "top_voxels": n,
        "top_target_hits": hit,
        "top_target_recall": safe_div(
            hit,
            truth_n,
        ),
        "top_target_precision": safe_div(
            hit,
            n,
        ),
        "top_dice": dice_from_masks(
            top,
            truth,
        ),
    }


# ============================================================================
# MODEL
# ============================================================================
def evaluate():
    banner(
        "PART 74 C2/C3 SPATIAL LOCALIZATION FORENSICS"
    )

    checkpoint = torch.load(
        CKPT,
        map_location="cpu",
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model = part11.create_model(
        DEVICE
    ).to(DEVICE)

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    threshold_records = []
    top_records = []
    confusion_records = []
    probability_records = []

    with torch.no_grad():
        for case_no, (_, row) in enumerate(
            val_rows.iterrows(),
            start=1,
        ):
            image, target, starts = load_case(
                row,
                part9,
                part11,
            )

            x = (
                image
                .unsqueeze(0)
                .unsqueeze(0)
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            with torch.autocast(
                device_type="cuda",
                enabled=DEVICE.type == "cuda",
            ):
                logits = model(x)

            probs = torch.softmax(
                logits.float(),
                dim=1,
            )[0].cpu().numpy()

            target_np = target.numpy()

            c2_prob = probs[2]
            c3_prob = probs[3]

            c2_truth = target_np == 2
            c3_truth = target_np == 3

            # --------------------------------------------------------------
            # C2 / C3 independent threshold analysis
            # --------------------------------------------------------------
            for c in FOCUS_CLASSES:
                probability = probs[c]
                truth = target_np == c

                p_stats = {
                    "condition": "R2_FULL_CLASS_BALANCED_E3",
                    "case_no": case_no,
                    "class_id": c,
                    "class_name": CLASSES[c],
                    "target_voxels": int(
                        truth.sum()
                    ),
                    "mean_target_probability": (
                        float(
                            probability[truth].mean()
                        )
                        if truth.any()
                        else 0.0
                    ),
                    "max_target_probability": (
                        float(
                            probability[truth].max()
                        )
                        if truth.any()
                        else 0.0
                    ),
                    "mean_non_target_probability": (
                        float(
                            probability[~truth].mean()
                        )
                    ),
                    "max_non_target_probability": (
                        float(
                            probability[~truth].max()
                        )
                    ),
                }

                probability_records.append(
                    p_stats
                )

                for threshold in THRESHOLDS[c]:
                    pred = probability >= threshold

                    metrics = binary_metrics(
                        pred,
                        truth,
                    )

                    distances = distance_metrics(
                        pred,
                        truth,
                    )

                    components = component_metrics(
                        pred,
                    )

                    threshold_records.append({
                        "condition":
                            "R2_FULL_CLASS_BALANCED_E3",
                        "case_no": case_no,
                        "class_id": c,
                        "class_name": CLASSES[c],
                        "threshold": threshold,
                        **metrics,
                        **distances,
                        **components,
                    })

                # Top probability spatial concentration.
                for fraction in TOP_FRACTIONS:
                    top_records.append({
                        "condition":
                            "R2_FULL_CLASS_BALANCED_E3",
                        "case_no": case_no,
                        "class_id": c,
                        "class_name": CLASSES[c],
                        **top_probability_metrics(
                            probability,
                            truth,
                            fraction,
                        ),
                    })

            # --------------------------------------------------------------
            # C2 <-> C3 confusion
            # --------------------------------------------------------------
            # At every voxel, compare C2 and C3 probabilities.
            c2_wins = (
                c2_prob > c3_prob
            )

            c3_wins = (
                c3_prob > c2_prob
            )

            c2_target_c3_wins = int(
                np.logical_and(
                    c2_truth,
                    c3_wins,
                ).sum()
            )

            c3_target_c2_wins = int(
                np.logical_and(
                    c3_truth,
                    c2_wins,
                ).sum()
            )

            c2_target_n = int(
                c2_truth.sum()
            )

            c3_target_n = int(
                c3_truth.sum()
            )

            c2c3_overlap = int(
                np.logical_and(
                    c2_truth,
                    c3_truth,
                ).sum()
            )

            confusion_records.append({
                "condition":
                    "R2_FULL_CLASS_BALANCED_E3",
                "case_no": case_no,
                "c2_target_voxels": c2_target_n,
                "c3_target_voxels": c3_target_n,
                "c2_target_c3_probability_wins":
                    c2_target_c3_wins,
                "c2_target_c3_probability_win_rate":
                    safe_div(
                        c2_target_c3_wins,
                        c2_target_n,
                    ),
                "c3_target_c2_probability_wins":
                    c3_target_c2_wins,
                "c3_target_c2_probability_win_rate":
                    safe_div(
                        c3_target_c2_wins,
                        c3_target_n,
                    ),
                "c2_c3_target_overlap_voxels":
                    c2c3_overlap,
            })

    threshold_df = pd.DataFrame(
        threshold_records
    )

    top_df = pd.DataFrame(
        top_records
    )

    confusion_df = pd.DataFrame(
        confusion_records
    )

    probability_df = pd.DataFrame(
        probability_records
    )

    # ========================================================================
    # AGGREGATE THRESHOLD RESULTS
    # ========================================================================
    summary_rows = []

    for c in FOCUS_CLASSES:
        for threshold in THRESHOLDS[c]:
            g = threshold_df[
                (threshold_df.class_id == c)
                & (
                    threshold_df.threshold
                    == threshold
                )
            ]

            pred_n = int(
                g.predicted_voxels.sum()
            )
            target_n = int(
                g.target_voxels.sum()
            )
            tp = int(
                g.tp.sum()
            )
            fp = int(
                g.fp.sum()
            )
            fn = int(
                g.fn.sum()
            )

            summary_rows.append({
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
                "prediction_target_ratio":
                    safe_div(
                        pred_n,
                        target_n,
                    ),
                "mean_pred_to_target_distance":
                    float(
                        g[
                            g.pred_to_target_mean_distance
                            >= 0
                        ].pred_to_target_mean_distance.mean()
                    ),
                "mean_target_to_pred_distance":
                    float(
                        g[
                            g.target_to_pred_mean_distance
                            >= 0
                        ].target_to_pred_mean_distance.mean()
                    ),
                "mean_symmetric_distance":
                    float(
                        g[
                            g.symmetric_mean_distance
                            >= 0
                        ].symmetric_mean_distance.mean()
                    ),
            })

    threshold_summary = pd.DataFrame(
        summary_rows
    )

    # Best threshold by Dice.
    best_rows = []

    for c in FOCUS_CLASSES:
        g = threshold_summary[
            threshold_summary.class_id == c
        ].sort_values(
            ["dice", "precision", "recall"],
            ascending=False,
        )

        best_rows.append(
            g.iloc[0].to_dict()
        )

    best_df = pd.DataFrame(
        best_rows
    )

    # ========================================================================
    # TOP-K AGGREGATES
    # ========================================================================
    top_summary_rows = []

    for c in FOCUS_CLASSES:
        for fraction in TOP_FRACTIONS:
            g = top_df[
                (top_df.class_id == c)
                & (
                    top_df.top_fraction
                    == fraction
                )
            ]

            top_summary_rows.append({
                "class_id": c,
                "class_name": CLASSES[c],
                "top_fraction": fraction,
                "mean_target_recall":
                    float(
                        g.top_target_recall.mean()
                    ),
                "mean_target_precision":
                    float(
                        g.top_target_precision.mean()
                    ),
                "mean_top_dice":
                    float(
                        g.top_dice.mean()
                    ),
            })

    top_summary = pd.DataFrame(
        top_summary_rows
    )

    # ========================================================================
    # CROSS-CLASS PROBABILITY CONFUSION
    # ========================================================================
    confusion_summary = {
        "c2_target_mean_c3_win_rate": float(
            confusion_df[
                "c2_target_c3_probability_win_rate"
            ].mean()
        ),
        "c3_target_mean_c2_win_rate": float(
            confusion_df[
                "c3_target_c2_probability_win_rate"
            ].mean()
        ),
        "mean_c2_c3_target_overlap_voxels": float(
            confusion_df[
                "c2_c3_target_overlap_voxels"
            ].mean()
        ),
    }

    # ========================================================================
    # DIAGNOSTIC CLASSIFICATION
    # ========================================================================
    best_c2 = best_df[
        best_df.class_id == 2
    ].iloc[0]

    best_c3 = best_df[
        best_df.class_id == 3
    ].iloc[0]

    c2_top1 = top_summary[
        (top_summary.class_id == 2)
        & (
            top_summary.top_fraction
            == 0.01
        )
    ].iloc[0]

    c3_top1 = top_summary[
        (top_summary.class_id == 3)
        & (
            top_summary.top_fraction
            == 0.01
        )
    ].iloc[0]

    c2c3_confusion = (
        confusion_summary[
            "c2_target_mean_c3_win_rate"
        ]
        + confusion_summary[
            "c3_target_mean_c2_win_rate"
        ]
    ) / 2.0

    best_mean_dice = float(
        (
            best_c2["dice"]
            + best_c3["dice"]
        ) / 2.0
    )

    top1_recall = float(
        (
            c2_top1["mean_target_recall"]
            + c3_top1["mean_target_recall"]
        ) / 2.0
    )

    if (
        best_mean_dice < 0.05
        and c2c3_confusion >= 0.30
    ):
        diagnosis = (
            "C2_C3_CLASS_CONFUSION_IS_A_MAJOR_FAILURE_MODE"
        )
    elif (
        best_mean_dice < 0.05
        and top1_recall < 0.20
    ):
        diagnosis = (
            "C2_C3_HIGH_PROBABILITY_ACTIVATION_IS_SPATIALLY_MISPLACED_OR_VERY_WEAK"
        )
    elif (
        best_mean_dice < 0.10
        and top1_recall >= 0.20
    ):
        diagnosis = (
            "C2_C3_HAVE_SOME_LOCALIZED_HIGH_PROBABILITY_SIGNAL_BUT_ARE_TOO_DIFFUSE"
        )
    else:
        diagnosis = (
            "C2_C3_SHOW_PARTIAL_LOCALIZATION_WITH_SUBSTANTIAL_RESIDUAL_ERROR"
        )

    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    banner(
        "PART 74 THRESHOLD SPATIAL SUMMARY"
    )

    print(
        f"{'Class':<42}"
        f"{'BestT':>8}"
        f"{'Dice':>10}"
        f"{'Prec':>10}"
        f"{'Recall':>10}"
        f"{'Pred/Target':>13}"
        f"{'SymDist':>10}"
    )

    for _, r in best_df.iterrows():
        print(
            f"C{int(r['class_id'])} "
            f"{r['class_name']:<36}"
            f"{r['threshold']:>8.2f}"
            f"{r['dice']:>10.6f}"
            f"{r['precision']:>10.6f}"
            f"{r['recall']:>10.6f}"
            f"{r['prediction_target_ratio']:>13.3f}"
            f"{r['mean_symmetric_distance']:>10.3f}"
        )

    banner(
        "PART 74 TOP-1% HIGH-PROBABILITY LOCALIZATION"
    )

    for c in FOCUS_CLASSES:
        r = top_summary[
            (top_summary.class_id == c)
            & (
                top_summary.top_fraction
                == 0.01
            )
        ].iloc[0]

        print(
            f"C{c} {CLASSES[c]}:"
        )
        print(
            f"  Top 1% target recall : "
            f"{r['mean_target_recall']:.6f}"
        )
        print(
            f"  Top 1% precision     : "
            f"{r['mean_target_precision']:.6f}"
        )
        print(
            f"  Top 1% Dice          : "
            f"{r['mean_top_dice']:.6f}"
        )

    banner(
        "PART 74 C2 <-> C3 CONFUSION"
    )

    print(
        "C2 target voxels where C3 probability wins : "
        f"{confusion_summary['c2_target_mean_c3_win_rate']:.6f}"
    )

    print(
        "C3 target voxels where C2 probability wins : "
        f"{confusion_summary['c3_target_mean_c2_win_rate']:.6f}"
    )

    print(
        "Mean C2/C3 cross-confusion rate             : "
        f"{c2c3_confusion:.6f}"
    )

    banner(
        "PART 74 FINAL DIAGNOSIS"
    )

    print(
        f"C2/C3 mean best-threshold Dice : "
        f"{best_mean_dice:.6f}"
    )

    print(
        f"C2/C3 mean top-1% recall       : "
        f"{top1_recall:.6f}"
    )

    print(
        f"C2/C3 mean cross-confusion     : "
        f"{c2c3_confusion:.6f}"
    )

    print(
        f"Diagnosis                       : "
        f"{diagnosis}"
    )

    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================
    threshold_df.to_csv(
        REPORT
        / "part74_case_threshold_spatial_metrics.csv",
        index=False,
    )

    threshold_summary.to_csv(
        REPORT
        / "part74_threshold_spatial_summary.csv",
        index=False,
    )

    best_df.to_csv(
        REPORT
        / "part74_best_threshold_spatial_metrics.csv",
        index=False,
    )

    top_df.to_csv(
        REPORT
        / "part74_case_top_probability_metrics.csv",
        index=False,
    )

    top_summary.to_csv(
        REPORT
        / "part74_top_probability_summary.csv",
        index=False,
    )

    probability_df.to_csv(
        REPORT
        / "part74_probability_statistics.csv",
        index=False,
    )

    confusion_df.to_csv(
        REPORT
        / "part74_c2_c3_confusion_case_metrics.csv",
        index=False,
    )

    summary = {
        "part": 74,
        "validation_cases": VAL_N,
        "full_shape": FULL,
        "crop": CROP,
        "condition": "R2_FULL_CLASS_BALANCED_E3",
        "focus_classes": FOCUS_CLASSES,
        "best_c2_dice": float(
            best_c2["dice"]
        ),
        "best_c2_threshold": float(
            best_c2["threshold"]
        ),
        "best_c3_dice": float(
            best_c3["dice"]
        ),
        "best_c3_threshold": float(
            best_c3["threshold"]
        ),
        "c2_c3_mean_best_dice": best_mean_dice,
        "c2_c3_mean_top1_recall": top1_recall,
        "c2_target_c3_win_rate":
            confusion_summary[
                "c2_target_mean_c3_win_rate"
            ],
        "c3_target_c2_win_rate":
            confusion_summary[
                "c3_target_mean_c2_win_rate"
            ],
        "mean_c2_c3_cross_confusion":
            c2c3_confusion,
        "diagnosis": diagnosis,
    }

    with (
        REPORT / "part74_summary.json"
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
        REPORT / "part74_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "PART 74\n"
            "RSNA-ONLY C2/C3 SPATIAL LOCALIZATION + "
            "CLASS-CONFUSION FORENSICS\n\n"
        )

        f.write(
            f"C2 best Dice: "
            f"{best_c2['dice']:.6f}\n"
        )

        f.write(
            f"C2 best threshold: "
            f"{best_c2['threshold']:.2f}\n"
        )

        f.write(
            f"C3 best Dice: "
            f"{best_c3['dice']:.6f}\n"
        )

        f.write(
            f"C3 best threshold: "
            f"{best_c3['threshold']:.2f}\n"
        )

        f.write(
            f"C2/C3 mean best Dice: "
            f"{best_mean_dice:.6f}\n"
        )

        f.write(
            f"C2/C3 mean top-1% recall: "
            f"{top1_recall:.6f}\n"
        )

        f.write(
            f"C2 target -> C3 win rate: "
            f"{confusion_summary['c2_target_mean_c3_win_rate']:.6f}\n"
        )

        f.write(
            f"C3 target -> C2 win rate: "
            f"{confusion_summary['c3_target_mean_c2_win_rate']:.6f}\n"
        )

        f.write(
            f"Diagnosis: {diagnosis}\n\n"
        )

        f.write(
            best_df.to_string(
                index=False
            )
        )

    banner(
        "PART 74 OUTPUTS"
    )

    for p in sorted(
        REPORT.iterdir()
    ):
        print(p)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    banner("PART 74")

    required = [
        P11,
        P9,
        VAL_CSV,
        CKPT,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

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

    part11 = load_module(
        P11,
        "part74_part11",
    )

    part9 = load_module(
        P9,
        "part74_part9",
    )

    evaluate()

    reset_cuda()
