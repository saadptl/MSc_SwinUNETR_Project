"""
PART 60 — FINE THRESHOLD STABILITY FORENSICS

Diagnostic-only experiment using existing Part 58 R2 checkpoints.
No training, optimizer, or backward pass is performed.

Purpose:
Determine whether the useful low-probability foreground signal found
in Part 59 is stable over a threshold range, across checkpoints, and
across both Part 58 learning-rate conditions.

Conditions:
  A_constant_5e5
  B_step_1e4_to_5e5

Checkpoints:
  epochs 1-5

Thresholds:
  0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225,
  0.250, 0.275, 0.300, 0.350, 0.400, 0.500

Validation:
  100 cases
  R2 pseudo-mask
  foreground-centered crop

Scientific note:
Targets are pseudo-masks and therefore this experiment does not
establish medical ground-truth segmentation accuracy.
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


ROOT = Path(
    r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"
)

PART11 = (
    ROOT
    / "src"
    / "segmentation_rsna_part11_controlled_pilot_training.py"
)

PART9 = (
    ROOT
    / "src"
    / "segmentation_rsna_part9_3d_dataset_loader.py"
)

P15 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

VACSV = P15 / "part15_validation_cohort.csv"

P58 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part58_r2_lr_stability_confirmation"
)

OUT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part60_fine_threshold_stability_forensics"
)

REPORT = OUT / "reports"

VAL_N = 100
FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)

NUM_CLASSES = 6
RADIUS = 2

THRESHOLDS = [
    0.050,
    0.075,
    0.100,
    0.125,
    0.150,
    0.175,
    0.200,
    0.225,
    0.250,
    0.275,
    0.300,
    0.350,
    0.400,
    0.500,
]

EPOCHS = [1, 2, 3, 4, 5]

CONDITIONS = {
    "A_constant_5e5": P58 / "A_constant_5e5",
    "B_step_1e4_to_5e5": P58 / "B_step_1e4_to_5e5",
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def dilate_binary_6(binary, iterations):
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


def dilate_multiclass(mask):
    if RADIUS == 0:
        return mask.copy()

    regions = []

    for cid in range(1, NUM_CLASSES):
        region = mask == cid

        if region.any():
            regions.append(
                (
                    int(region.sum()),
                    cid,
                    dilate_binary_6(region, RADIUS),
                )
            )

    regions.sort(key=lambda x: (-x[0], x[1]))

    result = np.zeros_like(mask, dtype=np.int64)
    occupied = np.zeros_like(mask, dtype=bool)

    for _, cid, region in regions:
        assignable = region & ~occupied
        result[assignable] = cid
        occupied |= assignable

    return result


def foreground_center_crop(image, mask):
    coords = np.argwhere(mask > 0)

    if coords.size:
        center = np.round(coords.mean(axis=0)).astype(int)
    else:
        center = np.array([s // 2 for s in image.shape])

    starts = []

    for i in range(3):
        start = int(center[i]) - CROP_SHAPE[i] // 2
        start = max(
            0,
            min(start, FULL_SHAPE[i] - CROP_SHAPE[i]),
        )
        starts.append(start)

    z, y, x = starts
    dz, dy, dx = CROP_SHAPE

    return (
        image[z:z + dz, y:y + dy, x:x + dx].astype(np.float32),
        mask[z:z + dz, y:y + dy, x:x + dx].astype(np.int64),
    )


def load_case(part11, part9, row):
    loaded = part11.load_tensor_case(row, part9)

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
        raise RuntimeError(f"Unexpected image shape: {image.shape}")

    if tuple(mask.shape) != FULL_SHAPE:
        raise RuntimeError(f"Unexpected mask shape: {mask.shape}")

    mask = dilate_multiclass(mask.astype(np.int64))

    return foreground_center_crop(image, mask)


def preload(part11, part9, dataframe):
    cases = []

    print()
    print("=" * 82)
    print("PART 60 VALIDATION PRELOAD")
    print("=" * 82)

    for i, (_, row) in enumerate(dataframe.iterrows(), 1):
        image, mask = load_case(part11, part9, row)

        cases.append(
            {
                "image": image,
                "mask": mask,
                "index": i,
            }
        )

        if i in (1, 25, 50, 75, 100):
            print(
                f"VALIDATION {i:03d}/{len(dataframe)} "
                f"FG={int((mask > 0).sum())}"
            )

    return cases


def threshold_prediction(probs, threshold):
    """
    probs shape: [6,D,H,W].

    Select the strongest foreground class among classes 1..5.
    A voxel is foreground only when that strongest foreground
    probability is at least the supplied threshold.
    Otherwise it is assigned background.
    """
    foreground = probs[1:, ...]

    class_index = (
        np.argmax(foreground, axis=0).astype(np.int64) + 1
    )

    max_foreground_probability = np.max(
        foreground,
        axis=0,
    )

    class_index[max_foreground_probability < threshold] = 0

    return class_index


def binary_metrics(target, prediction):
    target_fg = target > 0
    prediction_fg = prediction > 0

    tp = int(np.logical_and(target_fg, prediction_fg).sum())
    fp = int(np.logical_and(~target_fg, prediction_fg).sum())
    fn = int(np.logical_and(target_fg, ~prediction_fg).sum())

    pred_count = int(prediction_fg.sum())
    target_count = int(target_fg.sum())

    denominator = pred_count + target_count

    dice = (
        2.0 * tp / denominator
        if denominator > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_fg": pred_count,
        "target_fg": target_count,
    }


def class_dice(target, prediction, cid):
    target_class = target == cid
    prediction_class = prediction == cid

    denominator = (
        int(target_class.sum())
        + int(prediction_class.sum())
    )

    if denominator == 0:
        return 1.0

    return float(
        2.0
        * np.logical_and(
            target_class,
            prediction_class,
        ).sum()
        / denominator
    )


def evaluate(model, cases, device, threshold):
    model.eval()

    sample_metrics = []

    class_dice_values = {
        cid: []
        for cid in range(1, NUM_CLASSES)
    }

    class_prediction_counts = {
        cid: []
        for cid in range(1, NUM_CLASSES)
    }

    foreground_probability = []
    background_probability = []

    for case in cases:
        x = (
            torch.from_numpy(case["image"])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            logits = model(x)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0].detach().cpu().numpy()

        target = case["mask"]

        prediction = threshold_prediction(
            probabilities,
            threshold,
        )

        sample_metrics.append(
            binary_metrics(
                target,
                prediction,
            )
        )

        for cid in range(1, NUM_CLASSES):
            class_dice_values[cid].append(
                class_dice(
                    target,
                    prediction,
                    cid,
                )
            )

            class_prediction_counts[cid].append(
                int((prediction == cid).sum())
            )

        foreground_probability.append(
            float(
                probabilities[1:, ...].sum(axis=0).mean()
            )
        )

        background_probability.append(
            float(
                probabilities[0, ...].mean()
            )
        )

        del x, logits, probabilities, prediction

    mean_pred_fg = float(
        np.mean(
            [m["pred_fg"] for m in sample_metrics]
        )
    )

    mean_target_fg = float(
        np.mean(
            [m["target_fg"] for m in sample_metrics]
        )
    )

    return {
        "foreground_dice": float(
            np.mean(
                [m["dice"] for m in sample_metrics]
            )
        ),
        "precision": float(
            np.mean(
                [m["precision"] for m in sample_metrics]
            )
        ),
        "recall": float(
            np.mean(
                [m["recall"] for m in sample_metrics]
            )
        ),
        "tp": int(sum(m["tp"] for m in sample_metrics)),
        "fp": int(sum(m["fp"] for m in sample_metrics)),
        "fn": int(sum(m["fn"] for m in sample_metrics)),
        "predicted_foreground_voxels": mean_pred_fg,
        "target_foreground_voxels": mean_target_fg,
        "prediction_target_ratio": float(
            mean_pred_fg / max(mean_target_fg, 1e-12)
        ),
        "empty_cases": int(
            sum(m["pred_fg"] == 0 for m in sample_metrics)
        ),
        "total_cases": len(sample_metrics),
        "foreground_probability": float(
            np.mean(foreground_probability)
        ),
        "background_probability": float(
            np.mean(background_probability)
        ),
        "probability_gap": float(
            np.mean(foreground_probability)
            - np.mean(background_probability)
        ),
        "class_dice": {
            str(cid): float(
                np.mean(class_dice_values[cid])
            )
            for cid in range(1, NUM_CLASSES)
        },
        "class_predicted_voxels": {
            str(cid): float(
                np.mean(class_prediction_counts[cid])
            )
            for cid in range(1, NUM_CLASSES)
        },
    }


def main():
    random.seed(160)
    np.random.seed(160)
    torch.manual_seed(160)

    print("=" * 82)
    print("PART 60 PATH VALIDATION")
    print("=" * 82)

    required = [
        ("Project root", ROOT),
        ("Part 11", PART11),
        ("Part 9", PART9),
        ("Part 15 validation cohort", VACSV),
        ("Part 58 output", P58),
    ]

    for name, path in required:
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{name:<40}: {status}")

        if not path.exists():
            raise FileNotFoundError(path)

    for condition, directory in CONDITIONS.items():
        status = "FOUND" if directory.exists() else "MISSING"

        print(
            f"{condition} directory".ljust(40)
            + f": {status}"
        )

        if not directory.exists():
            raise FileNotFoundError(directory)

        for epoch in EPOCHS:
            checkpoint = (
                directory
                / "checkpoints"
                / f"epoch_{epoch:02d}.pth"
            )

            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)

    print()
    print("=" * 82)
    print("PART 60 — FINE THRESHOLD STABILITY FORENSICS")
    print("=" * 82)
    print(f"Validation cohort : {VAL_N}")
    print("Pseudo-mask radius : R2")
    print(f"Full volume : {FULL_SHAPE}")
    print(f"Crop : {CROP_SHAPE}")
    print(f"Thresholds : {THRESHOLDS}")
    print("Checkpoints : epochs 1-5")
    print("Training performed : NO")
    print("Optimizer used : NO")
    print("Backward pass : NO")
    print("SPIDER : NO")
    print("Test set : NO")
    print("Part 15 modified : NO")

    part11 = load_module(
        PART11,
        "part11_part60_runtime",
    )

    part9 = load_module(
        PART9,
        "part9_part60_runtime",
    )

    dataframe = pd.read_csv(
        VACSV
    ).head(VAL_N)

    cases = preload(
        part11,
        part9,
        dataframe,
    )

    print()
    print("=" * 82)
    print("PART 60 SHAPE / LABEL SMOKE TEST")
    print("=" * 82)
    print(f"Image : {cases[0]['image'].shape}")
    print(f"Mask : {cases[0]['mask'].shape}")
    print(
        "Labels : "
        f"{sorted(np.unique(cases[0]['mask']).tolist())}"
    )

    assert cases[0]["image"].shape == CROP_SHAPE
    assert cases[0]["mask"].shape == CROP_SHAPE
    assert cases[0]["mask"].min() >= 0
    assert cases[0]["mask"].max() < NUM_CLASSES

    print("✓ Shape / label smoke test PASSED.")

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"PyTorch : {torch.__version__}")
    print(f"Device : {device}")

    all_records = []

    for condition, directory in CONDITIONS.items():
        print()
        print("=" * 82)
        print(f"PART 60 CONDITION: {condition}")
        print("=" * 82)

        for epoch in EPOCHS:
            checkpoint = (
                directory
                / "checkpoints"
                / f"epoch_{epoch:02d}.pth"
            )

            model = part11.create_model(device)

            state = torch.load(
                checkpoint,
                map_location=device,
            )

            model.load_state_dict(
                state["model_state_dict"],
                strict=True,
            )

            print()
            print(
                f"CHECKPOINT EPOCH {epoch:02d} | "
                f"SHA256={sha256(checkpoint)}"
            )

            for threshold in THRESHOLDS:
                result = evaluate(
                    model,
                    cases,
                    device,
                    threshold,
                )

                print(
                    f"  T={threshold:.3f} | "
                    f"Dice={result['foreground_dice']:.6f} | "
                    f"P={result['precision']:.6f} | "
                    f"R={result['recall']:.6f} | "
                    f"PredFG={result['predicted_foreground_voxels']:.1f} | "
                    f"Empty={result['empty_cases']}/{VAL_N}"
                )

                row = {
                    "condition": condition,
                    "epoch": epoch,
                    "checkpoint": str(checkpoint),
                    "threshold": threshold,
                    "foreground_dice": result[
                        "foreground_dice"
                    ],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "tp": result["tp"],
                    "fp": result["fp"],
                    "fn": result["fn"],
                    "predicted_foreground_voxels": result[
                        "predicted_foreground_voxels"
                    ],
                    "target_foreground_voxels": result[
                        "target_foreground_voxels"
                    ],
                    "prediction_target_ratio": result[
                        "prediction_target_ratio"
                    ],
                    "empty_cases": result["empty_cases"],
                    "total_cases": result["total_cases"],
                    "foreground_probability": result[
                        "foreground_probability"
                    ],
                    "background_probability": result[
                        "background_probability"
                    ],
                    "probability_gap": result[
                        "probability_gap"
                    ],
                }

                for cid in range(1, NUM_CLASSES):
                    row[f"class{cid}_dice"] = result[
                        "class_dice"
                    ][str(cid)]

                    row[f"class{cid}_pred_voxels"] = result[
                        "class_predicted_voxels"
                    ][str(cid)]

                all_records.append(row)

            del model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print("=" * 82)
    print("PART 60 BEST THRESHOLD BY CHECKPOINT")
    print("=" * 82)

    best_by_checkpoint = []

    for condition in CONDITIONS:
        for epoch in EPOCHS:
            rows = [
                r
                for r in all_records
                if (
                    r["condition"] == condition
                    and r["epoch"] == epoch
                )
            ]

            best = max(
                rows,
                key=lambda r: (
                    r["foreground_dice"],
                    r["recall"],
                    -r["empty_cases"],
                ),
            )

            best_by_checkpoint.append(best)

            print(
                f"{condition:<24} "
                f"E{epoch} | "
                f"T={best['threshold']:.3f} | "
                f"Dice={best['foreground_dice']:.6f} | "
                f"P={best['precision']:.6f} | "
                f"R={best['recall']:.6f} | "
                f"PredFG={best['predicted_foreground_voxels']:.1f} | "
                f"Empty={best['empty_cases']}/{VAL_N}"
            )

    print()
    print("=" * 82)
    print("PART 60 90%-OF-BEST THRESHOLD STABILITY WINDOWS")
    print("=" * 82)

    stability_windows = []

    for condition in CONDITIONS:
        for epoch in EPOCHS:
            rows = [
                r
                for r in all_records
                if (
                    r["condition"] == condition
                    and r["epoch"] == epoch
                )
            ]

            best_dice = max(
                r["foreground_dice"]
                for r in rows
            )

            stable = [
                r
                for r in rows
                if (
                    r["foreground_dice"]
                    >= 0.90 * best_dice
                )
            ]

            threshold_values = [
                r["threshold"]
                for r in stable
            ]

            if threshold_values:
                low = min(threshold_values)
                high = max(threshold_values)
                width = high - low
            else:
                low = None
                high = None
                width = 0.0

            print(
                f"{condition:<24} "
                f"E{epoch} | "
                f"BestDice={best_dice:.6f} | "
                f"Window="
                f"{low if low is not None else 'NA'}"
                f"-"
                f"{high if high is not None else 'NA'} | "
                f"Width={width:.3f}"
            )

            stability_windows.append(
                {
                    "condition": condition,
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "stable_threshold_low": low,
                    "stable_threshold_high": high,
                    "stable_threshold_width": width,
                    "stable_threshold_count": len(stable),
                }
            )

    print()
    print("=" * 82)
    print("PART 60 CROSS-CHECKPOINT THRESHOLD ROBUSTNESS")
    print("=" * 82)

    threshold_aggregate = []

    for threshold in THRESHOLDS:
        rows = [
            r
            for r in all_records
            if abs(r["threshold"] - threshold) < 1e-9
        ]

        threshold_aggregate.append(
            {
                "threshold": threshold,
                "mean_dice": float(
                    np.mean(
                        [
                            r["foreground_dice"]
                            for r in rows
                        ]
                    )
                ),
                "median_dice": float(
                    np.median(
                        [
                            r["foreground_dice"]
                            for r in rows
                        ]
                    )
                ),
                "min_dice": float(
                    np.min(
                        [
                            r["foreground_dice"]
                            for r in rows
                        ]
                    )
                ),
                "max_dice": float(
                    np.max(
                        [
                            r["foreground_dice"]
                            for r in rows
                        ]
                    )
                ),
                "std_dice": float(
                    np.std(
                        [
                            r["foreground_dice"]
                            for r in rows
                        ]
                    )
                ),
                "mean_precision": float(
                    np.mean(
                        [
                            r["precision"]
                            for r in rows
                        ]
                    )
                ),
                "mean_recall": float(
                    np.mean(
                        [
                            r["recall"]
                            for r in rows
                        ]
                    )
                ),
                "mean_predicted_fg": float(
                    np.mean(
                        [
                            r["predicted_foreground_voxels"]
                            for r in rows
                        ]
                    )
                ),
                "mean_empty_cases": float(
                    np.mean(
                        [
                            r["empty_cases"]
                            for r in rows
                        ]
                    )
                ),
            }
        )

    best_mean = max(
        threshold_aggregate,
        key=lambda r: r["mean_dice"],
    )

    print(
        f"Best mean threshold : "
        f"{best_mean['threshold']:.3f}"
    )
    print(
        f"Mean FG Dice : "
        f"{best_mean['mean_dice']:.6f}"
    )
    print(
        f"Median FG Dice : "
        f"{best_mean['median_dice']:.6f}"
    )
    print(
        f"Dice range : "
        f"{best_mean['min_dice']:.6f}"
        f" - "
        f"{best_mean['max_dice']:.6f}"
    )
    print(
        f"Mean precision : "
        f"{best_mean['mean_precision']:.6f}"
    )
    print(
        f"Mean recall : "
        f"{best_mean['mean_recall']:.6f}"
    )

    global_best = max(
        all_records,
        key=lambda r: (
            r["foreground_dice"],
            r["recall"],
            -r["empty_cases"],
        ),
    )

    t50_rows = [
        r
        for r in all_records
        if abs(r["threshold"] - 0.50) < 1e-9
    ]

    best_t50 = max(
        t50_rows,
        key=lambda r: r["foreground_dice"],
    )

    threshold_gain = (
        global_best["foreground_dice"]
        - best_t50["foreground_dice"]
    )

    # Conservative interpretation:
    # A broad mean-Dice advantage indicates a useful stable operating
    # region; a high peak with low mean stability indicates a checkpoint-
    # dependent threshold effect.
    if (
        best_mean["mean_dice"] > 0.03
        and best_mean["std_dice"] < 0.01
    ):
        diagnosis = (
            "BROAD_CROSS_CHECKPOINT_THRESHOLD_SIGNAL"
        )
    elif threshold_gain > 0.01:
        diagnosis = (
            "THRESHOLD_SIGNAL_EXISTS_BUT_IS_NOT_STABLE_ACROSS_CHECKPOINTS"
        )
    else:
        diagnosis = (
            "NO_ROBUST_THRESHOLD_ADVANTAGE"
        )

    print()
    print("=" * 82)
    print("PART 60 DIAGNOSTIC INTERPRETATION")
    print("=" * 82)
    print(
        f"Global best checkpoint : "
        f"{global_best['condition']} "
        f"E{global_best['epoch']}"
    )
    print(
        f"Global best threshold : "
        f"{global_best['threshold']:.3f}"
    )
    print(
        f"Global best FG Dice : "
        f"{global_best['foreground_dice']:.6f}"
    )
    print(
        f"Best T=0.500 FG Dice : "
        f"{best_t50['foreground_dice']:.6f}"
    )
    print(
        f"Threshold gain : "
        f"{threshold_gain:+.6f}"
    )
    print(
        f"Diagnosis : {diagnosis}"
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_csv = (
        REPORT
        / "part60_fine_threshold_trajectory.csv"
    )

    with open(
        trajectory_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                all_records[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(all_records)

    best_csv = (
        REPORT
        / "part60_best_threshold_by_checkpoint.csv"
    )

    with open(
        best_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                best_by_checkpoint[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            best_by_checkpoint
        )

    stability_csv = (
        REPORT
        / "part60_threshold_stability_windows.csv"
    )

    with open(
        stability_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                stability_windows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            stability_windows
        )

    aggregate_csv = (
        REPORT
        / "part60_cross_checkpoint_threshold_aggregate.csv"
    )

    with open(
        aggregate_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                threshold_aggregate[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            threshold_aggregate
        )

    summary_json = (
        REPORT
        / "part60_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "part": 60,
                "purpose": (
                    "Fine probability-threshold stability "
                    "forensics using existing Part 58 R2 checkpoints."
                ),
                "configuration": {
                    "validation_cases": VAL_N,
                    "pseudo_mask_radius": RADIUS,
                    "full_shape": FULL_SHAPE,
                    "crop_shape": CROP_SHAPE,
                    "thresholds": THRESHOLDS,
                    "epochs": EPOCHS,
                    "training_performed": False,
                    "optimizer_used": False,
                    "backward_pass": False,
                    "spider": False,
                    "test_set": False,
                    "part15_modified": False,
                    "source_experiment": str(P58),
                },
                "best_by_checkpoint": best_by_checkpoint,
                "stability_windows": stability_windows,
                "threshold_aggregate": threshold_aggregate,
                "global_best": global_best,
                "best_t50": best_t50,
                "threshold_gain": threshold_gain,
                "diagnosis": diagnosis,
                "scientific_note": (
                    "Targets are pseudo-masks. "
                    "This experiment does not establish medical "
                    "ground-truth segmentation accuracy."
                ),
            },
            f,
            indent=2,
        )

    print()
    print("=" * 82)
    print("PART 60 COMPLETE")
    print("=" * 82)
    print(
        f"Output directory : {OUT}"
    )
    print(
        f"Trajectory CSV : {trajectory_csv}"
    )
    print(
        f"Best-checkpoint CSV : {best_csv}"
    )
    print(
        f"Stability-window CSV : {stability_csv}"
    )
    print(
        f"Threshold aggregate CSV : {aggregate_csv}"
    )
    print(
        f"Summary JSON : {summary_json}"
    )


if __name__ == "__main__":
    main()
