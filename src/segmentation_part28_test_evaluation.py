"""
==============================================================================
PHASE 3 - PART 28
PART 27 HYBRID IMAGE-ONLY SWIN-UNETR TEST SET EVALUATION
==============================================================================

Purpose
-------
Evaluate the best Part 27 model on the official 71-case test set.

Part 27:
    Hybrid image-only anatomical localization

Baseline:
    Fixed-center preprocessing

Important
---------
Ground-truth masks are NOT used for localization.

The same image-derived center is used for:
    MRI image
    Ground-truth mask

This script does NOT:
    - train the model
    - modify model weights
    - modify Part 10/11 outputs
    - modify the original checkpoint
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from monai.networks.nets import SwinUNETR


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "segmentation_split"
    / "test"
)

IMAGE_DIR = TEST_DIR / "images"
MASK_DIR = TEST_DIR / "masks"

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "training_part27"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part28_test_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PATCH_SIZE = (
    96,
    96,
    96
)

NUM_CLASSES = 4
IN_CHANNELS = 1
FEATURE_SIZE = 24

CLASS_NAMES = {
    1: "Vertebrae",
    2: "Spinal Canal",
    3: "Intervertebral Disc",
}


# =============================================================================
# HEADER
# =============================================================================

print("=" * 78)
print("PHASE 3 - PART 28")
print("PART 27 HYBRID IMAGE-ONLY SWIN-UNETR TEST SET EVALUATION")
print("=" * 78)

print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nTEST DIRECTORY")
print(TEST_DIR)

print("\nTEST IMAGE DIRECTORY")
print(IMAGE_DIR)

print("\nTEST MASK DIRECTORY")
print(MASK_DIR)

print("\nPART 27 CHECKPOINT")
print(CHECKPOINT)

print("\nOUTPUT DIRECTORY")
print(OUTPUT_DIR)


# =============================================================================
# DEVICE
# =============================================================================

print("\n" + "=" * 78)
print("DEVICE")
print("=" * 78)

if torch.cuda.is_available():

    DEVICE = torch.device("cuda:0")

    print("Device:", DEVICE)
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

else:

    DEVICE = torch.device("cpu")

    print("Device:", DEVICE)


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_volume(path):
    """
    Load MHA volume using SimpleITK.
    """

    import SimpleITK as sitk

    image = sitk.ReadImage(
        str(path)
    )

    array = sitk.GetArrayFromImage(
        image
    )

    return array.astype(
        np.float32
    )


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_image(image):
    """
    Same percentile-style normalization used by Part 27.
    """

    image = image.astype(
        np.float32
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    nonzero = image[image > 0]

    if nonzero.size == 0:
        return np.zeros_like(
            image,
            dtype=np.float32
        )

    low = np.percentile(
        nonzero,
        1
    )

    high = np.percentile(
        nonzero,
        99
    )

    if high <= low:
        high = low + 1e-6

    image = np.clip(
        image,
        low,
        high
    )

    image = (
        image - low
    ) / (
        high - low
    )

    image[image < 0] = 0
    image[image > 1] = 1

    return image.astype(
        np.float32
    )


# =============================================================================
# PART 27 HYBRID IMAGE-ONLY LOCALIZATION
# =============================================================================

def compute_hybrid_image_center(image):
    """
    Reproduce Part 27 image-only hybrid localization.

    Ground truth is NOT used.
    """

    image = image.astype(
        np.float32
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    nonzero = image > 0

    if nonzero.any():

        coords = np.argwhere(
            nonzero
        )

        nonzero_center = (
            coords.mean(axis=0)
        )

    else:

        nonzero_center = np.array(
            [
                size / 2.0
                for size in image.shape
            ],
            dtype=np.float32
        )

    positive_values = image[
        image > 0
    ]

    if positive_values.size > 0:

        low = np.percentile(
            positive_values,
            20
        )

        high = np.percentile(
            positive_values,
            80
        )

        robust_mask = (
            (image >= low)
            &
            (image <= high)
            &
            (image > 0)
        )

        if robust_mask.any():

            robust_coords = np.argwhere(
                robust_mask
            )

            robust_center = (
                robust_coords.mean(axis=0)
            )

        else:

            robust_center = nonzero_center

    else:

        robust_center = nonzero_center

    positive = np.maximum(
        image,
        0
    )

    total_intensity = positive.sum()

    if total_intensity > 0:

        grid = np.indices(
            image.shape,
            dtype=np.float32
        )

        weighted_center = np.array(
            [
                (
                    grid[axis]
                    * positive
                ).sum()
                / total_intensity

                for axis in range(
                    image.ndim
                )
            ],
            dtype=np.float32
        )

    else:

        weighted_center = nonzero_center

    hybrid_center = (
        0.50 * nonzero_center
        +
        0.30 * robust_center
        +
        0.20 * weighted_center
    )

    hybrid_center = np.asarray(
        hybrid_center,
        dtype=np.float32
    )

    for axis, size in enumerate(
        image.shape
    ):

        hybrid_center[axis] = np.clip(
            hybrid_center[axis],
            0,
            size - 1
        )

    return tuple(
        int(round(value))
        for value in hybrid_center
    )


# =============================================================================
# CROP / PAD
# =============================================================================

def crop_or_pad(
    array,
    target_shape,
    center,
    pad_value=0
):
    """
    Crop/pad around image-derived center.
    """

    result = np.full(
        target_shape,
        pad_value,
        dtype=array.dtype
    )

    source_shape = array.shape

    source_slices = []
    target_slices = []

    for axis, (
        source_size,
        target_size
    ) in enumerate(
        zip(
            source_shape,
            target_shape
        )
    ):

        center_coordinate = center[axis]

        half_target = target_size // 2

        source_start = (
            center_coordinate
            -
            half_target
        )

        source_end = (
            source_start
            +
            target_size
        )

        target_start = 0
        target_end = target_size

        if source_start < 0:

            target_start = -source_start
            source_start = 0

        if source_end > source_size:

            target_end -= (
                source_end
                -
                source_size
            )

            source_end = source_size

        source_slices.append(
            slice(
                source_start,
                source_end
            )
        )

        target_slices.append(
            slice(
                target_start,
                target_end
            )
        )

    result[
        tuple(target_slices)
    ] = array[
        tuple(source_slices)
    ]

    return result


# =============================================================================
# MODEL
# =============================================================================

print("\n" + "=" * 78)
print("CREATING SWIN-UNETR")
print("=" * 78)

model = SwinUNETR(
    in_channels=IN_CHANNELS,
    out_channels=NUM_CLASSES,
    feature_size=FEATURE_SIZE,
    use_checkpoint=False,
)

model = model.to(
    DEVICE
)

print("✓ Swin-UNETR created.")


# =============================================================================
# CHECKPOINT
# =============================================================================

print("\n" + "=" * 78)
print("LOADING PART 27 BEST CHECKPOINT")
print("=" * 78)

if not CHECKPOINT.exists():

    raise FileNotFoundError(
        f"Part 27 checkpoint not found:\n{CHECKPOINT}"
    )

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

print(
    "Checkpoint type:",
    type(checkpoint)
)

if isinstance(
    checkpoint,
    dict
):

    print(
        "Checkpoint keys:"
    )

    for key in checkpoint.keys():

        print(
            f"  - {key}"
        )

    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )

    if "best_val_dice" in checkpoint:

        print(
            "Part 27 best validation Dice:",
            checkpoint["best_val_dice"]
        )

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model = model.to(
    DEVICE
)

model.eval()

print("✓ Model weights loaded.")
print("✓ Model moved to:", DEVICE)
print("✓ Evaluation mode enabled.")


# =============================================================================
# FILE PAIRS
# =============================================================================

print("\n" + "=" * 78)
print("TEST DATASET")
print("=" * 78)

image_files = sorted(
    IMAGE_DIR.glob("*.mha")
)

mask_files = sorted(
    MASK_DIR.glob("*.mha")
)

image_map = {
    p.stem: p
    for p in image_files
}

mask_map = {
    p.stem: p
    for p in mask_files
}

common_names = sorted(
    set(image_map)
    &
    set(mask_map)
)

missing_masks = sorted(
    set(image_map)
    -
    set(mask_map)
)

orphan_masks = sorted(
    set(mask_map)
    -
    set(image_map)
)

print(
    "Test MRI files :",
    len(image_files)
)

print(
    "Test mask files:",
    len(mask_files)
)

print(
    "Matching pairs :",
    len(common_names)
)

print(
    "Missing masks  :",
    len(missing_masks)
)

print(
    "Orphan masks   :",
    len(orphan_masks)
)

if len(common_names) == 0:

    raise RuntimeError(
        "No matching test image/mask pairs found."
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_binary_metrics(
    prediction,
    target
):

    prediction = prediction.astype(
        bool
    )

    target = target.astype(
        bool
    )

    tp = np.logical_and(
        prediction,
        target
    ).sum()

    fp = np.logical_and(
        prediction,
        ~target
    ).sum()

    fn = np.logical_and(
        ~prediction,
        target
    ).sum()

    denominator = (
        2 * tp
        +
        fp
        +
        fn
    )

    if denominator == 0:

        dice = 1.0

    else:

        dice = (
            2 * tp
        ) / denominator

    union = (
        tp
        +
        fp
        +
        fn
    )

    if union == 0:

        iou = 1.0

    else:

        iou = tp / union

    precision_denominator = (
        tp + fp
    )

    if precision_denominator == 0:

        precision = 0.0

    else:

        precision = (
            tp
            /
            precision_denominator
        )

    recall_denominator = (
        tp + fn
    )

    if recall_denominator == 0:

        recall = 0.0

    else:

        recall = (
            tp
            /
            recall_denominator
        )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


# =============================================================================
# TEST EVALUATION
# =============================================================================

print("\n" + "=" * 78)
print("STARTING PART 27 TEST EVALUATION")
print("=" * 78)

print(
    "Test cases:",
    len(common_names)
)

print(
    "No optimizer or gradient computation will be performed."
)

results = []

start_time = time.time()

with torch.no_grad():

    for index, name in enumerate(
        common_names,
        start=1
    ):

        image_path = image_map[name]
        mask_path = mask_map[name]

        image = load_volume(
            image_path
        )

        mask = load_volume(
            mask_path
        )

        image = normalize_image(
            image
        )

        # ---------------------------------------------------------
        # CRITICAL:
        # image-only center
        # ---------------------------------------------------------

        center = compute_hybrid_image_center(
            image
        )

        image_crop = crop_or_pad(
            image,
            PATCH_SIZE,
            center,
            pad_value=0
        )

        mask_crop = crop_or_pad(
            mask,
            PATCH_SIZE,
            center,
            pad_value=0
        )

        image_tensor = torch.from_numpy(
            image_crop
        ).float()

        image_tensor = image_tensor.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        image_tensor = image_tensor.to(
            DEVICE
        )

        output = model(
            image_tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        )

        prediction = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
        )

        mask_crop = np.rint(
            mask_crop
        ).astype(
            np.int64
        )

        foreground_prediction = (
            prediction > 0
        )

        foreground_target = (
            mask_crop > 0
        )

        overall_metrics = calculate_binary_metrics(
            foreground_prediction,
            foreground_target
        )

        row = {
            "file": name,
            "mean_foreground_dice":
                overall_metrics["dice"],
            "mean_foreground_iou":
                overall_metrics["iou"],
        }

        per_class_dice = []

        for class_id, class_name in CLASS_NAMES.items():

            class_prediction = (
                prediction == class_id
            )

            class_target = (
                mask_crop == class_id
            )

            metrics = calculate_binary_metrics(
                class_prediction,
                class_target
            )

            row[
                f"{class_name}_dice"
            ] = metrics["dice"]

            row[
                f"{class_name}_iou"
            ] = metrics["iou"]

            row[
                f"{class_name}_precision"
            ] = metrics["precision"]

            row[
                f"{class_name}_recall"
            ] = metrics["recall"]

            per_class_dice.append(
                metrics["dice"]
            )

        row[
            "prediction_labels"
        ] = ",".join(
            map(
                str,
                sorted(
                    np.unique(
                        prediction
                    )
                )
            )
        )

        row[
            "target_labels"
        ] = ",".join(
            map(
                str,
                sorted(
                    np.unique(
                        mask_crop
                    )
                )
            )
        )

        results.append(
            row
        )

        if (
            index == 1
            or index % 10 == 0
            or index == len(common_names)
        ):

            print(
                f"Evaluated {index:3d} / {len(common_names)}"
            )


# =============================================================================
# RESULTS DATAFRAME
# =============================================================================

results_df = pd.DataFrame(
    results
)

elapsed_minutes = (
    time.time()
    -
    start_time
) / 60.0


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 78)
print("PART 27 TEST EVALUATION COMPLETE")
print("=" * 78)

print(
    "Successful cases:",
    len(results_df)
)

print(
    "Evaluation time:",
    f"{elapsed_minutes:.2f} minutes"
)


overall_dice = (
    results_df[
        "mean_foreground_dice"
    ]
    .mean()
)

overall_dice_std = (
    results_df[
        "mean_foreground_dice"
    ]
    .std()
)

overall_dice_median = (
    results_df[
        "mean_foreground_dice"
    ]
    .median()
)

overall_iou = (
    results_df[
        "mean_foreground_iou"
    ]
    .mean()
)

overall_iou_std = (
    results_df[
        "mean_foreground_iou"
    ]
    .std()
)


# =============================================================================
# OVERALL PERFORMANCE
# =============================================================================

print("\n" + "=" * 78)
print("OVERALL PART 27 TEST PERFORMANCE")
print("=" * 78)

print(
    "Mean Foreground Dice   :",
    f"{overall_dice:.6f}"
)

print(
    "Std Foreground Dice    :",
    f"{overall_dice_std:.6f}"
)

print(
    "Median Foreground Dice :",
    f"{overall_dice_median:.6f}"
)

print(
    "Mean Foreground IoU    :",
    f"{overall_iou:.6f}"
)

print(
    "Std Foreground IoU     :",
    f"{overall_iou_std:.6f}"
)


# =============================================================================
# PER-CLASS PERFORMANCE
# =============================================================================

print("\n" + "=" * 78)
print("PER-CLASS PART 27 TEST PERFORMANCE")
print("=" * 78)

class_summary = {}

for class_id, class_name in CLASS_NAMES.items():

    dice_column = (
        f"{class_name}_dice"
    )

    iou_column = (
        f"{class_name}_iou"
    )

    precision_column = (
        f"{class_name}_precision"
    )

    recall_column = (
        f"{class_name}_recall"
    )

    dice_mean = (
        results_df[
            dice_column
        ].mean()
    )

    iou_mean = (
        results_df[
            iou_column
        ].mean()
    )

    precision_mean = (
        results_df[
            precision_column
        ].mean()
    )

    recall_mean = (
        results_df[
            recall_column
        ].mean()
    )

    class_summary[
        class_name
    ] = {
        "dice": float(dice_mean),
        "iou": float(iou_mean),
        "precision": float(precision_mean),
        "recall": float(recall_mean),
    }

    print(
        f"{class_name:20s}"
        f"Dice: {dice_mean:.6f}  "
        f"IoU: {iou_mean:.6f}  "
        f"Precision: {precision_mean:.6f}  "
        f"Recall: {recall_mean:.6f}"
    )


# =============================================================================
# BEST / WORST CASES
# =============================================================================

print("\n" + "=" * 78)
print("BEST TEST CASES")
print("=" * 78)

best_cases = (
    results_df
    .sort_values(
        "mean_foreground_dice",
        ascending=False
    )
    .head(5)
)

for _, row in best_cases.iterrows():

    print(
        f"{row['file']:25s}"
        f"{row['mean_foreground_dice']:.6f}"
    )


print("\n" + "=" * 78)
print("WORST TEST CASES")
print("=" * 78)

worst_cases = (
    results_df
    .sort_values(
        "mean_foreground_dice",
        ascending=True
    )
    .head(5)
)

for _, row in worst_cases.iterrows():

    print(
        f"{row['file']:25s}"
        f"{row['mean_foreground_dice']:.6f}"
    )


# =============================================================================
# BASELINE COMPARISON
# =============================================================================

BASELINE_TEST_DICE = 0.822515
BASELINE_TEST_IOU = 0.731083

dice_difference = (
    overall_dice
    -
    BASELINE_TEST_DICE
)

iou_difference = (
    overall_iou
    -
    BASELINE_TEST_IOU
)

relative_dice_improvement = (
    dice_difference
    /
    BASELINE_TEST_DICE
) * 100.0


print("\n" + "=" * 78)
print("BASELINE VS PART 27")
print("=" * 78)

print(
    "Original test Dice:",
    f"{BASELINE_TEST_DICE:.6f}"
)

print(
    "Part 27 test Dice:",
    f"{overall_dice:.6f}"
)

print(
    "Dice difference:",
    f"{dice_difference:+.6f}"
)

print(
    "Relative Dice change:",
    f"{relative_dice_improvement:+.3f}%"
)

print(
    "Original test IoU:",
    f"{BASELINE_TEST_IOU:.6f}"
)

print(
    "Part 27 test IoU:",
    f"{overall_iou:.6f}"
)

print(
    "IoU difference:",
    f"{iou_difference:+.6f}"
)


# =============================================================================
# SAVE CASE RESULTS
# =============================================================================

case_results_path = (
    OUTPUT_DIR
    /
    "part27_test_case_results.csv"
)

results_df.to_csv(
    case_results_path,
    index=False
)

print(
    "\nSaved:",
    case_results_path
)


# =============================================================================
# SAVE CLASS SUMMARY
# =============================================================================

class_summary_df = pd.DataFrame(
    [
        {
            "class": class_name,
            **metrics
        }
        for class_name, metrics
        in class_summary.items()
    ]
)

class_summary_path = (
    OUTPUT_DIR
    /
    "part27_test_class_summary.csv"
)

class_summary_df.to_csv(
    class_summary_path,
    index=False
)

print(
    "Saved:",
    class_summary_path
)


# =============================================================================
# SAVE JSON SUMMARY
# =============================================================================

summary = {
    "phase": "Phase 3 - Part 28",
    "experiment": "Part 27 Hybrid Image-Only Crop",
    "checkpoint": str(CHECKPOINT),
    "checkpoint_epoch": (
        checkpoint.get("epoch")
        if isinstance(checkpoint, dict)
        else None
    ),
    "best_validation_dice": (
        float(
            checkpoint["best_val_dice"]
        )
        if isinstance(
            checkpoint,
            dict
        )
        and "best_val_dice"
        in checkpoint
        else None
    ),
    "test_cases": int(
        len(results_df)
    ),
    "mean_foreground_dice":
        float(overall_dice),
    "std_foreground_dice":
        float(overall_dice_std),
    "median_foreground_dice":
        float(overall_dice_median),
    "mean_foreground_iou":
        float(overall_iou),
    "std_foreground_iou":
        float(overall_iou_std),
    "baseline_test_dice":
        BASELINE_TEST_DICE,
    "dice_difference":
        float(dice_difference),
    "relative_dice_change_percent":
        float(relative_dice_improvement),
    "baseline_test_iou":
        BASELINE_TEST_IOU,
    "iou_difference":
        float(iou_difference),
    "class_summary":
        class_summary,
    "crop_strategy":
        "hybrid_image_center",
    "ground_truth_used_for_localization":
        False,
    "training_performed":
        False,
    "model_weights_modified":
        False,
}


json_path = (
    OUTPUT_DIR
    /
    "part28_test_evaluation_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print(
    "Saved:",
    json_path
)


# =============================================================================
# TEXT REPORT
# =============================================================================

report_path = (
    OUTPUT_DIR
    /
    "phase3_part28_test_evaluation_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PHASE 3 - PART 28\n"
    )

    f.write(
        "PART 27 HYBRID IMAGE-ONLY "
        "SWIN-UNETR TEST EVALUATION\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Test cases: {len(results_df)}\n"
    )

    f.write(
        f"Part 27 checkpoint epoch: "
        f"{summary['checkpoint_epoch']}\n"
    )

    f.write(
        f"Best validation Dice: "
        f"{summary['best_validation_dice']}\n\n"
    )

    f.write(
        "OVERALL TEST PERFORMANCE\n"
    )

    f.write(
        f"Mean Foreground Dice: "
        f"{overall_dice:.6f}\n"
    )

    f.write(
        f"Median Foreground Dice: "
        f"{overall_dice_median:.6f}\n"
    )

    f.write(
        f"Mean Foreground IoU: "
        f"{overall_iou:.6f}\n\n"
    )

    f.write(
        "BASELINE COMPARISON\n"
    )

    f.write(
        f"Baseline Dice: "
        f"{BASELINE_TEST_DICE:.6f}\n"
    )

    f.write(
        f"Part 27 Dice: "
        f"{overall_dice:.6f}\n"
    )

    f.write(
        f"Dice difference: "
        f"{dice_difference:+.6f}\n"
    )

    f.write(
        f"Relative Dice change: "
        f"{relative_dice_improvement:+.3f}%\n\n"
    )

    f.write(
        "PER-CLASS PERFORMANCE\n"
    )

    for class_name, metrics in class_summary.items():

        f.write(
            f"{class_name}: "
            f"Dice={metrics['dice']:.6f}, "
            f"IoU={metrics['iou']:.6f}, "
            f"Precision={metrics['precision']:.6f}, "
            f"Recall={metrics['recall']:.6f}\n"
        )

    f.write(
        "\nCROP STRATEGY\n"
    )

    f.write(
        "hybrid_image_center\n"
    )

    f.write(
        "Ground-truth used for localization: NO\n"
    )

    f.write(
        "Training performed: NO\n"
    )

    f.write(
        "Model weights modified: NO\n"
    )


print(
    "Saved:",
    report_path
)


# =============================================================================
# FINAL
# =============================================================================

print("\n" + "=" * 78)
print("PHASE 3 - PART 28 COMPLETE")
print("=" * 78)

print(
    "Test cases:",
    len(results_df)
)

print(
    "Part 27 Test Mean Dice:",
    f"{overall_dice:.6f}"
)

print(
    "Original Test Mean Dice:",
    f"{BASELINE_TEST_DICE:.6f}"
)

print(
    "Difference:",
    f"{dice_difference:+.6f}"
)

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print("=" * 78)