"""
PHASE 3 - PART 14
SPINAL CANAL FAILURE-CASE DEEP ANALYSIS

Uses the exact preprocessing pipeline from Phase 3 - Part 11.

Classes:
    0 - Background
    1 - Vertebrae
    2 - Spinal Canal
    3 - Intervertebral Disc
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import SwinUNETR


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "segmentation_split"
    / "test"
)

TEST_IMAGES_DIR = TEST_DIR / "images"
TEST_MASKS_DIR = TEST_DIR / "masks"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "training"
    / "checkpoints"
    / "best_model.pth"
)

PART11_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "test_evaluation"
    / "test_case_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "spinal_canal_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = (96, 96, 96)

IN_CHANNELS = 1
OUT_CHANNELS = 4
FEATURE_SIZE = 24

SPINAL_CANAL_CLASS = 2

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("PHASE 3 - PART 14")
print("SPINAL CANAL FAILURE-CASE DEEP ANALYSIS")
print("=" * 78)

print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nTEST DIRECTORY")
print(TEST_DIR)

print("\nCHECKPOINT")
print(CHECKPOINT_PATH)

print("\nOUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================
# VALIDATE PATHS
# ============================================================

if not TEST_IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Test image directory not found:\n{TEST_IMAGES_DIR}"
    )

if not TEST_MASKS_DIR.exists():
    raise FileNotFoundError(
        f"Test mask directory not found:\n{TEST_MASKS_DIR}"
    )

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Best checkpoint not found:\n{CHECKPOINT_PATH}"
    )

if not PART11_RESULTS.exists():
    raise FileNotFoundError(
        f"Part 11 results not found:\n{PART11_RESULTS}"
    )


# ============================================================
# DEVICE
# ============================================================

print("\n" + "=" * 78)
print("DEVICE")
print("=" * 78)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
    )
else:
    print("CUDA is not available. Analysis will use CPU.")


# ============================================================
# LOAD PART 11 RESULTS
# ============================================================

print("\n" + "=" * 78)
print("LOADING PART 11 RESULTS")
print("=" * 78)

part11_df = pd.read_csv(PART11_RESULTS)

print(f"Part 11 cases: {len(part11_df)}")
print(f"Columns: {list(part11_df.columns)}")


# ============================================================
# COLLECT TEST FILES
# ============================================================

image_files = sorted(TEST_IMAGES_DIR.glob("*.mha"))
mask_files = sorted(TEST_MASKS_DIR.glob("*.mha"))

image_map = {p.stem: p for p in image_files}
mask_map = {p.stem: p for p in mask_files}

common_names = sorted(
    set(image_map) & set(mask_map)
)

missing_masks = sorted(
    set(image_map) - set(mask_map)
)

orphan_masks = sorted(
    set(mask_map) - set(image_map)
)

print("\n" + "=" * 78)
print("TEST DATA")
print("=" * 78)

print(f"MRI files : {len(image_files)}")
print(f"Mask files: {len(mask_files)}")
print(f"Matching cases: {len(common_names)}")
print(f"Missing masks: {len(missing_masks)}")
print(f"Orphan masks : {len(orphan_masks)}")

if not common_names:
    raise RuntimeError("No valid MRI/mask pairs found.")

if missing_masks or orphan_masks:
    raise RuntimeError(
        "Test dataset pairing is incomplete."
    )


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 78)
print("LOADING SWIN-UNETR")
print("=" * 78)

model = SwinUNETR(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    feature_size=FEATURE_SIZE,
    use_checkpoint=False,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

print(f"Checkpoint type: {type(checkpoint)}")

if isinstance(checkpoint, dict):

    if "epoch" in checkpoint:
        print(
            f"Checkpoint epoch: {checkpoint['epoch']}"
        )

    if "best_val_dice" in checkpoint:
        print(
            f"Best validation Dice: "
            f"{checkpoint['best_val_dice']:.6f}"
        )

    model_state = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

else:

    model_state = checkpoint


model.load_state_dict(model_state)

model = model.to(DEVICE)

model.eval()

print("✓ Swin-UNETR loaded.")
print("✓ Evaluation mode enabled.")


# ============================================================
# PART 11 PREPROCESSING
# ============================================================

def load_volume(path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    return image, array


def center_crop_or_pad(
    array,
    target_shape=PATCH_SIZE,
    pad_value=0,
):
    """
    EXACT SAME CENTER CROP/PAD STRATEGY AS PART 11.

    Input:
        [Z, Y, X]

    Output:
        [96, 96, 96]
    """

    result = np.full(
        target_shape,
        pad_value,
        dtype=array.dtype,
    )

    source_slices = []
    target_slices = []

    for src_size, tgt_size in zip(
        array.shape,
        target_shape,
    ):

        if src_size >= tgt_size:

            src_start = (
                src_size - tgt_size
            ) // 2

            src_end = (
                src_start + tgt_size
            )

            source_slices.append(
                slice(
                    src_start,
                    src_end,
                )
            )

            target_slices.append(
                slice(
                    0,
                    tgt_size,
                )
            )

        else:

            src_start = 0
            src_end = src_size

            target_start = (
                tgt_size - src_size
            ) // 2

            target_end = (
                target_start + src_size
            )

            source_slices.append(
                slice(
                    src_start,
                    src_end,
                )
            )

            target_slices.append(
                slice(
                    target_start,
                    target_end,
                )
            )

    result[
        tuple(target_slices)
    ] = array[
        tuple(source_slices)
    ]

    return result


def normalize_image(array):
    """
    EXACT SAME NORMALIZATION AS PART 11.

    Uses the 1st and 99th percentile of non-zero voxels.
    """

    array = array.astype(
        np.float32
    )

    nonzero = array[
        array != 0
    ]

    if nonzero.size == 0:

        return np.zeros_like(
            array,
            dtype=np.float32,
        )

    low = np.percentile(
        nonzero,
        1.0,
    )

    high = np.percentile(
        nonzero,
        99.0,
    )

    if high <= low:

        return np.zeros_like(
            array,
            dtype=np.float32,
        )

    array = np.clip(
        array,
        low,
        high,
    )

    array = (
        array - low
    ) / (
        high - low
    )

    array[array < 0] = 0
    array[array > 1] = 1

    return array.astype(
        np.float32
    )


def prepare_image(array):
    """
    EXACT SAME IMAGE PREPARATION AS PART 11.

    Output:
        (1, 1, 96, 96, 96)
    """

    array = normalize_image(
        array
    )

    array = center_crop_or_pad(
        array,
        PATCH_SIZE,
        pad_value=0,
    )

    tensor = torch.from_numpy(
        array
    ).float()

    tensor = tensor.unsqueeze(0)

    tensor = tensor.unsqueeze(0)

    return tensor


def prepare_mask(array):
    """
    EXACT SAME MASK PREPARATION AS PART 11.
    """

    array = center_crop_or_pad(
        array.astype(np.int64),
        PATCH_SIZE,
        pad_value=0,
    )

    tensor = torch.from_numpy(
        array
    ).long()

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target,
    class_id=SPINAL_CANAL_CLASS,
):

    pred = (
        prediction == class_id
    )

    true = (
        target == class_id
    )

    tp = int(
        np.logical_and(
            pred,
            true,
        ).sum()
    )

    fp = int(
        np.logical_and(
            pred,
            np.logical_not(true),
        ).sum()
    )

    fn = int(
        np.logical_and(
            np.logical_not(pred),
            true,
        ).sum()
    )

    tn = int(
        np.logical_and(
            np.logical_not(pred),
            np.logical_not(true),
        ).sum()
    )

    pred_voxels = int(
        pred.sum()
    )

    target_voxels = int(
        true.sum()
    )

    dice_denominator = (
        2 * tp + fp + fn
    )

    if dice_denominator == 0:

        dice = 1.0

    else:

        dice = (
            2.0 * tp
        ) / dice_denominator

    iou_denominator = (
        tp + fp + fn
    )

    if iou_denominator == 0:

        iou = 1.0

    else:

        iou = (
            tp
            / iou_denominator
        )

    precision_denominator = (
        tp + fp
    )

    if precision_denominator == 0:

        precision = (
            1.0
            if target_voxels == 0
            else 0.0
        )

    else:

        precision = (
            tp
            / precision_denominator
        )

    recall_denominator = (
        tp + fn
    )

    if recall_denominator == 0:

        recall = (
            1.0
            if pred_voxels == 0
            else 0.0
        )

    else:

        recall = (
            tp
            / recall_denominator
        )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "prediction_voxels": pred_voxels,
        "target_voxels": target_voxels,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def centroid_distance(
    prediction,
    target,
    class_id=SPINAL_CANAL_CLASS,
):

    pred_coords = np.argwhere(
        prediction == class_id
    )

    target_coords = np.argwhere(
        target == class_id
    )

    if (
        len(pred_coords) == 0
        or len(target_coords) == 0
    ):

        return np.nan

    pred_center = (
        pred_coords.mean(
            axis=0
        )
    )

    target_center = (
        target_coords.mean(
            axis=0
        )
    )

    return float(
        np.linalg.norm(
            pred_center
            - target_center
        )
    )


# ============================================================
# INFERENCE ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("SPINAL CANAL INFERENCE ANALYSIS")
print("=" * 78)

print(
    "Using EXACT Part 11 preprocessing:"
)

print(
    "1. 1st-99th percentile normalization"
)

print(
    "2. Center crop/pad"
)

print(
    "3. 96 x 96 x 96 input"
)

analysis_rows = []

start_time = time.time()

with torch.no_grad():

    for index, name in enumerate(
        common_names,
        start=1,
    ):

        try:

            image_path = (
                image_map[name]
            )

            mask_path = (
                mask_map[name]
            )

            _, image_array = (
                load_volume(
                    image_path
                )
            )

            _, mask_array = (
                load_volume(
                    mask_path
                )
            )

            original_shape = tuple(
                image_array.shape
            )

            # ------------------------------------------------
            # IMPORTANT:
            # SAME PREPROCESSING AS PART 11
            # ------------------------------------------------

            image_tensor = (
                prepare_image(
                    image_array
                ).to(DEVICE)
            )

            mask_tensor = (
                prepare_mask(
                    mask_array
                )
            )

            # Safety check
            expected_shape = (
                1,
                1,
                96,
                96,
                96,
            )

            if tuple(
                image_tensor.shape
            ) != expected_shape:

                raise RuntimeError(
                    "Unexpected image tensor "
                    f"shape: "
                    f"{tuple(image_tensor.shape)}"
                )

            # ------------------------------------------------
            # MODEL INFERENCE
            # ------------------------------------------------

            output = model(
                image_tensor
            )

            prediction = (
                torch.argmax(
                    output,
                    dim=1,
                )
                .cpu()
                .numpy()[0]
                .astype(np.int64)
            )

            target = (
                mask_tensor
                .numpy()[0]
                .astype(np.int64)
            )

            # ------------------------------------------------
            # SPINAL CANAL METRICS
            # ------------------------------------------------

            metrics = calculate_metrics(
                prediction,
                target,
                SPINAL_CANAL_CLASS,
            )

            distance = (
                centroid_distance(
                    prediction,
                    target,
                    SPINAL_CANAL_CLASS,
                )
            )

            analysis_rows.append(
                {
                    "file": name,

                    "original_z":
                        original_shape[0],

                    "original_y":
                        original_shape[1],

                    "original_x":
                        original_shape[2],

                    "processed_z": 96,
                    "processed_y": 96,
                    "processed_x": 96,

                    "tp_voxels":
                        metrics["tp"],

                    "fp_voxels":
                        metrics["fp"],

                    "fn_voxels":
                        metrics["fn"],

                    "tn_voxels":
                        metrics["tn"],

                    "prediction_spinal_canal_voxels":
                        metrics[
                            "prediction_voxels"
                        ],

                    "target_spinal_canal_voxels":
                        metrics[
                            "target_voxels"
                        ],

                    "spinal_canal_dice":
                        metrics["dice"],

                    "spinal_canal_iou":
                        metrics["iou"],

                    "spinal_canal_precision":
                        metrics["precision"],

                    "spinal_canal_recall":
                        metrics["recall"],

                    "centroid_distance_voxels":
                        distance,
                }
            )

            if (
                index == 1
                or index % 10 == 0
                or index == len(common_names)
            ):

                print(
                    f"Analyzed "
                    f"{index:3d} / "
                    f"{len(common_names)}"
                )

        except Exception as exc:

            print(
                f"\nERROR: {name}"
            )

            print(
                f"       {exc}"
            )


analysis_df = pd.DataFrame(
    analysis_rows
)

if analysis_df.empty:

    raise RuntimeError(
        "No cases were successfully analyzed."
    )


# ============================================================
# MERGE WITH PART 11
# ============================================================

part11_columns = [
    "file",
    "mean_foreground_dice",
    "mean_foreground_iou",
    "Spinal Canal_dice",
    "Spinal Canal_iou",
    "Spinal Canal_precision",
    "Spinal Canal_recall",
]

available_columns = [
    col
    for col in part11_columns
    if col in part11_df.columns
]

merged_df = analysis_df.merge(
    part11_df[
        available_columns
    ],
    on="file",
    how="left",
)


# ============================================================
# FAILURE CATEGORY
# ============================================================

def failure_category(row):

    dice = (
        row["spinal_canal_dice"]
    )

    precision = (
        row["spinal_canal_precision"]
    )

    recall = (
        row["spinal_canal_recall"]
    )

    if (
        dice < 0.50
        and recall < 0.50
    ):

        return "Major under-segmentation"

    if (
        dice < 0.50
        and precision < 0.50
    ):

        return "Major over-segmentation"

    if (
        recall < 0.70
        and precision >= 0.70
    ):

        return "Under-segmentation"

    if (
        precision < 0.70
        and recall >= 0.70
    ):

        return "Over-segmentation"

    if dice < 0.70:

        return "Moderate segmentation error"

    return "Acceptable segmentation"


merged_df[
    "failure_category"
] = merged_df.apply(
    failure_category,
    axis=1,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("SPINAL CANAL ANALYSIS SUMMARY")
print("=" * 78)

mean_dice = (
    merged_df[
        "spinal_canal_dice"
    ].mean()
)

median_dice = (
    merged_df[
        "spinal_canal_dice"
    ].median()
)

std_dice = (
    merged_df[
        "spinal_canal_dice"
    ].std()
)

mean_iou = (
    merged_df[
        "spinal_canal_iou"
    ].mean()
)

mean_precision = (
    merged_df[
        "spinal_canal_precision"
    ].mean()
)

mean_recall = (
    merged_df[
        "spinal_canal_recall"
    ].mean()
)

mean_centroid = (
    merged_df[
        "centroid_distance_voxels"
    ].mean()
)

print(
    f"Cases analyzed             : "
    f"{len(merged_df)}"
)

print(
    f"Mean Spinal Canal Dice     : "
    f"{mean_dice:.6f}"
)

print(
    f"Median Spinal Canal Dice   : "
    f"{median_dice:.6f}"
)

print(
    f"Std Spinal Canal Dice      : "
    f"{std_dice:.6f}"
)

print(
    f"Mean Spinal Canal IoU      : "
    f"{mean_iou:.6f}"
)

print(
    f"Mean Precision             : "
    f"{mean_precision:.6f}"
)

print(
    f"Mean Recall                : "
    f"{mean_recall:.6f}"
)

print(
    f"Mean centroid distance     : "
    f"{mean_centroid:.4f} voxels"
)


# ============================================================
# FAILURE THRESHOLDS
# ============================================================

print("\n" + "=" * 78)
print("SPINAL CANAL FAILURE COUNTS")
print("=" * 78)

for threshold in [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]:

    count = int(
        (
            merged_df[
                "spinal_canal_dice"
            ]
            < threshold
        ).sum()
    )

    percentage = (
        count
        / len(merged_df)
        * 100.0
    )

    print(
        f"Dice < {threshold:.2f}: "
        f"{count:3d} cases "
        f"({percentage:.2f}%)"
    )


# ============================================================
# WORST / BEST CASES
# ============================================================

worst_cases = (
    merged_df
    .nsmallest(
        10,
        "spinal_canal_dice",
    )
)

best_cases = (
    merged_df
    .nlargest(
        10,
        "spinal_canal_dice",
    )
)

print("\n" + "=" * 78)
print("WORST SPINAL CANAL CASES")
print("=" * 78)

for _, row in worst_cases.iterrows():

    print(
        f"{row['file']:<25}"
        f"Dice: "
        f"{row['spinal_canal_dice']:.6f}  "
        f"Precision: "
        f"{row['spinal_canal_precision']:.6f}  "
        f"Recall: "
        f"{row['spinal_canal_recall']:.6f}"
    )


print("\n" + "=" * 78)
print("BEST SPINAL CANAL CASES")
print("=" * 78)

for _, row in best_cases.iterrows():

    print(
        f"{row['file']:<25}"
        f"Dice: "
        f"{row['spinal_canal_dice']:.6f}"
    )


# ============================================================
# PRIMARY FAILURE
# ============================================================

worst_case = merged_df.loc[
    merged_df[
        "spinal_canal_dice"
    ].idxmin()
]

print("\n" + "=" * 78)
print("PRIMARY SPINAL CANAL FAILURE")
print("=" * 78)

print(
    f"Case       : "
    f"{worst_case['file']}"
)

print(
    f"Dice       : "
    f"{worst_case['spinal_canal_dice']:.6f}"
)

print(
    f"IoU        : "
    f"{worst_case['spinal_canal_iou']:.6f}"
)

print(
    f"Precision  : "
    f"{worst_case['spinal_canal_precision']:.6f}"
)

print(
    f"Recall     : "
    f"{worst_case['spinal_canal_recall']:.6f}"
)

print(
    f"TP voxels  : "
    f"{int(worst_case['tp_voxels'])}"
)

print(
    f"FP voxels  : "
    f"{int(worst_case['fp_voxels'])}"
)

print(
    f"FN voxels  : "
    f"{int(worst_case['fn_voxels'])}"
)

print(
    f"Category   : "
    f"{worst_case['failure_category']}"
)


# ============================================================
# FAILURE CATEGORY DISTRIBUTION
# ============================================================

category_counts = (
    merged_df[
        "failure_category"
    ]
    .value_counts()
)

print("\n" + "=" * 78)
print("FAILURE CATEGORY DISTRIBUTION")
print("=" * 78)

for category, count in (
    category_counts.items()
):

    print(
        f"{category:<30}"
        f"{count:3d} cases "
        f"({count / len(merged_df) * 100:.2f}%)"
    )


# ============================================================
# SAVE TABLES
# ============================================================

analysis_csv = (
    OUTPUT_DIR
    / "spinal_canal_case_analysis.csv"
)

merged_df.to_csv(
    analysis_csv,
    index=False,
)


worst_csv = (
    OUTPUT_DIR
    / "spinal_canal_worst_cases.csv"
)

worst_cases.to_csv(
    worst_csv,
    index=False,
)


category_csv = (
    OUTPUT_DIR
    / "spinal_canal_failure_categories.csv"
)

category_df = (
    category_counts
    .rename("case_count")
    .reset_index()
)

category_df.columns = [
    "failure_category",
    "case_count",
]

category_df[
    "percentage"
] = (
    category_df["case_count"]
    / len(merged_df)
    * 100.0
)

category_df.to_csv(
    category_csv,
    index=False,
)


# ============================================================
# CHART 1 - DICE DISTRIBUTION
# ============================================================

print("\n" + "=" * 78)
print("CREATING SPINAL CANAL ANALYSIS CHARTS")
print("=" * 78)

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    merged_df[
        "spinal_canal_dice"
    ],
    bins=12,
)

plt.xlabel(
    "Spinal Canal Dice"
)

plt.ylabel(
    "Number of Test Cases"
)

plt.title(
    "Spinal Canal Dice Distribution"
)

plt.grid(
    alpha=0.25
)

dice_distribution_path = (
    OUTPUT_DIR
    / "spinal_canal_dice_distribution.png"
)

plt.tight_layout()

plt.savefig(
    dice_distribution_path,
    dpi=200,
)

plt.close()

print(
    f"Saved: "
    f"{dice_distribution_path}"
)


# ============================================================
# CHART 2 - PRECISION VS RECALL
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    merged_df[
        "spinal_canal_recall"
    ],
    merged_df[
        "spinal_canal_precision"
    ],
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Spinal Canal Precision vs Recall"
)

plt.grid(
    alpha=0.25
)

precision_recall_path = (
    OUTPUT_DIR
    / "spinal_canal_precision_recall.png"
)

plt.tight_layout()

plt.savefig(
    precision_recall_path,
    dpi=200,
)

plt.close()

print(
    f"Saved: "
    f"{precision_recall_path}"
)


# ============================================================
# CHART 3 - FP VS FN
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    merged_df[
        "fp_voxels"
    ],
    merged_df[
        "fn_voxels"
    ],
)

plt.xlabel(
    "False Positive Voxels"
)

plt.ylabel(
    "False Negative Voxels"
)

plt.title(
    "Spinal Canal False Positive vs False Negative Voxels"
)

plt.grid(
    alpha=0.25
)

fp_fn_path = (
    OUTPUT_DIR
    / "spinal_canal_fp_vs_fn.png"
)

plt.tight_layout()

plt.savefig(
    fp_fn_path,
    dpi=200,
)

plt.close()

print(
    f"Saved: "
    f"{fp_fn_path}"
)


# ============================================================
# CHART 4 - CENTROID DISTANCE VS DICE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    merged_df[
        "centroid_distance_voxels"
    ],
    merged_df[
        "spinal_canal_dice"
    ],
)

plt.xlabel(
    "Centroid Distance (voxels)"
)

plt.ylabel(
    "Spinal Canal Dice"
)

plt.title(
    "Centroid Distance vs Spinal Canal Dice"
)

plt.grid(
    alpha=0.25
)

centroid_path = (
    OUTPUT_DIR
    / "spinal_canal_centroid_vs_dice.png"
)

plt.tight_layout()

plt.savefig(
    centroid_path,
    dpi=200,
)

plt.close()

print(
    f"Saved: "
    f"{centroid_path}"
)


# ============================================================
# CHART 5 - WORST 10
# ============================================================

worst_plot_df = (
    worst_cases
    .sort_values(
        "spinal_canal_dice"
    )
)

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    worst_plot_df["file"],
    worst_plot_df[
        "spinal_canal_dice"
    ],
)

plt.xlabel(
    "Test Case"
)

plt.ylabel(
    "Spinal Canal Dice"
)

plt.title(
    "Worst 10 Spinal Canal Cases"
)

plt.xticks(
    rotation=45,
    ha="right",
)

plt.grid(
    axis="y",
    alpha=0.25,
)

worst_plot_path = (
    OUTPUT_DIR
    / "spinal_canal_worst_10.png"
)

plt.tight_layout()

plt.savefig(
    worst_plot_path,
    dpi=200,
)

plt.close()

print(
    f"Saved: "
    f"{worst_plot_path}"
)


# ============================================================
# JSON SUMMARY
# ============================================================

summary_json_path = (
    OUTPUT_DIR
    / "spinal_canal_analysis_summary.json"
)

summary_data = {

    "phase":
        "Phase 3 - Part 14",

    "analysis":
        "Spinal Canal Failure-Case Deep Analysis",

    "checkpoint":
        str(CHECKPOINT_PATH),

    "checkpoint_epoch":
        (
            int(checkpoint["epoch"])
            if isinstance(
                checkpoint,
                dict,
            )
            and "epoch" in checkpoint
            else None
        ),

    "best_validation_dice":
        (
            float(
                checkpoint[
                    "best_val_dice"
                ]
            )
            if isinstance(
                checkpoint,
                dict,
            )
            and "best_val_dice" in checkpoint
            else None
        ),

    "test_cases":
        int(len(merged_df)),

    "spinal_canal": {

        "mean_dice":
            float(mean_dice),

        "median_dice":
            float(median_dice),

        "std_dice":
            float(std_dice),

        "mean_iou":
            float(mean_iou),

        "mean_precision":
            float(mean_precision),

        "mean_recall":
            float(mean_recall),

        "mean_centroid_distance_voxels":
            (
                float(mean_centroid)
                if not np.isnan(
                    mean_centroid
                )
                else None
            ),
    },

    "worst_case": {

        "file":
            str(
                worst_case["file"]
            ),

        "dice":
            float(
                worst_case[
                    "spinal_canal_dice"
                ]
            ),

        "iou":
            float(
                worst_case[
                    "spinal_canal_iou"
                ]
            ),

        "precision":
            float(
                worst_case[
                    "spinal_canal_precision"
                ]
            ),

        "recall":
            float(
                worst_case[
                    "spinal_canal_recall"
                ]
            ),

        "tp_voxels":
            int(
                worst_case[
                    "tp_voxels"
                ]
            ),

        "fp_voxels":
            int(
                worst_case[
                    "fp_voxels"
                ]
            ),

        "fn_voxels":
            int(
                worst_case[
                    "fn_voxels"
                ]
            ),

        "failure_category":
            str(
                worst_case[
                    "failure_category"
                ]
            ),
    },

    "preprocessing": {

        "normalization":
            "1st-99th percentile of non-zero voxels",

        "spatial_strategy":
            "center crop/pad",

        "patch_size":
            [96, 96, 96],

        "input_tensor_shape":
            [1, 1, 96, 96, 96],
    },
}


with open(
    summary_json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary_data,
        f,
        indent=4,
    )


# ============================================================
# TEXT REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "phase3_part14_spinal_canal_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "PHASE 3 - PART 14\n"
    )

    f.write(
        "SPINAL CANAL FAILURE-CASE DEEP ANALYSIS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Project root:\n"
        f"{PROJECT_ROOT}\n\n"
    )

    f.write(
        f"Checkpoint:\n"
        f"{CHECKPOINT_PATH}\n\n"
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        if "epoch" in checkpoint:

            f.write(
                f"Checkpoint epoch: "
                f"{checkpoint['epoch']}\n"
            )

        if "best_val_dice" in checkpoint:

            f.write(
                f"Best validation Dice: "
                f"{checkpoint['best_val_dice']:.6f}\n"
            )

    f.write("\n")

    f.write(
        "TEST DATA\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Cases analyzed: "
        f"{len(merged_df)}\n"
    )

    f.write("\n")

    f.write(
        "SPINAL CANAL PERFORMANCE\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Mean Dice: "
        f"{mean_dice:.6f}\n"
    )

    f.write(
        f"Median Dice: "
        f"{median_dice:.6f}\n"
    )

    f.write(
        f"Std Dice: "
        f"{std_dice:.6f}\n"
    )

    f.write(
        f"Mean IoU: "
        f"{mean_iou:.6f}\n"
    )

    f.write(
        f"Mean Precision: "
        f"{mean_precision:.6f}\n"
    )

    f.write(
        f"Mean Recall: "
        f"{mean_recall:.6f}\n"
    )

    f.write(
        f"Mean centroid distance: "
        f"{mean_centroid:.4f} voxels\n"
    )

    f.write("\n")

    f.write(
        "PRIMARY FAILURE CASE\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Case: "
        f"{worst_case['file']}\n"
    )

    f.write(
        f"Dice: "
        f"{worst_case['spinal_canal_dice']:.6f}\n"
    )

    f.write(
        f"IoU: "
        f"{worst_case['spinal_canal_iou']:.6f}\n"
    )

    f.write(
        f"Precision: "
        f"{worst_case['spinal_canal_precision']:.6f}\n"
    )

    f.write(
        f"Recall: "
        f"{worst_case['spinal_canal_recall']:.6f}\n"
    )

    f.write(
        f"TP voxels: "
        f"{int(worst_case['tp_voxels'])}\n"
    )

    f.write(
        f"FP voxels: "
        f"{int(worst_case['fp_voxels'])}\n"
    )

    f.write(
        f"FN voxels: "
        f"{int(worst_case['fn_voxels'])}\n"
    )

    f.write(
        f"Failure category: "
        f"{worst_case['failure_category']}\n"
    )

    f.write("\n")

    f.write(
        "PREPROCESSING\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "Normalization: "
        "1st-99th percentile of non-zero voxels\n"
    )

    f.write(
        "Spatial processing: "
        "center crop/pad\n"
    )

    f.write(
        "Patch size: "
        "96 x 96 x 96\n"
    )

    f.write(
        "Model input: "
        "1 x 1 x 96 x 96 x 96\n"
    )

    f.write("\n")

    f.write(
        "OUTPUTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Analysis CSV:\n"
        f"{analysis_csv}\n\n"
    )

    f.write(
        f"Worst cases CSV:\n"
        f"{worst_csv}\n\n"
    )

    f.write(
        f"Failure categories CSV:\n"
        f"{category_csv}\n\n"
    )

    f.write(
        f"JSON summary:\n"
        f"{summary_json_path}\n\n"
    )

    f.write(
        f"Dice distribution:\n"
        f"{dice_distribution_path}\n\n"
    )

    f.write(
        f"Precision/Recall:\n"
        f"{precision_recall_path}\n\n"
    )

    f.write(
        f"FP/FN chart:\n"
        f"{fp_fn_path}\n\n"
    )

    f.write(
        f"Centroid/Dice chart:\n"
        f"{centroid_path}\n\n"
    )

    f.write(
        f"Worst 10 chart:\n"
        f"{worst_plot_path}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

total_time = (
    time.time() - start_time
)

print("\n" + "=" * 78)
print("PART 14 COMPLETE")
print("=" * 78)

print(
    f"Cases analyzed : "
    f"{len(merged_df)}"
)

print(
    f"Mean Spinal Canal Dice : "
    f"{mean_dice:.6f}"
)

print(
    f"Worst case : "
    f"{worst_case['file']}"
)

print(
    f"Worst Dice : "
    f"{worst_case['spinal_canal_dice']:.6f}"
)

print(
    f"Analysis time : "
    f"{total_time / 60:.2f} minutes"
)

print("\n" + "=" * 78)
print("OUTPUT FILES")
print("=" * 78)

print(
    f"Main analysis : "
    f"{analysis_csv}"
)

print(
    f"Worst cases : "
    f"{worst_csv}"
)

print(
    f"Categories : "
    f"{category_csv}"
)

print(
    f"JSON summary : "
    f"{summary_json_path}"
)

print(
    f"Report : "
    f"{report_path}"
)

print(
    f"Dice chart : "
    f"{dice_distribution_path}"
)

print(
    f"Precision/Recall : "
    f"{precision_recall_path}"
)

print(
    f"FP/FN chart : "
    f"{fp_fn_path}"
)

print(
    f"Centroid chart : "
    f"{centroid_path}"
)

print(
    f"Worst 10 chart : "
    f"{worst_plot_path}"
)

print("\n" + "=" * 78)
print("PHASE 3 - PART 14 COMPLETE")
print("=" * 78)