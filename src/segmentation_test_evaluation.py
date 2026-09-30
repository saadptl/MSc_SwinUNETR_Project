"""
PHASE 3 - PART 11
SWIN-UNETR TEST SET EVALUATION

Evaluates the best trained Swin-UNETR segmentation model on the
patient-disjoint held-out test set.

Classes:
    0 - Background
    1 - Vertebrae
    2 - Spinal Canal
    3 - Intervertebral Disc
"""

from pathlib import Path
import csv
import json
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "test_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = (96, 96, 96)

IN_CHANNELS = 1
OUT_CHANNELS = 4
FEATURE_SIZE = 24

CLASS_NAMES = {
    0: "Background",
    1: "Vertebrae",
    2: "Spinal Canal",
    3: "Intervertebral Disc",
}

FOREGROUND_CLASSES = [1, 2, 3]

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("PHASE 3 - PART 11")
print("SWIN-UNETR TEST SET EVALUATION")
print("=" * 78)

print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nTEST DIRECTORY")
print(TEST_DIR)

print("\nTEST IMAGE DIRECTORY")
print(TEST_IMAGES_DIR)

print("\nTEST MASK DIRECTORY")
print(TEST_MASKS_DIR)

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


# ============================================================
# DEVICE
# ============================================================

print("\n" + "=" * 78)
print("DEVICE")
print("=" * 78)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
    )
    print(f"CUDA: {torch.version.cuda}")
else:
    print("CUDA is not available. Evaluation will use CPU.")


# ============================================================
# COLLECT TEST FILES
# ============================================================

image_files = sorted(TEST_IMAGES_DIR.glob("*.mha"))
mask_files = sorted(TEST_MASKS_DIR.glob("*.mha"))

print("\n" + "=" * 78)
print("TEST DATASET")
print("=" * 78)

print(f"Test MRI files : {len(image_files)}")
print(f"Test mask files: {len(mask_files)}")

image_map = {p.stem: p for p in image_files}
mask_map = {p.stem: p for p in mask_files}

common_names = sorted(set(image_map) & set(mask_map))

missing_masks = sorted(set(image_map) - set(mask_map))
orphan_masks = sorted(set(mask_map) - set(image_map))

print(f"Matching pairs : {len(common_names)}")
print(f"Missing masks  : {len(missing_masks)}")
print(f"Orphan masks   : {len(orphan_masks)}")

if missing_masks:
    print("\nMissing masks:")
    for name in missing_masks[:20]:
        print(f"  {name}")

if orphan_masks:
    print("\nOrphan masks:")
    for name in orphan_masks[:20]:
        print(f"  {name}")

if not common_names:
    raise RuntimeError("No valid test MRI/mask pairs found.")

if missing_masks or orphan_masks:
    raise RuntimeError(
        "Test dataset pairing is incomplete."
    )


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 78)
print("CREATING SWIN-UNETR")
print("=" * 78)

model = SwinUNETR(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    feature_size=FEATURE_SIZE,
    use_checkpoint=False,
)

print("✓ Swin-UNETR created.")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\n" + "=" * 78)
print("LOADING BEST CHECKPOINT")
print("=" * 78)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

print(f"Checkpoint type: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print("Checkpoint keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

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

print("✓ Model weights loaded.")
print(f"✓ Model moved to: {DEVICE}")


# ============================================================
# HELPER FUNCTIONS
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
    Convert [Z, Y, X] volume to target shape.
    Uses the same center crop/pad strategy used by training.
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
            src_start = (src_size - tgt_size) // 2
            src_end = src_start + tgt_size

            source_slices.append(
                slice(src_start, src_end)
            )

            target_slices.append(
                slice(0, tgt_size)
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
                slice(src_start, src_end)
            )

            target_slices.append(
                slice(target_start, target_end)
            )

    result[
        tuple(target_slices)
    ] = array[
        tuple(source_slices)
    ]

    return result


def normalize_image(array):
    """
    Percentile normalization compatible with the
    segmentation preprocessing approach.
    """

    array = array.astype(np.float32)

    nonzero = array[array != 0]

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

    return array.astype(np.float32)


def prepare_image(array):
    array = normalize_image(array)

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


def dice_score(
    prediction,
    target,
    class_id,
    epsilon=1e-6,
):
    pred = prediction == class_id
    true = target == class_id

    intersection = np.logical_and(
        pred,
        true,
    ).sum()

    pred_sum = pred.sum()
    true_sum = true.sum()

    denominator = pred_sum + true_sum

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection + epsilon
    ) / (
        denominator + epsilon
    )


def iou_score(
    prediction,
    target,
    class_id,
    epsilon=1e-6,
):
    pred = prediction == class_id
    true = target == class_id

    intersection = np.logical_and(
        pred,
        true,
    ).sum()

    union = np.logical_or(
        pred,
        true,
    ).sum()

    if union == 0:
        return 1.0

    return (
        intersection + epsilon
    ) / (
        union + epsilon
    )


def precision_score(
    prediction,
    target,
    class_id,
    epsilon=1e-6,
):
    pred = prediction == class_id
    true = target == class_id

    tp = np.logical_and(
        pred,
        true,
    ).sum()

    fp = np.logical_and(
        pred,
        np.logical_not(true),
    ).sum()

    return (
        tp + epsilon
    ) / (
        tp + fp + epsilon
    )


def recall_score(
    prediction,
    target,
    class_id,
    epsilon=1e-6,
):
    pred = prediction == class_id
    true = target == class_id

    tp = np.logical_and(
        pred,
        true,
    ).sum()

    fn = np.logical_and(
        np.logical_not(pred),
        true,
    ).sum()

    return (
        tp + epsilon
    ) / (
        tp + fn + epsilon
    )


# ============================================================
# TEST LOOP
# ============================================================

print("\n" + "=" * 78)
print("STARTING TEST EVALUATION")
print("=" * 78)

print(f"Test cases: {len(common_names)}")
print("No optimizer or gradient computation will be performed.")
print("✓ Model is in evaluation mode.")

results = []

total_start = time.time()

with torch.no_grad():

    for index, name in enumerate(
        common_names,
        start=1,
    ):

        image_path = image_map[name]
        mask_path = mask_map[name]

        try:

            _, image_array = load_volume(
                image_path
            )

            _, mask_array = load_volume(
                mask_path
            )

            image_tensor = prepare_image(
                image_array
            ).to(DEVICE)

            mask_tensor = prepare_mask(
                mask_array
            )

            output = model(
                image_tensor
            )

            prediction = torch.argmax(
                output,
                dim=1,
            )

            prediction = (
                prediction
                .cpu()
                .numpy()[0]
                .astype(np.int64)
            )

            target = (
                mask_tensor
                .numpy()[0]
                .astype(np.int64)
            )

            unique_prediction = np.unique(
                prediction
            )

            unique_target = np.unique(
                target
            )

            class_metrics = {}

            for class_id in FOREGROUND_CLASSES:

                dice = dice_score(
                    prediction,
                    target,
                    class_id,
                )

                iou = iou_score(
                    prediction,
                    target,
                    class_id,
                )

                precision = precision_score(
                    prediction,
                    target,
                    class_id,
                )

                recall = recall_score(
                    prediction,
                    target,
                    class_id,
                )

                class_metrics[
                    class_id
                ] = {
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                }

            foreground_dice = [
                class_metrics[c]["dice"]
                for c in FOREGROUND_CLASSES
            ]

            mean_dice = float(
                np.mean(
                    foreground_dice
                )
            )

            foreground_iou = [
                class_metrics[c]["iou"]
                for c in FOREGROUND_CLASSES
            ]

            mean_iou = float(
                np.mean(
                    foreground_iou
                )
            )

            row = {
                "file": name,
                "mean_foreground_dice": mean_dice,
                "mean_foreground_iou": mean_iou,
                "prediction_labels": ",".join(
                    map(
                        str,
                        unique_prediction,
                    )
                ),
                "target_labels": ",".join(
                    map(
                        str,
                        unique_target,
                    )
                ),
            }

            for class_id in FOREGROUND_CLASSES:

                class_name = CLASS_NAMES[
                    class_id
                ]

                metrics = class_metrics[
                    class_id
                ]

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

            results.append(row)

            if (
                index == 1
                or index % 10 == 0
                or index == len(common_names)
            ):
                print(
                    f"Evaluated "
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

            results.append({
                "file": name,
                "error": str(exc),
            })


total_time = (
    time.time() - total_start
)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

successful_df = results_df[
    ~results_df["mean_foreground_dice"].isna()
].copy()

failed_count = len(
    results_df
) - len(
    successful_df
)


if successful_df.empty:
    raise RuntimeError(
        "No test cases were evaluated successfully."
    )


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for class_id in FOREGROUND_CLASSES:

    class_name = CLASS_NAMES[
        class_id
    ]

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

    summary_rows.append({
        "class_id": class_id,
        "class_name": class_name,
        "dice_mean": successful_df[
            dice_column
        ].mean(),
        "dice_std": successful_df[
            dice_column
        ].std(),
        "dice_median": successful_df[
            dice_column
        ].median(),
        "iou_mean": successful_df[
            iou_column
        ].mean(),
        "iou_std": successful_df[
            iou_column
        ].std(),
        "precision_mean": successful_df[
            precision_column
        ].mean(),
        "recall_mean": successful_df[
            recall_column
        ].mean(),
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

overall_mean_dice = successful_df[
    "mean_foreground_dice"
].mean()

overall_std_dice = successful_df[
    "mean_foreground_dice"
].std()

overall_median_dice = successful_df[
    "mean_foreground_dice"
].median()

overall_mean_iou = successful_df[
    "mean_foreground_iou"
].mean()

overall_std_iou = successful_df[
    "mean_foreground_iou"
].std()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 78)
print("TEST EVALUATION COMPLETE")
print("=" * 78)

print(
    f"Successful cases : "
    f"{len(successful_df)}"
)

print(
    f"Failed cases     : "
    f"{failed_count}"
)

print(
    f"Evaluation time  : "
    f"{total_time / 60:.2f} minutes"
)

print("\n" + "=" * 78)
print("OVERALL TEST PERFORMANCE")
print("=" * 78)

print(
    f"Mean Foreground Dice   : "
    f"{overall_mean_dice:.6f}"
)

print(
    f"Std Foreground Dice    : "
    f"{overall_std_dice:.6f}"
)

print(
    f"Median Foreground Dice : "
    f"{overall_median_dice:.6f}"
)

print(
    f"Mean Foreground IoU    : "
    f"{overall_mean_iou:.6f}"
)

print(
    f"Std Foreground IoU     : "
    f"{overall_std_iou:.6f}"
)

print("\n" + "=" * 78)
print("PER-CLASS TEST PERFORMANCE")
print("=" * 78)

for _, row in summary_df.iterrows():

    print(
        f"{row['class_name']:<22}"
        f"Dice: {row['dice_mean']:.6f}  "
        f"IoU: {row['iou_mean']:.6f}  "
        f"Precision: {row['precision_mean']:.6f}  "
        f"Recall: {row['recall_mean']:.6f}"
    )


# ============================================================
# BEST / WORST CASES
# ============================================================

best_cases = successful_df.nlargest(
    5,
    "mean_foreground_dice",
)

worst_cases = successful_df.nsmallest(
    5,
    "mean_foreground_dice",
)

print("\n" + "=" * 78)
print("BEST TEST CASES")
print("=" * 78)

for _, row in best_cases.iterrows():

    print(
        f"{row['file']:<25}"
        f"{row['mean_foreground_dice']:.6f}"
    )


print("\n" + "=" * 78)
print("WORST TEST CASES")
print("=" * 78)

for _, row in worst_cases.iterrows():

    print(
        f"{row['file']:<25}"
        f"{row['mean_foreground_dice']:.6f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_csv = (
    OUTPUT_DIR
    / "test_case_results.csv"
)

summary_csv = (
    OUTPUT_DIR
    / "test_class_summary.csv"
)

results_df.to_csv(
    results_csv,
    index=False,
)

summary_df.to_csv(
    summary_csv,
    index=False,
)


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

summary_json = (
    OUTPUT_DIR
    / "test_evaluation_summary.json"
)

json_data = {
    "checkpoint": str(
        CHECKPOINT_PATH
    ),
    "test_cases": len(
        common_names
    ),
    "successful_cases": len(
        successful_df
    ),
    "failed_cases": failed_count,
    "best_validation_dice": (
        float(
            checkpoint[
                "best_val_dice"
            ]
        )
        if isinstance(checkpoint, dict)
        and "best_val_dice" in checkpoint
        else None
    ),
    "overall": {
        "mean_foreground_dice":
            float(overall_mean_dice),
        "std_foreground_dice":
            float(overall_std_dice),
        "median_foreground_dice":
            float(overall_median_dice),
        "mean_foreground_iou":
            float(overall_mean_iou),
        "std_foreground_iou":
            float(overall_std_iou),
    },
    "classes": summary_df.to_dict(
        orient="records"
    ),
}

with open(
    summary_json,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_data,
        f,
        indent=4,
    )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "phase3_part11_test_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "PHASE 3 - PART 11\n"
    )

    f.write(
        "SWIN-UNETR TEST SET EVALUATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Checkpoint:\n"
        f"{CHECKPOINT_PATH}\n\n"
    )

    f.write(
        f"Test cases: "
        f"{len(common_names)}\n"
    )

    f.write(
        f"Successful cases: "
        f"{len(successful_df)}\n"
    )

    f.write(
        f"Failed cases: "
        f"{failed_count}\n\n"
    )

    f.write(
        "OVERALL TEST PERFORMANCE\n"
    )

    f.write(
        f"Mean Foreground Dice: "
        f"{overall_mean_dice:.6f}\n"
    )

    f.write(
        f"Std Foreground Dice: "
        f"{overall_std_dice:.6f}\n"
    )

    f.write(
        f"Median Foreground Dice: "
        f"{overall_median_dice:.6f}\n"
    )

    f.write(
        f"Mean Foreground IoU: "
        f"{overall_mean_iou:.6f}\n"
    )

    f.write(
        f"Std Foreground IoU: "
        f"{overall_std_iou:.6f}\n\n"
    )

    f.write(
        "PER-CLASS RESULTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for _, row in summary_df.iterrows():

        f.write(
            f"{row['class_name']}\n"
        )

        f.write(
            f"  Dice: "
            f"{row['dice_mean']:.6f}\n"
        )

        f.write(
            f"  Dice Std: "
            f"{row['dice_std']:.6f}\n"
        )

        f.write(
            f"  IoU: "
            f"{row['iou_mean']:.6f}\n"
        )

        f.write(
            f"  Precision: "
            f"{row['precision_mean']:.6f}\n"
        )

        f.write(
            f"  Recall: "
            f"{row['recall_mean']:.6f}\n\n"
        )

    f.write(
        "TRAINING REFERENCE\n"
    )

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:
            f.write(
                f"Best checkpoint epoch: "
                f"{checkpoint['epoch']}\n"
            )

        if "best_val_dice" in checkpoint:
            f.write(
                f"Best validation Dice: "
                f"{checkpoint['best_val_dice']:.6f}\n"
            )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 78)
print("OUTPUT FILES")
print("=" * 78)

print(
    f"Case results   : {results_csv}"
)

print(
    f"Class summary  : {summary_csv}"
)

print(
    f"JSON summary   : {summary_json}"
)

print(
    f"Test report    : {report_path}"
)

print("\n" + "=" * 78)
print("PHASE 3 - PART 11 COMPLETE")
print("=" * 78)