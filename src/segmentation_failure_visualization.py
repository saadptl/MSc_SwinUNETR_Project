"""
PHASE 3 - PART 15
SPINAL CANAL FAILURE-CASE VISUALIZATION

Purpose:
    Detailed qualitative analysis of difficult spinal canal
    segmentation cases.

Uses the same:
    - Swin-UNETR architecture
    - best epoch-28 checkpoint
    - normalization
    - center crop/pad
    - 96 x 96 x 96 input

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

PART14_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "spinal_canal_analysis"
    / "spinal_canal_case_analysis.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "failure_visualization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = (96, 96, 96)

IN_CHANNELS = 1
OUT_CHANNELS = 4
FEATURE_SIZE = 24

SPINAL_CANAL_CLASS = 2

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("PHASE 3 - PART 15")
print("SPINAL CANAL FAILURE-CASE VISUALIZATION")
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
# VALIDATION
# ============================================================

required_paths = [
    TEST_IMAGES_DIR,
    TEST_MASKS_DIR,
    CHECKPOINT_PATH,
    PART11_RESULTS,
    PART14_RESULTS,
]

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )


# ============================================================
# DEVICE
# ============================================================

print("\n" + "=" * 78)
print("DEVICE")
print("=" * 78)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

print("\n" + "=" * 78)
print("LOADING PART 11 / PART 14 RESULTS")
print("=" * 78)

part11_df = pd.read_csv(
    PART11_RESULTS
)

part14_df = pd.read_csv(
    PART14_RESULTS
)

print(
    f"Part 11 cases: {len(part11_df)}"
)

print(
    f"Part 14 cases: {len(part14_df)}"
)


# ============================================================
# TEST FILES
# ============================================================

image_files = sorted(
    TEST_IMAGES_DIR.glob("*.mha")
)

mask_files = sorted(
    TEST_MASKS_DIR.glob("*.mha")
)

image_map = {
    p.stem: p
    for p in image_files
}

mask_map = {
    p.stem: p
    for p in mask_files
}

common_cases = sorted(
    set(image_map)
    & set(mask_map)
)

print("\n" + "=" * 78)
print("TEST CASES")
print("=" * 78)

print(
    f"MRI files : {len(image_files)}"
)

print(
    f"Mask files: {len(mask_files)}"
)

print(
    f"Matching  : {len(common_cases)}"
)


# ============================================================
# LOAD MODEL
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

if isinstance(
    checkpoint,
    dict
):

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'N/A')}"
    )

    print(
        f"Best validation Dice: "
        f"{checkpoint.get('best_val_dice', 'N/A')}"
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model = model.to(
    DEVICE
)

model.eval()

print("✓ Model loaded.")
print("✓ Evaluation mode enabled.")


# ============================================================
# ARRAY UTILITIES
# ============================================================

def force_3d_volume(
    array,
    name="array",
):
    """
    Guarantee a 3-D [Z,Y,X] volume.

    This prevents accidental 1-D/2-D arrays from
    reaching the visualization stage.
    """

    array = np.asarray(array)

    # Remove dimensions of size 1.
    array = np.squeeze(array)

    if array.ndim == 3:

        return array

    if array.ndim == 2:

        # Treat a 2-D image as one axial slice.
        return array[np.newaxis, ...]

    if array.ndim == 1:

        raise ValueError(
            f"{name} became 1-D with shape "
            f"{array.shape}. "
            "A valid 2-D or 3-D volume is required."
        )

    raise ValueError(
        f"{name} has unsupported shape "
        f"{array.shape}."
    )


def force_2d_slice(
    array,
    slice_index,
    name="array",
):
    """
    Safely extract one [Y,X] axial slice.
    """

    volume = force_3d_volume(
        array,
        name,
    )

    if not (
        0 <= slice_index < volume.shape[0]
    ):

        raise IndexError(
            f"Slice {slice_index} is outside "
            f"volume shape {volume.shape}."
        )

    result = volume[
        slice_index
    ]

    result = np.asarray(
        result
    )

    if result.ndim != 2:

        raise ValueError(
            f"{name} slice is not 2-D. "
            f"Shape: {result.shape}"
        )

    return result


# ============================================================
# PREPROCESSING
# ============================================================

def center_crop_or_pad(
    array,
    target_shape=PATCH_SIZE,
    pad_value=0,
):

    array = force_3d_volume(
        array,
        "center_crop_or_pad input",
    )

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

            target_start = (
                tgt_size - src_size
            ) // 2

            target_end = (
                target_start + src_size
            )

            source_slices.append(
                slice(
                    0,
                    src_size,
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


def normalize_image(
    array,
):

    array = force_3d_volume(
        array,
        "image",
    )

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

    array = np.clip(
        array,
        0.0,
        1.0,
    )

    return array.astype(
        np.float32
    )


def prepare_image(
    array,
):

    array = force_3d_volume(
        array,
        "MRI",
    )

    array = normalize_image(
        array
    )

    array = center_crop_or_pad(
        array,
        PATCH_SIZE,
        0,
    )

    if array.shape != PATCH_SIZE:

        raise RuntimeError(
            f"Prepared image has wrong shape: "
            f"{array.shape}"
        )

    tensor = torch.from_numpy(
        array
    ).float()

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return tensor


def prepare_mask(
    array,
):

    array = force_3d_volume(
        array,
        "mask",
    )

    array = center_crop_or_pad(
        array.astype(np.int64),
        PATCH_SIZE,
        0,
    )

    if array.shape != PATCH_SIZE:

        raise RuntimeError(
            f"Prepared mask has wrong shape: "
            f"{array.shape}"
        )

    tensor = torch.from_numpy(
        array
    ).long()

    return tensor


# ============================================================
# LOAD VOLUME
# ============================================================

def load_volume(
    path,
):

    image = sitk.ReadImage(
        str(path)
    )

    array = sitk.GetArrayFromImage(
        image
    )

    return force_3d_volume(
        array,
        str(path),
    )


# ============================================================
# SLICE SELECTION
# ============================================================

def select_best_spinal_slice(
    target,
    prediction,
):

    target = force_3d_volume(
        target,
        "target",
    )

    prediction = force_3d_volume(
        prediction,
        "prediction",
    )

    target_mask = (
        target
        == SPINAL_CANAL_CLASS
    )

    target_counts = (
        target_mask.sum(
            axis=(1, 2)
        )
    )

    if (
        target_counts.size > 0
        and target_counts.max() > 0
    ):

        return int(
            np.argmax(
                target_counts
            )
        )

    prediction_mask = (
        prediction
        == SPINAL_CANAL_CLASS
    )

    prediction_counts = (
        prediction_mask.sum(
            axis=(1, 2)
        )
    )

    if (
        prediction_counts.size > 0
        and prediction_counts.max() > 0
    ):

        return int(
            np.argmax(
                prediction_counts
            )
        )

    # No spinal canal in either mask.
    return target.shape[0] // 2


# ============================================================
# METRICS
# ============================================================

def calculate_spinal_metrics(
    prediction,
    target,
):

    prediction = force_3d_volume(
        prediction,
        "prediction",
    )

    target = force_3d_volume(
        target,
        "target",
    )

    pred = (
        prediction
        == SPINAL_CANAL_CLASS
    )

    true = (
        target
        == SPINAL_CANAL_CLASS
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

    denominator = (
        2 * tp
        + fp
        + fn
    )

    if denominator == 0:

        dice = 1.0

    else:

        dice = (
            2.0 * tp
        ) / denominator

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": float(dice),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def create_case_visualization(
    case_name,
    image,
    target,
    prediction,
    output_path,
    dice,
):

    # --------------------------------------------------------
    # Force correct dimensions
    # --------------------------------------------------------

    image = force_3d_volume(
        image,
        "image",
    )

    target = force_3d_volume(
        target,
        "target",
    )

    prediction = force_3d_volume(
        prediction,
        "prediction",
    )

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    if image.shape != target.shape:

        raise ValueError(
            f"Image/target shape mismatch: "
            f"{image.shape} vs {target.shape}"
        )

    if target.shape != prediction.shape:

        raise ValueError(
            f"Target/prediction shape mismatch: "
            f"{target.shape} vs {prediction.shape}"
        )

    # --------------------------------------------------------
    # Select axial slice
    # --------------------------------------------------------

    slice_index = (
        select_best_spinal_slice(
            target,
            prediction,
        )
    )

    # --------------------------------------------------------
    # Extract true 2-D slices
    # --------------------------------------------------------

    mri_slice = force_2d_slice(
        image,
        slice_index,
        "MRI",
    )

    target_slice = force_2d_slice(
        target,
        slice_index,
        "target",
    )

    prediction_slice = force_2d_slice(
        prediction,
        slice_index,
        "prediction",
    )

    # --------------------------------------------------------
    # Spinal canal masks
    # --------------------------------------------------------

    gt_spinal = (
        target_slice
        == SPINAL_CANAL_CLASS
    )

    pred_spinal = (
        prediction_slice
        == SPINAL_CANAL_CLASS
    )

    true_positive = (
        gt_spinal
        & pred_spinal
    )

    false_positive = (
        ~gt_spinal
        & pred_spinal
    )

    false_negative = (
        gt_spinal
        & ~pred_spinal
    )

    # --------------------------------------------------------
    # Error map
    #
    # 0 = background
    # 1 = true positive
    # 2 = false positive
    # 3 = false negative
    # --------------------------------------------------------

    error_map = np.zeros(
        target_slice.shape,
        dtype=np.uint8,
    )

    error_map[
        true_positive
    ] = 1

    error_map[
        false_positive
    ] = 2

    error_map[
        false_negative
    ] = 3

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    # MRI
    axes[0].imshow(
        mri_slice,
        cmap="gray",
    )

    axes[0].set_title(
        f"MRI\nSlice {slice_index}"
    )

    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(
        target_slice,
        cmap="nipy_spectral",
        interpolation="nearest",
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[1].axis("off")

    # Prediction
    axes[2].imshow(
        prediction_slice,
        cmap="nipy_spectral",
        interpolation="nearest",
    )

    axes[2].set_title(
        "Prediction"
    )

    axes[2].axis("off")

    # Error map
    axes[3].imshow(
        error_map,
        cmap="viridis",
        interpolation="nearest",
    )

    axes[3].set_title(
        "Spinal Canal Error Map"
    )

    axes[3].axis("off")

    # Overlay
    axes[4].imshow(
        mri_slice,
        cmap="gray",
    )

    axes[4].imshow(
        gt_spinal.astype(
            np.float32
        ),
        alpha=0.35,
        cmap="winter",
    )

    axes[4].imshow(
        pred_spinal.astype(
            np.float32
        ),
        alpha=0.35,
        cmap="spring",
    )

    axes[4].set_title(
        "GT / Prediction Overlay"
    )

    axes[4].axis("off")

    fig.suptitle(
        f"{case_name} | "
        f"Spinal Canal Dice = "
        f"{dice:.4f}",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return slice_index


# ============================================================
# SELECT CASES
# ============================================================

print("\n" + "=" * 78)
print("SELECTING REPRESENTATIVE CASES")
print("=" * 78)

part14_df = part14_df[
    part14_df["file"].isin(
        common_cases
    )
].copy()

part14_df = part14_df.sort_values(
    "spinal_canal_dice"
)

worst_5 = (
    part14_df
    .head(5)
    ["file"]
    .tolist()
)

best_3 = (
    part14_df
    .tail(3)
    ["file"]
    .tolist()
)

median_index = (
    len(part14_df)
    // 2
)

median_case = (
    part14_df
    .iloc[
        median_index
    ]["file"]
)

selected_cases = []

for case_name in (
    worst_5
    + [median_case]
    + best_3
):

    if case_name not in selected_cases:

        selected_cases.append(
            case_name
        )


print(
    f"Worst cases: {worst_5}"
)

print(
    f"Median case: {median_case}"
)

print(
    f"Best cases: {best_3}"
)

print(
    f"Total selected: "
    f"{len(selected_cases)}"
)


# ============================================================
# INFERENCE + VISUALIZATION
# ============================================================

print("\n" + "=" * 78)
print("GENERATING FAILURE VISUALIZATIONS")
print("=" * 78)

visualization_rows = []

start_time = time.time()

with torch.no_grad():

    for counter, case_name in enumerate(
        selected_cases,
        start=1,
    ):

        print(
            f"\n[{counter}/{len(selected_cases)}] "
            f"{case_name}"
        )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        image_array = load_volume(
            image_map[case_name]
        )

        mask_array = load_volume(
            mask_map[case_name]
        )

        original_shape = (
            tuple(
                image_array.shape
            )
        )

        # ----------------------------------------------------
        # Prepare
        # ----------------------------------------------------

        image_tensor = (
            prepare_image(
                image_array
            )
            .to(DEVICE)
        )

        mask_tensor = (
            prepare_mask(
                mask_array
            )
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

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
            .numpy()
            .astype(np.int64)
        )

        # ----------------------------------------------------
        # FORCE CORRECT 3-D SHAPES
        # ----------------------------------------------------

        prediction = force_3d_volume(
            prediction,
            "prediction",
        )

        target = force_3d_volume(
            target,
            "target",
        )

        processed_image = (
            image_tensor[
                0, 0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        processed_image = (
            force_3d_volume(
                processed_image,
                "processed image",
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = (
            calculate_spinal_metrics(
                prediction,
                target,
            )
        )

        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        case_output_dir = (
            OUTPUT_DIR
            / case_name
        )

        case_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        visualization_path = (
            case_output_dir
            / f"{case_name}_analysis.png"
        )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        selected_slice = (
            create_case_visualization(
                case_name,
                processed_image,
                target,
                prediction,
                visualization_path,
                metrics["dice"],
            )
        )

        visualization_rows.append(
            {
                "file":
                    case_name,

                "original_shape":
                    str(
                        original_shape
                    ),

                "processed_shape":
                    str(
                        prediction.shape
                    ),

                "selected_slice":
                    selected_slice,

                "spinal_canal_dice":
                    metrics["dice"],

                "tp":
                    metrics["tp"],

                "fp":
                    metrics["fp"],

                "fn":
                    metrics["fn"],

                "visualization":
                    str(
                        visualization_path
                    ),
            }
        )

        print(
            f"Original shape: "
            f"{original_shape}"
        )

        print(
            f"Processed shape: "
            f"{prediction.shape}"
        )

        print(
            f"Selected slice: "
            f"{selected_slice}"
        )

        print(
            f"Dice: "
            f"{metrics['dice']:.6f}"
        )

        print(
            f"TP: "
            f"{metrics['tp']}"
        )

        print(
            f"FP: "
            f"{metrics['fp']}"
        )

        print(
            f"FN: "
            f"{metrics['fn']}"
        )

        print(
            f"Saved: "
            f"{visualization_path}"
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    visualization_rows
)

summary_csv = (
    OUTPUT_DIR
    / "failure_visualization_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False,
)


# ============================================================
# SAVE SELECTED CASES
# ============================================================

case_index_path = (
    OUTPUT_DIR
    / "selected_failure_cases.json"
)

case_index = {
    "worst_cases": worst_5,
    "median_case": median_case,
    "best_cases": best_3,
    "all_selected_cases":
        selected_cases,
}

with open(
    case_index_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        case_index,
        f,
        indent=4,
    )


# ============================================================
# REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "phase3_part15_failure_visualization_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "PHASE 3 - PART 15\n"
    )

    f.write(
        "SPINAL CANAL FAILURE-CASE VISUALIZATION\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Checkpoint:\n"
        f"{CHECKPOINT_PATH}\n\n"
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        f.write(
            f"Checkpoint epoch: "
            f"{checkpoint.get('epoch', 'N/A')}\n"
        )

        f.write(
            f"Best validation Dice: "
            f"{checkpoint.get('best_val_dice', 'N/A')}\n\n"
        )

    f.write(
        "PREPROCESSING\n"
    )

    f.write(
        "-" * 70
        + "\n"
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
        "96 x 96 x 96\n\n"
    )

    f.write(
        "SELECTED CASES\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    f.write(
        f"Worst 5: {worst_5}\n"
    )

    f.write(
        f"Median: {median_case}\n"
    )

    f.write(
        f"Best 3: {best_3}\n\n"
    )

    f.write(
        "CASE RESULTS\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    for _, row in (
        summary_df.iterrows()
    ):

        f.write(
            f"{row['file']}\n"
        )

        f.write(
            f"  Selected slice: "
            f"{row['selected_slice']}\n"
        )

        f.write(
            f"  Dice: "
            f"{row['spinal_canal_dice']:.6f}\n"
        )

        f.write(
            f"  TP: "
            f"{row['tp']}\n"
        )

        f.write(
            f"  FP: "
            f"{row['fp']}\n"
        )

        f.write(
            f"  FN: "
            f"{row['fn']}\n"
        )

        f.write(
            f"  Visualization: "
            f"{row['visualization']}\n\n"
        )

    f.write(
        "INTERPRETATION\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    f.write(
        "The visualizations compare the MRI, "
        "ground-truth segmentation, model prediction, "
        "spinal-canal error map, and ground-truth/"
        "prediction overlay.\n"
    )

    f.write(
        "The analysis is intended to identify "
        "under-segmentation, over-segmentation, "
        "spatial mismatch, and difficult anatomical cases.\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

elapsed = (
    time.time()
    - start_time
)

print("\n" + "=" * 78)
print("PART 15 COMPLETE")
print("=" * 78)

print(
    f"Selected cases : "
    f"{len(selected_cases)}"
)

print(
    f"Execution time : "
    f"{elapsed / 60:.2f} minutes"
)

print("\n" + "=" * 78)
print("OUTPUT FILES")
print("=" * 78)

print(
    f"Visualization directory:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"Summary CSV:\n"
    f"{summary_csv}"
)

print(
    f"Selected cases JSON:\n"
    f"{case_index_path}"
)

print(
    f"Report:\n"
    f"{report_path}"
)

print("\n" + "=" * 78)
print("PHASE 3 - PART 15 COMPLETE")
print("=" * 78)