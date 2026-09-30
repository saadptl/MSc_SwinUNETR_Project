"""
PHASE 3 - PART 19
T2 SPACE FAILURE CASE VISUAL INSPECTION

Purpose
-------
Generate qualitative visual comparisons for T2 SPACE MRI cases.

Cases:
    Failure:
        69_t2_SPACE
        166_t2_SPACE
        161_t2_SPACE
        162_t2_SPACE
        177_t2_SPACE
        107_t2_SPACE

    Successful:
        45_t2_SPACE
        127_t2_SPACE
        41_t2_SPACE

For every selected case:
    MRI
    Ground Truth Spinal Canal
    Prediction
    TP / FP / FN error map
    Overlay

Important
---------
This script performs inference using the same preprocessing
configuration used by Part 11.

It does NOT train the model.
It does NOT modify the checkpoint.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from monai.networks.nets import SwinUNETR


# ============================================================
# START TIME
# ============================================================

start_time = time.time()


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

IMAGE_DIR = TEST_DIR / "images"
MASK_DIR = TEST_DIR / "masks"

CHECKPOINT = (
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

PART18_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "t2_space_analysis"
    / "t2_space_case_analysis.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "t2_space_visual_inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SELECTED CASES
# ============================================================

FAILURE_CASES = [
    "69_t2_SPACE",
    "166_t2_SPACE",
    "161_t2_SPACE",
    "162_t2_SPACE",
    "177_t2_SPACE",
    "107_t2_SPACE",
]

SUCCESS_CASES = [
    "45_t2_SPACE",
    "127_t2_SPACE",
    "41_t2_SPACE",
]

ALL_CASES = (
    FAILURE_CASES
    + SUCCESS_CASES
)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = (
    96,
    96,
    96,
)

NUM_CLASSES = 4
FEATURE_SIZE = 24

SPINAL_CANAL_LABEL = 2

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("PHASE 3 - PART 19")
print("T2 SPACE FAILURE CASE VISUAL INSPECTION")
print("=" * 78)

print()
print("PROJECT ROOT")
print(PROJECT_ROOT)

print()
print("TEST DIRECTORY")
print(TEST_DIR)

print()
print("CHECKPOINT")
print(CHECKPOINT)

print()
print("OUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================
# VALIDATION
# ============================================================

for path in [
    TEST_DIR,
    IMAGE_DIR,
    MASK_DIR,
    CHECKPOINT,
    PART11_RESULTS,
    PART18_RESULTS,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )


# ============================================================
# DEVICE
# ============================================================

print()
print("=" * 78)
print("DEVICE")
print("=" * 78)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: "
        f"{torch.version.cuda}"
    )


# ============================================================
# LOAD PART 18 RESULTS
# ============================================================

print()
print("=" * 78)
print("LOADING PART 18 RESULTS")
print("=" * 78)

part18 = pd.read_csv(
    PART18_RESULTS
)

print(
    f"Part 18 cases: "
    f"{len(part18)}"
)


# ============================================================
# LOAD PART 11 RESULTS
# ============================================================

print()
print("=" * 78)
print("LOADING PART 11 RESULTS")
print("=" * 78)

part11 = pd.read_csv(
    PART11_RESULTS
)

print(
    f"Part 11 cases: "
    f"{len(part11)}"
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 78)
print("CREATING SWIN-UNETR")
print("=" * 78)

model = SwinUNETR(
    in_channels=1,
    out_channels=NUM_CLASSES,
    feature_size=FEATURE_SIZE,
    spatial_dims=3,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

print(
    f"Checkpoint epoch: "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Best validation Dice: "
    f"{checkpoint.get('best_val_dice', 'unknown')}"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(
    DEVICE
)

model.eval()

print("✓ Model loaded.")
print("✓ Evaluation mode enabled.")


# ============================================================
# FILE FINDER
# ============================================================

def find_case_file(directory, case_name):

    candidates = []

    for path in directory.iterdir():

        if not path.is_file():
            continue

        stem = path.stem

        if stem == case_name:

            candidates.append(path)

    if len(candidates) == 1:

        return candidates[0]

    # Exact filename stem may not match in some datasets.
    # Try prefix matching.

    for path in directory.iterdir():

        if not path.is_file():
            continue

        if path.stem.startswith(
            case_name
        ):

            candidates.append(path)

    if len(candidates) > 0:

        return candidates[0]

    return None


# ============================================================
# NUMPY LOAD
# ============================================================

def load_array(path):

    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # NumPy
    # --------------------------------------------------------

    if suffix == ".npy":

        return np.load(path)

    # --------------------------------------------------------
    # NumPy compressed
    # --------------------------------------------------------

    if suffix == ".npz":

        data = np.load(path)

        keys = list(data.keys())

        if len(keys) == 0:

            raise ValueError(
                f"Empty NPZ file: {path}"
            )

        return data[keys[0]]

    # --------------------------------------------------------
    # MetaImage (.mha / .mhd)
    # --------------------------------------------------------

    if suffix in [".mha", ".mhd"]:

        try:
            import SimpleITK as sitk

        except ImportError:

            raise ImportError(
                "SimpleITK is required to read .mha/.mhd files. "
                "Install it using: pip install SimpleITK"
            )

        image = sitk.ReadImage(
            str(path)
        )

        array = sitk.GetArrayFromImage(
            image
        )

        return np.asarray(
            array
        )

    # --------------------------------------------------------
    # Unsupported format
    # --------------------------------------------------------

    raise ValueError(
        f"Unsupported file format: {path}"
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):

    image = image.astype(
        np.float32
    )

    valid = image[
        np.isfinite(image)
    ]

    if valid.size == 0:

        return np.zeros_like(
            image,
            dtype=np.float32
        )

    low = np.percentile(
        valid,
        1
    )

    high = np.percentile(
        valid,
        99
    )

    if high <= low:

        return np.zeros_like(
            image,
            dtype=np.float32
        )

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

    return image.astype(
        np.float32
    )


# ============================================================
# CENTER CROP / PAD
# ============================================================

def center_crop_or_pad(
    volume,
    target_shape=PATCH_SIZE,
):

    result = np.zeros(
        target_shape,
        dtype=volume.dtype
    )

    source_slices = []
    target_slices = []

    for source_size, target_size in zip(
        volume.shape,
        target_shape
    ):

        if source_size >= target_size:

            source_start = (
                source_size
                - target_size
            ) // 2

            source_end = (
                source_start
                + target_size
            )

            source_slices.append(
                slice(
                    source_start,
                    source_end
                )
            )

            target_slices.append(
                slice(
                    0,
                    target_size
                )
            )

        else:

            target_start = (
                target_size
                - source_size
            ) // 2

            target_end = (
                target_start
                + source_size
            )

            source_slices.append(
                slice(
                    0,
                    source_size
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
    ] = volume[
        tuple(source_slices)
    ]

    return result


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_image(
    image
):

    image = normalize_image(
        image
    )

    original_shape = image.shape

    image = center_crop_or_pad(
        image,
        PATCH_SIZE
    )

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    return (
        tensor,
        original_shape,
        image
    )


# ============================================================
# PREPROCESS MASK
# ============================================================

def preprocess_mask(
    mask
):

    mask = mask.astype(
        np.int64
    )

    processed = center_crop_or_pad(
        mask,
        PATCH_SIZE
    )

    return processed


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict(
    image_tensor
):

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
        .astype(np.int64)
    )

    return prediction


# ============================================================
# SLICE SELECTION
# ============================================================

def select_best_slice(
    image,
    target,
    prediction
):

    target_spinal = (
        target
        == SPINAL_CANAL_LABEL
    )

    prediction_spinal = (
        prediction
        == SPINAL_CANAL_LABEL
    )

    # Calculate amount of spinal canal
    # in each axial slice.

    target_counts = (
        target_spinal
        .sum(axis=(0, 1))
    )

    prediction_counts = (
        prediction_spinal
        .sum(axis=(0, 1))
    )

    combined_counts = (
        target_counts
        + prediction_counts
    )

    if np.max(
        combined_counts
    ) > 0:

        return int(
            np.argmax(
                combined_counts
            )
        )

    # Fallback: choose middle slice.

    return image.shape[2] // 2


# ============================================================
# METRICS
# ============================================================

def calculate_binary_metrics(
    target,
    prediction
):

    target_mask = (
        target
        == SPINAL_CANAL_LABEL
    )

    prediction_mask = (
        prediction
        == SPINAL_CANAL_LABEL
    )

    tp = int(
        np.logical_and(
            target_mask,
            prediction_mask
        ).sum()
    )

    fp = int(
        np.logical_and(
            ~target_mask,
            prediction_mask
        ).sum()
    )

    fn = int(
        np.logical_and(
            target_mask,
            ~prediction_mask
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
            2 * tp
        ) / denominator

    iou_denominator = (
        tp
        + fp
        + fn
    )

    if iou_denominator == 0:

        iou = 1.0

    else:

        iou = (
            tp
        ) / iou_denominator

    precision_denominator = (
        tp + fp
    )

    if precision_denominator == 0:

        precision = 0.0

    else:

        precision = (
            tp
        ) / precision_denominator

    recall_denominator = (
        tp + fn
    )

    if recall_denominator == 0:

        recall = 0.0

    else:

        recall = (
            tp
        ) / recall_denominator

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# ERROR MAP
# ============================================================

def create_error_map(
    target_slice,
    prediction_slice
):

    target_mask = (
        target_slice
        == SPINAL_CANAL_LABEL
    )

    prediction_mask = (
        prediction_slice
        == SPINAL_CANAL_LABEL
    )

    tp = (
        target_mask
        & prediction_mask
    )

    fp = (
        ~target_mask
        & prediction_mask
    )

    fn = (
        target_mask
        & ~prediction_mask
    )

    error_map = np.zeros(
        target_slice.shape,
        dtype=np.uint8
    )

    # 1 = TP
    # 2 = FP
    # 3 = FN

    error_map[tp] = 1
    error_map[fp] = 2
    error_map[fn] = 3

    return error_map


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    case_name,
    image,
    target,
    prediction,
    selected_slice,
    metrics,
    group
):

    image_slice = image[
        :,
        :,
        selected_slice
    ]

    target_slice = target[
        :,
        :,
        selected_slice
    ]

    prediction_slice = prediction[
        :,
        :,
        selected_slice
    ]

    error_map = create_error_map(
        target_slice,
        prediction_slice
    )

    target_spinal = (
        target_slice
        == SPINAL_CANAL_LABEL
    )

    prediction_spinal = (
        prediction_slice
        == SPINAL_CANAL_LABEL
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5)
    )

    # MRI
    axes[0].imshow(
        image_slice.T,
        cmap="gray",
        origin="lower"
    )

    axes[0].set_title(
        "MRI"
    )

    # Ground truth
    axes[1].imshow(
        image_slice.T,
        cmap="gray",
        origin="lower"
    )

    axes[1].contour(
        target_spinal.T,
        levels=[0.5],
        linewidths=1.5
    )

    axes[1].set_title(
        "Ground Truth\nSpinal Canal"
    )

    # Prediction
    axes[2].imshow(
        image_slice.T,
        cmap="gray",
        origin="lower"
    )

    axes[2].contour(
        prediction_spinal.T,
        levels=[0.5],
        linewidths=1.5
    )

    axes[2].set_title(
        "Swin-UNETR\nPrediction"
    )

    # Error map
    axes[3].imshow(
        error_map.T,
        origin="lower",
        interpolation="nearest"
    )

    axes[3].set_title(
        "Error Map\n"
        "1=TP  2=FP  3=FN"
    )

    # Overlay
    axes[4].imshow(
        image_slice.T,
        cmap="gray",
        origin="lower"
    )

    axes[4].contour(
        target_spinal.T,
        levels=[0.5],
        linewidths=1.5
    )

    axes[4].contour(
        prediction_spinal.T,
        levels=[0.5],
        linewidths=1.5
    )

    axes[4].set_title(
        "GT + Prediction"
    )

    for ax in axes:

        ax.axis("off")

    fig.suptitle(
        (
            f"{case_name} | "
            f"T2 SPACE | "
            f"{group} | "
            f"Dice={metrics['dice']:.4f}"
        ),
        fontsize=14
    )

    fig.tight_layout()

    case_dir = (
        OUTPUT_DIR
        / case_name
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        case_dir
        / f"{case_name}_visual_analysis.png"
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# PROCESS CASES
# ============================================================

print()
print("=" * 78)
print("PROCESSING SELECTED T2 SPACE CASES")
print("=" * 78)

results = []

for index, case_name in enumerate(
    ALL_CASES,
    start=1
):

    print()
    print(
        f"[{index}/{len(ALL_CASES)}] "
        f"{case_name}"
    )

    image_path = find_case_file(
        IMAGE_DIR,
        case_name
    )

    mask_path = find_case_file(
        MASK_DIR,
        case_name
    )

    if image_path is None:

        print(
            f"WARNING: image not found "
            f"for {case_name}"
        )

        continue

    if mask_path is None:

        print(
            f"WARNING: mask not found "
            f"for {case_name}"
        )

        continue

    print(
        f"Image: {image_path.name}"
    )

    print(
        f"Mask : {mask_path.name}"
    )

    image = load_array(
        image_path
    )

    mask = load_array(
        mask_path
    )

    print(
        f"Original image shape: "
        f"{image.shape}"
    )

    print(
        f"Original mask shape: "
        f"{mask.shape}"
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_tensor, original_shape, processed_image = (
        preprocess_image(
            image
        )
    )

    processed_mask = preprocess_mask(
        mask
    )

    print(
        f"Processed shape: "
        f"{processed_image.shape}"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    prediction = predict(
        image_tensor
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_binary_metrics(
        processed_mask,
        prediction
    )

    # --------------------------------------------------------
    # Selected slice
    # --------------------------------------------------------

    selected_slice = select_best_slice(
        processed_image,
        processed_mask,
        prediction
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
        f"Precision: "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"TP: {metrics['tp']}"
    )

    print(
        f"FP: {metrics['fp']}"
    )

    print(
        f"FN: {metrics['fn']}"
    )

    if case_name in FAILURE_CASES:

        group = "Failure"

    else:

        group = "Successful"

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    output_path = create_visualization(
        case_name,
        processed_image,
        processed_mask,
        prediction,
        selected_slice,
        metrics,
        group
    )

    print(
        f"Saved: {output_path}"
    )

    results.append({
        "file": case_name,
        "group": group,
        "original_shape":
            str(original_shape),
        "processed_shape":
            str(processed_image.shape),
        "selected_slice":
            selected_slice,
        "dice":
            metrics["dice"],
        "iou":
            metrics["iou"],
        "precision":
            metrics["precision"],
        "recall":
            metrics["recall"],
        "tp":
            metrics["tp"],
        "fp":
            metrics["fp"],
        "fn":
            metrics["fn"],
        "visualization":
            str(output_path),
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

print()
print("=" * 78)
print("SAVING VISUAL INSPECTION RESULTS")
print("=" * 78)

results_df = pd.DataFrame(
    results
)

summary_csv = (
    OUTPUT_DIR
    / "t2_space_visual_inspection_summary.csv"
)

results_df.to_csv(
    summary_csv,
    index=False
)

print(
    f"Saved: {summary_csv}"
)


# ============================================================
# FAILURE VS SUCCESS SUMMARY
# ============================================================

if len(results_df) > 0:

    print()
    print("=" * 78)
    print("VISUAL INSPECTION SUMMARY")
    print("=" * 78)

    for group_name, group in (
        results_df.groupby("group")
    ):

        print()
        print(group_name)

        print(
            f"Cases: {len(group)}"
        )

        print(
            f"Mean Dice: "
            f"{group['dice'].mean():.6f}"
        )

        print(
            f"Mean Precision: "
            f"{group['precision'].mean():.6f}"
        )

        print(
            f"Mean Recall: "
            f"{group['recall'].mean():.6f}"
        )


# ============================================================
# JSON SUMMARY
# ============================================================

summary = {
    "phase":
        "Phase 3 - Part 19",

    "purpose":
        "T2 SPACE Failure Case Visual Inspection",

    "failure_cases":
        FAILURE_CASES,

    "successful_cases":
        SUCCESS_CASES,

    "processed_cases":
        int(len(results_df)),

    "results":
        results,

    "output_directory":
        str(OUTPUT_DIR),
}


json_path = (
    OUTPUT_DIR
    / "t2_space_visual_inspection_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        default=str
    )

print(
    f"Saved: {json_path}"
)


# ============================================================
# REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "phase3_part19_t2_space_visual_inspection_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 78
        + "\n"
    )

    f.write(
        "PHASE 3 - PART 19\n"
    )

    f.write(
        "T2 SPACE FAILURE CASE VISUAL INSPECTION\n"
    )

    f.write(
        "=" * 78
        + "\n\n"
    )

    f.write(
        "OBJECTIVE\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    f.write(
        "The objective of Part 19 is to qualitatively inspect "
        "representative T2 SPACE failure and successful cases "
        "using the same Swin-UNETR checkpoint used for official "
        "test evaluation.\n\n"
    )

    f.write(
        "FAILURE CASES\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    for case in FAILURE_CASES:

        f.write(
            f"{case}\n"
        )

    f.write(
        "\nSUCCESSFUL CASES\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    for case in SUCCESS_CASES:

        f.write(
            f"{case}\n"
        )

    f.write(
        "\nCASE RESULTS\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    if len(results_df) > 0:

        f.write(
            results_df.to_string(
                index=False
            )
        )

    f.write(
        "\n\nINTERPRETATION\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    f.write(
        "The visual inspection should be interpreted together "
        "with Parts 11-18. The purpose is to identify whether "
        "poor T2 SPACE performance is primarily associated with "
        "localization error, under-segmentation, over-segmentation, "
        "or other visible differences between the prediction and "
        "ground-truth spinal canal.\n\n"
    )

    f.write(
        "The visual inspection does not by itself establish the "
        "cause of segmentation failure. It provides qualitative "
        "evidence that can guide future model-improvement "
        "experiments.\n"
    )

print(
    f"Saved: {report_path}"
)


# ============================================================
# COMPLETION
# ============================================================

elapsed = (
    time.time()
    - start_time
)

print()
print("=" * 78)
print("PART 19 COMPLETE")
print("=" * 78)

print(
    f"Cases processed: "
    f"{len(results_df)}"
)

print(
    f"Execution time: "
    f"{elapsed / 60:.2f} minutes"
)

print()
print("=" * 78)
print("OUTPUT DIRECTORY")
print("=" * 78)

print(
    OUTPUT_DIR
)

print()
print("=" * 78)
print("PHASE 3 - PART 19 COMPLETE")
print("=" * 78)