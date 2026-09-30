"""
PHASE 3 - PART 12
QUALITATIVE SWIN-UNETR SEGMENTATION ANALYSIS

Creates visual comparisons between:
    1. MRI
    2. Ground-truth segmentation
    3. Swin-UNETR prediction
    4. Ground-truth overlay
    5. Prediction overlay
    6. Prediction error

Uses the official best_model.pth checkpoint.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
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

PART11_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "test_evaluation"
)

CASE_RESULTS = (
    PART11_DIR
    / "test_case_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "visual_analysis"
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

CLASS_NAMES = {
    0: "Background",
    1: "Vertebrae",
    2: "Spinal Canal",
    3: "Intervertebral Disc",
}

FOREGROUND_CLASSES = [1, 2, 3]

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("PHASE 3 - PART 12")
print("QUALITATIVE SWIN-UNETR SEGMENTATION ANALYSIS")
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
# PATH VALIDATION
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
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )

if not CASE_RESULTS.exists():
    raise FileNotFoundError(
        f"Part 11 results not found:\n{CASE_RESULTS}"
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
# LOAD PART 11 RESULTS
# ============================================================

print("\n" + "=" * 78)
print("LOADING PART 11 RESULTS")
print("=" * 78)

results_df = pd.read_csv(
    CASE_RESULTS
)

required_column = (
    "mean_foreground_dice"
)

if required_column not in results_df.columns:
    raise RuntimeError(
        f"Required column not found: "
        f"{required_column}"
    )

results_df = results_df[
    results_df[
        required_column
    ].notna()
].copy()

print(
    f"Successful test cases: "
    f"{len(results_df)}"
)


# ============================================================
# SELECT REPRESENTATIVE CASES
# ============================================================

best_case = results_df.loc[
    results_df[
        "mean_foreground_dice"
    ].idxmax()
]

worst_case = results_df.loc[
    results_df[
        "mean_foreground_dice"
    ].idxmin()
]

median_value = results_df[
    "mean_foreground_dice"
].median()

median_index = (
    results_df[
        "mean_foreground_dice"
    ]
    .sub(median_value)
    .abs()
    .idxmin()
)

median_case = results_df.loc[
    median_index
]


# ============================================================
# FIND STRUCTURE-SPECIFIC DIFFICULT CASES
# ============================================================

spinal_column = (
    "Spinal Canal_dice"
)

disc_column = (
    "Intervertebral Disc_dice"
)

vertebra_column = (
    "Vertebrae_dice"
)

spinal_case = None
disc_case = None
vertebra_case = None

if spinal_column in results_df.columns:
    spinal_case = results_df.loc[
        results_df[
            spinal_column
        ].idxmin()
    ]

if disc_column in results_df.columns:
    disc_case = results_df.loc[
        results_df[
            disc_column
        ].idxmin()
    ]

if vertebra_column in results_df.columns:
    vertebra_case = results_df.loc[
        results_df[
            vertebra_column
        ].idxmin()
    ]


# ============================================================
# PRINT SELECTED CASES
# ============================================================

print("\n" + "=" * 78)
print("SELECTED REPRESENTATIVE CASES")
print("=" * 78)

print(
    f"Best case   : "
    f"{best_case['file']} "
    f"(Dice = "
    f"{best_case['mean_foreground_dice']:.6f})"
)

print(
    f"Median case : "
    f"{median_case['file']} "
    f"(Dice = "
    f"{median_case['mean_foreground_dice']:.6f})"
)

print(
    f"Worst case  : "
    f"{worst_case['file']} "
    f"(Dice = "
    f"{worst_case['mean_foreground_dice']:.6f})"
)

if spinal_case is not None:
    print(
        f"Lowest spinal canal case: "
        f"{spinal_case['file']} "
        f"(Dice = "
        f"{spinal_case[spinal_column]:.6f})"
    )

if disc_case is not None:
    print(
        f"Lowest disc case: "
        f"{disc_case['file']} "
        f"(Dice = "
        f"{disc_case[disc_column]:.6f})"
    )

if vertebra_case is not None:
    print(
        f"Lowest vertebrae case: "
        f"{vertebra_case['file']} "
        f"(Dice = "
        f"{vertebra_case[vertebra_column]:.6f})"
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

if isinstance(checkpoint, dict):
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
print(
    f"✓ Checkpoint epoch: "
    f"{checkpoint.get('epoch', 'N/A')}"
)

if (
    isinstance(checkpoint, dict)
    and "best_val_dice" in checkpoint
):
    print(
        f"✓ Best validation Dice: "
        f"{checkpoint['best_val_dice']:.6f}"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_mha(path):
    image = sitk.ReadImage(
        str(path)
    )

    array = sitk.GetArrayFromImage(
        image
    )

    return image, array


def normalize_image(array):

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
        1.0
    )

    high = np.percentile(
        nonzero,
        99.0
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

    return np.clip(
        array,
        0,
        1,
    ).astype(
        np.float32
    )


def center_crop_or_pad(
    array,
    target_shape=PATCH_SIZE,
    pad_value=0,
):

    result = np.full(
        target_shape,
        pad_value,
        dtype=array.dtype,
    )

    source_slices = []
    target_slices = []

    for source_size, target_size in zip(
        array.shape,
        target_shape,
    ):

        if source_size >= target_size:

            start = (
                source_size
                - target_size
            ) // 2

            end = (
                start
                + target_size
            )

            source_slices.append(
                slice(start, end)
            )

            target_slices.append(
                slice(0, target_size)
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
                slice(0, source_size)
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


def prepare_image(array):

    array = normalize_image(
        array
    )

    array = center_crop_or_pad(
        array,
        PATCH_SIZE,
        0,
    )

    tensor = torch.from_numpy(
        array
    ).float()

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.unsqueeze(
        0
    )

    return tensor


def prepare_mask(array):

    array = center_crop_or_pad(
        array.astype(
            np.int64
        ),
        PATCH_SIZE,
        0,
    )

    return array


def dice_score(
    prediction,
    target,
    class_id,
):

    pred = (
        prediction == class_id
    )

    true = (
        target == class_id
    )

    intersection = np.logical_and(
        pred,
        true,
    ).sum()

    denominator = (
        pred.sum()
        + true.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection
    ) / denominator


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    case_name,
    case_output_name,
):

    image_path = (
        TEST_IMAGES_DIR
        / f"{case_name}.mha"
    )

    mask_path = (
        TEST_MASKS_DIR
        / f"{case_name}.mha"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found:\n"
            f"{mask_path}"
        )

    _, image_array = load_mha(
        image_path
    )

    _, mask_array = load_mha(
        mask_path
    )

    image_tensor = prepare_image(
        image_array
    ).to(DEVICE)

    target = prepare_mask(
        mask_array
    )

    with torch.no_grad():

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

    # --------------------------------------------------------
    # Determine informative slice
    # --------------------------------------------------------

    foreground = (
        target > 0
    )

    slice_scores = foreground.sum(
        axis=(1, 2)
    )

    if slice_scores.max() > 0:

        representative_slice = int(
            np.argmax(
                slice_scores
            )
        )

    else:

        representative_slice = (
            target.shape[0] // 2
        )

    image_slice = (
        center_crop_or_pad(
            normalize_image(
                image_array
            ),
            PATCH_SIZE,
            0,
        )
        [representative_slice]
    )

    target_slice = target[
        representative_slice
    ]

    prediction_slice = prediction[
        representative_slice
    ]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    dices = {}

    for class_id in FOREGROUND_CLASSES:

        dices[class_id] = dice_score(
            prediction,
            target,
            class_id,
        )

    mean_dice = float(
        np.mean(
            list(
                dices.values()
            )
        )
    )

    # --------------------------------------------------------
    # Error map
    # --------------------------------------------------------

    error = (
        prediction_slice
        != target_slice
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    fig.suptitle(
        (
            f"Swin-UNETR Segmentation Analysis\n"
            f"{case_name} | "
            f"Mean Foreground Dice = "
            f"{mean_dice:.4f}"
        ),
        fontsize=16,
    )

    # MRI
    axes[0, 0].imshow(
        image_slice,
        cmap="gray",
    )

    axes[0, 0].set_title(
        "MRI"
    )

    # Ground truth
    axes[0, 1].imshow(
        target_slice,
        interpolation="nearest",
    )

    axes[0, 1].set_title(
        "Ground Truth"
    )

    # Prediction
    axes[0, 2].imshow(
        prediction_slice,
        interpolation="nearest",
    )

    axes[0, 2].set_title(
        "Swin-UNETR Prediction"
    )

    # Ground truth overlay
    axes[1, 0].imshow(
        image_slice,
        cmap="gray",
    )

    axes[1, 0].imshow(
        target_slice,
        alpha=0.45,
        interpolation="nearest",
    )

    axes[1, 0].set_title(
        "Ground Truth Overlay"
    )

    # Prediction overlay
    axes[1, 1].imshow(
        image_slice,
        cmap="gray",
    )

    axes[1, 1].imshow(
        prediction_slice,
        alpha=0.45,
        interpolation="nearest",
    )

    axes[1, 1].set_title(
        "Prediction Overlay"
    )

    # Error
    axes[1, 2].imshow(
        error,
        interpolation="nearest",
    )

    axes[1, 2].set_title(
        "Prediction Error"
    )

    for ax in axes.ravel():
        ax.axis("off")

    metrics_text = (
        f"Vertebrae Dice: "
        f"{dices[1]:.4f}\n"
        f"Spinal Canal Dice: "
        f"{dices[2]:.4f}\n"
        f"Intervertebral Disc Dice: "
        f"{dices[3]:.4f}"
    )

    fig.text(
        0.5,
        0.015,
        metrics_text,
        ha="center",
        fontsize=12,
    )

    plt.tight_layout(
        rect=(0, 0.04, 1, 0.94)
    )

    output_path = (
        OUTPUT_DIR
        / case_output_name
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )

    return {
        "case": case_name,
        "slice": representative_slice,
        "mean_dice": mean_dice,
        "vertebrae_dice": dices[1],
        "spinal_canal_dice": dices[2],
        "disc_dice": dices[3],
        "output": str(output_path),
    }


# ============================================================
# GENERATE REPRESENTATIVE VISUALIZATIONS
# ============================================================

print("\n" + "=" * 78)
print("GENERATING REPRESENTATIVE VISUALIZATIONS")
print("=" * 78)

selected_cases = []

case_candidates = [
    (
        best_case["file"],
        "best_case.png",
        "Best",
    ),
    (
        median_case["file"],
        "median_case.png",
        "Median",
    ),
    (
        worst_case["file"],
        "worst_case.png",
        "Worst",
    ),
]

if spinal_case is not None:

    candidate = (
        spinal_case["file"],
        "lowest_spinal_canal_case.png",
        "Lowest Spinal Canal",
    )

    if candidate[0] not in [
        x[0]
        for x in case_candidates
    ]:

        case_candidates.append(
            candidate
        )


if disc_case is not None:

    candidate = (
        disc_case["file"],
        "lowest_disc_case.png",
        "Lowest Disc",
    )

    if candidate[0] not in [
        x[0]
        for x in case_candidates
    ]:

        case_candidates.append(
            candidate
        )


visual_results = []

for case_name, filename, label in case_candidates:

    print(
        f"\n{label}: {case_name}"
    )

    result = create_visualization(
        case_name,
        filename,
    )

    visual_results.append(
        result
    )


# ============================================================
# SAVE VISUALIZATION SUMMARY
# ============================================================

visual_df = pd.DataFrame(
    visual_results
)

visual_summary_path = (
    OUTPUT_DIR
    / "visual_analysis_summary.csv"
)

visual_df.to_csv(
    visual_summary_path,
    index=False,
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "phase3_part12_visual_analysis_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "PHASE 3 - PART 12\n"
    )

    f.write(
        "QUALITATIVE SWIN-UNETR "
        "SEGMENTATION ANALYSIS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Checkpoint:\n"
        f"{CHECKPOINT_PATH}\n\n"
    )

    f.write(
        "BEST TEST CASE\n"
    )

    f.write(
        f"{best_case['file']} | "
        f"Dice = "
        f"{best_case['mean_foreground_dice']:.6f}\n\n"
    )

    f.write(
        "MEDIAN TEST CASE\n"
    )

    f.write(
        f"{median_case['file']} | "
        f"Dice = "
        f"{median_case['mean_foreground_dice']:.6f}\n\n"
    )

    f.write(
        "WORST TEST CASE\n"
    )

    f.write(
        f"{worst_case['file']} | "
        f"Dice = "
        f"{worst_case['mean_foreground_dice']:.6f}\n\n"
    )

    f.write(
        "GENERATED VISUALIZATIONS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for result in visual_results:

        f.write(
            f"{result['case']}\n"
        )

        f.write(
            f"  Slice: "
            f"{result['slice']}\n"
        )

        f.write(
            f"  Mean Dice: "
            f"{result['mean_dice']:.6f}\n"
        )

        f.write(
            f"  Vertebrae: "
            f"{result['vertebrae_dice']:.6f}\n"
        )

        f.write(
            f"  Spinal Canal: "
            f"{result['spinal_canal_dice']:.6f}\n"
        )

        f.write(
            f"  Intervertebral Disc: "
            f"{result['disc_dice']:.6f}\n"
        )

        f.write(
            f"  Image: "
            f"{result['output']}\n\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 78)
print("OUTPUT FILES")
print("=" * 78)

print(
    f"Visualizations: "
    f"{OUTPUT_DIR}"
)

print(
    f"Summary CSV: "
    f"{visual_summary_path}"
)

print(
    f"Report: "
    f"{report_path}"
)

print("\n" + "=" * 78)
print("PHASE 3 - PART 12 COMPLETE")
print("=" * 78)