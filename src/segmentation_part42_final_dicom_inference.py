"""
PART 4.2
FINAL SWIN-UNETR DICOM INFERENCE

Final selected model:
    Part 3.3 Balanced LFNN/RFNN Symmetry Refinement

Checkpoint:
    outputs/segmentation/
    rsna_part33_balanced_symmetry_refinement/
    checkpoints/
    part33_best_development_macro.pth

Purpose
-------
Run the final selected Swin-UNETR model on an RSNA DICOM series
using the same physical-space/canonical preprocessing used during
Part 3.x training.

The script produces:
    - 6-class probability volume
    - predicted class volume
    - disease probability summaries
    - top predicted voxels for each disease
    - central-slice visualizations
    - JSON/CSV/NPZ outputs

IMPORTANT
---------
The model was trained with point supervision rather than manual
voxel masks.

Therefore:
    - model output is called disease localization/probability
    - voxel predictions are NOT claimed to be validated segmentation
    - no fake disease masks are generated
    - no dashboard is modified

Usage
-----
Interactive:
    python src/segmentation_part42_final_dicom_inference.py

The script asks for:
    study_id
    series_id

The selected series must exist in the RSNA manifest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)


# ============================================================================
# PART 2.20B GEOMETRY PIPELINE
# ============================================================================

import segmentation_rsna_part220b_geometry_corrected_training as part220b


# ============================================================================
# FINAL CHECKPOINT
# ============================================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part33_balanced_symmetry_refinement"
    / "checkpoints"
    / "part33_best_development_macro.pth"
)


# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part42_final_dicom_inference"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# MODEL CLASSES
# ============================================================================

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# MODEL
# ============================================================================

def build_final_model():

    print()
    print("=" * 80)
    print("LOADING FINAL SWIN-UNETR")
    print("=" * 80)

    print(
        f"Checkpoint:\n{CHECKPOINT}"
    )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            "Final Part 3.3 checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    model = part220b.build_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print(
        f"Missing keys: "
        f"{len(missing)}"
    )

    print(
        f"Unexpected keys: "
        f"{len(unexpected)}"
    )

    if missing or unexpected:

        raise RuntimeError(
            "Final checkpoint did not load cleanly."
        )

    model = (
        model
        .to(DEVICE)
        .eval()
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "Device: CPU"
        )

    return model


# ============================================================================
# MANIFEST
# ============================================================================

def load_manifest():

    manifest = (
        part220b.load_manifest()
    )

    print(
        f"Manifest rows: "
        f"{len(manifest)}"
    )

    return manifest


# ============================================================================
# FIND SERIES
# ============================================================================

def find_series(
    manifest,
    study_id,
    series_id,
):

    study_id = str(
        study_id
    )

    series_id = str(
        series_id
    )

    subset = manifest[
        (
            manifest["study_id"]
            .astype(str)
            == study_id
        )
        &
        (
            manifest["series_id"]
            .astype(str)
            == series_id
        )
    ].copy()

    if subset.empty:

        raise ValueError(
            "No annotated RSNA series found for:\n"
            f"study_id={study_id}\n"
            f"series_id={series_id}"
        )

    return subset


# ============================================================================
# LOAD CANONICAL CASE
# ============================================================================

def load_canonical_case(
    study_id,
    series_id,
    point_df,
):

    print()
    print(
        "Loading DICOM series through "
        "Part 2.20B physical-space pipeline..."
    )

    result = part220b.load_case(
        study_id,
        series_id,
        point_df,
    )

    if len(result) == 3:

        image, points, geometry = result

    elif len(result) == 4:

        image, points, geometry, _extra = result

    else:

        raise ValueError(
            "Unexpected load_case return length: "
            f"{len(result)}"
        )

    return (
        image,
        points,
        geometry,
    )


# ============================================================================
# IMAGE TENSOR
# ============================================================================

def image_to_tensor(
    image,
):

    if torch.is_tensor(image):

        tensor = image.float()

    else:

        tensor = torch.tensor(
            image,
            dtype=torch.float32,
        )

    # Expected final canonical image:
    #
    # D x H x W
    #
    # Convert to:
    #
    # 1 x 1 x D x H x W

    if tensor.ndim == 3:

        tensor = (
            tensor
            .unsqueeze(0)
            .unsqueeze(0)
        )

    elif tensor.ndim == 4:

        tensor = tensor.unsqueeze(0)

    else:

        raise ValueError(
            "Unexpected canonical image shape: "
            f"{tuple(tensor.shape)}"
        )

    return tensor.to(
        DEVICE
    )


# ============================================================================
# MODEL INFERENCE
# ============================================================================

@torch.no_grad()
def run_inference(
    model,
    image_tensor,
):

    print()
    print(
        "Running Swin-UNETR inference..."
    )

    if DEVICE.type == "cuda":

        with torch.amp.autocast(
            "cuda",
            enabled=True,
        ):

            output = model(
                image_tensor
            )

    else:

        output = model(
            image_tensor
        )

    if isinstance(
        output,
        (tuple, list),
    ):

        logits = output[0]

    else:

        logits = output

    if logits.ndim != 5:

        raise RuntimeError(
            "Expected 5D model output "
            "(B,C,D,H,W), received: "
            f"{tuple(logits.shape)}"
        )

    probabilities = torch.softmax(
        logits.float(),
        dim=1,
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1,
    )

    return (
        logits,
        probabilities,
        predicted_class,
    )


# ============================================================================
# SUMMARY
# ============================================================================

def probability_summary(
    probabilities,
):

    # probabilities:
    # 1 x 6 x D x H x W

    result = []

    for class_id in range(6):

        class_probability = (
            probabilities[
                0,
                class_id,
            ]
            .detach()
            .cpu()
            .numpy()
        )

        result.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    CLASS_NAMES[
                        class_id
                    ],

                "mean_probability":
                    float(
                        class_probability.mean()
                    ),

                "max_probability":
                    float(
                        class_probability.max()
                    ),

                "p99_probability":
                    float(
                        np.percentile(
                            class_probability,
                            99,
                        )
                    ),

                "voxels_above_0.50":
                    int(
                        (
                            class_probability
                            >= 0.50
                        ).sum()
                    ),

                "voxels_above_0.75":
                    int(
                        (
                            class_probability
                            >= 0.75
                        ).sum()
                    ),

                "voxels_above_0.90":
                    int(
                        (
                            class_probability
                            >= 0.90
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(
        result
    )


# ============================================================================
# TOP VOXELS
# ============================================================================

def get_top_voxels(
    probabilities,
    top_k=100,
):

    records = []

    probability_array = (
        probabilities
        .detach()
        .cpu()
        .numpy()
        [0]
    )

    for class_id in range(
        1,
        6,
    ):

        class_map = (
            probability_array[
                class_id
            ]
        )

        flat = class_map.reshape(
            -1
        )

        k = min(
            top_k,
            flat.size,
        )

        indices = np.argpartition(
            flat,
            -k,
        )[-k:]

        indices = indices[
            np.argsort(
                flat[indices]
            )[::-1]
        ]

        depth, height, width = (
            class_map.shape
        )

        for rank, flat_index in enumerate(
            indices,
            start=1,
        ):

            z, y, x = np.unravel_index(
                int(flat_index),
                (
                    depth,
                    height,
                    width,
                ),
            )

            records.append(
                {
                    "class_id":
                        class_id,

                    "class_name":
                        CLASS_NAMES[
                            class_id
                        ],

                    "rank":
                        rank,

                    "z":
                        int(z),

                    "y":
                        int(y),

                    "x":
                        int(x),

                    "probability":
                        float(
                            flat[
                                flat_index
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================================
# CENTRAL SLICE VISUALIZATION
# ============================================================================

def save_central_slices(
    image,
    probabilities,
    predicted_class,
    output_dir,
):

    image_np = (
        image
        .detach()
        .cpu()
        .numpy()
    )

    if image_np.ndim == 5:

        image_np = image_np[
            0,
            0,
        ]

    elif image_np.ndim == 4:

        image_np = image_np[
            0
        ]

    prediction_np = (
        predicted_class
        .detach()
        .cpu()
        .numpy()
    )

    if prediction_np.ndim == 4:

        prediction_np = prediction_np[
            0
        ]

    depth, height, width = (
        image_np.shape
    )

    z = depth // 2
    y = height // 2
    x = width // 2

    # ------------------------------------------------------------
    # Native canonical slices
    # ------------------------------------------------------------

    axial = image_np[
        z,
        :,
        :,
    ]

    sagittal = image_np[
        :,
        :,
        x,
    ]

    coronal = image_np[
        :,
        y,
        :,
    ]

    fig = plt.figure(
        figsize=(
            15,
            5,
        )
    )

    ax1 = fig.add_subplot(
        1,
        3,
        1,
    )

    ax1.imshow(
        axial,
        cmap="gray",
    )

    ax1.set_title(
        f"Axial canonical z={z}"
    )

    ax1.axis(
        "off"
    )

    ax2 = fig.add_subplot(
        1,
        3,
        2,
    )

    ax2.imshow(
        sagittal,
        cmap="gray",
    )

    ax2.set_title(
        f"Sagittal canonical x={x}"
    )

    ax2.axis(
        "off"
    )

    ax3 = fig.add_subplot(
        1,
        3,
        3,
    )

    ax3.imshow(
        coronal,
        cmap="gray",
    )

    ax3.set_title(
        f"Coronal canonical y={y}"
    )

    ax3.axis(
        "off"
    )

    fig.suptitle(
        "Part 3.3 Swin-UNETR Canonical MRI View",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "canonical_central_views.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ------------------------------------------------------------
    # Predicted-class central views
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(
            15,
            5,
        )
    )

    ax1 = fig.add_subplot(
        1,
        3,
        1,
    )

    ax1.imshow(
        prediction_np[
            z,
            :,
            :,
        ],
        interpolation="nearest",
    )

    ax1.set_title(
        f"Predicted class — axial z={z}"
    )

    ax1.axis(
        "off"
    )

    ax2 = fig.add_subplot(
        1,
        3,
        2,
    )

    ax2.imshow(
        prediction_np[
            :,
            :,
            x,
        ],
        interpolation="nearest",
    )

    ax2.set_title(
        f"Predicted class — sagittal x={x}"
    )

    ax2.axis(
        "off"
    )

    ax3 = fig.add_subplot(
        1,
        3,
        3,
    )

    ax3.imshow(
        prediction_np[
            :,
            y,
            :,
        ],
        interpolation="nearest",
    )

    ax3.set_title(
        f"Predicted class — coronal y={y}"
    )

    ax3.axis(
        "off"
    )

    fig.suptitle(
        "Predicted Disease Classes",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "predicted_class_central_views.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print(
        "PART 4.2"
    )
    print(
        "FINAL SWIN-UNETR DICOM INFERENCE"
    )
    print("=" * 80)

    print()
    print(
        "Model: Part 3.3"
    )

    print(
        "Task: point-supervised disease localization"
    )

    print(
        "Voxel-level segmentation claim: DISABLED"
    )

    # ------------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------------

    model = build_final_model()

    # ------------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------------

    manifest = load_manifest()

    # ------------------------------------------------------------------------
    # User input
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 80
    )
    print(
        "RSNA DICOM SERIES SELECTION"
    )
    print(
        "=" * 80
    )

    study_id = input(
        "Enter study_id: "
    ).strip()

    series_id = input(
        "Enter series_id: "
    ).strip()

    # ------------------------------------------------------------------------
    # Find series
    # ------------------------------------------------------------------------

    point_df = find_series(
        manifest,
        study_id,
        series_id,
    )

    print()
    print(
        f"Study ID: "
        f"{study_id}"
    )

    print(
        f"Series ID: "
        f"{series_id}"
    )

    print(
        f"Annotation rows: "
        f"{len(point_df)}"
    )

    if (
        "series_description"
        in point_df.columns
    ):

        descriptions = (
            point_df[
                "series_description"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        print(
            "Series description: "
            f"{descriptions}"
        )

    # ------------------------------------------------------------------------
    # Load canonical DICOM volume
    # ------------------------------------------------------------------------

    image, points, geometry = (
        load_canonical_case(
            study_id,
            series_id,
            point_df,
        )
    )

    print()
    print(
        "Canonical image shape:"
        f" {tuple(image.shape)}"
    )

    # ------------------------------------------------------------------------
    # Tensor
    # ------------------------------------------------------------------------

    image_tensor = image_to_tensor(
        image
    )

    print(
        "Model input shape:"
        f" {tuple(image_tensor.shape)}"
    )

    # ------------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------------

    (
        logits,
        probabilities,
        predicted_class,
    ) = run_inference(
        model,
        image_tensor,
    )

    print(
        "Model output shape:"
        f" {tuple(logits.shape)}"
    )

    # ------------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------------

    case_name = (
        f"study_{study_id}"
        f"_series_{series_id}"
    )

    case_dir = (
        OUTPUT_ROOT
        / case_name
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    summary_df = probability_summary(
        probabilities
    )

    print()
    print(
        "=" * 80
    )

    print(
        "DISEASE PROBABILITY SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    summary_df.to_csv(
        case_dir
        / "disease_probability_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Top voxels
    # ------------------------------------------------------------------------

    top_voxels = get_top_voxels(
        probabilities,
        top_k=100,
    )

    top_voxels.to_csv(
        case_dir
        / "top_disease_probability_voxels.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Save arrays
    # ------------------------------------------------------------------------

    probability_np = (
        probabilities
        .detach()
        .cpu()
        .numpy()
        [0]
        .astype(
            np.float32
        )
    )

    logits_np = (
        logits
        .detach()
        .cpu()
        .numpy()
        [0]
        .astype(
            np.float32
        )
    )

    prediction_np = (
        predicted_class
        .detach()
        .cpu()
        .numpy()
        [0]
        .astype(
            np.uint8
        )
    )

    image_np = (
        image_tensor
        .detach()
        .cpu()
        .numpy()
        [0, 0]
        .astype(
            np.float32
        )
    )

    np.savez_compressed(
        case_dir
        / "swinunetr_inference.npz",
        image=image_np,
        logits=logits_np,
        probabilities=probability_np,
        predicted_class=prediction_np,
    )

    # ------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------

    save_central_slices(
        image_tensor,
        probabilities,
        predicted_class,
        case_dir,
    )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    metadata = {
        "model":
            "Swin-UNETR",

        "model_variant":
            "Part 3.3 Balanced LFNN/RFNN Symmetry Refinement",

        "checkpoint":
            str(CHECKPOINT),

        "study_id":
            study_id,

        "series_id":
            series_id,

        "canonical_image_shape":
            list(
                image_np.shape
            ),

        "probability_shape":
            list(
                probability_np.shape
            ),

        "num_classes":
            6,

        "classes":
            CLASS_NAMES,

        "task":
            "point-supervised disease localization",

        "voxel_segmentation_validated":
            False,

        "manual_voxel_ground_truth_available":
            False,

        "physical_geometry_available":
            geometry is not None,

        "annotation_rows_used_for_case_loading":
            int(
                len(point_df)
            ),

        "outputs":
            {
                "npz":
                    "swinunetr_inference.npz",

                "summary":
                    "disease_probability_summary.csv",

                "top_voxels":
                    "top_disease_probability_voxels.csv",

                "canonical_views":
                    "canonical_central_views.png",

                "predicted_views":
                    "predicted_class_central_views.png",
            },
    }

    with open(
        case_dir
        / "inference_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "PART 4.2 COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Study:"
        f" {study_id}"
    )

    print(
        "Series:"
        f" {series_id}"
    )

    print(
        "Output directory:"
    )

    print(
        case_dir
    )

    print()
    print(
        "Important:"
    )

    print(
        "The probability volume is a model localization output."
    )

    print(
        "It is NOT claimed as validated voxel segmentation."
    )

    print(
        "Dashboard modified: NO"
    )


if __name__ == "__main__":

    main()