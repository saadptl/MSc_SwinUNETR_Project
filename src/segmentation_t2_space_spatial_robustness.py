"""
PHASE 3 - PART 21
T2 SPACE PREPROCESSING & SPATIAL ROBUSTNESS ANALYSIS

Purpose
-------
Investigate whether center crop / padding to 96x96x96
can contribute to T2 SPACE segmentation failures.

This is an analysis-only script.

IMPORTANT
---------
- No training
- No optimizer
- No checkpoint modification
- No model retraining

The analysis compares original MRI/mask volumes with the
same center-crop/pad strategy used for model inference.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# START
# ============================================================

START_TIME = time.time()


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

PART19_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "t2_space_visual_inspection"
    / "t2_space_visual_inspection_summary.csv"
)

PART20_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "t2_space_quality_analysis"
    / "t2_space_quality_case_analysis.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "t2_space_spatial_robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CASE GROUPS
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
# TARGET SIZE
# ============================================================

TARGET_SHAPE = (
    96,
    96,
    96
)


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("PHASE 3 - PART 21")
print("T2 SPACE PREPROCESSING & SPATIAL ROBUSTNESS ANALYSIS")
print("=" * 78)

print()
print("PROJECT ROOT")
print(PROJECT_ROOT)

print()
print("TEST DIRECTORY")
print(TEST_DIR)

print()
print("IMAGE DIRECTORY")
print(IMAGE_DIR)

print()
print("MASK DIRECTORY")
print(MASK_DIR)

print()
print("TARGET MODEL INPUT")
print(TARGET_SHAPE)

print()
print("OUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================
# VALIDATE
# ============================================================

required_paths = [
    TEST_DIR,
    IMAGE_DIR,
    MASK_DIR,
    PART11_RESULTS,
    PART18_RESULTS,
    PART19_RESULTS,
    PART20_RESULTS,
]

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

print()
print("=" * 78)
print("LOADING PREVIOUS RESULTS")
print("=" * 78)

part11 = pd.read_csv(
    PART11_RESULTS
)

part18 = pd.read_csv(
    PART18_RESULTS
)

part19 = pd.read_csv(
    PART19_RESULTS
)

part20 = pd.read_csv(
    PART20_RESULTS
)

print(
    f"Part 11 cases : {len(part11)}"
)

print(
    f"Part 18 cases : {len(part18)}"
)

print(
    f"Part 19 cases : {len(part19)}"
)

print(
    f"Part 20 cases : {len(part20)}"
)


# ============================================================
# FIND FILE
# ============================================================

def find_case_file(
    directory,
    case_name
):

    for extension in [
        ".mha",
        ".mhd",
        ".npy",
        ".npz",
    ]:

        path = (
            directory
            / (
                case_name
                + extension
            )
        )

        if path.exists():

            return path

    for path in directory.iterdir():

        if not path.is_file():
            continue

        if path.stem == case_name:

            return path

    return None


# ============================================================
# LOAD ARRAY
# ============================================================

def load_array(path):

    suffix = path.suffix.lower()

    if suffix in [
        ".mha",
        ".mhd",
    ]:

        try:

            import SimpleITK as sitk

        except ImportError:

            raise ImportError(
                "SimpleITK is required.\n"
                "Install with:\n"
                "pip install SimpleITK"
            )

        image = sitk.ReadImage(
            str(path)
        )

        return np.asarray(
            sitk.GetArrayFromImage(
                image
            ),
            dtype=np.float32
        )

    if suffix == ".npy":

        return np.asarray(
            np.load(path),
            dtype=np.float32
        )

    if suffix == ".npz":

        data = np.load(path)

        keys = list(
            data.keys()
        )

        if not keys:

            raise ValueError(
                f"Empty NPZ file: {path}"
            )

        return np.asarray(
            data[keys[0]],
            dtype=np.float32
        )

    raise ValueError(
        f"Unsupported file format: {path}"
    )


# ============================================================
# CENTER CROP / PAD
# ============================================================

def center_crop_pad(
    volume,
    target_shape
):
    """
    Apply center crop or symmetric padding
    to obtain target_shape.

    Returns:
        processed
        crop_start
        crop_end
        pad_before
        pad_after
    """

    volume = np.asarray(
        volume
    )

    original_shape = (
        np.array(
            volume.shape,
            dtype=int
        )
    )

    target = (
        np.array(
            target_shape,
            dtype=int
        )
    )

    crop_start = np.zeros(
        3,
        dtype=int
    )

    crop_end = original_shape.copy()

    pad_before = np.zeros(
        3,
        dtype=int
    )

    pad_after = np.zeros(
        3,
        dtype=int
    )

    slices = []

    for axis in range(3):

        current = (
            original_shape[axis]
        )

        desired = (
            target[axis]
        )

        if current > desired:

            excess = (
                current
                - desired
            )

            start = (
                excess
                // 2
            )

            end = (
                start
                + desired
            )

            crop_start[axis] = (
                start
            )

            crop_end[axis] = (
                end
            )

            slices.append(
                slice(
                    start,
                    end
                )
            )

        else:

            crop_start[axis] = 0
            crop_end[axis] = current

            slices.append(
                slice(
                    0,
                    current
                )
            )

            total_pad = (
                desired
                - current
            )

            before = (
                total_pad
                // 2
            )

            after = (
                total_pad
                - before
            )

            pad_before[axis] = (
                before
            )

            pad_after[axis] = (
                after
            )

    cropped = volume[
        tuple(slices)
    ]

    padding = tuple(
        (
            int(
                pad_before[i]
            ),
            int(
                pad_after[i]
            )
        )
        for i in range(3)
    )

    processed = np.pad(
        cropped,
        padding,
        mode="constant",
        constant_values=0
    )

    return (
        processed,
        crop_start,
        crop_end,
        pad_before,
        pad_after
    )


# ============================================================
# NORMALIZATION
# ============================================================

def percentile_normalize(
    image
):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    finite = image[
        np.isfinite(image)
    ]

    if finite.size == 0:

        return np.zeros_like(
            image
        )

    low = np.percentile(
        finite,
        1
    )

    high = np.percentile(
        finite,
        99
    )

    if high <= low:

        return np.zeros_like(
            image
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
# MASK LABEL INSPECTION
# ============================================================

def get_mask_labels(
    mask
):

    labels = np.unique(
        mask
    )

    return [
        int(x)
        for x in labels
    ]


# ============================================================
# BOUNDING BOX
# ============================================================

def get_bbox(
    binary_mask
):
    """
    Returns bounding box in
    min/max coordinates.

    Shape:
        (3, 2)

    [axis, 0] = minimum
    [axis, 1] = maximum
    """

    coordinates = np.argwhere(
        binary_mask
    )

    if coordinates.size == 0:

        return None

    minimum = (
        coordinates.min(
            axis=0
        )
    )

    maximum = (
        coordinates.max(
            axis=0
        )
    )

    return np.stack(
        [
            minimum,
            maximum,
        ],
        axis=1
    )


# ============================================================
# TRANSFORM BBOX TO PROCESSED SPACE
# ============================================================

def transform_bbox_to_processed(
    bbox,
    crop_start,
    pad_before,
    target_shape
):

    if bbox is None:

        return None

    processed_bbox = (
        bbox.astype(
            np.int64
        )
        - crop_start.reshape(
            3,
            1
        )
        + pad_before.reshape(
            3,
            1
        )
    )

    lower = (
        processed_bbox[
            :,
            0
        ]
    )

    upper = (
        processed_bbox[
            :,
            1
        ]
    )

    target = np.array(
        target_shape,
        dtype=int
    )

    # Clamp to processed volume

    lower = np.maximum(
        lower,
        0
    )

    upper = np.minimum(
        upper,
        target - 1
    )

    if np.any(
        lower > upper
    ):

        return None

    return np.stack(
        [
            lower,
            upper,
        ],
        axis=1
    )


# ============================================================
# RETENTION CALCULATION
# ============================================================

def calculate_retention(
    original_mask,
    processed_mask,
    label
):

    original_count = int(
        np.sum(
            original_mask == label
        )
    )

    processed_count = int(
        np.sum(
            processed_mask == label
        )
    )

    if original_count > 0:

        retention = (
            processed_count
            / original_count
        )

    else:

        retention = np.nan

    return (
        original_count,
        processed_count,
        retention
    )


# ============================================================
# BOUNDARY DISTANCE
# ============================================================

def calculate_boundary_distances(
    bbox,
    shape
):

    if bbox is None:

        return {

            "min_distance_boundary":
                np.nan,

            "distance_axis0_min":
                np.nan,

            "distance_axis0_max":
                np.nan,

            "distance_axis1_min":
                np.nan,

            "distance_axis1_max":
                np.nan,

            "distance_axis2_min":
                np.nan,

            "distance_axis2_max":
                np.nan,
        }

    shape = np.array(
        shape,
        dtype=int
    )

    lower = bbox[
        :,
        0
    ]

    upper = bbox[
        :,
        1
    ]

    distances = [

        int(
            lower[0]
        ),

        int(
            shape[0]
            - 1
            - upper[0]
        ),

        int(
            lower[1]
        ),

        int(
            shape[1]
            - 1
            - upper[1]
        ),

        int(
            lower[2]
        ),

        int(
            shape[2]
            - 1
            - upper[2]
        ),
    ]

    return {

        "min_distance_boundary":
            float(
                min(distances)
            ),

        "distance_axis0_min":
            distances[0],

        "distance_axis0_max":
            distances[1],

        "distance_axis1_min":
            distances[2],

        "distance_axis1_max":
            distances[3],

        "distance_axis2_min":
            distances[4],

        "distance_axis2_max":
            distances[5],
    }


# ============================================================
# CASE PROCESSING
# ============================================================

def process_case(
    case_name,
    group
):

    print()
    print("-" * 78)
    print(
        f"PROCESSING {case_name}"
    )
    print("-" * 78)

    image_path = find_case_file(
        IMAGE_DIR,
        case_name
    )

    mask_path = find_case_file(
        MASK_DIR,
        case_name
    )

    if image_path is None:

        raise FileNotFoundError(
            f"Image not found: "
            f"{case_name}"
        )

    if mask_path is None:

        raise FileNotFoundError(
            f"Mask not found: "
            f"{case_name}"
        )

    image = load_array(
        image_path
    )

    mask = load_array(
        mask_path
    )

    if image.shape != mask.shape:

        raise ValueError(
            f"Image/mask shape mismatch "
            f"for {case_name}: "
            f"{image.shape} vs "
            f"{mask.shape}"
        )

    original_shape = (
        np.array(
            image.shape,
            dtype=int
        )
    )

    print(
        "Original shape:",
        tuple(
            original_shape
        )
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = (
        percentile_normalize(
            image
        )
    )

    # --------------------------------------------------------
    # Crop/pad image
    # --------------------------------------------------------

    (
        processed_image,
        crop_start,
        crop_end,
        pad_before,
        pad_after
    ) = center_crop_pad(
        normalized,
        TARGET_SHAPE
    )

    # --------------------------------------------------------
    # Crop/pad mask using same transformation
    # --------------------------------------------------------

    (
        processed_mask,
        _,
        _,
        _,
        _
    ) = center_crop_pad(
        mask,
        TARGET_SHAPE
    )

    print(
        "Processed shape:",
        processed_image.shape
    )

    print(
        "Crop start:",
        crop_start.tolist()
    )

    print(
        "Crop end:",
        crop_end.tolist()
    )

    print(
        "Pad before:",
        pad_before.tolist()
    )

    print(
        "Pad after:",
        pad_after.tolist()
    )

    # --------------------------------------------------------
    # Foreground
    # --------------------------------------------------------

    original_foreground = (
        mask > 0
    )

    processed_foreground = (
        processed_mask > 0
    )

    original_foreground_count = int(
        original_foreground.sum()
    )

    processed_foreground_count = int(
        processed_foreground.sum()
    )

    if (
        original_foreground_count
        > 0
    ):

        foreground_retention = (
            processed_foreground_count
            / original_foreground_count
        )

    else:

        foreground_retention = np.nan

    # --------------------------------------------------------
    # Spinal canal
    # --------------------------------------------------------

    (
        original_canal_count,
        processed_canal_count,
        canal_retention
    ) = calculate_retention(
        mask,
        processed_mask,
        2
    )

    # --------------------------------------------------------
    # Vertebrae
    # --------------------------------------------------------

    (
        original_vertebrae_count,
        processed_vertebrae_count,
        vertebrae_retention
    ) = calculate_retention(
        mask,
        processed_mask,
        1
    )

    # --------------------------------------------------------
    # Disc
    # --------------------------------------------------------

    (
        original_disc_count,
        processed_disc_count,
        disc_retention
    ) = calculate_retention(
        mask,
        processed_mask,
        3
    )

    # --------------------------------------------------------
    # Bounding boxes
    # --------------------------------------------------------

    canal_bbox_original = get_bbox(
        mask == 2
    )

    canal_bbox_processed = (
        transform_bbox_to_processed(
            canal_bbox_original,
            crop_start,
            pad_before,
            TARGET_SHAPE
        )
    )

    foreground_bbox_original = get_bbox(
        mask > 0
    )

    foreground_bbox_processed = (
        transform_bbox_to_processed(
            foreground_bbox_original,
            crop_start,
            pad_before,
            TARGET_SHAPE
        )
    )

    vertebrae_bbox_original = get_bbox(
        mask == 1
    )

    vertebrae_bbox_processed = (
        transform_bbox_to_processed(
            vertebrae_bbox_original,
            crop_start,
            pad_before,
            TARGET_SHAPE
        )
    )

    disc_bbox_original = get_bbox(
        mask == 3
    )

    disc_bbox_processed = (
        transform_bbox_to_processed(
            disc_bbox_original,
            crop_start,
            pad_before,
            TARGET_SHAPE
        )
    )

    # --------------------------------------------------------
    # Boundary distances
    # --------------------------------------------------------

    canal_boundary = (
        calculate_boundary_distances(
            canal_bbox_processed,
            TARGET_SHAPE
        )
    )

    foreground_boundary = (
        calculate_boundary_distances(
            foreground_bbox_processed,
            TARGET_SHAPE
        )
    )

    # --------------------------------------------------------
    # Original canal percentage
    # --------------------------------------------------------

    total_original_voxels = int(
        np.prod(
            original_shape
        )
    )

    total_processed_voxels = int(
        np.prod(
            TARGET_SHAPE
        )
    )

    original_canal_percentage = (
        100.0
        * original_canal_count
        / total_original_voxels
    )

    processed_canal_percentage = (
        100.0
        * processed_canal_count
        / total_processed_voxels
    )

    # --------------------------------------------------------
    # Cropping information
    # --------------------------------------------------------

    crop_amount = (
        original_shape
        - np.minimum(
            original_shape,
            np.array(
                TARGET_SHAPE
            )
        )
    )

    # Total voxels removed due to cropping
    cropped_dimensions = np.maximum(
        original_shape
        - np.array(
            TARGET_SHAPE
        ),
        0
    )

    # --------------------------------------------------------
    # Case result
    # --------------------------------------------------------

    row = {

        "file":
            case_name,

        "group":
            group,

        # Original dimensions
        "original_dim_0":
            int(original_shape[0]),

        "original_dim_1":
            int(original_shape[1]),

        "original_dim_2":
            int(original_shape[2]),

        "original_voxels":
            total_original_voxels,

        # Processed dimensions
        "processed_dim_0":
            TARGET_SHAPE[0],

        "processed_dim_1":
            TARGET_SHAPE[1],

        "processed_dim_2":
            TARGET_SHAPE[2],

        "processed_voxels":
            total_processed_voxels,

        # Crop
        "crop_start_0":
            int(crop_start[0]),

        "crop_start_1":
            int(crop_start[1]),

        "crop_start_2":
            int(crop_start[2]),

        "crop_end_0":
            int(crop_end[0]),

        "crop_end_1":
            int(crop_end[1]),

        "crop_end_2":
            int(crop_end[2]),

        "cropped_dim_0":
            int(cropped_dimensions[0]),

        "cropped_dim_1":
            int(cropped_dimensions[1]),

        "cropped_dim_2":
            int(cropped_dimensions[2]),

        # Padding
        "pad_before_0":
            int(pad_before[0]),

        "pad_before_1":
            int(pad_before[1]),

        "pad_before_2":
            int(pad_before[2]),

        "pad_after_0":
            int(pad_after[0]),

        "pad_after_1":
            int(pad_after[1]),

        "pad_after_2":
            int(pad_after[2]),

        # Foreground
        "original_foreground_voxels":
            original_foreground_count,

        "processed_foreground_voxels":
            processed_foreground_count,

        "foreground_retention":
            foreground_retention,

        # Vertebrae
        "original_vertebrae_voxels":
            original_vertebrae_count,

        "processed_vertebrae_voxels":
            processed_vertebrae_count,

        "vertebrae_retention":
            vertebrae_retention,

        # Canal
        "original_spinal_canal_voxels":
            original_canal_count,

        "processed_spinal_canal_voxels":
            processed_canal_count,

        "spinal_canal_retention":
            canal_retention,

        "original_spinal_canal_percentage":
            original_canal_percentage,

        "processed_spinal_canal_percentage":
            processed_canal_percentage,

        # Disc
        "original_disc_voxels":
            original_disc_count,

        "processed_disc_voxels":
            processed_disc_count,

        "disc_retention":
            disc_retention,

        # Canal boundary
        **{
            "canal_" + key:
            value

            for key, value
            in canal_boundary.items()
        },

        # Foreground boundary
        **{
            "foreground_" + key:
            value

            for key, value
            in foreground_boundary.items()
        },
    }

    # --------------------------------------------------------
    # Original bbox values
    # --------------------------------------------------------

    if canal_bbox_original is not None:

        for axis in range(3):

            row[
                f"canal_original_min_{axis}"
            ] = int(
                canal_bbox_original[
                    axis,
                    0
                ]
            )

            row[
                f"canal_original_max_{axis}"
            ] = int(
                canal_bbox_original[
                    axis,
                    1
                ]
            )

    if canal_bbox_processed is not None:

        for axis in range(3):

            row[
                f"canal_processed_min_{axis}"
            ] = int(
                canal_bbox_processed[
                    axis,
                    0
                ]
            )

            row[
                f"canal_processed_max_{axis}"
            ] = int(
                canal_bbox_processed[
                    axis,
                    1
                ]
            )

    # --------------------------------------------------------
    # Boundary risk
    # --------------------------------------------------------

    minimum_boundary_distance = (
        row[
            "canal_min_distance_boundary"
        ]
    )

    if np.isnan(
        minimum_boundary_distance
    ):

        boundary_risk = (
            "Unknown"
        )

    elif (
        minimum_boundary_distance
        <= 2
    ):

        boundary_risk = (
            "High"
        )

    elif (
        minimum_boundary_distance
        <= 5
    ):

        boundary_risk = (
            "Moderate"
        )

    else:

        boundary_risk = (
            "Low"
        )

    row[
        "canal_boundary_risk"
    ] = boundary_risk

    # --------------------------------------------------------
    # Retention risk
    # --------------------------------------------------------

    if np.isnan(
        canal_retention
    ):

        retention_risk = (
            "Unknown"
        )

    elif canal_retention < 0.90:

        retention_risk = (
            "High"
        )

    elif canal_retention < 0.98:

        retention_risk = (
            "Moderate"
        )

    else:

        retention_risk = (
            "Low"
        )

    row[
        "canal_retention_risk"
    ] = retention_risk

    # --------------------------------------------------------
    # Combined risk
    # --------------------------------------------------------

    if (
        boundary_risk == "High"
        or retention_risk == "High"
    ):

        combined_risk = (
            "High"
        )

    elif (
        boundary_risk == "Moderate"
        or retention_risk == "Moderate"
    ):

        combined_risk = (
            "Moderate"
        )

    else:

        combined_risk = (
            "Low"
        )

    row[
        "spatial_risk"
    ] = combined_risk

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        f"Canal retention: "
        f"{canal_retention:.4f}"
    )

    print(
        f"Canal boundary distance: "
        f"{minimum_boundary_distance}"
    )

    print(
        f"Boundary risk: "
        f"{boundary_risk}"
    )

    print(
        f"Retention risk: "
        f"{retention_risk}"
    )

    print(
        f"Combined spatial risk: "
        f"{combined_risk}"
    )

    return (
        row,
        processed_image,
        processed_mask
    )


# ============================================================
# PROCESS ALL CASES
# ============================================================

print()
print("=" * 78)
print("ANALYZING PREPROCESSING ROBUSTNESS")
print("=" * 78)

rows = []

processed_examples = {}

for case_name in ALL_CASES:

    group = (
        "Failure"
        if case_name
        in FAILURE_CASES
        else "Successful"
    )

    (
        row,
        processed_image,
        processed_mask
    ) = process_case(
        case_name,
        group
    )

    rows.append(
        row
    )

    processed_examples[
        case_name
    ] = (
        processed_image,
        processed_mask
    )


results_df = pd.DataFrame(
    rows
)


# ============================================================
# MERGE MODEL DICE
# ============================================================

print()
print("=" * 78)
print("MERGING MODEL PERFORMANCE")
print("=" * 78)

part19_metrics = part19[
    [
        "file",
        "dice",
        "precision",
        "recall",
    ]
].copy()

part19_metrics = (
    part19_metrics
    .rename(
        columns={
            "dice":
                "model_dice",
            "precision":
                "model_precision",
            "recall":
                "model_recall",
        }
    )
)

results_df = results_df.merge(
    part19_metrics,
    on="file",
    how="left"
)


# ============================================================
# MERGE PART 20 FEATURES
# ============================================================

part20_columns = [
    "file",
    "intensity_mean",
    "intensity_std",
    "spinal_background_absolute_contrast",
]

part20_metrics = part20[
    part20_columns
].copy()

results_df = results_df.merge(
    part20_metrics,
    on="file",
    how="left"
)


# ============================================================
# SAVE CASE ANALYSIS
# ============================================================

case_csv = (
    OUTPUT_DIR
    / "t2_space_spatial_case_analysis.csv"
)

results_df.to_csv(
    case_csv,
    index=False
)

print(
    f"Saved: {case_csv}"
)


# ============================================================
# GROUP COMPARISON
# ============================================================

print()
print("=" * 78)
print("FAILURE VS SUCCESSFUL SPATIAL COMPARISON")
print("=" * 78)

comparison_columns = [

    "model_dice",

    "original_dim_0",
    "original_dim_1",
    "original_dim_2",

    "crop_start_0",
    "crop_start_1",
    "crop_start_2",

    "foreground_retention",

    "vertebrae_retention",

    "spinal_canal_retention",

    "original_spinal_canal_percentage",

    "processed_spinal_canal_percentage",

    "disc_retention",

    "canal_min_distance_boundary",

    "foreground_min_distance_boundary",
]


group_summary = (
    results_df
    .groupby("group")[
        comparison_columns
    ]
    .agg(
        [
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max",
        ]
    )
)

print(
    group_summary.to_string()
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_csv = (
    OUTPUT_DIR
    / "t2_space_spatial_group_summary.csv"
)

group_summary.to_csv(
    summary_csv
)

print(
    f"Saved: {summary_csv}"
)


# ============================================================
# SPATIAL RISK COUNTS
# ============================================================

print()
print("=" * 78)
print("SPATIAL RISK DISTRIBUTION")
print("=" * 78)

risk_counts = (
    results_df[
        "spatial_risk"
    ]
    .value_counts()
)

print(
    risk_counts.to_string()
)

risk_group = (
    pd.crosstab(
        results_df["group"],
        results_df["spatial_risk"]
    )
)

print()
print(
    risk_group.to_string()
)

risk_csv = (
    OUTPUT_DIR
    / "t2_space_spatial_risk_distribution.csv"
)

risk_group.to_csv(
    risk_csv
)

print(
    f"Saved: {risk_csv}"
)


# ============================================================
# CORRELATION WITH DICE
# ============================================================

print()
print("=" * 78)
print("SPATIAL FEATURE CORRELATION WITH DICE")
print("=" * 78)

correlation_features = [

    "foreground_retention",

    "vertebrae_retention",

    "spinal_canal_retention",

    "disc_retention",

    "original_spinal_canal_percentage",

    "processed_spinal_canal_percentage",

    "canal_min_distance_boundary",

    "foreground_min_distance_boundary",

    "crop_start_0",
    "crop_start_1",
    "crop_start_2",

    "intensity_mean",
    "intensity_std",

]


correlation_rows = []

for feature in correlation_features:

    valid = results_df[
        [
            feature,
            "model_dice"
        ]
    ].dropna()

    if (
        len(valid)
        >= 3
    ):

        correlation = (
            valid[
                feature
            ]
            .corr(
                valid[
                    "model_dice"
                ]
            )
        )

    else:

        correlation = np.nan

    correlation_rows.append({

        "feature":
            feature,

        "pearson_correlation_with_dice":
            correlation,

        "valid_cases":
            len(valid),
    })


correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_df = (
    correlation_df
    .sort_values(
        "pearson_correlation_with_dice",
        ascending=False
    )
)

print(
    correlation_df.to_string(
        index=False
    )
)


correlation_csv = (
    OUTPUT_DIR
    / "t2_space_spatial_correlations.csv"
)

correlation_df.to_csv(
    correlation_csv,
    index=False
)

print(
    f"Saved: {correlation_csv}"
)


# ============================================================
# CREATE BOX PLOT
# ============================================================

def create_boxplot(
    dataframe,
    column,
    title,
    ylabel,
    filename
):

    failure = dataframe[
        dataframe["group"]
        == "Failure"
    ][column].dropna()

    successful = dataframe[
        dataframe["group"]
        == "Successful"
    ][column].dropna()

    if (
        len(failure) == 0
        or len(successful) == 0
    ):

        return

    plt.figure(
        figsize=(8, 6)
    )

    plt.boxplot(
        [
            failure.values,
            successful.values,
        ],
        labels=[
            "Failure",
            "Successful",
        ]
    )

    plt.title(
        title
    )

    plt.ylabel(
        ylabel
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / filename
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# BOX PLOTS
# ============================================================

print()
print("=" * 78)
print("CREATING SPATIAL COMPARISON CHARTS")
print("=" * 78)

create_boxplot(
    results_df,
    "spinal_canal_retention",
    "Spinal Canal Retention: Failure vs Successful",
    "Canal Retention",
    "canal_retention_failure_vs_success.png"
)

create_boxplot(
    results_df,
    "foreground_retention",
    "Foreground Retention: Failure vs Successful",
    "Foreground Retention",
    "foreground_retention_failure_vs_success.png"
)

create_boxplot(
    results_df,
    "canal_min_distance_boundary",
    "Canal Boundary Distance: Failure vs Successful",
    "Minimum Boundary Distance",
    "canal_boundary_distance_failure_vs_success.png"
)

create_boxplot(
    results_df,
    "processed_spinal_canal_percentage",
    "Processed Canal Percentage: Failure vs Successful",
    "Canal Percentage",
    "processed_canal_percentage_failure_vs_success.png"
)


# ============================================================
# SCATTER PLOT
# ============================================================

def create_scatter(
    dataframe,
    x_column,
    title,
    xlabel,
    filename
):

    plt.figure(
        figsize=(8, 6)
    )

    for group in [
        "Failure",
        "Successful",
    ]:

        subset = dataframe[
            dataframe["group"]
            == group
        ]

        plt.scatter(
            subset[
                x_column
            ],
            subset[
                "model_dice"
            ],
            label=group,
            s=80
        )

    plt.title(
        title
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        "Model Dice"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / filename
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


create_scatter(
    results_df,
    "spinal_canal_retention",
    "Spinal Canal Retention vs Dice",
    "Spinal Canal Retention",
    "canal_retention_vs_dice.png"
)

create_scatter(
    results_df,
    "canal_min_distance_boundary",
    "Canal Boundary Distance vs Dice",
    "Minimum Boundary Distance",
    "canal_boundary_vs_dice.png"
)

create_scatter(
    results_df,
    "foreground_retention",
    "Foreground Retention vs Dice",
    "Foreground Retention",
    "foreground_retention_vs_dice.png"
)


# ============================================================
# CASE-BY-CASE SPATIAL TABLE
# ============================================================

print()
print("=" * 78)
print("CASE-BY-CASE SPATIAL SUMMARY")
print("=" * 78)

display_columns = [
    "file",
    "group",
    "model_dice",
    "spinal_canal_retention",
    "foreground_retention",
    "canal_min_distance_boundary",
    "canal_boundary_risk",
    "canal_retention_risk",
    "spatial_risk",
]

print(
    results_df[
        display_columns
    ]
    .sort_values(
        "model_dice"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE RISK TABLE
# ============================================================

risk_table_csv = (
    OUTPUT_DIR
    / "t2_space_spatial_case_risk_table.csv"
)

results_df[
    display_columns
].sort_values(
    "model_dice"
).to_csv(
    risk_table_csv,
    index=False
)

print(
    f"Saved: {risk_table_csv}"
)


# ============================================================
# JSON SAFE CONVERSION
# ============================================================

def make_json_safe(
    value
):

    if isinstance(
        value,
        dict
    ):

        return {
            str(k):
            make_json_safe(v)

            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple)
    ):

        return [
            make_json_safe(v)
            for v in value
        ]

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32
        )
    ):

        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32
        )
    ):

        if np.isnan(value):

            return None

        return float(value)

    if isinstance(
        value,
        np.bool_
    ):

        return bool(value)

    return value


# ============================================================
# JSON SUMMARY
# ============================================================

summary = {

    "phase":
        "Phase 3 - Part 21",

    "purpose":
        "T2 SPACE preprocessing and spatial robustness analysis",

    "target_shape":
        list(
            TARGET_SHAPE
        ),

    "total_cases":
        len(results_df),

    "failure_cases":
        FAILURE_CASES,

    "successful_cases":
        SUCCESS_CASES,

    "failure_count":
        len(
            FAILURE_CASES
        ),

    "successful_count":
        len(
            SUCCESS_CASES
        ),

    "failure_mean_dice":
        failure_mean
        if False
        else float(
            results_df[
                results_df["group"]
                == "Failure"
            ][
                "model_dice"
            ].mean()
        ),

    "successful_mean_dice":
        float(
            results_df[
                results_df["group"]
                == "Successful"
            ][
                "model_dice"
            ].mean()
        ),

    "spatial_risk_counts":
        risk_counts.to_dict(),

    "correlations":
        correlation_df.to_dict(
            orient="records"
        ),

    "output_directory":
        str(OUTPUT_DIR),
}

summary = make_json_safe(
    summary
)

json_path = (
    OUTPUT_DIR
    / "t2_space_spatial_robustness_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

print(
    f"Saved: {json_path}"
)


# ============================================================
# REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "phase3_part21_spatial_robustness_report.txt"
)

failure_df = results_df[
    results_df["group"]
    == "Failure"
]

successful_df = results_df[
    results_df["group"]
    == "Successful"
]

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "=" * 78
        + "\n"
    )

    file.write(
        "PHASE 3 - PART 21\n"
    )

    file.write(
        "T2 SPACE PREPROCESSING & SPATIAL ROBUSTNESS ANALYSIS\n"
    )

    file.write(
        "=" * 78
        + "\n\n"
    )

    file.write(
        "OBJECTIVE\n"
    )

    file.write(
        "-" * 78
        + "\n"
    )

    file.write(
        "This analysis investigates whether the center "
        "crop/padding operation to 96 x 96 x 96 may "
        "contribute to T2 SPACE segmentation failures.\n\n"
    )

    file.write(
        "MODEL INPUT SIZE\n"
    )

    file.write(
        f"{TARGET_SHAPE}\n\n"
    )

    file.write(
        "GROUP PERFORMANCE\n"
    )

    file.write(
        "-" * 78
        + "\n"
    )

    file.write(
        f"Failure cases: {len(failure_df)}\n"
    )

    file.write(
        f"Successful cases: {len(successful_df)}\n"
    )

    file.write(
        f"Failure mean Dice: "
        f"{failure_df['model_dice'].mean():.6f}\n"
    )

    file.write(
        f"Successful mean Dice: "
        f"{successful_df['model_dice'].mean():.6f}\n\n"
    )

    file.write(
        "SPATIAL RISK DISTRIBUTION\n"
    )

    file.write(
        "-" * 78
        + "\n"
    )

    file.write(
        risk_counts.to_string()
        + "\n\n"
    )

    file.write(
        "CASE SUMMARY\n"
    )

    file.write(
        "-" * 78
        + "\n"
    )

    file.write(
        results_df[
            display_columns
        ]
        .sort_values(
            "model_dice"
        )
        .to_string(
            index=False
        )
    )

    file.write(
        "\n\nSPATIAL CORRELATIONS WITH DICE\n"
    )

    file.write(
        "-" * 78
        + "\n"
    )

    file.write(
        correlation_df.to_string(
            index=False
        )
    )

    file.write(
        "\n\nINTERPRETATION CAUTION\n"
    )

    file.write(
        "-" * 78
        + "\n"
    )

    file.write(
        "Only nine T2 SPACE cases are analyzed. "
        "Therefore, observed spatial relationships are "
        "exploratory and should not be interpreted as "
        "causal evidence.\n"
    )

    file.write(
        "This analysis does not modify or retrain the model.\n"
    )

print(
    f"Saved: {report_path}"
)


# ============================================================
# COMPLETE
# ============================================================

elapsed = (
    time.time()
    - START_TIME
)

print()
print("=" * 78)
print("PART 21 COMPLETE")
print("=" * 78)

print(
    f"Cases analyzed: "
    f"{len(results_df)}"
)

print(
    f"Failure cases: "
    f"{len(failure_df)}"
)

print(
    f"Successful cases: "
    f"{len(successful_df)}"
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
print("PHASE 3 - PART 21 COMPLETE")
print("=" * 78)