"""
PHASE 3 - PART 22
T2 SPACE ANATOMY LOCALIZATION & CROP ALIGNMENT ANALYSIS

Purpose
-------
Investigate whether anatomical localization relative to the
fixed 96 x 96 x 96 center crop is associated with T2 SPACE
segmentation performance.

This is an analysis-only stage.

IMPORTANT
---------
- No training
- No optimizer
- No checkpoint modification
- No model retraining
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

PART21_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "t2_space_spatial_robustness"
    / "t2_space_spatial_case_analysis.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "t2_space_anatomy_alignment"
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
# TARGET MODEL INPUT
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
print("PHASE 3 - PART 22")
print("T2 SPACE ANATOMY LOCALIZATION & CROP ALIGNMENT ANALYSIS")
print("=" * 78)

print()
print("PROJECT ROOT")
print(PROJECT_ROOT)

print()
print("TEST DIRECTORY")
print(TEST_DIR)

print()
print("TARGET MODEL INPUT")
print(TARGET_SHAPE)

print()
print("OUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================
# VALIDATION
# ============================================================

required_paths = [
    TEST_DIR,
    IMAGE_DIR,
    MASK_DIR,
    PART19_RESULTS,
    PART20_RESULTS,
    PART21_RESULTS,
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

part19 = pd.read_csv(
    PART19_RESULTS
)

part20 = pd.read_csv(
    PART20_RESULTS
)

part21 = pd.read_csv(
    PART21_RESULTS
)

print(
    f"Part 19 cases : {len(part19)}"
)

print(
    f"Part 20 cases : {len(part20)}"
)

print(
    f"Part 21 cases : {len(part21)}"
)


# ============================================================
# FIND CASE FILE
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

        candidate = (
            directory
            / (
                case_name
                + extension
            )
        )

        if candidate.exists():

            return candidate

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
    Center crop or symmetric pad a 3-D volume.

    Returns
    -------
    processed
    crop_start
    crop_end
    pad_before
    pad_after
    """

    volume = np.asarray(
        volume
    )

    original_shape = np.array(
        volume.shape,
        dtype=int
    )

    target = np.array(
        target_shape,
        dtype=int
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

        current = int(
            original_shape[axis]
        )

        desired = int(
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
# CENTROID
# ============================================================

def calculate_centroid(
    binary_mask
):

    coordinates = np.argwhere(
        binary_mask
    )

    if coordinates.size == 0:

        return None

    return coordinates.mean(
        axis=0
    )


# ============================================================
# BOUNDING BOX
# ============================================================

def calculate_bbox(
    binary_mask
):

    coordinates = np.argwhere(
        binary_mask
    )

    if coordinates.size == 0:

        return None

    minimum = coordinates.min(
        axis=0
    )

    maximum = coordinates.max(
        axis=0
    )

    return np.stack(
        [
            minimum,
            maximum,
        ],
        axis=1
    )


# ============================================================
# TRANSFORM CENTROID
# ============================================================

def transform_centroid(
    centroid,
    crop_start,
    pad_before
):

    if centroid is None:

        return None

    return (
        centroid
        - crop_start
        + pad_before
    )


# ============================================================
# TRANSFORM BBOX
# ============================================================

def transform_bbox(
    bbox,
    crop_start,
    pad_before,
    target_shape
):

    if bbox is None:

        return None

    transformed = (
        bbox
        - crop_start.reshape(
            3,
            1
        )
        + pad_before.reshape(
            3,
            1
        )
    )

    lower = np.maximum(
        transformed[
            :,
            0
        ],
        0
    )

    upper = np.minimum(
        transformed[
            :,
            1
        ],
        np.array(
            target_shape
        ) - 1
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
# DISTANCE
# ============================================================

def euclidean_distance(
    a,
    b
):

    if (
        a is None
        or b is None
    ):

        return np.nan

    return float(
        np.linalg.norm(
            a - b
        )
    )


# ============================================================
# NORMALIZED DISTANCE
# ============================================================

def normalized_distance(
    distance,
    shape
):

    if np.isnan(
        distance
    ):

        return np.nan

    diagonal = np.linalg.norm(
        np.array(
            shape,
            dtype=float
        )
    )

    if diagonal == 0:

        return np.nan

    return float(
        distance
        / diagonal
    )


# ============================================================
# DISTANCE TO BOUNDARY
# ============================================================

def boundary_distances(
    bbox,
    shape
):

    if bbox is None:

        return {

            "min_boundary_distance":
                np.nan,

            "axis0_min_distance":
                np.nan,

            "axis0_max_distance":
                np.nan,

            "axis1_min_distance":
                np.nan,

            "axis1_max_distance":
                np.nan,

            "axis2_min_distance":
                np.nan,

            "axis2_max_distance":
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

        "min_boundary_distance":
            float(
                min(distances)
            ),

        "axis0_min_distance":
            distances[0],

        "axis0_max_distance":
            distances[1],

        "axis1_min_distance":
            distances[2],

        "axis1_max_distance":
            distances[3],

        "axis2_min_distance":
            distances[4],

        "axis2_max_distance":
            distances[5],
    }


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value
):

    if value is None:

        return None

    try:

        value = float(
            value
        )

    except Exception:

        return None

    if np.isnan(
        value
    ) or np.isinf(
        value
    ):

        return None

    return value


# ============================================================
# PROCESS CASE
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
            f"Image/mask mismatch "
            f"for {case_name}: "
            f"{image.shape} vs "
            f"{mask.shape}"
        )

    original_shape = np.array(
        image.shape,
        dtype=int
    )

    print(
        "Original shape:",
        tuple(
            original_shape
        )
    )

    (
        processed_image,
        crop_start,
        crop_end,
        pad_before,
        pad_after
    ) = center_crop_pad(
        image,
        TARGET_SHAPE
    )

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

    # --------------------------------------------------------
    # Centers
    # --------------------------------------------------------

    original_volume_center = (
        (
            original_shape
            - 1
        )
        / 2.0
    )

    processed_volume_center = (
        (
            np.array(
                TARGET_SHAPE
            )
            - 1
        )
        / 2.0
    )

    # --------------------------------------------------------
    # Anatomical masks
    # --------------------------------------------------------

    foreground_mask_original = (
        mask > 0
    )

    vertebrae_mask_original = (
        mask == 1
    )

    canal_mask_original = (
        mask == 2
    )

    disc_mask_original = (
        mask == 3
    )

    foreground_mask_processed = (
        processed_mask > 0
    )

    vertebrae_mask_processed = (
        processed_mask == 1
    )

    canal_mask_processed = (
        processed_mask == 2
    )

    disc_mask_processed = (
        processed_mask == 3
    )

    # --------------------------------------------------------
    # Original centroids
    # --------------------------------------------------------

    foreground_centroid_original = (
        calculate_centroid(
            foreground_mask_original
        )
    )

    vertebrae_centroid_original = (
        calculate_centroid(
            vertebrae_mask_original
        )
    )

    canal_centroid_original = (
        calculate_centroid(
            canal_mask_original
        )
    )

    disc_centroid_original = (
        calculate_centroid(
            disc_mask_original
        )
    )

    # --------------------------------------------------------
    # Processed centroids
    # --------------------------------------------------------

    foreground_centroid_processed = (
        calculate_centroid(
            foreground_mask_processed
        )
    )

    vertebrae_centroid_processed = (
        calculate_centroid(
            vertebrae_mask_processed
        )
    )

    canal_centroid_processed = (
        calculate_centroid(
            canal_mask_processed
        )
    )

    disc_centroid_processed = (
        calculate_centroid(
            disc_mask_processed
        )
    )

    # --------------------------------------------------------
    # Expected transformed centroids
    # --------------------------------------------------------

    expected_foreground_centroid = (
        transform_centroid(
            foreground_centroid_original,
            crop_start,
            pad_before
        )
    )

    expected_vertebrae_centroid = (
        transform_centroid(
            vertebrae_centroid_original,
            crop_start,
            pad_before
        )
    )

    expected_canal_centroid = (
        transform_centroid(
            canal_centroid_original,
            crop_start,
            pad_before
        )
    )

    expected_disc_centroid = (
        transform_centroid(
            disc_centroid_original,
            crop_start,
            pad_before
        )
    )

    # --------------------------------------------------------
    # Alignment distances
    # --------------------------------------------------------

    foreground_alignment = (
        euclidean_distance(
            expected_foreground_centroid,
            processed_volume_center
        )
    )

    vertebrae_alignment = (
        euclidean_distance(
            expected_vertebrae_centroid,
            processed_volume_center
        )
    )

    canal_alignment = (
        euclidean_distance(
            expected_canal_centroid,
            processed_volume_center
        )
    )

    disc_alignment = (
        euclidean_distance(
            expected_disc_centroid,
            processed_volume_center
        )
    )

    # --------------------------------------------------------
    # Normalized alignment
    # --------------------------------------------------------

    foreground_alignment_normalized = (
        normalized_distance(
            foreground_alignment,
            TARGET_SHAPE
        )
    )

    vertebrae_alignment_normalized = (
        normalized_distance(
            vertebrae_alignment,
            TARGET_SHAPE
        )
    )

    canal_alignment_normalized = (
        normalized_distance(
            canal_alignment,
            TARGET_SHAPE
        )
    )

    disc_alignment_normalized = (
        normalized_distance(
            disc_alignment,
            TARGET_SHAPE
        )
    )

    # --------------------------------------------------------
    # Boundary distances
    # --------------------------------------------------------

    foreground_bbox_original = (
        calculate_bbox(
            foreground_mask_original
        )
    )

    vertebrae_bbox_original = (
        calculate_bbox(
            vertebrae_mask_original
        )
    )

    canal_bbox_original = (
        calculate_bbox(
            canal_mask_original
        )
    )

    disc_bbox_original = (
        calculate_bbox(
            disc_mask_original
        )
    )

    foreground_bbox_processed = (
        transform_bbox(
            foreground_bbox_original,
            crop_start,
            pad_before,
            TARGET_SHAPE
        )
    )

    vertebrae_bbox_processed = (
        transform_bbox(
            vertebrae_bbox_original,
            crop_start,
            pad_before,
            TARGET_SHAPE
        )
    )

    canal_bbox_processed = (
        transform_bbox(
            canal_bbox_original,
            crop_start,
            pad_before,
            TARGET_SHAPE
        )
    )

    disc_bbox_processed = (
        transform_bbox(
            disc_bbox_original,
            crop_start,
            pad_before,
            TARGET_SHAPE
        )
    )

    foreground_boundary = (
        boundary_distances(
            foreground_bbox_processed,
            TARGET_SHAPE
        )
    )

    vertebrae_boundary = (
        boundary_distances(
            vertebrae_bbox_processed,
            TARGET_SHAPE
        )
    )

    canal_boundary = (
        boundary_distances(
            canal_bbox_processed,
            TARGET_SHAPE
        )
    )

    disc_boundary = (
        boundary_distances(
            disc_bbox_processed,
            TARGET_SHAPE
        )
    )

    # --------------------------------------------------------
    # Retention
    # --------------------------------------------------------

    def retention(
        original_mask,
        processed_mask
    ):

        original_count = int(
            original_mask.sum()
        )

        processed_count = int(
            processed_mask.sum()
        )

        if original_count == 0:

            return np.nan

        return (
            processed_count
            / original_count
        )

    foreground_retention = retention(
        foreground_mask_original,
        foreground_mask_processed
    )

    vertebrae_retention = retention(
        vertebrae_mask_original,
        vertebrae_mask_processed
    )

    canal_retention = retention(
        canal_mask_original,
        canal_mask_processed
    )

    disc_retention = retention(
        disc_mask_original,
        disc_mask_processed
    )

    # --------------------------------------------------------
    # Center offset components
    # --------------------------------------------------------

    def centroid_components(
        centroid
    ):

        if centroid is None:

            return (
                np.nan,
                np.nan,
                np.nan
            )

        offset = (
            centroid
            - processed_volume_center
        )

        return (
            float(offset[0]),
            float(offset[1]),
            float(offset[2])
        )

    (
        fg_offset_0,
        fg_offset_1,
        fg_offset_2
    ) = centroid_components(
        expected_foreground_centroid
    )

    (
        vert_offset_0,
        vert_offset_1,
        vert_offset_2
    ) = centroid_components(
        expected_vertebrae_centroid
    )

    (
        canal_offset_0,
        canal_offset_1,
        canal_offset_2
    ) = centroid_components(
        expected_canal_centroid
    )

    (
        disc_offset_0,
        disc_offset_1,
        disc_offset_2
    ) = centroid_components(
        expected_disc_centroid
    )

    # --------------------------------------------------------
    # Overall alignment score
    # --------------------------------------------------------

    available_alignment = [
        x
        for x in [
            foreground_alignment_normalized,
            vertebrae_alignment_normalized,
            canal_alignment_normalized,
            disc_alignment_normalized,
        ]
        if not np.isnan(x)
    ]

    if available_alignment:

        overall_alignment_score = float(
            np.mean(
                available_alignment
            )
        )

    else:

        overall_alignment_score = np.nan

    # --------------------------------------------------------
    # Model metrics
    # --------------------------------------------------------

    metric_row = part19[
        part19["file"]
        == case_name
    ]

    if len(metric_row) == 0:

        model_dice = np.nan
        model_precision = np.nan
        model_recall = np.nan

    else:

        metric_row = metric_row.iloc[0]

        model_dice = float(
            metric_row["dice"]
        )

        model_precision = float(
            metric_row["precision"]
        )

        model_recall = float(
            metric_row["recall"]
        )

    # --------------------------------------------------------
    # Image intensity
    # --------------------------------------------------------

    finite = processed_image[
        np.isfinite(
            processed_image
        )
    ]

    if finite.size:

        intensity_mean = float(
            finite.mean()
        )

        intensity_std = float(
            finite.std()
        )

    else:

        intensity_mean = np.nan
        intensity_std = np.nan

    # --------------------------------------------------------
    # Row
    # --------------------------------------------------------

    row = {

        "file":
            case_name,

        "group":
            group,

        "model_dice":
            model_dice,

        "model_precision":
            model_precision,

        "model_recall":
            model_recall,

        # Original dimensions
        "original_dim_0":
            int(original_shape[0]),

        "original_dim_1":
            int(original_shape[1]),

        "original_dim_2":
            int(original_shape[2]),

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

        # Foreground alignment
        "foreground_alignment_distance":
            foreground_alignment,

        "foreground_alignment_normalized":
            foreground_alignment_normalized,

        "foreground_offset_0":
            fg_offset_0,

        "foreground_offset_1":
            fg_offset_1,

        "foreground_offset_2":
            fg_offset_2,

        "foreground_retention":
            foreground_retention,

        # Vertebrae
        "vertebrae_alignment_distance":
            vertebrae_alignment,

        "vertebrae_alignment_normalized":
            vertebrae_alignment_normalized,

        "vertebrae_offset_0":
            vert_offset_0,

        "vertebrae_offset_1":
            vert_offset_1,

        "vertebrae_offset_2":
            vert_offset_2,

        "vertebrae_retention":
            vertebrae_retention,

        # Canal
        "canal_alignment_distance":
            canal_alignment,

        "canal_alignment_normalized":
            canal_alignment_normalized,

        "canal_offset_0":
            canal_offset_0,

        "canal_offset_1":
            canal_offset_1,

        "canal_offset_2":
            canal_offset_2,

        "canal_retention":
            canal_retention,

        # Disc
        "disc_alignment_distance":
            disc_alignment,

        "disc_alignment_normalized":
            disc_alignment_normalized,

        "disc_offset_0":
            disc_offset_0,

        "disc_offset_1":
            disc_offset_1,

        "disc_offset_2":
            disc_offset_2,

        "disc_retention":
            disc_retention,

        # Overall
        "overall_alignment_score":
            overall_alignment_score,

        # Intensity
        "intensity_mean":
            intensity_mean,

        "intensity_std":
            intensity_std,

        # Boundary distances
        "foreground_min_boundary_distance":
            foreground_boundary[
                "min_boundary_distance"
            ],

        "vertebrae_min_boundary_distance":
            vertebrae_boundary[
                "min_boundary_distance"
            ],

        "canal_min_boundary_distance":
            canal_boundary[
                "min_boundary_distance"
            ],

        "disc_min_boundary_distance":
            disc_boundary[
                "min_boundary_distance"
            ],
    }

    print(
        f"Model Dice: "
        f"{model_dice:.6f}"
    )

    print(
        f"Foreground alignment: "
        f"{foreground_alignment:.3f}"
    )

    print(
        f"Vertebrae alignment: "
        f"{vertebrae_alignment:.3f}"
    )

    print(
        f"Canal alignment: "
        f"{canal_alignment:.3f}"
    )

    print(
        f"Disc alignment: "
        f"{disc_alignment:.3f}"
    )

    print(
        f"Canal retention: "
        f"{canal_retention:.6f}"
    )

    return (
        row,
        processed_image,
        processed_mask,
        {
            "foreground_centroid":
                expected_foreground_centroid,

            "vertebrae_centroid":
                expected_vertebrae_centroid,

            "canal_centroid":
                expected_canal_centroid,

            "disc_centroid":
                expected_disc_centroid,

            "crop_center":
                processed_volume_center,
        }
    )


# ============================================================
# PROCESS ALL CASES
# ============================================================

print()
print("=" * 78)
print("ANALYZING ANATOMICAL ALIGNMENT")
print("=" * 78)

rows = []

visual_data = {}

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
        processed_mask,
        centers
    ) = process_case(
        case_name,
        group
    )

    rows.append(
        row
    )

    visual_data[
        case_name
    ] = {
        "image":
            processed_image,

        "mask":
            processed_mask,

        "centers":
            centers,

        "group":
            group,
    }


results_df = pd.DataFrame(
    rows
)


# ============================================================
# MERGE PART 21
# ============================================================

part21_columns = [
    "file",
    "spinal_canal_retention",
    "foreground_retention",
    "vertebrae_retention",
    "disc_retention",
]

part21_small = part21[
    part21_columns
].copy()

part21_small = (
    part21_small
    .rename(
        columns={
            "spinal_canal_retention":
                "part21_canal_retention",

            "foreground_retention":
                "part21_foreground_retention",

            "vertebrae_retention":
                "part21_vertebrae_retention",

            "disc_retention":
                "part21_disc_retention",
        }
    )
)

results_df = results_df.merge(
    part21_small,
    on="file",
    how="left"
)


# ============================================================
# SAVE CASE RESULTS
# ============================================================

case_csv = (
    OUTPUT_DIR
    / "t2_space_anatomy_alignment_case_analysis.csv"
)

results_df.to_csv(
    case_csv,
    index=False
)

print()
print(
    f"Saved: {case_csv}"
)


# ============================================================
# GROUP SUMMARY
# ============================================================

print()
print("=" * 78)
print("FAILURE VS SUCCESSFUL ALIGNMENT COMPARISON")
print("=" * 78)

comparison_features = [

    "model_dice",

    "foreground_alignment_distance",

    "vertebrae_alignment_distance",

    "canal_alignment_distance",

    "disc_alignment_distance",

    "foreground_alignment_normalized",

    "vertebrae_alignment_normalized",

    "canal_alignment_normalized",

    "disc_alignment_normalized",

    "overall_alignment_score",

    "foreground_retention",

    "vertebrae_retention",

    "canal_retention",

    "disc_retention",

    "foreground_min_boundary_distance",

    "vertebrae_min_boundary_distance",

    "canal_min_boundary_distance",

    "disc_min_boundary_distance",

    "intensity_mean",

    "intensity_std",
]

group_summary = (
    results_df
    .groupby("group")[
        comparison_features
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


group_summary_csv = (
    OUTPUT_DIR
    / "t2_space_anatomy_alignment_group_summary.csv"
)

group_summary.to_csv(
    group_summary_csv
)

print(
    f"Saved: {group_summary_csv}"
)


# ============================================================
# CORRELATIONS
# ============================================================

print()
print("=" * 78)
print("ALIGNMENT CORRELATION WITH MODEL DICE")
print("=" * 78)

correlation_features = [

    "foreground_alignment_distance",

    "vertebrae_alignment_distance",

    "canal_alignment_distance",

    "disc_alignment_distance",

    "foreground_alignment_normalized",

    "vertebrae_alignment_normalized",

    "canal_alignment_normalized",

    "disc_alignment_normalized",

    "overall_alignment_score",

    "foreground_retention",

    "vertebrae_retention",

    "canal_retention",

    "disc_retention",

    "foreground_min_boundary_distance",

    "vertebrae_min_boundary_distance",

    "canal_min_boundary_distance",

    "disc_min_boundary_distance",

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
        len(valid) >= 3
        and valid[feature].nunique() > 1
        and valid["model_dice"].nunique() > 1
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
            int(len(valid)),
    })


correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_df = (
    correlation_df
    .sort_values(
        "pearson_correlation_with_dice",
        ascending=False,
        na_position="last"
    )
)

print(
    correlation_df.to_string(
        index=False
    )
)


correlation_csv = (
    OUTPUT_DIR
    / "t2_space_anatomy_alignment_correlations.csv"
)

correlation_df.to_csv(
    correlation_csv,
    index=False
)

print(
    f"Saved: {correlation_csv}"
)


# ============================================================
# ALIGNMENT CATEGORY
# ============================================================

def alignment_category(
    normalized_distance
):

    if np.isnan(
        normalized_distance
    ):

        return "Unavailable"

    if normalized_distance < 0.10:

        return "Excellent"

    if normalized_distance < 0.20:

        return "Good"

    if normalized_distance < 0.30:

        return "Moderate"

    return "Poor"


results_df[
    "canal_alignment_category"
] = results_df[
    "canal_alignment_normalized"
].apply(
    alignment_category
)

results_df[
    "vertebrae_alignment_category"
] = results_df[
    "vertebrae_alignment_normalized"
].apply(
    alignment_category
)

results_df[
    "foreground_alignment_category"
] = results_df[
    "foreground_alignment_normalized"
].apply(
    alignment_category
)


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

print()
print("=" * 78)
print("ANATOMICAL ALIGNMENT CATEGORIES")
print("=" * 78)

print()
print("Spinal Canal")

print(
    results_df[
        "canal_alignment_category"
    ].value_counts()
)

print()
print("Vertebrae")

print(
    results_df[
        "vertebrae_alignment_category"
    ].value_counts()
)

print()
print("Foreground")

print(
    results_df[
        "foreground_alignment_category"
    ].value_counts()
)


category_table = pd.DataFrame({

    "canal_alignment":
        results_df[
            "canal_alignment_category"
        ].value_counts(),

    "vertebrae_alignment":
        results_df[
            "vertebrae_alignment_category"
        ].value_counts(),

    "foreground_alignment":
        results_df[
            "foreground_alignment_category"
        ].value_counts(),
})

category_csv = (
    OUTPUT_DIR
    / "t2_space_alignment_categories.csv"
)

category_table.to_csv(
    category_csv
)

print(
    f"Saved: {category_csv}"
)


# ============================================================
# SCATTER FUNCTION
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

        valid = subset[
            [
                x_column,
                "model_dice"
            ]
        ].dropna()

        if len(valid) == 0:

            continue

        plt.scatter(
            valid[
                x_column
            ],
            valid[
                "model_dice"
            ],
            label=group,
            s=85
        )

        for _, row in valid.iterrows():

            case_name = (
                row.name
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


# ============================================================
# ALIGNMENT CHARTS
# ============================================================

print()
print("=" * 78)
print("CREATING ALIGNMENT CHARTS")
print("=" * 78)

create_scatter(
    results_df,
    "canal_alignment_distance",
    "Spinal Canal Alignment Distance vs Dice",
    "Canal Distance from Crop Center (voxels)",
    "canal_alignment_distance_vs_dice.png"
)

create_scatter(
    results_df,
    "vertebrae_alignment_distance",
    "Vertebrae Alignment Distance vs Dice",
    "Vertebrae Distance from Crop Center (voxels)",
    "vertebrae_alignment_distance_vs_dice.png"
)

create_scatter(
    results_df,
    "foreground_alignment_distance",
    "Foreground Alignment Distance vs Dice",
    "Foreground Distance from Crop Center (voxels)",
    "foreground_alignment_distance_vs_dice.png"
)

create_scatter(
    results_df,
    "overall_alignment_score",
    "Overall Anatomical Alignment vs Dice",
    "Mean Normalized Alignment Distance",
    "overall_alignment_vs_dice.png"
)


# ============================================================
# BOX PLOT
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
        tick_labels=[
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


create_boxplot(
    results_df,
    "canal_alignment_distance",
    "Spinal Canal Alignment: Failure vs Successful",
    "Distance from Crop Center",
    "canal_alignment_failure_vs_success.png"
)

create_boxplot(
    results_df,
    "vertebrae_alignment_distance",
    "Vertebrae Alignment: Failure vs Successful",
    "Distance from Crop Center",
    "vertebrae_alignment_failure_vs_success.png"
)

create_boxplot(
    results_df,
    "overall_alignment_score",
    "Overall Alignment: Failure vs Successful",
    "Normalized Alignment Distance",
    "overall_alignment_failure_vs_success.png"
)


# ============================================================
# REPRESENTATIVE CASES
# ============================================================

print()
print("=" * 78)
print("SELECTING REPRESENTATIVE ALIGNMENT CASES")
print("=" * 78)

best_case = (
    results_df
    .sort_values(
        "model_dice",
        ascending=False
    )
    .iloc[0]
)

worst_case = (
    results_df
    .sort_values(
        "model_dice",
        ascending=True
    )
    .iloc[0]
)

median_dice = (
    results_df[
        "model_dice"
    ]
    .median()
)

median_case = (
    results_df
    .assign(
        dice_distance=lambda x:
        abs(
            x["model_dice"]
            - median_dice
        )
    )
    .sort_values(
        "dice_distance"
    )
    .iloc[0]
)

representative_cases = [
    best_case["file"],
    median_case["file"],
    worst_case["file"],
]

print(
    "Best case   :",
    best_case["file"],
    f"(Dice = {best_case['model_dice']:.6f})"
)

print(
    "Median case :",
    median_case["file"],
    f"(Dice = {median_case['model_dice']:.6f})"
)

print(
    "Worst case  :",
    worst_case["file"],
    f"(Dice = {worst_case['model_dice']:.6f})"
)


# ============================================================
# VISUALIZATION FUNCTION
# ============================================================

def create_alignment_visualization(
    case_name
):

    data = visual_data[
        case_name
    ]

    image = data[
        "image"
    ]

    mask = data[
        "mask"
    ]

    centers = data[
        "centers"
    ]

    group = data[
        "group"
    ]

    # Choose axial slice based on
    # strongest foreground presence.

    foreground = (
        mask > 0
    )

    counts = (
        foreground
        .sum(
            axis=(1, 2)
        )
    )

    if counts.max() > 0:

        slice_index = int(
            np.argmax(
                counts
            )
        )

    else:

        slice_index = (
            TARGET_SHAPE[0]
            // 2
        )

    image_slice = image[
        slice_index
    ]

    mask_slice = mask[
        slice_index
    ]

    figure = plt.figure(
        figsize=(16, 5)
    )

    # --------------------------------------------------------
    # MRI
    # --------------------------------------------------------

    ax1 = figure.add_subplot(
        1,
        3,
        1
    )

    ax1.imshow(
        image_slice,
        cmap="gray"
    )

    ax1.set_title(
        f"{case_name}\n"
        f"{group} | Slice {slice_index}"
    )

    ax1.axis(
        "off"
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ax2 = figure.add_subplot(
        1,
        3,
        2
    )

    ax2.imshow(
        image_slice,
        cmap="gray"
    )

    ax2.imshow(
        np.ma.masked_where(
            mask_slice == 0,
            mask_slice
        ),
        alpha=0.45,
        interpolation="nearest"
    )

    ax2.set_title(
        "Ground-Truth Anatomy"
    )

    ax2.axis(
        "off"
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    ax3 = figure.add_subplot(
        1,
        3,
        3
    )

    ax3.imshow(
        image_slice,
        cmap="gray"
    )

    ax3.axvline(
        TARGET_SHAPE[1] / 2,
        linestyle="--",
        linewidth=1.5
    )

    ax3.axhline(
        TARGET_SHAPE[0] / 2,
        linestyle="--",
        linewidth=1.5
    )

    # Centers use [axis0, axis1, axis2]
    # For an axial slice axis0 is fixed.
    # Plot axis1 horizontally and axis2 vertically.

    center = centers[
        "crop_center"
    ]

    ax3.scatter(
        center[2],
        center[1],
        s=100,
        marker="+",
        linewidths=2,
        label="Crop center"
    )

    plotted = False

    for label, centroid, marker in [

        (
            "Foreground",
            centers[
                "foreground_centroid"
            ],
            "o"
        ),

        (
            "Vertebrae",
            centers[
                "vertebrae_centroid"
            ],
            "s"
        ),

        (
            "Spinal Canal",
            centers[
                "canal_centroid"
            ],
            "^"
        ),

        (
            "Disc",
            centers[
                "disc_centroid"
            ],
            "D"
        ),
    ]:

        if centroid is None:

            continue

        ax3.scatter(
            centroid[2],
            centroid[1],
            s=70,
            marker=marker,
            label=label
        )

        plotted = True

    ax3.set_title(
        "Anatomy vs Crop Center"
    )

    if plotted:

        ax3.legend(
            fontsize=8
        )

    ax3.axis(
        "off"
    )

    plt.tight_layout()

    safe_name = (
        case_name
        .replace(
            "/",
            "_"
        )
        .replace(
            "\\",
            "_"
        )
    )

    path = (
        OUTPUT_DIR
        / f"{safe_name}_alignment_visualization.png"
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
# CREATE REPRESENTATIVE VISUALS
# ============================================================

print()
print("=" * 78)
print("GENERATING REPRESENTATIVE ALIGNMENT VISUALIZATIONS")
print("=" * 78)

for case_name in representative_cases:

    create_alignment_visualization(
        case_name
    )


# ============================================================
# CASE RANKING
# ============================================================

ranking_columns = [

    "file",
    "group",
    "model_dice",

    "foreground_alignment_distance",

    "vertebrae_alignment_distance",

    "canal_alignment_distance",

    "disc_alignment_distance",

    "overall_alignment_score",

    "foreground_retention",

    "vertebrae_retention",

    "canal_retention",

    "disc_retention",
]

ranking_df = (
    results_df[
        ranking_columns
    ]
    .sort_values(
        "overall_alignment_score"
    )
)

ranking_csv = (
    OUTPUT_DIR
    / "t2_space_alignment_ranking.csv"
)

ranking_df.to_csv(
    ranking_csv,
    index=False
)

print(
    f"Saved: {ranking_csv}"
)


# ============================================================
# SAFE JSON
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

        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32
        )
    ):

        if np.isnan(
            value
        ):

            return None

        return float(
            value
        )

    if isinstance(
        value,
        np.bool_
    ):

        return bool(
            value
        )

    return value


# ============================================================
# JSON SUMMARY
# ============================================================

failure_df = results_df[
    results_df["group"]
    == "Failure"
]

successful_df = results_df[
    results_df["group"]
    == "Successful"
]

summary = {

    "phase":
        "Phase 3 - Part 22",

    "purpose":
        "T2 SPACE anatomy localization and crop alignment analysis",

    "target_shape":
        list(
            TARGET_SHAPE
        ),

    "total_cases":
        int(
            len(results_df)
        ),

    "failure_cases":
        FAILURE_CASES,

    "successful_cases":
        SUCCESS_CASES,

    "failure_count":
        int(
            len(failure_df)
        ),

    "successful_count":
        int(
            len(successful_df)
        ),

    "failure_mean_dice":
        safe_float(
            failure_df[
                "model_dice"
            ].mean()
        ),

    "successful_mean_dice":
        safe_float(
            successful_df[
                "model_dice"
            ].mean()
        ),

    "failure_mean_canal_alignment":
        safe_float(
            failure_df[
                "canal_alignment_distance"
            ].mean()
        ),

    "successful_mean_canal_alignment":
        safe_float(
            successful_df[
                "canal_alignment_distance"
            ].mean()
        ),

    "failure_mean_vertebrae_alignment":
        safe_float(
            failure_df[
                "vertebrae_alignment_distance"
            ].mean()
        ),

    "successful_mean_vertebrae_alignment":
        safe_float(
            successful_df[
                "vertebrae_alignment_distance"
            ].mean()
        ),

    "failure_mean_overall_alignment":
        safe_float(
            failure_df[
                "overall_alignment_score"
            ].mean()
        ),

    "successful_mean_overall_alignment":
        safe_float(
            successful_df[
                "overall_alignment_score"
            ].mean()
        ),

    "correlations":
        correlation_df.to_dict(
            orient="records"
        ),

    "representative_cases":
        representative_cases,

    "output_directory":
        str(
            OUTPUT_DIR
        ),
}

summary = make_json_safe(
    summary
)

json_path = (
    OUTPUT_DIR
    / "t2_space_anatomy_alignment_summary.json"
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
    / "phase3_part22_anatomy_alignment_report.txt"
)

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
        "PHASE 3 - PART 22\n"
    )

    file.write(
        "T2 SPACE ANATOMY LOCALIZATION & CROP ALIGNMENT ANALYSIS\n"
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
        "Investigate whether the spatial position of "
        "vertebrae, spinal canal, discs and foreground "
        "anatomy relative to the fixed 96 x 96 x 96 "
        "model crop is associated with segmentation "
        "performance.\n\n"
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
        "ALIGNMENT SUMMARY\n"
    )

    file.write(
        "-" * 78
        + "\n"
    )

    file.write(
        f"Failure mean canal alignment: "
        f"{failure_df['canal_alignment_distance'].mean():.6f}\n"
    )

    file.write(
        f"Successful mean canal alignment: "
        f"{successful_df['canal_alignment_distance'].mean():.6f}\n"
    )

    file.write(
        f"Failure mean vertebrae alignment: "
        f"{failure_df['vertebrae_alignment_distance'].mean():.6f}\n"
    )

    file.write(
        f"Successful mean vertebrae alignment: "
        f"{successful_df['vertebrae_alignment_distance'].mean():.6f}\n"
    )

    file.write(
        f"Failure mean overall alignment: "
        f"{failure_df['overall_alignment_score'].mean():.6f}\n"
    )

    file.write(
        f"Successful mean overall alignment: "
        f"{successful_df['overall_alignment_score'].mean():.6f}\n\n"
    )

    file.write(
        "CASE-BY-CASE RESULTS\n"
    )

    file.write(
        "-" * 78
        + "\n"
    )

    file.write(
        ranking_df.to_string(
            index=False
        )
    )

    file.write(
        "\n\nCORRELATIONS WITH DICE\n"
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
        "Therefore, correlations and group differences "
        "are exploratory and must not be interpreted "
        "as causal evidence.\n"
    )

    file.write(
        "Part 22 does not modify the trained model, "
        "checkpoint, dataset or preprocessing pipeline.\n"
    )

print(
    f"Saved: {report_path}"
)


# ============================================================
# FINAL
# ============================================================

elapsed = (
    time.time()
    - START_TIME
)

print()
print("=" * 78)
print("PART 22 COMPLETE")
print("=" * 78)

print(
    f"Cases analyzed : "
    f"{len(results_df)}"
)

print(
    f"Failure cases  : "
    f"{len(failure_df)}"
)

print(
    f"Successful cases : "
    f"{len(successful_df)}"
)

print(
    f"Execution time : "
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
print("PHASE 3 - PART 22 COMPLETE")
print("=" * 78)