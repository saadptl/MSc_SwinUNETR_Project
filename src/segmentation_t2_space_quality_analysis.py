"""
PHASE 3 - PART 20
T2 SPACE IMAGE QUALITY & INTENSITY ANALYSIS

Purpose
-------
Compare successful and failed T2 SPACE MRI cases using:

    - Image dimensions
    - Number of slices
    - Intensity statistics
    - Percentile statistics
    - Intensity range
    - Standard deviation
    - Coefficient of variation
    - Non-zero voxel statistics
    - Normalized intensity statistics
    - Spinal-canal ground-truth volume
    - Spinal-canal volume percentage

This analysis is diagnostic only.

IMPORTANT
---------
This script DOES NOT train the model.
This script DOES NOT modify the checkpoint.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "t2_space_quality_analysis"
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
# HEADER
# ============================================================

print("=" * 78)
print("PHASE 3 - PART 20")
print("T2 SPACE IMAGE QUALITY & INTENSITY ANALYSIS")
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
print("OUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================
# VALIDATE PATHS
# ============================================================

required_paths = [
    TEST_DIR,
    IMAGE_DIR,
    MASK_DIR,
    PART11_RESULTS,
    PART18_RESULTS,
    PART19_RESULTS,
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

print(
    f"Part 11 cases : {len(part11)}"
)

print(
    f"Part 18 cases : {len(part18)}"
)

print(
    f"Part 19 cases : {len(part19)}"
)


# ============================================================
# FILE FINDER
# ============================================================

def find_case_file(
    directory,
    case_name
):

    exact = directory / (
        case_name + ".mha"
    )

    if exact.exists():

        return exact

    exact = directory / (
        case_name + ".mhd"
    )

    if exact.exists():

        return exact

    for path in directory.iterdir():

        if not path.is_file():
            continue

        if path.stem == case_name:

            return path

    for path in directory.iterdir():

        if not path.is_file():
            continue

        if path.stem.startswith(
            case_name
        ):

            return path

    return None


# ============================================================
# LOAD IMAGE
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

        array = sitk.GetArrayFromImage(
            image
        )

        return np.asarray(
            array,
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

        if len(keys) == 0:

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
# LOAD MASK
# ============================================================

def load_mask(path):

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

        array = sitk.GetArrayFromImage(
            image
        )

        return np.asarray(
            array
        )

    if suffix == ".npy":

        return np.load(
            path
        )

    if suffix == ".npz":

        data = np.load(path)

        keys = list(
            data.keys()
        )

        if len(keys) == 0:

            raise ValueError(
                f"Empty NPZ file: {path}"
            )

        return data[keys[0]]

    raise ValueError(
        f"Unsupported mask format: {path}"
    )


# ============================================================
# NORMALIZE
# ============================================================

def percentile_normalize(
    image
):

    image = image.astype(
        np.float32
    )

    finite = image[
        np.isfinite(image)
    ]

    if finite.size == 0:

        return np.zeros_like(
            image,
            dtype=np.float32
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
            image,
            dtype=np.float32
        )

    clipped = np.clip(
        image,
        low,
        high
    )

    normalized = (
        clipped - low
    ) / (
        high - low
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# ROBUST IMAGE STATISTICS
# ============================================================

def calculate_image_statistics(
    image
):

    image = image.astype(
        np.float32
    )

    finite = image[
        np.isfinite(image)
    ]

    if finite.size == 0:

        raise ValueError(
            "Image contains no finite values."
        )

    nonzero = finite[
        finite != 0
    ]

    if nonzero.size == 0:

        nonzero = finite

    normalized = percentile_normalize(
        finite
    )

    p01 = float(
        np.percentile(
            finite,
            1
        )
    )

    p05 = float(
        np.percentile(
            finite,
            5
        )
    )

    p10 = float(
        np.percentile(
            finite,
            10
        )
    )

    p25 = float(
        np.percentile(
            finite,
            25
        )
    )

    p50 = float(
        np.percentile(
            finite,
            50
        )
    )

    p75 = float(
        np.percentile(
            finite,
            75
        )
    )

    p90 = float(
        np.percentile(
            finite,
            90
        )
    )

    p95 = float(
        np.percentile(
            finite,
            95
        )
    )

    p99 = float(
        np.percentile(
            finite,
            99
        )
    )

    mean = float(
        np.mean(finite)
    )

    std = float(
        np.std(finite)
    )

    minimum = float(
        np.min(finite)
    )

    maximum = float(
        np.max(finite)
    )

    median = float(
        np.median(finite)
    )

    intensity_range = (
        maximum
        - minimum
    )

    percentile_range = (
        p99
        - p01
    )

    if abs(mean) > 1e-12:

        coefficient_variation = (
            std
            / abs(mean)
        )

    else:

        coefficient_variation = 0.0

    normalized_mean = float(
        np.mean(normalized)
    )

    normalized_std = float(
        np.std(normalized)
    )

    normalized_p25 = float(
        np.percentile(
            normalized,
            25
        )
    )

    normalized_p50 = float(
        np.percentile(
            normalized,
            50
        )
    )

    normalized_p75 = float(
        np.percentile(
            normalized,
            75
        )
    )

    return {

        "intensity_min":
            minimum,

        "intensity_max":
            maximum,

        "intensity_mean":
            mean,

        "intensity_std":
            std,

        "intensity_median":
            median,

        "intensity_range":
            intensity_range,

        "intensity_p01":
            p01,

        "intensity_p05":
            p05,

        "intensity_p10":
            p10,

        "intensity_p25":
            p25,

        "intensity_p50":
            p50,

        "intensity_p75":
            p75,

        "intensity_p90":
            p90,

        "intensity_p95":
            p95,

        "intensity_p99":
            p99,

        "percentile_range_1_99":
            percentile_range,

        "coefficient_variation":
            coefficient_variation,

        "normalized_mean":
            normalized_mean,

        "normalized_std":
            normalized_std,

        "normalized_p25":
            normalized_p25,

        "normalized_p50":
            normalized_p50,

        "normalized_p75":
            normalized_p75,
    }


# ============================================================
# SPINAL CANAL STATISTICS
# ============================================================

def calculate_spinal_canal_statistics(
    mask,
    image_shape
):

    spinal_mask = (
        mask == 2
    )

    spinal_voxels = int(
        spinal_mask.sum()
    )

    total_voxels = int(
        np.prod(image_shape)
    )

    if total_voxels > 0:

        spinal_percentage = (
            100.0
            * spinal_voxels
            / total_voxels
        )

    else:

        spinal_percentage = 0.0

    # Slice-wise distribution

    slice_counts = (
        spinal_mask
        .sum(axis=(1, 2))
    )

    nonzero_slices = np.where(
        slice_counts > 0
    )[0]

    if len(nonzero_slices) > 0:

        spinal_slice_count = (
            len(nonzero_slices)
        )

        max_slice_voxels = int(
            np.max(
                slice_counts
            )
        )

        mean_nonzero_slice_voxels = float(
            np.mean(
                slice_counts[
                    slice_counts > 0
                ]
            )
        )

    else:

        spinal_slice_count = 0
        max_slice_voxels = 0
        mean_nonzero_slice_voxels = 0.0

    return {

        "spinal_canal_voxels":
            spinal_voxels,

        "spinal_canal_percentage":
            spinal_percentage,

        "spinal_canal_slice_count":
            spinal_slice_count,

        "spinal_canal_max_slice_voxels":
            max_slice_voxels,

        "spinal_canal_mean_nonzero_slice_voxels":
            mean_nonzero_slice_voxels,
    }


# ============================================================
# IMAGE CONTRAST ESTIMATES
# ============================================================

def calculate_contrast_statistics(
    image,
    mask
):

    normalized = percentile_normalize(
        image
    )

    spinal_mask = (
        mask == 2
    )

    background_mask = (
        mask == 0
    )

    spinal_values = normalized[
        spinal_mask
    ]

    background_values = normalized[
        background_mask
    ]

    if spinal_values.size == 0:

        spinal_mean = np.nan
        spinal_std = np.nan

    else:

        spinal_mean = float(
            np.mean(
                spinal_values
            )
        )

        spinal_std = float(
            np.std(
                spinal_values
            )
        )

    if background_values.size == 0:

        background_mean = np.nan
        background_std = np.nan

    else:

        background_mean = float(
            np.mean(
                background_values
            )
        )

        background_std = float(
            np.std(
                background_values
            )
        )

    if (
        np.isfinite(
            spinal_mean
        )
        and np.isfinite(
            background_mean
        )
    ):

        absolute_contrast = abs(
            spinal_mean
            - background_mean
        )

    else:

        absolute_contrast = np.nan

    if (
        np.isfinite(
            spinal_mean
        )
        and np.isfinite(
            background_mean
        )
        and (
            background_std
            + spinal_std
        ) > 1e-12
    ):

        pooled_std = np.sqrt(
            (
                background_std ** 2
                + spinal_std ** 2
            )
            / 2
        )

        standardized_contrast = (
            abs(
                spinal_mean
                - background_mean
            )
            / pooled_std
        )

    else:

        standardized_contrast = np.nan

    return {

        "spinal_intensity_mean_normalized":
            spinal_mean,

        "spinal_intensity_std_normalized":
            spinal_std,

        "background_intensity_mean_normalized":
            background_mean,

        "background_intensity_std_normalized":
            background_std,

        "spinal_background_absolute_contrast":
            absolute_contrast,

        "spinal_background_standardized_contrast":
            standardized_contrast,
    }


# ============================================================
# PROCESS CASE
# ============================================================

def process_case(
    case_name,
    group
):

    print()
    print(
        f"Processing: {case_name}"
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

        raise FileNotFoundError(
            f"Image not found: "
            f"{case_name}"
        )

    if mask_path is None:

        raise FileNotFoundError(
            f"Mask not found: "
            f"{case_name}"
        )

    print(
        f"Image: {image_path.name}"
    )

    print(
        f"Mask : {mask_path.name}"
    )

    image = load_array(
        image_path
    )

    mask = load_mask(
        mask_path
    )

    if image.shape != mask.shape:

        raise ValueError(
            f"Image/mask shape mismatch "
            f"for {case_name}: "
            f"{image.shape} vs "
            f"{mask.shape}"
        )

    print(
        f"Shape: {image.shape}"
    )

    image_stats = (
        calculate_image_statistics(
            image
        )
    )

    canal_stats = (
        calculate_spinal_canal_statistics(
            mask,
            image.shape
        )
    )

    contrast_stats = (
        calculate_contrast_statistics(
            image,
            mask
        )
    )

    row = {

        "file":
            case_name,

        "group":
            group,

        "height":
            int(image.shape[0]),

        "width":
            int(image.shape[1]),

        "slices":
            int(image.shape[2]),

        "total_voxels":
            int(np.prod(image.shape)),

        **image_stats,

        **canal_stats,

        **contrast_stats,
    }

    print(
        f"Intensity mean: "
        f"{row['intensity_mean']:.6f}"
    )

    print(
        f"Intensity std: "
        f"{row['intensity_std']:.6f}"
    )

    print(
        f"1-99 percentile range: "
        f"{row['percentile_range_1_99']:.6f}"
    )

    print(
        f"Normalized mean: "
        f"{row['normalized_mean']:.6f}"
    )

    print(
        f"Normalized std: "
        f"{row['normalized_std']:.6f}"
    )

    print(
        f"Spinal canal voxels: "
        f"{row['spinal_canal_voxels']}"
    )

    print(
        f"Spinal canal %: "
        f"{row['spinal_canal_percentage']:.6f}"
    )

    print(
        f"Spinal/background contrast: "
        f"{row['spinal_background_absolute_contrast']}"
    )

    return row


# ============================================================
# RUN ANALYSIS
# ============================================================

print()
print("=" * 78)
print("ANALYZING T2 SPACE CASES")
print("=" * 78)

rows = []

for case_name in FAILURE_CASES:

    rows.append(
        process_case(
            case_name,
            "Failure"
        )
    )

for case_name in SUCCESS_CASES:

    rows.append(
        process_case(
            case_name,
            "Successful"
        )
    )


results_df = pd.DataFrame(
    rows
)


# ============================================================
# MERGE PART 19 PERFORMANCE
# ============================================================

print()
print("=" * 78)
print("MERGING MODEL PERFORMANCE")
print("=" * 78)

part19_metrics = part19[
    [
        "file",
        "group",
        "dice",
        "iou",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
    ]
].copy()

part19_metrics = (
    part19_metrics
    .rename(
        columns={
            "dice":
                "visual_dice",
            "iou":
                "visual_iou",
            "precision":
                "visual_precision",
            "recall":
                "visual_recall",
        }
    )
)

results_df = results_df.merge(
    part19_metrics,
    on=[
        "file",
        "group",
    ],
    how="left"
)


# ============================================================
# SAVE MAIN CSV
# ============================================================

main_csv = (
    OUTPUT_DIR
    / "t2_space_quality_case_analysis.csv"
)

results_df.to_csv(
    main_csv,
    index=False
)

print(
    f"Saved: {main_csv}"
)


# ============================================================
# GROUP SUMMARY
# ============================================================

print()
print("=" * 78)
print("GROUP COMPARISON")
print("=" * 78)

numeric_columns = [
    "visual_dice",
    "visual_precision",
    "visual_recall",
    "intensity_mean",
    "intensity_std",
    "intensity_median",
    "percentile_range_1_99",
    "coefficient_variation",
    "normalized_mean",
    "normalized_std",
    "spinal_canal_voxels",
    "spinal_canal_percentage",
    "spinal_background_absolute_contrast",
    "spinal_background_standardized_contrast",
]


group_summary = (
    results_df
    .groupby("group")[
        numeric_columns
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
# SAVE GROUP SUMMARY
# ============================================================

group_csv = (
    OUTPUT_DIR
    / "t2_space_quality_group_summary.csv"
)

group_summary.to_csv(
    group_csv
)

print(
    f"Saved: {group_csv}"
)


# ============================================================
# CORRELATIONS
# ============================================================

print()
print("=" * 78)
print("CORRELATION ANALYSIS")
print("=" * 78)

correlation_columns = [
    "visual_dice",
    "intensity_mean",
    "intensity_std",
    "percentile_range_1_99",
    "coefficient_variation",
    "normalized_mean",
    "normalized_std",
    "spinal_canal_voxels",
    "spinal_canal_percentage",
    "spinal_background_absolute_contrast",
    "spinal_background_standardized_contrast",
]

correlation_df = results_df[
    correlation_columns
].corr(
    method="pearson"
)

print(
    correlation_df[
        ["visual_dice"]
    ]
    .sort_values(
        "visual_dice",
        ascending=False
    )
    .to_string()
)


# ============================================================
# SAVE CORRELATIONS
# ============================================================

correlation_csv = (
    OUTPUT_DIR
    / "t2_space_quality_correlations.csv"
)

correlation_df.to_csv(
    correlation_csv
)

print(
    f"Saved: {correlation_csv}"
)


# ============================================================
# CREATE BOXPLOT HELPER
# ============================================================

def create_group_boxplot(
    dataframe,
    column,
    title,
    ylabel,
    filename
):

    failure_values = dataframe[
        dataframe["group"] == "Failure"
    ][column].dropna().values

    success_values = dataframe[
        dataframe["group"] == "Successful"
    ][column].dropna().values

    if (
        len(failure_values) == 0
        or len(success_values) == 0
    ):

        return

    plt.figure(
        figsize=(8, 6)
    )

    plt.boxplot(
        [
            failure_values,
            success_values,
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

    output_path = (
        OUTPUT_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# BOX PLOTS
# ============================================================

print()
print("=" * 78)
print("CREATING GROUP COMPARISON CHARTS")
print("=" * 78)

create_group_boxplot(
    results_df,
    "intensity_mean",
    "T2 SPACE Intensity Mean: Failure vs Successful",
    "Mean Intensity",
    "intensity_mean_failure_vs_success.png"
)

create_group_boxplot(
    results_df,
    "intensity_std",
    "T2 SPACE Intensity Variation: Failure vs Successful",
    "Intensity Standard Deviation",
    "intensity_std_failure_vs_success.png"
)

create_group_boxplot(
    results_df,
    "percentile_range_1_99",
    "T2 SPACE Intensity Range: Failure vs Successful",
    "1st-99th Percentile Range",
    "percentile_range_failure_vs_success.png"
)

create_group_boxplot(
    results_df,
    "normalized_std",
    "T2 SPACE Normalized Intensity Variation",
    "Normalized Standard Deviation",
    "normalized_std_failure_vs_success.png"
)

create_group_boxplot(
    results_df,
    "spinal_canal_voxels",
    "T2 SPACE Spinal Canal Volume",
    "Spinal Canal Voxels",
    "spinal_canal_volume_failure_vs_success.png"
)

create_group_boxplot(
    results_df,
    "spinal_background_absolute_contrast",
    "T2 SPACE Spinal Canal Contrast",
    "Absolute Normalized Contrast",
    "spinal_canal_contrast_failure_vs_success.png"
)


# ============================================================
# SCATTER PLOTS
# ============================================================

def create_scatter(
    dataframe,
    x_column,
    y_column,
    title,
    xlabel,
    ylabel,
    filename
):

    plt.figure(
        figsize=(8, 6)
    )

    for group_name in [
        "Failure",
        "Successful",
    ]:

        subset = dataframe[
            dataframe["group"]
            == group_name
        ]

        plt.scatter(
            subset[x_column],
            subset[y_column],
            label=group_name,
            s=80
        )

    plt.title(
        title
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        ylabel
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


create_scatter(
    results_df,
    "intensity_std",
    "visual_dice",
    "T2 SPACE Intensity Variation vs Dice",
    "Intensity Standard Deviation",
    "Spinal Canal Dice",
    "intensity_std_vs_dice.png"
)

create_scatter(
    results_df,
    "percentile_range_1_99",
    "visual_dice",
    "T2 SPACE Intensity Range vs Dice",
    "1st-99th Percentile Range",
    "Spinal Canal Dice",
    "percentile_range_vs_dice.png"
)

create_scatter(
    results_df,
    "spinal_canal_voxels",
    "visual_dice",
    "T2 SPACE Canal Volume vs Dice",
    "Spinal Canal Voxels",
    "Spinal Canal Dice",
    "canal_volume_vs_dice.png"
)

create_scatter(
    results_df,
    "spinal_background_absolute_contrast",
    "visual_dice",
    "T2 SPACE Contrast vs Dice",
    "Spinal/Background Contrast",
    "Spinal Canal Dice",
    "contrast_vs_dice.png"
)


# ============================================================
# AUTOMATIC DIFFERENCE ANALYSIS
# ============================================================

print()
print("=" * 78)
print("FAILURE VS SUCCESS DIFFERENCE ANALYSIS")
print("=" * 78)

difference_rows = []

failure_df = results_df[
    results_df["group"]
    == "Failure"
]

success_df = results_df[
    results_df["group"]
    == "Successful"
]

for column in numeric_columns:

    failure_mean = (
        failure_df[column]
        .mean()
    )

    success_mean = (
        success_df[column]
        .mean()
    )

    difference = (
        failure_mean
        - success_mean
    )

    if abs(success_mean) > 1e-12:

        percentage_difference = (
            100.0
            * difference
            / abs(success_mean)
        )

    else:

        percentage_difference = np.nan

    difference_rows.append({

        "feature":
            column,

        "failure_mean":
            failure_mean,

        "successful_mean":
            success_mean,

        "failure_minus_success":
            difference,

        "percentage_difference_vs_success":
            percentage_difference,
    })


difference_df = pd.DataFrame(
    difference_rows
)

difference_csv = (
    OUTPUT_DIR
    / "failure_vs_success_feature_differences.csv"
)

difference_df.to_csv(
    difference_csv,
    index=False
)

print(
    difference_df.to_string(
        index=False
    )
)

print(
    f"Saved: {difference_csv}"
)


# ============================================================
# IMPORTANT STATISTICAL CAUTION
# ============================================================

print()
print("=" * 78)
print("STATISTICAL INTERPRETATION")
print("=" * 78)

print(
    "Only 9 T2 SPACE cases are available."
)

print(
    "Therefore, group differences are exploratory."
)

print(
    "No feature should be interpreted as a causal factor "
    "without additional data or experiments."
)

print(
    "Correlation does not establish causation."
)


# ============================================================
# JSON SUMMARY
# ============================================================

summary = {
    "phase": "Phase 3 - Part 20",

    "purpose":
        "T2 SPACE Image Quality and Intensity Analysis",

    "total_cases":
        int(len(results_df)),

    "failure_cases":
        FAILURE_CASES,

    "successful_cases":
        SUCCESS_CASES,

    "failure_count":
        int(len(failure_df)),

    "successful_count":
        int(len(success_df)),

    "failure_mean_dice":
        float(
            failure_df["visual_dice"].mean()
        ),

    "successful_mean_dice":
        float(
            success_df["visual_dice"].mean()
        ),

    "correlations":
        correlation_df.to_dict(
            orient="index"
        ),

    "output_directory":
        str(OUTPUT_DIR),
}


json_path = (
    OUTPUT_DIR
    / "t2_space_quality_analysis_summary.json"
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
    / "phase3_part20_t2_space_quality_report.txt"
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
        "PHASE 3 - PART 20\n"
    )

    f.write(
        "T2 SPACE IMAGE QUALITY & INTENSITY ANALYSIS\n"
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
        "Part 20 compares successful and failed T2 SPACE "
        "cases using image dimensions, intensity statistics, "
        "normalized intensity statistics, spinal-canal "
        "ground-truth volume and spinal/background contrast.\n\n"
    )

    f.write(
        "CASE GROUPS\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    f.write(
        "Failure cases:\n"
    )

    for case in FAILURE_CASES:

        f.write(
            f"  {case}\n"
        )

    f.write(
        "\nSuccessful cases:\n"
    )

    for case in SUCCESS_CASES:

        f.write(
            f"  {case}\n"
        )

    f.write(
        "\nGROUP PERFORMANCE\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    f.write(
        f"Failure mean Dice: "
        f"{failure_df['visual_dice'].mean():.6f}\n"
    )

    f.write(
        f"Successful mean Dice: "
        f"{success_df['visual_dice'].mean():.6f}\n"
    )

    f.write(
        "\nFEATURE CORRELATIONS WITH DICE\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    dice_correlations = (
        correlation_df[
            ["visual_dice"]
        ]
        .sort_values(
            "visual_dice",
            ascending=False
        )
    )

    f.write(
        dice_correlations.to_string()
    )

    f.write(
        "\n\nSTATISTICAL CAUTION\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    f.write(
        "Only nine T2 SPACE cases are available. "
        "Consequently, this analysis is exploratory and "
        "should not be interpreted as proof of causation. "
        "Observed correlations may be unstable because "
        "of the small sample size.\n"
    )

    f.write(
        "\nMODEL STATUS\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    f.write(
        "No model training was performed in Part 20. "
        "The existing trained model and checkpoints remain "
        "unchanged.\n"
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
print("PART 20 COMPLETE")
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
    f"{len(success_df)}"
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
print("PHASE 3 - PART 20 COMPLETE")
print("=" * 78)