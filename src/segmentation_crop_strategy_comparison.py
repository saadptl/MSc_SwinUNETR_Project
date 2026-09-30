"""
PHASE 3 - PART 25
CROP STRATEGY COMPARISON

Purpose
-------
Compare different 96 x 96 x 96 crop strategies for the T2 SPACE
cases identified during Parts 19-24.

Strategies
----------
1. Fixed center crop
2. Foreground-centered crop
3. Vertebra-centered crop
4. Spinal-canal-centered crop
5. Disc-centered crop
6. Combined anatomy-centered crop

IMPORTANT
---------
This is an analysis-only experiment.

No model is loaded.
No model weights are changed.
No training is performed.

Ground-truth masks are used only to estimate the theoretical
retention capability of each crop strategy. Therefore these
results are feasibility / upper-bound results and are NOT
deployable inference results.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
)

PART24_DIR = (
    OUTPUT_ROOT
    / "adaptive_crop_analysis"
)

PART22_DIR = (
    OUTPUT_ROOT
    / "t2_space_anatomy_alignment"
)

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "segmentation_split"
    / "test"
)

IMAGE_DIR = TEST_DIR / "images"
MASK_DIR = TEST_DIR / "masks"

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "crop_strategy_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# INPUT
# ============================================================================

PART24_FILE = (
    PART24_DIR
    / "adaptive_crop_case_analysis.csv"
)

PART22_FILE = (
    PART22_DIR
    / "t2_space_anatomy_alignment_case_analysis.csv"
)


# ============================================================================
# TARGET INPUT SIZE
# ============================================================================

TARGET_SHAPE = np.array(
    [96, 96, 96],
    dtype=int
)


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

CLASS_IDS = {
    "vertebrae": 1,
    "spinal_canal": 2,
    "intervertebral_disc": 3,
}


# ============================================================================
# HELPERS
# ============================================================================

def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_mha(path):
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


def center_crop_or_pad(
    array,
    target_shape=TARGET_SHAPE
):
    """
    Center crop or zero-pad a 3-D array.

    Array convention:
        Z, Y, X
    """

    target_shape = np.asarray(
        target_shape,
        dtype=int
    )

    output = np.zeros(
        target_shape,
        dtype=array.dtype
    )

    input_shape = np.asarray(
        array.shape,
        dtype=int
    )

    src_start = np.maximum(
        (input_shape - target_shape) // 2,
        0
    )

    src_end = np.minimum(
        src_start + target_shape,
        input_shape
    )

    dst_start = np.maximum(
        (target_shape - input_shape) // 2,
        0
    )

    dst_end = (
        dst_start
        + (src_end - src_start)
    )

    output[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = array[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]

    return output


def crop_using_center(
    array,
    center,
    target_shape=TARGET_SHAPE
):
    """
    Crop/pad around a supplied center.

    center convention:
        Z, Y, X
    """

    target_shape = np.asarray(
        target_shape,
        dtype=int
    )

    center = np.asarray(
        center,
        dtype=float
    )

    center = np.round(
        center
    ).astype(int)

    output = np.zeros(
        target_shape,
        dtype=array.dtype
    )

    input_shape = np.asarray(
        array.shape,
        dtype=int
    )

    start = (
        center
        - target_shape // 2
    )

    end = (
        start
        + target_shape
    )

    src_start = np.maximum(
        start,
        0
    )

    src_end = np.minimum(
        end,
        input_shape
    )

    dst_start = np.maximum(
        -start,
        0
    )

    dst_end = (
        dst_start
        + (src_end - src_start)
    )

    output[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = array[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]

    return output


def get_centroid(
    binary_mask
):
    """
    Centroid of a binary mask.
    """

    coordinates = np.argwhere(
        binary_mask
    )

    if len(coordinates) == 0:
        return None

    return coordinates.mean(
        axis=0
    )


def get_bbox_center(
    binary_mask
):
    """
    Center of bounding box of a binary mask.
    """

    coordinates = np.argwhere(
        binary_mask
    )

    if len(coordinates) == 0:
        return None

    minimum = coordinates.min(
        axis=0
    )

    maximum = coordinates.max(
        axis=0
    )

    return (
        minimum
        + maximum
    ) / 2.0


def calculate_retention(
    original_mask,
    cropped_mask,
    class_id
):
    original_count = np.sum(
        original_mask == class_id
    )

    cropped_count = np.sum(
        cropped_mask == class_id
    )

    if original_count == 0:
        return np.nan

    return float(
        cropped_count
        / original_count
    )


def calculate_all_retention(
    original_mask,
    cropped_mask
):
    results = {}

    for name, class_id in CLASS_IDS.items():

        results[name] = calculate_retention(
            original_mask,
            cropped_mask,
            class_id
        )

    original_foreground = np.sum(
        original_mask > 0
    )

    cropped_foreground = np.sum(
        cropped_mask > 0
    )

    if original_foreground > 0:

        results["foreground"] = float(
            cropped_foreground
            / original_foreground
        )

    else:

        results["foreground"] = np.nan

    return results


def combined_score(
    retention
):
    """
    Balanced anatomical retention score.

    Equal weight is given to:
        Vertebrae
        Spinal Canal
        Intervertebral Disc

    Foreground is reported separately.
    """

    values = [
        retention["vertebrae"],
        retention["spinal_canal"],
        retention["intervertebral_disc"],
    ]

    values = [
        value
        for value in values
        if not pd.isna(value)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def safe_mean(series):
    return float(
        pd.to_numeric(
            series,
            errors="coerce"
        ).mean()
    )


# ============================================================================
# START
# ============================================================================

print_header(
    "PHASE 3 - PART 25"
)

print(
    "CROP STRATEGY COMPARISON"
)

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


# ============================================================================
# LOAD CASE LIST
# ============================================================================

print_header(
    "LOADING PREVIOUS ANALYSIS"
)

if PART24_FILE.exists():

    part24 = pd.read_csv(
        PART24_FILE
    )

    print(
        f"Part 24 cases: {len(part24)}"
    )

    if "file" in part24.columns:

        cases = (
            part24["file"]
            .astype(str)
            .tolist()
        )

    else:

        raise RuntimeError(
            "Part 24 file does not contain "
            "'file' column."
        )

elif PART22_FILE.exists():

    part22 = pd.read_csv(
        PART22_FILE
    )

    print(
        f"Part 22 cases: {len(part22)}"
    )

    cases = (
        part22["file"]
        .astype(str)
        .tolist()
    )

else:

    raise FileNotFoundError(
        "Neither Part 24 nor Part 22 "
        "analysis file was found."
    )


print()
print(
    f"Cases selected: {len(cases)}"
)


# ============================================================================
# MODEL DICE LOOKUP
# ============================================================================

dice_lookup = {}

if PART22_FILE.exists():

    part22 = pd.read_csv(
        PART22_FILE
    )

    if "file" in part22.columns:

        possible_dice_columns = [
            "model_dice",
            "mean_foreground_dice",
            "dice",
        ]

        dice_column = None

        for candidate in possible_dice_columns:

            if candidate in part22.columns:

                dice_column = candidate
                break

        if dice_column is not None:

            dice_lookup = dict(
                zip(
                    part22["file"].astype(str),
                    pd.to_numeric(
                        part22[dice_column],
                        errors="coerce"
                    )
                )
            )


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

print_header(
    "CROP STRATEGIES"
)

strategies = [
    "fixed_center",
    "foreground_centered",
    "vertebra_centered",
    "canal_centered",
    "disc_centered",
    "combined_anatomy",
]

for strategy in strategies:
    print(
        f"✓ {strategy}"
    )


# ============================================================================
# CASE ANALYSIS
# ============================================================================

print_header(
    "RUNNING CROP STRATEGY COMPARISON"
)

rows = []


for index, case in enumerate(
    cases,
    start=1
):

    print(
        f"Analyzing {index:2d} / "
        f"{len(cases)} : {case}"
    )

    image_path = (
        IMAGE_DIR
        / f"{case}.mha"
    )

    mask_path = (
        MASK_DIR
        / f"{case}.mha"
    )

    if not image_path.exists():

        print(
            "  WARNING: image missing"
        )

        continue

    if not mask_path.exists():

        print(
            "  WARNING: mask missing"
        )

        continue

    try:

        image = load_mha(
            image_path
        )

        mask = load_mha(
            mask_path
        )

    except Exception as exc:

        print(
            f"  ERROR: {exc}"
        )

        continue


    # ------------------------------------------------------------------------
    # CLASS MASKS
    # ------------------------------------------------------------------------

    vertebra_mask = (
        mask == CLASS_IDS["vertebrae"]
    )

    canal_mask = (
        mask == CLASS_IDS["spinal_canal"]
    )

    disc_mask = (
        mask == CLASS_IDS["intervertebral_disc"]
    )

    foreground_mask = (
        mask > 0
    )


    # ------------------------------------------------------------------------
    # CENTERS
    # ------------------------------------------------------------------------

    foreground_center = get_bbox_center(
        foreground_mask
    )

    vertebra_center = get_bbox_center(
        vertebra_mask
    )

    canal_center = get_bbox_center(
        canal_mask
    )

    disc_center = get_bbox_center(
        disc_mask
    )


    # ------------------------------------------------------------------------
    # COMBINED ANATOMY CENTER
    # ------------------------------------------------------------------------
    #
    # We calculate the center using the centroids of the three
    # anatomical classes rather than the full foreground bbox.
    #
    # This avoids allowing a single large structure to dominate.
    # ------------------------------------------------------------------------

    class_centers = []

    for class_mask in [
        vertebra_mask,
        canal_mask,
        disc_mask,
    ]:

        center = get_bbox_center(
            class_mask
        )

        if center is not None:

            class_centers.append(
                center
            )

    if len(class_centers) > 0:

        combined_center = np.mean(
            np.vstack(
                class_centers
            ),
            axis=0
        )

    else:

        combined_center = foreground_center


    # ------------------------------------------------------------------------
    # STRATEGY CENTERS
    # ------------------------------------------------------------------------

    strategy_centers = {

        "fixed_center":
            np.asarray(
                mask.shape
            ) / 2.0,

        "foreground_centered":
            foreground_center,

        "vertebra_centered":
            vertebra_center,

        "canal_centered":
            canal_center,

        "disc_centered":
            disc_center,

        "combined_anatomy":
            combined_center,
    }


    # ------------------------------------------------------------------------
    # PROCESS EACH STRATEGY
    # ------------------------------------------------------------------------

    for strategy, center in strategy_centers.items():

        if center is None:

            print(
                f"  WARNING: {strategy} "
                f"has no valid center."
            )

            continue


        cropped_mask = crop_using_center(
            mask,
            center
        )


        retention = calculate_all_retention(
            mask,
            cropped_mask
        )


        score = combined_score(
            retention
        )


        rows.append({

            "file":
                case,

            "model_dice":
                dice_lookup.get(
                    case,
                    np.nan
                ),

            "strategy":
                strategy,

            "foreground_retention":
                retention["foreground"],

            "vertebrae_retention":
                retention["vertebrae"],

            "spinal_canal_retention":
                retention["spinal_canal"],

            "intervertebral_disc_retention":
                retention[
                    "intervertebral_disc"
                ],

            "combined_anatomical_retention":
                score,

            "center_z":
                float(center[0]),

            "center_y":
                float(center[1]),

            "center_x":
                float(center[2]),

            "original_z":
                int(image.shape[0]),

            "original_y":
                int(image.shape[1]),

            "original_x":
                int(image.shape[2]),
        })


# ============================================================================
# DATAFRAME
# ============================================================================

results = pd.DataFrame(
    rows
)

if len(results) == 0:

    raise RuntimeError(
        "No crop strategy results were generated."
    )


# ============================================================================
# NUMERIC CLEANUP
# ============================================================================

numeric_columns = [
    "model_dice",
    "foreground_retention",
    "vertebrae_retention",
    "spinal_canal_retention",
    "intervertebral_disc_retention",
    "combined_anatomical_retention",
    "center_z",
    "center_y",
    "center_x",
]

for column in numeric_columns:

    results[column] = pd.to_numeric(
        results[column],
        errors="coerce"
    )


# ============================================================================
# SUMMARY BY STRATEGY
# ============================================================================

print_header(
    "STRATEGY PERFORMANCE SUMMARY"
)

summary_rows = []

for strategy in strategies:

    subset = results[
        results["strategy"]
        == strategy
    ]

    if len(subset) == 0:
        continue

    summary_rows.append({

        "strategy":
            strategy,

        "cases":
            len(subset),

        "mean_foreground_retention":
            safe_mean(
                subset[
                    "foreground_retention"
                ]
            ),

        "mean_vertebrae_retention":
            safe_mean(
                subset[
                    "vertebrae_retention"
                ]
            ),

        "mean_spinal_canal_retention":
            safe_mean(
                subset[
                    "spinal_canal_retention"
                ]
            ),

        "mean_disc_retention":
            safe_mean(
                subset[
                    "intervertebral_disc_retention"
                ]
            ),

        "mean_combined_anatomical_retention":
            safe_mean(
                subset[
                    "combined_anatomical_retention"
                ]
            ),

        "median_combined_anatomical_retention":
            float(
                subset[
                    "combined_anatomical_retention"
                ].median()
            ),

        "cases_canal_below_0.02":
            int(
                (
                    subset[
                        "spinal_canal_retention"
                    ]
                    < 0.02
                ).sum()
            ),

        "cases_combined_above_0.20":
            int(
                (
                    subset[
                        "combined_anatomical_retention"
                    ]
                    >= 0.20
                ).sum()
            ),
    })


summary = pd.DataFrame(
    summary_rows
)

summary = summary.sort_values(
    "mean_combined_anatomical_retention",
    ascending=False
).reset_index(
    drop=True
)

summary.insert(
    0,
    "rank",
    np.arange(
        1,
        len(summary) + 1
    )
)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================================
# BEST STRATEGY
# ============================================================================

print_header(
    "BEST CROP STRATEGY"
)

best_strategy = (
    summary.iloc[0]["strategy"]
)

best_score = (
    summary.iloc[0]
    ["mean_combined_anatomical_retention"]
)

print(
    f"Best strategy by combined anatomical retention:"
)

print(
    f"  {best_strategy}"
)

print(
    f"Mean combined retention:"
    f" {best_score:.6f}"
)


# ============================================================================
# COMPARE AGAINST FIXED CENTER
# ============================================================================

fixed_row = summary[
    summary["strategy"]
    == "fixed_center"
]

if len(fixed_row) > 0:

    fixed_score = float(
        fixed_row.iloc[0][
            "mean_combined_anatomical_retention"
        ]
    )

    print()

    print(
        f"Fixed center combined retention:"
        f" {fixed_score:.6f}"
    )

    print(
        f"Best strategy improvement:"
        f" {best_score - fixed_score:+.6f}"
    )


# ============================================================================
# SPINAL CANAL COMPARISON
# ============================================================================

print_header(
    "SPINAL CANAL RETENTION COMPARISON"
)

canal_table = summary[
    [
        "rank",
        "strategy",
        "mean_spinal_canal_retention",
        "cases_canal_below_0.02",
    ]
]

print(
    canal_table.to_string(
        index=False
    )
)


# ============================================================================
# ANATOMICAL BALANCE SCORE
# ============================================================================

print_header(
    "ANATOMICAL BALANCE ANALYSIS"
)

"""
A crop that retains only the canal is not sufficient.

We therefore calculate a conservative balanced score:

minimum(
    vertebrae retention,
    canal retention,
    disc retention
)

This identifies strategies that retain all three structures
reasonably well rather than optimizing only one structure.
"""

results[
    "minimum_class_retention"
] = results[
    [
        "vertebrae_retention",
        "spinal_canal_retention",
        "intervertebral_disc_retention",
    ]
].min(
    axis=1
)


balance_summary = []

for strategy in strategies:

    subset = results[
        results["strategy"]
        == strategy
    ]

    if len(subset) == 0:
        continue

    balance_summary.append({

        "strategy":
            strategy,

        "mean_minimum_class_retention":
            float(
                subset[
                    "minimum_class_retention"
                ].mean()
            ),

        "median_minimum_class_retention":
            float(
                subset[
                    "minimum_class_retention"
                ].median()
            ),
    })


balance_summary = pd.DataFrame(
    balance_summary
).sort_values(
    "mean_minimum_class_retention",
    ascending=False
).reset_index(
    drop=True
)

balance_summary.insert(
    0,
    "rank",
    np.arange(
        1,
        len(balance_summary) + 1
    )
)

print(
    balance_summary.to_string(
        index=False
    )
)


# ============================================================================
# SAVE RESULTS
# ============================================================================

print_header(
    "SAVING ANALYSIS TABLES"
)

case_path = (
    OUTPUT_DIR
    / "crop_strategy_case_analysis.csv"
)

results.to_csv(
    case_path,
    index=False
)

print(
    f"Saved: {case_path}"
)


summary_path = (
    OUTPUT_DIR
    / "crop_strategy_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    f"Saved: {summary_path}"
)


balance_path = (
    OUTPUT_DIR
    / "crop_strategy_balance_summary.csv"
)

balance_summary.to_csv(
    balance_path,
    index=False
)

print(
    f"Saved: {balance_path}"
)


# ============================================================================
# CHART 1
# ============================================================================

print_header(
    "CREATING CHARTS"
)

plot_data = summary.copy()

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    plot_data["strategy"],
    plot_data[
        "mean_combined_anatomical_retention"
    ]
)

plt.xlabel(
    "Crop Strategy"
)

plt.ylabel(
    "Mean Combined Anatomical Retention"
)

plt.title(
    "Crop Strategy Comparison"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "crop_strategy_combined_retention.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    f"Saved: {path}"
)


# ============================================================================
# CHART 2
# ============================================================================

plt.figure(
    figsize=(10, 6)
)

x = np.arange(
    len(summary)
)

width = 0.25

plt.bar(
    x - width,
    summary[
        "mean_vertebrae_retention"
    ],
    width,
    label="Vertebrae"
)

plt.bar(
    x,
    summary[
        "mean_spinal_canal_retention"
    ],
    width,
    label="Spinal Canal"
)

plt.bar(
    x + width,
    summary[
        "mean_disc_retention"
    ],
    width,
    label="Intervertebral Disc"
)

plt.xlabel(
    "Crop Strategy"
)

plt.ylabel(
    "Mean Retention"
)

plt.title(
    "Anatomical Retention by Crop Strategy"
)

plt.xticks(
    x,
    summary["strategy"],
    rotation=35,
    ha="right"
)

plt.legend()

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "crop_strategy_class_retention.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    f"Saved: {path}"
)


# ============================================================================
# CHART 3
# ============================================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    summary["strategy"],
    summary[
        "cases_canal_below_0.02"
    ]
)

plt.xlabel(
    "Crop Strategy"
)

plt.ylabel(
    "Number of Cases"
)

plt.title(
    "Cases with Spinal Canal Retention < 0.02"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "crop_strategy_canal_failures.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    f"Saved: {path}"
)


# ============================================================================
# CHART 4
# ============================================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    balance_summary["strategy"],
    balance_summary[
        "mean_minimum_class_retention"
    ]
)

plt.xlabel(
    "Crop Strategy"
)

plt.ylabel(
    "Mean Minimum Class Retention"
)

plt.title(
    "Balanced Anatomical Retention by Crop Strategy"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "crop_strategy_balanced_retention.png"
)

plt.savefig(
    path,
    dpi=200
)

plt.close()

print(
    f"Saved: {path}"
)


# ============================================================================
# JSON SUMMARY
# ============================================================================

print_header(
    "CREATING FINAL SUMMARY"
)

best_balance_strategy = (
    balance_summary.iloc[0]["strategy"]
)

best_balance_score = float(
    balance_summary.iloc[0][
        "mean_minimum_class_retention"
    ]
)

summary_json = {

    "part":
        "Phase 3 - Part 25",

    "purpose":
        "Compare candidate 96 x 96 x 96 crop strategies "
        "for T2 SPACE anatomical retention.",

    "cases_analyzed":
        int(
            results["file"].nunique()
        ),

    "strategies":
        strategies,

    "best_strategy_by_combined_retention":
        str(
            best_strategy
        ),

    "best_combined_retention":
        float(
            best_score
        ),

    "best_strategy_by_balanced_retention":
        str(
            best_balance_strategy
        ),

    "best_balanced_retention":
        best_balance_score,

    "strategy_summary":
        summary.to_dict(
            orient="records"
        ),

    "balance_summary":
        balance_summary.to_dict(
            orient="records"
        ),

    "methodological_note":
        "Ground-truth masks were used to estimate crop "
        "retention. Results represent feasibility and "
        "upper-bound preprocessing analysis, not "
        "deployable inference performance.",

    "training_performed":
        False,

    "model_weights_modified":
        False,
}


json_path = (
    OUTPUT_DIR
    / "phase3_part25_crop_strategy_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary_json,
        file,
        indent=4
    )

print(
    f"Saved: {json_path}"
)


# ============================================================================
# TEXT REPORT
# ============================================================================

report_path = (
    OUTPUT_DIR
    / "phase3_part25_crop_strategy_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "=" * 78 + "\n"
    )

    file.write(
        "PHASE 3 - PART 25\n"
    )

    file.write(
        "CROP STRATEGY COMPARISON\n"
    )

    file.write(
        "=" * 78 + "\n\n"
    )

    file.write(
        "PURPOSE\n"
    )

    file.write(
        "Compare candidate crop strategies for T2 SPACE "
        "anatomical retention before retraining.\n\n"
    )

    file.write(
        "STRATEGIES\n"
    )

    for strategy in strategies:

        file.write(
            f"- {strategy}\n"
        )

    file.write("\n")

    file.write(
        f"Cases analyzed: "
        f"{results['file'].nunique()}\n\n"
    )

    file.write(
        "BEST STRATEGY BY COMBINED RETENTION\n"
    )

    file.write(
        f"{best_strategy}\n"
    )

    file.write(
        f"Mean combined anatomical retention: "
        f"{best_score:.6f}\n\n"
    )

    file.write(
        "BEST STRATEGY BY BALANCED RETENTION\n"
    )

    file.write(
        f"{best_balance_strategy}\n"
    )

    file.write(
        f"Mean minimum-class retention: "
        f"{best_balance_score:.6f}\n\n"
    )

    file.write(
        "STRATEGY SUMMARY\n"
    )

    file.write(
        summary.to_string(
            index=False
        )
    )

    file.write("\n\n")

    file.write(
        "BALANCED RETENTION SUMMARY\n"
    )

    file.write(
        balance_summary.to_string(
            index=False
        )
    )

    file.write("\n\n")

    file.write(
        "IMPORTANT LIMITATION\n"
    )

    file.write(
        "Ground-truth masks were used to determine "
        "anatomical centers. Therefore this experiment "
        "does not represent a deployable inference "
        "pipeline. It only establishes which crop "
        "strategies are promising enough to investigate "
        "in a future training experiment.\n"
    )

print(
    f"Saved: {report_path}"
)


# ============================================================================
# FINAL
# ============================================================================

print_header(
    "PART 25 COMPLETE"
)

print(
    f"Cases analyzed: "
    f"{results['file'].nunique()}"
)

print()

print(
    "Best combined-retention strategy:"
)

print(
    f"  {best_strategy}"
)

print(
    f"  Score: {best_score:.6f}"
)

print()

print(
    "Best balanced-retention strategy:"
)

print(
    f"  {best_balance_strategy}"
)

print(
    f"  Score: {best_balance_score:.6f}"
)

print()

print(
    "Training performed: NO"
)

print(
    "Model weights modified: NO"
)

print()
print("=" * 78)
print("OUTPUT DIRECTORY")
print("=" * 78)
print(OUTPUT_DIR)

print()
print("=" * 78)
print("PHASE 3 - PART 25 COMPLETE")
print("=" * 78)