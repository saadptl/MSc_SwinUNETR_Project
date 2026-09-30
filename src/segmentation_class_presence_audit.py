"""
PHASE 3 - PART 16
GROUND-TRUTH CLASS PRESENCE & ANNOTATION AUDIT

Purpose:
    Audit the test-set segmentation masks to determine whether each
    anatomical class is actually present in the ground-truth mask.

Classes:
    0 - Background
    1 - Vertebrae
    2 - Spinal Canal
    3 - Intervertebral Disc

This analysis:
    1. Audits all 71 test masks.
    2. Counts voxels for every class.
    3. Determines class presence/absence.
    4. Identifies empty-class cases.
    5. Separates valid class-evaluation cases from empty-target cases.
    6. Recalculates Dice statistics using only cases where the
       corresponding ground-truth class is present.
    7. Compares the corrected statistics with Part 11.
    8. Produces tables, charts, JSON and a research report.

IMPORTANT:
    This script does NOT modify the model, checkpoint, dataset,
    masks or training results.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


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

TEST_MASKS_DIR = TEST_DIR / "masks"

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
    / "class_presence_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = {
    0: "Background",
    1: "Vertebrae",
    2: "Spinal Canal",
    3: "Intervertebral Disc",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("PHASE 3 - PART 16")
print("GROUND-TRUTH CLASS PRESENCE & ANNOTATION AUDIT")
print("=" * 78)

print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nTEST MASK DIRECTORY")
print(TEST_MASKS_DIR)

print("\nOUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================
# VALIDATE INPUTS
# ============================================================

required_paths = [
    TEST_MASKS_DIR,
    PART11_RESULTS,
    PART14_RESULTS,
]

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )


# ============================================================
# LOAD PART 11 RESULTS
# ============================================================

print("\n" + "=" * 78)
print("LOADING OFFICIAL TEST RESULTS")
print("=" * 78)

part11_df = pd.read_csv(
    PART11_RESULTS
)

print(
    f"Part 11 rows: "
    f"{len(part11_df)}"
)

print(
    f"Part 11 columns: "
    f"{len(part11_df.columns)}"
)


# ============================================================
# LOAD PART 14 RESULTS
# ============================================================

part14_df = pd.read_csv(
    PART14_RESULTS
)

print(
    f"Part 14 rows: "
    f"{len(part14_df)}"
)


# ============================================================
# TEST MASK FILES
# ============================================================

mask_files = sorted(
    TEST_MASKS_DIR.glob("*.mha")
)

print("\n" + "=" * 78)
print("TEST MASK DATASET")
print("=" * 78)

print(
    f"Mask files: "
    f"{len(mask_files)}"
)


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(path):

    image = sitk.ReadImage(
        str(path)
    )

    array = sitk.GetArrayFromImage(
        image
    )

    array = np.asarray(
        array
    )

    array = np.squeeze(
        array
    )

    if array.ndim == 2:

        array = array[
            np.newaxis,
            ...
        ]

    if array.ndim != 3:

        raise ValueError(
            f"Unexpected mask shape "
            f"for {path.name}: "
            f"{array.shape}"
        )

    return array.astype(
        np.int64
    )


# ============================================================
# MASK ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("ANALYZING GROUND-TRUTH MASKS")
print("=" * 78)

records = []

start_time = time.time()

for index, mask_path in enumerate(
    mask_files,
    start=1,
):

    case_name = mask_path.stem

    mask = load_mask(
        mask_path
    )

    unique_labels = np.unique(
        mask
    )

    total_voxels = int(
        mask.size
    )

    record = {
        "file": case_name,
        "shape": str(mask.shape),
        "total_voxels": total_voxels,
        "unique_labels": ",".join(
            str(int(x))
            for x in unique_labels
        ),
    }

    for class_id in (
        1,
        2,
        3,
    ):

        class_name = CLASS_NAMES[
            class_id
        ]

        count = int(
            (
                mask == class_id
            ).sum()
        )

        percentage = (
            count
            / total_voxels
            * 100.0
        )

        present = (
            count > 0
        )

        prefix = (
            class_name
            .lower()
            .replace(" ", "_")
        )

        record[
            f"{prefix}_voxels"
        ] = count

        record[
            f"{prefix}_percentage"
        ] = percentage

        record[
            f"{prefix}_present"
        ] = present

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    background_count = int(
        (
            mask == 0
        ).sum()
    )

    record[
        "background_voxels"
    ] = background_count

    record[
        "background_percentage"
    ] = (
        background_count
        / total_voxels
        * 100.0
    )

    # --------------------------------------------------------
    # Number of foreground classes present
    # --------------------------------------------------------

    foreground_presence = [
        record[
            "vertebrae_present"
        ],
        record[
            "spinal_canal_present"
        ],
        record[
            "intervertebral_disc_present"
        ],
    ]

    record[
        "foreground_classes_present"
    ] = int(
        sum(
            foreground_presence
        )
    )

    records.append(
        record
    )

    if (
        index == 1
        or index % 10 == 0
        or index == len(mask_files)
    ):

        print(
            f"Analyzed "
            f"{index:3d} / "
            f"{len(mask_files)}"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

audit_df = pd.DataFrame(
    records
)

audit_df = audit_df.sort_values(
    "file"
).reset_index(
    drop=True
)


# ============================================================
# CLASS PRESENCE SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("GROUND-TRUTH CLASS PRESENCE")
print("=" * 78)

summary_records = []

for class_id in (
    1,
    2,
    3,
):

    class_name = CLASS_NAMES[
        class_id
    ]

    prefix = (
        class_name
        .lower()
        .replace(" ", "_")
    )

    present_count = int(
        audit_df[
            f"{prefix}_present"
        ].sum()
    )

    absent_count = (
        len(audit_df)
        - present_count
    )

    mean_voxels_all = (
        audit_df[
            f"{prefix}_voxels"
        ].mean()
    )

    present_rows = audit_df[
        audit_df[
            f"{prefix}_present"
        ]
    ]

    if len(present_rows) > 0:

        mean_voxels_present = (
            present_rows[
                f"{prefix}_voxels"
            ].mean()
        )

        median_voxels_present = (
            present_rows[
                f"{prefix}_voxels"
            ].median()
        )

        min_voxels_present = (
            present_rows[
                f"{prefix}_voxels"
            ].min()
        )

        max_voxels_present = (
            present_rows[
                f"{prefix}_voxels"
            ].max()
        )

    else:

        mean_voxels_present = 0
        median_voxels_present = 0
        min_voxels_present = 0
        max_voxels_present = 0

    summary_records.append(
        {
            "class_id": class_id,
            "class": class_name,
            "present_cases": present_count,
            "absent_cases": absent_count,
            "presence_percentage":
                present_count
                / len(audit_df)
                * 100.0,
            "absence_percentage":
                absent_count
                / len(audit_df)
                * 100.0,
            "mean_voxels_all_cases":
                mean_voxels_all,
            "mean_voxels_present_cases":
                mean_voxels_present,
            "median_voxels_present_cases":
                median_voxels_present,
            "minimum_voxels_present_cases":
                min_voxels_present,
            "maximum_voxels_present_cases":
                max_voxels_present,
        }
    )

    print(
        f"\n{class_name}"
    )

    print(
        f"  Present: "
        f"{present_count} "
        f"({present_count / len(audit_df) * 100:.2f}%)"
    )

    print(
        f"  Absent : "
        f"{absent_count} "
        f"({absent_count / len(audit_df) * 100:.2f}%)"
    )

    print(
        f"  Mean voxels when present: "
        f"{mean_voxels_present:,.2f}"
    )

    print(
        f"  Median voxels when present: "
        f"{median_voxels_present:,.2f}"
    )


class_summary_df = pd.DataFrame(
    summary_records
)


# ============================================================
# EMPTY CLASS CASES
# ============================================================

print("\n" + "=" * 78)
print("EMPTY / ABSENT CLASS CASES")
print("=" * 78)

for class_id in (
    1,
    2,
    3,
):

    class_name = CLASS_NAMES[
        class_id
    ]

    prefix = (
        class_name
        .lower()
        .replace(" ", "_")
    )

    absent_cases = audit_df[
        ~audit_df[
            f"{prefix}_present"
        ]
    ]["file"].tolist()

    print(
        f"\n{class_name}"
    )

    print(
        f"Absent cases: "
        f"{len(absent_cases)}"
    )

    if absent_cases:

        print(
            "Cases:"
        )

        print(
            ", ".join(
                absent_cases
            )
        )


# ============================================================
# MERGE WITH PART 11
# ============================================================

print("\n" + "=" * 78)
print("MERGING OFFICIAL TEST METRICS")
print("=" * 78)

merged_df = pd.merge(
    part11_df,
    audit_df,
    on="file",
    how="left",
    validate="one_to_one",
)

if len(merged_df) != len(
    part11_df
):

    raise RuntimeError(
        "Merge changed the number "
        "of Part 11 cases."
    )

print(
    f"Merged cases: "
    f"{len(merged_df)}"
)
# Convert all Part 11 numerical metrics explicitly
numeric_metric_columns = [
    "mean_foreground_dice",
    "mean_foreground_iou",

    "Vertebrae_dice",
    "Vertebrae_iou",
    "Vertebrae_precision",
    "Vertebrae_recall",

    "Spinal Canal_dice",
    "Spinal Canal_iou",
    "Spinal Canal_precision",
    "Spinal Canal_recall",

    "Intervertebral Disc_dice",
    "Intervertebral Disc_iou",
    "Intervertebral Disc_precision",
    "Intervertebral Disc_recall",
]

for col in numeric_metric_columns:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(
            merged_df[col],
            errors="coerce"
        )


# ============================================================
# CLASS-SPECIFIC METRIC AUDIT
# ============================================================

print("\n" + "=" * 78)
print("CLASS-SPECIFIC DICE AUDIT")
print("=" * 78)

corrected_records = []

for class_id, class_name, metric_prefix in [
    (
        1,
        "Vertebrae",
        "Vertebrae_dice",
    ),
    (
        2,
        "Spinal Canal",
        "Spinal Canal_dice",
    ),
    (
        3,
        "Intervertebral Disc",
        "Intervertebral Disc_dice",
    ),
]:

    prefix = (
        class_name
        .lower()
        .replace(" ", "_")
    )

    dice_all = merged_df[
        metric_prefix
    ].astype(
        float
    )

    present_mask = merged_df[
        f"{prefix}_present"
    ].astype(
        bool
    )

    present_dice = (
        dice_all[
            present_mask
        ]
    )

    absent_dice = (
        dice_all[
            ~present_mask
        ]
    )

    print(
        f"\n{class_name}"
    )

    print(
        f"  All cases: "
        f"{len(dice_all)}"
    )

    print(
        f"  Ground-truth present: "
        f"{len(present_dice)}"
    )

    print(
        f"  Ground-truth absent: "
        f"{len(absent_dice)}"
    )

    if len(present_dice) > 0:

        print(
            f"  Dice when present - Mean: "
            f"{present_dice.mean():.6f}"
        )

        print(
            f"  Dice when present - Median: "
            f"{present_dice.median():.6f}"
        )

        print(
            f"  Dice when present - Std: "
            f"{present_dice.std():.6f}"
        )

        print(
            f"  Dice when present - Min: "
            f"{present_dice.min():.6f}"
        )

        print(
            f"  Dice when present - Max: "
            f"{present_dice.max():.6f}"
        )

    else:

        print(
            "  No ground-truth-present "
            "cases available."
        )

    if len(absent_dice) > 0:

        print(
            f"  Dice on absent-target cases - Mean: "
            f"{absent_dice.mean():.6f}"
        )

    else:

        print(
            "  No absent-target cases."
        )

    corrected_records.append(
        {
            "class_id": class_id,
            "class": class_name,
            "all_cases":
                len(dice_all),
            "ground_truth_present_cases":
                len(present_dice),
            "ground_truth_absent_cases":
                len(absent_dice),
            "original_mean_dice":
                float(
                    dice_all.mean()
                ),
            "present_only_mean_dice":
                float(
                    present_dice.mean()
                )
                if len(
                    present_dice
                ) > 0
                else None,
            "present_only_median_dice":
                float(
                    present_dice.median()
                )
                if len(
                    present_dice
                ) > 0
                else None,
            "present_only_std_dice":
                float(
                    present_dice.std()
                )
                if len(
                    present_dice
                ) > 1
                else 0.0,
            "present_only_min_dice":
                float(
                    present_dice.min()
                )
                if len(
                    present_dice
                ) > 0
                else None,
            "present_only_max_dice":
                float(
                    present_dice.max()
                )
                if len(
                    present_dice
                ) > 0
                else None,
        }
    )


corrected_dice_df = pd.DataFrame(
    corrected_records
)


# ============================================================
# SPINAL CANAL DEEP AUDIT
# ============================================================

print("\n" + "=" * 78)
print("SPINAL CANAL VALID-CASE AUDIT")
print("=" * 78)

spinal_present = merged_df[
    "spinal_canal_present"
].astype(bool)

spinal_present_df = merged_df[
    spinal_present
].copy()

spinal_absent_df = merged_df[
    ~spinal_present
].copy()

spinal_dice_present = (
    spinal_present_df[
        "Spinal Canal_dice"
    ].astype(float)
)

spinal_dice_absent = (
    spinal_absent_df[
        "Spinal Canal_dice"
    ].astype(float)
)

print(
    f"Total test cases: "
    f"{len(merged_df)}"
)

print(
    f"Spinal Canal present: "
    f"{len(spinal_present_df)}"
)

print(
    f"Spinal Canal absent: "
    f"{len(spinal_absent_df)}"
)

if len(
    spinal_dice_present
) > 0:

    print(
        f"\nSpinal Canal Dice "
        f"when ground truth is present:"
    )

    print(
        f"Mean   : "
        f"{spinal_dice_present.mean():.6f}"
    )

    print(
        f"Median : "
        f"{spinal_dice_present.median():.6f}"
    )

    print(
        f"Std    : "
        f"{spinal_dice_present.std():.6f}"
    )

    print(
        f"Minimum: "
        f"{spinal_dice_present.min():.6f}"
    )

    print(
        f"Maximum: "
        f"{spinal_dice_present.max():.6f}"
    )


# ============================================================
# TRUE SPINAL CANAL FAILURE CASES
# ============================================================

print("\n" + "=" * 78)
print("TRUE SPINAL CANAL FAILURE CASES")
print("=" * 78)

# Ensure official Part 11 metric columns are numeric
for col in [
    "Spinal Canal_dice",
    "Spinal Canal_iou",
    "Spinal Canal_precision",
    "Spinal Canal_recall",
]:
    merged_df[col] = pd.to_numeric(
        merged_df[col],
        errors="coerce"
    )

spinal_present_df = merged_df[
    merged_df["spinal_canal_present"].astype(bool)
].copy()

true_spinal_failures = (
    spinal_present_df[
        spinal_present_df["Spinal Canal_dice"] < 0.70
    ]
    .sort_values(
        "Spinal Canal_dice"
    )
)

print(
    f"Ground-truth-present cases "
    f"with Dice < 0.70: "
    f"{len(true_spinal_failures)}"
)

if len(
    true_spinal_failures
) > 0:

    print(
        true_spinal_failures[
            [
                "file",
                "Spinal Canal_dice",
                "Spinal Canal_iou",
                "Spinal Canal_precision",
                "Spinal Canal_recall",
                "spinal_canal_voxels",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# PRESENT-ONLY THRESHOLD DISTRIBUTION
# ============================================================

print("\n" + "=" * 78)
print("SPINAL CANAL PRESENT-ONLY DICE DISTRIBUTION")
print("=" * 78)

thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

threshold_records = []

for threshold in thresholds:

    count = int(
        (
            spinal_dice_present
            >= threshold
        ).sum()
    )

    total = len(
        spinal_dice_present
    )

    percentage = (
        count
        / total
        * 100.0
    ) if total > 0 else 0.0

    threshold_records.append(
        {
            "threshold":
                threshold,
            "cases_at_or_above":
                count,
            "total_present_cases":
                total,
            "percentage":
                percentage,
        }
    )

    print(
        f"Dice >= {threshold:.2f}: "
        f"{count} / {total} "
        f"({percentage:.2f}%)"
    )

threshold_df = pd.DataFrame(
    threshold_records
)


# ============================================================
# VOXEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 78)
print("CLASS VOXEL DISTRIBUTION")
print("=" * 78)

for class_id in (
    1,
    2,
    3,
):

    class_name = CLASS_NAMES[
        class_id
    ]

    prefix = (
        class_name
        .lower()
        .replace(" ", "_")
    )

    present_rows = audit_df[
        audit_df[
            f"{prefix}_present"
        ]
    ]

    if len(present_rows) == 0:

        continue

    print(
        f"\n{class_name}"
    )

    print(
        f"Minimum: "
        f"{present_rows[f'{prefix}_voxels'].min():,}"
    )

    print(
        f"25th percentile: "
        f"{present_rows[f'{prefix}_voxels'].quantile(0.25):,.0f}"
    )

    print(
        f"Median: "
        f"{present_rows[f'{prefix}_voxels'].median():,.0f}"
    )

    print(
        f"Mean: "
        f"{present_rows[f'{prefix}_voxels'].mean():,.0f}"
    )

    print(
        f"75th percentile: "
        f"{present_rows[f'{prefix}_voxels'].quantile(0.75):,.0f}"
    )

    print(
        f"Maximum: "
        f"{present_rows[f'{prefix}_voxels'].max():,}"
    )


# ============================================================
# SAVE TABLES
# ============================================================

print("\n" + "=" * 78)
print("SAVING ANALYSIS TABLES")
print("=" * 78)

audit_csv = (
    OUTPUT_DIR
    / "ground_truth_class_presence.csv"
)

class_summary_csv = (
    OUTPUT_DIR
    / "class_presence_summary.csv"
)

corrected_dice_csv = (
    OUTPUT_DIR
    / "class_dice_present_only_summary.csv"
)

threshold_csv = (
    OUTPUT_DIR
    / "spinal_canal_present_only_thresholds.csv"
)

spinal_failures_csv = (
    OUTPUT_DIR
    / "true_spinal_canal_failure_cases.csv"
)

audit_df.to_csv(
    audit_csv,
    index=False,
)

class_summary_df.to_csv(
    class_summary_csv,
    index=False,
)

corrected_dice_df.to_csv(
    corrected_dice_csv,
    index=False,
)

threshold_df.to_csv(
    threshold_csv,
    index=False,
)

true_spinal_failures.to_csv(
    spinal_failures_csv,
    index=False,
)

print(
    f"Ground-truth audit:\n{audit_csv}"
)

print(
    f"Class summary:\n{class_summary_csv}"
)

print(
    f"Present-only Dice summary:\n{corrected_dice_csv}"
)

print(
    f"Threshold summary:\n{threshold_csv}"
)

print(
    f"True spinal failures:\n{spinal_failures_csv}"
)


# ============================================================
# CHART 1 - CLASS PRESENCE
# ============================================================

print("\n" + "=" * 78)
print("CREATING CLASS PRESENCE CHART")
print("=" * 78)

plt.figure(
    figsize=(9, 6)
)

classes = (
    class_summary_df[
        "class"
    ]
    .tolist()
)

present_values = (
    class_summary_df[
        "present_cases"
    ]
    .tolist()
)

absent_values = (
    class_summary_df[
        "absent_cases"
    ]
    .tolist()
)

x = np.arange(
    len(classes)
)

width = 0.35

plt.bar(
    x - width / 2,
    present_values,
    width,
    label="Present",
)

plt.bar(
    x + width / 2,
    absent_values,
    width,
    label="Absent",
)

plt.xticks(
    x,
    classes,
    rotation=15,
)

plt.ylabel(
    "Number of Test Cases"
)

plt.title(
    "Ground-Truth Anatomical Class Presence"
)

plt.legend()

plt.tight_layout()

presence_chart = (
    OUTPUT_DIR
    / "class_presence_distribution.png"
)

plt.savefig(
    presence_chart,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {presence_chart}"
)


# ============================================================
# CHART 2 - PRESENT-ONLY SPINAL CANAL DICE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

if len(
    spinal_dice_present
) > 0:

    plt.hist(
        spinal_dice_present,
        bins=10,
    )

plt.xlabel(
    "Spinal Canal Dice"
)

plt.ylabel(
    "Number of Cases"
)

plt.title(
    "Spinal Canal Dice - "
    "Ground-Truth-Present Cases Only"
)

plt.tight_layout()

spinal_dice_chart = (
    OUTPUT_DIR
    / "spinal_canal_present_only_dice_distribution.png"
)

plt.savefig(
    spinal_dice_chart,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {spinal_dice_chart}"
)


# ============================================================
# CHART 3 - CLASS DICE COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 6)
)

original_values = (
    corrected_dice_df[
        "original_mean_dice"
    ]
    .tolist()
)

present_only_values = (
    corrected_dice_df[
        "present_only_mean_dice"
    ]
    .fillna(0)
    .tolist()
)

x = np.arange(
    len(classes)
)

plt.bar(
    x - width / 2,
    original_values,
    width,
    label="Original Mean Dice",
)

plt.bar(
    x + width / 2,
    present_only_values,
    width,
    label="Present-Only Mean Dice",
)

plt.xticks(
    x,
    classes,
    rotation=15,
)

plt.ylim(
    0,
    1.05,
)

plt.ylabel(
    "Dice"
)

plt.title(
    "Original vs Ground-Truth-Present Dice"
)

plt.legend()

plt.tight_layout()

dice_comparison_chart = (
    OUTPUT_DIR
    / "original_vs_present_only_dice.png"
)

plt.savefig(
    dice_comparison_chart,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {dice_comparison_chart}"
)


# ============================================================
# CHART 4 - SPINAL CANAL VOXEL COUNTS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

spinal_present_rows = audit_df[
    audit_df[
        "spinal_canal_present"
    ]
]

if len(
    spinal_present_rows
) > 0:

    plt.hist(
        spinal_present_rows[
            "spinal_canal_voxels"
        ],
        bins=15,
    )

plt.xlabel(
    "Spinal Canal Ground-Truth Voxels"
)

plt.ylabel(
    "Number of Cases"
)

plt.title(
    "Spinal Canal Ground-Truth Size "
    "Distribution"
)

plt.tight_layout()

voxel_chart = (
    OUTPUT_DIR
    / "spinal_canal_ground_truth_voxel_distribution.png"
)

plt.savefig(
    voxel_chart,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {voxel_chart}"
)


# ============================================================
# JSON SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("CREATING JSON SUMMARY")
print("=" * 78)

json_summary = {
    "phase": "Phase 3 - Part 16",
    "test_cases": int(
        len(audit_df)
    ),

    "classes": {},

    "spinal_canal": {
        "present_cases": int(
            len(spinal_present_df)
        ),
        "absent_cases": int(
            len(spinal_absent_df)
        ),
        "original_mean_dice":
            float(
                merged_df[
                    "Spinal Canal_dice"
                ].mean()
            ),
        "present_only_mean_dice":
            float(
                spinal_dice_present.mean()
            )
            if len(
                spinal_dice_present
            ) > 0
            else None,
        "present_only_median_dice":
            float(
                spinal_dice_present.median()
            )
            if len(
                spinal_dice_present
            ) > 0
            else None,
        "present_only_std_dice":
            float(
                spinal_dice_present.std()
            )
            if len(
                spinal_dice_present
            ) > 1
            else 0.0,
        "present_only_min_dice":
            float(
                spinal_dice_present.min()
            )
            if len(
                spinal_dice_present
            ) > 0
            else None,
        "present_only_max_dice":
            float(
                spinal_dice_present.max()
            )
            if len(
                spinal_dice_present
            ) > 0
            else None,
        "present_only_cases_below_0_70":
            int(
                (
                    spinal_dice_present
                    < 0.70
                ).sum()
            ),
    },

    "files": {
        "audit":
            str(audit_csv),
        "class_summary":
            str(class_summary_csv),
        "corrected_dice":
            str(corrected_dice_csv),
        "thresholds":
            str(threshold_csv),
        "true_spinal_failures":
            str(spinal_failures_csv),
        "presence_chart":
            str(presence_chart),
        "spinal_dice_chart":
            str(spinal_dice_chart),
        "dice_comparison_chart":
            str(dice_comparison_chart),
        "voxel_chart":
            str(voxel_chart),
    },
}

for row in (
    class_summary_df.to_dict(
        orient="records"
    )
):

    class_name = row[
        "class"
    ]

    json_summary[
        "classes"
    ][class_name] = {
        "present_cases":
            int(
                row[
                    "present_cases"
                ]
            ),
        "absent_cases":
            int(
                row[
                    "absent_cases"
                ]
            ),
        "presence_percentage":
            float(
                row[
                    "presence_percentage"
                ]
            ),
        "mean_voxels_present":
            float(
                row[
                    "mean_voxels_present_cases"
                ]
            ),
        "median_voxels_present":
            float(
                row[
                    "median_voxels_present_cases"
                ]
            ),
    }


json_path = (
    OUTPUT_DIR
    / "class_presence_audit_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_summary,
        f,
        indent=4,
    )

print(
    f"Saved: {json_path}"
)


# ============================================================
# FINAL REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "phase3_part16_class_presence_audit_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "PHASE 3 - PART 16\n"
    )

    f.write(
        "GROUND-TRUTH CLASS PRESENCE & ANNOTATION AUDIT\n"
    )

    f.write(
        "=" * 72
        + "\n\n"
    )

    f.write(
        f"Test cases: "
        f"{len(audit_df)}\n\n"
    )

    f.write(
        "CLASS PRESENCE SUMMARY\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    for _, row in (
        class_summary_df.iterrows()
    ):

        f.write(
            f"\n{row['class']}\n"
        )

        f.write(
            f"  Present cases: "
            f"{row['present_cases']}\n"
        )

        f.write(
            f"  Absent cases: "
            f"{row['absent_cases']}\n"
        )

        f.write(
            f"  Presence: "
            f"{row['presence_percentage']:.2f}%\n"
        )

        f.write(
            f"  Mean voxels when present: "
            f"{row['mean_voxels_present_cases']:.2f}\n"
        )

        f.write(
            f"  Median voxels when present: "
            f"{row['median_voxels_present_cases']:.2f}\n"
        )

    f.write(
        "\n\nSPINAL CANAL AUDIT\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        f"Total cases: "
        f"{len(merged_df)}\n"
    )

    f.write(
        f"Ground-truth-present cases: "
        f"{len(spinal_present_df)}\n"
    )

    f.write(
        f"Ground-truth-absent cases: "
        f"{len(spinal_absent_df)}\n"
    )

    f.write(
        f"Original mean Dice: "
        f"{merged_df['Spinal Canal_dice'].mean():.6f}\n"
    )

    if len(
        spinal_dice_present
    ) > 0:

        f.write(
            f"Present-only mean Dice: "
            f"{spinal_dice_present.mean():.6f}\n"
        )

        f.write(
            f"Present-only median Dice: "
            f"{spinal_dice_present.median():.6f}\n"
        )

        f.write(
            f"Present-only standard deviation: "
            f"{spinal_dice_present.std():.6f}\n"
        )

        f.write(
            f"Present-only minimum Dice: "
            f"{spinal_dice_present.min():.6f}\n"
        )

        f.write(
            f"Present-only maximum Dice: "
            f"{spinal_dice_present.max():.6f}\n"
        )

    f.write(
        "\n\nTRUE SPINAL CANAL FAILURE CASES\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        "A true failure case is defined here as a case "
        "where the ground-truth Spinal Canal class is "
        "present and the Dice is below 0.70.\n\n"
    )

    if len(
        true_spinal_failures
    ) == 0:

        f.write(
            "No ground-truth-present Spinal Canal "
            "cases had Dice below 0.70.\n"
        )

    else:

        for _, row in (
            true_spinal_failures.iterrows()
        ):

            f.write(
                f"{row['file']}: "
                f"Dice={row['Spinal Canal_dice']:.6f}, "
                f"IoU={row['Spinal Canal_iou']:.6f}, "
                f"Precision={row['Spinal Canal_precision']:.6f}, "
                f"Recall={row['Spinal Canal_recall']:.6f}\n"
            )

    f.write(
        "\n\nINTERPRETATION NOTE\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        "Dice values for cases where the target class is "
        "absent should be interpreted separately from "
        "cases where the anatomical class is actually "
        "present in the ground-truth annotation.\n"
    )

    f.write(
        "The present-only analysis provides a clearer "
        "measure of segmentation quality for that "
        "anatomical structure.\n"
    )


# ============================================================
# COMPLETION
# ============================================================

elapsed = (
    time.time()
    - start_time
)

print("\n" + "=" * 78)
print("PART 16 COMPLETE")
print("=" * 78)

print(
    f"Test cases analyzed: "
    f"{len(audit_df)}"
)

print(
    f"Spinal Canal present: "
    f"{len(spinal_present_df)}"
)

print(
    f"Spinal Canal absent: "
    f"{len(spinal_absent_df)}"
)

if len(
    spinal_dice_present
) > 0:

    print(
        f"Spinal Canal present-only "
        f"Mean Dice: "
        f"{spinal_dice_present.mean():.6f}"
    )

    print(
        f"Spinal Canal present-only "
        f"Median Dice: "
        f"{spinal_dice_present.median():.6f}"
    )

print(
    f"Execution time: "
    f"{elapsed / 60:.2f} minutes"
)

print("\n" + "=" * 78)
print("OUTPUT FILES")
print("=" * 78)

print(
    f"Ground-truth audit:\n"
    f"{audit_csv}"
)

print(
    f"Class summary:\n"
    f"{class_summary_csv}"
)

print(
    f"Present-only Dice summary:\n"
    f"{corrected_dice_csv}"
)

print(
    f"Spinal Canal thresholds:\n"
    f"{threshold_csv}"
)

print(
    f"True spinal failures:\n"
    f"{spinal_failures_csv}"
)

print(
    f"JSON summary:\n"
    f"{json_path}"
)

print(
    f"Report:\n"
    f"{report_path}"
)

print("\n" + "=" * 78)
print("PHASE 3 - PART 16 COMPLETE")
print("=" * 78)