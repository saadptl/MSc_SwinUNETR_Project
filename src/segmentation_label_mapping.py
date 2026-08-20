from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# PHASE 3 - PART 3
# ANATOMICAL LABEL MAPPING & DATASET STATISTICS
# ============================================================

print("=" * 75)
print("PHASE 3 - PART 3")
print("SPIDER ANATOMICAL LABEL MAPPING & DATASET STATISTICS")
print("=" * 75)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPIDER_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider"
)

MASKS_DIR = (
    SPIDER_DIR
    / "masks"
)

IMAGES_DIR = (
    SPIDER_DIR
    / "images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
)

REPORT_DIR = (
    OUTPUT_DIR
    / "statistics"
)

PROCESSED_MASK_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "masks_4class"
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LABEL DEFINITIONS
# ============================================================

ORIGINAL_LABELS = {
    0: "Background",

    1: "Vertebrae",
    2: "Vertebrae",
    3: "Vertebrae",
    4: "Vertebrae",
    5: "Vertebrae",
    6: "Vertebrae",
    7: "Vertebrae",
    8: "Vertebrae",
    9: "Vertebrae",

    100: "Spinal Canal",

    201: "Intervertebral Disc",
    202: "Intervertebral Disc",
    203: "Intervertebral Disc",
    204: "Intervertebral Disc",
    205: "Intervertebral Disc",
    206: "Intervertebral Disc",
    207: "Intervertebral Disc",
    208: "Intervertebral Disc",
    209: "Intervertebral Disc",
}


TRAINING_LABEL_MAP = {
    # Background
    0: 0,

    # Vertebrae
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,

    # Spinal canal
    100: 2,

    # Intervertebral discs
    201: 3,
    202: 3,
    203: 3,
    204: 3,
    205: 3,
    206: 3,
    207: 3,
    208: 3,
    209: 3,
}


TRAINING_CLASS_NAMES = {
    0: "Background",
    1: "Vertebrae",
    2: "Spinal Canal",
    3: "Intervertebral Disc",
}


EXPECTED_LABELS = set(
    ORIGINAL_LABELS.keys()
)


# ============================================================
# HEADER
# ============================================================

print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nSPIDER DIRECTORY")
print(SPIDER_DIR)

print("\nMASK DIRECTORY")
print(MASKS_DIR)

print("\nPROCESSED MASK DIRECTORY")
print(PROCESSED_MASK_DIR)


# ============================================================
# FIND MASKS
# ============================================================

mask_files = sorted(
    MASKS_DIR.rglob("*.mha")
)

image_files = sorted(
    IMAGES_DIR.rglob("*.mha")
)


print("\nDATASET FILE COUNTS")

print(
    "MRI volumes :",
    len(image_files)
)

print(
    "Mask volumes:",
    len(mask_files)
)


if not mask_files:

    raise RuntimeError(
        "No SPIDER mask files found."
    )


# ============================================================
# VERIFY MRI / MASK PAIRS
# ============================================================

image_names = {
    p.name
    for p in image_files
}

mask_names = {
    p.name
    for p in mask_files
}

common_names = sorted(
    image_names
    &
    mask_names
)


print("\nMRI / MASK PAIR CHECK")

print(
    "Matching pairs:",
    len(common_names)
)

print(
    "MRI without mask:",
    len(
        image_names
        -
        mask_names
    )
)

print(
    "Mask without MRI:",
    len(
        mask_names
        -
        image_names
    )
)


# ============================================================
# DATA COLLECTION
# ============================================================

case_statistics = []

label_statistics = []

training_statistics = []

unknown_labels = Counter()

global_original_counts = Counter()

global_training_counts = Counter()


# ============================================================
# PROCESS EVERY MASK
# ============================================================

print("\n" + "=" * 75)
print("SCANNING ALL SPIDER MASKS")
print("=" * 75)


for index, mask_path in enumerate(
    mask_files,
    start=1
):

    # --------------------------------------------------------
    # Read mask
    # --------------------------------------------------------

    mask_image = sitk.ReadImage(
        str(mask_path)
    )

    mask_array = sitk.GetArrayFromImage(
        mask_image
    )

    mask_array = mask_array.astype(
        np.int16,
        copy=False
    )

    # --------------------------------------------------------
    # Unique labels
    # --------------------------------------------------------

    unique_labels, counts = np.unique(
        mask_array,
        return_counts=True
    )

    case_label_counts = {
        int(label): int(count)
        for label, count in zip(
            unique_labels,
            counts
        )
    }

    # --------------------------------------------------------
    # Detect unknown labels
    # --------------------------------------------------------

    current_labels = set(
        case_label_counts.keys()
    )

    unexpected = (
        current_labels
        -
        EXPECTED_LABELS
    )

    if unexpected:

        for label in unexpected:

            unknown_labels[
                int(label)
            ] += 1

    # --------------------------------------------------------
    # Update global counts
    # --------------------------------------------------------

    for label, count in case_label_counts.items():

        global_original_counts[
            label
        ] += count

    # --------------------------------------------------------
    # Create 4-class mapping
    # --------------------------------------------------------

    training_mask = np.zeros(
        mask_array.shape,
        dtype=np.uint8
    )

    for original_label, training_label in (
        TRAINING_LABEL_MAP.items()
    ):

        training_mask[
            mask_array
            ==
            original_label
        ] = training_label

    # --------------------------------------------------------
    # Training counts
    # --------------------------------------------------------

    training_labels, training_counts = np.unique(
        training_mask,
        return_counts=True
    )

    case_training_counts = {
        int(label): int(count)
        for label, count in zip(
            training_labels,
            training_counts
        )
    }

    for label, count in case_training_counts.items():

        global_training_counts[
            label
        ] += count

    # --------------------------------------------------------
    # Case-level structure availability
    # --------------------------------------------------------

    has_vertebrae = any(
        case_label_counts.get(
            label,
            0
        ) > 0
        for label in range(1, 9)
    )

    has_canal = (
        case_label_counts.get(
            100,
            0
        ) > 0
    )

    has_discs = any(
        case_label_counts.get(
            label,
            0
        ) > 0
        for label in range(201, 209)
    )

    # --------------------------------------------------------
    # Volume information
    # --------------------------------------------------------

    total_voxels = int(
        mask_array.size
    )

    foreground_voxels = int(
        np.count_nonzero(
            mask_array
        )
    )

    foreground_percentage = (
        foreground_voxels
        /
        total_voxels
        *
        100
    )

    # --------------------------------------------------------
    # T1 / T2
    # --------------------------------------------------------

    filename = mask_path.name

    if "_t1" in filename.lower():

        sequence = "T1"

    elif "_t2" in filename.lower():

        sequence = "T2"

    else:

        sequence = "Unknown"

    # --------------------------------------------------------
    # Save case statistics
    # --------------------------------------------------------

    case_statistics.append(
        {
            "file": filename,
            "sequence": sequence,
            "shape": str(
                mask_array.shape
            ),
            "total_voxels":
                total_voxels,
            "foreground_voxels":
                foreground_voxels,
            "foreground_percentage":
                foreground_percentage,
            "vertebrae_present":
                has_vertebrae,
            "spinal_canal_present":
                has_canal,
            "discs_present":
                has_discs,
            "unique_labels":
                str(
                    sorted(
                        current_labels
                    )
                ),
        }
    )

    # --------------------------------------------------------
    # Original label statistics
    # --------------------------------------------------------

    for label in EXPECTED_LABELS:

        count = case_label_counts.get(
            label,
            0
        )

        label_statistics.append(
            {
                "file": filename,
                "sequence": sequence,
                "original_label": label,
                "structure":
                    ORIGINAL_LABELS[
                        label
                    ],
                "voxel_count":
                    count,
                "present":
                    count > 0,
            }
        )

    # --------------------------------------------------------
    # Training label statistics
    # --------------------------------------------------------

    for training_label, class_name in (
        TRAINING_CLASS_NAMES.items()
    ):

        count = case_training_counts.get(
            training_label,
            0
        )

        training_statistics.append(
            {
                "file": filename,
                "sequence": sequence,
                "training_label":
                    training_label,
                "class_name":
                    class_name,
                "voxel_count":
                    count,
                "present":
                    count > 0,
            }
        )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index == 1
        or index % 50 == 0
        or index == len(mask_files)
    ):

        print(
            f"Processed "
            f"{index:3d} / "
            f"{len(mask_files)}"
        )


# ============================================================
# DATAFRAMES
# ============================================================

case_df = pd.DataFrame(
    case_statistics
)

label_df = pd.DataFrame(
    label_statistics
)

training_df = pd.DataFrame(
    training_statistics
)


# ============================================================
# ORIGINAL LABEL SUMMARY
# ============================================================

original_summary_rows = []


for label in sorted(
    EXPECTED_LABELS
):

    total_voxels = (
        global_original_counts[
            label
        ]
    )

    cases_present = int(
        (
            label_df[
                label_df[
                    "original_label"
                ]
                ==
                label
            ][
                "present"
            ]
        ).sum()
    )

    original_summary_rows.append(
        {
            "original_label":
                label,

            "structure":
                ORIGINAL_LABELS[
                    label
                ],

            "total_voxels":
                total_voxels,

            "cases_present":
                cases_present,

            "cases_total":
                len(mask_files),

            "case_presence_percentage":
                (
                    cases_present
                    /
                    len(mask_files)
                    *
                    100
                ),
        }
    )


original_summary_df = pd.DataFrame(
    original_summary_rows
)


# ============================================================
# TRAINING LABEL SUMMARY
# ============================================================

training_summary_rows = []


for label, class_name in (
    TRAINING_CLASS_NAMES.items()
):

    total_voxels = (
        global_training_counts[
            label
        ]
    )

    cases_present = int(
        (
            training_df[
                training_df[
                    "training_label"
                ]
                ==
                label
            ][
                "present"
            ]
        ).sum()
    )

    training_summary_rows.append(
        {
            "training_label":
                label,

            "class_name":
                class_name,

            "total_voxels":
                total_voxels,

            "cases_present":
                cases_present,

            "cases_total":
                len(mask_files),

            "case_presence_percentage":
                (
                    cases_present
                    /
                    len(mask_files)
                    *
                    100
                ),
        }
    )


training_summary_df = pd.DataFrame(
    training_summary_rows
)


# ============================================================
# LABEL VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("LABEL VALIDATION")
print("=" * 75)


if unknown_labels:

    print(
        "WARNING: Unexpected labels found:"
    )

    for label, count in (
        unknown_labels.items()
    ):

        print(
            f"  Label {label}: "
            f"{count} cases"
        )

else:

    print(
        "✓ No unexpected labels found."
    )


# ============================================================
# GLOBAL ORIGINAL LABEL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("ORIGINAL SPIDER LABEL SUMMARY")
print("=" * 75)

print(
    original_summary_df.to_string(
        index=False
    )
)


# ============================================================
# 4-CLASS TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("4-CLASS TRAINING LABEL SUMMARY")
print("=" * 75)

print(
    training_summary_df.to_string(
        index=False
    )
)


# ============================================================
# SEQUENCE SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("MRI SEQUENCE SUMMARY")
print("=" * 75)

sequence_summary = (
    case_df[
        "sequence"
    ]
    .value_counts()
)

print(
    sequence_summary.to_string()
)


# ============================================================
# STRUCTURE AVAILABILITY
# ============================================================

print("\n" + "=" * 75)
print("STRUCTURE AVAILABILITY")
print("=" * 75)

structure_availability = pd.DataFrame(
    {
        "Structure": [
            "Vertebrae",
            "Spinal Canal",
            "Intervertebral Discs",
        ],

        "Cases Present": [
            case_df[
                "vertebrae_present"
            ].sum(),

            case_df[
                "spinal_canal_present"
            ].sum(),

            case_df[
                "discs_present"
            ].sum(),
        ],
    }
)


structure_availability[
    "Percentage"
] = (
    structure_availability[
        "Cases Present"
    ]
    /
    len(case_df)
    *
    100
)


print(
    structure_availability.to_string(
        index=False
    )
)


# ============================================================
# SAVE ORIGINAL LABEL SUMMARY
# ============================================================

original_summary_path = (
    REPORT_DIR
    /
    "original_label_summary.csv"
)

original_summary_df.to_csv(
    original_summary_path,
    index=False
)


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

training_summary_path = (
    REPORT_DIR
    /
    "training_4class_summary.csv"
)

training_summary_df.to_csv(
    training_summary_path,
    index=False
)


# ============================================================
# SAVE CASE STATISTICS
# ============================================================

case_statistics_path = (
    REPORT_DIR
    /
    "case_statistics.csv"
)

case_df.to_csv(
    case_statistics_path,
    index=False
)


# ============================================================
# SAVE ORIGINAL LABEL CASE STATISTICS
# ============================================================

label_statistics_path = (
    REPORT_DIR
    /
    "original_label_case_statistics.csv"
)

label_df.to_csv(
    label_statistics_path,
    index=False
)


# ============================================================
# SAVE TRAINING LABEL CASE STATISTICS
# ============================================================

training_statistics_path = (
    REPORT_DIR
    /
    "training_label_case_statistics.csv"
)

training_df.to_csv(
    training_statistics_path,
    index=False
)


# ============================================================
# SAVE STRUCTURE AVAILABILITY
# ============================================================

structure_path = (
    REPORT_DIR
    /
    "structure_availability.csv"
)

structure_availability.to_csv(
    structure_path,
    index=False
)


# ============================================================
# SAVE 4-CLASS MAPPING TABLE
# ============================================================

mapping_rows = []


for original_label in sorted(
    TRAINING_LABEL_MAP.keys()
):

    training_label = (
        TRAINING_LABEL_MAP[
            original_label
        ]
    )

    mapping_rows.append(
        {
            "original_label":
                original_label,

            "training_label":
                training_label,

            "structure":
                ORIGINAL_LABELS[
                    original_label
                ],

            "training_class":
                TRAINING_CLASS_NAMES[
                    training_label
                ],
        }
    )


mapping_df = pd.DataFrame(
    mapping_rows
)


mapping_path = (
    REPORT_DIR
    /
    "label_mapping.csv"
)

mapping_df.to_csv(
    mapping_path,
    index=False
)


# ============================================================
# CREATE 4-CLASS MASKS
# ============================================================

print("\n" + "=" * 75)
print("CREATING 4-CLASS MASKS")
print("=" * 75)


processed_count = 0


for mask_path in mask_files:

    mask_image = sitk.ReadImage(
        str(mask_path)
    )

    mask_array = sitk.GetArrayFromImage(
        mask_image
    )

    mask_array = mask_array.astype(
        np.int16,
        copy=False
    )

    training_mask = np.zeros(
        mask_array.shape,
        dtype=np.uint8
    )

    for (
        original_label,
        training_label
    ) in TRAINING_LABEL_MAP.items():

        training_mask[
            mask_array
            ==
            original_label
        ] = training_label

    training_image = (
        sitk.GetImageFromArray(
            training_mask
        )
    )

    training_image.CopyInformation(
        mask_image
    )

    output_path = (
        PROCESSED_MASK_DIR
        /
        mask_path.name
    )

    sitk.WriteImage(
        training_image,
        str(output_path),
        useCompression=True
    )

    processed_count += 1

    if (
        processed_count == 1
        or processed_count % 50 == 0
        or processed_count == len(mask_files)
    ):

        print(
            f"Saved "
            f"{processed_count:3d} / "
            f"{len(mask_files)}"
        )


# ============================================================
# VERIFY PROCESSED MASKS
# ============================================================

processed_masks = sorted(
    PROCESSED_MASK_DIR.glob(
        "*.mha"
    )
)


print("\nPROCESSED MASK COUNT")

print(
    "Original masks :",
    len(mask_files)
)

print(
    "4-class masks  :",
    len(processed_masks)
)


# ============================================================
# VISUALIZE MAPPING FOR FIRST CASE
# ============================================================

example_name = common_names[0]

# Rebuild filename → path mapping
mask_map = {
    path.name: path
    for path in mask_files
}

example_original = (
    mask_map[example_name]
)

example_processed = (
    PROCESSED_MASK_DIR
    / example_name
)


original_image = sitk.ReadImage(
    str(example_original)
)

original_array = sitk.GetArrayFromImage(
    original_image
)


processed_image = sitk.ReadImage(
    str(example_processed)
)

processed_array = sitk.GetArrayFromImage(
    processed_image
)


# ============================================================
# FIND REPRESENTATIVE SLICE
# ============================================================

slice_counts = np.count_nonzero(
    original_array,
    axis=(1, 2)
)

valid_slices = np.where(
    slice_counts > 0
)[0]


if len(valid_slices) > 0:

    visual_slice = int(
        valid_slices[
            len(valid_slices) // 2
        ]
    )

else:

    visual_slice = (
        original_array.shape[0] // 2
    )


original_slice = (
    original_array[
        visual_slice
    ]
)

processed_slice = (
    processed_array[
        visual_slice
    ]
)


# ============================================================
# CREATE LABEL-MAPPING VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 6)
)


axes[0].imshow(
    original_slice,
    interpolation="nearest",
    origin="lower"
)

axes[0].set_title(
    "Original SPIDER Labels",
    fontsize=15,
    fontweight="bold"
)

axes[0].axis("off")


axes[1].imshow(
    processed_slice,
    interpolation="nearest",
    origin="lower"
)

axes[1].set_title(
    "4-Class Training Labels",
    fontsize=15,
    fontweight="bold"
)

axes[1].axis("off")


fig.suptitle(
    (
        "SPIDER Label Mapping\n"
        f"{example_name} | Slice {visual_slice}"
    ),
    fontsize=17,
    fontweight="bold"
)


plt.tight_layout()


mapping_visualization_path = (
    REPORT_DIR
    / "phase3_part3_label_mapping.png"
)


plt.savefig(
    mapping_visualization_path,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# CREATE CLASS DISTRIBUTION CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


labels = (
    training_summary_df[
        "class_name"
    ]
)

values = (
    training_summary_df[
        "total_voxels"
    ]
)


ax.bar(
    labels,
    values
)


ax.set_title(
    "4-Class Segmentation Voxel Distribution",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Segmentation Class"
)

ax.set_ylabel(
    "Total Voxels"
)


ax.ticklabel_format(
    style="plain",
    axis="y"
)


plt.xticks(
    rotation=15
)

plt.tight_layout()


distribution_chart_path = (
    REPORT_DIR
    / "phase3_part3_class_distribution.png"
)


plt.savefig(
    distribution_chart_path,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print(
    "Original label summary:",
    original_summary_path
)

print(
    "Training label summary:",
    training_summary_path
)

print(
    "Case statistics:",
    case_statistics_path
)

print(
    "Structure availability:",
    structure_path
)

print(
    "Label mapping:",
    mapping_path
)

print(
    "4-class masks:",
    PROCESSED_MASK_DIR
)

print(
    "Mapping visualization:",
    mapping_visualization_path
)

print(
    "Distribution chart:",
    distribution_chart_path
)


print("\n" + "=" * 75)
print("PHASE 3 - PART 3 COMPLETE")
print("=" * 75)