from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# PHASE 3 - PART 5
# SPIDER 3D SEGMENTATION PREPROCESSING
# ============================================================

print("=" * 75)
print("PHASE 3 - PART 5")
print("SPIDER 3D SEGMENTATION PREPROCESSING")
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

MRI_DIR = (
    SPIDER_DIR
    / "images"
)

MASK_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "masks_4class"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
)


PROCESSED_MRI_DIR = (
    OUTPUT_ROOT
    / "images_preprocessed"
)

PROCESSED_MASK_DIR = (
    OUTPUT_ROOT
    / "masks_preprocessed"
)

STATISTICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "preprocessing"
)


PROCESSED_MRI_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STATISTICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PREPROCESSING CONFIGURATION
# ============================================================

# Robust intensity clipping
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0


# Foreground threshold after normalization
FOREGROUND_THRESHOLD = 0.02


# Safety limits
MAX_DIMENSION = 512


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nMRI DIRECTORY")
print(MRI_DIR)

print("\nMASK DIRECTORY")
print(MASK_DIR)

print("\nPROCESSED MRI DIRECTORY")
print(PROCESSED_MRI_DIR)

print("\nPROCESSED MASK DIRECTORY")
print(PROCESSED_MASK_DIR)

print("\nPREPROCESSING CONFIGURATION")

print(
    f"Intensity clipping: "
    f"{LOW_PERCENTILE}th - "
    f"{HIGH_PERCENTILE}th percentile"
)

print(
    "Foreground threshold:",
    FOREGROUND_THRESHOLD
)

print(
    "Maximum dimension:",
    MAX_DIMENSION
)


# ============================================================
# FIND FILES
# ============================================================

mri_files = sorted(
    MRI_DIR.rglob("*.mha")
)

mask_files = sorted(
    MASK_DIR.rglob("*.mha")
)


print("\n" + "=" * 75)
print("FILE COUNTS")
print("=" * 75)

print(
    "MRI files:",
    len(mri_files)
)

print(
    "4-class mask files:",
    len(mask_files)
)


if not mri_files:

    raise RuntimeError(
        "No MRI .mha files found."
    )


if not mask_files:

    raise RuntimeError(
        "No 4-class masks found."
    )


# ============================================================
# CREATE MASK LOOKUP
# ============================================================

mask_map = {
    path.name: path
    for path in mask_files
}


# ============================================================
# VERIFY PAIRING
# ============================================================

paired_files = [
    path
    for path in mri_files
    if path.name in mask_map
]


missing_masks = [
    path.name
    for path in mri_files
    if path.name not in mask_map
]


print("\nMRI / MASK PAIRING")

print(
    "Matching pairs:",
    len(paired_files)
)

print(
    "MRI without mask:",
    len(missing_masks)
)


if missing_masks:

    print("\nFirst missing masks:")

    for name in missing_masks[:10]:

        print(
            " ",
            name
        )


if not paired_files:

    raise RuntimeError(
        "No MRI/mask pairs found."
    )


# ============================================================
# HELPER: ORIENTATION
# ============================================================

def standardize_orientation(
    image
):
    """
    Convert image to a consistent
    SimpleITK orientation.

    DICOM/MHA orientation differences can
    otherwise introduce inconsistent spatial
    arrangements between cases.
    """

    return sitk.DICOMOrient(
        image,
        "LPS"
    )


# ============================================================
# HELPER: NORMALIZE MRI
# ============================================================

def normalize_mri(
    array
):
    """
    Robust MRI intensity normalization.

    Steps:
        1. Replace non-finite values.
        2. Compute percentile limits.
        3. Clip intensities.
        4. Min-max normalize to [0, 1].
    """

    array = np.asarray(
        array,
        dtype=np.float32
    )

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Use non-zero voxels when possible
    # --------------------------------------------------------

    nonzero = array[
        np.abs(array) > 1e-8
    ]

    if nonzero.size < 100:

        nonzero = array.reshape(
            -1
        )

    low = np.percentile(
        nonzero,
        LOW_PERCENTILE
    )

    high = np.percentile(
        nonzero,
        HIGH_PERCENTILE
    )

    if high <= low:

        low = float(
            np.min(nonzero)
        )

        high = float(
            np.max(nonzero)
        )

    if high <= low:

        return np.zeros_like(
            array,
            dtype=np.float32
        ), low, high

    array = np.clip(
        array,
        low,
        high
    )

    array = (
        array - low
    ) / (
        high - low
    )

    array = np.clip(
        array,
        0.0,
        1.0
    )

    return (
        array.astype(
            np.float32
        ),
        float(low),
        float(high)
    )


# ============================================================
# HELPER: FOREGROUND BOUNDING BOX
# ============================================================

def get_foreground_bbox(
    array,
    threshold=FOREGROUND_THRESHOLD
):
    """
    Find the bounding box of non-background MRI.

    Returns:
        (z_min, z_max,
         y_min, y_max,
         x_min, x_max)
    """

    foreground = (
        array > threshold
    )

    coordinates = np.where(
        foreground
    )

    if len(coordinates[0]) == 0:

        return (
            0,
            array.shape[0],
            0,
            array.shape[1],
            0,
            array.shape[2],
        )

    z_min = int(
        coordinates[0].min()
    )

    z_max = int(
        coordinates[0].max()
    ) + 1

    y_min = int(
        coordinates[1].min()
    )

    y_max = int(
        coordinates[1].max()
    ) + 1

    x_min = int(
        coordinates[2].min()
    )

    x_max = int(
        coordinates[2].max()
    ) + 1

    return (
        z_min,
        z_max,
        y_min,
        y_max,
        x_min,
        x_max,
    )


# ============================================================
# HELPER: LIMIT EXTREME DIMENSIONS
# ============================================================

def limit_crop(
    array,
    bbox,
    max_dimension=MAX_DIMENSION
):
    """
    Prevent an unexpectedly large crop from
    producing excessive memory requirements.

    This is a safety mechanism for the
    4 GB GPU workflow.
    """

    (
        z_min,
        z_max,
        y_min,
        y_max,
        x_min,
        x_max,
    ) = bbox

    shape = array.shape

    # --------------------------------------------------------
    # Z
    # --------------------------------------------------------

    if (
        z_max - z_min
        >
        max_dimension
    ):

        center = (
            z_min + z_max
        ) // 2

        half = (
            max_dimension
            // 2
        )

        z_min = max(
            0,
            center - half
        )

        z_max = min(
            shape[0],
            z_min + max_dimension
        )

    # --------------------------------------------------------
    # Y
    # --------------------------------------------------------

    if (
        y_max - y_min
        >
        max_dimension
    ):

        center = (
            y_min + y_max
        ) // 2

        half = (
            max_dimension
            // 2
        )

        y_min = max(
            0,
            center - half
        )

        y_max = min(
            shape[1],
            y_min + max_dimension
        )

    # --------------------------------------------------------
    # X
    # --------------------------------------------------------

    if (
        x_max - x_min
        >
        max_dimension
    ):

        center = (
            x_min + x_max
        ) // 2

        half = (
            max_dimension
            // 2
        )

        x_min = max(
            0,
            center - half
        )

        x_max = min(
            shape[2],
            x_min + max_dimension
        )

    return (
        z_min,
        z_max,
        y_min,
        y_max,
        x_min,
        x_max,
    )


# ============================================================
# HELPER: CROP
# ============================================================

def crop_array(
    array,
    bbox
):

    (
        z_min,
        z_max,
        y_min,
        y_max,
        x_min,
        x_max,
    ) = bbox

    return array[
        z_min:z_max,
        y_min:y_max,
        x_min:x_max,
    ]


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(
    mri_path,
    mask_path
):

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    mri_image = sitk.ReadImage(
        str(mri_path)
    )

    mask_image = sitk.ReadImage(
        str(mask_path)
    )

    # --------------------------------------------------------
    # STANDARDIZE ORIENTATION
    # --------------------------------------------------------

    mri_image = standardize_orientation(
        mri_image
    )

    mask_image = standardize_orientation(
        mask_image
    )

    # --------------------------------------------------------
    # VERIFY IMAGE / MASK GEOMETRY
    # --------------------------------------------------------

    if (
        mri_image.GetSize()
        !=
        mask_image.GetSize()
    ):

        raise RuntimeError(
            "MRI/mask size mismatch: "
            f"{mri_path.name}"
        )

    # --------------------------------------------------------
    # CONVERT TO ARRAYS
    # --------------------------------------------------------

    mri_array = sitk.GetArrayFromImage(
        mri_image
    ).astype(
        np.float32
    )

    mask_array = sitk.GetArrayFromImage(
        mask_image
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    normalized, low, high = (
        normalize_mri(
            mri_array
        )
    )

    # --------------------------------------------------------
    # FOREGROUND BBOX
    # --------------------------------------------------------

    bbox = get_foreground_bbox(
        normalized
    )

    bbox = limit_crop(
        normalized,
        bbox
    )

    # --------------------------------------------------------
    # CROP MRI
    # --------------------------------------------------------

    cropped_mri = crop_array(
        normalized,
        bbox
    )

    # --------------------------------------------------------
    # CROP MASK USING SAME BBOX
    # --------------------------------------------------------

    cropped_mask = crop_array(
        mask_array,
        bbox
    )

    # --------------------------------------------------------
    # VALIDATE LABELS
    # --------------------------------------------------------

    valid_labels = {
        0,
        1,
        2,
        3,
    }

    actual_labels = set(
        np.unique(
            cropped_mask
        ).tolist()
    )

    invalid_labels = (
        actual_labels
        -
        valid_labels
    )

    if invalid_labels:

        raise RuntimeError(
            f"Invalid labels in "
            f"{mask_path.name}: "
            f"{sorted(invalid_labels)}"
        )

    # --------------------------------------------------------
    # SAVE MRI
    # --------------------------------------------------------

    output_mri = (
        PROCESSED_MRI_DIR
        /
        mri_path.name
    )

    output_mask = (
        PROCESSED_MASK_DIR
        /
        mask_path.name
    )

    # --------------------------------------------------------
    # MRI IMAGE
    # --------------------------------------------------------

    mri_output_image = (
        sitk.GetImageFromArray(
            cropped_mri
        )
    )

    mri_output_image.SetSpacing(
        mri_image.GetSpacing()
    )

    mri_output_image.SetOrigin(
        mri_image.GetOrigin()
    )

    mri_output_image.SetDirection(
        mri_image.GetDirection()
    )

    mri_output_image = sitk.Cast(
        mri_output_image,
        sitk.sitkFloat32
    )

    sitk.WriteImage(
        mri_output_image,
        str(output_mri)
    )

    # --------------------------------------------------------
    # MASK IMAGE
    # --------------------------------------------------------

    mask_output_image = (
        sitk.GetImageFromArray(
            cropped_mask
        )
    )

    mask_output_image.SetSpacing(
        mri_image.GetSpacing()
    )

    mask_output_image.SetOrigin(
        mri_image.GetOrigin()
    )

    mask_output_image.SetDirection(
        mri_image.GetDirection()
    )

    mask_output_image = sitk.Cast(
        mask_output_image,
        sitk.sitkUInt8
    )

    sitk.WriteImage(
        mask_output_image,
        str(output_mask)
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    return {
        "file": mri_path.name,

        "original_z":
            int(mri_array.shape[0]),

        "original_y":
            int(mri_array.shape[1]),

        "original_x":
            int(mri_array.shape[2]),

        "processed_z":
            int(cropped_mri.shape[0]),

        "processed_y":
            int(cropped_mri.shape[1]),

        "processed_x":
            int(cropped_mri.shape[2]),

        "spacing_x":
            float(
                mri_image.GetSpacing()[0]
            ),

        "spacing_y":
            float(
                mri_image.GetSpacing()[1]
            ),

        "spacing_z":
            float(
                mri_image.GetSpacing()[2]
            ),

        "clip_low":
            low,

        "clip_high":
            high,

        "foreground_voxels":
            int(
                np.count_nonzero(
                    cropped_mri
                    >
                    FOREGROUND_THRESHOLD
                )
            ),

        "mask_foreground_voxels":
            int(
                np.count_nonzero(
                    cropped_mask
                )
            ),

        "mask_labels":
            ",".join(
                str(x)
                for x in sorted(
                    actual_labels
                )
            ),
    }


# ============================================================
# PROCESS DATASET
# ============================================================

records = []


print("\n" + "=" * 75)
print("PROCESSING DATASET")
print("=" * 75)


for index, mri_path in enumerate(
    paired_files,
    start=1
):

    mask_path = mask_map[
        mri_path.name
    ]

    try:

        result = process_case(
            mri_path,
            mask_path
        )

        records.append(
            result
        )

    except Exception as exc:

        print(
            "\nERROR:",
            mri_path.name
        )

        print(
            exc
        )

    if (
        index == 1
        or index % 25 == 0
        or index == len(paired_files)
    ):

        print(
            f"Processed "
            f"{index:3d} / "
            f"{len(paired_files)}"
        )


# ============================================================
# CREATE STATISTICS DATAFRAME
# ============================================================

stats_df = pd.DataFrame(
    records
)


if stats_df.empty:

    raise RuntimeError(
        "No cases were successfully processed."
    )


# ============================================================
# SAVE STATISTICS
# ============================================================

stats_path = (
    STATISTICS_DIR
    /
    "preprocessing_statistics.csv"
)

stats_df.to_csv(
    stats_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PREPROCESSING SUMMARY")
print("=" * 75)

print(
    "Successfully processed:",
    len(stats_df)
)

print(
    "Failed:",
    len(paired_files)
    -
    len(stats_df)
)


print("\nOriginal dimensions")

print(
    stats_df[
        [
            "original_z",
            "original_y",
            "original_x",
        ]
    ]
    .describe()
    .to_string()
)


print("\nProcessed dimensions")

print(
    stats_df[
        [
            "processed_z",
            "processed_y",
            "processed_x",
        ]
    ]
    .describe()
    .to_string()
)


print("\nSpacing")

print(
    stats_df[
        [
            "spacing_x",
            "spacing_y",
            "spacing_z",
        ]
    ]
    .describe()
    .to_string()
)


# ============================================================
# LABEL VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("PROCESSED MASK LABEL VALIDATION")
print("=" * 75)


all_labels = set()

for value in stats_df[
    "mask_labels"
]:

    if not value:

        continue

    for label in value.split(","):

        all_labels.add(
            int(label)
        )


print(
    "Labels found:",
    sorted(all_labels)
)


expected_labels = {
    0,
    1,
    2,
    3,
}


unexpected = (
    all_labels
    -
    expected_labels
)


if unexpected:

    print(
        "WARNING: Unexpected labels:",
        sorted(unexpected)
    )

else:

    print(
        "✓ All processed masks contain "
        "only labels 0, 1, 2, 3."
    )


# ============================================================
# OUTPUT COUNTS
# ============================================================

processed_mri_count = len(
    list(
        PROCESSED_MRI_DIR.glob(
            "*.mha"
        )
    )
)

processed_mask_count = len(
    list(
        PROCESSED_MASK_DIR.glob(
            "*.mha"
        )
    )
)


print("\n" + "=" * 75)
print("PROCESSED DATASET")
print("=" * 75)

print(
    "Processed MRI files:",
    processed_mri_count
)

print(
    "Processed mask files:",
    processed_mask_count
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT FILE")
print("=" * 75)

print(
    "Statistics:",
    stats_path
)

print(
    "Processed MRI:",
    PROCESSED_MRI_DIR
)

print(
    "Processed masks:",
    PROCESSED_MASK_DIR
)


print("\n" + "=" * 75)
print("PHASE 3 - PART 5 COMPLETE")
print("=" * 75)