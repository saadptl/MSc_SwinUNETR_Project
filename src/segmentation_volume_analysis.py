from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# PHASE 3 - PART 4
# SPIDER 3D MRI VOLUME & SPACING ANALYSIS
# ============================================================

print("=" * 75)
print("PHASE 3 - PART 4")
print("SPIDER 3D MRI VOLUME & SPACING ANALYSIS")
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

IMAGES_DIR = (
    SPIDER_DIR
    / "images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "volume_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nSPIDER DIRECTORY")
print(SPIDER_DIR)

print("\nMRI DIRECTORY")
print(IMAGES_DIR)

print("\nOUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================
# FIND MRI VOLUMES
# ============================================================

image_files = sorted(
    IMAGES_DIR.rglob("*.mha")
)


print("\nMRI FILE COUNT")

print(
    "MRI volumes:",
    len(image_files)
)


if not image_files:

    raise RuntimeError(
        "No .mha MRI volumes were found."
    )


# ============================================================
# ANALYSIS STORAGE
# ============================================================

records = []


# ============================================================
# SCAN ALL VOLUMES
# ============================================================

print("\n" + "=" * 75)
print("SCANNING MRI VOLUMES")
print("=" * 75)


for index, image_path in enumerate(
    image_files,
    start=1
):

    try:

        image = sitk.ReadImage(
            str(image_path)
        )

        array = sitk.GetArrayFromImage(
            image
        )

        # ----------------------------------------------------
        # SimpleITK dimensions
        # [X, Y, Z]
        # ----------------------------------------------------

        size_x, size_y, size_z = (
            image.GetSize()
        )

        # ----------------------------------------------------
        # Spacing
        # ----------------------------------------------------

        spacing_x, spacing_y, spacing_z = (
            image.GetSpacing()
        )

        # ----------------------------------------------------
        # Physical field of view
        # ----------------------------------------------------

        fov_x = (
            size_x
            *
            spacing_x
        )

        fov_y = (
            size_y
            *
            spacing_y
        )

        fov_z = (
            size_z
            *
            spacing_z
        )

        # ----------------------------------------------------
        # Intensity statistics
        # ----------------------------------------------------

        minimum = float(
            np.min(array)
        )

        maximum = float(
            np.max(array)
        )

        mean = float(
            np.mean(array)
        )

        std = float(
            np.std(array)
        )

        percentile_1 = float(
            np.percentile(
                array,
                1
            )
        )

        percentile_99 = float(
            np.percentile(
                array,
                99
            )
        )

        nonzero_voxels = int(
            np.count_nonzero(array)
        )

        total_voxels = int(
            array.size
        )

        # ----------------------------------------------------
        # Sequence
        # ----------------------------------------------------

        filename = image_path.name

        filename_lower = (
            filename.lower()
        )

        if "_t2_space" in filename_lower:

            sequence = "T2_SPACE"

        elif "_t2" in filename_lower:

            sequence = "T2"

        elif "_t1" in filename_lower:

            sequence = "T1"

        else:

            sequence = "Unknown"

        # ----------------------------------------------------
        # Patient/study identifier
        # ----------------------------------------------------

        stem = image_path.stem

        if "_" in stem:

            patient_id = (
                stem.split("_")[0]
            )

        else:

            patient_id = stem

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        records.append(
            {
                "file": filename,
                "patient_id": patient_id,
                "sequence": sequence,

                "size_x": size_x,
                "size_y": size_y,
                "size_z": size_z,

                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "spacing_z": spacing_z,

                "fov_x_mm": fov_x,
                "fov_y_mm": fov_y,
                "fov_z_mm": fov_z,

                "total_voxels":
                    total_voxels,

                "nonzero_voxels":
                    nonzero_voxels,

                "intensity_min":
                    minimum,

                "intensity_max":
                    maximum,

                "intensity_mean":
                    mean,

                "intensity_std":
                    std,

                "intensity_p01":
                    percentile_1,

                "intensity_p99":
                    percentile_99,
            }
        )

    except Exception as exc:

        print(
            "\nERROR:",
            image_path.name
        )

        print(
            exc
        )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index == 1
        or index % 50 == 0
        or index == len(image_files)
    ):

        print(
            f"Processed "
            f"{index:3d} / "
            f"{len(image_files)}"
        )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    records
)


if df.empty:

    raise RuntimeError(
        "No MRI volumes could be analyzed."
    )


# ============================================================
# BASIC DATASET SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("DATASET SUMMARY")
print("=" * 75)

print(
    "Analyzed volumes:",
    len(df)
)

print(
    "Unique patients:",
    df["patient_id"].nunique()
)


# ============================================================
# SEQUENCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("SEQUENCE DISTRIBUTION")
print("=" * 75)

sequence_counts = (
    df["sequence"]
    .value_counts()
)


print(
    sequence_counts.to_string()
)


# ============================================================
# DIMENSION STATISTICS
# ============================================================

dimension_columns = [
    "size_x",
    "size_y",
    "size_z",
]


dimension_summary = (
    df[
        dimension_columns
    ]
    .describe()
    .T
)


print("\n" + "=" * 75)
print("VOLUME DIMENSION STATISTICS")
print("=" * 75)

print(
    dimension_summary.to_string()
)


# ============================================================
# SPACING STATISTICS
# ============================================================

spacing_columns = [
    "spacing_x",
    "spacing_y",
    "spacing_z",
]


spacing_summary = (
    df[
        spacing_columns
    ]
    .describe()
    .T
)


print("\n" + "=" * 75)
print("VOXEL SPACING STATISTICS")
print("=" * 75)

print(
    spacing_summary.to_string()
)


# ============================================================
# PHYSICAL FOV STATISTICS
# ============================================================

fov_columns = [
    "fov_x_mm",
    "fov_y_mm",
    "fov_z_mm",
]


fov_summary = (
    df[
        fov_columns
    ]
    .describe()
    .T
)


print("\n" + "=" * 75)
print("PHYSICAL FIELD-OF-VIEW STATISTICS")
print("=" * 75)

print(
    fov_summary.to_string()
)


# ============================================================
# INTENSITY STATISTICS
# ============================================================

intensity_columns = [
    "intensity_min",
    "intensity_max",
    "intensity_mean",
    "intensity_std",
    "intensity_p01",
    "intensity_p99",
]


intensity_summary = (
    df[
        intensity_columns
    ]
    .describe()
    .T
)


print("\n" + "=" * 75)
print("INTENSITY STATISTICS")
print("=" * 75)

print(
    intensity_summary.to_string()
)


# ============================================================
# SEQUENCE-WISE SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SEQUENCE-WISE DIMENSION SUMMARY")
print("=" * 75)


sequence_dimension_summary = (
    df.groupby(
        "sequence"
    )[
        [
            "size_x",
            "size_y",
            "size_z",
            "spacing_x",
            "spacing_y",
            "spacing_z",
        ]
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)


print(
    sequence_dimension_summary.to_string()
)


# ============================================================
# UNIQUE VOLUME SHAPES
# ============================================================

shape_counts = (
    df[
        [
            "size_x",
            "size_y",
            "size_z",
        ]
    ]
    .value_counts()
    .reset_index(
        name="count"
    )
)


print("\n" + "=" * 75)
print("MOST COMMON VOLUME SHAPES")
print("=" * 75)

print(
    shape_counts.head(
        20
    ).to_string(
        index=False
    )
)


# ============================================================
# UNIQUE SPACING PATTERNS
# ============================================================

spacing_counts = (
    df[
        [
            "spacing_x",
            "spacing_y",
            "spacing_z",
        ]
    ]
    .round(4)
    .value_counts()
    .reset_index(
        name="count"
    )
)


print("\n" + "=" * 75)
print("MOST COMMON SPACING PATTERNS")
print("=" * 75)

print(
    spacing_counts.head(
        20
    ).to_string(
        index=False
    )
)


# ============================================================
# ANISOTROPY ANALYSIS
# ============================================================

df["spacing_max"] = (
    df[
        [
            "spacing_x",
            "spacing_y",
            "spacing_z",
        ]
    ]
    .max(
        axis=1
    )
)

df["spacing_min"] = (
    df[
        [
            "spacing_x",
            "spacing_y",
            "spacing_z",
        ]
    ]
    .min(
        axis=1
    )
)

df["anisotropy_ratio"] = (
    df["spacing_max"]
    /
    df["spacing_min"]
)


print("\n" + "=" * 75)
print("SPACING ANISOTROPY")
print("=" * 75)

print(
    df[
        "anisotropy_ratio"
    ]
    .describe()
    .to_string()
)


# ============================================================
# HIGHLY ANISOTROPIC CASES
# ============================================================

high_anisotropy = df[
    df[
        "anisotropy_ratio"
    ]
    >= 3.0
].copy()


print(
    "\nCases with spacing anisotropy ratio >= 3:",
    len(high_anisotropy)
)


# ============================================================
# Z-AXIS ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("Z-AXIS / SLICE ANALYSIS")
print("=" * 75)

print(
    "Minimum Z slices:",
    df["size_z"].min()
)

print(
    "Maximum Z slices:",
    df["size_z"].max()
)

print(
    "Median Z slices:",
    df["size_z"].median()
)

print(
    "Minimum Z spacing:",
    df["spacing_z"].min()
)

print(
    "Maximum Z spacing:",
    df["spacing_z"].max()
)

print(
    "Median Z spacing:",
    df["spacing_z"].median()
)


# ============================================================
# PHYSICAL DEPTH
# ============================================================

print("\nPhysical Z coverage:")

print(
    "Minimum:",
    df["fov_z_mm"].min(),
    "mm"
)

print(
    "Maximum:",
    df["fov_z_mm"].max(),
    "mm"
)

print(
    "Median:",
    df["fov_z_mm"].median(),
    "mm"
)


# ============================================================
# SAVE COMPLETE VOLUME TABLE
# ============================================================

volume_table_path = (
    OUTPUT_DIR
    /
    "spider_volume_analysis.csv"
)


df.to_csv(
    volume_table_path,
    index=False
)


# ============================================================
# SAVE SUMMARY TABLES
# ============================================================

dimension_summary_path = (
    OUTPUT_DIR
    /
    "dimension_summary.csv"
)

dimension_summary.to_csv(
    dimension_summary_path
)


spacing_summary_path = (
    OUTPUT_DIR
    /
    "spacing_summary.csv"
)

spacing_summary.to_csv(
    spacing_summary_path
)


fov_summary_path = (
    OUTPUT_DIR
    /
    "fov_summary.csv"
)

fov_summary.to_csv(
    fov_summary_path
)


intensity_summary_path = (
    OUTPUT_DIR
    /
    "intensity_summary.csv"
)

intensity_summary.to_csv(
    intensity_summary_path
)


sequence_summary_path = (
    OUTPUT_DIR
    /
    "sequence_dimension_summary.csv"
)

sequence_dimension_summary.to_csv(
    sequence_summary_path
)


# ============================================================
# SAVE SHAPE FREQUENCIES
# ============================================================

shape_counts_path = (
    OUTPUT_DIR
    /
    "volume_shape_distribution.csv"
)

shape_counts.to_csv(
    shape_counts_path,
    index=False
)


# ============================================================
# SAVE SPACING FREQUENCIES
# ============================================================

spacing_counts_path = (
    OUTPUT_DIR
    /
    "spacing_distribution.csv"
)

spacing_counts.to_csv(
    spacing_counts_path,
    index=False
)


# ============================================================
# SAVE ANISOTROPY CASES
# ============================================================

anisotropy_path = (
    OUTPUT_DIR
    /
    "high_anisotropy_cases.csv"
)

high_anisotropy.to_csv(
    anisotropy_path,
    index=False
)


# ============================================================
# VISUALIZATION 1
# VOLUME SIZE DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    df["size_x"],
    bins=20
)

ax.set_title(
    "SPIDER MRI X-Dimension Distribution",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "X Dimension (voxels)"
)

ax.set_ylabel(
    "Number of Volumes"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    /
    "x_dimension_distribution.png"
)

plt.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# VISUALIZATION 2
# Y DIMENSION
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    df["size_y"],
    bins=20
)

ax.set_title(
    "SPIDER MRI Y-Dimension Distribution",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Y Dimension (voxels)"
)

ax.set_ylabel(
    "Number of Volumes"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    /
    "y_dimension_distribution.png"
)

plt.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# VISUALIZATION 3
# Z DIMENSION
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    df["size_z"],
    bins=20
)

ax.set_title(
    "SPIDER MRI Z-Dimension Distribution",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Number of Slices"
)

ax.set_ylabel(
    "Number of Volumes"
)

plt.tight_layout()

path = (
    OUTPUT_DIR
    /
    "z_dimension_distribution.png"
)

plt.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# VISUALIZATION 4
# SPACING DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    df["spacing_x"],
    bins=20,
    alpha=0.7,
    label="X spacing"
)

ax.hist(
    df["spacing_y"],
    bins=20,
    alpha=0.7,
    label="Y spacing"
)

ax.hist(
    df["spacing_z"],
    bins=20,
    alpha=0.7,
    label="Z spacing"
)

ax.set_title(
    "SPIDER MRI Voxel Spacing Distribution",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Voxel Spacing (mm)"
)

ax.set_ylabel(
    "Number of Volumes"
)

ax.legend()

plt.tight_layout()

path = (
    OUTPUT_DIR
    /
    "spacing_distribution.png"
)

plt.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# VISUALIZATION 5
# PHYSICAL FOV
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    df["fov_x_mm"],
    bins=20,
    alpha=0.7,
    label="FOV X"
)

ax.hist(
    df["fov_y_mm"],
    bins=20,
    alpha=0.7,
    label="FOV Y"
)

ax.hist(
    df["fov_z_mm"],
    bins=20,
    alpha=0.7,
    label="FOV Z"
)

ax.set_title(
    "SPIDER MRI Physical Field-of-View",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Physical Size (mm)"
)

ax.set_ylabel(
    "Number of Volumes"
)

ax.legend()

plt.tight_layout()

path = (
    OUTPUT_DIR
    /
    "physical_fov_distribution.png"
)

plt.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# VISUALIZATION 6
# ANISOTROPY
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    df["anisotropy_ratio"],
    bins=20
)

ax.axvline(
    3.0,
    linestyle="--",
    label="Ratio = 3"
)

ax.set_title(
    "SPIDER MRI Spacing Anisotropy",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Maximum / Minimum Spacing Ratio"
)

ax.set_ylabel(
    "Number of Volumes"
)

ax.legend()

plt.tight_layout()

path = (
    OUTPUT_DIR
    /
    "spacing_anisotropy_distribution.png"
)

plt.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print(
    "Complete volume table:",
    volume_table_path
)

print(
    "Dimension summary:",
    dimension_summary_path
)

print(
    "Spacing summary:",
    spacing_summary_path
)

print(
    "FOV summary:",
    fov_summary_path
)

print(
    "Intensity summary:",
    intensity_summary_path
)

print(
    "Sequence summary:",
    sequence_summary_path
)

print(
    "Shape distribution:",
    shape_counts_path
)

print(
    "Spacing distribution:",
    spacing_counts_path
)

print(
    "High anisotropy cases:",
    anisotropy_path
)


print("\n" + "=" * 75)
print("PHASE 3 - PART 4 COMPLETE")
print("=" * 75)