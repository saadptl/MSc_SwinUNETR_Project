from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# PHASE 3 - PART 2
# SPIDER MRI + SEGMENTATION MASK INSPECTION
# ============================================================

print("=" * 70)
print("PHASE 3 - PART 2")
print("SPIDER MRI + SEGMENTATION MASK INSPECTION")
print("=" * 70)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPIDER_DIR = PROJECT_ROOT / "dataset" / "spider"

IMAGES_DIR = SPIDER_DIR / "images"
MASKS_DIR = SPIDER_DIR / "masks"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BASIC PATH VERIFICATION
# ============================================================

print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nSPIDER DIRECTORY")
print(SPIDER_DIR)

print("\nIMAGE DIRECTORY")
print(IMAGES_DIR)

print("\nMASK DIRECTORY")
print(MASKS_DIR)


if not IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGES_DIR}"
    )

if not MASKS_DIR.exists():
    raise FileNotFoundError(
        f"Mask directory not found:\n{MASKS_DIR}"
    )


# ============================================================
# FIND MHA FILES
# ============================================================

image_files = sorted(
    IMAGES_DIR.glob("*.mha")
)

mask_files = sorted(
    MASKS_DIR.glob("*.mha")
)


print("\nFILE COUNTS")

print(
    "MRI .mha files :",
    len(image_files)
)

print(
    "Mask .mha files:",
    len(mask_files)
)


if not image_files:
    raise RuntimeError(
        "No .mha MRI files were found."
    )

if not mask_files:
    raise RuntimeError(
        "No .mha mask files were found."
    )


# ============================================================
# DISPLAY FIRST FILES
# ============================================================

print("\nFIRST 10 MRI FILES")

for path in image_files[:10]:
    print(
        " ",
        path.name
    )


print("\nFIRST 10 MASK FILES")

for path in mask_files[:10]:
    print(
        " ",
        path.name
    )


# ============================================================
# MATCH MRI AND MASK FILES
# ============================================================

image_map = {
    path.name: path
    for path in image_files
}

mask_map = {
    path.name: path
    for path in mask_files
}


common_names = sorted(
    set(image_map.keys())
    &
    set(mask_map.keys())
)


missing_masks = sorted(
    set(image_map.keys())
    -
    set(mask_map.keys())
)

missing_images = sorted(
    set(mask_map.keys())
    -
    set(image_map.keys())
)


print("\nMRI / MASK PAIRING")

print(
    "Matching pairs:",
    len(common_names)
)

print(
    "MRI without mask:",
    len(missing_masks)
)

print(
    "Mask without MRI:",
    len(missing_images)
)


if missing_masks:

    print("\nFIRST MISSING MASKS")

    for name in missing_masks[:10]:
        print(
            " ",
            name
        )


if missing_images:

    print("\nFIRST MISSING IMAGES")

    for name in missing_images[:10]:
        print(
            " ",
            name
        )


if not common_names:

    raise RuntimeError(
        "No matching MRI-mask pairs were found."
    )


# ============================================================
# SELECT FIRST MATCHING CASE
# ============================================================

selected_name = common_names[0]

image_path = image_map[selected_name]
mask_path = mask_map[selected_name]


print("\nSELECTED CASE")

print(
    "MRI :",
    image_path
)

print(
    "MASK:",
    mask_path
)


# ============================================================
# LOAD MHA IMAGE
# ============================================================

print("\nLOADING MRI")

image_sitk = sitk.ReadImage(
    str(image_path)
)

image_array = sitk.GetArrayFromImage(
    image_sitk
)


# ============================================================
# LOAD MHA MASK
# ============================================================

print("LOADING MASK")

mask_sitk = sitk.ReadImage(
    str(mask_path)
)

mask_array = sitk.GetArrayFromImage(
    mask_sitk
)


# ============================================================
# MRI INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MRI INFORMATION")
print("=" * 70)

print(
    "File:",
    image_path.name
)

print(
    "Array shape [Z,Y,X]:",
    image_array.shape
)

print(
    "SimpleITK size [X,Y,Z]:",
    image_sitk.GetSize()
)

print(
    "Voxel spacing:",
    image_sitk.GetSpacing()
)

print(
    "Origin:",
    image_sitk.GetOrigin()
)

print(
    "Direction:",
    image_sitk.GetDirection()
)

print(
    "Pixel type:",
    image_sitk.GetPixelIDTypeAsString()
)

print(
    "Minimum:",
    float(np.min(image_array))
)

print(
    "Maximum:",
    float(np.max(image_array))
)

print(
    "Mean:",
    float(np.mean(image_array))
)

print(
    "Non-zero voxels:",
    int(np.count_nonzero(image_array))
)


# ============================================================
# MASK INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MASK INFORMATION")
print("=" * 70)

print(
    "File:",
    mask_path.name
)

print(
    "Array shape [Z,Y,X]:",
    mask_array.shape
)

print(
    "SimpleITK size [X,Y,Z]:",
    mask_sitk.GetSize()
)

print(
    "Voxel spacing:",
    mask_sitk.GetSpacing()
)

print(
    "Origin:",
    mask_sitk.GetOrigin()
)

print(
    "Direction:",
    mask_sitk.GetDirection()
)

print(
    "Pixel type:",
    mask_sitk.GetPixelIDTypeAsString()
)


# ============================================================
# DIMENSION CHECK
# ============================================================

print("\n" + "=" * 70)
print("MRI / MASK COMPATIBILITY")
print("=" * 70)


if image_array.shape == mask_array.shape:

    print(
        "✓ Array dimensions MATCH"
    )

else:

    print(
        "✗ Array dimensions DO NOT MATCH"
    )

    print(
        "MRI shape :",
        image_array.shape
    )

    print(
        "Mask shape:",
        mask_array.shape
    )


# ============================================================
# SPATIAL METADATA CHECK
# ============================================================

same_size = (
    image_sitk.GetSize()
    ==
    mask_sitk.GetSize()
)

same_spacing = np.allclose(
    image_sitk.GetSpacing(),
    mask_sitk.GetSpacing(),
    atol=1e-5
)

same_origin = np.allclose(
    image_sitk.GetOrigin(),
    mask_sitk.GetOrigin(),
    atol=1e-5
)

same_direction = np.allclose(
    image_sitk.GetDirection(),
    mask_sitk.GetDirection(),
    atol=1e-5
)


print(
    "Same size      :",
    same_size
)

print(
    "Same spacing   :",
    same_spacing
)

print(
    "Same origin    :",
    same_origin
)

print(
    "Same direction :",
    same_direction
)


# ============================================================
# UNIQUE MASK LABELS
# ============================================================

print("\n" + "=" * 70)
print("SEGMENTATION MASK LABELS")
print("=" * 70)


unique_labels, label_counts = np.unique(
    mask_array,
    return_counts=True
)


print(
    "Number of unique labels:",
    len(unique_labels)
)


for label, count in zip(
    unique_labels,
    label_counts
):

    print(
        f"Label {label}"
        f" -> {count:,} voxels"
    )


# ============================================================
# NON-ZERO LABELS
# ============================================================

nonzero_labels = (
    unique_labels[
        unique_labels != 0
    ]
)


print("\nNON-ZERO LABELS")

if len(nonzero_labels) == 0:

    print(
        "WARNING: mask contains only background."
    )

else:

    for label in nonzero_labels:

        print(
            f"  {label}"
        )


# ============================================================
# MASK COVERAGE
# ============================================================

total_voxels = mask_array.size

foreground_voxels = np.count_nonzero(
    mask_array
)

foreground_percentage = (
    foreground_voxels
    /
    total_voxels
    *
    100
)


print("\nMASK COVERAGE")

print(
    f"Total voxels      : {total_voxels:,}"
)

print(
    f"Foreground voxels : {foreground_voxels:,}"
)

print(
    f"Foreground ratio  : "
    f"{foreground_percentage:.4f}%"
)


# ============================================================
# SELECT REPRESENTATIVE SLICE
# ============================================================

num_slices = image_array.shape[0]


# Find slices containing segmentation
mask_slice_counts = np.count_nonzero(
    mask_array,
    axis=(1, 2)
)


nonzero_slice_indices = np.where(
    mask_slice_counts > 0
)[0]


if len(nonzero_slice_indices) > 0:

    representative_index = int(
        nonzero_slice_indices[
            len(nonzero_slice_indices) // 2
        ]
    )

    print(
        "\nRepresentative segmentation slice:",
        representative_index
    )

else:

    representative_index = (
        num_slices // 2
    )

    print(
        "\nNo foreground mask detected."
    )

    print(
        "Using middle MRI slice:",
        representative_index
    )


# ============================================================
# EXTRACT SLICE
# ============================================================

mri_slice = image_array[
    representative_index,
    :,
    :
]

mask_slice = mask_array[
    representative_index,
    :,
    :
]


# ============================================================
# MRI DISPLAY NORMALIZATION
# ============================================================

low = np.percentile(
    mri_slice,
    1
)

high = np.percentile(
    mri_slice,
    99
)


if high > low:

    display_slice = np.clip(
        mri_slice,
        low,
        high
    )

    display_slice = (
        display_slice - low
    ) / (
        high - low
    )

else:

    display_slice = (
        mri_slice
        - np.min(mri_slice)
    )

    denominator = (
        np.max(mri_slice)
        -
        np.min(mri_slice)
    )

    if denominator > 0:

        display_slice = (
            display_slice
            /
            denominator
        )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


# ============================================================
# MRI
# ============================================================

axes[0].imshow(
    display_slice,
    cmap="gray",
    origin="lower"
)

axes[0].set_title(
    "MRI Slice",
    fontsize=15,
    fontweight="bold"
)

axes[0].axis("off")


# ============================================================
# GROUND-TRUTH MASK
# ============================================================

axes[1].imshow(
    mask_slice,
    origin="lower",
    interpolation="nearest"
)

axes[1].set_title(
    "Ground-Truth Segmentation",
    fontsize=15,
    fontweight="bold"
)

axes[1].axis("off")


# ============================================================
# OVERLAY
# ============================================================

axes[2].imshow(
    display_slice,
    cmap="gray",
    origin="lower"
)

mask_overlay = np.ma.masked_where(
    mask_slice == 0,
    mask_slice
)


axes[2].imshow(
    mask_overlay,
    alpha=0.55,
    origin="lower",
    interpolation="nearest"
)

axes[2].set_title(
    "MRI + Ground-Truth Overlay",
    fontsize=15,
    fontweight="bold"
)

axes[2].axis("off")


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    (
        "SPIDER Dataset — MRI Segmentation Inspection\n"
        f"{selected_name} | Slice {representative_index}"
    ),
    fontsize=17,
    fontweight="bold"
)


plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

output_path = (
    OUTPUT_DIR
    / "phase3_part2_mri_mask_overlay.png"
)


plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)


print("\n" + "=" * 70)
print("VISUALIZATION")
print("=" * 70)

print(
    "Saved:",
    output_path
)


# ============================================================
# SAVE INSPECTION SUMMARY
# ============================================================

summary = {
    "case": selected_name,
    "image_shape": str(
        image_array.shape
    ),
    "mask_shape": str(
        mask_array.shape
    ),
    "voxel_spacing": str(
        image_sitk.GetSpacing()
    ),
    "same_size": same_size,
    "same_spacing": same_spacing,
    "same_origin": same_origin,
    "same_direction": same_direction,
    "unique_mask_labels": str(
        unique_labels.tolist()
    ),
    "foreground_voxels": foreground_voxels,
    "total_voxels": total_voxels,
    "foreground_percentage":
        foreground_percentage,
    "representative_slice":
        representative_index,
}


summary_df = pd.DataFrame(
    [summary]
)


summary_path = (
    OUTPUT_DIR
    / "phase3_part2_inspection_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


print(
    "Summary:",
    summary_path
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3 - PART 2 COMPLETE")
print("=" * 70)


plt.show()