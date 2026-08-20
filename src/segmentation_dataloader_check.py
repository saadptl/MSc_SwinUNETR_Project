from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PHASE 3 - PART 6
# SWIN-UNETR DATASET & DATALOADER VALIDATION
# ============================================================

print("=" * 75)
print("PHASE 3 - PART 6")
print("SWIN-UNETR DATASET & DATALOADER VALIDATION")
print("=" * 75)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MRI_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "images_preprocessed"
)

MASK_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "masks_preprocessed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "dataloader"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nPREPROCESSED MRI DIRECTORY")
print(MRI_DIR)

print("\nPREPROCESSED MASK DIRECTORY")
print(MASK_DIR)

print("\nOUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = (
    96,
    96,
    96
)

BATCH_SIZE = 1

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()

RANDOM_SEED = 42

# Expected 4-class segmentation labels
EXPECTED_LABELS = {
    0,
    1,
    2,
    3,
}


np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


print("\n" + "=" * 75)
print("DATALOADER CONFIGURATION")
print("=" * 75)

print(
    "Patch size:",
    PATCH_SIZE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Workers:",
    NUM_WORKERS
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

print(
    "Pin memory:",
    PIN_MEMORY
)


# ============================================================
# FILE DISCOVERY
# ============================================================

mri_files = sorted(
    MRI_DIR.glob("*.mha")
)

mask_files = sorted(
    MASK_DIR.glob("*.mha")
)


print("\n" + "=" * 75)
print("DATASET FILE COUNTS")
print("=" * 75)

print(
    "MRI files:",
    len(mri_files)
)

print(
    "Mask files:",
    len(mask_files)
)


if not mri_files:

    raise RuntimeError(
        "No preprocessed MRI files found."
    )


if not mask_files:

    raise RuntimeError(
        "No preprocessed mask files found."
    )


# ============================================================
# MASK LOOKUP
# ============================================================

mask_map = {
    path.name: path
    for path in mask_files
}


pairs = [
    (
        mri_path,
        mask_map[mri_path.name]
    )
    for mri_path in mri_files
    if mri_path.name in mask_map
]


print("\nMRI / MASK PAIRS")

print(
    "Matching pairs:",
    len(pairs)
)

print(
    "Missing masks:",
    len(mri_files) - len(pairs)
)


if not pairs:

    raise RuntimeError(
        "No MRI/mask pairs found."
    )


# ============================================================
# HELPER
# ============================================================

def pad_to_patch(
    array,
    patch_size,
    constant_value=0
):
    """
    Pad a 3D array so that every spatial
    dimension is at least the requested
    patch size.

    Array order:
        [Z, Y, X]
    """

    target_z, target_y, target_x = (
        patch_size
    )

    z, y, x = array.shape

    pad_z = max(
        0,
        target_z - z
    )

    pad_y = max(
        0,
        target_y - y
    )

    pad_x = max(
        0,
        target_x - x
    )

    before_z = pad_z // 2
    after_z = (
        pad_z - before_z
    )

    before_y = pad_y // 2
    after_y = (
        pad_y - before_y
    )

    before_x = pad_x // 2
    after_x = (
        pad_x - before_x
    )

    padded = np.pad(
        array,
        (
            (before_z, after_z),
            (before_y, after_y),
            (before_x, after_x),
        ),
        mode="constant",
        constant_values=constant_value
    )

    return padded


# ============================================================
# HELPER
# ============================================================

def get_foreground_bbox(
    array
):
    """
    Locate non-zero foreground in an MRI.
    """

    foreground = (
        array > 0.02
    )

    coordinates = np.where(
        foreground
    )

    if coordinates[0].size == 0:

        return (
            0,
            array.shape[0],
            0,
            array.shape[1],
            0,
            array.shape[2],
        )

    return (
        int(coordinates[0].min()),
        int(coordinates[0].max()) + 1,

        int(coordinates[1].min()),
        int(coordinates[1].max()) + 1,

        int(coordinates[2].min()),
        int(coordinates[2].max()) + 1,
    )


# ============================================================
# HELPER
# ============================================================

def extract_center_patch(
    image,
    mask,
    patch_size
):
    """
    Extract a centered foreground-aware
    patch from an MRI and corresponding mask.
    """

    image = pad_to_patch(
        image,
        patch_size,
        constant_value=0
    )

    mask = pad_to_patch(
        mask,
        patch_size,
        constant_value=0
    )

    (
        z_min,
        z_max,
        y_min,
        y_max,
        x_min,
        x_max,
    ) = get_foreground_bbox(
        image
    )

    center_z = (
        z_min + z_max
    ) // 2

    center_y = (
        y_min + y_max
    ) // 2

    center_x = (
        x_min + x_max
    ) // 2

    pz, py, px = patch_size

    start_z = (
        center_z - pz // 2
    )

    start_y = (
        center_y - py // 2
    )

    start_x = (
        center_x - px // 2
    )

    start_z = max(
        0,
        min(
            start_z,
            image.shape[0] - pz
        )
    )

    start_y = max(
        0,
        min(
            start_y,
            image.shape[1] - py
        )
    )

    start_x = max(
        0,
        min(
            start_x,
            image.shape[2] - px
        )
    )

    end_z = (
        start_z + pz
    )

    end_y = (
        start_y + py
    )

    end_x = (
        start_x + px
    )

    image_patch = image[
        start_z:end_z,
        start_y:end_y,
        start_x:end_x
    ]

    mask_patch = mask[
        start_z:end_z,
        start_y:end_y,
        start_x:end_x
    ]

    return (
        image_patch,
        mask_patch
    )


# ============================================================
# DATASET
# ============================================================

class SpineSegmentationDataset(
    Dataset
):

    def __init__(
        self,
        pairs,
        patch_size=(96, 96, 96)
    ):

        self.pairs = pairs

        self.patch_size = (
            patch_size
        )

    def __len__(
        self
    ):

        return len(
            self.pairs
        )

    def __getitem__(
        self,
        index
    ):

        mri_path, mask_path = (
            self.pairs[index]
        )

        # ----------------------------------------------------
        # Load MRI
        # ----------------------------------------------------

        mri_image = sitk.ReadImage(
            str(mri_path)
        )

        mri = sitk.GetArrayFromImage(
            mri_image
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        mask_image = sitk.ReadImage(
            str(mask_path)
        )

        mask = sitk.GetArrayFromImage(
            mask_image
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Geometry validation
        # ----------------------------------------------------

        if mri.shape != mask.shape:

            raise RuntimeError(
                "MRI/mask shape mismatch: "
                f"{mri_path.name}"
            )

        # ----------------------------------------------------
        # Mask validation
        # ----------------------------------------------------

        labels = set(
            np.unique(mask).tolist()
        )

        expected_labels = {
            0,
            1,
            2,
            3
        }

        invalid_labels = (
            labels
            -
            EXPECTED_LABELS
        )

        if invalid_labels:

            raise RuntimeError(
                "Invalid segmentation labels "
                f"in {mask_path.name}: "
                f"{sorted(invalid_labels)}"
            )

        # ----------------------------------------------------
        # Patch extraction
        # ----------------------------------------------------

        image_patch, mask_patch = (
            extract_center_patch(
                mri,
                mask,
                self.patch_size
            )
        )

        # ----------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image_patch.copy()
        ).float()

        mask_tensor = torch.from_numpy(
            mask_patch.copy()
        ).long()

        # ----------------------------------------------------
        # Add channel dimension
        #
        # [Z,Y,X]
        #      ↓
        # [C,Z,Y,X]
        # ----------------------------------------------------

        image_tensor = (
            image_tensor
            .unsqueeze(0)
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "filename": mri_path.name,
        }


# ============================================================
# CREATE DATASET
# ============================================================

dataset = SpineSegmentationDataset(
    pairs=pairs,
    patch_size=PATCH_SIZE
)


print("\n" + "=" * 75)
print("PYTORCH DATASET")
print("=" * 75)

print(
    "Dataset length:",
    len(dataset)
)


# ============================================================
# TEST FIRST SAMPLE
# ============================================================

print("\n" + "=" * 75)
print("FIRST SAMPLE VALIDATION")
print("=" * 75)


sample = dataset[0]

image = sample["image"]

mask = sample["mask"]

filename = sample["filename"]


print(
    "File:",
    filename
)

print(
    "Image shape:",
    tuple(image.shape)
)

print(
    "Mask shape:",
    tuple(mask.shape)
)

print(
    "Image dtype:",
    image.dtype
)

print(
    "Mask dtype:",
    mask.dtype
)

print(
    "Image minimum:",
    float(image.min())
)

print(
    "Image maximum:",
    float(image.max())
)

print(
    "Mask labels:",
    torch.unique(
        mask
    ).tolist()
)


# ============================================================
# SHAPE VALIDATION
# ============================================================

expected_image_shape = (
    1,
    PATCH_SIZE[0],
    PATCH_SIZE[1],
    PATCH_SIZE[2],
)

expected_mask_shape = (
    PATCH_SIZE[0],
    PATCH_SIZE[1],
    PATCH_SIZE[2],
)


if tuple(image.shape) != (
    expected_image_shape
):

    raise RuntimeError(
        "Unexpected MRI tensor shape: "
        f"{tuple(image.shape)}"
    )


if tuple(mask.shape) != (
    expected_mask_shape
):

    raise RuntimeError(
        "Unexpected mask tensor shape: "
        f"{tuple(mask.shape)}"
    )


print(
    "✓ MRI tensor shape is correct."
)

print(
    "✓ Mask tensor shape is correct."
)


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


print("\n" + "=" * 75)
print("DATALOADER")
print("=" * 75)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Number of batches:",
    len(loader)
)


# ============================================================
# FIRST BATCH
# ============================================================

print("\n" + "=" * 75)
print("FIRST BATCH VALIDATION")
print("=" * 75)


batch = next(
    iter(loader)
)


batch_image = batch[
    "image"
]

batch_mask = batch[
    "mask"
]


print(
    "Batch image shape:",
    tuple(
        batch_image.shape
    )
)

print(
    "Batch mask shape:",
    tuple(
        batch_mask.shape
    )
)

print(
    "Batch image dtype:",
    batch_image.dtype
)

print(
    "Batch mask dtype:",
    batch_mask.dtype
)

print(
    "Batch image min:",
    float(
        batch_image.min()
    )
)

print(
    "Batch image max:",
    float(
        batch_image.max()
    )
)

print(
    "Batch mask labels:",
    torch.unique(
        batch_mask
    ).tolist()
)


# ============================================================
# EXPECTED BATCH SHAPES
# ============================================================

expected_batch_image = (
    BATCH_SIZE,
    1,
    PATCH_SIZE[0],
    PATCH_SIZE[1],
    PATCH_SIZE[2],
)

expected_batch_mask = (
    BATCH_SIZE,
    PATCH_SIZE[0],
    PATCH_SIZE[1],
    PATCH_SIZE[2],
)


if tuple(
    batch_image.shape
) != expected_batch_image:

    raise RuntimeError(
        "Unexpected batch image shape."
    )


if tuple(
    batch_mask.shape
) != expected_batch_mask:

    raise RuntimeError(
        "Unexpected batch mask shape."
    )


print(
    "✓ Batch image shape verified."
)

print(
    "✓ Batch mask shape verified."
)


# ============================================================
# CLASS DISTRIBUTION IN FIRST BATCH
# ============================================================

print("\n" + "=" * 75)
print("FIRST BATCH SEGMENTATION DISTRIBUTION")
print("=" * 75)


unique_labels, counts = torch.unique(
    batch_mask,
    return_counts=True
)


for label, count in zip(
    unique_labels.tolist(),
    counts.tolist()
):

    percentage = (
        100.0
        *
        count
        /
        batch_mask.numel()
    )

    class_name = {
        0: "Background",
        1: "Vertebrae",
        2: "Spinal Canal",
        3: "Intervertebral Disc",
    }.get(
        label,
        "Unknown"
    )

    print(
        f"Class {label} "
        f"({class_name}): "
        f"{count:,} voxels "
        f"({percentage:.4f}%)"
    )


# ============================================================
# ITERATE THROUGH ENTIRE DATASET
# ============================================================

print("\n" + "=" * 75)
print("FULL DATALOADER VALIDATION")
print("=" * 75)


successful_batches = 0

failed_batches = []


for batch_index, batch_data in enumerate(
    loader
):

    try:

        images = batch_data[
            "image"
        ]

        masks = batch_data[
            "mask"
        ]

        if images.ndim != 5:

            raise RuntimeError(
                "MRI batch must be 5D."
            )

        if masks.ndim != 4:

            raise RuntimeError(
                "Mask batch must be 4D."
            )

        if not torch.isfinite(
            images
        ).all():

            raise RuntimeError(
                "MRI tensor contains "
                "NaN or Inf."
            )

        labels = set(
            torch.unique(
                masks
            ).tolist()
        )

        if not labels.issubset(
    EXPECTED_LABELS
):

            raise RuntimeError(
                "Invalid mask labels."
            )

        successful_batches += 1

    except Exception as exc:

        failed_batches.append(
            {
                "batch": batch_index,
                "error": str(exc)
            }
        )


print(
    "Successful batches:",
    successful_batches
)

print(
    "Failed batches:",
    len(failed_batches)
)


if failed_batches:

    print("\nFAILED BATCHES")

    for failure in failed_batches[:20]:

        print(
            failure
        )

    raise RuntimeError(
        "Full DataLoader validation failed."
    )


# ============================================================
# GPU TENSOR TEST
# ============================================================

print("\n" + "=" * 75)
print("GPU TENSOR VALIDATION")
print("=" * 75)


if torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

    gpu_images = (
        batch_image
        .to(
            device,
            non_blocking=True
        )
    )

    gpu_masks = (
        batch_mask
        .to(
            device,
            non_blocking=True
        )
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )

    print(
        "GPU image shape:",
        tuple(
            gpu_images.shape
        )
    )

    print(
        "GPU mask shape:",
        tuple(
            gpu_masks.shape
        )
    )

    print(
        "GPU image dtype:",
        gpu_images.dtype
    )

    print(
        "GPU mask dtype:",
        gpu_masks.dtype
    )

    print(
        "GPU memory allocated:",
        round(
            torch.cuda.memory_allocated()
            /
            1024**2,
            2
        ),
        "MB"
    )

    del gpu_images
    del gpu_masks

    torch.cuda.empty_cache()

    print(
        "✓ GPU tensor transfer successful."
    )

else:

    print(
        "CUDA unavailable."
    )

    print(
        "GPU validation skipped."
    )


# ============================================================
# SAVE DATASET MANIFEST
# ============================================================

manifest_records = []

for mri_path, mask_path in pairs:

    manifest_records.append(
        {
            "mri_file":
                mri_path.name,

            "mask_file":
                mask_path.name,

            "patch_z":
                PATCH_SIZE[0],

            "patch_y":
                PATCH_SIZE[1],

            "patch_x":
                PATCH_SIZE[2],

            "classes":
                "0,1,2,3",
        }
    )


manifest_df = pd.DataFrame(
    manifest_records
)


manifest_path = (
    OUTPUT_DIR
    /
    "segmentation_dataset_manifest.csv"
)


manifest_df.to_csv(
    manifest_path,
    index=False
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config_path = (
    OUTPUT_DIR
    /
    "dataloader_configuration.txt"
)


config_path.write_text(
    (
        "PHASE 3 - PART 6\n"
        "Swin-UNETR Dataset Configuration\n\n"
        f"Dataset size: {len(dataset)}\n"
        f"Patch size: {PATCH_SIZE}\n"
        f"Batch size: {BATCH_SIZE}\n"
        f"Workers: {NUM_WORKERS}\n"
        f"Classes: 4\n"
        "Class 0: Background\n"
        "Class 1: Vertebrae\n"
        "Class 2: Spinal Canal\n"
        "Class 3: Intervertebral Disc\n"
    ),
    encoding="utf-8"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print(
    "Dataset manifest:",
    manifest_path
)

print(
    "DataLoader configuration:",
    config_path
)


print("\n" + "=" * 75)
print("PHASE 3 - PART 6 COMPLETE")
print("=" * 75)