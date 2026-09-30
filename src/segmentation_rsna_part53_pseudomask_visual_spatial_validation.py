"""
PART 53 — PSEUDO-MASK VISUAL & SPATIAL VALIDATION

Purpose
-------
Part 52 demonstrated that pseudo-mask density is highly sensitive to
dilation radius. Part 53 performs the next required validation step:
visual/spatial inspection of the same pseudo-masks before any retraining.

For representative RSNA cases, this script creates PNG overlays for:

    R0 = current Part 11 pseudo-mask
    R1 = 1-voxel 3D 6-connected dilation
    R2 = 2-voxel dilation
    R3 = 3-voxel dilation

The image is the same foreground-centered 32x64x64 crop used in
Parts 42-52. The slice selected for visualization is the slice with
the greatest foreground occupancy in the original R0 mask.

Each output figure contains:
    - MRI grayscale slice
    - R0, R1, R2, R3 mask contours
    - per-radius foreground counts
    - per-class counts

This is a VISUAL/SPATIAL diagnostic only.

NO:
    - model training
    - optimizer
    - checkpoint modification
    - SPIDER
    - test set
    - Part 15 modification

The script does not overwrite existing pseudo-masks.

Scientific caution
------------------
Dilation is a sensitivity analysis. A larger region is NOT assumed to
be medically correct ground truth. Visual plausibility is evaluated
relative to the RSNA point-derived target and surrounding MRI anatomy.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ================================================================
# CONFIGURATION
# ================================================================

ROOT = Path(
    r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"
)

PART11_PATH = (
    ROOT
    / "src"
    / "segmentation_rsna_part11_controlled_pilot_training.py"
)

PART9_PATH = (
    ROOT
    / "src"
    / "segmentation_rsna_part9_3d_dataset_loader.py"
)

P15 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

TRCSV = P15 / "part15_train_cohort.csv"
VACSV = P15 / "part15_validation_cohort.csv"

OUT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part53_pseudomask_visual_spatial_validation"
)

FIG_DIR = OUT / "figures"
REPORT_DIR = OUT / "reports"

TRAIN_N = 100
VAL_N = 50

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)

NUM_CLASSES = 6

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

# Representative cases are deliberately spread through each cohort.
TRAIN_CASES = [1, 25, 50, 75, 100]
VAL_CASES = [1, 25, 50]

RADII = [0, 1, 2, 3]

# Conservative 6-connected dilation, matching Part 52.
DILATION_CONNECTIVITY = "3D 6-connected"


# ================================================================
# REPRODUCIBILITY
# ================================================================

def seed_everything(seed: int = 153):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ================================================================
# MODULE LOADING
# ================================================================

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


# ================================================================
# PATH VALIDATION
# ================================================================

def validate_paths():
    print("=" * 82)
    print("PART 53 PATH VALIDATION")
    print("=" * 82)

    paths = [
        ("Project root", ROOT),
        ("Part 11", PART11_PATH),
        ("Part 9", PART9_PATH),
        ("Part 15 train cohort", TRCSV),
        ("Part 15 validation cohort", VACSV),
    ]

    for label, path in paths:
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{label:<40}: {status}")

        if not path.exists():
            raise FileNotFoundError(path)


# ================================================================
# CROP HELPERS
# ================================================================

def clamp_start(center: int, crop: int, total: int) -> int:
    return max(
        0,
        min(
            center - crop // 2,
            total - crop,
        ),
    )


def foreground_centered_crop(
    image: np.ndarray,
    mask: np.ndarray,
):
    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        center = [
            total // 2
            for total in image.shape
        ]
    else:
        center = np.round(
            coords.mean(axis=0)
        ).astype(int)

    starts = [
        clamp_start(
            int(center[i]),
            CROP_SHAPE[i],
            image.shape[i],
        )
        for i in range(3)
    ]

    z, y, x = starts
    dz, dy, dx = CROP_SHAPE

    return (
        image[
            z:z + dz,
            y:y + dy,
            x:x + dx,
        ].astype(np.float32),
        mask[
            z:z + dz,
            y:y + dy,
            x:x + dx,
        ].astype(np.int64),
    )


# ================================================================
# DILATION
# ================================================================

def dilate_binary_6_connected(
    binary: np.ndarray,
    iterations: int,
):
    result = binary.astype(bool).copy()

    for _ in range(iterations):
        expanded = result.copy()

        expanded[1:, :, :] |= result[:-1, :, :]
        expanded[:-1, :, :] |= result[1:, :, :]

        expanded[:, 1:, :] |= result[:, :-1, :]
        expanded[:, :-1, :] |= result[:, 1:, :]

        expanded[:, :, 1:] |= result[:, :, :-1]
        expanded[:, :, :-1] |= result[:, :, 1:]

        result = expanded

    return result


def dilate_multiclass_mask(
    mask: np.ndarray,
    radius: int,
):
    if radius == 0:
        return mask.copy()

    candidates = []

    for class_id in range(1, NUM_CLASSES):
        original = mask == class_id

        if not original.any():
            continue

        dilated = dilate_binary_6_connected(
            original,
            radius,
        )

        candidates.append(
            (
                int(original.sum()),
                class_id,
                dilated,
            )
        )

    # Deterministic overlap handling: larger original classes first,
    # then class id.
    candidates.sort(
        key=lambda x: (-x[0], x[1])
    )

    result = np.zeros_like(
        mask,
        dtype=np.int64,
    )

    occupied = np.zeros_like(
        mask,
        dtype=bool,
    )

    for _, class_id, region in candidates:
        assignable = region & (~occupied)

        result[assignable] = class_id
        occupied |= assignable

    return result


# ================================================================
# CASE LOADING
# ================================================================

def load_case(
    part11,
    part9,
    row: pd.Series,
):
    loaded = part11.load_tensor_case(
        row,
        part9,
    )

    image = loaded[0]
    mask = loaded[1]

    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    image = np.asarray(image)
    mask = np.asarray(mask)

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]

    image = image.astype(np.float32)
    mask = mask.astype(np.int64)

    if tuple(image.shape) != FULL_SHAPE:
        raise RuntimeError(
            f"Unexpected image shape: {image.shape}"
        )

    if tuple(mask.shape) != FULL_SHAPE:
        raise RuntimeError(
            f"Unexpected mask shape: {mask.shape}"
        )

    return foreground_centered_crop(
        image,
        mask,
    )


# ================================================================
# IMAGE NORMALIZATION
# ================================================================

def normalize_slice(image_slice: np.ndarray):
    values = image_slice.astype(np.float32)

    finite = np.isfinite(values)

    if not finite.any():
        return np.zeros_like(values)

    lo = np.percentile(
        values[finite],
        1,
    )

    hi = np.percentile(
        values[finite],
        99,
    )

    if hi <= lo:
        return np.zeros_like(values)

    return np.clip(
        (values - lo) / (hi - lo),
        0,
        1,
    )


# ================================================================
# SLICE SELECTION
# ================================================================

def choose_visual_slice(mask: np.ndarray):
    occupancy = (
        (mask > 0)
        .sum(axis=(1, 2))
    )

    maximum = occupancy.max()

    candidates = np.where(
        occupancy == maximum
    )[0]

    if len(candidates) == 0:
        return mask.shape[0] // 2

    return int(
        candidates[len(candidates) // 2]
    )


# ================================================================
# PER-CLASS COUNTS
# ================================================================

def class_counts(mask: np.ndarray):
    return {
        str(class_id): int(
            (mask == class_id).sum()
        )
        for class_id in range(NUM_CLASSES)
    }


# ================================================================
# FIGURE CREATION
# ================================================================

def save_case_figure(
    image: np.ndarray,
    base_mask: np.ndarray,
    split: str,
    case_number: int,
):
    masks = {
        radius: dilate_multiclass_mask(
            base_mask,
            radius,
        )
        for radius in RADII
    }

    slice_index = choose_visual_slice(
        base_mask
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 11),
    )

    axes = axes.ravel()

    for ax, radius in zip(
        axes,
        RADII,
    ):
        img = normalize_slice(
            image[slice_index]
        )

        ax.imshow(
            img,
            cmap="gray",
        )

        current = masks[radius][slice_index]

        # One contour per class. Contours are intentionally drawn
        # without filling so MRI anatomy remains visible.
        for class_id in range(
            1,
            NUM_CLASSES,
        ):
            binary = (
                current == class_id
            )

            if binary.any():
                ax.contour(
                    binary.astype(float),
                    levels=[0.5],
                    linewidths=1.1,
                )

        fg = int(
            (current > 0).sum()
        )

        per_class = [
            f"C{cid}:{int((current == cid).sum())}"
            for cid in range(1, NUM_CLASSES)
            if (current == cid).any()
        ]

        ax.set_title(
            f"Radius {radius} | "
            f"slice {slice_index}/{image.shape[0]-1}\n"
            f"FG={fg} | "
            + " ".join(per_class),
            fontsize=10,
        )

        ax.axis("off")

    fig.suptitle(
        f"Part 53 — {split.upper()} Case {case_number} | "
        f"Pseudo-mask spatial sensitivity",
        fontsize=14,
    )

    fig.text(
        0.5,
        0.015,
        "Contours: foreground pseudo-mask classes 1–5 | "
        "Same MRI crop and same original R0 target across radii",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.035, 1, 0.95)
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{split.lower()}_case_"
        f"{case_number:03d}_"
        f"slice_{slice_index:02d}.png"
    )

    path = FIG_DIR / filename

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return (
        path,
        slice_index,
        masks,
    )


# ================================================================
# NUMERICAL SPATIAL REPORT
# ================================================================

def case_report(
    split: str,
    case_number: int,
    image: np.ndarray,
    base_mask: np.ndarray,
    masks,
    slice_index: int,
):
    result = {
        "split": split,
        "case_number": case_number,
        "image_shape": list(image.shape),
        "crop_shape": list(base_mask.shape),
        "visual_slice": slice_index,
        "radii": {},
    }

    for radius, mask in masks.items():
        fg = mask > 0

        active_z = int(
            fg.any(axis=(1, 2)).sum()
        )

        slice_fg = int(
            fg[slice_index].sum()
        )

        result["radii"][str(radius)] = {
            "foreground_voxels": int(
                fg.sum()
            ),
            "foreground_fraction": float(
                fg.mean()
            ),
            "active_z_slices": active_z,
            "active_z_fraction": float(
                active_z / mask.shape[0]
            ),
            "visual_slice_foreground": slice_fg,
            "class_counts": class_counts(mask),
        }

    return result


# ================================================================
# MAIN
# ================================================================

def main():
    seed_everything()

    validate_paths()

    print()
    print("=" * 82)
    print("PART 53 — PSEUDO-MASK VISUAL & SPATIAL VALIDATION")
    print("=" * 82)

    print(f"Train cohort : {TRAIN_N}")
    print(f"Validation cohort : {VAL_N}")
    print(f"Full shape : {FULL_SHAPE}")
    print(f"Crop shape : {CROP_SHAPE}")
    print(f"Representative train cases : {TRAIN_CASES}")
    print(f"Representative validation cases : {VAL_CASES}")
    print(f"Radii : {RADII}")
    print(
        f"Dilation : {DILATION_CONNECTIVITY}"
    )
    print("Training : NO")
    print("Optimizer : NO")
    print("SPIDER : NO")
    print("Test set : NO")
    print("Part 15 modified : NO")

    part11 = load_module(
        PART11_PATH,
        "segmentation_rsna_part11_part53_runtime",
    )

    part9 = load_module(
        PART9_PATH,
        "segmentation_rsna_part9_part53_runtime",
    )

    train_df = pd.read_csv(
        TRCSV
    ).head(TRAIN_N)

    val_df = pd.read_csv(
        VACSV
    ).head(VAL_N)

    # ------------------------------------------------------------
    # SMOKE TEST
    # ------------------------------------------------------------

    print()
    print("=" * 82)
    print("PART 53 SHAPE / LABEL SMOKE TEST")
    print("=" * 82)

    smoke_image, smoke_mask = load_case(
        part11,
        part9,
        train_df.iloc[0],
    )

    print(
        f"Image : {smoke_image.shape}"
    )

    print(
        f"Mask : {smoke_mask.shape}"
    )

    print(
        f"Labels : "
        f"{sorted(np.unique(smoke_mask).tolist())}"
    )

    assert tuple(smoke_image.shape) == CROP_SHAPE
    assert tuple(smoke_mask.shape) == CROP_SHAPE
    assert smoke_mask.min() >= 0
    assert smoke_mask.max() < NUM_CLASSES

    print("✓ Shape / label smoke test PASSED.")

    # ------------------------------------------------------------
    # REPRESENTATIVE CASES
    # ------------------------------------------------------------

    all_reports = []

    cases_to_process = [
        (
            "train",
            train_df,
            TRAIN_CASES,
        ),
        (
            "validation",
            val_df,
            VAL_CASES,
        ),
    ]

    for split, dataframe, case_numbers in cases_to_process:

        print()
        print("=" * 82)
        print(
            f"PART 53 {split.upper()} VISUAL VALIDATION"
        )
        print("=" * 82)

        for case_number in case_numbers:

            row = dataframe.iloc[
                case_number - 1
            ]

            print()
            print(
                f"{split.upper()} CASE {case_number}"
            )

            image, mask = load_case(
                part11,
                part9,
                row,
            )

            print(
                f"R0 foreground : "
                f"{int((mask > 0).sum())}"
            )

            figure_path, slice_index, masks = (
                save_case_figure(
                    image,
                    mask,
                    split,
                    case_number,
                )
            )

            report = case_report(
                split,
                case_number,
                image,
                mask,
                masks,
                slice_index,
            )

            all_reports.append(
                report
            )

            print(
                f"Visual slice : {slice_index}"
            )

            for radius in RADII:
                stats = report[
                    "radii"
                ][str(radius)]

                print(
                    f"  R{radius}: "
                    f"FG={stats['foreground_voxels']} "
                    f"sliceFG={stats['visual_slice_foreground']} "
                    f"activeZ={stats['active_z_fraction']:.3f}"
                )

            print(
                f"Saved figure : {figure_path}"
            )

    # ------------------------------------------------------------
    # SAVE JSON REPORT
    # ------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "part": 53,
        "purpose": (
            "Visual and spatial validation of pseudo-mask "
            "dilation sensitivity before retraining."
        ),
        "configuration": {
            "train_cases": TRAIN_CASES,
            "validation_cases": VAL_CASES,
            "full_shape": FULL_SHAPE,
            "crop_shape": CROP_SHAPE,
            "radii": RADII,
            "dilation_connectivity": DILATION_CONNECTIVITY,
            "crop_policy": "foreground-centered",
        },
        "method": {
            "training": False,
            "optimizer": False,
            "spider": False,
            "test_set": False,
            "part15_modified": False,
            "production_masks_modified": False,
        },
        "case_reports": all_reports,
        "figure_directory": str(FIG_DIR),
        "scientific_note": (
            "The visualizations test spatial plausibility and "
            "sensitivity of synthetic dilation around the current "
            "RSNA point-derived pseudo-mask. They do not establish "
            "medical ground-truth segmentation accuracy."
        ),
    }

    summary_path = (
        REPORT_DIR
        / "part53_visual_spatial_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 82)
    print("PART 53 COMPLETE")
    print("=" * 82)

    print(
        f"Figures : {FIG_DIR}"
    )

    print(
        f"Summary : {summary_path}"
    )

    print()
    print(
        "NEXT DECISION:"
    )

    print(
        "Inspect R0/R1/R2/R3 overlays before selecting "
        "any pseudo-mask radius for training."
    )


if __name__ == "__main__":
    main()
