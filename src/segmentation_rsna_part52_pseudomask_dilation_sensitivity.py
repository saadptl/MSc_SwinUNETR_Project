"""
PART 52 — PSEUDO-MASK DILATION SENSITIVITY ANALYSIS

Purpose
-------
Part 51 showed that the current pseudo-masks are highly sparse.
Part 52 therefore tests whether target sparsity is strongly dependent
on the radius used to construct/expand the point-based pseudo-masks.

NO MODEL TRAINING IS PERFORMED.

This script starts from the pseudo-mask already produced by the
current Part 11 pipeline and applies controlled 3-D binary dilation
to each foreground class independently.

It compares:
    radius 0 : current pseudo-mask
    radius 1 : one-voxel 6-connected dilation
    radius 2 : two iterations
    radius 3 : three iterations

Important:
    This is a SENSITIVITY ANALYSIS of the existing pseudo-mask.
    It does NOT claim that a larger mask is medically correct.
    It does NOT modify the production masks or Part 15.

For every radius it measures:
    - total foreground voxels
    - foreground occupancy
    - per-class voxel counts
    - class presence
    - connected components
    - largest component
    - small-component fraction
    - active Z-slice fraction

The exact first 100 Part-15 training cases and first 50 validation
cases are used, with the same foreground-centered 32x64x64 crop.

No SPIDER, no test set, no optimizer, no checkpoint changes.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

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
    ROOT / "src" / "segmentation_rsna_part11_controlled_pilot_training.py"
)

PART9_PATH = (
    ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py"
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
    / "rsna_part52_pseudomask_dilation_sensitivity"
)

REPORT = OUT / "reports"
CSV_OUT = OUT / "csv"

SEED = 152

TRAIN_N = 100
VAL_N = 50

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)

NUM_CLASSES = 6

CLASS_NAMES = [
    "Background",
    "Spinal_Canal_Stenosis",
    "Left_Neural_Foraminal_Narrowing",
    "Right_Neural_Foraminal_Narrowing",
    "Left_Subarticular_Stenosis",
    "Right_Subarticular_Stenosis",
]

# Radius is implemented as repeated 6-connected dilation iterations.
RADII = [0, 1, 2, 3]

SMALL_COMPONENT_THRESHOLD = 5


# ================================================================
# REPRODUCIBILITY
# ================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

def validate_paths() -> None:
    print("=" * 82)
    print("PART 52 PATH VALIDATION")
    print("=" * 82)

    checks = [
        ("Project root", ROOT),
        ("Part 11", PART11_PATH),
        ("Part 9", PART9_PATH),
        ("Part 15 train cohort", TRCSV),
        ("Part 15 validation cohort", VACSV),
    ]

    for label, path in checks:
        status = "FOUND" if path.exists() else "MISSING"

        print(
            f"{label:<40}: {status}"
        )

        if not path.exists():
            raise FileNotFoundError(path)


# ================================================================
# FOREGROUND-CENTERED CROP
# ================================================================

def clamp_start(
    center: int,
    crop: int,
    total: int,
) -> int:
    start = center - crop // 2

    return max(
        0,
        min(
            start,
            total - crop,
        ),
    )


def foreground_centered_crop(
    image: np.ndarray,
    mask: np.ndarray,
):
    foreground = np.argwhere(mask > 0)

    if foreground.size == 0:
        starts = [
            max(
                0,
                (total - crop) // 2,
            )
            for total, crop in zip(
                image.shape,
                CROP_SHAPE,
            )
        ]
    else:
        center = np.round(
            foreground.mean(axis=0)
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
# LOAD DATA
# ================================================================

def load_masks(
    part11,
    part9,
    csv_path: Path,
    n: int,
    label: str,
):
    df = pd.read_csv(csv_path).head(n).copy()

    if len(df) != n:
        raise RuntimeError(
            f"{label}: expected {n} rows, found {len(df)}"
        )

    masks = []

    print()
    print("=" * 82)
    print(f"PART 52 {label.upper()} DATA PRELOAD")
    print("-" * 82)

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        loaded = part11.load_tensor_case(
            row,
            part9,
        )

        image = loaded[0]
        mask = loaded[1]

        if torch.is_tensor(image):
            image = (
                image.detach()
                .cpu()
                .numpy()
            )

        if torch.is_tensor(mask):
            mask = (
                mask.detach()
                .cpu()
                .numpy()
            )

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
                f"{label} case {i}: image={image.shape}"
            )

        if tuple(mask.shape) != FULL_SHAPE:
            raise RuntimeError(
                f"{label} case {i}: mask={mask.shape}"
            )

        _, cropped_mask = foreground_centered_crop(
            image,
            mask,
        )

        masks.append(cropped_mask)

        if (
            i == 1
            or i == n
            or i % 25 == 0
        ):
            print(
                f"{label.upper()} "
                f"{i:03d}/{n} "
                f"FG={(cropped_mask > 0).sum()}"
            )

    return masks


# ================================================================
# 6-CONNECTED DILATION
# ================================================================

def dilate_binary_6_connected(
    binary: np.ndarray,
    iterations: int,
):
    """
    Repeated 6-connected binary dilation.

    Each iteration expands a component by one voxel in:
        +/- Z, +/- Y, +/- X

    This is intentionally explicit and dependency-free.
    """

    result = np.asarray(
        binary,
        dtype=bool,
    ).copy()

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


# ================================================================
# CONNECTED COMPONENTS
# ================================================================

def connected_component_sizes_3d(
    binary: np.ndarray,
):
    binary = np.asarray(
        binary,
        dtype=bool,
    )

    if not binary.any():
        return []

    visited = np.zeros(
        binary.shape,
        dtype=bool,
    )

    depth, height, width = binary.shape

    components = []

    for z in range(depth):
        for y in range(height):
            for x in range(width):

                if not binary[z, y, x]:
                    continue

                if visited[z, y, x]:
                    continue

                stack = [(z, y, x)]
                visited[z, y, x] = True
                size = 0

                while stack:
                    cz, cy, cx = stack.pop()
                    size += 1

                    neighbors = (
                        (cz - 1, cy, cx),
                        (cz + 1, cy, cx),
                        (cz, cy - 1, cx),
                        (cz, cy + 1, cx),
                        (cz, cy, cx - 1),
                        (cz, cy, cx + 1),
                    )

                    for nz, ny, nx in neighbors:
                        if (
                            nz < 0
                            or nz >= depth
                            or ny < 0
                            or ny >= height
                            or nx < 0
                            or nx >= width
                        ):
                            continue

                        if not binary[
                            nz,
                            ny,
                            nx,
                        ]:
                            continue

                        if visited[
                            nz,
                            ny,
                            nx,
                        ]:
                            continue

                        visited[
                            nz,
                            ny,
                            nx,
                        ] = True

                        stack.append(
                            (nz, ny, nx)
                        )

                components.append(size)

    return sorted(
        components,
        reverse=True,
    )


# ================================================================
# SINGLE CLASS STATISTICS
# ================================================================

def class_stats(
    mask: np.ndarray,
    class_id: int,
):
    binary = mask == class_id

    voxel_count = int(
        binary.sum()
    )

    if voxel_count == 0:
        return {
            "voxels": 0,
            "fraction": 0.0,
            "components": 0,
            "largest_component": 0,
            "median_component": 0.0,
            "small_component_fraction": 0.0,
            "active_z_fraction": 0.0,
            "bbox_z": 0,
            "bbox_y": 0,
            "bbox_x": 0,
        }

    components = (
        connected_component_sizes_3d(
            binary
        )
    )

    coordinates = np.argwhere(
        binary
    )

    lo = coordinates.min(axis=0)
    hi = coordinates.max(axis=0)

    bbox_z = int(
        hi[0] - lo[0] + 1
    )

    bbox_y = int(
        hi[1] - lo[1] + 1
    )

    bbox_x = int(
        hi[2] - lo[2] + 1
    )

    small_voxels = sum(
        value
        for value in components
        if value < SMALL_COMPONENT_THRESHOLD
    )

    active_z = int(
        binary.any(axis=(1, 2)).sum()
    )

    return {
        "voxels": voxel_count,
        "fraction": float(
            voxel_count / binary.size
        ),
        "components": len(components),
        "largest_component": int(
            components[0]
        ),
        "median_component": float(
            np.median(components)
        ),
        "small_component_fraction": float(
            small_voxels / voxel_count
        ),
        "active_z_fraction": float(
            active_z / binary.shape[0]
        ),
        "bbox_z": bbox_z,
        "bbox_y": bbox_y,
        "bbox_x": bbox_x,
    }


# ================================================================
# APPLY CLASS-WISE DILATION
# ================================================================

def make_dilated_mask(
    mask: np.ndarray,
    radius: int,
):
    if radius == 0:
        return mask.copy()

    output = np.zeros_like(
        mask,
        dtype=np.int64,
    )

    # Dilate each class independently.
    # If independently dilated regions touch/overlap, the class with
    # the larger original component is assigned first. This avoids
    # producing an invalid multi-label mask.
    class_components = []

    for class_id in range(
        1,
        NUM_CLASSES,
    ):
        binary = mask == class_id

        if not binary.any():
            continue

        dilated = dilate_binary_6_connected(
            binary,
            radius,
        )

        original_size = int(
            binary.sum()
        )

        class_components.append(
            (
                original_size,
                class_id,
                dilated,
            )
        )

    class_components.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    occupied = np.zeros_like(
        mask,
        dtype=bool,
    )

    for _, class_id, dilated in class_components:
        assignable = (
            dilated
            & (~occupied)
        )

        output[
            assignable
        ] = class_id

        occupied |= assignable

    return output


# ================================================================
# CASE ANALYSIS
# ================================================================

def analyze_case(
    mask: np.ndarray,
    split: str,
    case_index: int,
    radius: int,
):
    foreground = mask > 0

    foreground_voxels = int(
        foreground.sum()
    )

    components = (
        connected_component_sizes_3d(
            foreground
        )
    )

    active_z = int(
        foreground.any(
            axis=(1, 2)
        ).sum()
    )

    z_activity = foreground.any(
        axis=(1, 2)
    ).astype(np.int8)

    transitions = int(
        np.abs(
            np.diff(z_activity)
        ).sum()
    )

    record = {
        "split": split,
        "case_index": case_index,
        "radius": radius,
        "foreground_voxels": foreground_voxels,
        "foreground_fraction": float(
            foreground_voxels / mask.size
        ),
        "foreground_components": len(
            components
        ),
        "largest_foreground_component": (
            int(components[0])
            if components
            else 0
        ),
        "active_z_fraction": float(
            active_z / mask.shape[0]
        ),
        "z_transitions": transitions,
        "invalid_labels": int(
            (
                (mask < 0)
                | (mask >= NUM_CLASSES)
            ).sum()
        ),
    }

    for class_id in range(
        NUM_CLASSES
    ):
        stats = class_stats(
            mask,
            class_id,
        )

        for key, value in stats.items():
            record[
                f"class_{class_id}_{key}"
            ] = value

    return record


# ================================================================
# SUMMARY BY SPLIT AND RADIUS
# ================================================================

def mean_of(
    records,
    key,
):
    values = [
        record[key]
        for record in records
    ]

    return float(
        np.mean(values)
    )


def summarize(
    records,
    split,
    radius,
):
    subset = [
        record
        for record in records
        if (
            record["split"] == split
            and record["radius"] == radius
        )
    ]

    result = {
        "split": split,
        "radius": radius,
        "n_cases": len(subset),
        "mean_foreground_voxels": mean_of(
            subset,
            "foreground_voxels",
        ),
        "median_foreground_voxels": float(
            np.median(
                [
                    r["foreground_voxels"]
                    for r in subset
                ]
            )
        ),
        "mean_foreground_fraction": mean_of(
            subset,
            "foreground_fraction",
        ),
        "mean_foreground_components": mean_of(
            subset,
            "foreground_components",
        ),
        "mean_largest_foreground_component": mean_of(
            subset,
            "largest_foreground_component",
        ),
        "mean_active_z_fraction": mean_of(
            subset,
            "active_z_fraction",
        ),
        "mean_z_transitions": mean_of(
            subset,
            "z_transitions",
        ),
        "zero_foreground_cases": int(
            sum(
                r["foreground_voxels"] == 0
                for r in subset
            )
        ),
        "invalid_label_cases": int(
            sum(
                r["invalid_labels"] > 0
                for r in subset
            )
        ),
        "classes": {},
    }

    for class_id in range(
        NUM_CLASSES
    ):
        class_records = [
            r
            for r in subset
        ]

        voxel_values = [
            r[
                f"class_{class_id}_voxels"
            ]
            for r in class_records
        ]

        component_values = [
            r[
                f"class_{class_id}_components"
            ]
            for r in class_records
        ]

        largest_values = [
            r[
                f"class_{class_id}_largest_component"
            ]
            for r in class_records
        ]

        small_values = [
            r[
                f"class_{class_id}_small_component_fraction"
            ]
            for r in class_records
            if r[
                f"class_{class_id}_voxels"
            ] > 0
        ]

        active_values = [
            r[
                f"class_{class_id}_active_z_fraction"
            ]
            for r in class_records
            if r[
                f"class_{class_id}_voxels"
            ] > 0
        ]

        result["classes"][
            str(class_id)
        ] = {
            "name": CLASS_NAMES[class_id],
            "total_voxels": int(
                sum(voxel_values)
            ),
            "mean_voxels": float(
                np.mean(voxel_values)
            ),
            "median_voxels": float(
                np.median(voxel_values)
            ),
            "cases_present": int(
                sum(
                    value > 0
                    for value in voxel_values
                )
            ),
            "mean_components": float(
                np.mean(component_values)
            ),
            "mean_largest_component": float(
                np.mean(largest_values)
            ),
            "mean_small_component_fraction": (
                float(np.mean(small_values))
                if small_values
                else 0.0
            ),
            "mean_active_z_fraction": (
                float(np.mean(active_values))
                if active_values
                else 0.0
            ),
        }

    return result


# ================================================================
# MAIN
# ================================================================

def main():
    seed_everything(SEED)

    validate_paths()

    print()
    print("=" * 82)
    print(
        "PART 52 — PSEUDO-MASK DILATION "
        "SENSITIVITY ANALYSIS"
    )
    print("=" * 82)

    print(f"Train subset : {TRAIN_N}")
    print(f"Validation subset : {VAL_N}")
    print(f"Full volume : {FULL_SHAPE}")
    print(f"Crop : {CROP_SHAPE}")
    print(f"Tested radii : {RADII}")
    print(
        "Dilation connectivity : "
        "3-D 6-connected"
    )
    print(
        f"Small component threshold : "
        f"< {SMALL_COMPONENT_THRESHOLD} voxels"
    )
    print("Training performed : NO")
    print("Optimizer used : NO")
    print("SPIDER used : NO")
    print("Test set used : NO")
    print("Part 15 overwritten : NO")

    part11 = load_module(
        PART11_PATH,
        "segmentation_rsna_part11_part52_runtime",
    )

    part9 = load_module(
        PART9_PATH,
        "segmentation_rsna_part9_part52_runtime",
    )

    train_masks = load_masks(
        part11,
        part9,
        TRCSV,
        TRAIN_N,
        "train",
    )

    val_masks = load_masks(
        part11,
        part9,
        VACSV,
        VAL_N,
        "validation",
    )

    print()
    print("=" * 82)
    print("PART 52 SHAPE / LABEL SMOKE TEST")
    print("-" * 82)

    for i, mask in enumerate(
        train_masks[:3],
        start=1,
    ):
        print(
            f"Case {i}: "
            f"mask={mask.shape} "
            f"FG={(mask > 0).sum()} "
            f"labels={sorted(np.unique(mask).tolist())}"
        )

        assert tuple(mask.shape) == CROP_SHAPE
        assert mask.min() >= 0
        assert mask.max() < NUM_CLASSES

    print("✓ Shape / label smoke test PASSED.")

    # ------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------

    all_records = []

    for radius in RADII:

        print()
        print("=" * 82)
        print(
            f"PART 52 RADIUS {radius} ANALYSIS"
        )
        print("=" * 82)

        for split, masks in (
            ("train", train_masks),
            ("validation", val_masks),
        ):
            print(
                f"\n{split.upper()} radius={radius}"
            )

            for case_index, original_mask in enumerate(
                masks,
                start=1,
            ):
                new_mask = make_dilated_mask(
                    original_mask,
                    radius,
                )

                record = analyze_case(
                    new_mask,
                    split,
                    case_index,
                    radius,
                )

                all_records.append(
                    record
                )

                if (
                    case_index == 1
                    or case_index == len(masks)
                    or case_index % 25 == 0
                ):
                    print(
                        f"{split.upper()} "
                        f"{case_index:03d}/"
                        f"{len(masks)} "
                        f"FG="
                        f"{record['foreground_voxels']} "
                        f"components="
                        f"{record['foreground_components']} "
                        f"largest="
                        f"{record['largest_foreground_component']} "
                        f"activeZ="
                        f"{record['active_z_fraction']:.3f}"
                    )

    # ------------------------------------------------------------
    # SAVE CASE CSV
    # ------------------------------------------------------------

    CSV_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = []

    for record in all_records:
        for key in record.keys():
            if key not in fields:
                fields.append(key)

    case_csv = (
        CSV_OUT
        / "part52_dilation_case_geometry.csv"
    )

    with open(
        case_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(all_records)

    # ------------------------------------------------------------
    # SUMMARIES
    # ------------------------------------------------------------

    summaries = []

    for split in (
        "train",
        "validation",
    ):
        for radius in RADII:
            summaries.append(
                summarize(
                    all_records,
                    split,
                    radius,
                )
            )

    # ------------------------------------------------------------
    # CONSOLE COMPARISON
    # ------------------------------------------------------------

    print()
    print("=" * 82)
    print("PART 52 TRAIN RADIUS COMPARISON")
    print("=" * 82)

    for radius in RADII:
        s = next(
            x
            for x in summaries
            if (
                x["split"] == "train"
                and x["radius"] == radius
            )
        )

        print(
            f"Radius {radius}: "
            f"mean FG={s['mean_foreground_voxels']:.2f}, "
            f"occupancy={s['mean_foreground_fraction']:.6f}, "
            f"components={s['mean_foreground_components']:.2f}, "
            f"largest={s['mean_largest_foreground_component']:.2f}, "
            f"activeZ={s['mean_active_z_fraction']:.4f}"
        )

    print()
    print("=" * 82)
    print("PART 52 VALIDATION RADIUS COMPARISON")
    print("=" * 82)

    for radius in RADII:
        s = next(
            x
            for x in summaries
            if (
                x["split"] == "validation"
                and x["radius"] == radius
            )
        )

        print(
            f"Radius {radius}: "
            f"mean FG={s['mean_foreground_voxels']:.2f}, "
            f"occupancy={s['mean_foreground_fraction']:.6f}, "
            f"components={s['mean_foreground_components']:.2f}, "
            f"largest={s['mean_largest_foreground_component']:.2f}, "
            f"activeZ={s['mean_active_z_fraction']:.4f}"
        )

    # ------------------------------------------------------------
    # PER-CLASS SUMMARY
    # ------------------------------------------------------------

    print()
    print("=" * 82)
    print("PART 52 PER-CLASS RADIUS RESPONSE")
    print("=" * 82)

    for class_id in range(
        1,
        NUM_CLASSES,
    ):
        print()
        print(
            f"Class {class_id} — "
            f"{CLASS_NAMES[class_id]}"
        )

        for radius in RADII:
            train_s = next(
                x
                for x in summaries
                if (
                    x["split"] == "train"
                    and x["radius"] == radius
                )
            )

            val_s = next(
                x
                for x in summaries
                if (
                    x["split"] == "validation"
                    and x["radius"] == radius
                )
            )

            a = train_s["classes"][
                str(class_id)
            ]

            b = val_s["classes"][
                str(class_id)
            ]

            print(
                f"  R{radius}: "
                f"train vox={a['mean_voxels']:.2f}, "
                f"present={a['cases_present']}/{TRAIN_N}, "
                f"comp={a['mean_components']:.2f}; "
                f"val vox={b['mean_voxels']:.2f}, "
                f"present={b['cases_present']}/{VAL_N}, "
                f"comp={b['mean_components']:.2f}"
            )

    # ------------------------------------------------------------
    # DIAGNOSTIC INTERPRETATION
    # ------------------------------------------------------------

    train_r0 = next(
        x
        for x in summaries
        if (
            x["split"] == "train"
            and x["radius"] == 0
        )
    )

    train_r3 = next(
        x
        for x in summaries
        if (
            x["split"] == "train"
            and x["radius"] == 3
        )
    )

    val_r0 = next(
        x
        for x in summaries
        if (
            x["split"] == "validation"
            and x["radius"] == 0
        )
    )

    val_r3 = next(
        x
        for x in summaries
        if (
            x["split"] == "validation"
            and x["radius"] == 3
        )
    )

    train_growth = (
        train_r3["mean_foreground_voxels"]
        / max(
            train_r0["mean_foreground_voxels"],
            1e-9,
        )
    )

    val_growth = (
        val_r3["mean_foreground_voxels"]
        / max(
            val_r0["mean_foreground_voxels"],
            1e-9,
        )
    )

    if (
        train_growth >= 3.0
        or val_growth >= 3.0
    ):
        diagnosis = (
            "PSEUDOMASK_DENSITY_IS_HIGHLY_SENSITIVE_TO_DILATION"
        )
    elif (
        train_growth >= 1.5
        or val_growth >= 1.5
    ):
        diagnosis = (
            "PSEUDOMASK_DENSITY_IS_MODERATELY_SENSITIVE_TO_DILATION"
        )
    else:
        diagnosis = (
            "PSEUDOMASK_DENSITY_REMAINS_LOW_UNDER_TESTED_DILATION"
        )

    result = {
        "part": 52,
        "configuration": {
            "train_subset": TRAIN_N,
            "validation_subset": VAL_N,
            "full_shape": FULL_SHAPE,
            "crop_shape": CROP_SHAPE,
            "crop_policy": "foreground-centered",
            "radii": RADII,
            "connectivity": "3D 6-connected",
            "small_component_threshold": SMALL_COMPONENT_THRESHOLD,
            "class_names": CLASS_NAMES,
        },
        "method": {
            "training": False,
            "optimizer": False,
            "spider": False,
            "test_set": False,
            "part15_overwritten": False,
            "base_masks": "current Part 11 pseudo-masks",
        },
        "train_summaries": [
            x
            for x in summaries
            if x["split"] == "train"
        ],
        "validation_summaries": [
            x
            for x in summaries
            if x["split"] == "validation"
        ],
        "radius_growth": {
            "train_radius_0_to_3": train_growth,
            "validation_radius_0_to_3": val_growth,
        },
        "diagnosis": diagnosis,
        "note": (
            "Dilation is a sensitivity analysis only. "
            "It does not establish that any tested radius is "
            "medically correct ground truth."
        ),
        "files": {
            "case_csv": str(case_csv),
            "summary_json": str(
                REPORT / "part52_summary.json"
            ),
        },
    }

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT / "part52_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 82)
    print("PART 52 DIAGNOSTIC INTERPRETATION")
    print("=" * 82)

    print(
        f"Train radius 0→3 FG growth : "
        f"{train_growth:.3f}x"
    )

    print(
        f"Validation radius 0→3 FG growth : "
        f"{val_growth:.3f}x"
    )

    print(
        f"Diagnosis : {diagnosis}"
    )

    print()
    print("=" * 82)
    print("PART 52 COMPLETE")
    print("=" * 82)

    print(
        f"Case geometry CSV : {case_csv}"
    )

    print(
        f"Summary JSON : "
        f"{REPORT / 'part52_summary.json'}"
    )


if __name__ == "__main__":
    main()
