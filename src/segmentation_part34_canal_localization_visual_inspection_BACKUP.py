"""
PHASE 3 - PART 34
IMAGE-ONLY CANAL-PRESERVING LOCALIZATION VISUAL INSPECTION

Purpose
-------
Part 34 visually inspects representative cases from Part 33.

The objective is to determine whether image-only localization strategies
actually place the 96 x 96 x 96 crop over the spinal anatomy, especially
the spinal canal.

This is an EVALUATION / VISUALIZATION ONLY stage.

Ground truth:
    - NEVER used to calculate or choose the localization center.
    - Used ONLY to overlay anatomy and quantify/display retained anatomy.

Model:
    - No Swin-UNETR inference is performed.
    - No model weights are modified.
    - No training is performed.

Representative groups:
    1. Critical residual canal cases
    2. Strongest canal-retention improvements
    3. T2 SPACE cases
    4. Cases where canal improves but combined retention decreases
    5. Stable / good fixed-center cases

Outputs:
    - Per-case visualization PNGs
    - Representative-case CSV
    - Strategy comparison CSV
    - JSON summary
    - Text report
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError(
        "SimpleITK is required. Install with: pip install SimpleITK"
    ) from exc


# ============================================================================
# CONFIGURATION
# ============================================================================

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

PART33_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part33_high_risk_spinal_canal_analysis"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part34_canal_localization_visual_inspection"
)

PATCH_SIZE = (96, 96, 96)

CLASS_NAMES = {
    0: "Background",
    1: "Vertebrae",
    2: "Spinal Canal",
    3: "Intervertebral Disc",
}

# Number of cases selected from each group.
MAX_PER_GROUP = 3

# Maximum total cases to visualize.
MAX_TOTAL_CASES = 15


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def banner(text: str) -> None:
    print("=" * 78)
    print(text)
    print("=" * 78)


def safe_float(value, default=np.nan):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
        return default
    except Exception:
        return default


def sequence_from_case(case_id: str) -> str:
    name = str(case_id).upper()

    if "T2_SPACE" in name:
        return "T2 SPACE"
    if "_T2" in name:
        return "T2"
    if "_T1" in name:
        return "T1"

    return "OTHER"


def load_mha(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    return np.asarray(sitk.GetArrayFromImage(image))


def find_file(directory: Path, case_id: str) -> Path | None:
    candidates = [
        directory / f"{case_id}.mha",
        directory / f"{case_id}.nii.gz",
        directory / f"{case_id}.nii",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Fallback for unusual extensions.
    matches = list(directory.glob(f"{case_id}.*"))
    return matches[0] if matches else None


# ============================================================================
# IMAGE-ONLY LOCALIZATION
# ============================================================================

def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)

    finite = np.isfinite(image)

    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    values = image[finite]

    lo = np.percentile(values, 1)
    hi = np.percentile(values, 99)

    if hi <= lo:
        lo = float(values.min())
        hi = float(values.max())

    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    result = (image - lo) / (hi - lo)
    result = np.clip(result, 0.0, 1.0)
    result[~finite] = 0.0

    return result.astype(np.float32)


def fixed_center(image: np.ndarray):
    z, y, x = image.shape
    return (x / 2.0, y / 2.0, z / 2.0)


def image_nonzero_center(image: np.ndarray):
    norm = normalize_image(image)
    coords = np.argwhere(norm > 0.03)

    if len(coords) == 0:
        return fixed_center(image)

    z, y, x = np.median(coords, axis=0)
    return (float(x), float(y), float(z))


def robust_intensity_center(image: np.ndarray):
    norm = normalize_image(image)

    positive = norm[norm > 0]

    if positive.size == 0:
        return fixed_center(image)

    threshold = np.percentile(positive, 60)
    coords = np.argwhere(norm >= threshold)

    if len(coords) == 0:
        return fixed_center(image)

    z, y, x = np.median(coords, axis=0)
    return (float(x), float(y), float(z))


def intensity_weighted_center(image: np.ndarray):
    norm = normalize_image(image)
    weights = norm.astype(np.float64)

    total = weights.sum()

    if total <= 1e-8:
        return fixed_center(image)

    z_axis = np.arange(image.shape[0], dtype=np.float64)
    y_axis = np.arange(image.shape[1], dtype=np.float64)
    x_axis = np.arange(image.shape[2], dtype=np.float64)

    wz = weights.sum(axis=(1, 2))
    wy = weights.sum(axis=(0, 2))
    wx = weights.sum(axis=(0, 1))

    z = float((z_axis * wz).sum() / total)
    y = float((y_axis * wy).sum() / total)
    x = float((x_axis * wx).sum() / total)

    return (x, y, z)


def canal_axis_image_center(image: np.ndarray):
    norm = normalize_image(image)

    z, y, x = image.shape

    yy, xx = np.mgrid[0:y, 0:x]

    cx = (x - 1) / 2.0
    cy = (y - 1) / 2.0

    sigma_x = max(x * 0.30, 1.0)
    sigma_y = max(y * 0.30, 1.0)

    central_prior = np.exp(
        -(
            ((xx - cx) ** 2) / (2 * sigma_x**2)
            + ((yy - cy) ** 2) / (2 * sigma_y**2)
        )
    )

    xy_score = norm.mean(axis=0) * central_prior

    if np.all(xy_score <= 0):
        return fixed_center(image)

    y_idx, x_idx = np.unravel_index(
        np.argmax(xy_score),
        xy_score.shape,
    )

    profile = norm[:, int(y_idx), int(x_idx)]

    if profile.sum() > 0:
        z_idx = float(
            np.sum(np.arange(z) * profile)
            / profile.sum()
        )
    else:
        z_idx = z / 2.0

    return (
        float(x_idx),
        float(y_idx),
        float(z_idx),
    )


def hybrid_image_center(image: np.ndarray):
    a = np.asarray(image_nonzero_center(image))
    b = np.asarray(robust_intensity_center(image))
    c = np.asarray(fixed_center(image))

    result = 0.45 * a + 0.40 * b + 0.15 * c

    return tuple(float(v) for v in result)


def canal_preserving_hybrid(image: np.ndarray):
    a = np.asarray(hybrid_image_center(image))
    b = np.asarray(canal_axis_image_center(image))
    c = np.asarray(fixed_center(image))

    result = 0.55 * a + 0.35 * b + 0.10 * c

    return tuple(float(v) for v in result)


def multi_candidate_canal_center(image: np.ndarray):
    candidates = [
        fixed_center(image),
        hybrid_image_center(image),
        intensity_weighted_center(image),
        canal_axis_image_center(image),
        canal_preserving_hybrid(image),
    ]

    norm = normalize_image(image)

    z, y, x = image.shape

    scored = []

    for center in candidates:
        cx, cy, cz = center

        xi = int(np.clip(round(cx), 0, x - 1))
        yi = int(np.clip(round(cy), 0, y - 1))
        zi = int(np.clip(round(cz), 0, z - 1))

        z0 = max(0, zi - 12)
        z1 = min(z, zi + 13)
        y0 = max(0, yi - 12)
        y1 = min(y, yi + 13)
        x0 = max(0, xi - 12)
        x1 = min(x, xi + 13)

        local_mean = float(
            norm[z0:z1, y0:y1, x0:x1].mean()
        )

        penalty = 0.0

        if cx < PATCH_SIZE[2] / 2 or cx > x - PATCH_SIZE[2] / 2:
            penalty += 0.05

        if cy < PATCH_SIZE[1] / 2 or cy > y - PATCH_SIZE[1] / 2:
            penalty += 0.05

        if cz < PATCH_SIZE[0] / 2 or cz > z - PATCH_SIZE[0] / 2:
            penalty += 0.05

        scored.append((local_mean - penalty, center))

    scored.sort(key=lambda item: item[0], reverse=True)

    return tuple(float(v) for v in scored[0][1])


LOCALIZATION_FUNCTIONS = {
    "fixed_center": fixed_center,
    "hybrid_image_center": hybrid_image_center,
    "intensity_weighted_center": intensity_weighted_center,
    "canal_axis_image_center": canal_axis_image_center,
    "canal_preserving_hybrid": canal_preserving_hybrid,
    "multi_candidate_canal_center": multi_candidate_canal_center,
}


# ============================================================================
# CROP / RETENTION
# ============================================================================

def center_crop_pad(
    volume: np.ndarray,
    center_xyz,
    patch_size=PATCH_SIZE,
):
    target_z, target_y, target_x = patch_size

    cx, cy, cz = center_xyz

    cx = int(round(cx))
    cy = int(round(cy))
    cz = int(round(cz))

    z0 = cz - target_z // 2
    y0 = cy - target_y // 2
    x0 = cx - target_x // 2

    z1 = z0 + target_z
    y1 = y0 + target_y
    x1 = x0 + target_x

    output = np.zeros(
        (target_z, target_y, target_x),
        dtype=volume.dtype,
    )

    src_z0 = max(0, z0)
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)

    src_z1 = min(volume.shape[0], z1)
    src_y1 = min(volume.shape[1], y1)
    src_x1 = min(volume.shape[2], x1)

    dst_z0 = src_z0 - z0
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0

    dst_z1 = dst_z0 + max(0, src_z1 - src_z0)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)

    if (
        src_z1 > src_z0
        and src_y1 > src_y0
        and src_x1 > src_x0
    ):
        output[
            dst_z0:dst_z1,
            dst_y0:dst_y1,
            dst_x0:dst_x1,
        ] = volume[
            src_z0:src_z1,
            src_y0:src_y1,
            src_x0:src_x1,
        ]

    return output


def class_retention(
    original_mask: np.ndarray,
    cropped_mask: np.ndarray,
    label: int,
) -> float:
    total = int(np.count_nonzero(original_mask == label))

    if total == 0:
        return np.nan

    retained = int(np.count_nonzero(cropped_mask == label))

    return retained / total


def crop_metrics(mask: np.ndarray, center):
    cropped = center_crop_pad(mask, center)

    values = {
        "vertebrae_retention": class_retention(mask, cropped, 1),
        "spinal_canal_retention": class_retention(mask, cropped, 2),
        "disc_retention": class_retention(mask, cropped, 3),
    }

    foreground_total = np.count_nonzero(mask > 0)
    foreground_retained = np.count_nonzero(cropped > 0)

    values["foreground_retention"] = (
        foreground_retained / foreground_total
        if foreground_total > 0
        else np.nan
    )

    class_values = [
        values["vertebrae_retention"],
        values["spinal_canal_retention"],
        values["disc_retention"],
    ]

    finite = [
        value for value in class_values
        if np.isfinite(value)
    ]

    values["combined_retention"] = (
        float(np.mean(finite))
        if finite else np.nan
    )

    values["minimum_class_retention"] = (
        float(np.min(finite))
        if finite else np.nan
    )

    return values, cropped


def distance(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float64)
            - np.asarray(b, dtype=np.float64)
        )
    )


# ============================================================================
# SLICE SELECTION
# ============================================================================

def select_best_slice(
    image: np.ndarray,
    mask: np.ndarray,
    center,
) -> int:
    """
    Select the axial slice closest to the image-only localization center
    that contains meaningful spinal-canal ground truth.

    Ground truth is used ONLY to select a useful visualization slice,
    not to select the localization center.
    """
    _, _, zc = center

    canal_counts = np.sum(mask == 2, axis=(1, 2))

    nonzero = np.where(canal_counts > 0)[0]

    if len(nonzero) == 0:
        return int(np.clip(round(zc), 0, image.shape[0] - 1))

    # Prefer slices near the localization center.
    order = sorted(
        nonzero.tolist(),
        key=lambda z: (
            abs(z - zc),
            -int(canal_counts[z]),
        ),
    )

    return int(order[0])


def axial_crop_bounds(center, shape):
    """
    Return x/y crop rectangle corresponding to the 96x96 crop
    in the selected axial slice.
    """
    cx, cy, _ = center

    half_x = PATCH_SIZE[2] / 2.0
    half_y = PATCH_SIZE[1] / 2.0

    x0 = cx - half_x
    x1 = cx + half_x
    y0 = cy - half_y
    y1 = cy + half_y

    return x0, x1, y0, y1


# ============================================================================
# REPRESENTATIVE CASE SELECTION
# ============================================================================

def select_representative_cases(case_df: pd.DataFrame):
    selected = {}

    # 1. Critical residual cases.
    critical = (
        case_df[
            case_df["failure_category"]
            == "Critical canal retention"
        ]
        .sort_values("best_spinal_canal_retention")
        .head(MAX_PER_GROUP)
    )

    selected["critical_residual"] = critical["file"].tolist()

    # 2. Strongest improvement.
    improvement = (
        case_df
        .sort_values(
            "canal_retention_improvement",
            ascending=False,
        )
        .head(MAX_PER_GROUP)
    )

    selected["strongest_improvement"] = improvement["file"].tolist()

    # 3. T2 SPACE cases.
    t2_space = (
        case_df[
            case_df["sequence"] == "T2 SPACE"
        ]
        .sort_values(
            "canal_retention_improvement",
            ascending=False,
        )
        .head(MAX_PER_GROUP)
    )

    selected["t2_space"] = t2_space["file"].tolist()

    # 4. Canal improves but combined retention decreases.
    tradeoff = (
        case_df[
            (case_df["canal_retention_improvement"] > 0)
            & (case_df["combined_retention_improvement"] < 0)
        ]
        .sort_values(
            "combined_retention_improvement"
        )
        .head(MAX_PER_GROUP)
    )

    selected["canal_tradeoff"] = tradeoff["file"].tolist()

    # 5. Stable good fixed-center cases.
    stable = (
        case_df[
            case_df["fixed_spinal_canal_retention"] >= 0.30
        ]
        .sort_values(
            "fixed_spinal_canal_retention",
            ascending=False,
        )
        .head(MAX_PER_GROUP)
    )

    selected["stable_fixed_center"] = stable["file"].tolist()

    # Preserve order and remove duplicates.
    final = []

    for group_cases in selected.values():
        for case in group_cases:
            if case not in final:
                final.append(case)

    final = final[:MAX_TOTAL_CASES]

    return selected, final


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_case_visualization(
    case_id: str,
    group_names: list[str],
    image: np.ndarray,
    mask: np.ndarray,
    case_row: pd.Series,
    output_dir: Path,
):
    """
    Creates one multi-panel visualization.

    Panels:
        1. MRI
        2. Ground-truth anatomy
        3. Fixed-center crop boundary
        4. Best-strategy crop boundary
        5. Difference between fixed and best crop
        6. Anatomical retention comparison
    """
    best_strategy = str(case_row["best_strategy"])

    fixed_center_value = LOCALIZATION_FUNCTIONS[
        "fixed_center"
    ](image)

    best_center = LOCALIZATION_FUNCTIONS[
        best_strategy
    ](image)

    slice_index = select_best_slice(
        image,
        mask,
        best_center,
    )

    image_slice = normalize_image(
        image[slice_index]
    )

    mask_slice = mask[slice_index]

    # Build RGB-like anatomy overlay.
    overlay = np.dstack(
        [
            image_slice,
            image_slice,
            image_slice,
        ]
    )

    # Red = vertebrae
    overlay[mask_slice == 1, 0] = 1.0
    overlay[mask_slice == 1, 1] *= 0.35
    overlay[mask_slice == 1, 2] *= 0.35

    # Green = spinal canal
    overlay[mask_slice == 2, 0] *= 0.35
    overlay[mask_slice == 2, 1] = 1.0
    overlay[mask_slice == 2, 2] *= 0.35

    # Blue = discs
    overlay[mask_slice == 3, 0] *= 0.35
    overlay[mask_slice == 3, 1] *= 0.35
    overlay[mask_slice == 3, 2] = 1.0

    fixed_bounds = axial_crop_bounds(
        fixed_center_value,
        image.shape,
    )

    best_bounds = axial_crop_bounds(
        best_center,
        image.shape,
    )

    fixed_metrics, fixed_crop_mask = crop_metrics(
        mask,
        fixed_center_value,
    )

    best_metrics, best_crop_mask = crop_metrics(
        mask,
        best_center,
    )

    # Difference visualization:
    # positive = anatomy gained by best crop
    # negative = anatomy lost compared with fixed crop
    fixed_slice_mask = fixed_crop_mask[slice_index]
    best_slice_mask = best_crop_mask[slice_index]

    gain = (
        (best_slice_mask > 0)
        & (fixed_slice_mask == 0)
    )

    loss = (
        (fixed_slice_mask > 0)
        & (best_slice_mask == 0)
    )

    difference = np.zeros(
        (*mask_slice.shape, 3),
        dtype=np.float32,
    )

    difference[gain] = [0.0, 1.0, 0.0]
    difference[loss] = [1.0, 0.0, 0.0]

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    # Panel 1.
    axes[0, 0].imshow(
        image_slice,
        cmap="gray",
    )
    axes[0, 0].set_title(
        f"MRI — slice {slice_index}"
    )
    axes[0, 0].axis("off")

    # Panel 2.
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(
        "Ground Truth Anatomy\n"
        "Red=Vertebrae | Green=Canal | Blue=Disc"
    )
    axes[0, 1].axis("off")

    # Panel 3.
    axes[0, 2].imshow(
        image_slice,
        cmap="gray",
    )

    x0, x1, y0, y1 = fixed_bounds

    axes[0, 2].add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            linewidth=2,
            linestyle="--",
        )
    )

    axes[0, 2].scatter(
        [fixed_center_value[0]],
        [fixed_center_value[1]],
        s=50,
        marker="x",
    )

    axes[0, 2].set_title(
        "Fixed Center\n96×96 crop"
    )
    axes[0, 2].set_xlim(0, image.shape[2])
    axes[0, 2].set_ylim(image.shape[1], 0)
    axes[0, 2].axis("off")

    # Panel 4.
    axes[1, 0].imshow(
        image_slice,
        cmap="gray",
    )

    x0, x1, y0, y1 = best_bounds

    axes[1, 0].add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            linewidth=2,
            linestyle="--",
        )
    )

    axes[1, 0].scatter(
        [best_center[0]],
        [best_center[1]],
        s=50,
        marker="x",
    )

    axes[1, 0].set_title(
        f"{best_strategy}\n96×96 crop"
    )
    axes[1, 0].set_xlim(0, image.shape[2])
    axes[1, 0].set_ylim(image.shape[1], 0)
    axes[1, 0].axis("off")

    # Panel 5.
    axes[1, 1].imshow(
        image_slice,
        cmap="gray",
    )
    axes[1, 1].imshow(
        difference,
        alpha=0.55,
    )
    axes[1, 1].set_title(
        "Crop Difference\n"
        "Green = anatomy gained | Red = anatomy lost"
    )
    axes[1, 1].axis("off")

    # Panel 6.
    class_labels = [
        "Vertebrae",
        "Canal",
        "Disc",
    ]

    fixed_values = [
        fixed_metrics["vertebrae_retention"],
        fixed_metrics["spinal_canal_retention"],
        fixed_metrics["disc_retention"],
    ]

    best_values = [
        best_metrics["vertebrae_retention"],
        best_metrics["spinal_canal_retention"],
        best_metrics["disc_retention"],
    ]

    x = np.arange(len(class_labels))
    width = 0.36

    axes[1, 2].bar(
        x - width / 2,
        fixed_values,
        width,
        label="Fixed",
    )

    axes[1, 2].bar(
        x + width / 2,
        best_values,
        width,
        label=best_strategy,
    )

    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(class_labels)

    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].set_ylabel("Retention")
    axes[1, 2].set_title(
        "Anatomical Retention"
    )
    axes[1, 2].legend()

    figure.suptitle(
        f"Part 34 — {case_id} | {sequence_from_case(case_id)}\n"
        f"Groups: {', '.join(group_names)}",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    case_output = output_dir / case_id
    case_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        case_output
        / f"{case_id}_localization_visual_analysis.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return {
        "slice_index": slice_index,
        "best_strategy": best_strategy,
        "fixed_center_x": fixed_center_value[0],
        "fixed_center_y": fixed_center_value[1],
        "fixed_center_z": fixed_center_value[2],
        "best_center_x": best_center[0],
        "best_center_y": best_center[1],
        "best_center_z": best_center[2],
        "localization_shift_voxels": distance(
            fixed_center_value,
            best_center,
        ),
        "output_path": str(output_path),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    start = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    banner("PHASE 3 - PART 34")
    print("IMAGE-ONLY CANAL-PRESERVING LOCALIZATION VISUAL INSPECTION")
    banner("")

    print("PROJECT ROOT")
    print(PROJECT_ROOT)
    print()

    print("TEST DIRECTORY")
    print(TEST_DIR)
    print()

    print("PART 33 DIRECTORY")
    print(PART33_DIR)
    print()

    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)
    print()

    # ------------------------------------------------------------------
    banner("LOADING PART 33 RESULTS")

    best_path = (
        PART33_DIR
        / "part33_best_strategy_per_case.csv"
    )

    case_path = (
        PART33_DIR
        / "part33_high_risk_case_analysis.csv"
    )

    if not best_path.exists():
        raise FileNotFoundError(
            f"Missing Part 33 best-case results:\n{best_path}"
        )

    if not case_path.exists():
        raise FileNotFoundError(
            f"Missing Part 33 case analysis:\n{case_path}"
        )

    best_df = pd.read_csv(best_path)
    case_df = pd.read_csv(case_path)

    print(f"Part 33 case rows: {len(case_df)}")
    print(f"Part 33 best-case rows: {len(best_df)}")

    # ------------------------------------------------------------------
    banner("SELECTING REPRESENTATIVE CASES")

    selected_groups, selected_cases = (
        select_representative_cases(best_df)
    )

    for group_name, cases in selected_groups.items():
        print(
            f"{group_name}: {cases}"
        )

    print()
    print(
        f"Unique selected cases: {len(selected_cases)}"
    )

    # Map case -> groups.
    case_groups = {}

    for group_name, cases in selected_groups.items():
        for case_id in cases:
            case_groups.setdefault(
                case_id,
                [],
            ).append(group_name)

    # ------------------------------------------------------------------
    banner("PROCESSING VISUAL INSPECTION")

    visualization_rows = []

    for index, case_id in enumerate(
        selected_cases,
        start=1,
    ):
        print(
            f"[{index}/{len(selected_cases)}] {case_id}"
        )

        image_path = find_file(
            IMAGE_DIR,
            case_id,
        )

        mask_path = find_file(
            MASK_DIR,
            case_id,
        )

        if image_path is None or mask_path is None:
            print(
                f"WARNING: missing image or mask for {case_id}"
            )
            continue

        image = load_mha(image_path)
        mask = load_mha(mask_path)

        if image.shape != mask.shape:
            print(
                f"WARNING: shape mismatch for {case_id}: "
                f"{image.shape} vs {mask.shape}"
            )
            continue

        row = best_df[
            best_df["file"].astype(str) == str(case_id)
        ]

        if row.empty:
            print(
                f"WARNING: Part 33 row missing for {case_id}"
            )
            continue

        row = row.iloc[0]

        result = create_case_visualization(
            case_id=case_id,
            group_names=case_groups.get(
                case_id,
                [],
            ),
            image=image,
            mask=mask,
            case_row=row,
            output_dir=OUTPUT_DIR,
        )

        visualization_rows.append(
            {
                "file": case_id,
                "sequence": sequence_from_case(
                    case_id
                ),
                "groups": "|".join(
                    case_groups.get(
                        case_id,
                        [],
                    )
                ),
                "best_strategy": row[
                    "best_strategy"
                ],
                "best_spinal_canal_retention": safe_float(
                    row[
                        "best_spinal_canal_retention"
                    ]
                ),
                "fixed_spinal_canal_retention": safe_float(
                    row[
                        "fixed_spinal_canal_retention"
                    ]
                ),
                "canal_retention_improvement": safe_float(
                    row[
                        "canal_retention_improvement"
                    ]
                ),
                "best_combined_retention": safe_float(
                    row[
                        "best_combined_retention"
                    ]
                ),
                "fixed_combined_retention": safe_float(
                    row[
                        "fixed_combined_retention"
                    ]
                ),
                "combined_retention_improvement": safe_float(
                    row[
                        "combined_retention_improvement"
                    ]
                ),
                **result,
            }
        )

        print(
            f"  Best strategy : "
            f"{row['best_strategy']}"
        )
        print(
            f"  Canal retention: "
            f"{safe_float(row['best_spinal_canal_retention']):.6f}"
        )
        print(
            f"  Improvement    : "
            f"{safe_float(row['canal_retention_improvement']):+.6f}"
        )
        print(
            f"  Slice          : "
            f"{result['slice_index']}"
        )
        print(
            f"  Saved          : "
            f"{result['output_path']}"
        )

    visual_df = pd.DataFrame(
        visualization_rows
    )

    if visual_df.empty:
        raise RuntimeError(
            "No visualizations were generated."
        )

    # ------------------------------------------------------------------
    banner("VISUAL INSPECTION SUMMARY")

    print(
        f"Cases visualized: {len(visual_df)}"
    )

    print()
    print(
        "Best strategy distribution:"
    )

    print(
        visual_df["best_strategy"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Mean canal retention:"
    )

    print(
        f"Fixed center : "
        f"{visual_df['fixed_spinal_canal_retention'].mean():.6f}"
    )

    print(
        f"Best strategy: "
        f"{visual_df['best_spinal_canal_retention'].mean():.6f}"
    )

    print(
        f"Mean improvement: "
        f"{visual_df['canal_retention_improvement'].mean():+.6f}"
    )

    # ------------------------------------------------------------------
    banner("SAVING VISUAL INSPECTION TABLES")

    summary_csv = (
        OUTPUT_DIR
        / "part34_visual_inspection_summary.csv"
    )

    visual_df.to_csv(
        summary_csv,
        index=False,
    )

    # Selected group table.
    group_rows = []

    for group_name, cases in selected_groups.items():
        for case_id in cases:
            group_rows.append(
                {
                    "group": group_name,
                    "file": case_id,
                }
            )

    groups_csv = (
        OUTPUT_DIR
        / "part34_selected_case_groups.csv"
    )

    pd.DataFrame(group_rows).to_csv(
        groups_csv,
        index=False,
    )

    print(f"Saved: {summary_csv}")
    print(f"Saved: {groups_csv}")

    # ------------------------------------------------------------------
    banner("CREATING SUMMARY CHARTS")

    # Strategy distribution.
    strategy_counts = (
        visual_df["best_strategy"]
        .value_counts()
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        strategy_counts.index,
        strategy_counts.values,
    )

    plt.xlabel("Best localization strategy")
    plt.ylabel("Number of representative cases")
    plt.title(
        "Part 34 - Best Strategy Distribution"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    strategy_chart = (
        OUTPUT_DIR
        / "part34_best_strategy_distribution.png"
    )

    plt.savefig(
        strategy_chart,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {strategy_chart}"
    )

    # Canal retention comparison.
    sorted_visual = visual_df.sort_values(
        "canal_retention_improvement"
    )

    plt.figure(figsize=(12, 7))

    x = np.arange(
        len(sorted_visual)
    )

    width = 0.36

    plt.bar(
        x - width / 2,
        sorted_visual[
            "fixed_spinal_canal_retention"
        ],
        width,
        label="Fixed center",
    )

    plt.bar(
        x + width / 2,
        sorted_visual[
            "best_spinal_canal_retention"
        ],
        width,
        label="Best image-only strategy",
    )

    plt.xticks(
        x,
        sorted_visual["file"],
        rotation=45,
        ha="right",
    )

    plt.ylabel("Spinal-canal retention")
    plt.xlabel("Representative case")
    plt.title(
        "Part 34 - Fixed vs Best Image-Only Canal Retention"
    )

    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    retention_chart = (
        OUTPUT_DIR
        / "part34_fixed_vs_best_canal_retention.png"
    )

    plt.savefig(
        retention_chart,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {retention_chart}"
    )

    # Localization shift.
    plt.figure(figsize=(12, 7))

    shift_sorted = visual_df.sort_values(
        "localization_shift_voxels"
    )

    plt.bar(
        shift_sorted["file"],
        shift_sorted[
            "localization_shift_voxels"
        ],
    )

    plt.ylabel(
        "Localization shift from fixed center (voxels)"
    )
    plt.xlabel("Representative case")
    plt.title(
        "Part 34 - Image-Only Localization Shift"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    shift_chart = (
        OUTPUT_DIR
        / "part34_localization_shift.png"
    )

    plt.savefig(
        shift_chart,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved: {shift_chart}"
    )

    # ------------------------------------------------------------------
    banner("CREATING FINAL SUMMARY")

    critical_count = int(
        (
            visual_df[
                "best_spinal_canal_retention"
            ] < 0.02
        ).sum()
    )

    low_count = int(
        (
            visual_df[
                "best_spinal_canal_retention"
            ] < 0.10
        ).sum()
    )

    improved_count = int(
        (
            visual_df[
                "canal_retention_improvement"
            ] > 0
        ).sum()
    )

    tradeoff_count = int(
        (
            visual_df[
                "combined_retention_improvement"
            ] < 0
        ).sum()
    )

    mean_fixed = float(
        visual_df[
            "fixed_spinal_canal_retention"
        ].mean()
    )

    mean_best = float(
        visual_df[
            "best_spinal_canal_retention"
        ].mean()
    )

    mean_improvement = float(
        visual_df[
            "canal_retention_improvement"
        ].mean()
    )

    json_summary = {
        "phase": "Phase 3 - Part 34",
        "title": (
            "Image-Only Canal-Preserving "
            "Localization Visual Inspection"
        ),
        "cases_visualized": int(
            len(visual_df)
        ),
        "selected_groups": selected_groups,
        "patch_size": list(PATCH_SIZE),
        "ground_truth_used_for_localization": False,
        "ground_truth_used_for_evaluation": True,
        "training_performed": False,
        "model_weights_modified": False,
        "mean_fixed_canal_retention": mean_fixed,
        "mean_best_canal_retention": mean_best,
        "mean_canal_improvement": mean_improvement,
        "cases_with_canal_improvement": improved_count,
        "critical_cases_after_best_strategy": critical_count,
        "low_retention_cases_after_best_strategy": low_count,
        "canal_combined_tradeoff_cases": tradeoff_count,
        "output_directory": str(
            OUTPUT_DIR
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "phase3_part34_summary.json"
    )

    json_path.write_text(
        json.dumps(
            json_summary,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved: {json_path}"
    )

    # ------------------------------------------------------------------
    elapsed = (
        time.time() - start
    ) / 60.0

    report_lines = [
        "=" * 78,
        "PHASE 3 - PART 34",
        "IMAGE-ONLY CANAL-PRESERVING LOCALIZATION VISUAL INSPECTION",
        "=" * 78,
        "",
        f"Cases visualized: {len(visual_df)}",
        f"Mean fixed canal retention: {mean_fixed:.6f}",
        f"Mean best canal retention: {mean_best:.6f}",
        f"Mean canal improvement: {mean_improvement:+.6f}",
        f"Cases improved: {improved_count}",
        f"Critical residual cases: {critical_count}",
        f"Low-retention cases: {low_count}",
        f"Canal/combined tradeoff cases: {tradeoff_count}",
        "",
        "Purpose:",
        (
            "Visual inspection of representative high-risk cases "
            "identified by Part 33. Crop boundaries and anatomical "
            "retention are displayed to assess whether image-only "
            "localization places the 96^3 crop over the spinal anatomy."
        ),
        "",
        "Interpretation:",
        (
            "This part is a qualitative/visual validation stage. "
            "It does not establish segmentation performance because "
            "no Swin-UNETR inference or retraining is performed."
        ),
        "",
        "Ground-truth used for localization: NO",
        "Ground-truth used for evaluation/overlay: YES",
        "Training performed: NO",
        "Model weights modified: NO",
        "",
        f"Execution time: {elapsed:.2f} minutes",
        "",
        "=" * 78,
        "PHASE 3 - PART 34 COMPLETE",
        "=" * 78,
    ]

    report_path = (
        OUTPUT_DIR
        / "phase3_part34_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(
        f"Saved: {report_path}"
    )

    # ------------------------------------------------------------------
    banner("PART 34 COMPLETE")

    print(
        f"Cases visualized              : "
        f"{len(visual_df)}"
    )
    print(
        f"Mean fixed canal retention    : "
        f"{mean_fixed:.6f}"
    )
    print(
        f"Mean best canal retention     : "
        f"{mean_best:.6f}"
    )
    print(
        f"Mean canal improvement        : "
        f"{mean_improvement:+.6f}"
    )
    print(
        f"Cases improved                : "
        f"{improved_count}"
    )
    print(
        f"Critical residual cases      : "
        f"{critical_count}"
    )

    print()
    print(
        "Ground-truth used for localization: NO"
    )
    print(
        "Ground-truth used for evaluation: YES"
    )
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
    print("PHASE 3 - PART 34 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
