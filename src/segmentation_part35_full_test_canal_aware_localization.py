"""
==============================================================================
PHASE 3 - PART 35
FULL TEST-SET CANAL-AWARE IMAGE-ONLY LOCALIZATION EVALUATION
==============================================================================

Purpose
-------
Evaluate canal-aware, image-only localization strategies on the COMPLETE
71-case test set before any new Swin-UNETR training is performed.

This is a localization/retention experiment only:
    - Ground truth is NEVER used to determine crop centers.
    - Ground truth IS used only to calculate retention/evaluation metrics.
    - No model is trained.
    - No model weights are modified.

Strategies
----------
1. fixed_center
2. hybrid_image_center
3. intensity_weighted_center
4. canal_axis_image_center
5. canal_preserving_hybrid

The script is deliberately self-contained and does not depend on Part 34's
13-case visual inspection subset.

Outputs
-------
outputs/segmentation/part35_full_test_canal_aware_localization/
    part35_case_analysis.csv
    part35_strategy_summary.csv
    part35_per_case_best_strategy.csv
    part35_canal_retention_comparison.png
    part35_combined_retention_comparison.png
    part35_balanced_retention_comparison.png
    part35_localization_error_comparison.png
    part35_canal_failure_count.png
    phase3_part35_summary.json
    phase3_part35_report.txt
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
except ImportError:
    raise ImportError(
        "SimpleITK is required. Install it with: pip install SimpleITK"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

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

PART34_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part34_canal_localization_visual_inspection"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part35_full_test_canal_aware_localization"
)

PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 4

CLASS_NAMES = {
    1: "Vertebrae",
    2: "Spinal Canal",
    3: "Intervertebral Disc",
}

STRATEGIES = [
    "fixed_center",
    "hybrid_image_center",
    "intensity_weighted_center",
    "canal_axis_image_center",
    "canal_preserving_hybrid",
]


# =============================================================================
# PRINT HELPERS
# =============================================================================

def banner(text: str) -> None:
    print("=" * 78)
    print(text)
    print("=" * 78)


def section(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


# =============================================================================
# IMAGE / MASK LOADING
# =============================================================================

def read_mha(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    # SimpleITK returns [z, y, x].
    return np.asarray(array)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)

    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    values = image[finite]

    p1 = float(np.percentile(values, 1))
    p99 = float(np.percentile(values, 99))

    if p99 <= p1:
        lo = float(values.min())
        hi = float(values.max())
    else:
        lo, hi = p1, p99

    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo)

    return image.astype(np.float32)


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def clamp_center(center, shape, patch_size=PATCH_SIZE):
    """Clamp center to a valid crop center in [z,y,x] coordinates."""
    center = np.asarray(center, dtype=np.float64)
    shape = np.asarray(shape, dtype=np.int64)
    patch = np.asarray(patch_size, dtype=np.int64)

    result = np.zeros(3, dtype=np.float64)

    for i in range(3):
        half = patch[i] / 2.0
        low = half
        high = shape[i] - half

        if high < low:
            result[i] = shape[i] / 2.0
        else:
            result[i] = np.clip(center[i], low, high)

    return result


def crop_volume(volume, center, patch_size=PATCH_SIZE):
    """
    Center crop/pad to exactly patch_size.

    volume coordinates are [z,y,x].
    """
    patch = tuple(int(v) for v in patch_size)
    shape = np.asarray(volume.shape, dtype=np.int64)

    center = clamp_center(center, shape, patch)

    starts = np.floor(center - np.asarray(patch) / 2.0).astype(int)

    out = np.zeros(patch, dtype=volume.dtype)

    src_slices = []
    dst_slices = []

    for axis in range(3):
        start = int(starts[axis])
        end = start + patch[axis]

        src_start = max(0, start)
        src_end = min(int(shape[axis]), end)

        dst_start = max(0, -start)
        dst_end = dst_start + max(0, src_end - src_start)

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    if all(s.stop > s.start for s in src_slices):
        out[tuple(dst_slices)] = volume[tuple(src_slices)]

    return out


def crop_bounds(center, shape, patch_size=PATCH_SIZE):
    patch = np.asarray(patch_size, dtype=np.int64)
    shape = np.asarray(shape, dtype=np.int64)
    center = clamp_center(center, shape, patch)

    starts = np.floor(center - patch / 2.0).astype(int)
    ends = starts + patch

    return starts, ends


# =============================================================================
# IMAGE-ONLY LOCALIZATION
# =============================================================================

def robust_nonzero_center(image: np.ndarray):
    mask = image > np.percentile(image, 5)

    coords = np.argwhere(mask)

    if len(coords) == 0:
        return np.asarray(image.shape, dtype=float) / 2.0

    return np.median(coords, axis=0)


def robust_intensity_center(image: np.ndarray):
    threshold = np.percentile(image, 70)
    coords = np.argwhere(image >= threshold)

    if len(coords) == 0:
        return np.asarray(image.shape, dtype=float) / 2.0

    return np.mean(coords, axis=0)


def intensity_weighted_center(image: np.ndarray):
    """
    Weighted image center.

    This is intentionally image-only: no mask information is used.
    """
    z, y, x = np.indices(image.shape, dtype=np.float32)

    weights = np.clip(image, 0.0, None).astype(np.float64)
    weights = np.power(weights, 2.0)

    total = float(weights.sum())

    if total <= 1e-12:
        return np.asarray(image.shape, dtype=float) / 2.0

    cz = float((z * weights).sum() / total)
    cy = float((y * weights).sum() / total)
    cx = float((x * weights).sum() / total)

    return np.asarray([cz, cy, cx])


def hybrid_image_center(image: np.ndarray):
    """
    Robust hybrid image-only center.

    Combines:
        - non-zero anatomical extent
        - robust intensity center

    No ground-truth information is used.
    """
    nonzero = robust_nonzero_center(image)
    intensity = robust_intensity_center(image)

    center = 0.55 * nonzero + 0.45 * intensity

    return center


def canal_axis_image_center(image: np.ndarray):
    """
    Image-only approximation of the spinal canal axis.

    The method searches transverse slices for a stable central high-contrast
    anatomical corridor and aggregates the most central candidates.

    It uses ONLY image intensities.
    """
    image = np.asarray(image, dtype=np.float32)

    # Work on a robustly normalized image.
    norm = normalize_image(image)

    z_scores = []

    z_mid = image.shape[0] / 2.0
    y_mid = image.shape[1] / 2.0
    x_mid = image.shape[2] / 2.0

    # Evaluate a sparse set of axial slices to keep the RTX-2050/CPU workflow
    # lightweight.
    step = max(1, image.shape[0] // 24)

    for z in range(0, image.shape[0], step):
        plane = norm[z]

        # Ignore the very dark background.
        q = np.percentile(plane, 60)
        candidate = plane >= q

        coords = np.argwhere(candidate)

        if len(coords) < 20:
            continue

        cy, cx = np.median(coords, axis=0)

        # Prefer candidates near the anatomical center while avoiding the
        # outer image boundary.
        radial = math.sqrt(
            ((cy - y_mid) / max(y_mid, 1.0)) ** 2
            + ((cx - x_mid) / max(x_mid, 1.0)) ** 2
        )

        intensity = float(np.mean(plane[candidate]))

        score = intensity / (1.0 + radial)

        z_scores.append((score, z, cy, cx))

    if not z_scores:
        return hybrid_image_center(image)

    z_scores.sort(reverse=True)

    top = z_scores[: max(3, min(8, len(z_scores)))]

    weights = np.asarray([max(v[0], 1e-6) for v in top])
    weights /= weights.sum()

    cz = float(np.sum([w * v[1] for w, v in zip(weights, top)]))
    cy = float(np.sum([w * v[2] for w, v in zip(weights, top)]))
    cx = float(np.sum([w * v[3] for w, v in zip(weights, top)]))

    # Keep the z component closer to the robust anatomical extent.
    robust = robust_nonzero_center(image)
    cz = 0.65 * cz + 0.35 * robust[0]

    return np.asarray([cz, cy, cx])


def canal_preserving_hybrid(image: np.ndarray):
    """
    Canal-preserving image-only center.

    Combines the most stable image-only estimates and biases the center
    toward the canal-axis estimate.

    Ground truth is NOT used.
    """
    hybrid = hybrid_image_center(image)
    canal = canal_axis_image_center(image)
    weighted = intensity_weighted_center(image)

    center = (
        0.25 * hybrid
        + 0.55 * canal
        + 0.20 * weighted
    )

    return center


def get_strategy_center(image: np.ndarray, strategy: str):
    if strategy == "fixed_center":
        return np.asarray(image.shape, dtype=float) / 2.0

    if strategy == "hybrid_image_center":
        return hybrid_image_center(image)

    if strategy == "intensity_weighted_center":
        return intensity_weighted_center(image)

    if strategy == "canal_axis_image_center":
        return canal_axis_image_center(image)

    if strategy == "canal_preserving_hybrid":
        return canal_preserving_hybrid(image)

    raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# RETENTION METRICS
# =============================================================================

def class_voxel_counts(mask: np.ndarray):
    return {
        class_id: int(np.sum(mask == class_id))
        for class_id in CLASS_NAMES
    }


def retention_metrics(mask: np.ndarray, center):
    cropped = crop_volume(mask, center, PATCH_SIZE)

    total = max(int(np.sum(mask > 0)), 1)

    counts = class_voxel_counts(mask)
    cropped_counts = class_voxel_counts(cropped)

    result = {}

    foreground = int(np.sum(cropped > 0)) / total

    result["foreground_retention"] = float(foreground)

    for class_id, name in CLASS_NAMES.items():
        denom = max(counts[class_id], 1)
        result[f"{name.lower().replace(' ', '_')}_retention"] = (
            float(cropped_counts[class_id] / denom)
            if counts[class_id] > 0
            else 0.0
        )

    values = [
        result[f"{name.lower().replace(' ', '_')}_retention"]
        for name in CLASS_NAMES.values()
    ]

    result["combined_retention"] = float(np.mean(values))
    result["minimum_class_retention"] = float(np.min(values))

    return result


# =============================================================================
# LOCALIZATION ERROR
# =============================================================================

def centroid(mask: np.ndarray, class_id: int):
    coords = np.argwhere(mask == class_id)

    if len(coords) == 0:
        return None

    return np.mean(coords, axis=0)


def center_error_to_canal(mask: np.ndarray, center):
    canal = centroid(mask, 2)

    if canal is None:
        return float("nan")

    center = np.asarray(center, dtype=float)

    return float(np.linalg.norm(center - canal))


# =============================================================================
# CASE PROCESSING
# =============================================================================

def find_matching_cases():
    images = {
        p.stem: p
        for p in IMAGE_DIR.glob("*.mha")
    }

    masks = {
        p.stem: p
        for p in MASK_DIR.glob("*.mha")
    }

    common = sorted(set(images) & set(masks))

    return images, masks, common


def process_case(case_id: str, image_path: Path, mask_path: Path):
    image = read_mha(image_path)
    mask = read_mha(mask_path)

    if image.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch for {case_id}: "
            f"image={image.shape}, mask={mask.shape}"
        )

    image_norm = normalize_image(image)

    rows = []

    for strategy in STRATEGIES:
        center = get_strategy_center(image_norm, strategy)
        center = clamp_center(center, image.shape, PATCH_SIZE)

        metrics = retention_metrics(mask, center)

        error = center_error_to_canal(mask, center)

        row = {
            "file": case_id,
            "strategy": strategy,
            "center_z": float(center[0]),
            "center_y": float(center[1]),
            "center_x": float(center[2]),
            "localization_error_to_gt_canal_voxels": error,
            **metrics,
        }

        rows.append(row)

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()

    banner("PHASE 3 - PART 35")
    print("FULL TEST-SET CANAL-AWARE IMAGE-ONLY LOCALIZATION EVALUATION")

    section("PROJECT PATHS")
    print(f"PROJECT ROOT : {PROJECT_ROOT}")
    print(f"TEST DIRECTORY : {TEST_DIR}")
    print(f"IMAGE DIRECTORY : {IMAGE_DIR}")
    print(f"MASK DIRECTORY : {MASK_DIR}")
    print(f"PART 33 DIRECTORY : {PART33_DIR}")
    print(f"PART 34 DIRECTORY : {PART34_DIR}")
    print(f"OUTPUT DIRECTORY : {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("EXPERIMENT RULES")
    print("Cases evaluated: COMPLETE TEST SET")
    print("Ground-truth used for localization: NO")
    print("Ground-truth used for evaluation: YES")
    print("Training performed: NO")
    print("Model weights modified: NO")
    print(f"Patch size: {PATCH_SIZE}")

    section("IMAGE-ONLY STRATEGIES")
    for strategy in STRATEGIES:
        print(f"✓ {strategy}")

    images, masks, common = find_matching_cases()

    section("TEST DATASET")
    print(f"Image files : {len(images)}")
    print(f"Mask files  : {len(masks)}")
    print(f"Matching    : {len(common)}")
    print(f"Missing masks: {len(set(images) - set(masks))}")
    print(f"Orphan masks : {len(set(masks) - set(images))}")

    if not common:
        raise RuntimeError("No matching image/mask cases found.")

    all_rows = []

    section("RUNNING FULL 71-CASE LOCALIZATION EVALUATION")

    for index, case_id in enumerate(common, start=1):
        print(f"Analyzing {index:3d} / {len(common)} : {case_id}")

        try:
            rows = process_case(
                case_id,
                images[case_id],
                masks[case_id],
            )
            all_rows.extend(rows)

        except Exception as exc:
            print(f"  WARNING: failed case {case_id}: {exc}")

    df = pd.DataFrame(all_rows)

    if df.empty:
        raise RuntimeError("No case results were produced.")

    # -------------------------------------------------------------------------
    # Strategy summary
    # -------------------------------------------------------------------------

    summary_rows = []

    for strategy in STRATEGIES:
        part = df[df["strategy"] == strategy].copy()

        summary_rows.append({
            "strategy": strategy,
            "cases": int(len(part)),
            "mean_foreground_retention": float(
                part["foreground_retention"].mean()
            ),
            "mean_vertebrae_retention": float(
                part["vertebrae_retention"].mean()
            ),
            "mean_spinal_canal_retention": float(
                part["spinal_canal_retention"].mean()
            ),
            "mean_disc_retention": float(
                part["intervertebral_disc_retention"].mean()
            ),
            "mean_combined_retention": float(
                part["combined_retention"].mean()
            ),
            "median_combined_retention": float(
                part["combined_retention"].median()
            ),
            "mean_minimum_class_retention": float(
                part["minimum_class_retention"].mean()
            ),
            "median_minimum_class_retention": float(
                part["minimum_class_retention"].median()
            ),
            "mean_canal_localization_error": float(
                part["localization_error_to_gt_canal_voxels"].mean()
            ),
            "median_canal_localization_error": float(
                part["localization_error_to_gt_canal_voxels"].median()
            ),
            "cases_canal_below_0.02": int(
                (part["spinal_canal_retention"] < 0.02).sum()
            ),
            "cases_canal_below_0.05": int(
                (part["spinal_canal_retention"] < 0.05).sum()
            ),
            "cases_canal_above_0.20": int(
                (part["spinal_canal_retention"] >= 0.20).sum()
            ),
            "cases_combined_above_0.20": int(
                (part["combined_retention"] >= 0.20).sum()
            ),
        })

    summary = pd.DataFrame(summary_rows)

    # Primary ranking:
    # 1. canal retention
    # 2. minimum class retention
    # 3. combined retention
    summary = summary.sort_values(
        [
            "mean_spinal_canal_retention",
            "mean_minimum_class_retention",
            "mean_combined_retention",
        ],
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(0, "rank", np.arange(1, len(summary) + 1))

    # -------------------------------------------------------------------------
    # Per-case best strategies
    # -------------------------------------------------------------------------

    best_rows = []

    for case_id, part in df.groupby("file"):
        # Primary criterion: spinal canal retention.
        # Secondary: minimum class retention.
        # Tertiary: combined retention.
        ranked = part.sort_values(
            [
                "spinal_canal_retention",
                "minimum_class_retention",
                "combined_retention",
            ],
            ascending=False,
        )

        best = ranked.iloc[0]

        fixed = part[part["strategy"] == "fixed_center"].iloc[0]

        best_rows.append({
            "file": case_id,
            "fixed_canal_retention": float(
                fixed["spinal_canal_retention"]
            ),
            "best_strategy": best["strategy"],
            "best_canal_retention": float(
                best["spinal_canal_retention"]
            ),
            "canal_improvement": float(
                best["spinal_canal_retention"]
                - fixed["spinal_canal_retention"]
            ),
            "fixed_combined_retention": float(
                fixed["combined_retention"]
            ),
            "best_combined_retention": float(
                best["combined_retention"]
            ),
            "fixed_minimum_class_retention": float(
                fixed["minimum_class_retention"]
            ),
            "best_minimum_class_retention": float(
                best["minimum_class_retention"]
            ),
            "best_localization_error": float(
                best["localization_error_to_gt_canal_voxels"]
            ),
        })

    per_case_best = pd.DataFrame(best_rows)

    # -------------------------------------------------------------------------
    # Baseline comparison
    # -------------------------------------------------------------------------

    fixed_summary = summary[
        summary["strategy"] == "fixed_center"
    ].iloc[0]

    canal_aware_candidates = summary[
        summary["strategy"].isin(
            [
                "canal_axis_image_center",
                "canal_preserving_hybrid",
            ]
        )
    ].copy()

    best_canal_aware = canal_aware_candidates.iloc[0]

    best_overall = summary.iloc[0]

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    section("STRATEGY PERFORMANCE SUMMARY")

    display_columns = [
        "rank",
        "strategy",
        "cases",
        "mean_foreground_retention",
        "mean_vertebrae_retention",
        "mean_spinal_canal_retention",
        "mean_disc_retention",
        "mean_combined_retention",
        "mean_minimum_class_retention",
        "cases_canal_below_0.02",
        "cases_canal_above_0.20",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    section("BEST CANAL-PRESERVING STRATEGY")

    print(f"Best strategy overall by canal retention:")
    print(f"  {best_overall['strategy']}")
    print(
        f"Mean canal retention: "
        f"{best_overall['mean_spinal_canal_retention']:.6f}"
    )

    print()
    print("Best canal-aware candidate:")
    print(f"  {best_canal_aware['strategy']}")
    print(
        f"Mean canal retention: "
        f"{best_canal_aware['mean_spinal_canal_retention']:.6f}"
    )

    fixed_canal = float(fixed_summary["mean_spinal_canal_retention"])

    print()
    print(f"Fixed-center canal retention: {fixed_canal:.6f}")
    print(
        "Canal-aware improvement: "
        f"+{best_canal_aware['mean_spinal_canal_retention'] - fixed_canal:.6f}"
    )

    section("CANAL RETENTION COMPARISON")

    canal_table = summary[
        [
            "rank",
            "strategy",
            "mean_spinal_canal_retention",
            "median_combined_retention",
            "cases_canal_below_0.02",
            "cases_canal_above_0.20",
        ]
    ]

    print(
        canal_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    section("PER-CASE BEST CANAL STRATEGY")

    print(
        per_case_best[
            [
                "file",
                "best_strategy",
                "fixed_canal_retention",
                "best_canal_retention",
                "canal_improvement",
            ]
        ].sort_values(
            "canal_improvement",
            ascending=False,
        ).head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    improved_cases = int(
        (per_case_best["canal_improvement"] > 0.0).sum()
    )

    worsened_cases = int(
        (per_case_best["canal_improvement"] < 0.0).sum()
    )

    section("FIXED CENTER VS BEST STRATEGY")

    print(f"Improved cases : {improved_cases}")
    print(
        f"Unchanged cases: "
        f"{len(per_case_best) - improved_cases - worsened_cases}"
    )
    print(f"Worsened cases : {worsened_cases}")

    print(
        f"Mean fixed canal retention: "
        f"{per_case_best['fixed_canal_retention'].mean():.6f}"
    )

    print(
        f"Mean best canal retention: "
        f"{per_case_best['best_canal_retention'].mean():.6f}"
    )

    print(
        f"Mean improvement: "
        f"{per_case_best['canal_improvement'].mean():+.6f}"
    )

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------

    section("SAVING ANALYSIS TABLES")

    case_csv = OUTPUT_DIR / "part35_case_analysis.csv"
    summary_csv = OUTPUT_DIR / "part35_strategy_summary.csv"
    best_csv = OUTPUT_DIR / "part35_per_case_best_strategy.csv"

    df.to_csv(case_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    per_case_best.to_csv(best_csv, index=False)

    print(f"Saved: {case_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {best_csv}")

    # -------------------------------------------------------------------------
    # Charts
    # -------------------------------------------------------------------------

    section("CREATING SUMMARY CHARTS")

    # 1. Canal retention
    plt.figure(figsize=(11, 6))
    plt.bar(
        summary["strategy"],
        summary["mean_spinal_canal_retention"],
    )
    plt.ylabel("Mean Spinal Canal Retention")
    plt.xlabel("Strategy")
    plt.title("Part 35 - Spinal Canal Retention by Localization Strategy")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    chart = OUTPUT_DIR / "part35_canal_retention_comparison.png"
    plt.savefig(chart, dpi=200)
    plt.close()
    print(f"Saved: {chart}")

    # 2. Combined retention
    plt.figure(figsize=(11, 6))
    plt.bar(
        summary["strategy"],
        summary["mean_combined_retention"],
    )
    plt.ylabel("Mean Combined Anatomical Retention")
    plt.xlabel("Strategy")
    plt.title("Part 35 - Combined Anatomical Retention")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    chart = OUTPUT_DIR / "part35_combined_retention_comparison.png"
    plt.savefig(chart, dpi=200)
    plt.close()
    print(f"Saved: {chart}")

    # 3. Balanced retention
    plt.figure(figsize=(11, 6))
    plt.bar(
        summary["strategy"],
        summary["mean_minimum_class_retention"],
    )
    plt.ylabel("Mean Minimum-Class Retention")
    plt.xlabel("Strategy")
    plt.title("Part 35 - Balanced Anatomical Retention")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    chart = OUTPUT_DIR / "part35_balanced_retention_comparison.png"
    plt.savefig(chart, dpi=200)
    plt.close()
    print(f"Saved: {chart}")

    # 4. Localization error
    plt.figure(figsize=(11, 6))
    plt.bar(
        summary["strategy"],
        summary["mean_canal_localization_error"],
    )
    plt.ylabel("Mean Canal Localization Error (voxels)")
    plt.xlabel("Strategy")
    plt.title("Part 35 - Image-Only Canal Localization Error")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    chart = OUTPUT_DIR / "part35_localization_error_comparison.png"
    plt.savefig(chart, dpi=200)
    plt.close()
    print(f"Saved: {chart}")

    # 5. Canal failure count
    plt.figure(figsize=(11, 6))
    plt.bar(
        summary["strategy"],
        summary["cases_canal_below_0.02"],
    )
    plt.ylabel("Number of Cases")
    plt.xlabel("Strategy")
    plt.title("Part 35 - Severe Canal Retention Failures (< 0.02)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    chart = OUTPUT_DIR / "part35_canal_failure_count.png"
    plt.savefig(chart, dpi=200)
    plt.close()
    print(f"Saved: {chart}")

    # -------------------------------------------------------------------------
    # JSON summary
    # -------------------------------------------------------------------------

    elapsed = time.time() - start_time

    summary_json = {
        "phase": "Phase 3",
        "part": 35,
        "title": "Full Test-Set Canal-Aware Image-Only Localization Evaluation",
        "cases_analyzed": int(len(common)),
        "strategies": STRATEGIES,
        "patch_size": list(PATCH_SIZE),
        "ground_truth_used_for_localization": False,
        "ground_truth_used_for_evaluation": True,
        "training_performed": False,
        "model_weights_modified": False,
        "fixed_center_mean_canal_retention": float(
            fixed_summary["mean_spinal_canal_retention"]
        ),
        "best_strategy": str(best_overall["strategy"]),
        "best_strategy_mean_canal_retention": float(
            best_overall["mean_spinal_canal_retention"]
        ),
        "best_strategy_mean_combined_retention": float(
            best_overall["mean_combined_retention"]
        ),
        "best_canal_aware_strategy": str(best_canal_aware["strategy"]),
        "best_canal_aware_mean_canal_retention": float(
            best_canal_aware["mean_spinal_canal_retention"]
        ),
        "best_canal_aware_improvement_over_fixed": float(
            best_canal_aware["mean_spinal_canal_retention"] - fixed_canal
        ),
        "improved_cases_vs_fixed": improved_cases,
        "worsened_cases_vs_fixed": worsened_cases,
        "mean_best_case_canal_retention": float(
            per_case_best["best_canal_retention"].mean()
        ),
        "mean_casewise_canal_improvement": float(
            per_case_best["canal_improvement"].mean()
        ),
        "execution_time_minutes": float(elapsed / 60.0),
    }

    json_path = OUTPUT_DIR / "phase3_part35_summary.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)

    print(f"Saved: {json_path}")

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    report_path = OUTPUT_DIR / "phase3_part35_report.txt"

    report = f"""
PHASE 3 - PART 35
FULL TEST-SET CANAL-AWARE IMAGE-ONLY LOCALIZATION EVALUATION
======================================================================

Purpose
-------
Evaluate canal-aware image-only localization strategies on the complete
test set before any additional Swin-UNETR training.

Dataset
-------
Test cases analyzed: {len(common)}
Patch size: {PATCH_SIZE}

Ground-truth used for localization: NO
Ground-truth used for evaluation: YES
Training performed: NO
Model weights modified: NO

Strategy ranking
----------------
{summary[['rank', 'strategy', 'mean_spinal_canal_retention',
           'mean_combined_retention', 'mean_minimum_class_retention',
           'mean_canal_localization_error',
           'cases_canal_below_0.02']].to_string(index=False)}

Primary result
--------------
Best strategy by mean spinal-canal retention:
{best_overall['strategy']}

Mean canal retention:
{best_overall['mean_spinal_canal_retention']:.6f}

Fixed-center mean canal retention:
{fixed_canal:.6f}

Improvement:
{best_overall['mean_spinal_canal_retention'] - fixed_canal:+.6f}

Canal-aware candidate
---------------------
{best_canal_aware['strategy']}

Mean canal retention:
{best_canal_aware['mean_spinal_canal_retention']:.6f}

Improvement over fixed center:
{best_canal_aware['mean_spinal_canal_retention'] - fixed_canal:+.6f}

Per-case comparison
-------------------
Improved cases versus fixed center: {improved_cases}
Worsened cases versus fixed center: {worsened_cases}

Mean case-wise canal improvement:
{per_case_best['canal_improvement'].mean():+.6f}

Interpretation
--------------
This experiment evaluates localization only. It does not establish that
canal-aware localization improves final segmentation performance.

A training experiment should only be initiated after selecting the strategy
from this full test-set localization analysis and documenting the trade-off
between spinal-canal retention and balanced anatomical retention.

Execution time
--------------
{elapsed / 60.0:.2f} minutes
"""

    report_path.write_text(report.strip() + "\n", encoding="utf-8")
    print(f"Saved: {report_path}")

    section("PART 35 COMPLETE")

    print(f"Cases analyzed              : {len(common)}")
    print(
        f"Best strategy               : "
        f"{best_overall['strategy']}"
    )
    print(
        f"Best mean canal retention   : "
        f"{best_overall['mean_spinal_canal_retention']:.6f}"
    )
    print(
        f"Fixed mean canal retention  : "
        f"{fixed_canal:.6f}"
    )
    print(
        f"Mean improvement            : "
        f"{best_overall['mean_spinal_canal_retention'] - fixed_canal:+.6f}"
    )
    print(f"Improved cases              : {improved_cases}")
    print(f"Worsened cases              : {worsened_cases}")
    print()
    print("Ground-truth used for localization: NO")
    print("Ground-truth used for evaluation: YES")
    print("Training performed: NO")
    print("Model weights modified: NO")

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)
    print()
    print("PHASE 3 - PART 35 COMPLETE")


if __name__ == "__main__":
    main()
