"""
PHASE 3 - PART 33
HIGH-RISK SPINAL-CANAL CASE DEEP ANALYSIS

Purpose
-------
Part 32 identified 14 high-risk test cases where image-only localization
still provides poor spinal-canal retention.

Part 33 performs a deeper, evaluation-only analysis of those cases:
1. Loads the official Part 32 high-risk cases.
2. Loads Part 32 per-case strategy analysis.
3. Re-evaluates the selected image-only localization strategies.
4. Measures anatomical retention for all available strategies.
5. Identifies the best strategy per high-risk case.
6. Quantifies how much the best strategy improves over fixed-center.
7. Groups failures by sequence.
8. Creates charts, CSV files, JSON summary and a text report.

IMPORTANT
---------
- Ground truth is NEVER used to choose a crop/localization.
- Ground truth is used ONLY for evaluation/retention measurements.
- No model training is performed.
- No model weights are modified.
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

PART32_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part32_spinal_canal_aware_localization"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "part33_high_risk_spinal_canal_analysis"
)

PATCH_SIZE = (96, 96, 96)

CLASS_NAMES = {
    0: "Background",
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
    "multi_candidate_canal_center",
]

HIGH_RISK_THRESHOLD = 0.20


# ============================================================================
# UTILITIES
# ============================================================================

def banner(text: str) -> None:
    print("=" * 78)
    print(text)
    print("=" * 78)


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isfinite(value):
            return value
        return default
    except Exception:
        return default


def case_id_from_path(path: Path) -> str:
    return path.stem


def sequence_from_case(case_id: str) -> str:
    name = case_id.upper()

    if "T2_SPACE" in name:
        return "T2 SPACE"
    if "_T2" in name:
        return "T2"
    if "_T1" in name:
        return "T1"

    return "OTHER"


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Robust 1st-99th percentile normalization.
    This is used only for image-based localization.
    """
    image = np.asarray(image, dtype=np.float32)

    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    valid = image[finite]

    lo = np.percentile(valid, 1)
    hi = np.percentile(valid, 99)

    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max())

    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    out = (image - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.0

    return out.astype(np.float32)


def load_mha(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    # SimpleITK returns [z, y, x].
    return np.asarray(array)


def find_matching_image(case_id: str) -> Path | None:
    candidates = [
        IMAGE_DIR / f"{case_id}.mha",
        IMAGE_DIR / f"{case_id}.nii.gz",
        IMAGE_DIR / f"{case_id}.nii",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def find_matching_mask(case_id: str) -> Path | None:
    candidates = [
        MASK_DIR / f"{case_id}.mha",
        MASK_DIR / f"{case_id}.nii.gz",
        MASK_DIR / f"{case_id}.nii",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


# ============================================================================
# IMAGE-ONLY LOCALIZATION
# ============================================================================

def center_crop_pad(
    volume: np.ndarray,
    center_xyz: tuple[float, float, float],
    patch_size=PATCH_SIZE,
) -> np.ndarray:
    """
    Crop/pad a 3-D array around an image-derived center.

    Input volume convention:
        [z, y, x]

    center_xyz:
        (x, y, z)
    """
    target_z, target_y, target_x = patch_size

    zc = int(round(center_xyz[2]))
    yc = int(round(center_xyz[1]))
    xc = int(round(center_xyz[0]))

    z0 = zc - target_z // 2
    y0 = yc - target_y // 2
    x0 = xc - target_x // 2

    z1 = z0 + target_z
    y1 = y0 + target_y
    x1 = x0 + target_x

    out = np.zeros(
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

    dst_z1 = dst_z0 + (src_z1 - src_z0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if (
        src_z1 > src_z0
        and src_y1 > src_y0
        and src_x1 > src_x0
    ):
        out[
            dst_z0:dst_z1,
            dst_y0:dst_y1,
            dst_x0:dst_x1,
        ] = volume[
            src_z0:src_z1,
            src_y0:src_y1,
            src_x0:src_x1,
        ]

    return out


def fixed_center(image: np.ndarray) -> tuple[float, float, float]:
    z, y, x = image.shape
    return (x / 2.0, y / 2.0, z / 2.0)


def image_nonzero_center(image: np.ndarray) -> tuple[float, float, float]:
    norm = normalize_image(image)
    coords = np.argwhere(norm > 0.03)

    if len(coords) == 0:
        return fixed_center(image)

    zc, yc, xc = np.median(coords, axis=0)
    return (float(xc), float(yc), float(zc))


def robust_intensity_center(image: np.ndarray) -> tuple[float, float, float]:
    norm = normalize_image(image)

    threshold = np.percentile(norm[norm > 0], 60) if np.any(norm > 0) else 0.0
    coords = np.argwhere(norm >= threshold)

    if len(coords) == 0:
        return fixed_center(image)

    zc, yc, xc = np.median(coords, axis=0)
    return (float(xc), float(yc), float(zc))


def intensity_weighted_center(image: np.ndarray) -> tuple[float, float, float]:
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

    zc = float((z_axis * wz).sum() / total)
    yc = float((y_axis * wy).sum() / total)
    xc = float((x_axis * wx).sum() / total)

    return (xc, yc, zc)


def canal_axis_image_center(image: np.ndarray) -> tuple[float, float, float]:
    """
    Image-only approximation of the spinal canal axis.

    The method uses robust intensity information and gives additional
    importance to the central x/y region. It does NOT inspect the mask.
    """
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

    z_profile = norm[:, int(y_idx), int(x_idx)]

    if np.sum(z_profile) > 0:
        z_idx = float(
            np.sum(np.arange(z) * z_profile)
            / np.sum(z_profile)
        )
    else:
        z_idx = z / 2.0

    return (float(x_idx), float(y_idx), float(z_idx))


def canal_preserving_hybrid(image: np.ndarray) -> tuple[float, float, float]:
    """
    Image-only hybrid center emphasizing:
    - robust image center
    - central anatomical prior
    - canal-axis estimate
    """
    a = np.asarray(hybrid_image_center(image))
    b = np.asarray(canal_axis_image_center(image))

    center = np.asarray(fixed_center(image))

    # More weight to the hybrid anatomical center, with a canal-axis
    # correction. No ground truth is consulted.
    result = 0.55 * a + 0.35 * b + 0.10 * center

    return tuple(float(v) for v in result)


def hybrid_image_center(image: np.ndarray) -> tuple[float, float, float]:
    """
    Image-only hybrid localization.
    """
    a = np.asarray(image_nonzero_center(image))
    b = np.asarray(robust_intensity_center(image))
    c = np.asarray(fixed_center(image))

    result = 0.45 * a + 0.40 * b + 0.15 * c

    return tuple(float(v) for v in result)


def multi_candidate_canal_center(image: np.ndarray) -> tuple[float, float, float]:
    """
    Deterministic multi-candidate image-only localization.

    Candidate centers are generated from several image-derived estimates,
    then selected using an image-only anatomical score.
    """
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

        # Image-only score: high intensity near the candidate center,
        # plus preference for staying inside the volume.
        cx_i = int(np.clip(round(cx), 0, x - 1))
        cy_i = int(np.clip(round(cy), 0, y - 1))
        cz_i = int(np.clip(round(cz), 0, z - 1))

        z0 = max(0, cz_i - 12)
        z1 = min(z, cz_i + 13)
        y0 = max(0, cy_i - 12)
        y1 = min(y, cy_i + 13)
        x0 = max(0, cx_i - 12)
        x1 = min(x, cx_i + 13)

        local_mean = float(
            norm[z0:z1, y0:y1, x0:x1].mean()
        )

        boundary_penalty = 0.0

        if cx < PATCH_SIZE[2] / 2 or cx > x - PATCH_SIZE[2] / 2:
            boundary_penalty += 0.05
        if cy < PATCH_SIZE[1] / 2 or cy > y - PATCH_SIZE[1] / 2:
            boundary_penalty += 0.05
        if cz < PATCH_SIZE[0] / 2 or cz > z - PATCH_SIZE[0] / 2:
            boundary_penalty += 0.05

        score = local_mean - boundary_penalty
        scored.append((score, center))

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
# RETENTION / ANATOMICAL ANALYSIS
# ============================================================================

def class_voxels(mask: np.ndarray, label: int) -> int:
    return int(np.count_nonzero(mask == label))


def retention_for_class(
    original_mask: np.ndarray,
    cropped_mask: np.ndarray,
    label: int,
) -> float:
    total = class_voxels(original_mask, label)

    if total <= 0:
        return np.nan

    retained = class_voxels(cropped_mask, label)

    return float(retained / total)


def evaluate_crop(
    original_mask: np.ndarray,
    center: tuple[float, float, float],
) -> dict:
    cropped_mask = center_crop_pad(
        original_mask,
        center,
        PATCH_SIZE,
    )

    foreground_total = np.count_nonzero(original_mask > 0)
    foreground_retained = np.count_nonzero(cropped_mask > 0)

    if foreground_total > 0:
        foreground_retention = foreground_retained / foreground_total
    else:
        foreground_retention = np.nan

    vertebra_retention = retention_for_class(
        original_mask,
        cropped_mask,
        1,
    )

    canal_retention = retention_for_class(
        original_mask,
        cropped_mask,
        2,
    )

    disc_retention = retention_for_class(
        original_mask,
        cropped_mask,
        3,
    )

    values = [
        vertebra_retention,
        canal_retention,
        disc_retention,
    ]

    finite_values = [v for v in values if np.isfinite(v)]

    combined = (
        float(np.mean(finite_values))
        if finite_values
        else np.nan
    )

    minimum = (
        float(np.min(finite_values))
        if finite_values
        else np.nan
    )

    return {
        "foreground_retention": float(foreground_retention),
        "vertebrae_retention": float(vertebra_retention),
        "spinal_canal_retention": float(canal_retention),
        "disc_retention": float(disc_retention),
        "combined_retention": combined,
        "minimum_class_retention": minimum,
    }


def localization_error(
    center_a: tuple[float, float, float],
    center_b: tuple[float, float, float],
) -> float:
    a = np.asarray(center_a, dtype=np.float64)
    b = np.asarray(center_b, dtype=np.float64)

    return float(np.linalg.norm(a - b))


# ============================================================================
# LOAD PART 32 RESULTS
# ============================================================================

def load_part32_high_risk() -> pd.DataFrame:
    path = PART32_DIR / "part32_high_risk_cases.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Part 32 high-risk file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "file" not in df.columns:
        raise ValueError(
            "Part 32 high-risk CSV does not contain a 'file' column."
        )

    return df


def load_part32_case_analysis() -> pd.DataFrame:
    path = PART32_DIR / "part32_case_analysis.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Part 32 case-analysis file not found:\n{path}"
        )

    return pd.read_csv(path)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main() -> None:
    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    banner("PHASE 3 - PART 33")
    print("HIGH-RISK SPINAL-CANAL CASE DEEP ANALYSIS")
    banner("")

    print("PROJECT ROOT")
    print(PROJECT_ROOT)
    print()

    print("TEST DIRECTORY")
    print(TEST_DIR)
    print()

    print("PART 32 DIRECTORY")
    print(PART32_DIR)
    print()

    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)
    print()

    # ------------------------------------------------------------------
    banner("LOADING PART 32 RESULTS")

    high_risk_df = load_part32_high_risk()
    part32_case_df = load_part32_case_analysis()

    high_risk_cases = high_risk_df["file"].astype(str).tolist()

    print(f"Part 32 high-risk cases: {len(high_risk_cases)}")
    print(f"Part 32 case-analysis rows: {len(part32_case_df)}")

    # ------------------------------------------------------------------
    banner("HIGH-RISK CASES")

    print(", ".join(high_risk_cases))
    print()

    # ------------------------------------------------------------------
    banner("IMAGE / MASK MATCHING")

    image_files = {
        p.stem: p
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".mha", ".nii", ".gz"}
    }

    mask_files = {
        p.stem: p
        for p in MASK_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".mha", ".nii", ".gz"}
    }

    available = [
        case
        for case in high_risk_cases
        if case in image_files and case in mask_files
    ]

    print(f"High-risk cases: {len(high_risk_cases)}")
    print(f"Matched image/mask cases: {len(available)}")
    print(f"Missing cases: {len(high_risk_cases) - len(available)}")

    if not available:
        raise RuntimeError(
            "No high-risk cases have matching image and mask files."
        )

    # ------------------------------------------------------------------
    banner("RUNNING HIGH-RISK STRATEGY ANALYSIS")

    rows = []

    for index, case_id in enumerate(available, start=1):
        print(
            f"Analyzing {index:2d} / {len(available)} : {case_id}"
        )

        image = load_mha(image_files[case_id])
        mask = load_mha(mask_files[case_id])

        if image.shape != mask.shape:
            print(
                f"WARNING: shape mismatch for {case_id}: "
                f"{image.shape} vs {mask.shape}"
            )
            continue

        fixed = fixed_center(image)

        for strategy in STRATEGIES:
            center = LOCALIZATION_FUNCTIONS[strategy](image)

            metrics = evaluate_crop(mask, center)

            row = {
                "file": case_id,
                "sequence": sequence_from_case(case_id),
                "strategy": strategy,
                "image_shape": str(tuple(image.shape)),
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "localization_error_from_fixed_voxels": localization_error(
                    center,
                    fixed,
                ),
                **metrics,
            }

            rows.append(row)

    case_df = pd.DataFrame(rows)

    if case_df.empty:
        raise RuntimeError("No high-risk case analysis was produced.")

    # ------------------------------------------------------------------
    banner("HIGH-RISK STRATEGY SUMMARY")

    summary = (
        case_df
        .groupby("strategy", as_index=False)
        .agg(
            cases=("file", "nunique"),
            mean_localization_error_voxels=(
                "localization_error_from_fixed_voxels",
                "mean",
            ),
            median_localization_error_voxels=(
                "localization_error_from_fixed_voxels",
                "median",
            ),
            mean_foreground_retention=(
                "foreground_retention",
                "mean",
            ),
            mean_vertebrae_retention=(
                "vertebrae_retention",
                "mean",
            ),
            mean_spinal_canal_retention=(
                "spinal_canal_retention",
                "mean",
            ),
            mean_disc_retention=(
                "disc_retention",
                "mean",
            ),
            mean_combined_retention=(
                "combined_retention",
                "mean",
            ),
            mean_minimum_class_retention=(
                "minimum_class_retention",
                "mean",
            ),
        )
    )

    summary = summary.sort_values(
        "mean_spinal_canal_retention",
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(0, "rank", np.arange(1, len(summary) + 1))

    print(summary.to_string(index=False))

    # ------------------------------------------------------------------
    banner("BEST STRATEGY FOR EACH HIGH-RISK CASE")

    best_rows = []

    for case_id, group in case_df.groupby("file"):
        group = group.sort_values(
            ["spinal_canal_retention", "combined_retention"],
            ascending=False,
        )

        best = group.iloc[0]

        fixed_row = group[
            group["strategy"] == "fixed_center"
        ]

        if len(fixed_row):
            fixed_row = fixed_row.iloc[0]
            fixed_canal = safe_float(
                fixed_row["spinal_canal_retention"]
            )
            fixed_combined = safe_float(
                fixed_row["combined_retention"]
            )
        else:
            fixed_canal = np.nan
            fixed_combined = np.nan

        best_canal = safe_float(best["spinal_canal_retention"])
        best_combined = safe_float(best["combined_retention"])

        best_rows.append(
            {
                "file": case_id,
                "sequence": best["sequence"],
                "best_strategy": best["strategy"],
                "best_spinal_canal_retention": best_canal,
                "fixed_spinal_canal_retention": fixed_canal,
                "canal_retention_improvement": (
                    best_canal - fixed_canal
                    if np.isfinite(fixed_canal)
                    else np.nan
                ),
                "best_combined_retention": best_combined,
                "fixed_combined_retention": fixed_combined,
                "combined_retention_improvement": (
                    best_combined - fixed_combined
                    if np.isfinite(fixed_combined)
                    else np.nan
                ),
                "best_localization_error_from_fixed": safe_float(
                    best["localization_error_from_fixed_voxels"]
                ),
            }
        )

    best_case_df = pd.DataFrame(best_rows)

    print(best_case_df.to_string(index=False))

    # ------------------------------------------------------------------
    banner("HIGH-RISK FAILURE CATEGORIES")

    categories = []

    for _, row in best_case_df.iterrows():
        canal = safe_float(row["best_spinal_canal_retention"])
        improvement = safe_float(
            row["canal_retention_improvement"]
        )

        if canal < 0.02:
            category = "Critical canal retention"
        elif canal < 0.10:
            category = "Severe canal retention"
        elif canal < 0.20:
            category = "Moderate canal retention"
        elif improvement > 0.05:
            category = "Improved by localization"
        else:
            category = "Residual high-risk case"

        categories.append(category)

    best_case_df["failure_category"] = categories

    category_summary = (
        best_case_df
        .groupby("failure_category")
        .size()
        .reset_index(name="cases")
        .sort_values("cases", ascending=False)
    )

    print(category_summary.to_string(index=False))

    # ------------------------------------------------------------------
    banner("SEQUENCE-WISE HIGH-RISK ANALYSIS")

    sequence_summary = (
        best_case_df
        .groupby("sequence", as_index=False)
        .agg(
            cases=("file", "count"),
            mean_best_canal_retention=(
                "best_spinal_canal_retention",
                "mean",
            ),
            mean_fixed_canal_retention=(
                "fixed_spinal_canal_retention",
                "mean",
            ),
            mean_canal_improvement=(
                "canal_retention_improvement",
                "mean",
            ),
            mean_best_combined_retention=(
                "best_combined_retention",
                "mean",
            ),
            mean_combined_improvement=(
                "combined_retention_improvement",
                "mean",
            ),
        )
    )

    print(sequence_summary.to_string(index=False))

    # ------------------------------------------------------------------
    banner("BEST OVERALL HIGH-RISK STRATEGY")

    best_strategy_row = (
        summary
        .sort_values(
            "mean_spinal_canal_retention",
            ascending=False,
        )
        .iloc[0]
    )

    best_strategy = str(best_strategy_row["strategy"])
    best_canal = float(
        best_strategy_row["mean_spinal_canal_retention"]
    )

    fixed_strategy_row = summary[
        summary["strategy"] == "fixed_center"
    ]

    if len(fixed_strategy_row):
        fixed_canal = float(
            fixed_strategy_row.iloc[0][
                "mean_spinal_canal_retention"
            ]
        )
    else:
        fixed_canal = np.nan

    print(f"Best high-risk canal strategy : {best_strategy}")
    print(f"Mean canal retention          : {best_canal:.6f}")

    if np.isfinite(fixed_canal):
        print(
            f"Fixed-center canal retention : {fixed_canal:.6f}"
        )
        print(
            f"Improvement                  : "
            f"{best_canal - fixed_canal:+.6f}"
        )

    # ------------------------------------------------------------------
    banner("SAVING ANALYSIS TABLES")

    case_path = OUTPUT_DIR / "part33_high_risk_case_analysis.csv"
    summary_path = OUTPUT_DIR / "part33_high_risk_strategy_summary.csv"
    best_path = OUTPUT_DIR / "part33_best_strategy_per_case.csv"
    category_path = OUTPUT_DIR / "part33_failure_category_summary.csv"
    sequence_path = OUTPUT_DIR / "part33_sequence_summary.csv"

    case_df.to_csv(case_path, index=False)
    summary.to_csv(summary_path, index=False)
    best_case_df.to_csv(best_path, index=False)
    category_summary.to_csv(category_path, index=False)
    sequence_summary.to_csv(sequence_path, index=False)

    print(f"Saved: {case_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {best_path}")
    print(f"Saved: {category_path}")
    print(f"Saved: {sequence_path}")

    # ------------------------------------------------------------------
    banner("CREATING CHARTS")

    # 1. Canal retention by strategy
    plt.figure(figsize=(11, 6))
    plt.bar(
        summary["strategy"],
        summary["mean_spinal_canal_retention"],
    )
    plt.axhline(
        HIGH_RISK_THRESHOLD,
        linestyle="--",
        linewidth=1.5,
        label="0.20 high-risk threshold",
    )
    plt.ylabel("Mean spinal-canal retention")
    plt.xlabel("Localization strategy")
    plt.title(
        "Part 33 - High-Risk Spinal Canal Retention by Strategy"
    )
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()

    chart1 = OUTPUT_DIR / "part33_high_risk_canal_retention.png"
    plt.savefig(chart1, dpi=200)
    plt.close()
    print(f"Saved: {chart1}")

    # 2. Canal improvement per case
    plot_df = best_case_df.sort_values(
        "canal_retention_improvement"
    )

    plt.figure(figsize=(12, 7))
    plt.barh(
        plot_df["file"],
        plot_df["canal_retention_improvement"],
    )
    plt.axvline(0.0, linewidth=1.0)
    plt.xlabel("Best strategy improvement over fixed center")
    plt.ylabel("High-risk case")
    plt.title(
        "Part 33 - Spinal Canal Retention Improvement per High-Risk Case"
    )
    plt.tight_layout()

    chart2 = OUTPUT_DIR / "part33_casewise_canal_improvement.png"
    plt.savefig(chart2, dpi=200)
    plt.close()
    print(f"Saved: {chart2}")

    # 3. Strategy vs combined retention
    plt.figure(figsize=(11, 6))
    plt.bar(
        summary["strategy"],
        summary["mean_combined_retention"],
    )
    plt.ylabel("Mean combined anatomical retention")
    plt.xlabel("Localization strategy")
    plt.title(
        "Part 33 - High-Risk Combined Anatomical Retention"
    )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    chart3 = OUTPUT_DIR / "part33_combined_retention.png"
    plt.savefig(chart3, dpi=200)
    plt.close()
    print(f"Saved: {chart3}")

    # 4. Sequence-wise canal retention
    if not sequence_summary.empty:
        plt.figure(figsize=(9, 6))
        x = np.arange(len(sequence_summary))

        width = 0.35

        plt.bar(
            x - width / 2,
            sequence_summary["mean_fixed_canal_retention"],
            width,
            label="Fixed center",
        )

        plt.bar(
            x + width / 2,
            sequence_summary["mean_best_canal_retention"],
            width,
            label="Best high-risk strategy",
        )

        plt.xticks(
            x,
            sequence_summary["sequence"],
        )
        plt.ylabel("Mean spinal-canal retention")
        plt.xlabel("Sequence")
        plt.title(
            "Part 33 - High-Risk Canal Retention by Sequence"
        )
        plt.legend()
        plt.tight_layout()

        chart4 = OUTPUT_DIR / "part33_sequence_canal_comparison.png"
        plt.savefig(chart4, dpi=200)
        plt.close()
        print(f"Saved: {chart4}")

    # 5. Failure categories
    plt.figure(figsize=(9, 6))
    plt.bar(
        category_summary["failure_category"],
        category_summary["cases"],
    )
    plt.ylabel("Number of cases")
    plt.xlabel("Failure category")
    plt.title(
        "Part 33 - High-Risk Spinal Canal Failure Categories"
    )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    chart5 = OUTPUT_DIR / "part33_failure_categories.png"
    plt.savefig(chart5, dpi=200)
    plt.close()
    print(f"Saved: {chart5}")

    print(f"Saved: {chart1}")
    print(f"Saved: {chart2}")
    print(f"Saved: {chart3}")

    # ------------------------------------------------------------------
    banner("CREATING FINAL SUMMARY")

    mean_best_canal = float(
        best_case_df["best_spinal_canal_retention"].mean()
    )

    mean_fixed_canal_highrisk = float(
        best_case_df["fixed_spinal_canal_retention"].mean()
    )

    mean_improvement = float(
        best_case_df["canal_retention_improvement"].mean()
    )

    improved_cases = int(
        (best_case_df["canal_retention_improvement"] > 0).sum()
    )

    critical_cases = int(
        (
            best_case_df["best_spinal_canal_retention"] < 0.02
        ).sum()
    )

    severe_cases = int(
        (
            best_case_df["best_spinal_canal_retention"] < 0.10
        ).sum()
    )

    summary_json = {
        "phase": "Phase 3 - Part 33",
        "title": "High-Risk Spinal-Canal Case Deep Analysis",
        "cases_analyzed": int(len(best_case_df)),
        "strategies_analyzed": STRATEGIES,
        "patch_size": list(PATCH_SIZE),
        "ground_truth_used_for_localization": False,
        "ground_truth_used_for_evaluation": True,
        "training_performed": False,
        "model_weights_modified": False,
        "best_high_risk_canal_strategy": best_strategy,
        "best_high_risk_mean_canal_retention": best_canal,
        "fixed_center_mean_canal_retention": fixed_canal,
        "best_vs_fixed_canal_improvement": (
            best_canal - fixed_canal
            if np.isfinite(fixed_canal)
            else None
        ),
        "mean_best_per_case_canal_retention": mean_best_canal,
        "mean_fixed_high_risk_canal_retention": mean_fixed_canal_highrisk,
        "mean_per_case_improvement": mean_improvement,
        "cases_improved_over_fixed": improved_cases,
        "critical_cases_after_best_strategy": critical_cases,
        "severe_cases_after_best_strategy": severe_cases,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "output_directory": str(OUTPUT_DIR),
    }

    json_path = OUTPUT_DIR / "phase3_part33_summary.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            summary_json,
            f,
            indent=4,
        )

    print(f"Saved: {json_path}")

    # ------------------------------------------------------------------
    report_path = OUTPUT_DIR / "phase3_part33_report.txt"

    elapsed = (time.time() - start_time) / 60.0

    report_lines = [
        "=" * 78,
        "PHASE 3 - PART 33",
        "HIGH-RISK SPINAL-CANAL CASE DEEP ANALYSIS",
        "=" * 78,
        "",
        f"Cases analyzed: {len(best_case_df)}",
        f"Best high-risk canal strategy: {best_strategy}",
        f"Best strategy mean canal retention: {best_canal:.6f}",
        f"Fixed-center mean canal retention: {fixed_canal:.6f}",
        f"Best-vs-fixed improvement: "
        f"{best_canal - fixed_canal:+.6f}",
        "",
        f"Mean best-per-case canal retention: "
        f"{mean_best_canal:.6f}",
        f"Mean fixed high-risk canal retention: "
        f"{mean_fixed_canal_highrisk:.6f}",
        f"Mean per-case improvement: "
        f"{mean_improvement:+.6f}",
        f"Cases improved over fixed center: {improved_cases}",
        f"Critical cases after best strategy: {critical_cases}",
        f"Severe cases after best strategy: {severe_cases}",
        "",
        "Interpretation:",
        (
            "Part 33 focuses specifically on the high-risk cases "
            "identified by Part 32. The analysis is evaluation-only. "
            "Ground truth is never used to select the localization "
            "strategy; it is used only to calculate anatomical retention."
        ),
        "",
        "Training performed: NO",
        "Model weights modified: NO",
        "",
        f"Execution time: {elapsed:.2f} minutes",
        "",
        "=" * 78,
        "PHASE 3 - PART 33 COMPLETE",
        "=" * 78,
    ]

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"Saved: {report_path}")

    # ------------------------------------------------------------------
    banner("PART 33 COMPLETE")

    print(f"Cases analyzed                 : {len(best_case_df)}")
    print(f"Best high-risk canal strategy  : {best_strategy}")
    print(f"Best canal retention           : {best_canal:.6f}")
    print(
        f"Fixed-center canal retention   : "
        f"{fixed_canal:.6f}"
    )
    print(
        f"Best vs fixed improvement      : "
        f"{best_canal - fixed_canal:+.6f}"
    )
    print(f"Improved high-risk cases       : {improved_cases}")
    print(f"Critical residual cases        : {critical_cases}")
    print(f"Severe residual cases          : {severe_cases}")
    print(f"Execution time                 : {elapsed:.2f} minutes")

    print()
    print("Ground-truth used for localization: NO")
    print("Ground-truth used for evaluation: YES")
    print("Training performed: NO")
    print("Model weights modified: NO")

    print()
    print("=" * 78)
    print("OUTPUT DIRECTORY")
    print("=" * 78)
    print(OUTPUT_DIR)

    print()
    print("=" * 78)
    print("PHASE 3 - PART 33 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
