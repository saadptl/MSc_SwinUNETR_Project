"""
PHASE 4 - PART 4
RSNA PSEUDO-TARGET ANATOMICAL VALIDATION

RSNA-only research stage.

Purpose
-------
The RSNA 2024 Lumbar Spine Degenerative Classification dataset provides
point annotations, not manually drawn segmentation masks. This script
tests several reproducible point-derived pseudo-target constructions on
real RSNA DICOM images before any Swin-UNETR segmentation training.

IMPORTANT
---------
* RSNA ONLY.
* SPIDER is not used.
* No model is trained.
* No model weights are modified.
* Generated regions are PSEUDO-TARGET CANDIDATES, not manual ground truth.
* This phase is an analysis/design phase.

Candidate strategies
---------------------
1. point_disk
2. gaussian
3. adaptive_disk
4. level_band
5. gaussian_adaptive

Outputs
-------
outputs/segmentation/rsna_part4_point_to_target_design/
    rsna_part5_pseudotarget_case_analysis.csv
    rsna_part5_visual_case_index.csv
    rsna_part5_pseudotarget_validation_summary.json
    phase4_part5_pseudotarget_validation_report.txt
    visual_cases/*.png
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES = RSNA_ROOT / "train_images"
COORD_CSV = RSNA_ROOT / "train_label_coordinates.csv"
SERIES_CSV = RSNA_ROOT / "train_series_descriptions.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part4_point_to_target_design"
)

VISUAL_DIR = OUTPUT_DIR / "visual_cases"


# ================================================================
# CONFIGURATION
# ================================================================

MAX_CASES = 120
MAX_VISUAL_CASES = 20
RANDOM_SEED = 42

POINT_RADIUS = 7
GAUSSIAN_SIGMA = 5.0

ADAPTIVE_MIN_RADIUS = 5
ADAPTIVE_MAX_RADIUS = 14

LEVEL_BAND_HALF_HEIGHT = 18
LEVEL_BAND_HALF_WIDTH = 28

CONDITIONS = [
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]

LEVELS = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]

STRATEGIES = [
    "point_disk",
    "gaussian",
    "adaptive_disk",
    "level_band",
    "gaussian_adaptive",
]


# ================================================================
# GENERAL HELPERS
# ================================================================

def banner(text):
    print("=" * 78)
    print(text)
    print("=" * 78)


def normalize_level(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .upper()
        .replace(" ", "")
        .replace("_", "/")
        .replace("-", "/")
    )


def normalize_condition(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


# ================================================================
# DICOM
# ================================================================

def load_dicom(dicom_path):
    import pydicom

    ds = pydicom.dcmread(str(dicom_path))

    image = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    image = image * slope + intercept

    return ds, image


def normalize_image(image):
    image = np.asarray(image, dtype=np.float32)

    finite = np.isfinite(image)

    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    values = image[finite]

    low, high = np.percentile(values, [1, 99])

    if high <= low:
        low = float(values.min())
        high = float(values.max())

    normalized = (image - low) / (high - low + 1e-8)

    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.astype(np.float32)


def find_dicom(series_dir, instance_number):
    """
    RSNA directories normally contain files named 1.dcm, 2.dcm, ...
    but the function also falls back to sorted DICOM files.
    """

    if not series_dir.exists():
        return None

    exact = series_dir / f"{int(instance_number)}.dcm"

    if exact.exists():
        return exact

    files = sorted(series_dir.glob("*.dcm"))

    if not files:
        return None

    instance_number = int(instance_number)

    if 1 <= instance_number <= len(files):
        return files[instance_number - 1]

    return None


# ================================================================
# TARGET GENERATION
# ================================================================

def point_disk(shape, x, y, radius):
    height, width = shape

    yy, xx = np.ogrid[:height, :width]

    distance_squared = (
        (xx - x) ** 2
        + (yy - y) ** 2
    )

    return distance_squared <= radius ** 2


def gaussian_target(shape, x, y, sigma):
    height, width = shape

    yy, xx = np.ogrid[:height, :width]

    distance_squared = (
        (xx - x) ** 2
        + (yy - y) ** 2
    )

    return np.exp(
        -distance_squared / (2.0 * sigma ** 2)
    ).astype(np.float32)


def adaptive_disk(shape, x, y, image):
    """
    Conservative image-dependent radius.

    The image is used only to estimate local structure. This does not
    claim to discover an anatomical boundary.
    """

    height, width = shape

    cx = int(round(x))
    cy = int(round(y))

    x0 = max(0, cx - 10)
    x1 = min(width, cx + 11)

    y0 = max(0, cy - 10)
    y1 = min(height, cy + 11)

    patch = image[y0:y1, x0:x1]

    if patch.size < 9:
        radius = POINT_RADIUS
    else:
        gy, gx = np.gradient(patch)

        gradient = np.sqrt(
            gx ** 2 + gy ** 2
        )

        mean_gradient = float(np.mean(gradient))

        radius = POINT_RADIUS

        if mean_gradient < 0.025:
            radius += 3

        elif mean_gradient > 0.10:
            radius -= 2

        radius = int(
            np.clip(
                radius,
                ADAPTIVE_MIN_RADIUS,
                ADAPTIVE_MAX_RADIUS,
            )
        )

    mask = point_disk(
        shape,
        x,
        y,
        radius,
    )

    return mask, radius


def level_band(shape, x, y):
    """
    Narrow local band around the RSNA level annotation.

    This is a localization baseline, not a proposed final anatomical
    segmentation boundary.
    """

    height, width = shape

    yy, xx = np.ogrid[:height, :width]

    vertical = np.abs(yy - y) <= LEVEL_BAND_HALF_HEIGHT
    horizontal = np.abs(xx - x) <= LEVEL_BAND_HALF_WIDTH

    return vertical & horizontal


# ================================================================
# CASE SELECTION
# ================================================================

def choose_representative_cases(df):
    """Stratified RSNA sampling across condition and MRI series type."""
    rng = np.random.default_rng(RANDOM_SEED)

    candidates = df.dropna(
        subset=["x", "y", "instance_number"]
    ).copy()

    candidates = candidates[
        candidates["condition"].isin(CONDITIONS)
    ]

    candidates = candidates[
        candidates["series_description"].isin(
            ["Sagittal T1", "Sagittal T2/STIR", "Axial T2"]
        )
    ]

    candidates = candidates.drop_duplicates(
        subset=[
            "study_id",
            "series_id",
            "condition",
            "level",
            "instance_number",
        ]
    )

    if candidates.empty:
        return candidates

    selected = []

    # First guarantee representation from every condition/series cell.
    for condition in CONDITIONS:
        for series_type in [
            "Sagittal T1",
            "Sagittal T2/STIR",
            "Axial T2",
        ]:
            subset = candidates[
                (candidates["condition"] == condition)
                & (candidates["series_description"] == series_type)
            ]

            if subset.empty:
                continue

            take = min(8, len(subset))
            selected.extend(
                rng.choice(
                    subset.index.to_numpy(),
                    size=take,
                    replace=False,
                ).tolist()
            )

    selected = list(dict.fromkeys(selected))

    remaining = candidates[
        ~candidates.index.isin(selected)
    ]

    need = MAX_CASES - len(selected)

    if need > 0 and not remaining.empty:
        take = min(need, len(remaining))
        selected.extend(
            rng.choice(
                remaining.index.to_numpy(),
                size=take,
                replace=False,
            ).tolist()
        )

    return (
        candidates.loc[selected]
        .head(MAX_CASES)
        .reset_index(drop=True)
    )


# ================================================================
# METRICS
# ================================================================

def target_metrics(target):
    array = np.asarray(target)

    area = int(
        np.count_nonzero(array > 0.5)
    )

    fraction = float(
        area / array.size
    )

    return area, fraction


def geometric_metrics(target, x, y):
    """Compute geometry relative to the original RSNA point annotation."""
    support = np.asarray(target) > 0.5
    area = int(np.count_nonzero(support))
    h, w = support.shape

    px = int(np.clip(round(x), 0, w - 1))
    py = int(np.clip(round(y), 0, h - 1))
    point_inside = int(support[py, px])

    yy, xx = np.nonzero(support)
    if len(xx) == 0:
        return {
            "point_inside": point_inside,
            "centroid_distance": np.nan,
            "radial_mean": np.nan,
            "radial_median": np.nan,
            "radial_max": np.nan,
            "bbox_width": 0,
            "bbox_height": 0,
            "area": area,
            "area_fraction": float(area / support.size),
        }

    distances = np.hypot(
        xx.astype(np.float32) - float(x),
        yy.astype(np.float32) - float(y),
    )

    centroid_x = float(xx.mean())
    centroid_y = float(yy.mean())

    return {
        "point_inside": point_inside,
        "centroid_distance": float(
            np.hypot(centroid_x - float(x), centroid_y - float(y))
        ),
        "radial_mean": float(distances.mean()),
        "radial_median": float(np.median(distances)),
        "radial_max": float(distances.max()),
        "bbox_width": int(xx.max() - xx.min() + 1),
        "bbox_height": int(yy.max() - yy.min() + 1),
        "area": area,
        "area_fraction": float(area / support.size),
    }


def add_geometric_validation_score(summary):
    """Rank target geometry only; this is NOT segmentation accuracy."""
    area = summary["mean_area"].to_numpy(dtype=float)
    centroid = summary["mean_centroid_distance"].to_numpy(dtype=float)
    radial = summary["mean_radial_distance"].to_numpy(dtype=float)
    containment = summary["point_containment"].to_numpy(dtype=float)

    def inverse_minmax(values):
        lo = np.nanmin(values)
        hi = np.nanmax(values)
        if hi <= lo:
            return np.ones_like(values, dtype=float)
        return 1.0 - ((values - lo) / (hi - lo))

    summary["area_compactness"] = inverse_minmax(area)
    summary["centroid_compactness"] = inverse_minmax(centroid)
    summary["radial_compactness"] = inverse_minmax(radial)

    summary["geometric_validation_score"] = (
        0.40 * containment
        + 0.25 * summary["area_compactness"]
        + 0.20 * summary["centroid_compactness"]
        + 0.15 * summary["radial_compactness"]
    )

    return summary.sort_values(
        "geometric_validation_score",
        ascending=False,
    ).reset_index(drop=True)



# ================================================================
# MAIN
# ================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VISUAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    banner("PHASE 4 - PART 5")
    print("RSNA PSEUDO-TARGET ANATOMICAL VALIDATION")
    banner("")

    print("PROJECT ROOT")
    print(PROJECT_ROOT)

    print()
    print("RSNA DATASET")
    print(RSNA_ROOT)

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    banner("DATASET VALIDATION")

    required_paths = {
        "RSNA root": RSNA_ROOT,
        "train_images": TRAIN_IMAGES,
        "train_label_coordinates.csv": COORD_CSV,
        "train_series_descriptions.csv": SERIES_CSV,
    }

    for name, path in required_paths.items():
        status = "FOUND" if path.exists() else "MISSING"

        print(
            f"{name:<36}: {status}"
        )

    if not all(path.exists() for path in required_paths.values()):
        raise FileNotFoundError(
            "Required RSNA dataset components are missing."
        )

    # ------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------

    banner("LOADING RSNA METADATA")

    coords = pd.read_csv(COORD_CSV)
    series = pd.read_csv(SERIES_CSV)

    coords["study_id"] = (
        coords["study_id"].astype(str)
    )

    coords["series_id"] = (
        coords["series_id"].astype(str)
    )

    coords["condition"] = (
        coords["condition"].map(normalize_condition)
    )

    coords["level"] = (
        coords["level"].map(normalize_level)
    )

    coords["x"] = pd.to_numeric(
        coords["x"],
        errors="coerce",
    )

    coords["y"] = pd.to_numeric(
        coords["y"],
        errors="coerce",
    )

    coords["instance_number"] = pd.to_numeric(
        coords["instance_number"],
        errors="coerce",
    )

    series["study_id"] = (
        series["study_id"].astype(str)
    )

    series["series_id"] = (
        series["series_id"].astype(str)
    )

    series["series_description"] = (
        series["series_description"]
        .astype(str)
        .str.strip()
    )

    df = coords.merge(
        series,
        on=["study_id", "series_id"],
        how="left",
        validate="many_to_one",
    )

    df["series_description"] = (
        df["series_description"]
        .fillna("UNKNOWN")
    )

    print(
        f"Coordinate annotations: {len(df)}"
    )

    print(
        f"Annotated studies     : {df.study_id.nunique()}"
    )

    print(
        "Annotated series      : "
        f"{df[['study_id','series_id']].drop_duplicates().shape[0]}"
    )

    # ------------------------------------------------------------
    # Select cases
    # ------------------------------------------------------------

    banner("SELECTING REPRESENTATIVE CASES")

    cases = choose_representative_cases(df)

    print(
        f"Representative cases selected: {len(cases)}"
    )

    # ------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------

    banner("CANDIDATE TARGET STRATEGIES")

    for strategy in STRATEGIES:
        print(
            "✓",
            strategy,
        )

    # ------------------------------------------------------------
    # Process
    # ------------------------------------------------------------

    banner("PROCESSING RSNA DICOM ANNOTATIONS")

    results = []
    validation_results = []
    visual_cases = []

    for index, row in cases.iterrows():

        study_id = row["study_id"]
        series_id = row["series_id"]

        condition = row["condition"]
        level = row["level"]

        instance = int(
            row["instance_number"]
        )

        x = float(row["x"])
        y = float(row["y"])

        series_dir = (
            TRAIN_IMAGES
            / study_id
            / series_id
        )

        dicom_path = find_dicom(
            series_dir,
            instance,
        )

        print(
            f"[{index + 1}/{len(cases)}] "
            f"{study_id} | "
            f"{row['series_description']} | "
            f"{condition} | "
            f"{level} | "
            f"instance={instance}"
        )

        if dicom_path is None:

            print(
                "  WARNING: DICOM instance not found."
            )

            results.append({
                "study_id": study_id,
                "series_id": series_id,
                "condition": condition,
                "level": level,
                "instance_number": instance,
                "status": "missing_dicom",
            })

            continue

        try:
            _, image = load_dicom(
                dicom_path
            )

        except Exception as exc:

            print(
                "  WARNING: DICOM read failed:",
                exc,
            )

            results.append({
                "study_id": study_id,
                "series_id": series_id,
                "condition": condition,
                "level": level,
                "instance_number": instance,
                "status": "dicom_read_failed",
                "error": str(exc),
            })

            continue

        image_norm = normalize_image(
            image
        )

        height, width = image.shape

        coordinate_valid = (
            0 <= x < width
            and
            0 <= y < height
        )

        if not coordinate_valid:

            print(
                "  WARNING: coordinate outside DICOM."
            )

            continue

        disk = point_disk(
            (height, width),
            x,
            y,
            POINT_RADIUS,
        )

        gaussian = gaussian_target(
            (height, width),
            x,
            y,
            GAUSSIAN_SIGMA,
        )

        adaptive, adaptive_radius = (
            adaptive_disk(
                (height, width),
                x,
                y,
                image_norm,
            )
        )

        band = level_band(
            (height, width),
            x,
            y,
        )

        gaussian_adaptive = gaussian_target(
            (height, width),
            x,
            y,
            max(
                3.0,
                adaptive_radius * 0.8,
            ),
        )

        targets = {
            "point_disk": disk,
            "gaussian": gaussian,
            "adaptive_disk": adaptive,
            "level_band": band,
            "gaussian_adaptive": gaussian_adaptive,
        }

        result = {
            "study_id": study_id,
            "series_id": series_id,
            "series_description": row[
                "series_description"
            ],
            "condition": condition,
            "level": level,
            "instance_number": instance,
            "image_rows": height,
            "image_columns": width,
            "x": x,
            "y": y,
            "adaptive_radius": adaptive_radius,
            "status": "success",
        }

        for name, target in targets.items():

            area, fraction = target_metrics(
                target
            )

            result[
                f"{name}_area_pixels"
            ] = area

            result[
                f"{name}_area_fraction"
            ] = fraction

            # Part 5 uses a long-format table for strategy-level
            # geometric validation. The original wide-format
            # `results` table is retained for Part 4-compatible outputs.
            geometry = geometric_metrics(
                target,
                x,
                y,
            )

            validation_results.append({
                "study_id": study_id,
                "series_id": series_id,
                "series_description": row["series_description"],
                "condition": condition,
                "level": level,
                "instance_number": instance,
                "x": x,
                "y": y,
                "strategy": name,
                "status": "success",
                **geometry,
            })

        results.append(result)

        if len(visual_cases) < MAX_VISUAL_CASES:

            visual_cases.append({
                "row": row,
                "image": image_norm,
                "targets": targets,
                "dicom_path": dicom_path,
            })

    analysis = pd.DataFrame(results)

    success = analysis[
        analysis["status"] == "success"
    ].copy()

    if success.empty:
        raise RuntimeError(
            "No RSNA DICOM annotations were successfully processed."
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    banner("TARGET GEOMETRY SUMMARY")

    for strategy in STRATEGIES:

        area = success[
            f"{strategy}_area_pixels"
        ].mean()

        fraction = success[
            f"{strategy}_area_fraction"
        ].mean()

        print(
            f"{strategy:<22}"
            f"mean area = {area:8.2f} px | "
            f"mean image fraction = {fraction:.6f}"
        )

    # ------------------------------------------------------------
    # Visual validation
    # ------------------------------------------------------------

    banner("CREATING VISUAL VALIDATION")

    visual_index = []

    for number, case in enumerate(
        visual_cases,
        start=1,
    ):

        row = case["row"]
        image = case["image"]
        targets = case["targets"]
        dicom_path = case["dicom_path"]

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 9),
            constrained_layout=True,
        )

        axes = axes.ravel()

        axes[0].imshow(
            image,
            cmap="gray",
        )

        axes[0].scatter(
            [row["x"]],
            [row["y"]],
            s=35,
        )

        axes[0].set_title(
            "Original DICOM + RSNA Point"
        )

        axes[0].axis("off")

        for axis, strategy in zip(
            axes[1:],
            STRATEGIES,
        ):

            axis.imshow(
                image,
                cmap="gray",
            )

            target = targets[
                strategy
            ]

            if strategy in {
                "gaussian",
                "gaussian_adaptive",
            }:

                axis.imshow(
                    target,
                    alpha=0.45,
                )

            else:

                axis.imshow(
                    target,
                    alpha=0.35,
                )

            axis.scatter(
                [row["x"]],
                [row["y"]],
                s=20,
            )

            axis.set_title(
                strategy
            )

            axis.axis("off")

        fig.suptitle(
            "RSNA Point-to-Target Design | "
            f"Study {row['study_id']} | "
            f"{row['condition']} | "
            f"{row['level']} | "
            f"{row['series_description']}",
            fontsize=12,
        )

        safe_condition = (
            str(row["condition"])
            .replace(" ", "_")
            .replace("/", "_")
        )

        output_file = (
            VISUAL_DIR
            / (
                f"case_{number:03d}_"
                f"{row['study_id']}_"
                f"{safe_condition}.png"
            )
        )

        fig.savefig(
            output_file,
            dpi=160,
        )

        plt.close(fig)

        visual_index.append({
            "visual_case": number,
            "study_id": row["study_id"],
            "series_id": row["series_id"],
            "condition": row["condition"],
            "level": row["level"],
            "series_description": row[
                "series_description"
            ],
            "instance_number": row[
                "instance_number"
            ],
            "x": row["x"],
            "y": row["y"],
            "dicom_path": str(dicom_path),
            "figure_path": str(output_file),
        })

        print(
            f"Saved: {output_file}"
        )

    visual_df = pd.DataFrame(
        visual_index
    )

    # ------------------------------------------------------------
    # Strategy interpretation
    # ------------------------------------------------------------

    banner("STRATEGY INTERPRETATION")

    interpretation = {

        "point_disk": {
            "strength":
                "Simple and fully deterministic.",
            "weakness":
                "Fixed radius does not represent true anatomy.",
            "role":
                "Baseline weak pseudo-target."
        },

        "gaussian": {
            "strength":
                "Provides a spatial confidence field around the annotation.",
            "weakness":
                "Does not define an anatomical boundary.",
            "role":
                "Useful for localization or soft-label supervision."
        },

        "adaptive_disk": {
            "strength":
                "Allows conservative radius variation using local image structure.",
            "weakness":
                "Image intensity/gradient is not equivalent to anatomical boundary.",
            "role":
                "Candidate adaptive pseudo-target."
        },

        "level_band": {
            "strength":
                "Represents local level context.",
            "weakness":
                "Can include substantial unrelated anatomy.",
            "role":
                "Localization baseline rather than final segmentation target."
        },

        "gaussian_adaptive": {
            "strength":
                "Combines spatial confidence and adaptive scale.",
            "weakness":
                "Adds assumptions and remains a pseudo-target.",
            "role":
                "Advanced candidate requiring validation."
        },
    }

    for strategy, info in interpretation.items():

        print()
        print(strategy)
        print(
            "  Strength:",
            info["strength"],
        )
        print(
            "  Weakness:",
            info["weakness"],
        )
        print(
            "  Role    :",
            info["role"],
        )

    # ------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------

    banner("SCIENTIFIC RECOMMENDATION")

    recommendation = (
        "Do not choose a segmentation target only from pixel area. "
        "The RSNA annotations are points, so a generated region must be "
        "treated as a pseudo-label. The safest progression is to validate "
        "point_disk and gaussian targets first, then evaluate adaptive "
        "targets. The level_band strategy should remain a localization "
        "baseline because it may include unrelated anatomy."
    )

    print(recommendation)

    # ------------------------------------------------------------
    # Part 5 quantitative summaries
    # ------------------------------------------------------------

    banner("PART 5 GEOMETRIC VALIDATION RANKING")

    validation_df = pd.DataFrame(validation_results)

    if validation_df.empty:
        raise RuntimeError(
            "No geometric validation records were created."
        )

    strategy_summary = (
        validation_df.groupby("strategy")
        .agg(
            cases=("study_id", "count"),
            point_containment=("point_inside", "mean"),
            mean_area=("area", "mean"),
            median_area=("area", "median"),
            mean_centroid_distance=("centroid_distance", "mean"),
            mean_radial_distance=("radial_mean", "mean"),
        )
        .reset_index()
    )

    strategy_summary = add_geometric_validation_score(
        strategy_summary
    )

    strategy_summary["rank"] = (
        np.arange(len(strategy_summary)) + 1
    )

    print(
        strategy_summary[
            [
                "rank",
                "strategy",
                "cases",
                "point_containment",
                "mean_area",
                "mean_centroid_distance",
                "mean_radial_distance",
                "geometric_validation_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    condition_summary = (
        validation_df.groupby(["condition", "strategy"])
        .agg(
            cases=("study_id", "count"),
            mean_area=("area", "mean"),
            mean_centroid_distance=("centroid_distance", "mean"),
            point_containment=("point_inside", "mean"),
        )
        .reset_index()
    )

    series_summary = (
        validation_df.groupby(["series_description", "strategy"])
        .agg(
            cases=("study_id", "count"),
            mean_area=("area", "mean"),
            mean_centroid_distance=("centroid_distance", "mean"),
            point_containment=("point_inside", "mean"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # Save files
    # ------------------------------------------------------------

    banner("SAVING ANALYSIS OUTPUTS")

    analysis_path = (
        OUTPUT_DIR
        / "rsna_part5_pseudotarget_case_analysis.csv"
    )

    visual_index_path = (
        OUTPUT_DIR
        / "rsna_part5_visual_case_index.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "rsna_part5_pseudotarget_validation_summary.json"
    )

    report_path = (
        OUTPUT_DIR
        / "phase4_part5_pseudotarget_validation_report.txt"
    )

    analysis.to_csv(
        analysis_path,
        index=False,
    )

    visual_df.to_csv(
        visual_index_path,
        index=False,
    )

    summary = {
        "phase": "Phase 4 - Part 5",
        "title":
            "RSNA Pseudo-Target Anatomical Validation",
        "dataset":
            "RSNA 2024 Lumbar Spine Degenerative Classification",
        "spider_used": False,
        "training_performed": False,
        "model_weights_modified": False,
        "manual_ground_truth_masks_used": False,
        "total_annotations": int(len(df)),
        "representative_cases_selected": int(len(cases)),
        "successful_dicom_cases": int(len(success)),
        "strategies": STRATEGIES,
        "mean_area_pixels": {
            strategy: float(
                success[
                    f"{strategy}_area_pixels"
                ].mean()
            )
            for strategy in STRATEGIES
        },
        "mean_image_fraction": {
            strategy: float(
                success[
                    f"{strategy}_area_fraction"
                ].mean()
            )
            for strategy in STRATEGIES
        },
        "interpretation": interpretation,
        "recommendation": recommendation,
        "scientific_warning":
            "Generated targets are pseudo-label candidates, not manual ground truth masks.",
        "next_phase":
            "Construct a controlled RSNA pseudo-target benchmark and perform visual/quantitative validation before Swin-UNETR training.",
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    report_lines = [
        "PHASE 4 - PART 5",
        "RSNA PSEUDO-TARGET ANATOMICAL VALIDATION",
        "",
        f"RSNA dataset: {RSNA_ROOT}",
        f"Total coordinate annotations: {len(df)}",
        f"Representative cases selected: {len(cases)}",
        f"Successful DICOM cases: {len(success)}",
        "",
        "TARGET STRATEGIES",
        "",
    ]

    for strategy in STRATEGIES:

        report_lines.append(
            f"{strategy}: "
            f"mean area = "
            f"{success[f'{strategy}_area_pixels'].mean():.2f} px; "
            f"mean image fraction = "
            f"{success[f'{strategy}_area_fraction'].mean():.6f}"
        )

    report_lines.extend([
        "",
        "SCIENTIFIC INTERPRETATION",
        "RSNA coordinates are point annotations.",
        "Point annotations are not manual segmentation masks.",
        "Generated regions are therefore pseudo-target candidates.",
        "",
        "RECOMMENDATION",
        recommendation,
        "",
        "SPIDER used: NO",
        "Training performed: NO",
        "Model weights modified: NO",
        "Manual ground-truth masks used: NO",
        "",
        "NEXT PHASE",
        "Validate the target-generation strategy on a controlled",
        "RSNA benchmark before beginning Swin-UNETR training.",
    ])

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(
        f"Saved: {analysis_path}"
    )
    print(
        f"Saved: {visual_index_path}"
    )
    print(
        f"Saved: {summary_path}"
    )
    print(
        f"Saved: {report_path}"
    )
    validation_df.to_csv(
        OUTPUT_DIR / "rsna_part5_geometric_validation.csv",
        index=False,
    )

    strategy_summary.to_csv(
        OUTPUT_DIR / "rsna_part5_strategy_summary.csv",
        index=False,
    )

    condition_summary.to_csv(
        OUTPUT_DIR / "rsna_part5_condition_summary.csv",
        index=False,
    )

    series_summary.to_csv(
        OUTPUT_DIR / "rsna_part5_series_summary.csv",
        index=False,
    )

    best_strategy = str(
        strategy_summary.iloc[0]["strategy"]
    )
    best_score = float(
        strategy_summary.iloc[0]["geometric_validation_score"]
    )

    print(
        "Saved: "
        f"{OUTPUT_DIR / 'rsna_part5_strategy_summary.csv'}"
    )
    print(
        "Saved: "
        f"{OUTPUT_DIR / 'rsna_part5_condition_summary.csv'}"
    )
    print(
        "Saved: "
        f"{OUTPUT_DIR / 'rsna_part5_series_summary.csv'}"
    )

    # Geometric ranking chart.
    plt.figure(figsize=(10, 6))
    chart = strategy_summary.sort_values(
        "geometric_validation_score"
    )
    plt.barh(
        chart["strategy"],
        chart["geometric_validation_score"],
    )
    plt.xlabel("Geometric validation score")
    plt.ylabel("Pseudo-target strategy")
    plt.title("RSNA Pseudo-Target Geometric Validation")
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "part5_geometric_validation_score.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()
    print(f"Saved: {chart_path}")

    # Centroid-distance chart.
    plt.figure(figsize=(10, 6))
    chart = strategy_summary.sort_values(
        "mean_centroid_distance"
    )
    plt.bar(
        chart["strategy"],
        chart["mean_centroid_distance"],
    )
    plt.ylabel("Mean centroid distance (pixels)")
    plt.xlabel("Pseudo-target strategy")
    plt.title("RSNA Point-to-Target Centroid Distance")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "part5_centroid_distance.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()
    print(f"Saved: {chart_path}")



    # ------------------------------------------------------------
    # Final
    # ------------------------------------------------------------

    print()
    print(
        "Scientific decision:"
    )
    print(
        f"Best geometric candidate: {best_strategy}"
    )
    print(
        f"Geometric validation score: {best_score:.6f}"
    )
    print(
        "This score measures geometric consistency with RSNA "
        "point annotations; it is NOT segmentation accuracy."
    )
    print(
        "Do not treat these pseudo-targets as manual ground truth."
    )

    banner("PART 5 COMPLETE")

    print(
        f"Representative cases processed : {len(success)}"
    )

    print(
        "RSNA only                       : YES"
    )

    print(
        "SPIDER used                     : NO"
    )

    print(
        "Training performed              : NO"
    )

    print(
        "Model weights modified         : NO"
    )

    print(
        "Manual masks used               : NO"
    )

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    banner("PHASE 4 - PART 5 COMPLETE")


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(1)

    except Exception as exc:
        print()
        print("ERROR:")
        print(type(exc).__name__, str(exc))
        raise
