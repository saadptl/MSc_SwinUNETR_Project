"""
==============================================================================
PHASE 4 - PART 7
==============================================================================
RSNA PSEUDO-MASK QUALITY VALIDATION
==============================================================================

Purpose
-------
Validate the pseudo-masks generated in Phase 4 - Part 6 before model training.

Important scientific distinction
--------------------------------
These masks are point-derived pseudo-labels generated from RSNA coordinate
annotations. They are NOT manually annotated pixel-wise segmentation masks.

Part 7 performs:
    1. Pseudo-mask file integrity validation
    2. Array-shape validation
    3. Class-ID validation
    4. Empty-mask / near-empty-mask detection
    5. Per-class voxel statistics
    6. Connected-component analysis
    7. Annotation-to-mask spatial validation
    8. Representative visual inspection
    9. Quality flag generation
   10. Final training-readiness decision

Part 7 does NOT:
    - train a model
    - modify model weights
    - create new pseudo-masks
    - use Spider
    - use ground-truth segmentation masks

Dataset:
    RSNA 2024 Lumbar Spine Degenerative Classification

Input:
    outputs/segmentation/rsna_part6_pseudomask_generation/pseudo_masks

Outputs:
    outputs/segmentation/rsna_part7_pseudomask_quality_validation
==============================================================================
"""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES = RSNA_ROOT / "train_images"

COORDINATES_CSV = RSNA_ROOT / "train_label_coordinates.csv"

SERIES_CSV = RSNA_ROOT / "train_series_descriptions.csv"

PART6_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part6_pseudomask_generation"
)

PSEUDOMASK_DIR = PART6_DIR / "pseudo_masks"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part7_pseudomask_quality_validation"
)

VIS_DIR = OUTPUT_DIR / "visual_inspection"

REPORT_DIR = OUTPUT_DIR / "reports"


# =============================================================================
# CONSTANTS
# =============================================================================

CLASS_MAP = {
    "Spinal Canal Stenosis": 1,
    "Left Neural Foraminal Narrowing": 2,
    "Right Neural Foraminal Narrowing": 3,
    "Left Subarticular Stenosis": 4,
    "Right Subarticular Stenosis": 5,
}

CLASS_NAMES = {
    0: "background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

VALID_CLASS_IDS = set(CLASS_NAMES.keys()) - {0}

# Quality thresholds.
# These are validation thresholds, not medical ground-truth criteria.

MIN_TOTAL_FOREGROUND_VOXELS = 10

MIN_CLASS_VOXELS = 2

MAX_CONNECTED_COMPONENTS_PER_CLASS = 50

MAX_COMPONENT_FRAGMENTATION_RATIO = 0.50

MAX_POINT_DISTANCE_PIXELS = 20.0

REPRESENTATIVE_CASE_COUNT = 20


# =============================================================================
# UTILITIES
# =============================================================================

def print_header(title: str):
    print("=" * 78)
    print(title)
    print("=" * 78)


def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def normalize_image(image):
    image = np.asarray(image, dtype=np.float32)

    finite = np.isfinite(image)

    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    values = image[finite]

    lo = np.percentile(values, 1)
    hi = np.percentile(values, 99)

    if hi <= lo:
        lo = values.min()
        hi = values.max()

    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo)

    return image


def load_npz_mask(path: Path):
    """
    Load Part 6 NPZ while supporting several reasonable key conventions.
    """

    with np.load(path, allow_pickle=True) as data:

        keys = list(data.keys())

        preferred = [
            "mask",
            "pseudo_mask",
            "segmentation",
            "labels",
            "label",
        ]

        mask_key = None

        for key in preferred:
            if key in keys:
                mask_key = key
                break

        if mask_key is None:
            arrays = []

            for key in keys:
                arr = data[key]

                if isinstance(arr, np.ndarray):
                    arrays.append((key, arr))

            if not arrays:
                raise ValueError("NPZ contains no ndarray")

            # Prefer integer-like multidimensional array.
            candidates = [
                item
                for item in arrays
                if item[1].ndim >= 2
            ]

            if not candidates:
                candidates = arrays

            mask_key, mask = candidates[0]
        else:
            mask = data[mask_key]

        metadata = {}

        for key in keys:
            if key == mask_key:
                continue

            try:
                value = data[key]

                if np.ndim(value) == 0:
                    value = value.item()

                metadata[key] = value
            except Exception:
                pass

    return np.asarray(mask), metadata


def parse_pseudomask_filename(path: Path):
    """
    Expected Part 6 naming:

        study_id_series_id_Series_Type.npz

    Series descriptions are encoded as:
        Sagittal_T1
        Axial_T2
        Sagittal_T2_STIR
    """

    stem = path.stem

    parts = stem.split("_")

    if len(parts) < 3:
        return None, None, None

    study_id = parts[0]
    series_id = parts[1]

    description = "_".join(parts[2:])

    description_map = {
        "Sagittal_T1": "Sagittal T1",
        "Axial_T2": "Axial T2",
        "Sagittal_T2_STIR": "Sagittal T2/STIR",
    }

    description = description_map.get(description, description)

    return study_id, series_id, description


def load_metadata():
    coords = pd.read_csv(COORDINATES_CSV)
    series = pd.read_csv(SERIES_CSV)

    coords["study_id"] = coords["study_id"].astype(str)
    coords["series_id"] = coords["series_id"].astype(str)

    series["study_id"] = series["study_id"].astype(str)
    series["series_id"] = series["series_id"].astype(str)

    return coords, series


def build_annotation_lookup(coords):
    lookup = {}

    for series_id, group in coords.groupby("series_id"):

        rows = []

        for _, row in group.iterrows():

            condition = str(row["condition"])

            class_id = CLASS_MAP.get(condition)

            if class_id is None:
                continue

            rows.append(
                {
                    "study_id": str(row["study_id"]),
                    "series_id": str(row["series_id"]),
                    "instance_number": int(row["instance_number"]),
                    "condition": condition,
                    "class_id": int(class_id),
                    "level": str(row["level"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
            )

        lookup[str(series_id)] = rows

    return lookup


def load_dicom_series(series_dir: Path):
    """
    Load DICOM files ordered approximately by InstanceNumber.

    Returns:
        volume
        instance_numbers
        paths
    """

    if pydicom is None:
        raise ImportError(
            "pydicom is required for DICOM validation. "
            "Install it with: pip install pydicom"
        )

    files = list(series_dir.glob("*.dcm"))

    if not files:
        raise FileNotFoundError(f"No DICOM files found: {series_dir}")

    records = []

    for path in files:

        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=False,
            )

            instance = getattr(ds, "InstanceNumber", None)

            try:
                instance = int(instance)
            except Exception:
                instance = 0

            records.append((instance, path, ds))

        except Exception:
            continue

    if not records:
        raise RuntimeError(f"Unable to read DICOM series: {series_dir}")

    records.sort(key=lambda x: (x[0], x[1].name))

    images = []

    instance_numbers = []

    paths = []

    for instance, path, ds in records:

        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue

        images.append(arr)
        instance_numbers.append(instance)
        paths.append(path)

    if not images:
        raise RuntimeError(
            f"No readable DICOM pixel arrays: {series_dir}"
        )

    try:
        volume = np.stack(images, axis=0)
    except Exception:
        # Different in-plane dimensions.
        min_h = min(x.shape[0] for x in images)
        min_w = min(x.shape[1] for x in images)

        images = [
            x[:min_h, :min_w]
            for x in images
        ]

        volume = np.stack(images, axis=0)

    return volume, instance_numbers, paths


def get_mask_voxel_counts(mask):
    values, counts = np.unique(mask, return_counts=True)

    result = {}

    for value, count in zip(values, counts):

        try:
            class_id = int(value)
        except Exception:
            continue

        result[class_id] = int(count)

    return result


def connected_component_statistics(binary_mask):
    if not binary_mask.any():
        return {
            "components": 0,
            "largest_component": 0,
            "total_voxels": 0,
            "fragmentation_ratio": 0.0,
        }

    if ndimage is None:
        return {
            "components": np.nan,
            "largest_component": np.nan,
            "total_voxels": int(binary_mask.sum()),
            "fragmentation_ratio": np.nan,
        }

    structure = ndimage.generate_binary_structure(
        binary_mask.ndim,
        1,
    )

    labels, num = ndimage.label(
        binary_mask,
        structure=structure,
    )

    if num == 0:
        return {
            "components": 0,
            "largest_component": 0,
            "total_voxels": int(binary_mask.sum()),
            "fragmentation_ratio": 0.0,
        }

    component_sizes = np.bincount(
        labels.ravel()
    )[1:]

    largest = int(component_sizes.max())

    total = int(binary_mask.sum())

    fragmentation = (
        1.0 - largest / total
        if total > 0
        else 0.0
    )

    return {
        "components": int(num),
        "largest_component": largest,
        "total_voxels": total,
        "fragmentation_ratio": float(fragmentation),
    }


def get_slice_axis(mask, instance_number, annotation_instance):
    """
    Determine the slice index corresponding to an annotation instance.

    Part 6 masks are expected to have shape:

        [slice, height, width]

    If exact instance mapping cannot be recovered, fall back to nearest
    instance number.
    """

    try:
        instances = np.asarray(instance_number, dtype=np.int64)

        if len(instances) == mask.shape[0]:

            matches = np.where(
                instances == int(annotation_instance)
            )[0]

            if len(matches):
                return int(matches[0])

            distances = np.abs(
                instances - int(annotation_instance)
            )

            return int(np.argmin(distances))

    except Exception:
        pass

    return None


def point_to_mask_distance(
    class_mask,
    x,
    y,
    z,
):
    """
    Measure whether the annotation point lands inside the corresponding
    pseudo-mask or close to it.

    x/y are DICOM image coordinates.
    z is the slice index.
    """

    if z is None:
        return np.nan, False

    if z < 0 or z >= class_mask.shape[0]:
        return np.nan, False

    h, w = class_mask.shape[1:]

    x = float(x)
    y = float(y)

    ix = int(round(x))
    iy = int(round(y))

    if 0 <= iy < h and 0 <= ix < w:

        if class_mask[z, iy, ix]:
            return 0.0, True

    positions = np.argwhere(
        class_mask
    )

    if positions.size == 0:
        return np.nan, False

    same_slice = positions[
        positions[:, 0] == z
    ]

    if same_slice.size == 0:
        return np.nan, False

    dy = same_slice[:, 1].astype(float) - y
    dx = same_slice[:, 2].astype(float) - x

    distances = np.sqrt(
        dx * dx + dy * dy
    )

    minimum = float(distances.min())

    return minimum, bool(minimum <= MAX_POINT_DISTANCE_PIXELS)


def choose_representative_cases(summary_df):
    """
    Select cases from different quality strata rather than only taking the
    first N rows.
    """

    if summary_df.empty:
        return summary_df

    candidates = []

    # Worst foreground counts
    candidates.append(
        summary_df.nsmallest(
            5,
            "total_foreground_voxels",
        )
    )

    # Highest fragmentation
    candidates.append(
        summary_df.nlargest(
            5,
            "mean_fragmentation_ratio",
        )
    )

    # Best point coverage
    candidates.append(
        summary_df.nlargest(
            5,
            "point_inside_or_near_rate",
        )
    )

    # Lowest point coverage
    candidates.append(
        summary_df.nsmallest(
            5,
            "point_inside_or_near_rate",
        )
    )

    combined = pd.concat(
        candidates,
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["study_id", "series_id"]
    )

    return combined.head(REPRESENTATIVE_CASE_COUNT)


def visualize_case(
    row,
    coords_lookup,
):
    study_id = str(row["study_id"])
    series_id = str(row["series_id"])

    mask_path = Path(row["mask_path"])

    try:
        mask, metadata = load_npz_mask(mask_path)
    except Exception as exc:
        print(
            f"  Visualization load failed: {exc}"
        )
        return None

    annotations = coords_lookup.get(
        series_id,
        [],
    )

    if mask.ndim != 3:
        return None

    # Choose the slice containing the largest foreground area.
    foreground = mask > 0

    slice_counts = foreground.reshape(
        foreground.shape[0],
        -1,
    ).sum(axis=1)

    if slice_counts.max() == 0:
        slice_index = 0
    else:
        slice_index = int(
            np.argmax(slice_counts)
        )

    # DICOM image.
    series_dir = (
        TRAIN_IMAGES
        / study_id
        / series_id
    )

    image = None
    instance_numbers = None

    try:
        volume, instance_numbers, _ = load_dicom_series(
            series_dir
        )

        if volume.shape[0] == mask.shape[0]:
            image = volume[slice_index]

    except Exception:
        image = None

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    if image is not None:

        ax.imshow(
            normalize_image(image),
            cmap="gray",
        )

    else:

        # If DICOM loading fails, display the mask directly.
        ax.imshow(
            mask[slice_index] > 0,
            cmap="gray",
        )

    # Mask contours.
    for class_id in sorted(VALID_CLASS_IDS):

        binary = (
            mask[slice_index] == class_id
        )

        if binary.any():

            ax.contour(
                binary,
                levels=[0.5],
                linewidths=1.2,
            )

    # Annotation points.
    for annotation in annotations:

        z = None

        if instance_numbers is not None:

            try:
                z = get_mask_slice_for_annotation(
                    mask,
                    instance_numbers,
                    annotation["instance_number"],
                )
            except Exception:
                z = None

        if z is None:

            # Try metadata slice mapping.
            z = annotation.get(
                "slice_index",
                None,
            )

        if z is None:
            continue

        if int(z) != slice_index:
            continue

        ax.scatter(
            annotation["x"],
            annotation["y"],
            s=45,
            marker="x",
            linewidths=2,
        )

    ax.set_title(
        f"RSNA pseudo-mask validation\n"
        f"Study {study_id} | Series {series_id}\n"
        f"Slice {slice_index}"
    )

    ax.axis("off")

    out_path = (
        VIS_DIR
        / f"{study_id}_{series_id}_quality_validation.png"
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    return out_path


def get_mask_slice_for_annotation(
    mask,
    instance_numbers,
    annotation_instance,
):
    return get_slice_axis(
        mask,
        instance_numbers,
        annotation_instance,
    )


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def main():

    print_header(
        "PHASE 4 - PART 7"
    )

    print(
        "RSNA PSEUDO-MASK QUALITY VALIDATION"
    )

    print_header(
        "PROJECT PATHS"
    )

    print(
        f"PROJECT ROOT\n{PROJECT_ROOT}\n"
    )

    print(
        f"RSNA DATASET\n{RSNA_ROOT}\n"
    )

    print(
        f"PART 6 PSEUDO-MASK DIRECTORY\n"
        f"{PSEUDOMASK_DIR}\n"
    )

    print(
        f"OUTPUT DIRECTORY\n{OUTPUT_DIR}\n"
    )

    ensure_directories()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    print_header(
        "DATASET PATH VALIDATION"
    )

    checks = {
        "RSNA root": RSNA_ROOT.exists(),
        "train_images": TRAIN_IMAGES.exists(),
        "train_label_coordinates.csv": COORDINATES_CSV.exists(),
        "train_series_descriptions.csv": SERIES_CSV.exists(),
        "Part 6 pseudo_masks": PSEUDOMASK_DIR.exists(),
    }

    for name, status in checks.items():
        print(
            f"{name:<35}: "
            f"{'FOUND' if status else 'MISSING'}"
        )

    if not all(checks.values()):
        raise FileNotFoundError(
            "One or more required paths are missing."
        )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    print_header(
        "LOADING RSNA METADATA"
    )

    coords, series = load_metadata()

    print(
        f"Coordinate rows       : {len(coords)}"
    )

    print(
        f"Series metadata rows  : {len(series)}"
    )

    coords_lookup = build_annotation_lookup(
        coords
    )

    # -------------------------------------------------------------------------
    # Part 6 masks
    # -------------------------------------------------------------------------

    print_header(
        "LOADING PART 6 PSEUDO-MASK FILES"
    )

    mask_files = sorted(
        PSEUDOMASK_DIR.glob("*.npz")
    )

    print(
        f"Pseudo-mask files found : {len(mask_files)}"
    )

    if not mask_files:
        raise RuntimeError(
            "No Part 6 pseudo-mask files found."
        )

    # -------------------------------------------------------------------------
    # Validation loop
    # -------------------------------------------------------------------------

    print_header(
        "PSEUDO-MASK STRUCTURAL VALIDATION"
    )

    records = []

    class_total_voxels = Counter()

    total_valid = 0
    total_invalid = 0

    total_empty = 0

    point_distances = []

    for index, mask_path in enumerate(
        mask_files,
        start=1,
    ):

        if index == 1 or index % 250 == 0:
            print(
                f"[{index}/{len(mask_files)}] "
                f"Validating pseudo-mask files..."
            )

        study_id, series_id, description = (
            parse_pseudomask_filename(
                mask_path
            )
        )

        record = {
            "study_id": study_id,
            "series_id": series_id,
            "series_description": description,
            "mask_path": str(mask_path),
            "status": "PASS",
            "quality_flags": "",
        }

        flags = []

        try:

            mask, metadata = load_npz_mask(
                mask_path
            )

            record["mask_ndim"] = int(
                mask.ndim
            )

            record["mask_shape"] = str(
                tuple(mask.shape)
            )

            if mask.ndim != 3:

                flags.append(
                    "INVALID_DIMENSION"
                )

            unique_values = np.unique(mask)

            invalid_values = [
                int(x)
                for x in unique_values
                if int(x) not in CLASS_NAMES
            ]

            record["unique_class_ids"] = ",".join(
                str(int(x))
                for x in unique_values
            )

            record["invalid_class_ids"] = ",".join(
                str(x)
                for x in invalid_values
            )

            if invalid_values:

                flags.append(
                    "INVALID_CLASS_ID"
                )

            counts = get_mask_voxel_counts(
                mask
            )

            foreground = int(
                sum(
                    count
                    for class_id, count
                    in counts.items()
                    if class_id != 0
                )
            )

            record[
                "total_foreground_voxels"
            ] = foreground

            if foreground < MIN_TOTAL_FOREGROUND_VOXELS:

                flags.append(
                    "NEAR_EMPTY_MASK"
                )

                total_empty += 1

            # Per-class counts.
            for class_id in sorted(
                VALID_CLASS_IDS
            ):

                count = int(
                    counts.get(
                        class_id,
                        0,
                    )
                )

                record[
                    f"class_{class_id}_voxels"
                ] = count

                class_total_voxels[
                    class_id
                ] += count

                if 0 < count < MIN_CLASS_VOXELS:

                    flags.append(
                        f"CLASS_{class_id}_VERY_SMALL"
                    )

            # Connected components.
            component_values = []

            for class_id in sorted(
                VALID_CLASS_IDS
            ):

                binary = (
                    mask == class_id
                )

                stats = (
                    connected_component_statistics(
                        binary
                    )
                )

                record[
                    f"class_{class_id}_components"
                ] = stats["components"]

                record[
                    f"class_{class_id}_largest_component"
                ] = stats[
                    "largest_component"
                ]

                record[
                    f"class_{class_id}_fragmentation"
                ] = stats[
                    "fragmentation_ratio"
                ]

                if (
                    not pd.isna(
                        stats["components"]
                    )
                    and stats["components"]
                    > MAX_CONNECTED_COMPONENTS_PER_CLASS
                ):

                    flags.append(
                        f"CLASS_{class_id}_HIGH_COMPONENT_COUNT"
                    )

                if (
                    not pd.isna(
                        stats[
                            "fragmentation_ratio"
                        ]
                    )
                    and stats[
                        "fragmentation_ratio"
                    ]
                    > MAX_COMPONENT_FRAGMENTATION_RATIO
                ):

                    flags.append(
                        f"CLASS_{class_id}_HIGH_FRAGMENTATION"
                    )

                component_values.append(
                    stats["fragmentation_ratio"]
                )

            valid_fragmentations = [
                x
                for x in component_values
                if not pd.isna(x)
            ]

            record[
                "mean_fragmentation_ratio"
            ] = (
                float(
                    np.mean(
                        valid_fragmentations
                    )
                )
                if valid_fragmentations
                else np.nan
            )

            # Annotation-point validation.
            annotations = coords_lookup.get(
                str(series_id),
                [],
            )

            record[
                "annotation_count"
            ] = len(annotations)

            inside_or_near = 0

            distances_for_case = []

            # We need DICOM instance mapping.
            instance_numbers = None

            if mask.ndim == 3:

                series_dir = (
                    TRAIN_IMAGES
                    / str(study_id)
                    / str(series_id)
                )

                try:

                    _, instance_numbers, _ = (
                        load_dicom_series(
                            series_dir
                        )
                    )

                except Exception:

                    instance_numbers = None

            for annotation in annotations:

                z = get_mask_slice_for_annotation(
                    mask,
                    instance_numbers,
                    annotation[
                        "instance_number"
                    ],
                ) if instance_numbers is not None else None

                if z is None:
                    continue

                class_id = annotation[
                    "class_id"
                ]

                class_mask = (
                    mask == class_id
                )

                distance, near = (
                    point_to_mask_distance(
                        class_mask,
                        annotation["x"],
                        annotation["y"],
                        z,
                    )
                )

                if not pd.isna(distance):

                    distances_for_case.append(
                        distance
                    )

                    point_distances.append(
                        distance
                    )

                if near:
                    inside_or_near += 1

            record[
                "point_inside_or_near_count"
            ] = inside_or_near

            record[
                "point_inside_or_near_rate"
            ] = (
                inside_or_near / len(annotations)
                if annotations
                else np.nan
            )

            record[
                "mean_point_distance_pixels"
            ] = (
                float(
                    np.mean(
                        distances_for_case
                    )
                )
                if distances_for_case
                else np.nan
            )

            record[
                "max_point_distance_pixels"
            ] = (
                float(
                    np.max(
                        distances_for_case
                    )
                )
                if distances_for_case
                else np.nan
            )

            if (
                annotations
                and record[
                    "point_inside_or_near_rate"
                ] < 0.80
            ):

                flags.append(
                    "LOW_POINT_ALIGNMENT"
                )

            if flags:

                record["status"] = (
                    "CAUTION"
                )

            total_valid += 1

        except Exception as exc:

            total_invalid += 1

            record["status"] = "FAIL"

            flags.append(
                "MASK_READ_ERROR"
            )

            record[
                "error"
            ] = str(exc)

        record[
            "quality_flags"
        ] = ";".join(
            sorted(
                set(flags)
            )
        )

        records.append(record)

    summary_df = pd.DataFrame(
        records
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print_header(
        "QUALITY VALIDATION SUMMARY"
    )

    total_files = len(
        summary_df
    )

    pass_count = int(
        (
            summary_df["status"]
            == "PASS"
        ).sum()
    )

    caution_count = int(
        (
            summary_df["status"]
            == "CAUTION"
        ).sum()
    )

    fail_count = int(
        (
            summary_df["status"]
            == "FAIL"
        ).sum()
    )

    print(
        f"Pseudo-mask files        : {total_files}"
    )

    print(
        f"PASS                     : {pass_count}"
    )

    print(
        f"CAUTION                  : {caution_count}"
    )

    print(
        f"FAIL                     : {fail_count}"
    )

    if "total_foreground_voxels" in summary_df:

        print(
            f"Mean foreground voxels : "
            f"{summary_df['total_foreground_voxels'].mean():.2f}"
        )

        print(
            f"Median foreground voxels : "
            f"{summary_df['total_foreground_voxels'].median():.2f}"
        )

    if point_distances:

        print(
            f"Mean point-to-mask distance : "
            f"{np.mean(point_distances):.3f} pixels"
        )

        print(
            f"Median point-to-mask distance : "
            f"{np.median(point_distances):.3f} pixels"
        )

    # -------------------------------------------------------------------------
    # Class summary
    # -------------------------------------------------------------------------

    print_header(
        "PER-CLASS PSEUDO-MASK SUMMARY"
    )

    class_rows = []

    for class_id in sorted(
        VALID_CLASS_IDS
    ):

        voxel_column = (
            f"class_{class_id}_voxels"
        )

        component_column = (
            f"class_{class_id}_components"
        )

        fragmentation_column = (
            f"class_{class_id}_fragmentation"
        )

        row = {
            "class_id": class_id,
            "class_name": CLASS_NAMES[
                class_id
            ],
            "total_voxels": int(
                summary_df[
                    voxel_column
                ].sum()
            ),
            "mean_voxels_per_series": float(
                summary_df[
                    voxel_column
                ].mean()
            ),
            "median_voxels_per_series": float(
                summary_df[
                    voxel_column
                ].median()
            ),
            "series_with_class": int(
                (
                    summary_df[
                        voxel_column
                    ] > 0
                ).sum()
            ),
            "mean_components": float(
                summary_df[
                    component_column
                ].mean()
            ),
            "mean_fragmentation": float(
                summary_df[
                    fragmentation_column
                ].mean()
            ),
        }

        class_rows.append(row)

        print(
            f"{class_id}: "
            f"{CLASS_NAMES[class_id]}"
        )

        print(
            f"  Total voxels : "
            f"{row['total_voxels']}"
        )

        print(
            f"  Mean voxels  : "
            f"{row['mean_voxels_per_series']:.2f}"
        )

        print(
            f"  Series       : "
            f"{row['series_with_class']}"
        )

    class_df = pd.DataFrame(
        class_rows
    )

    # -------------------------------------------------------------------------
    # Quality flags
    # -------------------------------------------------------------------------

    print_header(
        "QUALITY FLAG FREQUENCY"
    )

    flag_counter = Counter()

    for value in summary_df[
        "quality_flags"
    ].fillna(""):

        if not value:
            continue

        for flag in value.split(";"):

            if flag:
                flag_counter[flag] += 1

    flag_rows = []

    for flag, count in sorted(
        flag_counter.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    ):

        flag_rows.append(
            {
                "quality_flag": flag,
                "series_count": int(count),
                "percentage": float(
                    100.0
                    * count
                    / total_files
                ),
            }
        )

        print(
            f"{flag:<45} : "
            f"{count}"
        )

    flags_df = pd.DataFrame(
        flag_rows
    )

    # -------------------------------------------------------------------------
    # Representative visual cases
    # -------------------------------------------------------------------------

    print_header(
        "SELECTING REPRESENTATIVE VISUAL CASES"
    )

    representatives = choose_representative_cases(
        summary_df
    )

    print(
        f"Cases selected : "
        f"{len(representatives)}"
    )

    visual_rows = []

    for _, row in representatives.iterrows():

        study_id = str(
            row["study_id"]
        )

        series_id = str(
            row["series_id"]
        )

        print(
            f"  Visualizing "
            f"{study_id} | {series_id}"
        )

        output_path = visualize_case(
            row,
            coords_lookup,
        )

        visual_rows.append(
            {
                "study_id": study_id,
                "series_id": series_id,
                "series_description": row[
                    "series_description"
                ],
                "status": row[
                    "status"
                ],
                "visualization": (
                    str(output_path)
                    if output_path
                    else ""
                ),
            }
        )

    visual_df = pd.DataFrame(
        visual_rows
    )

    # -------------------------------------------------------------------------
    # Final decision
    # -------------------------------------------------------------------------

    print_header(
        "FINAL TRAINING-READINESS DECISION"
    )

    severe_failures = (
        fail_count > 0
    )

    excessive_empty = (
        total_files > 0
        and (
            total_empty
            / total_files
        ) > 0.05
    )

    mean_alignment = (
        summary_df[
            "point_inside_or_near_rate"
        ].mean()
    )

    alignment_problem = (
        not pd.isna(mean_alignment)
        and mean_alignment < 0.80
    )

    if severe_failures:

        decision = (
            "FAIL - pseudo-mask integrity "
            "problems require correction"
        )

    elif excessive_empty:

        decision = (
            "CAUTION - excessive empty/"
            "near-empty pseudo-masks"
        )

    elif alignment_problem:

        decision = (
            "CAUTION - point-to-mask spatial "
            "alignment requires review"
        )

    else:

        decision = (
            "PASS - pseudo-mask generation "
            "is structurally suitable for "
            "controlled model-training preparation"
        )

    print(
        f"Decision : {decision}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This decision does NOT mean the pseudo-masks "
        "are equivalent to manual segmentation ground truth."
    )

    print(
        "It only evaluates whether the generated "
        "pseudo-labels are structurally and spatially "
        "reasonable enough to continue to the next "
        "experimental stage."
    )

    # -------------------------------------------------------------------------
    # Save reports
    # -------------------------------------------------------------------------

    print_header(
        "SAVING VALIDATION TABLES"
    )

    summary_csv = (
        OUTPUT_DIR
        / "rsna_part7_pseudomask_quality_summary.csv"
    )

    class_csv = (
        OUTPUT_DIR
        / "rsna_part7_class_quality_summary.csv"
    )

    flags_csv = (
        OUTPUT_DIR
        / "rsna_part7_quality_flag_summary.csv"
    )

    visual_csv = (
        OUTPUT_DIR
        / "rsna_part7_visual_case_selection.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    class_df.to_csv(
        class_csv,
        index=False,
    )

    flags_df.to_csv(
        flags_csv,
        index=False,
    )

    visual_df.to_csv(
        visual_csv,
        index=False,
    )

    print(
        f"Saved: {summary_csv}"
    )

    print(
        f"Saved: {class_csv}"
    )

    print(
        f"Saved: {flags_csv}"
    )

    print(
        f"Saved: {visual_csv}"
    )

    # -------------------------------------------------------------------------
    # JSON summary
    # -------------------------------------------------------------------------

    print_header(
        "CREATING FINAL SUMMARY"
    )

    json_summary = {
        "phase": "Phase 4 - Part 7",
        "title": (
            "RSNA Pseudo-Mask Quality Validation"
        ),
        "dataset": "RSNA 2024 Lumbar Spine Degenerative Classification",
        "spider_used": False,
        "training_performed": False,
        "model_weights_modified": False,
        "manual_segmentation_ground_truth_used": False,
        "point_annotations_used": True,
        "part6_pseudomasks": {
            "files_found": total_files,
            "pass": pass_count,
            "caution": caution_count,
            "fail": fail_count,
            "near_empty": total_empty,
        },
        "point_alignment": {
            "mean_distance_pixels": (
                float(
                    np.mean(point_distances)
                )
                if point_distances
                else None
            ),
            "median_distance_pixels": (
                float(
                    np.median(point_distances)
                )
                if point_distances
                else None
            ),
            "mean_inside_or_near_rate": (
                float(mean_alignment)
                if not pd.isna(mean_alignment)
                else None
            ),
        },
        "decision": decision,
        "quality_thresholds": {
            "min_total_foreground_voxels": (
                MIN_TOTAL_FOREGROUND_VOXELS
            ),
            "min_class_voxels": (
                MIN_CLASS_VOXELS
            ),
            "max_components_per_class": (
                MAX_CONNECTED_COMPONENTS_PER_CLASS
            ),
            "max_fragmentation_ratio": (
                MAX_COMPONENT_FRAGMENTATION_RATIO
            ),
            "max_point_distance_pixels": (
                MAX_POINT_DISTANCE_PIXELS
            ),
        },
        "output_directory": str(
            OUTPUT_DIR
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "phase4_part7_pseudomask_quality_validation_summary.json"
    )

    save_json(
        json_path,
        json_summary,
    )

    print(
        f"Saved: {json_path}"
    )

    # -------------------------------------------------------------------------
    # Text report
    # -------------------------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "phase4_part7_pseudomask_quality_validation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 4 - PART 7\n"
        )

        f.write(
            "RSNA PSEUDO-MASK QUALITY VALIDATION\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            f"Pseudo-mask files : {total_files}\n"
        )

        f.write(
            f"PASS              : {pass_count}\n"
        )

        f.write(
            f"CAUTION           : {caution_count}\n"
        )

        f.write(
            f"FAIL              : {fail_count}\n"
        )

        f.write(
            f"Near-empty masks  : {total_empty}\n\n"
        )

        f.write(
            f"Mean point distance : "
            f"{np.mean(point_distances) if point_distances else 'N/A'} pixels\n"
        )

        f.write(
            f"Median point distance : "
            f"{np.median(point_distances) if point_distances else 'N/A'} pixels\n"
        )

        f.write(
            f"Mean point alignment rate : "
            f"{mean_alignment if not pd.isna(mean_alignment) else 'N/A'}\n\n"
        )

        f.write(
            "FINAL DECISION\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            decision
            + "\n\n"
        )

        f.write(
            "SCIENTIFIC NOTE\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "The evaluated masks are point-derived pseudo-labels "
            "constructed from RSNA coordinate annotations. "
            "They are not equivalent to manually drawn segmentation "
            "ground-truth masks. Therefore, Part 7 evaluates "
            "structural integrity, spatial consistency and "
            "pseudo-label usability rather than clinical segmentation "
            "accuracy.\n\n"
        )

        f.write(
            "SPIDER USED: NO\n"
        )

        f.write(
            "TRAINING PERFORMED: NO\n"
        )

        f.write(
            "MODEL WEIGHTS MODIFIED: NO\n"
        )

        f.write(
            "MANUAL SEGMENTATION GROUND TRUTH USED: NO\n"
        )

    print(
        f"Saved: {report_path}"
    )

    # -------------------------------------------------------------------------
    # Completion
    # -------------------------------------------------------------------------

    print_header(
        "PART 7 COMPLETE"
    )

    print(
        f"Pseudo-mask files validated : {total_files}"
    )

    print(
        f"PASS                        : {pass_count}"
    )

    print(
        f"CAUTION                     : {caution_count}"
    )

    print(
        f"FAIL                        : {fail_count}"
    )

    print(
        f"Near-empty masks            : {total_empty}"
    )

    print(
        f"Final decision              : {decision}"
    )

    print(
        "\nSPIDER used                 : NO"
    )

    print(
        "Training performed          : NO"
    )

    print(
        "Model weights modified     : NO"
    )

    print(
        "\nOUTPUT DIRECTORY"
    )

    print(
        OUTPUT_DIR
    )

    print_header(
        "PHASE 4 - PART 7 COMPLETE"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print_header(
            "ERROR"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        raise