"""
PHASE 4 - PART 6
RSNA VALIDATED PSEUDO-MASK GENERATION - CORRECTED

Purpose
-------
Part 5 validated candidate point-to-region pseudo-target strategies.
Part 6 converts the selected RSNA Gaussian strategy into reproducible
3-D sparse pseudo-masks at the annotated DICOM instances.

Scientific scope
----------------
RSNA 2024 provides point annotations for five degenerative findings:
    1. Spinal Canal Stenosis
    2. Left Neural Foraminal Narrowing
    3. Right Neural Foraminal Narrowing
    4. Left Subarticular Stenosis
    5. Right Subarticular Stenosis

It does NOT provide manual pixel-wise masks for vertebrae, discs, or canal.

Therefore this script:
    - uses RSNA only;
    - creates five-condition pseudo-labels;
    - does NOT claim them to be manual ground truth;
    - does NOT create the previous 4-class SPIDER mask format;
    - does NOT train Swin-UNETR;
    - does NOT modify model weights.

The generated volume is intentionally sparse:
only DICOM instances carrying an RSNA point annotation receive a
Gaussian pseudo-region. Unannotated slices remain background.

This conservative construction avoids inventing anatomical extent
where RSNA supplies no annotation.

Output
------
For every processed series:
    *.npz containing:
        image_shape
        pseudo_mask
        annotated_slice_indices
        annotation_table

Global CSV/JSON/TXT summaries are also created.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES = RSNA_ROOT / "train_images"

COORD_CSV = RSNA_ROOT / "train_label_coordinates.csv"
SERIES_CSV = RSNA_ROOT / "train_series_descriptions.csv"

PART5_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part5_pseudotarget_anatomical_validation"
)

# Accept the previous accidental Part-4 location too.
PART5_FALLBACK_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part4_point_to_target_design"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part6_pseudomask_generation"
)

MASK_DIR = OUTPUT_DIR / "pseudo_masks"


# ======================================================================
# CONFIGURATION
# ======================================================================

RANDOM_SEED = 42

# Part 5 selected Gaussian.
SELECTED_STRATEGY = "gaussian"

# Must match the validated Part 5 Gaussian construction.
GAUSSIAN_SIGMA = 5.0

# Gaussian is converted to a binary pseudo-region at this threshold.
GAUSSIAN_THRESHOLD = 0.30

# To prevent accidental overwriting of a pixel by a lower-priority
# condition when two annotations overlap.
CLASS_PRIORITY = {
    "Spinal Canal Stenosis": 5,
    "Left Neural Foraminal Narrowing": 4,
    "Right Neural Foraminal Narrowing": 3,
    "Left Subarticular Stenosis": 2,
    "Right Subarticular Stenosis": 1,
}

CLASS_MAP = {
    "Spinal Canal Stenosis": 1,
    "Left Neural Foraminal Narrowing": 2,
    "Right Neural Foraminal Narrowing": 3,
    "Left Subarticular Stenosis": 4,
    "Right Subarticular Stenosis": 5,
}

CONDITIONS = list(CLASS_MAP.keys())

SERIES_TYPES = [
    "Sagittal T1",
    "Sagittal T2/STIR",
    "Axial T2",
]

# Process every eligible annotated series by default.
# Set to an integer for a pilot run.
MAX_SERIES = None


# ======================================================================
# HELPERS
# ======================================================================

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


def gaussian_disk(shape, x, y, sigma, threshold):
    """Create a binary Gaussian-derived region around an RSNA point."""
    height, width = shape

    yy, xx = np.ogrid[:height, :width]

    distance_squared = (
        (xx.astype(np.float32) - float(x)) ** 2
        + (yy.astype(np.float32) - float(y)) ** 2
    )

    gaussian = np.exp(
        -distance_squared
        / (2.0 * float(sigma) ** 2)
    )

    return gaussian >= float(threshold)


def find_part5_summary():
    candidates = [
        PART5_DIR / "rsna_part5_strategy_summary.csv",
        PART5_FALLBACK_DIR / "rsna_part5_strategy_summary.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def find_dicom_files(series_dir):
    import pydicom

    records = []

    if not series_dir.exists():
        return records

    for path in series_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
            )

            instance = int(
                getattr(ds, "InstanceNumber", 0)
            )

            records.append(
                {
                    "path": path,
                    "instance_number": instance,
                    "rows": int(
                        getattr(ds, "Rows", 0)
                    ),
                    "columns": int(
                        getattr(ds, "Columns", 0)
                    ),
                }
            )

        except Exception:
            continue

    records.sort(
        key=lambda item: (
            item["instance_number"],
            item["path"].name,
        )
    )

    return records


def choose_series(coords, series):
    """Return annotated RSNA series eligible for pseudo-mask creation.

    ``coords`` is the merged coordinate/series table returned by
    ``load_metadata()`` and therefore already contains
    ``series_description``.  Re-merging it with ``series`` would create
    pandas suffixes (``series_description_x/y``), which caused the
    previous ``KeyError``.
    """

    required_columns = {
        "study_id",
        "series_id",
        "series_description",
        "condition",
    }

    missing = required_columns.difference(coords.columns)

    if missing:
        raise KeyError(
            "Metadata table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    eligible = coords[
        coords["condition"].isin(CONDITIONS)
        & coords["series_description"].isin(SERIES_TYPES)
    ][
        [
            "study_id",
            "series_id",
            "series_description",
        ]
    ].drop_duplicates()

    eligible = eligible.sort_values(
        [
            "study_id",
            "series_id",
        ]
    ).reset_index(drop=True)

    if MAX_SERIES is not None:
        eligible = eligible.head(
            int(MAX_SERIES)
        )

    return eligible


def load_metadata():
    coords = pd.read_csv(COORD_CSV)
    series = pd.read_csv(SERIES_CSV)

    coords["study_id"] = (
        coords["study_id"].astype(str)
    )

    coords["series_id"] = (
        coords["series_id"].astype(str)
    )

    series["study_id"] = (
        series["study_id"].astype(str)
    )

    series["series_id"] = (
        series["series_id"].astype(str)
    )

    coords["condition"] = (
        coords["condition"]
        .map(normalize_condition)
    )

    coords["level"] = (
        coords["level"]
        .map(normalize_level)
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

    series["series_description"] = (
        series["series_description"]
        .astype(str)
        .str.strip()
    )

    merged = coords.merge(
        series,
        on=[
            "study_id",
            "series_id",
        ],
        how="left",
        validate="many_to_one",
    )

    return merged, series


def make_series_mask(
    dicom_records,
    annotations,
):
    """Create a sparse 3-D condition pseudo-mask.

    Axis convention:
        [slice, y, x]

    Class 0 = background
    Classes 1-5 = the five RSNA conditions.
    """

    if not dicom_records:
        return None, None

    first_rows = int(
        dicom_records[0]["rows"]
    )

    first_cols = int(
        dicom_records[0]["columns"]
    )

    if first_rows <= 0 or first_cols <= 0:
        return None, None

    mask = np.zeros(
        (
            len(dicom_records),
            first_rows,
            first_cols,
        ),
        dtype=np.uint8,
    )

    instance_to_slice = {
        int(record["instance_number"]): index
        for index, record in enumerate(dicom_records)
    }

    generated_rows = []

    for row in annotations.to_dict("records"):

        if not np.isfinite(row["x"]):
            continue

        if not np.isfinite(row["y"]):
            continue

        if not np.isfinite(
            row["instance_number"]
        ):
            continue

        instance = int(
            row["instance_number"]
        )

        if instance not in instance_to_slice:
            continue

        slice_index = instance_to_slice[
            instance
        ]

        x = float(row["x"])
        y = float(row["y"])

        if not (
            0 <= x < first_cols
            and 0 <= y < first_rows
        ):
            continue

        condition = row["condition"]

        if condition not in CLASS_MAP:
            continue

        class_id = int(
            CLASS_MAP[condition]
        )

        target = gaussian_disk(
            (
                first_rows,
                first_cols,
            ),
            x,
            y,
            GAUSSIAN_SIGMA,
            GAUSSIAN_THRESHOLD,
        )

        existing = mask[
            slice_index
        ]

        # Only replace background or lower-priority labels.
        # This keeps the construction deterministic.
        if np.any(target):
            existing_priority = np.zeros_like(
                existing,
                dtype=np.int16,
            )

            for condition_name, cid in CLASS_MAP.items():
                priority = CLASS_PRIORITY[
                    condition_name
                ]

                existing_priority[
                    existing == cid
                ] = priority

            write_pixels = (
                target
                & (
                    existing_priority
                    <= CLASS_PRIORITY[condition]
                )
            )

            existing[
                write_pixels
            ] = class_id

        generated_rows.append(
            {
                "study_id": row["study_id"],
                "series_id": row["series_id"],
                "series_description":
                    row["series_description"],
                "condition": condition,
                "level": row["level"],
                "instance_number": instance,
                "slice_index": slice_index,
                "x": x,
                "y": y,
                "class_id": class_id,
                "pseudo_area_pixels": int(
                    np.count_nonzero(target)
                ),
            }
        )

    return mask, generated_rows


def save_npz(
    path,
    mask,
    annotations,
):
    annotation_json = json.dumps(
        annotations,
        default=str,
    )

    np.savez_compressed(
        path,
        pseudo_mask=mask,
        annotation_table=np.array(
            annotation_json
        ),
        gaussian_sigma=np.array(
            GAUSSIAN_SIGMA,
            dtype=np.float32,
        ),
        gaussian_threshold=np.array(
            GAUSSIAN_THRESHOLD,
            dtype=np.float32,
        ),
        selected_strategy=np.array(
            SELECTED_STRATEGY
        ),
        class_map=np.array(
            json.dumps(CLASS_MAP)
        ),
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    np.random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MASK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    banner("PHASE 4 - PART 6")
    print("RSNA VALIDATED PSEUDO-MASK GENERATION")
    banner("")

    print("PROJECT ROOT")
    print(PROJECT_ROOT)

    print()
    print("RSNA DATASET")
    print(RSNA_ROOT)

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    banner("DATASET PATH VALIDATION")

    required = {
        "RSNA root": RSNA_ROOT,
        "train_images": TRAIN_IMAGES,
        "train_label_coordinates.csv": COORD_CSV,
        "train_series_descriptions.csv": SERIES_CSV,
    }

    for name, path in required.items():
        print(
            f"{name:<36}: "
            f"{'FOUND' if path.exists() else 'MISSING'}"
        )

    if not all(
        path.exists()
        for path in required.values()
    ):
        raise FileNotFoundError(
            "Required RSNA dataset files are missing."
        )

    # ------------------------------------------------------------------
    # Part 5 source
    # ------------------------------------------------------------------

    banner("LOADING PART 5 VALIDATION")

    part5_summary = find_part5_summary()

    if part5_summary is None:
        raise FileNotFoundError(
            "Part 5 strategy summary was not found. "
            "Run Part 5 successfully before Part 6."
        )

    print(
        "Part 5 strategy summary:"
    )
    print(part5_summary)

    part5_df = pd.read_csv(
        part5_summary
    )

    if "strategy" not in part5_df.columns:
        raise ValueError(
            "Part 5 strategy summary does not contain "
            "the required 'strategy' column."
        )

    print()
    print(
        part5_df[
            [
                "strategy",
                "geometric_validation_score",
            ]
        ].to_string(index=False)
    )

    if SELECTED_STRATEGY not in set(
        part5_df["strategy"]
    ):
        raise ValueError(
            "Selected Part 5 strategy "
            f"'{SELECTED_STRATEGY}' is absent."
        )

    selected_row = part5_df[
        part5_df["strategy"]
        == SELECTED_STRATEGY
    ].iloc[0]

    print()
    print(
        f"Selected strategy: {SELECTED_STRATEGY}"
    )

    print(
        "Part 5 geometric validation score: "
        f"{float(selected_row['geometric_validation_score']):.6f}"
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    banner("LOADING RSNA METADATA")

    metadata, series = load_metadata()

    print(
        f"Coordinate rows: {len(metadata)}"
    )

    print(
        f"Annotated studies: "
        f"{metadata['study_id'].nunique()}"
    )

    print(
        f"Annotated series: "
        f"{metadata[['study_id','series_id']].drop_duplicates().shape[0]}"
    )

    # ------------------------------------------------------------------
    # Explain target definition
    # ------------------------------------------------------------------

    banner("PSEUDO-MASK DEFINITION")

    print(
        "Strategy:"
        f" {SELECTED_STRATEGY}"
    )

    print(
        "Gaussian sigma:"
        f" {GAUSSIAN_SIGMA}"
    )

    print(
        "Binary threshold:"
        f" {GAUSSIAN_THRESHOLD}"
    )

    print()
    print("Classes:")

    for condition, class_id in CLASS_MAP.items():
        print(
            f"  {class_id}: {condition}"
        )

    print()
    print(
        "Important: these are RSNA point-derived "
        "pseudo-labels, not manual segmentation masks."
    )

    # ------------------------------------------------------------------
    # Select series
    # ------------------------------------------------------------------

    banner("SELECTING ANNOTATED RSNA SERIES")

    series_cases = choose_series(
        metadata,
        series,
    )

    print(
        f"Eligible series: {len(series_cases)}"
    )

    if series_cases.empty:
        raise RuntimeError(
            "No eligible RSNA annotated series were found for "
            "pseudo-mask generation."
        )

    # ------------------------------------------------------------------
    # Process series
    # ------------------------------------------------------------------

    banner("GENERATING 3-D SPARSE PSEUDO-MASKS")

    series_summary = []
    annotation_records = []

    successful = 0
    failed = 0

    for number, item in enumerate(
        series_cases.to_dict("records"),
        start=1,
    ):

        study_id = str(
            item["study_id"]
        )

        series_id = str(
            item["series_id"]
        )

        description = str(
            item["series_description"]
        )

        print(
            f"[{number}/{len(series_cases)}] "
            f"{study_id} | "
            f"{series_id} | "
            f"{description}"
        )

        series_dir = (
            TRAIN_IMAGES
            / study_id
            / series_id
        )

        dicom_records = find_dicom_files(
            series_dir
        )

        if not dicom_records:
            failed += 1
            print(
                "  WARNING: no readable DICOM files"
            )
            continue

        annotations = metadata[
            (metadata["study_id"] == study_id)
            & (metadata["series_id"] == series_id)
            & (
                metadata["condition"].isin(
                    CONDITIONS
                )
            )
        ].copy()

        if annotations.empty:
            failed += 1
            print(
                "  WARNING: no eligible annotations"
            )
            continue

        mask, generated = make_series_mask(
            dicom_records,
            annotations,
        )

        if mask is None:
            failed += 1
            print(
                "  WARNING: mask generation failed"
            )
            continue

        if not generated:
            failed += 1
            print(
                "  WARNING: no valid point annotations "
                "mapped to DICOM instances"
            )
            continue

        safe_description = (
            description
            .replace("/", "_")
            .replace(" ", "_")
            .replace("\\", "_")
        )

        output_path = (
            MASK_DIR
            / (
                f"{study_id}_"
                f"{series_id}_"
                f"{safe_description}.npz"
            )
        )

        save_npz(
            output_path,
            mask,
            generated,
        )

        mask_voxels = int(
            np.count_nonzero(mask)
        )

        foreground_fraction = float(
            mask_voxels
            / mask.size
        )

        annotated_slices = int(
            len(
                np.unique(
                    [
                        row["slice_index"]
                        for row in generated
                    ]
                )
            )
        )

        condition_counts = (
            pd.DataFrame(generated)
            ["condition"]
            .value_counts()
            .to_dict()
        )

        summary_row = {
            "study_id": study_id,
            "series_id": series_id,
            "series_description": description,
            "num_dicom_slices": len(
                dicom_records
            ),
            "height": int(mask.shape[1]),
            "width": int(mask.shape[2]),
            "annotation_rows_used": len(
                generated
            ),
            "annotated_slices": annotated_slices,
            "pseudo_mask_voxels": mask_voxels,
            "pseudo_mask_foreground_fraction":
                foreground_fraction,
            "spinal_canal_annotations":
                int(
                    condition_counts.get(
                        "Spinal Canal Stenosis",
                        0,
                    )
                ),
            "left_foraminal_annotations":
                int(
                    condition_counts.get(
                        "Left Neural Foraminal Narrowing",
                        0,
                    )
                ),
            "right_foraminal_annotations":
                int(
                    condition_counts.get(
                        "Right Neural Foraminal Narrowing",
                        0,
                    )
                ),
            "left_subarticular_annotations":
                int(
                    condition_counts.get(
                        "Left Subarticular Stenosis",
                        0,
                    )
                ),
            "right_subarticular_annotations":
                int(
                    condition_counts.get(
                        "Right Subarticular Stenosis",
                        0,
                    )
                ),
            "output_file": str(
                output_path
            ),
        }

        series_summary.append(
            summary_row
        )

        annotation_records.extend(
            generated
        )

        successful += 1

        print(
            f"  DICOM slices       : {len(dicom_records)}"
        )

        print(
            f"  Annotation points  : {len(generated)}"
        )

        print(
            f"  Annotated slices   : {annotated_slices}"
        )

        print(
            f"  Pseudo-mask voxels : {mask_voxels}"
        )

        print(
            f"  Saved              : {output_path}"
        )

    # ------------------------------------------------------------------
    # DataFrames
    # ------------------------------------------------------------------

    series_summary_df = pd.DataFrame(
        series_summary
    )

    annotation_df = pd.DataFrame(
        annotation_records
    )

    if series_summary_df.empty:
        raise RuntimeError(
            "No pseudo-mask volumes were successfully generated."
        )

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------

    banner("PSEUDO-MASK GENERATION SUMMARY")

    total_annotation_rows = int(
        len(annotation_df)
    )

    total_mask_voxels = int(
        series_summary_df[
            "pseudo_mask_voxels"
        ].sum()
    )

    total_dicom_slices = int(
        series_summary_df[
            "num_dicom_slices"
        ].sum()
    )

    total_annotated_slices = int(
        series_summary_df[
            "annotated_slices"
        ].sum()
    )

    overall_foreground_fraction = float(
        total_mask_voxels
        / max(
            1,
            sum(
                int(row["num_dicom_slices"])
                * int(row["height"])
                * int(row["width"])
                for row in series_summary
            ),
        )
    )

    print(
        f"Successful series       : {successful}"
    )

    print(
        f"Failed series           : {failed}"
    )

    print(
        f"Annotation rows used    : {total_annotation_rows}"
    )

    print(
        f"Total DICOM slices      : {total_dicom_slices}"
    )

    print(
        f"Annotated slices        : {total_annotated_slices}"
    )

    print(
        f"Pseudo-mask voxels      : {total_mask_voxels}"
    )

    print(
        "Foreground fraction     : "
        f"{overall_foreground_fraction:.8f}"
    )

    # ------------------------------------------------------------------
    # Condition summary
    # ------------------------------------------------------------------

    banner("CONDITION-WISE PSEUDO-TARGET SUMMARY")

    if not annotation_df.empty:

        condition_summary = (
            annotation_df
            .groupby(
                [
                    "condition",
                    "class_id",
                ]
            )
            .agg(
                annotation_rows=(
                    "condition",
                    "count",
                ),
                mean_pseudo_area_pixels=(
                    "pseudo_area_pixels",
                    "mean",
                ),
                median_pseudo_area_pixels=(
                    "pseudo_area_pixels",
                    "median",
                ),
                unique_studies=(
                    "study_id",
                    "nunique",
                ),
                unique_series=(
                    "series_id",
                    "nunique",
                ),
            )
            .reset_index()
        )

        print(
            condition_summary.to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.4f}",
            )
        )

    else:
        condition_summary = pd.DataFrame()

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    banner("SAVING PART 6 OUTPUTS")

    series_path = (
        OUTPUT_DIR
        / "rsna_part6_series_summary.csv"
    )

    annotation_path = (
        OUTPUT_DIR
        / "rsna_part6_annotation_summary.csv"
    )

    condition_path = (
        OUTPUT_DIR
        / "rsna_part6_condition_summary.csv"
    )

    config_path = (
        OUTPUT_DIR
        / "rsna_part6_pseudomask_config.json"
    )

    summary_path = (
        OUTPUT_DIR
        / "phase4_part6_pseudomask_generation_summary.json"
    )

    report_path = (
        OUTPUT_DIR
        / "phase4_part6_pseudomask_generation_report.txt"
    )

    series_summary_df.to_csv(
        series_path,
        index=False,
    )

    annotation_df.to_csv(
        annotation_path,
        index=False,
    )

    condition_summary.to_csv(
        condition_path,
        index=False,
    )

    config = {
        "phase": "Phase 4 - Part 6",
        "title":
            "RSNA Validated Pseudo-Mask Generation",
        "dataset":
            "RSNA 2024 Lumbar Spine Degenerative Classification",
        "rsna_only": True,
        "spider_used": False,
        "selected_strategy": SELECTED_STRATEGY,
        "part5_strategy_summary":
            str(part5_summary),
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "gaussian_threshold":
            GAUSSIAN_THRESHOLD,
        "class_map": CLASS_MAP,
        "pseudo_mask_definition":
            "Binary Gaussian-derived regions generated only on "
            "DICOM instances containing RSNA point annotations.",
        "unannotated_slices":
            "Remain background.",
        "manual_masks_used": False,
        "training_performed": False,
        "model_weights_modified": False,
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    summary = {
        **config,
        "eligible_series":
            int(len(series_cases)),
        "successful_series":
            int(successful),
        "failed_series":
            int(failed),
        "annotation_rows_used":
            total_annotation_rows,
        "total_dicom_slices":
            total_dicom_slices,
        "annotated_slices":
            total_annotated_slices,
        "pseudo_mask_voxels":
            total_mask_voxels,
        "overall_foreground_fraction":
            overall_foreground_fraction,
        "output_mask_directory":
            str(MASK_DIR),
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
            default=str,
        )

    report_lines = [
        "PHASE 4 - PART 6",
        "RSNA VALIDATED PSEUDO-MASK GENERATION",
        "",
        f"RSNA root: {RSNA_ROOT}",
        f"Part 5 source: {part5_summary}",
        f"Selected strategy: {SELECTED_STRATEGY}",
        "",
        "TARGET DEFINITION",
        "The selected Gaussian point-to-region strategy is",
        "converted into sparse 3-D pseudo-masks.",
        "Only annotated DICOM instances receive pseudo-regions.",
        "Unannotated slices remain background.",
        "",
        "IMPORTANT SCIENTIFIC LIMITATION",
        "RSNA supplies point annotations, not manual pixel-wise masks.",
        "The generated volumes are pseudo-labels and must not be",
        "reported as ground-truth segmentation masks.",
        "",
        "RESULTS",
        f"Eligible series: {len(series_cases)}",
        f"Successful series: {successful}",
        f"Failed series: {failed}",
        f"Annotation rows used: {total_annotation_rows}",
        f"Total DICOM slices: {total_dicom_slices}",
        f"Annotated slices: {total_annotated_slices}",
        f"Pseudo-mask voxels: {total_mask_voxels}",
        f"Foreground fraction: {overall_foreground_fraction:.8f}",
        "",
        "CLASS MAP",
    ]

    for condition, class_id in CLASS_MAP.items():
        report_lines.append(
            f"{class_id}: {condition}"
        )

    report_lines.extend([
        "",
        "SPIDER used: NO",
        "Manual segmentation masks used: NO",
        "Training performed: NO",
        "Model weights modified: NO",
        "",
        "NEXT STEP",
        "Perform pseudo-label quality-control and train/validation/test",
        "split construction before any Swin-UNETR training.",
    ])

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(
        f"Saved: {series_path}"
    )

    print(
        f"Saved: {annotation_path}"
    )

    print(
        f"Saved: {condition_path}"
    )

    print(
        f"Saved: {config_path}"
    )

    print(
        f"Saved: {summary_path}"
    )

    print(
        f"Saved: {report_path}"
    )

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------

    banner("PART 6 COMPLETE")

    print(
        f"Eligible series             : {len(series_cases)}"
    )

    print(
        f"Successful series           : {successful}"
    )

    print(
        f"Failed series               : {failed}"
    )

    print(
        f"Annotation rows used        : {total_annotation_rows}"
    )

    print(
        f"Pseudo-mask voxels          : {total_mask_voxels}"
    )

    print(
        f"Selected strategy           : {SELECTED_STRATEGY}"
    )

    print()
    print(
        "RSNA only                   : YES"
    )

    print(
        "SPIDER used                 : NO"
    )

    print(
        "Manual masks used           : NO"
    )

    print(
        "Training performed          : NO"
    )

    print(
        "Model weights modified      : NO"
    )

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    banner("PHASE 4 - PART 6 COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print()
        print("=" * 78)
        print("ERROR")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise
