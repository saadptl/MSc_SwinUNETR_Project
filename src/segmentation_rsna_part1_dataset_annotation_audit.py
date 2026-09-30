"""
PHASE 4 - PART 1
RSNA-ONLY SEGMENTATION DATASET & ANNOTATION AUDIT

Purpose
-------
Audit the locally available RSNA 2024 Lumbar Spine Degenerative
Classification dataset before any segmentation preprocessing/training.

This script:
- audits RSNA CSV files
- counts studies / series / DICOM files
- analyzes MRI sequence distribution
- analyzes disease-label distribution
- analyzes coordinate annotations
- checks coordinate -> series consistency
- checks coordinate -> DICOM instance consistency
- checks coordinate bounds against DICOM dimensions
- checks important DICOM geometry fields
- analyzes spinal-canal annotations
- creates CSV/JSON/TXT reports

This script DOES NOT:
- use SPIDER
- create segmentation masks
- train a model
- modify model weights

Run:
    python -m py_compile src\segmentation_rsna_part1_dataset_annotation_audit.py

Then:
    python src\segmentation_rsna_part1_dataset_annotation_audit.py
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

try:
    import pydicom
except ImportError:
    raise SystemExit(
        "\nERROR: pydicom is not installed.\n"
        "Install it with:\n\n"
        "pip install pydicom\n"
    )


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES = RSNA_ROOT / "train_images"

TRAIN_CSV = RSNA_ROOT / "train.csv"
COORDINATES_CSV = RSNA_ROOT / "train_label_coordinates.csv"
SERIES_CSV = RSNA_ROOT / "train_series_descriptions.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part1_dataset_annotation_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# HELPERS
# ============================================================================

def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def safe_int(value):
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def load_dicom_header(path):
    try:
        return pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
        )
    except Exception:
        return None


def find_dicom_instance(series_dir, instance_number):
    """
    Try to find the DICOM corresponding to the RSNA instance_number.
    """

    direct = series_dir / f"{instance_number}.dcm"

    if direct.exists():
        return direct

    # Filename-based fallback
    for path in series_dir.glob("*.dcm"):
        try:
            if int(path.stem) == instance_number:
                return path
        except Exception:
            pass

    # Header-based fallback
    for path in series_dir.glob("*.dcm"):

        ds = load_dicom_header(path)

        if ds is None:
            continue

        dicom_instance = safe_int(
            getattr(ds, "InstanceNumber", None)
        )

        if dicom_instance == instance_number:
            return path

    return None


# ============================================================================
# MAIN
# ============================================================================

def main():

    start_time = time.time()

    print("=" * 78)
    print("PHASE 4 - PART 1")
    print("RSNA-ONLY SEGMENTATION DATASET & ANNOTATION AUDIT")
    print("=" * 78)

    print("\nPROJECT ROOT")
    print(PROJECT_ROOT)

    print("\nRSNA DATASET")
    print(RSNA_ROOT)

    print("\nTRAIN IMAGE DIRECTORY")
    print(TRAIN_IMAGES)

    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    # ========================================================================
    # PATH CHECK
    # ========================================================================

    header("DATASET PATH VALIDATION")

    required_paths = {
        "RSNA root": RSNA_ROOT,
        "train_images": TRAIN_IMAGES,
        "train.csv": TRAIN_CSV,
        "train_label_coordinates.csv": COORDINATES_CSV,
        "train_series_descriptions.csv": SERIES_CSV,
    }

    for name, path in required_paths.items():

        if path.exists():
            print(f"{name:<38}: FOUND")
        else:
            print(f"{name:<38}: MISSING")
            raise FileNotFoundError(
                f"Required RSNA path does not exist:\n{path}"
            )

    # ========================================================================
    # LOAD CSV FILES
    # ========================================================================

    header("LOADING RSNA CSV FILES")

    train_df = pd.read_csv(TRAIN_CSV)

    coordinates_df = pd.read_csv(
        COORDINATES_CSV
    )

    series_df = pd.read_csv(
        SERIES_CSV
    )

    print(
        f"train.csv rows                     : "
        f"{len(train_df)}"
    )

    print(
        f"train_label_coordinates rows       : "
        f"{len(coordinates_df)}"
    )

    print(
        f"train_series_descriptions rows      : "
        f"{len(series_df)}"
    )

    print("\ntrain.csv columns")

    for column in train_df.columns:
        print("  -", column)

    print("\nCoordinate CSV columns")

    for column in coordinates_df.columns:
        print("  -", column)

    print("\nSeries CSV columns")

    for column in series_df.columns:
        print("  -", column)

    # ========================================================================
    # FILESYSTEM DATASET COUNTS
    # ========================================================================

    header("RSNA FILESYSTEM COUNTS")

    study_dirs = [
        path
        for path in TRAIN_IMAGES.iterdir()
        if path.is_dir()
    ]

    series_dirs = []

    filesystem_rows = []

    total_dicom_images = 0

    for study_dir in study_dirs:

        for series_dir in study_dir.iterdir():

            if not series_dir.is_dir():
                continue

            dicom_files = list(
                series_dir.glob("*.dcm")
            )

            count = len(dicom_files)

            total_dicom_images += count

            series_dirs.append(series_dir)

            filesystem_rows.append(
                {
                    "study_id": study_dir.name,
                    "series_id": series_dir.name,
                    "dicom_count": count,
                }
            )

    filesystem_studies = len(study_dirs)
    filesystem_series = len(series_dirs)

    print(
        f"Filesystem studies                 : "
        f"{filesystem_studies}"
    )

    print(
        f"Filesystem series                  : "
        f"{filesystem_series}"
    )

    print(
        f"Filesystem DICOM images            : "
        f"{total_dicom_images}"
    )

    print(
        f"CSV unique studies                 : "
        f"{series_df['study_id'].nunique()}"
    )

    print(
        f"CSV unique series                  : "
        f"{series_df['series_id'].nunique()}"
    )

    print(
        f"Coordinate unique studies          : "
        f"{coordinates_df['study_id'].nunique()}"
    )

    print(
        f"Coordinate unique series           : "
        f"{coordinates_df['series_id'].nunique()}"
    )

    # ========================================================================
    # MRI SERIES DISTRIBUTION
    # ========================================================================

    header("MRI SERIES DISTRIBUTION")

    series_distribution = (
        series_df["series_description"]
        .fillna("MISSING")
        .value_counts()
    )

    print(series_distribution.to_string())

    # ========================================================================
    # DISEASE LABEL DISTRIBUTION
    # ========================================================================

    header("DISEASE LABEL DISTRIBUTION")

    disease_columns = [
        column
        for column in train_df.columns
        if column != "study_id"
    ]

    disease_rows = []

    for column in disease_columns:

        counts = (
            train_df[column]
            .fillna("MISSING")
            .value_counts()
        )

        for label, count in counts.items():

            disease_rows.append(
                {
                    "label_column": column,
                    "label": str(label),
                    "count": int(count),
                    "percentage": (
                        float(count / len(train_df) * 100)
                    ),
                }
            )

    disease_distribution = pd.DataFrame(
        disease_rows
    )

    print(
        disease_distribution.to_string(
            index=False
        )
    )

    # ========================================================================
    # COORDINATE DISTRIBUTION
    # ========================================================================

    header("SPATIAL ANNOTATION DISTRIBUTION")

    print("\nCONDITIONS")

    print(
        coordinates_df["condition"]
        .fillna("MISSING")
        .value_counts()
        .to_string()
    )

    print("\nLEVELS")

    print(
        coordinates_df["level"]
        .fillna("MISSING")
        .value_counts()
        .to_string()
    )

    # ========================================================================
    # COORDINATE QUALITY
    # ========================================================================

    header("COORDINATE QUALITY AUDIT")

    coordinate_quality = {
        "total_rows": len(coordinates_df),
        "missing_x": int(
            coordinates_df["x"].isna().sum()
        ),
        "missing_y": int(
            coordinates_df["y"].isna().sum()
        ),
        "missing_instance_number": int(
            coordinates_df["instance_number"].isna().sum()
        ),
        "missing_condition": int(
            coordinates_df["condition"].isna().sum()
        ),
        "missing_level": int(
            coordinates_df["level"].isna().sum()
        ),
    }

    for key, value in coordinate_quality.items():

        print(
            f"{key:<38}: {value}"
        )

    # ========================================================================
    # ANNOTATION -> SERIES CONSISTENCY
    # ========================================================================

    header("ANNOTATION -> SERIES CONSISTENCY")

    series_keys = set(
        zip(
            series_df["study_id"].astype(int),
            series_df["series_id"].astype(int),
        )
    )

    coordinate_keys = list(
        zip(
            coordinates_df["study_id"].astype(int),
            coordinates_df["series_id"].astype(int),
        )
    )

    missing_series_references = [
        key
        for key in coordinate_keys
        if key not in series_keys
    ]

    print(
        f"Coordinate rows                    : "
        f"{len(coordinates_df)}"
    )

    print(
        f"Missing series references           : "
        f"{len(missing_series_references)}"
    )

    # ========================================================================
    # ANNOTATION SERIES TYPE
    # ========================================================================

    header("ANNOTATION SERIES-TYPE ANALYSIS")

    annotation_with_series = coordinates_df.merge(
        series_df[
            [
                "study_id",
                "series_id",
                "series_description",
            ]
        ],
        on=[
            "study_id",
            "series_id",
        ],
        how="left",
    )

    annotation_series_distribution = (
        annotation_with_series[
            "series_description"
        ]
        .fillna("MISSING")
        .value_counts()
    )

    print(
        annotation_series_distribution.to_string()
    )

    # ========================================================================
    # FILESYSTEM <-> CSV CONSISTENCY
    # ========================================================================

    header("FILESYSTEM <-> CSV CONSISTENCY")

    filesystem_study_ids = {
        int(path.name)
        for path in study_dirs
        if path.name.isdigit()
    }

    csv_study_ids = set(
        series_df["study_id"].astype(int)
    )

    missing_filesystem_studies = (
        csv_study_ids
        - filesystem_study_ids
    )

    orphan_filesystem_studies = (
        filesystem_study_ids
        - csv_study_ids
    )

    print(
        f"CSV studies                       : "
        f"{len(csv_study_ids)}"
    )

    print(
        f"Filesystem studies                : "
        f"{len(filesystem_study_ids)}"
    )

    print(
        f"CSV studies missing on filesystem : "
        f"{len(missing_filesystem_studies)}"
    )

    print(
        f"Filesystem-only studies           : "
        f"{len(orphan_filesystem_studies)}"
    )

    # ========================================================================
    # DICOM / COORDINATE VALIDATION
    # ========================================================================

    header("DICOM GEOMETRY & COORDINATE VALIDATION")

    validation_rows = []

    header_cache = {}

    total_coordinates = len(
        coordinates_df
    )

    for index, row in coordinates_df.iterrows():

        study_id = safe_int(
            row["study_id"]
        )

        series_id = safe_int(
            row["series_id"]
        )

        instance_number = safe_int(
            row["instance_number"]
        )

        x = safe_float(
            row["x"]
        )

        y = safe_float(
            row["y"]
        )

        series_dir = (
            TRAIN_IMAGES
            / str(study_id)
            / str(series_id)
        )

        result = {
            "row_index": int(index),
            "study_id": study_id,
            "series_id": series_id,
            "instance_number": instance_number,
            "condition": str(row["condition"]),
            "level": str(row["level"]),
            "x": x,
            "y": y,
            "series_exists": False,
            "instance_file_exists": False,
            "dicom_readable": False,
            "rows": None,
            "columns": None,
            "coordinate_in_bounds": False,
            "instance_number_matches_dicom": False,
            "pixel_spacing_row": None,
            "pixel_spacing_column": None,
            "slice_thickness": None,
            "has_image_position": False,
            "has_image_orientation": False,
            "error": "",
        }

        if not series_dir.exists():

            result["error"] = (
                "series_directory_missing"
            )

            validation_rows.append(
                result
            )

            continue

        result["series_exists"] = True

        instance_file = find_dicom_instance(
            series_dir,
            instance_number,
        )

        if instance_file is None:

            result["error"] = (
                "instance_file_missing"
            )

            validation_rows.append(
                result
            )

            continue

        result[
            "instance_file_exists"
        ] = True

        cache_key = str(
            instance_file
        )

        if cache_key not in header_cache:

            header_cache[
                cache_key
            ] = load_dicom_header(
                instance_file
            )

        ds = header_cache[
            cache_key
        ]

        if ds is None:

            result["error"] = (
                "dicom_unreadable"
            )

            validation_rows.append(
                result
            )

            continue

        result[
            "dicom_readable"
        ] = True

        rows = safe_int(
            getattr(
                ds,
                "Rows",
                None,
            )
        )

        columns = safe_int(
            getattr(
                ds,
                "Columns",
                None,
            )
        )

        result["rows"] = rows
        result["columns"] = columns

        dicom_instance = safe_int(
            getattr(
                ds,
                "InstanceNumber",
                None,
            )
        )

        if (
            dicom_instance is not None
            and instance_number is not None
        ):

            result[
                "instance_number_matches_dicom"
            ] = (
                dicom_instance
                == instance_number
            )

        if (
            x is not None
            and y is not None
            and rows is not None
            and columns is not None
        ):

            result[
                "coordinate_in_bounds"
            ] = (
                0 <= x < columns
                and
                0 <= y < rows
            )

        try:

            pixel_spacing = list(
                ds.PixelSpacing
            )

            if len(pixel_spacing) >= 2:

                result[
                    "pixel_spacing_row"
                ] = safe_float(
                    pixel_spacing[0]
                )

                result[
                    "pixel_spacing_column"
                ] = safe_float(
                    pixel_spacing[1]
                )

        except Exception:
            pass

        result[
            "slice_thickness"
        ] = safe_float(
            getattr(
                ds,
                "SliceThickness",
                None,
            )
        )

        result[
            "has_image_position"
        ] = hasattr(
            ds,
            "ImagePositionPatient",
        )

        result[
            "has_image_orientation"
        ] = hasattr(
            ds,
            "ImageOrientationPatient",
        )

        validation_rows.append(
            result
        )

        if (
            (index + 1) % 500 == 0
            or index + 1 == total_coordinates
        ):

            print(
                f"Audited {index + 1:5d} / "
                f"{total_coordinates}"
            )

    validation_df = pd.DataFrame(
        validation_rows
    )

    # ========================================================================
    # VALIDATION SUMMARY
    # ========================================================================

    header("ANNOTATION -> DICOM VALIDATION SUMMARY")

    validation_summary = {
        "coordinate_rows": len(
            validation_df
        ),
        "series_directory_missing": int(
            (
                ~validation_df[
                    "series_exists"
                ]
            ).sum()
        ),
        "instance_file_missing": int(
            (
                validation_df[
                    "series_exists"
                ]
                &
                ~validation_df[
                    "instance_file_exists"
                ]
            ).sum()
        ),
        "dicom_unreadable": int(
            (
                validation_df[
                    "instance_file_exists"
                ]
                &
                ~validation_df[
                    "dicom_readable"
                ]
            ).sum()
        ),
        "coordinate_in_bounds": int(
            validation_df[
                "coordinate_in_bounds"
            ].sum()
        ),
        "coordinate_out_of_bounds": int(
            (
                validation_df[
                    "dicom_readable"
                ]
                &
                ~validation_df[
                    "coordinate_in_bounds"
                ]
            ).sum()
        ),
        "instance_number_matches_dicom": int(
            validation_df[
                "instance_number_matches_dicom"
            ].sum()
        ),
        "image_position_available": int(
            validation_df[
                "has_image_position"
            ].sum()
        ),
        "image_orientation_available": int(
            validation_df[
                "has_image_orientation"
            ].sum()
        ),
    }

    for key, value in validation_summary.items():

        print(
            f"{key:<38}: {value}"
        )

    # ========================================================================
    # SPINAL CANAL SUBSET
    # ========================================================================

    header("SPINAL CANAL ANNOTATION AUDIT")

    spinal_canal_mask = (
        coordinates_df[
            "condition"
        ]
        .astype(str)
        .str.contains(
            "Spinal Canal",
            case=False,
            na=False,
        )
    )

    spinal_canal_df = (
        coordinates_df[
            spinal_canal_mask
        ].copy()
    )

    print(
        f"Spinal canal coordinate rows      : "
        f"{len(spinal_canal_df)}"
    )

    print(
        f"Spinal canal studies              : "
        f"{spinal_canal_df['study_id'].nunique()}"
    )

    print(
        f"Spinal canal series               : "
        f"{spinal_canal_df['series_id'].nunique()}"
    )

    print("\nSpinal canal annotations by level")

    print(
        spinal_canal_df[
            "level"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================================
    # ANNOTATION CONDITION / LEVEL TABLE
    # ========================================================================

    annotation_level_summary = (
        coordinates_df
        .groupby(
            [
                "condition",
                "level",
            ]
        )
        .size()
        .reset_index(
            name="annotation_count"
        )
    )

    # ========================================================================
    # SAVE CSV OUTPUTS
    # ========================================================================

    header("SAVING ANALYSIS TABLES")

    series_distribution_df = (
        series_distribution
        .rename("series_count")
        .reset_index()
        .rename(
            columns={
                "index":
                "series_description"
            }
        )
    )

    annotation_series_df = (
        annotation_series_distribution
        .rename("annotation_count")
        .reset_index()
        .rename(
            columns={
                "index":
                "series_description"
            }
        )
    )

    series_distribution_df.to_csv(
        OUTPUT_DIR
        / "rsna_series_distribution.csv",
        index=False,
    )

    disease_distribution.to_csv(
        OUTPUT_DIR
        / "rsna_disease_label_distribution.csv",
        index=False,
    )

    annotation_series_df.to_csv(
        OUTPUT_DIR
        / "rsna_annotation_series_distribution.csv",
        index=False,
    )

    annotation_level_summary.to_csv(
        OUTPUT_DIR
        / "rsna_annotation_level_condition_summary.csv",
        index=False,
    )

    validation_df.to_csv(
        OUTPUT_DIR
        / "rsna_coordinate_dicom_validation.csv",
        index=False,
    )

    pd.DataFrame(
        filesystem_rows
    ).to_csv(
        OUTPUT_DIR
        / "rsna_filesystem_series_image_counts.csv",
        index=False,
    )

    spinal_canal_df.to_csv(
        OUTPUT_DIR
        / "rsna_spinal_canal_annotations.csv",
        index=False,
    )

    # ========================================================================
    # JSON SUMMARY
    # ========================================================================

    elapsed = (
        time.time()
        - start_time
    )

    summary = {
        "phase":
            "Phase 4 - Part 1",

        "title":
            "RSNA-only Dataset and Annotation Audit",

        "dataset": {

            "filesystem_studies":
                filesystem_studies,

            "filesystem_series":
                filesystem_series,

            "filesystem_dicom_images":
                total_dicom_images,

            "csv_unique_studies":
                int(
                    series_df[
                        "study_id"
                    ].nunique()
                ),

            "csv_unique_series":
                int(
                    series_df[
                        "series_id"
                    ].nunique()
                ),

            "train_csv_rows":
                len(train_df),

            "coordinate_rows":
                len(coordinates_df),
        },

        "coordinate_quality":
            coordinate_quality,

        "filesystem_consistency": {

            "csv_studies_missing_on_filesystem":
                len(
                    missing_filesystem_studies
                ),

            "filesystem_only_studies":
                len(
                    orphan_filesystem_studies
                ),
        },

        "validation":
            validation_summary,

        "spinal_canal": {

            "coordinate_rows":
                len(
                    spinal_canal_df
                ),

            "studies":
                int(
                    spinal_canal_df[
                        "study_id"
                    ].nunique()
                ),

            "series":
                int(
                    spinal_canal_df[
                        "series_id"
                    ].nunique()
                ),
        },

        "methodology": {

            "spider_used":
                False,

            "ground_truth_masks_used":
                False,

            "segmentation_masks_created":
                False,

            "training_performed":
                False,

            "model_weights_modified":
                False,

        },

        "execution_seconds":
            elapsed,
    }

    with open(
        OUTPUT_DIR
        / "phase4_part1_rsna_audit_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )

    # ========================================================================
    # TEXT REPORT
    # ========================================================================

    report = []

    report.append(
        "PHASE 4 - PART 1"
    )

    report.append(
        "RSNA-ONLY SEGMENTATION DATASET & ANNOTATION AUDIT"
    )

    report.append("")

    report.append(
        f"RSNA root: {RSNA_ROOT}"
    )

    report.append("")

    report.append(
        "DATASET COUNTS"
    )

    report.append(
        f"Filesystem studies: {filesystem_studies}"
    )

    report.append(
        f"Filesystem series: {filesystem_series}"
    )

    report.append(
        f"DICOM images: {total_dicom_images}"
    )

    report.append(
        f"CSV unique studies: "
        f"{series_df['study_id'].nunique()}"
    )

    report.append(
        f"CSV unique series: "
        f"{series_df['series_id'].nunique()}"
    )

    report.append(
        f"Coordinate rows: "
        f"{len(coordinates_df)}"
    )

    report.append("")

    report.append(
        "COORDINATE QUALITY"
    )

    for key, value in coordinate_quality.items():

        report.append(
            f"{key}: {value}"
        )

    report.append("")

    report.append(
        "DICOM VALIDATION"
    )

    for key, value in validation_summary.items():

        report.append(
            f"{key}: {value}"
        )

    report.append("")

    report.append(
        "SPINAL CANAL ANNOTATIONS"
    )

    report.append(
        f"Rows: {len(spinal_canal_df)}"
    )

    report.append(
        f"Studies: "
        f"{spinal_canal_df['study_id'].nunique()}"
    )

    report.append(
        f"Series: "
        f"{spinal_canal_df['series_id'].nunique()}"
    )

    report.append("")

    report.append(
        "METHODOLOGY"
    )

    report.append(
        "SPIDER used: NO"
    )

    report.append(
        "Ground-truth segmentation masks used: NO"
    )

    report.append(
        "Segmentation masks created: NO"
    )

    report.append(
        "Training performed: NO"
    )

    report.append(
        "Model weights modified: NO"
    )

    report.append("")

    report.append(
        "NEXT STEP"
    )

    report.append(
        "Validate RSNA coordinate-to-DICOM geometry "
        "visually before creating any segmentation "
        "or localization targets."
    )

    report.append("")

    report.append(
        f"Execution time: "
        f"{elapsed / 60:.2f} minutes"
    )

    with open(
        OUTPUT_DIR
        / "phase4_part1_rsna_audit_report.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(report)
        )

    # ========================================================================
    # FINAL
    # ========================================================================

    header("PART 1 COMPLETE")

    print(
        f"Studies                         : "
        f"{filesystem_studies}"
    )

    print(
        f"Series                          : "
        f"{filesystem_series}"
    )

    print(
        f"DICOM images                    : "
        f"{total_dicom_images}"
    )

    print(
        f"Coordinate annotations          : "
        f"{len(coordinates_df)}"
    )

    print(
        f"Spinal canal annotations        : "
        f"{len(spinal_canal_df)}"
    )

    print(
        f"Coordinates in bounds            : "
        f"{validation_summary['coordinate_in_bounds']}"
    )

    print(
        f"Coordinates out of bounds        : "
        f"{validation_summary['coordinate_out_of_bounds']}"
    )

    print("")
    print(
        "SPIDER used             : NO"
    )
    print(
        "Training performed      : NO"
    )
    print(
        "Masks created           : NO"
    )
    print(
        "Model weights modified  : NO"
    )

    print("")
    print(
        "OUTPUT DIRECTORY"
    )

    print(
        OUTPUT_DIR
    )

    print("\n" + "=" * 78)
    print(
        "PHASE 4 - PART 1 COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":

    main()