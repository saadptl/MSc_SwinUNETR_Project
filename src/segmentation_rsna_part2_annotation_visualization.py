"""
==============================================================================
PHASE 4 - PART 2
RSNA ANNOTATION VISUALIZATION AND ANATOMICAL LOCALIZATION AUDIT
==============================================================================

Purpose
-------
This script performs a visual audit of the RSNA 2024 lumbar-spine annotations.

It does NOT:
    - train a model
    - modify model weights
    - create segmentation masks
    - use the SPIDER dataset
    - use ground-truth segmentation masks

It DOES:
    1. Load RSNA train.csv
    2. Load train_label_coordinates.csv
    3. Load train_series_descriptions.csv
    4. Match annotation -> study -> series -> DICOM instance
    5. Read the corresponding DICOM image
    6. Plot the RSNA (x, y) annotation
    7. Display condition and lumbar level
    8. Verify coordinate validity
    9. Save individual visualizations
   10. Create a summary CSV/JSON/report

The purpose is to establish a reliable foundation for later
RSNA-only segmentation-target generation.

==============================================================================

IMPORTANT
---------
RSNA provides coordinate annotations, NOT ready-made pixel-wise
segmentation masks.

Therefore this part is intentionally a VISUAL AUDIT.

Do not interpret the output of this script as segmentation ground truth.
==============================================================================

"""

from __future__ import annotations

import json
import math
import random
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    import pydicom
except ImportError:
    raise ImportError(
        "pydicom is required.\n"
        "Install it using:\n"
        "pip install pydicom"
    )


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES_DIR = RSNA_ROOT / "train_images"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part2_annotation_visualization"
)


# =============================================================================
# INPUT FILES
# =============================================================================

TRAIN_CSV = RSNA_ROOT / "train.csv"

COORDINATE_CSV = RSNA_ROOT / "train_label_coordinates.csv"

SERIES_CSV = RSNA_ROOT / "train_series_descriptions.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Number of representative studies to visualize.
MAX_STUDIES = 20

# Maximum number of annotation images to save per study.
MAX_ANNOTATIONS_PER_STUDY = 10

# Random seed for reproducibility.
RANDOM_SEED = 42

# Image display window.
FIGURE_SIZE = (9, 9)

# Whether to create an overview contact sheet.
CREATE_CONTACT_SHEET = True

# Maximum number of contact-sheet images.
CONTACT_SHEET_MAX_IMAGES = 20

# Preferred series order.
PREFERRED_SERIES = [
    "Sagittal T2/STIR",
    "Sagittal T1",
    "Axial T2",
]


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_int(value) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def safe_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


# =============================================================================
# DICOM HELPERS
# =============================================================================


def find_dicom_for_instance(
    series_dir: Path,
    instance_number: int,
) -> Optional[Path]:
    """
    Find the DICOM file whose DICOM InstanceNumber matches instance_number.

    We do not assume that the filename itself is the instance number.
    """

    if not series_dir.exists():
        return None

    # First try the common filename convention.
    direct_file = series_dir / f"{instance_number}.dcm"

    if direct_file.exists():
        try:
            ds = pydicom.dcmread(
                str(direct_file),
                stop_before_pixels=True,
                force=True,
            )

            dicom_instance = safe_int(
                getattr(ds, "InstanceNumber", None)
            )

            if dicom_instance == instance_number:
                return direct_file
        except Exception:
            pass

    # Reliable fallback: inspect DICOM metadata.
    try:
        files = sorted(series_dir.glob("*.dcm"))
    except Exception:
        files = []

    for file_path in files:
        try:
            ds = pydicom.dcmread(
                str(file_path),
                stop_before_pixels=True,
                force=True,
            )

            dicom_instance = safe_int(
                getattr(ds, "InstanceNumber", None)
            )

            if dicom_instance == instance_number:
                return file_path

        except Exception:
            continue

    return None


def load_dicom_image(file_path: Path):
    """
    Read a DICOM image and apply basic presentation handling.

    Returns:
        image: numpy array
        ds: DICOM dataset
    """

    ds = pydicom.dcmread(
        str(file_path),
        force=True,
    )

    image = ds.pixel_array.astype(np.float32)

    # Handle MONOCHROME1.
    photometric = str(
        getattr(ds, "PhotometricInterpretation", "")
    ).upper()

    if photometric == "MONOCHROME1":
        image = np.max(image) - image

    # Apply rescale slope/intercept when available.
    slope = safe_float(
        getattr(ds, "RescaleSlope", 1.0)
    )

    intercept = safe_float(
        getattr(ds, "RescaleIntercept", 0.0)
    )

    if slope is None:
        slope = 1.0

    if intercept is None:
        intercept = 0.0

    image = image * slope + intercept

    return image, ds


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Robust percentile normalization for visualization.
    """

    image = np.asarray(image, dtype=np.float32)

    finite = np.isfinite(image)

    if not np.any(finite):
        return np.zeros_like(image)

    valid = image[finite]

    low = np.percentile(valid, 1.0)
    high = np.percentile(valid, 99.0)

    if high <= low:
        low = float(valid.min())
        high = float(valid.max())

    if high <= low:
        return np.zeros_like(image)

    normalized = (image - low) / (high - low)

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return normalized


# =============================================================================
# CSV LOADING
# =============================================================================


def validate_required_files() -> None:

    required = [
        RSNA_ROOT,
        TRAIN_IMAGES_DIR,
        TRAIN_CSV,
        COORDINATE_CSV,
        SERIES_CSV,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:

        print_header("MISSING REQUIRED FILES")

        for path in missing:
            print(path)

        raise FileNotFoundError(
            "One or more RSNA dataset files/directories are missing."
        )


def load_rsna_tables():

    print_header("LOADING RSNA CSV FILES")

    train_df = pd.read_csv(TRAIN_CSV)

    coordinates_df = pd.read_csv(COORDINATE_CSV)

    series_df = pd.read_csv(SERIES_CSV)

    print(f"train.csv rows                : {len(train_df)}")
    print(
        "train_label_coordinates rows : "
        f"{len(coordinates_df)}"
    )
    print(
        "train_series_descriptions rows: "
        f"{len(series_df)}"
    )

    return train_df, coordinates_df, series_df


# =============================================================================
# TABLE PREPARATION
# =============================================================================


def prepare_coordinate_table(
    coordinates_df: pd.DataFrame,
    series_df: pd.DataFrame,
) -> pd.DataFrame:

    df = coordinates_df.copy()

    # Normalize identifier columns.
    df["study_id"] = df["study_id"].astype(str)
    df["series_id"] = df["series_id"].astype(str)

    series_lookup = series_df.copy()

    series_lookup["study_id"] = (
        series_lookup["study_id"].astype(str)
    )

    series_lookup["series_id"] = (
        series_lookup["series_id"].astype(str)
    )

    series_lookup = series_lookup[
        [
            "study_id",
            "series_id",
            "series_description",
        ]
    ].drop_duplicates()

    df = df.merge(
        series_lookup,
        on=["study_id", "series_id"],
        how="left",
        validate="many_to_one",
    )

    df["instance_number_int"] = (
        df["instance_number"]
        .apply(safe_int)
    )

    df["x_float"] = (
        df["x"]
        .apply(safe_float)
    )

    df["y_float"] = (
        df["y"]
        .apply(safe_float)
    )

    return df


# =============================================================================
# SERIES SELECTION
# =============================================================================


def select_series_for_study(
    study_id: str,
    study_annotations: pd.DataFrame,
) -> List[Tuple[str, str]]:

    """
    Return series ordered according to our preferred visualization order.

    A study can contain:
        Sagittal T1
        Sagittal T2/STIR
        Axial T2

    We retain every series that actually contains annotations.
    """

    rows = []

    unique_series = (
        study_annotations[
            [
                "series_id",
                "series_description",
            ]
        ]
        .drop_duplicates()
        .to_dict("records")
    )

    preference = {
        name: index
        for index, name in enumerate(PREFERRED_SERIES)
    }

    for row in unique_series:

        series_id = str(row["series_id"])

        description = str(
            row["series_description"]
        )

        priority = preference.get(
            description,
            len(PREFERRED_SERIES),
        )

        rows.append(
            (
                priority,
                description,
                series_id,
            )
        )

    rows.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2],
        )
    )

    return [
        (description, series_id)
        for _, description, series_id in rows
    ]


# =============================================================================
# ANNOTATION VALIDATION
# =============================================================================


def validate_annotation_against_image(
    x: float,
    y: float,
    image_shape: Tuple[int, ...],
) -> Tuple[bool, str]:

    if len(image_shape) < 2:
        return False, "image_dimension_error"

    height = image_shape[0]
    width = image_shape[1]

    if x < 0 or x >= width:
        return False, "x_out_of_bounds"

    if y < 0 or y >= height:
        return False, "y_out_of_bounds"

    return True, "in_bounds"


# =============================================================================
# VISUALIZATION
# =============================================================================


def create_annotation_visualization(
    image: np.ndarray,
    x: float,
    y: float,
    condition: str,
    level: str,
    study_id: str,
    series_id: str,
    series_description: str,
    instance_number: int,
    output_path: Path,
) -> Dict:

    display_image = normalize_for_display(image)

    valid, status = validate_annotation_against_image(
        x,
        y,
        image.shape,
    )

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE
    )

    ax.imshow(
        display_image,
        cmap="gray",
    )

    if valid:

        ax.scatter(
            [x],
            [y],
            s=120,
            marker="+",
            linewidths=3,
        )

        # Circle around the coordinate.
        circle = plt.Circle(
            (x, y),
            radius=20,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(circle)

    else:

        ax.text(
            0.5,
            0.5,
            "ANNOTATION OUT OF BOUNDS",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )

    title = (
        f"RSNA Annotation Audit\n"
        f"Study: {study_id} | "
        f"Level: {level}\n"
        f"Condition: {condition}\n"
        f"Series: {series_description}\n"
        f"Instance: {instance_number}\n"
        f"Coordinate: ({x:.1f}, {y:.1f})"
    )

    ax.set_title(title)

    ax.set_xlabel("X / Column")

    ax.set_ylabel("Y / Row")

    ax.set_xlim(
        0,
        image.shape[1] - 1,
    )

    ax.set_ylim(
        image.shape[0] - 1,
        0,
    )

    ax.grid(
        False
    )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "study_id": study_id,
        "series_id": series_id,
        "series_description": series_description,
        "instance_number": instance_number,
        "condition": condition,
        "level": level,
        "x": float(x),
        "y": float(y),
        "image_height": int(image.shape[0]),
        "image_width": int(image.shape[1]),
        "coordinate_status": status,
        "visualization_path": str(output_path),
    }


# =============================================================================
# STUDY SAMPLING
# =============================================================================


def choose_representative_studies(
    coordinates_df: pd.DataFrame,
    train_df: pd.DataFrame,
    max_studies: int,
) -> List[str]:

    """
    Select representative studies.

    Selection attempts to cover:
        - different conditions
        - different levels
        - different imaging series
        - different disease severity

    No segmentation information is used.
    """

    rng = random.Random(
        RANDOM_SEED
    )

    available_studies = sorted(
        coordinates_df[
            "study_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not available_studies:
        return []

    selected = []

    # -------------------------------------------------------------------------
    # 1. Sample across annotation conditions.
    # -------------------------------------------------------------------------

    conditions = sorted(
        coordinates_df[
            "condition"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    for condition in conditions:

        subset = coordinates_df[
            coordinates_df["condition"].astype(str)
            == condition
        ]

        studies = (
            subset["study_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        rng.shuffle(studies)

        for study in studies:

            if study not in selected:

                selected.append(study)
                break

            if len(selected) >= max_studies:
                return selected[:max_studies]

    # -------------------------------------------------------------------------
    # 2. Sample across levels.
    # -------------------------------------------------------------------------

    levels = sorted(
        coordinates_df[
            "level"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    for level in levels:

        subset = coordinates_df[
            coordinates_df["level"].astype(str)
            == level
        ]

        studies = (
            subset["study_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        rng.shuffle(studies)

        for study in studies:

            if study not in selected:

                selected.append(study)
                break

            if len(selected) >= max_studies:
                return selected[:max_studies]

    # -------------------------------------------------------------------------
    # 3. Fill remaining positions randomly.
    # -------------------------------------------------------------------------

    remaining = [
        study
        for study in available_studies
        if study not in selected
    ]

    rng.shuffle(
        remaining
    )

    selected.extend(
        remaining[
            : max(0, max_studies - len(selected))
        ]
    )

    return selected[:max_studies]


# =============================================================================
# CONTACT SHEET
# =============================================================================


def create_contact_sheet(
    records: List[Dict],
    output_path: Path,
    max_images: int,
) -> None:

    if not records:
        return

    records = records[:max_images]

    columns = 4

    rows = math.ceil(
        len(records) / columns
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 4 * rows),
    )

    axes = np.asarray(
        axes
    ).reshape(
        rows,
        columns,
    )

    for ax in axes.flat:
        ax.axis("off")

    for index, record in enumerate(records):

        row = index // columns
        col = index % columns

        ax = axes[row, col]

        try:

            image, _ = load_dicom_image(
                Path(
                    record["dicom_path"]
                )
            )

            display_image = normalize_for_display(
                image
            )

            ax.imshow(
                display_image,
                cmap="gray",
            )

            x = record["x"]
            y = record["y"]

            ax.scatter(
                [x],
                [y],
                s=100,
                marker="+",
                linewidths=2,
            )

            ax.set_title(
                f'{record["study_id"]}\n'
                f'{record["level"]} | '
                f'{record["series_description"]}',
                fontsize=9,
            )

            ax.axis("off")

        except Exception as exc:

            ax.text(
                0.5,
                0.5,
                f"Load error\n{exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    fig.suptitle(
        "RSNA Part 2 - Annotation Visualization Overview",
        fontsize=16,
    )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN AUDIT
# =============================================================================


def main():

    print_header(
        "PHASE 4 - PART 2"
    )

    print(
        "RSNA ANNOTATION VISUALIZATION "
        "AND ANATOMICAL LOCALIZATION AUDIT"
    )

    print_header(
        "PROJECT PATHS"
    )

    print(
        f"PROJECT ROOT\n{PROJECT_ROOT}"
    )

    print(
        f"\nRSNA DATASET\n{RSNA_ROOT}"
    )

    print(
        f"\nTRAIN IMAGES\n{TRAIN_IMAGES_DIR}"
    )

    print(
        f"\nOUTPUT DIRECTORY\n{OUTPUT_DIR}"
    )

    # -------------------------------------------------------------------------
    # Validate paths.
    # -------------------------------------------------------------------------

    print_header(
        "DATASET PATH VALIDATION"
    )

    validate_required_files()

    print(
        "RSNA root                    : FOUND"
    )

    print(
        "train_images                 : FOUND"
    )

    print(
        "train.csv                    : FOUND"
    )

    print(
        "train_label_coordinates.csv  : FOUND"
    )

    print(
        "train_series_descriptions.csv: FOUND"
    )

    # -------------------------------------------------------------------------
    # Load tables.
    # -------------------------------------------------------------------------

    train_df, coordinates_df, series_df = (
        load_rsna_tables()
    )

    # Avoid unused-variable warnings.
    _ = train_df

    coordinates_df = prepare_coordinate_table(
        coordinates_df,
        series_df,
    )

    # -------------------------------------------------------------------------
    # Basic coordinate audit.
    # -------------------------------------------------------------------------

    print_header(
        "ANNOTATION QUALITY PRE-CHECK"
    )

    total = len(coordinates_df)

    missing_x = int(
        coordinates_df["x_float"]
        .isna()
        .sum()
    )

    missing_y = int(
        coordinates_df["y_float"]
        .isna()
        .sum()
    )

    missing_instance = int(
        coordinates_df[
            "instance_number_int"
        ]
        .isna()
        .sum()
    )

    missing_series_description = int(
        coordinates_df[
            "series_description"
        ]
        .isna()
        .sum()
    )

    print(
        f"Total annotation rows          : {total}"
    )

    print(
        f"Missing X                      : {missing_x}"
    )

    print(
        f"Missing Y                      : {missing_y}"
    )

    print(
        f"Missing instance number        : {missing_instance}"
    )

    print(
        "Missing series description     : "
        f"{missing_series_description}"
    )

    # -------------------------------------------------------------------------
    # Representative study selection.
    # -------------------------------------------------------------------------

    print_header(
        "SELECTING REPRESENTATIVE STUDIES"
    )

    selected_studies = (
        choose_representative_studies(
            coordinates_df,
            train_df,
            MAX_STUDIES,
        )
    )

    print(
        f"Selected studies: "
        f"{len(selected_studies)}"
    )

    for index, study_id in enumerate(
        selected_studies,
        start=1,
    ):
        print(
            f"{index:02d}. {study_id}"
        )

    # -------------------------------------------------------------------------
    # Process annotations.
    # -------------------------------------------------------------------------

    print_header(
        "PROCESSING ANNOTATIONS"
    )

    result_records = []

    contact_records = []

    processed_annotations = 0

    successful_annotations = 0

    failed_annotations = 0

    out_of_bounds = 0

    study_errors = 0

    for study_index, study_id in enumerate(
        selected_studies,
        start=1,
    ):

        print(
            f"\n[{study_index}/{len(selected_studies)}] "
            f"Study {study_id}"
        )

        study_annotations = coordinates_df[
            coordinates_df["study_id"].astype(str)
            == str(study_id)
        ].copy()

        if study_annotations.empty:

            print(
                "  No annotations found."
            )

            continue

        series_list = select_series_for_study(
            study_id,
            study_annotations,
        )

        print(
            f"  Annotation rows: "
            f"{len(study_annotations)}"
        )

        print(
            f"  Annotated series: "
            f"{len(series_list)}"
        )

        annotations_processed_for_study = 0

        try:

            for (
                series_description,
                series_id,
            ) in series_list:

                series_annotations = (
                    study_annotations[
                        study_annotations[
                            "series_id"
                        ].astype(str)
                        == str(series_id)
                    ]
                    .copy()
                )

                if series_annotations.empty:
                    continue

                print(
                    f"  Series: "
                    f"{series_description} "
                    f"({series_id})"
                )

                series_dir = (
                    TRAIN_IMAGES_DIR
                    / str(study_id)
                    / str(series_id)
                )

                if not series_dir.exists():

                    print(
                        "    ERROR: series directory "
                        "does not exist."
                    )

                    for _, row in (
                        series_annotations.iterrows()
                    ):

                        result_records.append(
                            {
                                "study_id": study_id,
                                "series_id": series_id,
                                "series_description":
                                    series_description,
                                "instance_number":
                                    row[
                                        "instance_number_int"
                                    ],
                                "condition":
                                    row["condition"],
                                "level":
                                    row["level"],
                                "x":
                                    row["x_float"],
                                "y":
                                    row["y_float"],
                                "dicom_path":
                                    "",
                                "status":
                                    "series_directory_missing",
                            }
                        )

                    failed_annotations += len(
                        series_annotations
                    )

                    continue

                # Limit number of annotations visualized
                # within one series.
                series_annotations = (
                    series_annotations
                    .head(
                        MAX_ANNOTATIONS_PER_STUDY
                    )
                )

                for annotation_index, row in (
                    series_annotations.iterrows()
                ):

                    processed_annotations += 1

                    instance_number = (
                        row[
                            "instance_number_int"
                        ]
                    )

                    x = row["x_float"]
                    y = row["y_float"]

                    condition = str(
                        row["condition"]
                    )

                    level = str(
                        row["level"]
                    )

                    if (
                        instance_number is None
                        or x is None
                        or y is None
                    ):

                        failed_annotations += 1

                        result_records.append(
                            {
                                "study_id": study_id,
                                "series_id": series_id,
                                "series_description":
                                    series_description,
                                "instance_number":
                                    instance_number,
                                "condition":
                                    condition,
                                "level":
                                    level,
                                "x": x,
                                "y": y,
                                "dicom_path":
                                    "",
                                "status":
                                    "invalid_annotation_values",
                            }
                        )

                        continue

                    dicom_path = (
                        find_dicom_for_instance(
                            series_dir,
                            instance_number,
                        )
                    )

                    if dicom_path is None:

                        print(
                            f"    Instance "
                            f"{instance_number}: "
                            "DICOM NOT FOUND"
                        )

                        failed_annotations += 1

                        result_records.append(
                            {
                                "study_id": study_id,
                                "series_id": series_id,
                                "series_description":
                                    series_description,
                                "instance_number":
                                    instance_number,
                                "condition":
                                    condition,
                                "level":
                                    level,
                                "x":
                                    x,
                                "y":
                                    y,
                                "dicom_path":
                                    "",
                                "status":
                                    "dicom_instance_missing",
                            }
                        )

                        continue

                    try:

                        image, ds = (
                            load_dicom_image(
                                dicom_path
                            )
                        )

                        valid, coordinate_status = (
                            validate_annotation_against_image(
                                x,
                                y,
                                image.shape,
                            )
                        )

                        if not valid:

                            out_of_bounds += 1

                            print(
                                f"    Instance "
                                f"{instance_number}: "
                                f"{coordinate_status}"
                            )

                        # Output filename.
                        safe_condition = (
                            condition
                            .replace("/", "_")
                            .replace(" ", "_")
                        )

                        safe_level = (
                            level
                            .replace("/", "_")
                        )

                        filename = (
                            f"{study_id}_"
                            f"{series_id}_"
                            f"inst_{instance_number}_"
                            f"{safe_condition}_"
                            f"{safe_level}.png"
                        )

                        output_path = (
                            OUTPUT_DIR
                            / "annotations"
                            / str(study_id)
                            / filename
                        )

                        visualization_record = (
                            create_annotation_visualization(
                                image=image,
                                x=x,
                                y=y,
                                condition=condition,
                                level=level,
                                study_id=str(
                                    study_id
                                ),
                                series_id=str(
                                    series_id
                                ),
                                series_description=(
                                    series_description
                                ),
                                instance_number=(
                                    instance_number
                                ),
                                output_path=(
                                    output_path
                                ),
                            )
                        )

                        visualization_record[
                            "dicom_path"
                        ] = str(
                            dicom_path
                        )

                        visualization_record[
                            "status"
                        ] = (
                            "success"
                            if valid
                            else coordinate_status
                        )

                        result_records.append(
                            visualization_record
                        )

                        contact_records.append(
                            visualization_record
                        )

                        if valid:
                            successful_annotations += 1
                        else:
                            failed_annotations += 1

                        annotations_processed_for_study += 1

                        print(
                            f"    ✓ Level {level} | "
                            f"Instance {instance_number} | "
                            f"({x:.1f}, {y:.1f})"
                        )

                    except Exception as exc:

                        failed_annotations += 1

                        print(
                            f"    ERROR reading "
                            f"instance {instance_number}: "
                            f"{exc}"
                        )

                        result_records.append(
                            {
                                "study_id": study_id,
                                "series_id": series_id,
                                "series_description":
                                    series_description,
                                "instance_number":
                                    instance_number,
                                "condition":
                                    condition,
                                "level":
                                    level,
                                "x":
                                    x,
                                "y":
                                    y,
                                "dicom_path":
                                    str(
                                        dicom_path
                                    ),
                                "status":
                                    "dicom_read_error",
                                "error":
                                    str(exc),
                            }
                        )

                # End annotation loop.

        except Exception as exc:

            study_errors += 1

            print(
                f"  STUDY ERROR: {exc}"
            )

            traceback.print_exc()

        print(
            f"  Visualized annotations: "
            f"{annotations_processed_for_study}"
        )

    # -------------------------------------------------------------------------
    # Save detailed table.
    # -------------------------------------------------------------------------

    print_header(
        "SAVING ANALYSIS TABLE"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        result_records
    )

    detailed_csv = (
        OUTPUT_DIR
        / "rsna_part2_annotation_visualization_results.csv"
    )

    results_df.to_csv(
        detailed_csv,
        index=False,
    )

    print(
        f"Saved: {detailed_csv}"
    )

    # -------------------------------------------------------------------------
    # Create contact sheet.
    # -------------------------------------------------------------------------

    if CREATE_CONTACT_SHEET:

        print_header(
            "CREATING CONTACT SHEET"
        )

        contact_sheet_path = (
            OUTPUT_DIR
            / "rsna_part2_annotation_contact_sheet.png"
        )

        create_contact_sheet(
            contact_records,
            contact_sheet_path,
            CONTACT_SHEET_MAX_IMAGES,
        )

        print(
            f"Saved: {contact_sheet_path}"
        )

    # -------------------------------------------------------------------------
    # Calculate summary.
    # -------------------------------------------------------------------------

    print_header(
        "VISUAL ANNOTATION SUMMARY"
    )

    if not results_df.empty:

        total_results = len(
            results_df
        )

        successful_results = int(
            (
                results_df["status"]
                == "success"
            ).sum()
        )

        missing_instances = int(
            (
                results_df["status"]
                == "dicom_instance_missing"
            ).sum()
        )

        missing_series = int(
            (
                results_df["status"]
                == "series_directory_missing"
            ).sum()
        )

        coordinate_errors = int(
            results_df[
                "status"
            ]
            .astype(str)
            .str.contains(
                "out_of_bounds",
                na=False,
            )
            .sum()
        )

    else:

        total_results = 0
        successful_results = 0
        missing_instances = 0
        missing_series = 0
        coordinate_errors = 0

    print(
        f"Studies selected              : "
        f"{len(selected_studies)}"
    )

    print(
        f"Annotations processed         : "
        f"{processed_annotations}"
    )

    print(
        f"Successful visualizations     : "
        f"{successful_results}"
    )

    print(
        f"Failed annotations             : "
        f"{failed_annotations}"
    )

    print(
        f"DICOM instances missing       : "
        f"{missing_instances}"
    )

    print(
        f"Series directories missing    : "
        f"{missing_series}"
    )

    print(
        f"Coordinate out-of-bounds      : "
        f"{coordinate_errors}"
    )

    print(
        f"Study-level errors             : "
        f"{study_errors}"
    )

    # -------------------------------------------------------------------------
    # Condition summary.
    # -------------------------------------------------------------------------

    print_header(
        "VISUALIZED CONDITIONS"
    )

    if not results_df.empty:

        condition_counts = (
            results_df[
                "condition"
            ]
            .value_counts()
        )

        print(
            condition_counts.to_string()
        )

    # -------------------------------------------------------------------------
    # Series summary.
    # -------------------------------------------------------------------------

    print_header(
        "VISUALIZED SERIES TYPES"
    )

    if not results_df.empty:

        series_counts = (
            results_df[
                "series_description"
            ]
            .value_counts()
        )

        print(
            series_counts.to_string()
        )

    # -------------------------------------------------------------------------
    # Level summary.
    # -------------------------------------------------------------------------

    print_header(
        "VISUALIZED LUMBAR LEVELS"
    )

    if not results_df.empty:

        level_counts = (
            results_df[
                "level"
            ]
            .value_counts()
        )

        print(
            level_counts.to_string()
        )

    # -------------------------------------------------------------------------
    # Save JSON summary.
    # -------------------------------------------------------------------------

    print_header(
        "SAVING FINAL SUMMARY"
    )

    summary = {
        "phase": 4,
        "part": 2,
        "title": (
            "RSNA Annotation Visualization "
            "and Anatomical Localization Audit"
        ),
        "dataset": "RSNA 2024 Lumbar Spine Degenerative Classification",
        "project_root": str(
            PROJECT_ROOT
        ),
        "rsna_root": str(
            RSNA_ROOT
        ),
        "studies_selected": int(
            len(selected_studies)
        ),
        "annotations_processed": int(
            processed_annotations
        ),
        "successful_visualizations": int(
            successful_results
        ),
        "failed_annotations": int(
            failed_annotations
        ),
        "dicom_instances_missing": int(
            missing_instances
        ),
        "series_directories_missing": int(
            missing_series
        ),
        "coordinate_out_of_bounds": int(
            coordinate_errors
        ),
        "study_errors": int(
            study_errors
        ),
        "spider_used": False,
        "training_performed": False,
        "masks_created": False,
        "model_weights_modified": False,
        "ground_truth_segmentation_used": False,
        "purpose": (
            "Visual verification of RSNA coordinate "
            "annotations before segmentation-target generation."
        ),
    }

    summary_json = (
        OUTPUT_DIR
        / "phase4_part2_annotation_visualization_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"Saved: {summary_json}"
    )

    # -------------------------------------------------------------------------
    # Save report.
    # -------------------------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "phase4_part2_annotation_visualization_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 4 - PART 2\n"
        )

        f.write(
            "RSNA ANNOTATION VISUALIZATION "
            "AND ANATOMICAL LOCALIZATION AUDIT\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            f"Project root:\n"
            f"{PROJECT_ROOT}\n\n"
        )

        f.write(
            f"RSNA dataset:\n"
            f"{RSNA_ROOT}\n\n"
        )

        f.write(
            "SUMMARY\n"
            + "-" * 78
            + "\n"
        )

        f.write(
            f"Studies selected          : "
            f"{len(selected_studies)}\n"
        )

        f.write(
            f"Annotations processed     : "
            f"{processed_annotations}\n"
        )

        f.write(
            f"Successful visualizations : "
            f"{successful_results}\n"
        )

        f.write(
            f"Failed annotations        : "
            f"{failed_annotations}\n"
        )

        f.write(
            f"Missing DICOM instances   : "
            f"{missing_instances}\n"
        )

        f.write(
            f"Missing series directories: "
            f"{missing_series}\n"
        )

        f.write(
            f"Out-of-bounds coordinates : "
            f"{coordinate_errors}\n"
        )

        f.write(
            f"Study-level errors        : "
            f"{study_errors}\n"
        )

        f.write(
            "\nMETHODOLOGICAL STATUS\n"
            + "-" * 78
            + "\n"
        )

        f.write(
            "SPIDER used               : NO\n"
        )

        f.write(
            "Training performed        : NO\n"
        )

        f.write(
            "Segmentation masks created: NO\n"
        )

        f.write(
            "Model weights modified    : NO\n"
        )

        f.write(
            "Ground-truth segmentation : NO\n"
        )

        f.write(
            "\nIMPORTANT INTERPRETATION\n"
            + "-" * 78
            + "\n"
        )

        f.write(
            "The RSNA dataset provides point-coordinate "
            "annotations rather than ready-made pixel-wise "
            "segmentation masks. This part therefore performs "
            "visual verification only. The results must be "
            "used to determine a defensible strategy for "
            "generating segmentation targets in subsequent "
            "parts.\n"
        )

    print(
        f"Saved: {report_path}"
    )

    # -------------------------------------------------------------------------
    # Final status.
    # -------------------------------------------------------------------------

    print_header(
        "PART 2 COMPLETE"
    )

    print(
        f"Studies visualized           : "
        f"{len(selected_studies)}"
    )

    print(
        f"Annotations processed        : "
        f"{processed_annotations}"
    )

    print(
        f"Successful visualizations    : "
        f"{successful_results}"
    )

    print(
        f"Coordinate out-of-bounds     : "
        f"{coordinate_errors}"
    )

    print(
        f"Missing DICOM instances      : "
        f"{missing_instances}"
    )

    print()
    print(
        "SPIDER used                  : NO"
    )

    print(
        "Training performed           : NO"
    )

    print(
        "Masks created                : NO"
    )

    print(
        "Model weights modified       : NO"
    )

    print()
    print(
        "OUTPUT DIRECTORY"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "=" * 78
    )

    print(
        "PHASE 4 - PART 2 COMPLETE"
    )

    print(
        "=" * 78
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()