"""
==============================================================================
PHASE 4 - PART 8
RSNA 3D SEGMENTATION DATASET CONSTRUCTION
==============================================================================

Purpose
-------
Construct an RSNA-only 3D segmentation dataset using:

    RSNA DICOM MRI series
        +
    RSNA point-derived pseudo-masks from Part 6
        +
    Part 7 pseudo-mask quality validation

IMPORTANT
---------
The Part 6 pseudo-masks are stored as compressed NumPy .npz files.

They are NOT manually drawn segmentation masks.

Spider dataset is intentionally NOT used.

No model training is performed.
No model weights are modified.

Classes
-------
0 : Background
1 : Spinal Canal Stenosis
2 : Left Neural Foraminal Narrowing
3 : Right Neural Foraminal Narrowing
4 : Left Subarticular Stenosis
5 : Right Subarticular Stenosis

The split is performed at STUDY level to prevent patient/study leakage.
==============================================================================

"""

from __future__ import annotations

import json
import math
import random
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pydicom
except ImportError:
    pydicom = None


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES_DIR = RSNA_ROOT / "train_images"

TRAIN_CSV = RSNA_ROOT / "train.csv"
COORDINATES_CSV = RSNA_ROOT / "train_label_coordinates.csv"
SERIES_CSV = RSNA_ROOT / "train_series_descriptions.csv"

PART6_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part6_pseudomask_generation"
)

PSEUDOMASK_DIR = PART6_DIR / "pseudo_masks"

PART7_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part7_pseudomask_quality_validation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
)

MANIFEST_DIR = OUTPUT_DIR / "manifests"
REPORT_DIR = OUTPUT_DIR / "reports"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EXPECTED_SERIES_TYPES = {
    "Sagittal T1",
    "Sagittal T2/STIR",
    "Axial T2",
}

CLASS_MAP = {
    "Spinal Canal Stenosis": 1,
    "Left Neural Foraminal Narrowing": 2,
    "Right Neural Foraminal Narrowing": 3,
    "Left Subarticular Stenosis": 4,
    "Right Subarticular Stenosis": 5,
}

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# =============================================================================
# DISPLAY
# =============================================================================

def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

def prepare_output_dirs() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# PATH VALIDATION
# =============================================================================

def validate_paths() -> None:

    header("DATASET PATH VALIDATION")

    required = {
        "RSNA root": RSNA_ROOT,
        "train_images": TRAIN_IMAGES_DIR,
        "train.csv": TRAIN_CSV,
        "train_label_coordinates.csv": COORDINATES_CSV,
        "train_series_descriptions.csv": SERIES_CSV,
        "Part 6 pseudo_masks": PSEUDOMASK_DIR,
        "Part 7 quality validation": PART7_DIR,
    }

    failed = False

    for name, path in required.items():

        found = path.exists()

        print(
            f"{name:<36}: "
            f"{'FOUND' if found else 'MISSING'}"
        )

        if not found:
            failed = True

    if failed:

        raise FileNotFoundError(
            "One or more required RSNA/Part 6/Part 7 paths are missing."
        )


# =============================================================================
# LOAD METADATA
# =============================================================================

def load_metadata():

    header("LOADING RSNA METADATA")

    coordinates = pd.read_csv(
        COORDINATES_CSV
    )

    series = pd.read_csv(
        SERIES_CSV
    )

    train = pd.read_csv(
        TRAIN_CSV
    )

    print(
        f"Coordinate rows       : {len(coordinates)}"
    )

    print(
        f"Series metadata rows  : {len(series)}"
    )

    print(
        f"Train label rows      : {len(train)}"
    )

    required_coordinates = {
        "study_id",
        "series_id",
        "instance_number",
        "condition",
        "level",
        "x",
        "y",
    }

    required_series = {
        "study_id",
        "series_id",
        "series_description",
    }

    missing_coordinates = (
        required_coordinates
        - set(coordinates.columns)
    )

    missing_series = (
        required_series
        - set(series.columns)
    )

    if missing_coordinates:

        raise ValueError(
            "Missing coordinate columns: "
            f"{sorted(missing_coordinates)}"
        )

    if missing_series:

        raise ValueError(
            "Missing series columns: "
            f"{sorted(missing_series)}"
        )

    coordinates["study_id"] = (
        coordinates["study_id"].astype(str)
    )

    coordinates["series_id"] = (
        coordinates["series_id"].astype(str)
    )

    series["study_id"] = (
        series["study_id"].astype(str)
    )

    series["series_id"] = (
        series["series_id"].astype(str)
    )

    return coordinates, series, train


# =============================================================================
# PART 7 VALIDATION INFORMATION
# =============================================================================

def load_part7_quality():

    header("LOADING PART 7 QUALITY VALIDATION")

    quality_file = (
        PART7_DIR
        / "rsna_part7_pseudomask_quality_summary.csv"
    )

    class_file = (
        PART7_DIR
        / "rsna_part7_class_quality_summary.csv"
    )

    flag_file = (
        PART7_DIR
        / "rsna_part7_quality_flag_summary.csv"
    )

    quality = None
    class_quality = None
    flags = None

    if quality_file.exists():

        quality = pd.read_csv(
            quality_file
        )

        print(
            f"Quality summary rows : {len(quality)}"
        )

    else:

        print(
            "Quality summary not found."
        )

    if class_file.exists():

        class_quality = pd.read_csv(
            class_file
        )

        print(
            f"Class quality rows   : {len(class_quality)}"
        )

    else:

        print(
            "Class quality summary not found."
        )

    if flag_file.exists():

        flags = pd.read_csv(
            flag_file
        )

        print(
            f"Quality flag rows    : {len(flags)}"
        )

    else:

        print(
            "Quality flag summary not found."
        )

    return (
        quality,
        class_quality,
        flags,
    )


# =============================================================================
# PSEUDO-MASK DISCOVERY
# =============================================================================

def discover_pseudomasks(
    series_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Discover Part 6 pseudo-mask .npz files and robustly map each
    pseudo-mask to the authoritative RSNA series_id.

    Part 6 stores pseudo-masks as .npz files.

    The filename convention is not assumed to be exactly equal
    to series_id. Instead, all known RSNA series IDs are extracted
    from the filenames and matched against train_series_descriptions.csv.
    """

    header("DISCOVERING PART 6 PSEUDO-MASKS")

    files = sorted(
        PSEUDOMASK_DIR.glob("*.npz")
    )

    print(
        f"Pseudo-mask files found : {len(files)}"
    )

    if not files:
        raise RuntimeError(
            "No Part 6 pseudo-mask .npz files found."
        )

    # -------------------------------------------------------------------------
    # Authoritative RSNA series IDs
    # -------------------------------------------------------------------------

    known_series_ids = set(
        series_metadata["series_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    print(
        f"Known RSNA series IDs     : "
        f"{len(known_series_ids)}"
    )

    rows = []

    unmatched = []
    ambiguous = []

    # -------------------------------------------------------------------------
    # Match every NPZ filename against known RSNA series IDs
    # -------------------------------------------------------------------------

    for path in files:

        filename = path.name
        stem = path.stem

        # First try exact stem.
        if stem in known_series_ids:

            matched_series_id = stem

        else:

            # Search for a known series_id inside the filename.
            matches = [
                series_id
                for series_id in known_series_ids
                if series_id in stem
            ]

            if len(matches) == 1:

                matched_series_id = matches[0]

            elif len(matches) > 1:

                ambiguous.append(
                    {
                        "filename": filename,
                        "matches": "|".join(
                            sorted(matches)
                        ),
                    }
                )

                continue

            else:

                matched_series_id = None

        if matched_series_id is None:

            unmatched.append(
                {
                    "filename": filename,
                    "stem": stem,
                }
            )

            continue

        rows.append(
            {
                "series_id": str(
                    matched_series_id
                ),
                "pseudo_mask_path": str(
                    path.resolve()
                ),
                "pseudo_mask_filename": filename,
            }
        )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    print(
        f"Successfully matched      : {len(rows)}"
    )

    print(
        f"Unmatched pseudo-masks    : {len(unmatched)}"
    )

    print(
        f"Ambiguous pseudo-masks    : {len(ambiguous)}"
    )

    if unmatched:

        print()
        print(
            "First unmatched pseudo-mask filenames:"
        )

        for item in unmatched[:10]:

            print(
                f"  {item['filename']}"
            )

    if ambiguous:

        print()
        print(
            "First ambiguous pseudo-mask filenames:"
        )

        for item in ambiguous[:10]:

            print(
                f"  {item['filename']} -> "
                f"{item['matches']}"
            )

    if not rows:

        raise RuntimeError(
            "No Part 6 pseudo-mask files could be "
            "mapped to RSNA series IDs."
        )

    result = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------------------
    # Duplicate series-ID protection
    # -------------------------------------------------------------------------

    duplicates = result[
        result["series_id"].duplicated(
            keep=False
        )
    ]

    if not duplicates.empty:

        print()
        print(
            "Duplicate series IDs detected:"
        )

        print(
            duplicates.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Multiple pseudo-mask files were mapped "
            "to the same RSNA series_id."
        )

    # -------------------------------------------------------------------------
    # Save mapping diagnostics
    # -------------------------------------------------------------------------

    mapping_path = (
        MANIFEST_DIR
        / "rsna_part8_pseudomask_mapping.csv"
    )

    result.to_csv(
        mapping_path,
        index=False,
    )

    print(
        f"Saved pseudo-mask mapping: "
        f"{mapping_path}"
    )

    return result


# =============================================================================
# NPZ MASK EXTRACTION
# =============================================================================

def load_pseudomask(
    path: Path,
) -> Tuple[np.ndarray, str]:

    """
    Load a Part 6 .npz pseudo-mask.

    Part 7 supports several possible keys. We reproduce that
    robust loading behavior here.
    """

    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        keys = list(
            data.keys()
        )

        preferred = [
            "mask",
            "pseudo_mask",
            "segmentation",
            "labels",
        ]

        selected_key = None

        for key in preferred:

            if key in keys:

                selected_key = key
                break

        if selected_key is None:

            # Fallback to first array.
            if not keys:

                raise ValueError(
                    f"Empty NPZ file: {path}"
                )

            selected_key = keys[0]

        mask = np.asarray(
            data[selected_key]
        )

    return mask, selected_key


# =============================================================================
# PSEUDO-MASK VALIDATION
# =============================================================================

def inspect_pseudomask(
    path: Path,
):

    try:

        mask, key = load_pseudomask(
            path
        )

        shape = tuple(
            mask.shape
        )

        dtype = mask.dtype

        if mask.ndim != 3:

            return {
                "valid": False,
                "shape": shape,
                "dtype": str(dtype),
                "foreground_voxels": 0,
                "labels": [],
                "selected_key": key,
                "error": (
                    "Mask is not 3-dimensional"
                ),
            }

        if not np.isfinite(
            mask
        ).all():

            return {
                "valid": False,
                "shape": shape,
                "dtype": str(dtype),
                "foreground_voxels": 0,
                "labels": [],
                "selected_key": key,
                "error": (
                    "Mask contains non-finite values"
                ),
            }

        unique = np.unique(
            mask
        )

        integer_like = np.all(
            np.isclose(
                unique,
                np.round(unique),
            )
        )

        if not integer_like:

            return {
                "valid": False,
                "shape": shape,
                "dtype": str(dtype),
                "foreground_voxels": 0,
                "labels": [],
                "selected_key": key,
                "error": (
                    "Mask contains non-integer labels"
                ),
            }

        labels = sorted(
            int(v)
            for v in unique
        )

        valid_labels = set(
            labels
        ).issubset(
            set(range(0, 6))
        )

        foreground = int(
            np.count_nonzero(
                mask
            )
        )

        if not valid_labels:

            return {
                "valid": False,
                "shape": shape,
                "dtype": str(dtype),
                "foreground_voxels": foreground,
                "labels": labels,
                "selected_key": key,
                "error": (
                    "Unexpected class labels"
                ),
            }

        if foreground <= 0:

            return {
                "valid": False,
                "shape": shape,
                "dtype": str(dtype),
                "foreground_voxels": foreground,
                "labels": labels,
                "selected_key": key,
                "error": (
                    "Empty foreground"
                ),
            }

        return {
            "valid": True,
            "shape": shape,
            "dtype": str(dtype),
            "foreground_voxels": foreground,
            "labels": labels,
            "selected_key": key,
            "error": "",
        }

    except Exception as exc:

        return {
            "valid": False,
            "shape": None,
            "dtype": "",
            "foreground_voxels": 0,
            "labels": [],
            "selected_key": "",
            "error": str(exc),
        }


def validate_pseudomasks(
    manifest: pd.DataFrame,
) -> pd.DataFrame:

    header("VALIDATING PART 6 PSEUDO-MASK FILES")

    results = []

    total = len(manifest)

    for n, (_, row) in enumerate(
        manifest.iterrows(),
        start=1,
    ):

        path = Path(
            row["pseudo_mask_path"]
        )

        result = inspect_pseudomask(
            path
        )

        results.append(
            {
                "series_id": row["series_id"],
                "pseudo_mask_valid": result[
                    "valid"
                ],
                "mask_shape": str(
                    result["shape"]
                ),
                "mask_dtype": result[
                    "dtype"
                ],
                "mask_selected_key": result[
                    "selected_key"
                ],
                "foreground_voxels": result[
                    "foreground_voxels"
                ],
                "mask_labels": ",".join(
                    map(
                        str,
                        result["labels"],
                    )
                ),
                "pseudo_mask_error": result[
                    "error"
                ],
            }
        )

        if (
            n == 1
            or n % 500 == 0
            or n == total
        ):

            print(
                f"Validated {n:5d} / {total}"
            )

    validation = pd.DataFrame(
        results
    )

    manifest = manifest.merge(
        validation,
        on="series_id",
        how="left",
    )

    manifest[
        "pseudo_mask_valid"
    ] = manifest[
        "pseudo_mask_valid"
    ].fillna(False)

    return manifest


# =============================================================================
# DICOM DISCOVERY
# =============================================================================

def get_dicom_files(
    series_dir: Path,
):

    if not series_dir.exists():

        return []

    files = list(
        series_dir.glob("*.dcm")
    )

    if not files:

        files = [
            p
            for p in series_dir.iterdir()
            if p.is_file()
        ]

    return sorted(
        files,
        key=lambda p: p.name,
    )


def discover_dicom_series():

    header("DISCOVERING RSNA DICOM SERIES")

    rows = []

    study_dirs = [
        p
        for p in TRAIN_IMAGES_DIR.iterdir()
        if p.is_dir()
    ]

    print(
        f"Study directories : {len(study_dirs)}"
    )

    for study_dir in sorted(
        study_dirs,
        key=lambda p: p.name,
    ):

        study_id = str(
            study_dir.name
        )

        series_dirs = [
            p
            for p in study_dir.iterdir()
            if p.is_dir()
        ]

        for series_dir in sorted(
            series_dirs,
            key=lambda p: p.name,
        ):

            series_id = str(
                series_dir.name
            )

            dicom_files = (
                get_dicom_files(
                    series_dir
                )
            )

            if not dicom_files:

                continue

            rows.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "dicom_dir": str(
                        series_dir.resolve()
                    ),
                    "dicom_count": len(
                        dicom_files
                    ),
                    "first_dicom_path": str(
                        dicom_files[0].resolve()
                    ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    print(
        f"DICOM series discovered : {len(result)}"
    )

    if result.empty:

        raise RuntimeError(
            "No RSNA DICOM series discovered."
        )

    return result


# =============================================================================
# BUILD SERIES MANIFEST
# =============================================================================

def build_series_manifest(
    coordinates,
    series,
    dicom_series,
    pseudomasks,
):

    header(
        "BUILDING RSNA SERIES MANIFEST"
    )

    annotated_ids = set(
        coordinates[
            "series_id"
        ].astype(str)
    )

    metadata = series.copy()

    metadata[
        "series_id"
    ] = metadata[
        "series_id"
    ].astype(str)

    metadata[
        "study_id"
    ] = metadata[
        "study_id"
    ].astype(str)

    metadata = metadata[
        metadata[
            "series_id"
        ].isin(
            annotated_ids
        )
    ].copy()

    metadata = metadata[
        metadata[
            "series_description"
        ].isin(
            EXPECTED_SERIES_TYPES
        )
    ].copy()

    print(
        f"Annotated expected MRI series : "
        f"{len(metadata)}"
    )

    manifest = metadata.merge(
        dicom_series,
        on=[
            "study_id",
            "series_id",
        ],
        how="left",
    )

    manifest = manifest.merge(
        pseudomasks,
        on="series_id",
        how="left",
    )

    annotation_stats = (
        coordinates
        .groupby("series_id")
        .agg(
            annotation_count=(
                "series_id",
                "size",
            ),
            coordinate_study_id=(
                "study_id",
                "first",
            ),
            condition_count=(
                "condition",
                "nunique",
            ),
            level_count=(
                "level",
                "nunique",
            ),
        )
        .reset_index()
    )

    manifest = manifest.merge(
        annotation_stats,
        on="series_id",
        how="left",
    )

    class_presence = (
        coordinates
        .assign(
            class_id=
            coordinates[
                "condition"
            ].map(
                CLASS_MAP
            )
        )
        .groupby(
            "series_id"
        )[
            "class_id"
        ]
        .apply(
            lambda values:
            sorted(
                set(
                    int(v)
                    for v
                    in values.dropna()
                )
            )
        )
        .reset_index(
            name="present_class_ids"
        )
    )

    manifest = manifest.merge(
        class_presence,
        on="series_id",
        how="left",
    )

    manifest[
        "has_dicom"
    ] = (
        manifest[
            "dicom_dir"
        ].notna()
        &
        (
            manifest[
                "dicom_count"
            ].fillna(0)
            > 0
        )
    )

    manifest[
        "has_pseudomask"
    ] = (
        manifest[
            "pseudo_mask_path"
        ].notna()
    )

    manifest[
        "study_id_match"
    ] = (
        manifest[
            "study_id"
        ].astype(str)
        ==
        manifest[
            "coordinate_study_id"
        ]
        .fillna("")
        .astype(str)
    )

    manifest[
        "eligible_before_mask_validation"
    ] = (
        manifest[
            "has_dicom"
        ]
        &
        manifest[
            "has_pseudomask"
        ]
        &
        manifest[
            "study_id_match"
        ]
    )

    print(
        f"Series with DICOM       : "
        f"{int(manifest['has_dicom'].sum())}"
    )

    print(
        f"Series with pseudo-mask : "
        f"{int(manifest['has_pseudomask'].sum())}"
    )

    print(
        f"Study IDs matched       : "
        f"{int(manifest['study_id_match'].sum())}"
    )

    print(
        f"Eligible before mask validation : "
        f"{int(manifest['eligible_before_mask_validation'].sum())}"
    )

    return manifest


# =============================================================================
# FINAL ELIGIBILITY
# =============================================================================

def finalize_eligibility(
    manifest,
):

    header(
        "FINAL SERIES ELIGIBILITY"
    )

    manifest[
        "dataset_eligible"
    ] = (
        manifest[
            "eligible_before_mask_validation"
        ]
        &
        manifest[
            "pseudo_mask_valid"
        ].fillna(False)
    )

    total = len(manifest)

    eligible = int(
        manifest[
            "dataset_eligible"
        ].sum()
    )

    print(
        f"Total manifest series : {total}"
    )

    print(
        f"Eligible series       : {eligible}"
    )

    print(
        f"Excluded series       : "
        f"{total - eligible}"
    )

    return manifest


# =============================================================================
# STUDY-LEVEL SPLIT
# =============================================================================

def create_study_split(
    manifest,
):

    header(
        "CREATING STUDY-LEVEL DATA SPLIT"
    )

    eligible = manifest[
        manifest[
            "dataset_eligible"
        ]
    ].copy()

    studies = sorted(
        eligible[
            "study_id"
        ].astype(str)
        .unique()
    )

    if not studies:

        raise RuntimeError(
            "No eligible studies available."
        )

    rng = random.Random(
        RANDOM_SEED
    )

    rng.shuffle(
        studies
    )

    n = len(studies)

    n_train = int(
        math.floor(
            n * TRAIN_RATIO
        )
    )

    n_val = int(
        math.floor(
            n * VAL_RATIO
        )
    )

    train_studies = set(
        studies[
            :n_train
        ]
    )

    validation_studies = set(
        studies[
            n_train:
            n_train + n_val
        ]
    )

    test_studies = set(
        studies[
            n_train + n_val:
        ]
    )

    def assign_split(
        study_id
    ):

        study_id = str(
            study_id
        )

        if study_id in train_studies:
            return "train"

        if study_id in validation_studies:
            return "validation"

        if study_id in test_studies:
            return "test"

        return "unassigned"

    manifest[
        "split"
    ] = (
        manifest[
            "study_id"
        ].astype(str)
        .map(
            assign_split
        )
    )

    print(
        f"Total eligible studies : {n}"
    )

    print(
        f"Train studies          : "
        f"{len(train_studies)}"
    )

    print(
        f"Validation studies     : "
        f"{len(validation_studies)}"
    )

    print(
        f"Test studies           : "
        f"{len(test_studies)}"
    )

    eligible = manifest[
        manifest[
            "dataset_eligible"
        ]
    ]

    print()
    print(
        "Series distribution:"
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        count = int(
            (
                eligible[
                    "split"
                ]
                == split
            ).sum()
        )

        print(
            f"  {split:<12}: {count}"
        )

    return manifest


# =============================================================================
# LEAKAGE AUDIT
# =============================================================================

def perform_leakage_audit(
    manifest,
):

    header(
        "STUDY-LEVEL LEAKAGE AUDIT"
    )

    eligible = manifest[
        manifest[
            "dataset_eligible"
        ]
    ].copy()

    split_studies = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        split_studies[
            split
        ] = set(
            eligible.loc[
                eligible[
                    "split"
                ] == split,
                "study_id",
            ]
            .astype(str)
        )

    train_val = (
        split_studies[
            "train"
        ]
        &
        split_studies[
            "validation"
        ]
    )

    train_test = (
        split_studies[
            "train"
        ]
        &
        split_studies[
            "test"
        ]
    )

    val_test = (
        split_studies[
            "validation"
        ]
        &
        split_studies[
            "test"
        ]
    )

    result = {
        "train_validation_overlap": len(
            train_val
        ),
        "train_test_overlap": len(
            train_test
        ),
        "validation_test_overlap": len(
            val_test
        ),
    }

    print(
        f"Train/Validation overlap : "
        f"{result['train_validation_overlap']}"
    )

    print(
        f"Train/Test overlap       : "
        f"{result['train_test_overlap']}"
    )

    print(
        f"Validation/Test overlap  : "
        f"{result['validation_test_overlap']}"
    )

    if any(
        value > 0
        for value in result.values()
    ):

        raise RuntimeError(
            "Study-level leakage detected."
        )

    print(
        "✓ No study-level leakage detected."
    )

    return result


# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

def create_class_distribution(
    coordinates,
    manifest,
):

    header(
        "CLASS DISTRIBUTION"
    )

    eligible_series = set(
        manifest.loc[
            manifest[
                "dataset_eligible"
            ],
            "series_id",
        ]
        .astype(str)
    )

    coords = coordinates.copy()

    coords[
        "series_id"
    ] = coords[
        "series_id"
    ].astype(str)

    coords = coords[
        coords[
            "series_id"
        ].isin(
            eligible_series
        )
    ].copy()

    coords[
        "class_id"
    ] = coords[
        "condition"
    ].map(
        CLASS_MAP
    )

    split_lookup = (
        manifest
        .set_index(
            "series_id"
        )[
            "split"
        ]
        .to_dict()
    )

    coords[
        "split"
    ] = coords[
        "series_id"
    ].map(
        split_lookup
    )

    rows = []

    for class_id, class_name in (
        CLASS_NAMES.items()
    ):

        if class_id == 0:
            continue

        subset = coords[
            coords[
                "class_id"
            ] == class_id
        ]

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "annotation_count": len(
                    subset
                ),
                "series_count": subset[
                    "series_id"
                ].nunique(),
                "study_count": subset[
                    "study_id"
                ].nunique(),
                "train_annotations": int(
                    (
                        subset[
                            "split"
                        ]
                        == "train"
                    ).sum()
                ),
                "validation_annotations": int(
                    (
                        subset[
                            "split"
                        ]
                        == "validation"
                    ).sum()
                ),
                "test_annotations": int(
                    (
                        subset[
                            "split"
                        ]
                        == "test"
                    ).sum()
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =============================================================================
# SERIES DISTRIBUTION
# =============================================================================

def create_series_distribution(
    manifest,
):

    header(
        "SERIES TYPE DISTRIBUTION"
    )

    eligible = manifest[
        manifest[
            "dataset_eligible"
        ]
    ].copy()

    result = (
        eligible
        .groupby(
            [
                "split",
                "series_description",
            ]
        )
        .size()
        .reset_index(
            name="series_count"
        )
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =============================================================================
# FOREGROUND STATISTICS
# =============================================================================

def create_foreground_statistics(
    manifest,
):

    header(
        "PSEUDO-MASK FOREGROUND STATISTICS"
    )

    eligible = manifest[
        manifest[
            "dataset_eligible"
        ]
    ].copy()

    rows = []

    for split in [
        "train",
        "validation",
        "test",
    ]:

        values = (
            eligible.loc[
                eligible[
                    "split"
                ] == split,
                "foreground_voxels",
            ]
            .dropna()
            .astype(float)
        )

        if values.empty:
            continue

        rows.append(
            {
                "split": split,
                "series_count": len(
                    values
                ),
                "mean_foreground_voxels": float(
                    values.mean()
                ),
                "median_foreground_voxels": float(
                    values.median()
                ),
                "minimum_foreground_voxels": int(
                    values.min()
                ),
                "maximum_foreground_voxels": int(
                    values.max()
                ),
                "std_foreground_voxels": float(
                    values.std(
                        ddof=0
                    )
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =============================================================================
# DICOM GEOMETRY AUDIT
# =============================================================================

def inspect_dicom_geometry(
    manifest,
):

    header(
        "DICOM GEOMETRY AUDIT"
    )

    if pydicom is None:

        print(
            "pydicom is not installed."
        )

        print(
            "DICOM geometry audit skipped."
        )

        return pd.DataFrame()

    eligible = manifest[
        manifest[
            "dataset_eligible"
        ]
    ].copy()

    rows = []

    total = len(
        eligible
    )

    for n, (_, row) in enumerate(
        eligible.iterrows(),
        start=1,
    ):

        path = Path(
            row[
                "first_dicom_path"
            ]
        )

        result = {
            "study_id": row[
                "study_id"
            ],
            "series_id": row[
                "series_id"
            ],
            "series_description": row[
                "series_description"
            ],
            "dicom_count": row[
                "dicom_count"
            ],
            "dicom_readable": False,
            "rows": None,
            "columns": None,
            "pixel_spacing": "",
            "slice_thickness": None,
            "has_image_position": False,
            "has_image_orientation": False,
            "has_instance_number": False,
        }

        try:

            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
            )

            result[
                "dicom_readable"
            ] = True

            result[
                "rows"
            ] = safe_int(
                getattr(
                    ds,
                    "Rows",
                    None,
                )
            )

            result[
                "columns"
            ] = safe_int(
                getattr(
                    ds,
                    "Columns",
                    None,
                )
            )

            spacing = getattr(
                ds,
                "PixelSpacing",
                None,
            )

            if spacing is not None:

                result[
                    "pixel_spacing"
                ] = (
                    f"{spacing[0]},{spacing[1]}"
                )

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

            result[
                "has_instance_number"
            ] = hasattr(
                ds,
                "InstanceNumber",
            )

        except Exception:

            pass

        rows.append(
            result
        )

        if (
            n == 1
            or n % 500 == 0
            or n == total
        ):

            print(
                f"Audited {n:5d} / {total}"
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# REPRESENTATIVE CASES
# =============================================================================

def select_representative_cases(
    manifest,
):

    header(
        "SELECTING REPRESENTATIVE CASES"
    )

    eligible = manifest[
        manifest[
            "dataset_eligible"
        ]
    ].copy()

    selected = []

    # Highest foreground.
    high = (
        eligible
        .sort_values(
            "foreground_voxels",
            ascending=False,
        )
        .head(3)
    )

    for _, row in high.iterrows():

        selected.append(
            {
                "series_id": row[
                    "series_id"
                ],
                "study_id": row[
                    "study_id"
                ],
                "series_description": row[
                    "series_description"
                ],
                "split": row[
                    "split"
                ],
                "foreground_voxels": row[
                    "foreground_voxels"
                ],
                "reason": "high_foreground",
            }
        )

    # Lowest foreground.
    low = (
        eligible[
            eligible[
                "foreground_voxels"
            ] > 0
        ]
        .sort_values(
            "foreground_voxels",
            ascending=True,
        )
        .head(3)
    )

    for _, row in low.iterrows():

        selected.append(
            {
                "series_id": row[
                    "series_id"
                ],
                "study_id": row[
                    "study_id"
                ],
                "series_description": row[
                    "series_description"
                ],
                "split": row[
                    "split"
                ],
                "foreground_voxels": row[
                    "foreground_voxels"
                ],
                "reason": "low_foreground",
            }
        )

    # One representative from each sequence type.
    for sequence in sorted(
        EXPECTED_SERIES_TYPES
    ):

        subset = eligible[
            eligible[
                "series_description"
            ] == sequence
        ]

        if subset.empty:
            continue

        row = subset.iloc[0]

        selected.append(
            {
                "series_id": row[
                    "series_id"
                ],
                "study_id": row[
                    "study_id"
                ],
                "series_description": row[
                    "series_description"
                ],
                "split": row[
                    "split"
                ],
                "foreground_voxels": row[
                    "foreground_voxels"
                ],
                "reason": (
                    "series_type_representative"
                ),
            }
        )

    result = pd.DataFrame(
        selected
    ).drop_duplicates(
        subset=[
            "series_id"
        ]
    )

    print(
        f"Representative cases : {len(result)}"
    )

    if not result.empty:

        print(
            result.to_string(
                index=False
            )
        )

    return result


# =============================================================================
# SAVE TABLES
# =============================================================================

def save_tables(
    manifest,
    class_df,
    series_df,
    foreground_df,
    dicom_df,
    representative_df,
):

    header(
        "SAVING DATASET MANIFESTS"
    )

    paths = {}

    files = {
        "series_manifest":
            "rsna_part8_series_manifest.csv",
        "train_manifest":
            "rsna_part8_train_manifest.csv",
        "validation_manifest":
            "rsna_part8_validation_manifest.csv",
        "test_manifest":
            "rsna_part8_test_manifest.csv",
    }

    for key, filename in files.items():

        if key == "series_manifest":

            data = manifest

        elif key == "train_manifest":

            data = manifest[
                manifest[
                    "split"
                ] == "train"
            ]

        elif key == "validation_manifest":

            data = manifest[
                manifest[
                    "split"
                ] == "validation"
            ]

        else:

            data = manifest[
                manifest[
                    "split"
                ] == "test"
            ]

        path = (
            MANIFEST_DIR
            / filename
        )

        data.to_csv(
            path,
            index=False,
        )

        paths[key] = str(
            path
        )

        print(
            f"Saved: {path}"
        )

    other = {
        "class_distribution":
            (
                "rsna_part8_class_distribution.csv",
                class_df,
            ),
        "series_distribution":
            (
                "rsna_part8_series_distribution.csv",
                series_df,
            ),
        "foreground_statistics":
            (
                "rsna_part8_foreground_statistics.csv",
                foreground_df,
            ),
        "representative_cases":
            (
                "rsna_part8_representative_cases.csv",
                representative_df,
            ),
    }

    for key, (
        filename,
        data,
    ) in other.items():

        path = (
            MANIFEST_DIR
            / filename
        )

        data.to_csv(
            path,
            index=False,
        )

        paths[key] = str(
            path
        )

        print(
            f"Saved: {path}"
        )

    if not dicom_df.empty:

        path = (
            MANIFEST_DIR
            / "rsna_part8_dicom_geometry_audit.csv"
        )

        dicom_df.to_csv(
            path,
            index=False,
        )

        paths[
            "dicom_geometry"
        ] = str(path)

        print(
            f"Saved: {path}"
        )

    return paths


# =============================================================================
# SUMMARY
# =============================================================================

def create_summary(
    manifest,
    leakage,
    class_df,
    series_df,
    foreground_df,
    dicom_df,
):

    eligible = manifest[
        manifest[
            "dataset_eligible"
        ]
    ].copy()

    split_series = {}

    split_studies = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        split_series[
            split
        ] = int(
            (
                eligible[
                    "split"
                ] == split
            ).sum()
        )

        split_studies[
            split
        ] = int(
            eligible.loc[
                eligible[
                    "split"
                ] == split,
                "study_id",
            ].nunique()
        )

    summary = {
        "phase": 4,
        "part": 8,
        "title":
            "RSNA 3D Segmentation Dataset Construction",

        "dataset":
            "RSNA 2024 Lumbar Spine Degenerative Classification",

        "spider_used": False,
        "training_performed": False,
        "model_weights_modified": False,

        "manual_segmentation_masks": False,
        "point_derived_pseudomasks": True,

        "pseudomask_format": ".npz",

        "random_seed":
            RANDOM_SEED,

        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },

        "total_manifest_series":
            len(manifest),

        "eligible_series":
            len(eligible),

        "excluded_series":
            len(manifest) - len(eligible),

        "eligible_studies":
            int(
                eligible[
                    "study_id"
                ].nunique()
            ),

        "split_series":
            split_series,

        "split_studies":
            split_studies,

        "leakage_audit":
            leakage,

        "dicom_geometry_rows":
            len(dicom_df),

        "class_distribution_rows":
            len(class_df),

        "series_distribution_rows":
            len(series_df),

        "foreground_statistics_rows":
            len(foreground_df),

        "decision":
            (
                "PASS - RSNA-only segmentation "
                "dataset constructed and study-level "
                "split validated"
                if (
                    len(eligible) > 0
                    and leakage[
                        "train_validation_overlap"
                    ] == 0
                    and leakage[
                        "train_test_overlap"
                    ] == 0
                    and leakage[
                        "validation_test_overlap"
                    ] == 0
                )
                else
                "FAIL - dataset construction requires correction"
            ),
    }

    return summary


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    summary,
    manifest,
    representative_df,
):

    path = (
        REPORT_DIR
        / "phase4_part8_dataset_construction_report.txt"
    )

    eligible = manifest[
        manifest[
            "dataset_eligible"
        ]
    ]

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 4 - PART 8\n"
        )

        f.write(
            "RSNA 3D SEGMENTATION DATASET "
            "CONSTRUCTION\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            "DATASET\n"
        )

        f.write(
            "RSNA 2024 Lumbar Spine "
            "Degenerative Classification\n\n"
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
            "MANUAL SEGMENTATION MASKS: NO\n"
        )

        f.write(
            "POINT-DERIVED PSEUDOMASKS: YES\n"
        )

        f.write(
            "PSEUDOMASK FORMAT: .npz\n\n"
        )

        f.write(
            "DATASET\n"
        )

        f.write(
            f"Total manifest series : "
            f"{summary['total_manifest_series']}\n"
        )

        f.write(
            f"Eligible series       : "
            f"{summary['eligible_series']}\n"
        )

        f.write(
            f"Excluded series       : "
            f"{summary['excluded_series']}\n"
        )

        f.write(
            f"Eligible studies      : "
            f"{summary['eligible_studies']}\n\n"
        )

        f.write(
            "SPLIT\n"
        )

        for split in [
            "train",
            "validation",
            "test",
        ]:

            f.write(
                f"{split} studies : "
                f"{summary['split_studies'][split]}\n"
            )

            f.write(
                f"{split} series  : "
                f"{summary['split_series'][split]}\n"
            )

        f.write("\n")

        f.write(
            "LEAKAGE AUDIT\n"
        )

        for key, value in (
            summary[
                "leakage_audit"
            ].items()
        ):

            f.write(
                f"{key}: {value}\n"
            )

        f.write("\n")

        f.write(
            "REPRESENTATIVE CASES\n"
        )

        if representative_df.empty:

            f.write(
                "None selected.\n"
            )

        else:

            f.write(
                representative_df.to_string(
                    index=False
                )
            )

            f.write("\n")

        f.write("\n")

        f.write(
            "FINAL DECISION\n"
        )

        f.write(
            summary["decision"]
            + "\n"
        )

        f.write("\n")

        f.write(
            "SCIENTIFIC NOTE\n"
        )

        f.write(
            "The segmentation targets are derived "
            "from RSNA point annotations. They are "
            "pseudo-labels and must not be described "
            "as manually segmented ground truth. "
            "The subsequent training and evaluation "
            "stages must preserve this distinction.\n"
        )

    return path


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "PHASE 4 - PART 8"
    )

    print(
        "RSNA 3D SEGMENTATION DATASET CONSTRUCTION"
    )

    header(
        "PROJECT PATHS"
    )

    print(
        f"PROJECT ROOT\n{PROJECT_ROOT}"
    )

    print(
        f"\nRSNA DATASET\n{RSNA_ROOT}"
    )

    print(
        f"\nPART 6 PSEUDO-MASKS\n"
        f"{PSEUDOMASK_DIR}"
    )

    print(
        f"\nPART 7 QUALITY VALIDATION\n"
        f"{PART7_DIR}"
    )

    print(
        f"\nOUTPUT DIRECTORY\n"
        f"{OUTPUT_DIR}"
    )

    prepare_output_dirs()

    validate_paths()

    coordinates, series, train = (
        load_metadata()
    )

    (
        quality,
        class_quality,
        flags,
    ) = load_part7_quality()

    pseudomasks = (
        discover_pseudomasks(series)
    )

    dicom_series = (
        discover_dicom_series()
    )

    manifest = build_series_manifest(
        coordinates,
        series,
        dicom_series,
        pseudomasks,
    )

    manifest = validate_pseudomasks(
        manifest
    )

    manifest = finalize_eligibility(
        manifest
    )

    manifest = create_study_split(
        manifest
    )

    leakage = perform_leakage_audit(
        manifest
    )

    class_df = (
        create_class_distribution(
            coordinates,
            manifest,
        )
    )

    series_df = (
        create_series_distribution(
            manifest
        )
    )

    foreground_df = (
        create_foreground_statistics(
            manifest
        )
    )

    dicom_df = (
        inspect_dicom_geometry(
            manifest
        )
    )

    representative_df = (
        select_representative_cases(
            manifest
        )
    )

    save_tables(
        manifest,
        class_df,
        series_df,
        foreground_df,
        dicom_df,
        representative_df,
    )

    summary = create_summary(
        manifest,
        leakage,
        class_df,
        series_df,
        foreground_df,
        dicom_df,
    )

    summary_json = (
        OUTPUT_DIR
        / "phase4_part8_dataset_construction_summary.json"
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

    report_path = write_report(
        summary,
        manifest,
        representative_df,
    )

    header(
        "PART 8 FINAL SUMMARY"
    )

    print(
        f"Total manifest series : "
        f"{summary['total_manifest_series']}"
    )

    print(
        f"Eligible series       : "
        f"{summary['eligible_series']}"
    )

    print(
        f"Excluded series       : "
        f"{summary['excluded_series']}"
    )

    print(
        f"Eligible studies      : "
        f"{summary['eligible_studies']}"
    )

    print()
    print(
        "STUDY SPLIT"
    )

    print(
        f"Train studies        : "
        f"{summary['split_studies']['train']}"
    )

    print(
        f"Validation studies   : "
        f"{summary['split_studies']['validation']}"
    )

    print(
        f"Test studies         : "
        f"{summary['split_studies']['test']}"
    )

    print()
    print(
        "SERIES SPLIT"
    )

    print(
        f"Train series         : "
        f"{summary['split_series']['train']}"
    )

    print(
        f"Validation series    : "
        f"{summary['split_series']['validation']}"
    )

    print(
        f"Test series          : "
        f"{summary['split_series']['test']}"
    )

    print()
    print(
        "LEAKAGE AUDIT"
    )

    print(
        f"Train/Validation : "
        f"{leakage['train_validation_overlap']}"
    )

    print(
        f"Train/Test       : "
        f"{leakage['train_test_overlap']}"
    )

    print(
        f"Validation/Test  : "
        f"{leakage['validation_test_overlap']}"
    )

    print()
    print(
        "PSEUDO-MASK FORMAT : .npz"
    )

    print(
        "SPIDER USED        : NO"
    )

    print(
        "TRAINING PERFORMED : NO"
    )

    print(
        "MODEL WEIGHTS MODIFIED : NO"
    )

    print()
    print(
        "FINAL DECISION"
    )

    print(
        summary["decision"]
    )

    header(
        "SAVING FINAL SUMMARY"
    )

    print(
        f"Saved: {summary_json}"
    )

    print(
        f"Saved: {report_path}"
    )

    header(
        "PART 8 COMPLETE"
    )

    print(
        "RSNA-only dataset construction completed."
    )

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Process interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        header(
            "ERROR"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        sys.exit(1)