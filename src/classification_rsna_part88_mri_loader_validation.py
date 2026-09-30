from pathlib import Path
import json
import random
import sys

import numpy as np
import pandas as pd
import pydicom
import torch


# ============================================================
# PART 88
# RSNA CLASSIFICATION MRI LOADER & PREPROCESSING VALIDATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RSNA_ROOT = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES = RSNA_ROOT / "train_images"

TRAIN_MANIFEST = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part87_study_level_split"
    / "part87_train_manifest.csv"
)

VAL_MANIFEST = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part87_study_level_split"
    / "part87_validation_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part88_mri_loader_validation"
)

REPORT_DIR = ROOT / "reports"

SAMPLE_RESULTS_CSV = (
    OUTPUT_DIR
    / "part88_sample_loading_results.csv"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "part88_mri_loader_validation_summary.json"
)

REPORT_TXT = (
    REPORT_DIR
    / "part88_mri_loader_validation_report.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

# Small validation sample.
# We deliberately do not scan all 1,975 studies yet.
TRAIN_SAMPLE_SIZE = 20
VAL_SAMPLE_SIZE = 20

# Target spatial size for the classification preprocessing
# validation. This is a validation target, not a final model
# architecture decision.
TARGET_DEPTH = 32
TARGET_HEIGHT = 224
TARGET_WIDTH = 224

EXPECTED_SERIES = [
    "Sagittal T2/STIR",
    "Sagittal T1",
    "Axial T2",
]

TARGET_COLUMNS = [
    "spinal_canal_stenosis_l1_l2",
    "spinal_canal_stenosis_l2_l3",
    "spinal_canal_stenosis_l3_l4",
    "spinal_canal_stenosis_l4_l5",
    "spinal_canal_stenosis_l5_s1",

    "left_neural_foraminal_narrowing_l1_l2",
    "left_neural_foraminal_narrowing_l2_l3",
    "left_neural_foraminal_narrowing_l3_l4",
    "left_neural_foraminal_narrowing_l4_l5",
    "left_neural_foraminal_narrowing_l5_s1",

    "right_neural_foraminal_narrowing_l1_l2",
    "right_neural_foraminal_narrowing_l2_l3",
    "right_neural_foraminal_narrowing_l3_l4",
    "right_neural_foraminal_narrowing_l4_l5",
    "right_neural_foraminal_narrowing_l5_s1",

    "left_subarticular_stenosis_l1_l2",
    "left_subarticular_stenosis_l2_l3",
    "left_subarticular_stenosis_l3_l4",
    "left_subarticular_stenosis_l4_l5",
    "left_subarticular_stenosis_l5_s1",

    "right_subarticular_stenosis_l1_l2",
    "right_subarticular_stenosis_l2_l3",
    "right_subarticular_stenosis_l3_l4",
    "right_subarticular_stenosis_l4_l5",
    "right_subarticular_stenosis_l5_s1",
]


# ============================================================
# UTILITIES
# ============================================================

def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return None


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_paths():

    banner("PART 88 PATH VALIDATION")

    paths = {
        "RSNA root": RSNA_ROOT,
        "train_images": TRAIN_IMAGES,
        "train manifest": TRAIN_MANIFEST,
        "validation manifest": VAL_MANIFEST,
    }

    missing = []

    for name, path in paths.items():

        exists = path.exists()

        print(
            f"{name:<35}: "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:
            missing.append(str(path))

    if missing:

        raise FileNotFoundError(
            "Required Part 87 input(s) missing:\n"
            + "\n".join(missing)
        )


# ============================================================
# LOAD MANIFESTS
# ============================================================

def load_manifests():

    banner("LOADING PART 87 MANIFESTS")

    train = pd.read_csv(
        TRAIN_MANIFEST
    )

    val = pd.read_csv(
        VAL_MANIFEST
    )

    print(
        f"Training manifest rows   : {len(train)}"
    )

    print(
        f"Validation manifest rows : {len(val)}"
    )

    required = {
        "study_id",
        "series_id",
        "series_description",
        "series_path",
        "selected",
        "series_exists",
        "dicom_file_count",
    }

    required.update(
        TARGET_COLUMNS
    )

    missing_train = (
        required - set(train.columns)
    )

    missing_val = (
        required - set(val.columns)
    )

    if missing_train:

        raise RuntimeError(
            "Training manifest missing columns:\n"
            + "\n".join(sorted(missing_train))
        )

    if missing_val:

        raise RuntimeError(
            "Validation manifest missing columns:\n"
            + "\n".join(sorted(missing_val))
        )

    return train, val


# ============================================================
# DICOM FILE DISCOVERY
# ============================================================

def discover_dicom_files(series_path):

    path = Path(
        str(series_path)
    )

    if not path.exists():

        return []

    files = []

    try:

        for p in path.iterdir():

            if p.is_file():

                files.append(p)

    except Exception:

        return []

    # The RSNA data normally contains DICOM files without
    # requiring a particular filename extension.
    files.sort(
        key=lambda x: x.name
    )

    return files


# ============================================================
# DICOM LOADING
# ============================================================

def load_dicom_series(series_path):

    files = discover_dicom_files(
        series_path
    )

    if not files:

        raise FileNotFoundError(
            f"No files found in {series_path}"
        )

    slices = []

    failed_files = []

    for path in files:

        try:

            ds = pydicom.dcmread(
                str(path),
                force=True
            )

            if not hasattr(
                ds,
                "PixelData"
            ):

                continue

            array = ds.pixel_array

            if array.ndim != 2:

                continue

            instance_number = getattr(
                ds,
                "InstanceNumber",
                None
            )

            image_position = getattr(
                ds,
                "ImagePositionPatient",
                None
            )

            z_position = None

            if (
                image_position is not None
                and len(image_position) >= 3
            ):

                z_position = safe_float(
                    image_position[2]
                )

            slices.append(
                {
                    "path": path,
                    "array": np.asarray(
                        array,
                        dtype=np.float32
                    ),
                    "instance_number":
                        instance_number,
                    "z_position":
                        z_position,
                }
            )

        except Exception:

            failed_files.append(
                str(path)
            )

    if not slices:

        raise RuntimeError(
            f"No readable DICOM pixel arrays found in "
            f"{series_path}"
        )

    # Prefer physical z-position when available.
    if all(
        s["z_position"] is not None
        for s in slices
    ):

        slices.sort(
            key=lambda s: s["z_position"]
        )

        ordering_method = (
            "ImagePositionPatient_z"
        )

    elif all(
        s["instance_number"] is not None
        for s in slices
    ):

        slices.sort(
            key=lambda s: int(
                s["instance_number"]
            )
        )

        ordering_method = (
            "InstanceNumber"
        )

    else:

        slices.sort(
            key=lambda s: s["path"].name
        )

        ordering_method = (
            "filename"
        )

    shapes = [
        s["array"].shape
        for s in slices
    ]

    unique_shapes = sorted(
        set(shapes)
    )

    # DICOM series should normally have consistent
    # in-plane dimensions. If they do not, pad them to
    # the largest H/W rather than silently dropping slices.
    max_height = max(
        shape[0]
        for shape in shapes
    )

    max_width = max(
        shape[1]
        for shape in shapes
    )

    volume = np.zeros(
        (
            len(slices),
            max_height,
            max_width,
        ),
        dtype=np.float32
    )

    for i, item in enumerate(slices):

        image = item["array"]

        h, w = image.shape

        volume[
            i,
            :h,
            :w
        ] = image

    return (
        volume,
        ordering_method,
        len(files),
        len(slices),
        len(failed_files),
        unique_shapes,
        failed_files,
    )


# ============================================================
# ROBUST INTENSITY NORMALIZATION
# ============================================================

def normalize_volume(volume):

    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    finite = np.isfinite(
        volume
    )

    if not finite.any():

        raise ValueError(
            "Volume contains no finite values."
        )

    valid = volume[finite]

    low = np.percentile(
        valid,
        1.0
    )

    high = np.percentile(
        valid,
        99.0
    )

    if high <= low:

        low = float(
            valid.min()
        )

        high = float(
            valid.max()
        )

    clipped = np.clip(
        volume,
        low,
        high
    )

    if high > low:

        normalized = (
            clipped - low
        ) / (
            high - low
        )

    else:

        normalized = np.zeros_like(
            clipped,
            dtype=np.float32
        )

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# CENTER CROP / PAD
# ============================================================

def center_crop_or_pad_3d(
    volume,
    target_depth,
    target_height,
    target_width
):

    source = np.asarray(
        volume,
        dtype=np.float32
    )

    src_d, src_h, src_w = (
        source.shape
    )

    output = np.zeros(
        (
            target_depth,
            target_height,
            target_width,
        ),
        dtype=np.float32
    )

    # Source crop coordinates.
    d_start = max(
        0,
        (src_d - target_depth) // 2
    )

    h_start = max(
        0,
        (src_h - target_height) // 2
    )

    w_start = max(
        0,
        (src_w - target_width) // 2
    )

    d_end = min(
        src_d,
        d_start + target_depth
    )

    h_end = min(
        src_h,
        h_start + target_height
    )

    w_end = min(
        src_w,
        w_start + target_width
    )

    cropped = source[
        d_start:d_end,
        h_start:h_end,
        w_start:w_end
    ]

    # Destination coordinates.
    out_d_start = max(
        0,
        (target_depth - cropped.shape[0]) // 2
    )

    out_h_start = max(
        0,
        (target_height - cropped.shape[1]) // 2
    )

    out_w_start = max(
        0,
        (target_width - cropped.shape[2]) // 2
    )

    out_d_end = (
        out_d_start
        + cropped.shape[0]
    )

    out_h_end = (
        out_h_start
        + cropped.shape[1]
    )

    out_w_end = (
        out_w_start
        + cropped.shape[2]
    )

    output[
        out_d_start:out_d_end,
        out_h_start:out_h_end,
        out_w_start:out_w_end
    ] = cropped

    return output


# ============================================================
# TENSOR PREPROCESSING
# ============================================================

def preprocess_volume(
    volume
):

    normalized = normalize_volume(
        volume
    )

    resized = center_crop_or_pad_3d(
        normalized,
        TARGET_DEPTH,
        TARGET_HEIGHT,
        TARGET_WIDTH
    )

    # Classification tensor convention:
    #
    # [C, D, H, W]
    #
    # C = 1 grayscale MRI channel.
    tensor = torch.from_numpy(
        resized
    ).unsqueeze(0)

    return tensor.float()


# ============================================================
# LABEL VALIDATION
# ============================================================

def validate_labels(row):

    present = 0
    missing = 0

    severity_counts = {
        "Normal/Mild": 0,
        "Moderate": 0,
        "Severe": 0,
    }

    unknown = 0

    for column in TARGET_COLUMNS:

        value = row[column]

        if pd.isna(value):

            missing += 1
            continue

        present += 1

        value_string = str(
            value
        ).strip()

        if value_string in severity_counts:

            severity_counts[
                value_string
            ] += 1

        else:

            unknown += 1

    return {
        "label_count":
            present,

        "missing_label_count":
            missing,

        "normal_mild_count":
            severity_counts["Normal/Mild"],

        "moderate_count":
            severity_counts["Moderate"],

        "severe_count":
            severity_counts["Severe"],

        "unknown_label_count":
            unknown,
    }


# ============================================================
# SINGLE STUDY VALIDATION
# ============================================================

def validate_one_study(
    row,
    split
):

    study_id = str(
        row["study_id"]
    )

    series_id = str(
        row["series_id"]
    )

    series_description = str(
        row["series_description"]
    )

    series_path = Path(
        str(row["series_path"])
    )

    result = {
        "split": split,
        "study_id": study_id,
        "series_id": series_id,
        "series_description":
            series_description,

        "manifest_selected":
            bool(row["selected"]),

        "manifest_series_exists":
            bool(row["series_exists"]),

        "manifest_dicom_file_count":
            int(
                row["dicom_file_count"]
            ),

        "filesystem_series_exists":
            series_path.exists(),

        "filesystem_file_count":
            0,

        "readable_dicom_count":
            0,

        "failed_dicom_count":
            0,

        "ordering_method":
            "",

        "original_depth":
            0,

        "original_height":
            0,

        "original_width":
            0,

        "unique_original_shapes":
            "",

        "normalized_min":
            None,

        "normalized_max":
            None,

        "normalized_mean":
            None,

        "normalized_std":
            None,

        "tensor_shape":
            "",

        "tensor_min":
            None,

        "tensor_max":
            None,

        "tensor_mean":
            None,

        "tensor_std":
            None,

        "label_count":
            0,

        "missing_label_count":
            0,

        "normal_mild_count":
            0,

        "moderate_count":
            0,

        "severe_count":
            0,

        "unknown_label_count":
            0,

        "status":
            "FAILED",

        "error":
            "",
    }

    try:

        if not series_path.exists():

            raise FileNotFoundError(
                f"Series directory missing: "
                f"{series_path}"
            )

        (
            volume,
            ordering_method,
            filesystem_file_count,
            readable_count,
            failed_count,
            unique_shapes,
            failed_files,
        ) = load_dicom_series(
            series_path
        )

        result[
            "filesystem_file_count"
        ] = filesystem_file_count

        result[
            "readable_dicom_count"
        ] = readable_count

        result[
            "failed_dicom_count"
        ] = failed_count

        result[
            "ordering_method"
        ] = ordering_method

        result[
            "original_depth"
        ] = int(volume.shape[0])

        result[
            "original_height"
        ] = int(volume.shape[1])

        result[
            "original_width"
        ] = int(volume.shape[2])

        result[
            "unique_original_shapes"
        ] = ";".join(
            str(x)
            for x in unique_shapes
        )

        normalized = normalize_volume(
            volume
        )

        result[
            "normalized_min"
        ] = float(
            normalized.min()
        )

        result[
            "normalized_max"
        ] = float(
            normalized.max()
        )

        result[
            "normalized_mean"
        ] = float(
            normalized.mean()
        )

        result[
            "normalized_std"
        ] = float(
            normalized.std()
        )

        tensor = preprocess_volume(
            volume
        )

        result[
            "tensor_shape"
        ] = str(
            tuple(
                tensor.shape
            )
        )

        result[
            "tensor_min"
        ] = float(
            tensor.min()
        )

        result[
            "tensor_max"
        ] = float(
            tensor.max()
        )

        result[
            "tensor_mean"
        ] = float(
            tensor.mean()
        )

        result[
            "tensor_std"
        ] = float(
            tensor.std()
        )

        labels = validate_labels(
            row
        )

        result.update(
            labels
        )

        if (
            result["readable_dicom_count"]
            <= 0
        ):

            raise RuntimeError(
                "No readable DICOM slices."
            )

        if (
            result["tensor_shape"]
            != str(
                (
                    1,
                    TARGET_DEPTH,
                    TARGET_HEIGHT,
                    TARGET_WIDTH,
                )
            )
        ):

            raise RuntimeError(
                "Unexpected tensor shape."
            )

        if (
            result["tensor_min"] is None
            or result["tensor_max"] is None
        ):

            raise RuntimeError(
                "Invalid tensor statistics."
            )

        result[
            "status"
        ] = "PASS"

    except Exception as exc:

        result[
            "error"
        ] = str(exc)

    return result


# ============================================================
# SAMPLE SELECTION
# ============================================================

def select_samples(
    train,
    val
):

    banner("SELECTING VALIDATION SAMPLE")

    rng = np.random.default_rng(
        SEED
    )

    train_n = min(
        TRAIN_SAMPLE_SIZE,
        len(train)
    )

    val_n = min(
        VAL_SAMPLE_SIZE,
        len(val)
    )

    train_indices = rng.choice(
        len(train),
        size=train_n,
        replace=False
    )

    val_indices = rng.choice(
        len(val),
        size=val_n,
        replace=False
    )

    train_sample = (
        train.iloc[
            train_indices
        ]
        .copy()
        .reset_index(drop=True)
    )

    val_sample = (
        val.iloc[
            val_indices
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"Training samples selected   : "
        f"{len(train_sample)}"
    )

    print(
        f"Validation samples selected : "
        f"{len(val_sample)}"
    )

    return (
        train_sample,
        val_sample
    )


# ============================================================
# SAMPLE VALIDATION
# ============================================================

def validate_samples(
    train_sample,
    val_sample
):

    banner("VALIDATING TRAINING SAMPLES")

    results = []

    for index, (_, row) in enumerate(
        train_sample.iterrows(),
        start=1
    ):

        print(
            f"TRAIN {index:02d}/"
            f"{len(train_sample):02d} "
            f"study={row['study_id']} "
            f"series={row['series_description']}"
        )

        result = validate_one_study(
            row,
            "train"
        )

        results.append(
            result
        )

        print(
            f"    status={result['status']} "
            f"slices={result['readable_dicom_count']} "
            f"shape="
            f"{result['original_depth']}x"
            f"{result['original_height']}x"
            f"{result['original_width']}"
        )

        if result["error"]:

            print(
                f"    ERROR: "
                f"{result['error']}"
            )

    banner("VALIDATING VALIDATION SAMPLES")

    for index, (_, row) in enumerate(
        val_sample.iterrows(),
        start=1
    ):

        print(
            f"VAL   {index:02d}/"
            f"{len(val_sample):02d} "
            f"study={row['study_id']} "
            f"series={row['series_description']}"
        )

        result = validate_one_study(
            row,
            "validation"
        )

        results.append(
            result
        )

        print(
            f"    status={result['status']} "
            f"slices={result['readable_dicom_count']} "
            f"shape="
            f"{result['original_depth']}x"
            f"{result['original_height']}x"
            f"{result['original_width']}"
        )

        if result["error"]:

            print(
                f"    ERROR: "
                f"{result['error']}"
            )

    return pd.DataFrame(
        results
    )


# ============================================================
# GLOBAL CHECKS
# ============================================================

def perform_global_checks(
    train,
    val,
    results
):

    banner("PART 88 GLOBAL CHECKS")

    checks = {}

    # --------------------------------------------------------
    # Study leakage
    # --------------------------------------------------------

    train_ids = set(
        train["study_id"]
        .astype(str)
    )

    val_ids = set(
        val["study_id"]
        .astype(str)
    )

    overlap = (
        train_ids.intersection(
            val_ids
        )
    )

    checks[
        "study_leakage"
    ] = len(overlap) == 0

    print(
        f"Study overlap in Part 87 manifests : "
        f"{len(overlap)}"
    )

    # --------------------------------------------------------
    # Sample loading
    # --------------------------------------------------------

    pass_count = int(
        (
            results["status"]
            == "PASS"
        ).sum()
    )

    fail_count = int(
        (
            results["status"]
            == "FAILED"
        ).sum()
    )

    checks[
        "sample_loading"
    ] = fail_count == 0

    print(
        f"Sample loading PASS : {pass_count}"
    )

    print(
        f"Sample loading FAIL : {fail_count}"
    )

    # --------------------------------------------------------
    # Tensor shapes
    # --------------------------------------------------------

    expected_shape = str(
        (
            1,
            TARGET_DEPTH,
            TARGET_HEIGHT,
            TARGET_WIDTH,
        )
    )

    shape_ok = (
        results[
            "tensor_shape"
        ]
        == expected_shape
    )

    shape_fail_count = int(
        (
            ~shape_ok
        ).sum()
    )

    checks[
        "tensor_shape"
    ] = shape_fail_count == 0

    print(
        f"Tensor shape failures : "
        f"{shape_fail_count}"
    )

    # --------------------------------------------------------
    # Numeric validity
    # --------------------------------------------------------

    numeric_ok = True

    for column in [
        "tensor_min",
        "tensor_max",
        "tensor_mean",
        "tensor_std",
    ]:

        values = pd.to_numeric(
            results[column],
            errors="coerce"
        )

        if values.isna().any():

            numeric_ok = False

        if not np.isfinite(
            values.fillna(0).to_numpy()
        ).all():

            numeric_ok = False

    checks[
        "numeric_tensor_values"
    ] = numeric_ok

    print(
        f"Tensor numeric validity : "
        f"{'PASS' if numeric_ok else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Normalization range
    # --------------------------------------------------------

    normalized_ok = True

    for column in [
        "normalized_min",
        "normalized_max",
    ]:

        values = pd.to_numeric(
            results[column],
            errors="coerce"
        )

        if values.isna().any():

            normalized_ok = False
            continue

        if (
            values.min() < -1e-5
            or values.max() > 1.00001
        ):

            normalized_ok = False

    checks[
        "normalization_range"
    ] = normalized_ok

    print(
        f"Normalization [0,1] : "
        f"{'PASS' if normalized_ok else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    unknown_labels = int(
        results[
            "unknown_label_count"
        ]
        .sum()
    )

    checks[
        "known_labels"
    ] = unknown_labels == 0

    print(
        f"Unknown label values : "
        f"{unknown_labels}"
    )

    # --------------------------------------------------------
    # Series type
    # --------------------------------------------------------

    unexpected_series = sorted(
        set(
            results[
                "series_description"
            ]
        )
        - set(EXPECTED_SERIES)
    )

    checks[
        "expected_series_types"
    ] = len(
        unexpected_series
    ) == 0

    print(
        "Unexpected selected series : "
        f"{unexpected_series}"
    )

    return checks, sorted(
        overlap
    )


# ============================================================
# REPORT
# ============================================================

def write_report(
    train,
    val,
    results,
    checks,
    overlap
):

    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "PART 88 — RSNA MRI LOADER & PREPROCESSING VALIDATION"
    )

    lines.append(
        "=" * 78
    )

    lines.append("")

    lines.append(
        f"Project root: {ROOT}"
    )

    lines.append(
        f"RSNA root: {RSNA_ROOT}"
    )

    lines.append("")

    lines.append(
        "PART 87 SPLIT"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Training studies: {len(train)}"
    )

    lines.append(
        f"Validation studies: {len(val)}"
    )

    lines.append(
        f"Study overlap: {len(overlap)}"
    )

    lines.append("")

    lines.append(
        "SAMPLE VALIDATION"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Total tested studies: {len(results)}"
    )

    lines.append(
        f"Passed: "
        f"{int((results['status'] == 'PASS').sum())}"
    )

    lines.append(
        f"Failed: "
        f"{int((results['status'] == 'FAILED').sum())}"
    )

    lines.append("")

    lines.append(
        "PREPROCESSING TARGET"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        "Input: DICOM series"
    )

    lines.append(
        "Normalization: percentile clipping "
        "1st–99th percentile"
    )

    lines.append(
        "Output tensor: "
        f"(1, {TARGET_DEPTH}, "
        f"{TARGET_HEIGHT}, {TARGET_WIDTH})"
    )

    lines.append(
        "Tensor range: [0, 1]"
    )

    lines.append("")

    lines.append(
        "SERIES DISTRIBUTION IN SAMPLE"
    )

    lines.append(
        "-" * 78
    )

    distribution = (
        results[
            "series_description"
        ]
        .value_counts()
    )

    for series_name, count in distribution.items():

        lines.append(
            f"{series_name:25s} : "
            f"{int(count):3d}"
        )

    lines.append("")

    lines.append(
        "SLICE STATISTICS"
    )

    lines.append(
        "-" * 78
    )

    readable = pd.to_numeric(
        results[
            "readable_dicom_count"
        ],
        errors="coerce"
    )

    depths = pd.to_numeric(
        results[
            "original_depth"
        ],
        errors="coerce"
    )

    lines.append(
        f"Readable slices — mean: "
        f"{readable.mean():.2f}"
    )

    lines.append(
        f"Readable slices — min: "
        f"{int(readable.min())}"
    )

    lines.append(
        f"Readable slices — max: "
        f"{int(readable.max())}"
    )

    lines.append(
        f"Original depth — mean: "
        f"{depths.mean():.2f}"
    )

    lines.append(
        f"Original depth — min: "
        f"{int(depths.min())}"
    )

    lines.append(
        f"Original depth — max: "
        f"{int(depths.max())}"
    )

    lines.append("")

    lines.append(
        "GLOBAL CHECKS"
    )

    lines.append(
        "-" * 78
    )

    for name, passed in checks.items():

        lines.append(
            f"{name:<30}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    overall = all(
        checks.values()
    )

    lines.append("")

    lines.append(
        "FINAL STATUS"
    )

    lines.append(
        "-" * 78
    )

    if overall:

        lines.append(
            "PASS — MRI loading and preprocessing "
            "validation completed successfully."
        )

    else:

        lines.append(
            "REVIEW REQUIRED — One or more "
            "Part 88 checks failed."
        )

    lines.append("")

    lines.append(
        "No neural-network training was performed."
    )

    lines.append(
        "Part 84/85 segmentation files were not modified."
    )

    return "\n".join(lines), overall


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    banner(
        "PART 88 — RSNA MRI LOADER & PREPROCESSING VALIDATION"
    )

    print(
        f"Device available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"CUDA device: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    validate_paths()

    # --------------------------------------------------------
    # Manifests
    # --------------------------------------------------------

    train, val = load_manifests()

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    (
        train_sample,
        val_sample
    ) = select_samples(
        train,
        val
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    results = validate_samples(
        train_sample,
        val_sample
    )

    # --------------------------------------------------------
    # Global checks
    # --------------------------------------------------------

    (
        checks,
        overlap
    ) = perform_global_checks(
        train,
        val,
        results
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results.to_csv(
        SAMPLE_RESULTS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    (
        report,
        overall_pass
    ) = write_report(
        train,
        val,
        results,
        checks,
        overlap
    )

    with open(
        REPORT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    summary = {
        "part": 88,

        "title":
            "RSNA MRI Loader and Preprocessing Validation",

        "seed": SEED,

        "dataset": {
            "training_studies":
                int(len(train)),

            "validation_studies":
                int(len(val)),
        },

        "sample": {
            "training_samples":
                int(len(train_sample)),

            "validation_samples":
                int(len(val_sample)),

            "total_samples":
                int(len(results)),
        },

        "preprocessing": {
            "normalization":
                "1st-99th percentile clipping",

            "normalization_range":
                [0.0, 1.0],

            "target_tensor_shape":
                [
                    1,
                    TARGET_DEPTH,
                    TARGET_HEIGHT,
                    TARGET_WIDTH,
                ],
        },

        "checks": {
            key: bool(value)
            for key, value in checks.items()
        },

        "study_overlap_count":
            int(len(overlap)),

        "passed_samples":
            int(
                (
                    results["status"]
                    == "PASS"
                ).sum()
            ),

        "failed_samples":
            int(
                (
                    results["status"]
                    == "FAILED"
                ).sum()
            ),

        "series_distribution":
            {
                str(k): int(v)
                for k, v
                in results[
                    "series_description"
                ].value_counts().items()
            },

        "outputs": {
            "sample_results":
                str(SAMPLE_RESULTS_CSV),

            "report":
                str(REPORT_TXT),

            "summary_json":
                str(SUMMARY_JSON),
        },

        "training_performed":
            False,

        "segmentation_modified":
            False,

        "overall_status":
            "PASS"
            if overall_pass
            else "REVIEW_REQUIRED",
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    banner(
        "PART 88 COMPLETE"
    )

    print(
        f"Samples tested : {len(results)}"
    )

    print(
        f"Samples passed : "
        f"{int((results['status'] == 'PASS').sum())}"
    )

    print(
        f"Samples failed : "
        f"{int((results['status'] == 'FAILED').sum())}"
    )

    print(
        f"Study overlap  : {len(overlap)}"
    )

    print(
        f"Target tensor  : "
        f"(1, {TARGET_DEPTH}, "
        f"{TARGET_HEIGHT}, {TARGET_WIDTH})"
    )

    print(
        f"\nDetailed results:\n"
        f"{SAMPLE_RESULTS_CSV}"
    )

    print(
        f"\nReport:\n"
        f"{REPORT_TXT}"
    )

    print(
        f"\nSummary JSON:\n"
        f"{SUMMARY_JSON}"
    )

    if overall_pass:

        print(
            "\nSTATUS: PASS — "
            "READY FOR CLASSIFICATION DATASET PIPELINE"
        )

    else:

        print(
            "\nSTATUS: REVIEW REQUIRED — "
            "DO NOT START CLASSIFIER TRAINING YET"
        )


if __name__ == "__main__":
    main()