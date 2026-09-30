"""
PHASE 4 - PART 33
RSNA-ONLY PREPROCESSING MASK-INTEGRITY / FOREGROUND-RETENTION AUDIT

This audit is designed specifically against the validated Part 11 pipeline.

Part 11's validated flow is:
    Part 9 load_case()
        -> local mixed-DICOM-shape fallback when required
        -> pseudo-mask loading/harmonization
        -> preprocess_case()
        -> final spatial size (64, 96, 96)

Part 33 isolates the mask transformation performed by preprocess_case().
It compares:
    RAW / ALIGNED MASK
        ->
    Part 11 preprocess_case()
        ->
    PROCESSED MASK

No model is created and no training is performed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# =============================================================================
# PATHS
# =============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

PART11_SOURCE = (
    SRC_DIR
    / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

PART8_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
)

PART8_TRAIN_MANIFEST = (
    PART8_DIR
    / "manifests"
    / "rsna_part8_train_manifest.csv"
)

PART11_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part11_controlled_pilot_training"
)

PART11_RESULT = (
    PART11_DIR
    / "part11_training_result.json"
)

PART11_HISTORY = (
    PART11_DIR
    / "part11_training_history.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part33_preprocessing_mask_integrity_audit"
)

CASE_CSV = (
    OUTPUT_DIR
    / "part33_case_preprocessing_integrity.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "phase4_part33_summary.json"
)

REPORT_TXT = (
    OUTPUT_DIR
    / "phase4_part33_report.txt"
)


# =============================================================================
# LOCKED PART 11 PREPROCESSING CONTRACT
# =============================================================================

PATCH_SIZE = (64, 96, 96)
NUM_CLASSES = 6

FOREGROUND_CLASSES = [1, 2, 3, 4, 5]

LABELS = {
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


def line(name: str, value: Any) -> None:
    print(f"{name:<42}: {value}")


# =============================================================================
# VALUE HELPERS
# =============================================================================

def first_value(
    row: pd.Series,
    names: List[str],
    default: Any = None,
) -> Any:
    for name in names:
        if name not in row.index:
            continue

        value = row[name]

        try:
            if pd.isna(value):
                continue
        except Exception:
            pass

        return value

    return default


def as_id(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


# =============================================================================
# MASK NORMALIZATION / STATISTICS
# =============================================================================

def normalize_mask(mask: Any) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)

    # Remove only singleton batch/channel dimensions.
    while arr.ndim > 3:
        if arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        elif arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        else:
            raise RuntimeError(
                f"Cannot reduce mask to 3-D: shape={arr.shape}"
            )

    if arr.ndim != 3:
        raise RuntimeError(
            f"Expected 3-D mask, got shape={arr.shape}"
        )

    if not np.isfinite(arr).all():
        raise RuntimeError(
            "Mask contains non-finite values."
        )

    rounded = np.rint(arr)

    if not np.array_equal(arr, rounded):
        raise RuntimeError(
            "Mask contains non-integer label values."
        )

    arr = rounded.astype(np.int64)

    unique = np.unique(arr)

    invalid = [
        int(x)
        for x in unique
        if int(x) < 0 or int(x) >= NUM_CLASSES
    ]

    if invalid:
        raise RuntimeError(
            f"Mask contains invalid class IDs: {invalid}"
        )

    return arr


def bbox_size(
    mask: np.ndarray,
    class_id: int,
) -> str:
    coords = np.argwhere(
        mask == class_id
    )

    if coords.size == 0:
        return ""

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    size = maxs - mins + 1

    return "x".join(
        str(int(v))
        for v in size
    )


def mask_statistics(
    mask: Any,
) -> Dict[str, Any]:
    arr = normalize_mask(mask)

    total = int(arr.size)

    result: Dict[str, Any] = {
        "shape": tuple(
            int(x)
            for x in arr.shape
        ),
        "total_voxels": total,
        "unique_labels": [
            int(x)
            for x in np.unique(arr)
        ],
    }

    foreground = 0

    for class_id in range(NUM_CLASSES):
        count = int(
            np.count_nonzero(
                arr == class_id
            )
        )

        result[
            f"class_{class_id}_voxels"
        ] = count

        if class_id in FOREGROUND_CLASSES:
            foreground += count

        if class_id in FOREGROUND_CLASSES:
            result[
                f"class_{class_id}_bbox_size"
            ] = bbox_size(
                arr,
                class_id,
            )

    result[
        "foreground_voxels"
    ] = foreground

    result[
        "background_voxels"
    ] = result[
        "class_0_voxels"
    ]

    result[
        "foreground_fraction"
    ] = (
        foreground / total
        if total
        else 0.0
    )

    return result


# =============================================================================
# PART 11 IMPORT
# =============================================================================

def import_part11():
    header("IMPORTING VALIDATED PART 11")

    if not PART11_SOURCE.exists():
        raise FileNotFoundError(
            f"Part 11 source not found: {PART11_SOURCE}"
        )

    spec = importlib.util.spec_from_file_location(
        "part11_for_part33",
        str(PART11_SOURCE),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import Part 11: {PART11_SOURCE}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        "part11_for_part33"
    ] = module

    spec.loader.exec_module(module)

    required = [
        "load_part9_module",
        "preprocess_case",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise AttributeError(
            "Part 11 implementation is missing: "
            + ", ".join(missing)
        )

    print("✓ Part 11 imported.")
    print("✓ load_part9_module() available.")
    print("✓ preprocess_case() available.")

    return module


# =============================================================================
# PART 9
# =============================================================================

def load_part9(part11: Any) -> Any:
    header("LOADING PART 9 THROUGH PART 11")

    try:
        part9 = part11.load_part9_module()
    except TypeError:
        part9 = part11.load_part9_module(
            RSNA_ROOT
        )

    if not hasattr(part9, "load_case"):
        raise RuntimeError(
            "Part 9 loader does not expose load_case(row)."
        )

    print("✓ Part 9 loader imported.")
    print("✓ load_case(row) available.")

    return part9


# =============================================================================
# RAW CASE LOADING
# =============================================================================

def load_raw_case(
    row: pd.Series,
    part9: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Obtain the Part 9-aligned image/mask pair.

    Important:
    Part 11's load_tensor_case() calls preprocess_case().
    Therefore Part 33 deliberately calls Part 9 load_case() directly
    so that preprocessing is measured exactly once here.
    """

    image, mask, info = part9.load_case(
        row
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    mask = normalize_mask(
        mask
    )

    info = (
        dict(info)
        if isinstance(info, dict)
        else {}
    )

    return image, mask, info


# =============================================================================
# EXACT PART 11 PREPROCESSING
# =============================================================================

def apply_part11_preprocessing(
    part11: Any,
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the exact Part 11 preprocess_case() implementation once.

    Part 11 source confirms:
        image -> resize_3d(..., PATCH_SIZE, is_mask=False)
        mask  -> resize_3d(..., PATCH_SIZE, is_mask=True)

    The mask path uses nearest-neighbor interpolation and converts the
    result back to integer class IDs.
    """

    result = part11.preprocess_case(
        image,
        mask,
    )

    if not isinstance(result, tuple):
        raise RuntimeError(
            "Part 11 preprocess_case() did not return a tuple."
        )

    if len(result) < 2:
        raise RuntimeError(
            "Part 11 preprocess_case() did not return "
            "(image_tensor, mask_tensor)."
        )

    processed_image = result[0]
    processed_mask = result[1]

    processed_mask = normalize_mask(
        processed_mask
    )

    processed_image = np.asarray(
        processed_image.detach().cpu().numpy()
        if isinstance(
            processed_image,
            torch.Tensor,
        )
        else processed_image,
        dtype=np.float32,
    )

    return (
        processed_image,
        processed_mask,
    )


# =============================================================================
# SINGLE CASE AUDIT
# =============================================================================

def audit_case(
    index: int,
    row: pd.Series,
    part11: Any,
    part9: Any,
) -> Dict[str, Any]:

    study_id = as_id(
        first_value(
            row,
            ["study_id", "study"],
        )
    )

    series_id = as_id(
        first_value(
            row,
            ["series_id", "series"],
        )
    )

    result: Dict[str, Any] = {
        "case_index": index,
        "study_id": study_id,
        "series_id": series_id,
        "status": "ERROR",
        "error": "",
    }

    try:
        # -------------------------------------------------------------
        # RAW / ALIGNED
        # -------------------------------------------------------------
        image, raw_mask, info = load_raw_case(
            row,
            part9,
        )

        raw_stats = mask_statistics(
            raw_mask
        )

        result[
            "raw_image_shape"
        ] = str(
            tuple(image.shape)
        )

        result[
            "raw_mask_shape"
        ] = str(
            raw_stats["shape"]
        )

        result[
            "raw_unique_labels"
        ] = ",".join(
            str(x)
            for x in raw_stats[
                "unique_labels"
            ]
        )

        result[
            "raw_foreground_voxels"
        ] = raw_stats[
            "foreground_voxels"
        ]

        result[
            "raw_background_voxels"
        ] = raw_stats[
            "background_voxels"
        ]

        result[
            "raw_foreground_fraction"
        ] = raw_stats[
            "foreground_fraction"
        ]

        for class_id in range(
            NUM_CLASSES
        ):
            result[
                f"raw_class_{class_id}_voxels"
            ] = raw_stats[
                f"class_{class_id}_voxels"
            ]

        for class_id in FOREGROUND_CLASSES:
            result[
                f"raw_class_{class_id}_bbox_size"
            ] = raw_stats[
                f"class_{class_id}_bbox_size"
            ]

        # -------------------------------------------------------------
        # EXACT PART 11 PREPROCESSING
        # -------------------------------------------------------------
        processed_image, processed_mask = (
            apply_part11_preprocessing(
                part11,
                image,
                raw_mask,
            )
        )

        processed_stats = mask_statistics(
            processed_mask
        )

        result[
            "processed_image_shape"
        ] = str(
            tuple(processed_image.shape)
        )

        result[
            "processed_mask_shape"
        ] = str(
            processed_stats["shape"]
        )

        result[
            "processed_unique_labels"
        ] = ",".join(
            str(x)
            for x in processed_stats[
                "unique_labels"
            ]
        )

        result[
            "processed_foreground_voxels"
        ] = processed_stats[
            "foreground_voxels"
        ]

        result[
            "processed_background_voxels"
        ] = processed_stats[
            "background_voxels"
        ]

        result[
            "processed_foreground_fraction"
        ] = processed_stats[
            "foreground_fraction"
        ]

        for class_id in range(
            NUM_CLASSES
        ):
            result[
                f"processed_class_{class_id}_voxels"
            ] = processed_stats[
                f"class_{class_id}_voxels"
            ]

        for class_id in FOREGROUND_CLASSES:
            result[
                f"processed_class_{class_id}_bbox_size"
            ] = processed_stats[
                f"class_{class_id}_bbox_size"
            ]

        # -------------------------------------------------------------
        # OVERALL FOREGROUND RETENTION
        # -------------------------------------------------------------
        raw_fg = int(
            raw_stats[
                "foreground_voxels"
            ]
        )

        processed_fg = int(
            processed_stats[
                "foreground_voxels"
            ]
        )

        result[
            "foreground_voxel_difference"
        ] = (
            processed_fg - raw_fg
        )

        result[
            "foreground_voxels_lost"
        ] = max(
            0,
            raw_fg - processed_fg,
        )

        result[
            "foreground_retention_ratio"
        ] = (
            processed_fg / raw_fg
            if raw_fg > 0
            else np.nan
        )

        result[
            "foreground_loss_fraction"
        ] = (
            1.0
            - (
                processed_fg / raw_fg
            )
            if raw_fg > 0
            else np.nan
        )

        result[
            "foreground_destroyed"
        ] = bool(
            raw_fg > 0
            and processed_fg == 0
        )

        result[
            "spatial_shape_changed"
        ] = bool(
            raw_stats["shape"]
            != processed_stats["shape"]
        )

        # -------------------------------------------------------------
        # CLASS-WISE RETENTION
        # -------------------------------------------------------------
        raw_foreground_classes = []
        processed_foreground_classes = []
        removed_classes = []

        for class_id in FOREGROUND_CLASSES:
            raw_count = int(
                raw_stats[
                    f"class_{class_id}_voxels"
                ]
            )

            processed_count = int(
                processed_stats[
                    f"class_{class_id}_voxels"
                ]
            )

            if raw_count > 0:
                raw_foreground_classes.append(
                    class_id
                )

            if processed_count > 0:
                processed_foreground_classes.append(
                    class_id
                )

            result[
                f"class_{class_id}_voxel_difference"
            ] = (
                processed_count
                - raw_count
            )

            result[
                f"class_{class_id}_voxels_lost"
            ] = max(
                0,
                raw_count
                - processed_count,
            )

            result[
                f"class_{class_id}_retention_ratio"
            ] = (
                processed_count / raw_count
                if raw_count > 0
                else np.nan
            )

            result[
                f"class_{class_id}_destroyed"
            ] = bool(
                raw_count > 0
                and processed_count == 0
            )

            if (
                raw_count > 0
                and processed_count == 0
            ):
                removed_classes.append(
                    class_id
                )

        result[
            "raw_foreground_classes"
        ] = ",".join(
            str(x)
            for x in raw_foreground_classes
        )

        result[
            "processed_foreground_classes"
        ] = ",".join(
            str(x)
            for x in processed_foreground_classes
        )

        result[
            "classes_removed_by_preprocessing"
        ] = ",".join(
            str(x)
            for x in removed_classes
        )

        result[
            "any_foreground_class_destroyed"
        ] = bool(
            removed_classes
        )

        # -------------------------------------------------------------
        # LABEL-SET INTEGRITY
        # -------------------------------------------------------------
        raw_labels = set(
            raw_stats[
                "unique_labels"
            ]
        )

        processed_labels = set(
            processed_stats[
                "unique_labels"
            ]
        )

        invalid_processed = sorted(
            processed_labels
            - set(range(NUM_CLASSES))
        )

        result[
            "invalid_processed_labels"
        ] = ",".join(
            str(x)
            for x in invalid_processed
        )

        result[
            "label_ids_valid"
        ] = bool(
            not invalid_processed
        )

        result[
            "no_new_label_ids"
        ] = bool(
            processed_labels.issubset(
                raw_labels
            )
        )

        # -------------------------------------------------------------
        # PART 11 LOADER METADATA
        # -------------------------------------------------------------
        result[
            "part11_loader"
        ] = str(
            info.get(
                "part11_loader",
                "part9_direct",
            )
        )

        result[
            "dicom_harmonized"
        ] = bool(
            info.get(
                "dicom_harmonized",
                False,
            )
        )

        result[
            "status"
        ] = "OK"

    except Exception as exc:
        result[
            "error"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    summary: Dict[str, Any] = {
        "phase": "Phase 4 - Part 33",
        "description": (
            "RSNA-only preprocessing mask-integrity "
            "and foreground-retention audit"
        ),
        "audited_cases": int(
            len(df)
        ),
        "successful_cases": 0,
        "failed_cases": 0,
    }

    if df.empty:
        summary[
            "diagnosis"
        ] = "NO_CASES_AUDITED"
        summary[
            "explanation"
        ] = "No cohort rows were available."
        return summary

    ok = df[
        df["status"] == "OK"
    ].copy()

    summary[
        "successful_cases"
    ] = int(len(ok))

    summary[
        "failed_cases"
    ] = int(
        len(df) - len(ok)
    )

    if ok.empty:
        summary[
            "diagnosis"
        ] = "NO_SUCCESSFUL_CASES"
        summary[
            "explanation"
        ] = (
            "Every attempted case failed before the "
            "raw-to-processed mask comparison."
        )
        return summary

    retention = pd.to_numeric(
        ok[
            "foreground_retention_ratio"
        ],
        errors="coerce",
    )

    raw_fg = pd.to_numeric(
        ok[
            "raw_foreground_voxels"
        ],
        errors="coerce",
    )

    processed_fg = pd.to_numeric(
        ok[
            "processed_foreground_voxels"
        ],
        errors="coerce",
    )

    destroyed = ok[
        "foreground_destroyed"
    ].astype(bool)

    shape_changed = ok[
        "spatial_shape_changed"
    ].astype(bool)

    summary.update(
        {
            "mean_raw_foreground_voxels": float(
                raw_fg.mean()
            ),
            "median_raw_foreground_voxels": float(
                raw_fg.median()
            ),
            "mean_processed_foreground_voxels": float(
                processed_fg.mean()
            ),
            "median_processed_foreground_voxels": float(
                processed_fg.median()
            ),
            "mean_foreground_retention_ratio": float(
                retention.mean()
            ),
            "median_foreground_retention_ratio": float(
                retention.median()
            ),
            "minimum_foreground_retention_ratio": float(
                retention.min()
            ),
            "maximum_foreground_retention_ratio": float(
                retention.max()
            ),
            "foreground_destroyed_cases": int(
                destroyed.sum()
            ),
            "foreground_destroyed_rate": float(
                destroyed.mean()
            ),
            "spatial_shape_changed_cases": int(
                shape_changed.sum()
            ),
        }
    )

    for class_id in FOREGROUND_CLASSES:
        raw_col = (
            f"raw_class_{class_id}_voxels"
        )
        processed_col = (
            f"processed_class_{class_id}_voxels"
        )
        retention_col = (
            f"class_{class_id}_retention_ratio"
        )

        raw_class = pd.to_numeric(
            ok[raw_col],
            errors="coerce",
        )

        processed_class = pd.to_numeric(
            ok[processed_col],
            errors="coerce",
        )

        class_retention = pd.to_numeric(
            ok[retention_col],
            errors="coerce",
        )

        summary[
            f"class_{class_id}_name"
        ] = LABELS[class_id]

        summary[
            f"class_{class_id}_raw_total_voxels"
        ] = int(
            raw_class.sum()
        )

        summary[
            f"class_{class_id}_processed_total_voxels"
        ] = int(
            processed_class.sum()
        )

        summary[
            f"class_{class_id}_mean_retention_ratio"
        ] = float(
            class_retention.mean()
        )

        summary[
            f"class_{class_id}_destroyed_cases"
        ] = int(
            (
                (raw_class > 0)
                & (processed_class == 0)
            ).sum()
        )

    # -------------------------------------------------------------
    # Diagnosis
    # -------------------------------------------------------------
    if destroyed.any():
        diagnosis = (
            "PREPROCESSING_CAN_DESTROY_FOREGROUND"
        )

        explanation = (
            "At least one successful case loses all "
            "foreground voxels during the Part 11 "
            "preprocessing transformation."
        )

    elif shape_changed.any():
        diagnosis = (
            "SPATIAL_RESIZING_PRESENT"
        )

        explanation = (
            "The preprocessing stage changes spatial "
            "dimensions. Because Part 11 resizes masks "
            "to the locked patch size, foreground voxel "
            "counts must be interpreted in the context "
            "of that spatial transformation."
        )

    else:
        diagnosis = (
            "NO_SPATIAL_SHAPE_CHANGE_DETECTED"
        )

        explanation = (
            "The raw and processed masks have the same "
            "spatial shape for the successful cases."
        )

    summary[
        "diagnosis"
    ] = diagnosis

    summary[
        "explanation"
    ] = explanation

    summary[
        "part11_patch_size"
    ] = list(PATCH_SIZE)

    summary[
        "part11_num_classes"
    ] = NUM_CLASSES

    summary[
        "training_performed"
    ] = False

    summary[
        "model_weights_modified"
    ] = False

    summary[
        "optimizer_created"
    ] = False

    summary[
        "spider_used"
    ] = False

    summary[
        "rsna_test_set_used"
    ] = False

    return summary


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    summary: Dict[str, Any],
    df: pd.DataFrame,
) -> None:

    with REPORT_TXT.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 4 - PART 33\n"
        )
        f.write(
            "RSNA-ONLY PREPROCESSING MASK-INTEGRITY / "
            "FOREGROUND-RETENTION AUDIT\n\n"
        )

        f.write(
            "Purpose\n"
        )
        f.write(
            "-------\n"
        )
        f.write(
            "Measure how the exact Part 11 preprocess_case() "
            "transforms the mask.\n\n"
        )

        f.write(
            "Controls\n"
        )
        f.write(
            "--------\n"
        )
        f.write(
            "Training performed: NO\n"
        )
        f.write(
            "Model weights modified: NO\n"
        )
        f.write(
            "Optimizer created: NO\n"
        )
        f.write(
            "SPIDER used: NO\n"
        )
        f.write(
            "RSNA test set used: NO\n\n"
        )

        f.write(
            "DIAGNOSIS\n"
        )
        f.write(
            f"{summary.get('diagnosis', '')}\n\n"
        )

        f.write(
            "EXPLANATION\n"
        )
        f.write(
            f"{summary.get('explanation', '')}\n\n"
        )

        f.write(
            "SUMMARY VALUES\n"
        )
        f.write(
            "-" * 78
            + "\n"
        )

        for key, value in summary.items():
            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            "\nCASE-LEVEL RESULTS\n"
        )
        f.write(
            "-" * 78
            + "\n"
        )

        preferred = [
            "case_index",
            "study_id",
            "series_id",
            "status",
            "raw_image_shape",
            "raw_mask_shape",
            "processed_image_shape",
            "processed_mask_shape",
            "raw_unique_labels",
            "processed_unique_labels",
            "raw_foreground_voxels",
            "processed_foreground_voxels",
            "foreground_voxels_lost",
            "foreground_retention_ratio",
            "foreground_loss_fraction",
            "foreground_destroyed",
            "spatial_shape_changed",
            "raw_foreground_classes",
            "processed_foreground_classes",
            "classes_removed_by_preprocessing",
            "any_foreground_class_destroyed",
            "part11_loader",
            "dicom_harmonized",
            "error",
        ]

        columns = [
            c
            for c in preferred
            if c in df.columns
        ]

        if columns:
            f.write(
                df[
                    columns
                ].to_string(
                    index=False
                )
            )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    header("PHASE 4 - PART 33")

    print(
        "RSNA-ONLY PREPROCESSING MASK-INTEGRITY / "
        "FOREGROUND-RETENTION AUDIT"
    )

    print()
    print("Evaluation / audit only.")
    print("No training.")
    print("No model creation.")
    print("No model-weight modification.")
    print("No optimizer.")
    print("No SPIDER.")
    print("No RSNA test set.")

    line(
        "PROJECT ROOT",
        PROJECT_ROOT,
    )
    line(
        "RSNA DATASET",
        RSNA_ROOT,
    )
    line(
        "PART 11 SOURCE",
        PART11_SOURCE,
    )
    line(
        "PART 8 TRAIN MANIFEST",
        PART8_TRAIN_MANIFEST,
    )
    line(
        "OUTPUT DIRECTORY",
        OUTPUT_DIR,
    )

    # -----------------------------------------------------------------
    # Validate paths
    # -----------------------------------------------------------------
    header("PATH VALIDATION")

    required = {
        "RSNA root": RSNA_ROOT,
        "train images": RSNA_ROOT / "train_images",
        "Part 8 train manifest": PART8_TRAIN_MANIFEST,
        "Part 11 source": PART11_SOURCE,
    }

    missing = []

    for name, path in required.items():
        exists = path.exists()

        print(
            f"{name:<30}: "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:
            missing.append(
                f"{name}: {path}"
            )

    if missing:
        raise FileNotFoundError(
            "Missing required Part 33 input(s):\n"
            + "\n".join(missing)
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Environment
    # -----------------------------------------------------------------
    header("PYTORCH ENVIRONMENT")

    print(
        f"PyTorch version : {torch.__version__}"
    )
    print(
        f"CUDA available  : {torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------------------
    # Import validated modules
    # -----------------------------------------------------------------
    part11 = import_part11()
    part9 = load_part9(part11)

    # -----------------------------------------------------------------
    # Load cohort
    # -----------------------------------------------------------------
    header("LOADING PART 8 TRAIN MANIFEST")

    manifest = pd.read_csv(
        PART8_TRAIN_MANIFEST
    )

    if manifest.empty:
        raise RuntimeError(
            "Part 8 training manifest is empty."
        )

    line(
        "Manifest rows",
        len(manifest),
    )

    # -----------------------------------------------------------------
    # Audit all Part 8 training rows
    # -----------------------------------------------------------------
    header(
        "RUNNING RAW -> PART 11 PREPROCESSING AUDIT"
    )

    print(
        "Important: Part 9 load_case() is used directly."
    )
    print(
        "Part 11 load_tensor_case() is NOT used because "
        "it already calls preprocess_case()."
    )
    print(
        "Therefore each successful case is preprocessed exactly once."
    )

    results: List[Dict[str, Any]] = []

    total = len(manifest)

    for index, (_, row) in enumerate(
        manifest.iterrows(),
        start=1,
    ):

        result = audit_case(
            index,
            row,
            part11,
            part9,
        )

        results.append(
            result
        )

        if result["status"] == "OK":
            print(
                f"[{index:>4}/{total}] "
                f"study={result['study_id']} "
                f"series={result['series_id']} | "
                f"raw_fg="
                f"{result['raw_foreground_voxels']} | "
                f"processed_fg="
                f"{result['processed_foreground_voxels']} | "
                f"retention="
                f"{result['foreground_retention_ratio']:.6f}"
            )
        else:
            print(
                f"[{index:>4}/{total}] "
                f"ERROR | "
                f"{result['study_id']} "
                f"{result['series_id']} | "
                f"{result['error']}"
            )

    df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    header("PART 33 DIAGNOSIS")

    summary = build_summary(
        df
    )

    for key, value in summary.items():
        line(
            key,
            value,
        )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    header("SAVING PART 33 OUTPUTS")

    df.to_csv(
        CASE_CSV,
        index=False,
    )

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    write_report(
        summary,
        df,
    )

    line(
        "Case CSV",
        CASE_CSV,
    )
    line(
        "Summary JSON",
        SUMMARY_JSON,
    )
    line(
        "Report TXT",
        REPORT_TXT,
    )

    header("PHASE 4 - PART 33 COMPLETE")

    print(
        "✓ Raw/aligned masks audited."
    )
    print(
        "✓ Exact Part 11 preprocess_case() audited."
    )
    print(
        "✓ Per-class foreground retention calculated."
    )
    print(
        "✓ No training performed."
    )
    print(
        "✓ No model weights modified."
    )
    print(
        "✓ No optimizer created."
    )
    print(
        "✓ SPIDER not used."
    )
    print(
        "✓ RSNA test set not used."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPART 33 INTERRUPTED")
        raise
    except Exception:
        print("\nPART 33 ERROR")
        traceback.print_exc()
        raise
