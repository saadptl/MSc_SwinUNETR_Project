"""
PHASE 4 - PART 34
RSNA-ONLY SPATIAL RESAMPLING / CLASS-AWARE FOREGROUND CONSERVATION AUDIT

Purpose
-------
Part 33 established that the raw/loaded mask is transformed during the
validated Part 11 preprocessing step and that the final mask is resized to
the locked patch size (64, 96, 96).

Part 34 investigates whether the foreground-count reduction is explained
by the change of voxel grid and whether individual foreground classes have
additional disproportionate retention loss.

Audit only:
    - No training
    - No model creation
    - No optimizer
    - No optimizer step
    - No checkpoint modification
    - No SPIDER
    - No RSNA test set

Pipeline:
    Part 8 manifest
        -> Part 9 load_case()
        -> raw/aligned image + mask
        -> Part 11 preprocess_case() exactly once
        -> processed mask

Important:
Part 11's load_tensor_case() is intentionally NOT used because it already
calls preprocess_case(). Using it here would preprocess twice.
"""

from __future__ import annotations

import importlib.util
import inspect
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

PART8_TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
    / "rsna_part8_train_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part34_spatial_resampling_class_aware_foreground_conservation_audit"
)

CASE_CSV = OUTPUT_DIR / "part34_case_spatial_resampling_audit.csv"
FAILED_CSV = OUTPUT_DIR / "part34_failed_cases.csv"
CLASS_CSV = OUTPUT_DIR / "part34_class_retention_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "phase4_part34_summary.json"
REPORT_TXT = OUTPUT_DIR / "phase4_part34_report.txt"


# =============================================================================
# LOCKED CONSTANTS
# =============================================================================

PATCH_SIZE = (64, 96, 96)
NUM_CLASSES = 6
FOREGROUND_CLASSES = [1, 2, 3, 4, 5]

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


def line(name: str, value: Any) -> None:
    print(f"{name:<44}: {value}")


# =============================================================================
# VALUE HELPERS
# =============================================================================

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


def first_value(
    row: pd.Series,
    candidates: List[str],
) -> Any:
    for name in candidates:
        if name not in row.index:
            continue

        value = row[name]

        try:
            if pd.isna(value):
                continue
        except Exception:
            pass

        return value

    return ""


# =============================================================================
# MASK NORMALIZATION
# =============================================================================

def normalize_mask(mask: Any) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)

    # Remove only singleton dimensions.
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

    invalid = sorted(
        int(x)
        for x in np.unique(arr)
        if int(x) < 0 or int(x) >= NUM_CLASSES
    )

    if invalid:
        raise RuntimeError(
            f"Mask contains invalid labels: {invalid}"
        )

    return arr


# =============================================================================
# MASK STATISTICS
# =============================================================================

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
        str(int(x))
        for x in size
    )


def mask_stats(
    mask: Any,
) -> Dict[str, Any]:
    arr = normalize_mask(mask)

    stats: Dict[str, Any] = {
        "shape": tuple(
            int(x)
            for x in arr.shape
        ),
        "total_voxels": int(arr.size),
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

        stats[
            f"class_{class_id}_voxels"
        ] = count

        if class_id in FOREGROUND_CLASSES:
            foreground += count
            stats[
                f"class_{class_id}_bbox_size"
            ] = bbox_size(
                arr,
                class_id,
            )

    stats[
        "foreground_voxels"
    ] = int(foreground)

    stats[
        "background_voxels"
    ] = int(
        stats["class_0_voxels"]
    )

    stats[
        "foreground_fraction"
    ] = (
        foreground / arr.size
        if arr.size
        else 0.0
    )

    return stats


# =============================================================================
# IMPORT PART 11
# =============================================================================

def import_part11() -> Any:
    header("IMPORTING VALIDATED PART 11")

    if not PART11_SOURCE.exists():
        raise FileNotFoundError(
            f"Part 11 source not found: {PART11_SOURCE}"
        )

    spec = importlib.util.spec_from_file_location(
        "part11_for_part34",
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
        "part11_for_part34"
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
        raise RuntimeError(
            "Part 11 missing required APIs: "
            + ", ".join(missing)
        )

    print("✓ Corrected Part 11 imported.")
    print(
        "preprocess_case:",
        inspect.signature(
            module.preprocess_case
        ),
    )

    return module


# =============================================================================
# IMPORT PART 9
# =============================================================================

def import_part9(
    part11: Any,
) -> Any:
    header("LOADING PART 9 THROUGH PART 11")

    try:
        part9 = part11.load_part9_module()
    except TypeError:
        part9 = part11.load_part9_module(
            RSNA_ROOT
        )

    if not hasattr(
        part9,
        "load_case",
    ):
        raise RuntimeError(
            "Part 9 loader does not expose load_case(row)."
        )

    print("✓ Part 9 loader imported.")
    print("✓ load_case(row) available.")

    return part9


# =============================================================================
# RAW CASE
# =============================================================================

def load_raw_case(
    part9: Any,
    row: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Obtain raw/aligned 3-D image and mask directly from Part 9.
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

    if image.ndim != 3:
        raise RuntimeError(
            f"Expected raw image to be 3-D, got {image.shape}"
        )

    info = (
        dict(info)
        if isinstance(info, dict)
        else {}
    )

    return image, mask, info


# =============================================================================
# PREPROCESS EXACTLY ONCE
# =============================================================================

def preprocess_once(
    part11: Any,
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Apply the validated Part 11 preprocessing exactly once.
    """

    result = part11.preprocess_case(
        image,
        mask,
    )

    if not isinstance(result, tuple):
        raise RuntimeError(
            "preprocess_case() did not return a tuple."
        )

    if len(result) < 2:
        raise RuntimeError(
            "preprocess_case() returned fewer than two values."
        )

    return normalize_mask(
        result[1]
    )


# =============================================================================
# CASE AUDIT
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
        image, raw_mask, info = load_raw_case(
            part9,
            row,
        )

        raw = mask_stats(
            raw_mask
        )

        processed_mask = preprocess_once(
            part11,
            image,
            raw_mask,
        )

        processed = mask_stats(
            processed_mask
        )

        raw_shape = raw["shape"]
        processed_shape = processed["shape"]

        raw_grid_voxels = int(
            np.prod(raw_shape)
        )

        processed_grid_voxels = int(
            np.prod(processed_shape)
        )

        # This is the fraction of the original voxel grid represented
        # by the final fixed patch grid.
        grid_volume_scale = (
            processed_grid_voxels
            / raw_grid_voxels
            if raw_grid_voxels
            else np.nan
        )

        raw_fg = int(
            raw["foreground_voxels"]
        )

        processed_fg = int(
            processed["foreground_voxels"]
        )

        observed_retention = (
            processed_fg / raw_fg
            if raw_fg > 0
            else np.nan
        )

        # If foreground density were conserved perfectly, its voxel count
        # would scale approximately with the total voxel-grid ratio.
        expected_processed_fg = (
            raw_fg
            * grid_volume_scale
            if np.isfinite(grid_volume_scale)
            else np.nan
        )

        grid_normalized_retention = (
            processed_fg
            / expected_processed_fg
            if expected_processed_fg > 0
            else np.nan
        )

        result.update(
            {
                "raw_image_shape": str(
                    tuple(image.shape)
                ),
                "raw_mask_shape": str(
                    raw_shape
                ),
                "processed_mask_shape": str(
                    processed_shape
                ),
                "raw_grid_voxels": raw_grid_voxels,
                "processed_grid_voxels": processed_grid_voxels,
                "grid_volume_scale": grid_volume_scale,
                "expected_processed_foreground_voxels": (
                    expected_processed_fg
                ),
                "raw_foreground_voxels": raw_fg,
                "processed_foreground_voxels": processed_fg,
                "foreground_voxel_difference": (
                    processed_fg - raw_fg
                ),
                "foreground_voxels_lost": max(
                    0,
                    raw_fg - processed_fg,
                ),
                "observed_foreground_retention_ratio": (
                    observed_retention
                ),
                "grid_normalized_foreground_retention": (
                    grid_normalized_retention
                ),
                "foreground_minus_grid_expected": (
                    processed_fg
                    - expected_processed_fg
                    if np.isfinite(
                        expected_processed_fg
                    )
                    else np.nan
                ),
                "shape_changed": bool(
                    raw_shape != processed_shape
                ),
                "foreground_destroyed": bool(
                    raw_fg > 0
                    and processed_fg == 0
                ),
            }
        )

        # -------------------------------------------------------------
        # Label integrity
        # -------------------------------------------------------------
        raw_labels = set(
            raw["unique_labels"]
        )

        processed_labels = set(
            processed["unique_labels"]
        )

        invalid_processed = sorted(
            int(x)
            for x in processed_labels
            if x < 0 or x >= NUM_CLASSES
        )

        result[
            "raw_unique_labels"
        ] = ",".join(
            str(x)
            for x in sorted(raw_labels)
        )

        result[
            "processed_unique_labels"
        ] = ",".join(
            str(x)
            for x in sorted(processed_labels)
        )

        result[
            "invalid_processed_labels"
        ] = ",".join(
            str(x)
            for x in invalid_processed
        )

        result[
            "processed_has_invalid_labels"
        ] = bool(
            invalid_processed
        )

        # -------------------------------------------------------------
        # Per-class analysis
        # -------------------------------------------------------------
        raw_classes = set()
        processed_classes = set()

        for class_id in FOREGROUND_CLASSES:

            raw_count = int(
                raw[
                    f"class_{class_id}_voxels"
                ]
            )

            processed_count = int(
                processed[
                    f"class_{class_id}_voxels"
                ]
            )

            if raw_count > 0:
                raw_classes.add(
                    class_id
                )

            if processed_count > 0:
                processed_classes.add(
                    class_id
                )

            class_observed_retention = (
                processed_count / raw_count
                if raw_count > 0
                else np.nan
            )

            expected_class_count = (
                raw_count
                * grid_volume_scale
                if raw_count > 0
                and np.isfinite(
                    grid_volume_scale
                )
                else np.nan
            )

            class_normalized_retention = (
                processed_count
                / expected_class_count
                if expected_class_count > 0
                else np.nan
            )

            result[
                f"class_{class_id}_raw_voxels"
            ] = raw_count

            result[
                f"class_{class_id}_processed_voxels"
            ] = processed_count

            result[
                f"class_{class_id}_expected_processed_voxels"
            ] = expected_class_count

            result[
                f"class_{class_id}_observed_retention"
            ] = class_observed_retention

            result[
                f"class_{class_id}_grid_normalized_retention"
            ] = class_normalized_retention

            result[
                f"class_{class_id}_destroyed"
            ] = bool(
                raw_count > 0
                and processed_count == 0
            )

            result[
                f"class_{class_id}_bbox_raw"
            ] = raw[
                f"class_{class_id}_bbox_size"
            ]

            result[
                f"class_{class_id}_bbox_processed"
            ] = processed[
                f"class_{class_id}_bbox_size"
            ]

        removed_classes = (
            raw_classes
            - processed_classes
        )

        result[
            "raw_foreground_classes"
        ] = ",".join(
            str(x)
            for x in sorted(raw_classes)
        )

        result[
            "processed_foreground_classes"
        ] = ",".join(
            str(x)
            for x in sorted(processed_classes)
        )

        result[
            "classes_removed"
        ] = ",".join(
            str(x)
            for x in sorted(removed_classes)
        )

        result[
            "any_class_removed"
        ] = bool(
            removed_classes
        )

        # -------------------------------------------------------------
        # Loader information
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
# CLASS SUMMARY
# =============================================================================

def build_class_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    ok = df[
        df["status"] == "OK"
    ]

    rows = []

    for class_id in FOREGROUND_CLASSES:

        raw = pd.to_numeric(
            ok[
                f"class_{class_id}_raw_voxels"
            ],
            errors="coerce",
        )

        processed = pd.to_numeric(
            ok[
                f"class_{class_id}_processed_voxels"
            ],
            errors="coerce",
        )

        observed = pd.to_numeric(
            ok[
                f"class_{class_id}_observed_retention"
            ],
            errors="coerce",
        )

        normalized = pd.to_numeric(
            ok[
                f"class_{class_id}_grid_normalized_retention"
            ],
            errors="coerce",
        )

        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[
                    class_id
                ],
                "cases_with_raw_class": int(
                    (raw > 0).sum()
                ),
                "raw_total_voxels": int(
                    raw.sum()
                ),
                "processed_total_voxels": int(
                    processed.sum()
                ),
                "mean_observed_retention": float(
                    observed.mean()
                ),
                "median_observed_retention": float(
                    observed.median()
                ),
                "mean_grid_normalized_retention": float(
                    normalized.mean()
                ),
                "median_grid_normalized_retention": float(
                    normalized.median()
                ),
                "destroyed_cases": int(
                    (
                        (raw > 0)
                        & (processed == 0)
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FAILED CASES
# =============================================================================

def build_failed_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    failed = df[
        df["status"] != "OK"
    ]

    if failed.empty:
        return pd.DataFrame(
            columns=[
                "case_index",
                "study_id",
                "series_id",
                "status",
                "error",
            ]
        )

    return failed[
        [
            "case_index",
            "study_id",
            "series_id",
            "status",
            "error",
        ]
    ].copy()


# =============================================================================
# OVERALL SUMMARY
# =============================================================================

def build_summary(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    ok = df[
        df["status"] == "OK"
    ]

    failed = df[
        df["status"] != "OK"
    ]

    summary: Dict[str, Any] = {
        "phase": "Phase 4 - Part 34",
        "description": (
            "RSNA-only spatial resampling and "
            "class-aware foreground conservation audit"
        ),
        "audited_cases": int(
            len(df)
        ),
        "successful_cases": int(
            len(ok)
        ),
        "failed_cases": int(
            len(failed)
        ),
        "patch_size": list(
            PATCH_SIZE
        ),
        "num_classes": NUM_CLASSES,
        "training_performed": False,
        "model_created": False,
        "model_weights_modified": False,
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "spider_used": False,
        "rsna_test_set_used": False,
    }

    if ok.empty:
        summary[
            "diagnosis"
        ] = "NO_SUCCESSFUL_CASES"

        summary[
            "explanation"
        ] = (
            "No successful cases reached the "
            "raw-to-processed comparison."
        )

        return summary

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

    grid_scale = pd.to_numeric(
        ok[
            "grid_volume_scale"
        ],
        errors="coerce",
    )

    expected_fg = pd.to_numeric(
        ok[
            "expected_processed_foreground_voxels"
        ],
        errors="coerce",
    )

    observed = pd.to_numeric(
        ok[
            "observed_foreground_retention_ratio"
        ],
        errors="coerce",
    )

    normalized = pd.to_numeric(
        ok[
            "grid_normalized_foreground_retention"
        ],
        errors="coerce",
    )

    destroyed = ok[
        "foreground_destroyed"
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
            "mean_grid_volume_scale": float(
                grid_scale.mean()
            ),
            "median_grid_volume_scale": float(
                grid_scale.median()
            ),
            "mean_expected_processed_foreground_voxels": float(
                expected_fg.mean()
            ),
            "mean_observed_foreground_retention": float(
                observed.mean()
            ),
            "median_observed_foreground_retention": float(
                observed.median()
            ),
            "mean_grid_normalized_foreground_retention": float(
                normalized.mean()
            ),
            "median_grid_normalized_foreground_retention": float(
                normalized.median()
            ),
            "minimum_grid_normalized_foreground_retention": float(
                normalized.min()
            ),
            "maximum_grid_normalized_foreground_retention": float(
                normalized.max()
            ),
            "foreground_destroyed_cases": int(
                destroyed.sum()
            ),
            "class_removed_cases": int(
                ok[
                    "any_class_removed"
                ].astype(bool).sum()
            ),
            "invalid_processed_label_cases": int(
                ok[
                    "processed_has_invalid_labels"
                ].astype(bool).sum()
            ),
            "shape_changed_cases": int(
                ok[
                    "shape_changed"
                ].astype(bool).sum()
            ),
        }
    )

    # -----------------------------------------------------------------
    # Class summary in JSON
    # -----------------------------------------------------------------
    class_df = build_class_summary(
        df
    )

    for _, row in class_df.iterrows():
        c = int(
            row["class_id"]
        )

        summary[
            f"class_{c}_name"
        ] = row[
            "class_name"
        ]

        summary[
            f"class_{c}_mean_observed_retention"
        ] = float(
            row[
                "mean_observed_retention"
            ]
        )

        summary[
            f"class_{c}_mean_grid_normalized_retention"
        ] = float(
            row[
                "mean_grid_normalized_retention"
            ]
        )

        summary[
            f"class_{c}_destroyed_cases"
        ] = int(
            row[
                "destroyed_cases"
            ]
        )

    # -----------------------------------------------------------------
    # Diagnosis
    # -----------------------------------------------------------------
    normalized_mean = float(
        normalized.mean()
    )

    if destroyed.any():
        diagnosis = (
            "FOREGROUND_CAN_BE_DESTROYED_BY_RESAMPLING"
        )

        explanation = (
            "At least one successful case loses all foreground "
            "voxels during preprocessing. This requires case-level "
            "review before treating the preprocessing as safe for "
            "small structures."
        )

    elif normalized_mean < 0.75:
        diagnosis = (
            "ADDITIONAL_FOREGROUND_REDUCTION_BEYOND_GRID_SCALE"
        )

        explanation = (
            "After accounting for the change in total voxel-grid "
            "volume, the processed foreground remains substantially "
            "below the simple volume-scaling expectation. This is "
            "consistent with discrete nearest-neighbor resampling "
            "and the geometry of relatively small structures. "
            "The effect should be evaluated per anatomical class."
        )

    elif normalized_mean <= 1.25:
        diagnosis = (
            "FOREGROUND_REDUCTION_LARGELY_EXPLAINED_BY_GRID_SCALE"
        )

        explanation = (
            "The processed foreground is broadly consistent with "
            "the change in voxel-grid volume. Residual variation "
            "is expected because the transformation is discrete."
        )

    else:
        diagnosis = (
            "FOREGROUND_INCREASE_REQUIRES_REVIEW"
        )

        explanation = (
            "The processed foreground is substantially greater than "
            "the simple grid-volume expectation. Review spatial "
            "mapping and label preservation before training."
        )

    summary[
        "diagnosis"
    ] = diagnosis

    summary[
        "explanation"
    ] = explanation

    return summary


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    summary: Dict[str, Any],
    df: pd.DataFrame,
    class_df: pd.DataFrame,
    failed_df: pd.DataFrame,
) -> None:

    with REPORT_TXT.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 4 - PART 34\n"
        )

        f.write(
            "RSNA-ONLY SPATIAL RESAMPLING / "
            "CLASS-AWARE FOREGROUND CONSERVATION AUDIT\n\n"
        )

        f.write(
            "CONTROLS\n"
        )
        f.write(
            "--------\n"
        )

        f.write(
            "Training performed: NO\n"
        )
        f.write(
            "Model created: NO\n"
        )
        f.write(
            "Model weights modified: NO\n"
        )
        f.write(
            "Optimizer created: NO\n"
        )
        f.write(
            "Optimizer step performed: NO\n"
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
            str(
                summary.get(
                    "diagnosis",
                    "",
                )
            )
            + "\n\n"
        )

        f.write(
            "EXPLANATION\n"
        )
        f.write(
            str(
                summary.get(
                    "explanation",
                    "",
                )
            )
            + "\n\n"
        )

        f.write(
            "OVERALL SUMMARY\n"
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
            "\nCLASS SUMMARY\n"
        )
        f.write(
            "-" * 78
            + "\n"
        )

        if class_df.empty:
            f.write(
                "No class results.\n"
            )
        else:
            f.write(
                class_df.to_string(
                    index=False
                )
            )

        f.write(
            "\n\nFAILED CASES\n"
        )
        f.write(
            "-" * 78
            + "\n"
        )

        if failed_df.empty:
            f.write(
                "No failed cases.\n"
            )
        else:
            f.write(
                failed_df.to_string(
                    index=False
                )
            )

        f.write(
            "\n\nCASE-LEVEL RESULTS\n"
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
            "raw_mask_shape",
            "processed_mask_shape",
            "raw_grid_voxels",
            "processed_grid_voxels",
            "grid_volume_scale",
            "expected_processed_foreground_voxels",
            "raw_foreground_voxels",
            "processed_foreground_voxels",
            "observed_foreground_retention_ratio",
            "grid_normalized_foreground_retention",
            "foreground_minus_grid_expected",
            "foreground_destroyed",
            "classes_removed",
            "any_class_removed",
            "processed_has_invalid_labels",
            "error",
        ]

        columns = [
            c
            for c in preferred
            if c in df.columns
        ]

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

    header("PHASE 4 - PART 34")

    print(
        "RSNA-ONLY SPATIAL RESAMPLING / "
        "CLASS-AWARE FOREGROUND CONSERVATION AUDIT"
    )

    print()
    print("Evaluation / audit only.")
    print("No training is performed.")
    print("No model is created.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("No optimizer step is performed.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")

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
        "PATCH SIZE",
        PATCH_SIZE,
    )

    line(
        "OUTPUT DIRECTORY",
        OUTPUT_DIR,
    )

    # -----------------------------------------------------------------
    # Paths
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

        line(
            name,
            "FOUND" if exists else "MISSING",
        )

        if not exists:
            missing.append(
                f"{name}: {path}"
            )

    if missing:
        raise FileNotFoundError(
            "Missing required Part 34 input(s):\n"
            + "\n".join(missing)
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Environment
    # -----------------------------------------------------------------
    header("PYTORCH / GPU ENVIRONMENT")

    line(
        "PyTorch version",
        torch.__version__,
    )

    line(
        "CUDA available",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        line(
            "GPU",
            torch.cuda.get_device_name(0),
        )

    # -----------------------------------------------------------------
    # Modules
    # -----------------------------------------------------------------
    part11 = import_part11()
    part9 = import_part9(
        part11
    )

    # -----------------------------------------------------------------
    # Manifest
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
    # Audit
    # -----------------------------------------------------------------
    header(
        "RUNNING SPATIAL RESAMPLING AUDIT"
    )

    print(
        "Part 9 load_case() is used directly."
    )

    print(
        "Part 11 load_tensor_case() is NOT used."
    )

    print(
        "preprocess_case() is therefore called exactly once "
        "per successful case."
    )

    results: List[
        Dict[str, Any]
    ] = []

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
                f"{result['raw_mask_shape']} -> "
                f"{result['processed_mask_shape']} | "
                f"raw_fg="
                f"{result['raw_foreground_voxels']} | "
                f"processed_fg="
                f"{result['processed_foreground_voxels']} | "
                f"grid="
                f"{result['grid_volume_scale']:.6f} | "
                f"normalized="
                f"{result['grid_normalized_foreground_retention']:.6f}"
            )

        else:

            print(
                f"[{index:>4}/{total}] "
                f"FAILED | "
                f"study={result['study_id']} "
                f"series={result['series_id']} | "
                f"{result['error']}"
            )

    df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------------------
    # Summaries
    # -----------------------------------------------------------------
    summary = build_summary(
        df
    )

    class_df = build_class_summary(
        df
    )

    failed_df = build_failed_summary(
        df
    )

    # -----------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------
    header("PART 34 OVERALL DIAGNOSIS")

    for key in [
        "audited_cases",
        "successful_cases",
        "failed_cases",
        "mean_raw_foreground_voxels",
        "mean_processed_foreground_voxels",
        "mean_grid_volume_scale",
        "mean_expected_processed_foreground_voxels",
        "mean_observed_foreground_retention",
        "median_observed_foreground_retention",
        "mean_grid_normalized_foreground_retention",
        "median_grid_normalized_foreground_retention",
        "minimum_grid_normalized_foreground_retention",
        "maximum_grid_normalized_foreground_retention",
        "foreground_destroyed_cases",
        "class_removed_cases",
        "invalid_processed_label_cases",
        "shape_changed_cases",
    ]:
        if key in summary:
            line(
                key,
                summary[key],
            )

    line(
        "DIAGNOSIS",
        summary["diagnosis"],
    )

    print()
    print(
        summary["explanation"]
    )

    # -----------------------------------------------------------------
    # Class summary
    # -----------------------------------------------------------------
    header("CLASS-AWARE RETENTION SUMMARY")

    if class_df.empty:
        print(
            "No successful class results."
        )
    else:
        print(
            class_df[
                [
                    "class_id",
                    "class_name",
                    "cases_with_raw_class",
                    "mean_observed_retention",
                    "mean_grid_normalized_retention",
                    "destroyed_cases",
                ]
            ].to_string(
                index=False
            )
        )

    # -----------------------------------------------------------------
    # Failed cases
    # -----------------------------------------------------------------
    header("FAILED CASE SUMMARY")

    if failed_df.empty:
        print(
            "No failed cases."
        )
    else:
        print(
            failed_df.to_string(
                index=False
            )
        )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    header("SAVING PART 34 OUTPUTS")

    df.to_csv(
        CASE_CSV,
        index=False,
    )

    failed_df.to_csv(
        FAILED_CSV,
        index=False,
    )

    class_df.to_csv(
        CLASS_CSV,
        index=False,
    )

    summary[
        "case_csv"
    ] = str(
        CASE_CSV
    )

    summary[
        "failed_csv"
    ] = str(
        FAILED_CSV
    )

    summary[
        "class_csv"
    ] = str(
        CLASS_CSV
    )

    summary[
        "report_txt"
    ] = str(
        REPORT_TXT
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
        class_df,
        failed_df,
    )

    line(
        "Case audit CSV",
        CASE_CSV,
    )

    line(
        "Failed cases CSV",
        FAILED_CSV,
    )

    line(
        "Class summary CSV",
        CLASS_CSV,
    )

    line(
        "Summary JSON",
        SUMMARY_JSON,
    )

    line(
        "Report TXT",
        REPORT_TXT,
    )

    header("PHASE 4 - PART 34 COMPLETE")

    print(
        "✓ Grid-volume scaling calculated."
    )

    print(
        "✓ Expected foreground count calculated."
    )

    print(
        "✓ Grid-normalized foreground retention calculated."
    )

    print(
        "✓ Class 1-5 retention calculated."
    )

    print(
        "✓ Failed cases recorded separately."
    )

    print(
        "✓ No training performed."
    )

    print(
        "✓ No model weights modified."
    )

    print(
        "✓ No optimizer step performed."
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
        print(
            "\nPART 34 INTERRUPTED"
        )
        raise
    except Exception:
        header("PART 34 ERROR")
        traceback.print_exc()
        raise
