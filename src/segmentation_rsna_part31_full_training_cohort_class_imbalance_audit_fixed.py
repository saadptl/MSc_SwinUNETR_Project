"""
PHASE 4 - PART 31
RSNA-ONLY FULL TRAINING-COHORT CLASS IMBALANCE AUDIT

Evaluation / audit only.
No training is performed.
No model weights are modified.
No optimizer is created.
No optimizer step is performed.
SPIDER is not used.
RSNA test set is not used.

Purpose:
Determine whether the complete Part 15 training cohort contains severe
foreground/class imbalance and verify that the real pseudo-masks loaded
through the validated Part 11 -> Part 9 pipeline retain foreground labels.

IMPORTANT FIX
-------------
The previous Part 31 failed in consistency_audit() with:

    TypeError: replace() argument 1 must be str, not float

The cause was calling pandas Series.replace(np.nan, 0) on an object/string
series. This version never uses that unsafe pattern. Numeric columns are
explicitly converted with pd.to_numeric(..., errors="coerce"), followed by
fillna(0). String/object columns are handled separately.

This file is intentionally self-contained and does not modify Part 11.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# =============================================================================
# PROJECT PATHS
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

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART15_CHECKPOINT = (
    PART15_DIR
    / "checkpoints"
    / "best_model.pth"
)

PART15_TRAIN_COHORT = (
    PART15_DIR
    / "part15_train_cohort.csv"
)

PART8_MANIFEST = (
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
    / "rsna_part31_full_training_cohort_class_imbalance_audit"
)

REPORT_DIR = OUTPUT_DIR / "reports"

CASE_CSV = OUTPUT_DIR / "part31_full_training_case_metrics.csv"
CLASS_CSV = OUTPUT_DIR / "part31_full_training_class_distribution.csv"
CONSISTENCY_CSV = OUTPUT_DIR / "part31_metadata_loaded_mask_consistency.csv"
SUMMARY_JSON = OUTPUT_DIR / "phase4_part31_full_training_cohort_summary.json"
REPORT_TXT = REPORT_DIR / "phase4_part31_full_training_cohort_class_imbalance_report.txt"


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_CLASSES = 6

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

FOREGROUND_CLASSES = list(range(1, NUM_CLASSES))

# Part 15 cohort is expected to contain 500 training cases.
EXPECTED_TRAIN_CASES = 500

# Do not use the RSNA test set.
# This is a full Part 15 training-cohort audit.
USE_CHECKPOINT_HASH = True


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def line(label: str, value: Any) -> None:
    print(f"{label:<38}: {value}")


# =============================================================================
# SAFE DATAFRAME HELPERS
# =============================================================================

def numeric_series(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """
    Safely convert a dataframe column to numeric.

    This is the central fix for the Part 31 crash.

    NEVER use:
        series.replace(np.nan, 0)

    on object/string columns.

    Instead:
        pd.to_numeric(..., errors='coerce').fillna(default)
    """
    if column not in df.columns:
        return pd.Series(
            np.full(len(df), default, dtype=np.float64),
            index=df.index,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return default
        return int(value)

    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        x = float(value)
        if not np.isfinite(x):
            return default
        return x
    except Exception:
        return default


def safe_string(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value)


def parse_present_classes(value: Any) -> List[int]:
    """
    Parse Part 15/Part 8 present_class_ids values such as:
        [1]
        [2, 3]
        1,2,3
        1
        nan

    This function is deliberately conservative.
    """
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    text = str(value).strip()

    if not text:
        return []

    text = (
        text.replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace("{", "")
        .replace("}", "")
    )

    result: List[int] = []

    for token in text.split(","):
        token = token.strip()

        if not token:
            continue

        try:
            number = int(float(token))
        except Exception:
            continue

        if 0 <= number < NUM_CLASSES:
            result.append(number)

    return sorted(set(result))


# =============================================================================
# PATH VALIDATION
# =============================================================================

def validate_paths() -> None:
    header("PATH VALIDATION")

    required = {
        "RSNA root": RSNA_ROOT,
        "Part 11 source": PART11_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 train cohort": PART15_TRAIN_COHORT,
        "Part 8 train manifest": PART8_MANIFEST,
    }

    for name, path in required.items():
        line(
            name,
            "FOUND" if path.exists() else "MISSING",
        )

    missing = [
        name
        for name, path in required.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required Part 31 input(s):\n"
            + "\n".join(missing)
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# PART 11 IMPORT
# =============================================================================

def import_part11():
    header("IMPORTING VALIDATED PART 11")

    spec = importlib.util.spec_from_file_location(
        "part11_corrected_part31",
        str(PART11_SOURCE),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to import Part 11: {PART11_SOURCE}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules["part11_corrected_part31"] = module

    spec.loader.exec_module(module)

    required = [
        "load_part9_module",
        "load_tensor_case",
        "preprocess_case",
        "create_model",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise AttributeError(
            "Part 11 missing required API: "
            + ", ".join(missing)
        )

    print("✓ Corrected Part 11 imported.")

    print(
        "load_tensor_case:",
        inspect.signature(module.load_tensor_case),
    )

    print(
        "preprocess_case:",
        inspect.signature(module.preprocess_case),
    )

    print(
        "create_model:",
        inspect.signature(module.create_model),
    )

    return module


# =============================================================================
# PART 9
# =============================================================================

def load_part9(part11):
    header("LOADING PART 9")

    if hasattr(part11, "load_part9_module"):
        part9 = part11.load_part9_module()
        print("✓ Part 9 loader imported through Part 11.")
        return part9

    fallback = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"

    if not fallback.exists():
        raise RuntimeError(
            "Could not obtain Part 9 through Part 11 or fallback source."
        )

    spec = importlib.util.spec_from_file_location(
        "part9_for_part31",
        str(fallback),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to import Part 9: {fallback}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules["part9_for_part31"] = module

    spec.loader.exec_module(module)

    print("✓ Part 9 loader imported from source.")

    return module


# =============================================================================
# ROBUST PART 11 LOAD CALL
# =============================================================================

def call_load_tensor_case(
    part11,
    row: pd.Series,
    part9: Any,
):
    """
    Call the validated Part 11 load_tensor_case API.

    Current validated API:
        load_tensor_case(row, part9)

    Signature inspection prevents accidental argument-count errors.
    """
    fn = part11.load_tensor_case

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if len(params) == 1:
        return fn(row)

    kwargs = {}

    for parameter in params:
        if parameter.name in {
            "row",
            "case",
            "case_row",
            "record",
        }:
            kwargs[parameter.name] = row

        elif parameter.name in {
            "part9",
            "part9_module",
            "loader",
            "module",
        }:
            kwargs[parameter.name] = part9

        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(
                "Unsupported required load_tensor_case argument: "
                + parameter.name
            )

    return fn(**kwargs)


# =============================================================================
# TENSOR / MASK NORMALIZATION
# =============================================================================

def mask_to_numpy(mask: Any) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        x = mask.detach().cpu()

        # Common model/data-loader forms.
        while x.ndim > 3 and x.shape[0] == 1:
            x = x.squeeze(0)

        if x.ndim != 3:
            raise ValueError(
                f"Unexpected tensor mask shape: {tuple(x.shape)}"
            )

        return x.numpy()

    x = np.asarray(mask)

    while x.ndim > 3 and x.shape[0] == 1:
        x = np.squeeze(x, axis=0)

    if x.ndim != 3:
        raise ValueError(
            f"Unexpected numpy mask shape: {x.shape}"
        )

    return x


def image_to_numpy(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        x = image.detach().cpu()

        while x.ndim > 3 and x.shape[0] == 1:
            x = x.squeeze(0)

        if x.ndim != 3:
            raise ValueError(
                f"Unexpected image shape: {tuple(x.shape)}"
            )

        return x.numpy()

    x = np.asarray(image)

    while x.ndim > 3 and x.shape[0] == 1:
        x = np.squeeze(x, axis=0)

    if x.ndim != 3:
        raise ValueError(
            f"Unexpected image shape: {x.shape}"
        )

    return x


# =============================================================================
# SAFE PREPROCESSING CHECK
# =============================================================================

def call_preprocess_case(
    part11,
    image_np: np.ndarray,
    mask_np: np.ndarray,
):
    """
    Use Part 11's actual preprocessing API.

    Current validated signature:
        preprocess_case(image, mask)

    This function is only used to inspect label retention. It does not
    train or modify model parameters.
    """
    fn = part11.preprocess_case

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if len(params) == 2:
        return fn(
            image_np,
            mask_np,
        )

    kwargs = {}

    for parameter in params:
        if parameter.name in {
            "image",
            "image_array",
            "img",
            "image_np",
        }:
            kwargs[parameter.name] = image_np

        elif parameter.name in {
            "mask",
            "mask_array",
            "label",
            "target",
            "mask_np",
        }:
            kwargs[parameter.name] = mask_np

        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(
                "Unsupported required preprocess_case argument: "
                + parameter.name
            )

    return fn(**kwargs)


# =============================================================================
# CHECKPOINT HASH
# =============================================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


# =============================================================================
# MASK STATISTICS
# =============================================================================

def mask_statistics(mask: np.ndarray) -> Dict[str, Any]:
    mask = np.asarray(mask)

    finite = np.isfinite(mask)

    if not finite.all():
        finite_count = int(finite.sum())
        total = int(mask.size)
        raise ValueError(
            f"Mask contains non-finite values: "
            f"{total - finite_count}/{total}"
        )

    # Convert to integer labels.
    rounded = np.rint(mask)

    if not np.array_equal(
        mask.astype(np.float64),
        rounded.astype(np.float64),
    ):
        raise ValueError(
            "Mask contains non-integer label values."
        )

    mask_i = rounded.astype(np.int64)

    unique_values = np.unique(mask_i)

    invalid = [
        int(v)
        for v in unique_values
        if int(v) < 0 or int(v) >= NUM_CLASSES
    ]

    if invalid:
        raise ValueError(
            "Mask contains invalid class IDs: "
            + str(invalid)
        )

    class_voxels = {}

    for class_id in range(NUM_CLASSES):
        class_voxels[class_id] = int(
            np.count_nonzero(mask_i == class_id)
        )

    foreground = int(
        sum(
            class_voxels[c]
            for c in FOREGROUND_CLASSES
        )
    )

    total = int(mask_i.size)

    background = int(
        class_voxels[0]
    )

    foreground_fraction = (
        foreground / total
        if total
        else 0.0
    )

    if foreground:
        present_foreground = [
            c
            for c in FOREGROUND_CLASSES
            if class_voxels[c] > 0
        ]
    else:
        present_foreground = []

    return {
        "shape": str(tuple(int(x) for x in mask_i.shape)),
        "total_voxels": total,
        "foreground_voxels": foreground,
        "background_voxels": background,
        "foreground_fraction": foreground_fraction,
        "background_fraction": (
            background / total
            if total
            else 0.0
        ),
        "foreground_classes": present_foreground,
        "num_foreground_classes": len(
            present_foreground
        ),
        "class_voxels": class_voxels,
        "unique_labels": [
            int(x)
            for x in unique_values.tolist()
        ],
    }


# =============================================================================
# CASE AUDIT
# =============================================================================

def audit_case(
    index: int,
    row: pd.Series,
    part11,
    part9,
) -> Dict[str, Any]:
    study_id = safe_string(
        row.get("study_id", "")
    )

    series_id = safe_string(
        row.get("series_id", "")
    )

    result: Dict[str, Any] = {
        "case_index": index,
        "study_id": study_id,
        "series_id": series_id,
        "series_description": safe_string(
            row.get("series_description", "")
        ),
        "status": "FAILED",
        "error": "",
    }

    try:
        image, mask, info = call_load_tensor_case(
            part11,
            row,
            part9,
        )

        mask_np = mask_to_numpy(mask)
        image_np = image_to_numpy(image)

        loaded_stats = mask_statistics(
            mask_np
        )

        result.update(
            {
                "status": "SUCCESS",
                "image_shape": str(
                    tuple(
                        int(x)
                        for x in image_np.shape
                    )
                ),
                "loaded_mask_shape": loaded_stats[
                    "shape"
                ],
                "loaded_total_voxels": loaded_stats[
                    "total_voxels"
                ],
                "loaded_foreground_voxels": loaded_stats[
                    "foreground_voxels"
                ],
                "loaded_background_voxels": loaded_stats[
                    "background_voxels"
                ],
                "loaded_foreground_fraction": loaded_stats[
                    "foreground_fraction"
                ],
                "loaded_background_fraction": loaded_stats[
                    "background_fraction"
                ],
                "loaded_num_foreground_classes": loaded_stats[
                    "num_foreground_classes"
                ],
                "loaded_foreground_classes": ",".join(
                    map(
                        str,
                        loaded_stats[
                            "foreground_classes"
                        ],
                    )
                ),
                "loaded_unique_labels": ",".join(
                    map(
                        str,
                        loaded_stats[
                            "unique_labels"
                        ],
                    )
                ),
            }
        )

        for class_id in range(NUM_CLASSES):
            result[
                f"loaded_class_{class_id}_voxels"
            ] = loaded_stats[
                "class_voxels"
            ][class_id]

        # ------------------------------------------------------------------
        # Metadata from Part 15 cohort
        # ------------------------------------------------------------------

        metadata_fg = safe_int(
            row.get(
                "foreground_voxels",
                0,
            )
        )

        result[
            "metadata_foreground_voxels"
        ] = metadata_fg

        result[
            "metadata_present_classes"
        ] = ",".join(
            map(
                str,
                parse_present_classes(
                    row.get(
                        "present_class_ids",
                        "",
                    )
                ),
            )
        )

        metadata_labels = safe_string(
            row.get(
                "mask_labels",
                "",
            )
        )

        result[
            "metadata_mask_labels"
        ] = metadata_labels

        # Difference and retention are calculated only after both quantities
        # are known.
        result[
            "foreground_voxel_difference_loaded_minus_metadata"
        ] = (
            loaded_stats["foreground_voxels"]
            - metadata_fg
        )

        if metadata_fg > 0:
            result[
                "loaded_to_metadata_foreground_ratio"
            ] = (
                loaded_stats["foreground_voxels"]
                / metadata_fg
            )
        else:
            result[
                "loaded_to_metadata_foreground_ratio"
            ] = np.nan

        # ------------------------------------------------------------------
        # Preprocessing retention
        # ------------------------------------------------------------------

        preprocess_result = call_preprocess_case(
            part11,
            image_np,
            mask_np,
        )

        if isinstance(
            preprocess_result,
            tuple,
        ):
            processed_mask = (
                preprocess_result[1]
                if len(preprocess_result) > 1
                else mask_np
            )
        else:
            # Some historical APIs returned image only.
            # In that case, the original mask is retained for the audit.
            processed_mask = mask_np

        processed_mask_np = mask_to_numpy(
            processed_mask
        )

        processed_stats = mask_statistics(
            processed_mask_np
        )

        processed_fg = processed_stats[
            "foreground_voxels"
        ]

        result[
            "processed_mask_shape"
        ] = processed_stats[
            "shape"
        ]

        result[
            "processed_foreground_voxels"
        ] = processed_fg

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

        result[
            "processed_foreground_classes"
        ] = ",".join(
            map(
                str,
                processed_stats[
                    "foreground_classes"
                ],
            )
        )

        for class_id in range(NUM_CLASSES):
            result[
                f"processed_class_{class_id}_voxels"
            ] = processed_stats[
                "class_voxels"
            ][class_id]

        raw_fg = loaded_stats[
            "foreground_voxels"
        ]

        if raw_fg > 0:
            retention = (
                processed_fg / raw_fg
            )
        else:
            retention = np.nan

        result[
            "foreground_retention_ratio"
        ] = retention

        result[
            "foreground_destroyed_by_preprocessing"
        ] = bool(
            raw_fg > 0
            and processed_fg == 0
        )

        result[
            "processed_empty_foreground"
        ] = bool(
            processed_fg == 0
        )

        # Part 9 / Part 11 information, if available.
        if isinstance(info, dict):
            for key in [
                "raw_mask_shape",
                "alignment_mode",
                "aligned_slice_index",
                "volume_shape",
            ]:
                if key in info:
                    result[
                        f"loader_{key}"
                    ] = safe_string(
                        info[key]
                    )

        return result

    except Exception as exc:
        result["error"] = repr(exc)
        return result


# =============================================================================
# CONSISTENCY AUDIT
# =============================================================================

def consistency_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare Part 15 metadata with values obtained from the real loaded masks.

    FIXED IMPLEMENTATION:
    ---------------------
    The old implementation used something equivalent to:

        series.replace(np.nan, 0)

    That is unsafe for an object/string Series.

    We now explicitly coerce the relevant values to numeric first.
    """
    header(
        "METADATA VS LOADED MASK CONSISTENCY"
    )

    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "study_id": df[
                "study_id"
            ].astype(str),
            "series_id": df[
                "series_id"
            ].astype(str),
        }
    )

    # ----------------------------------------------------------------------
    # SAFE NUMERIC CONVERSION
    # ----------------------------------------------------------------------

    metadata_fg = numeric_series(
        df,
        "metadata_foreground_voxels",
        default=0,
    )

    loaded_fg = numeric_series(
        df,
        "loaded_foreground_voxels",
        default=0,
    )

    processed_fg = numeric_series(
        df,
        "processed_foreground_voxels",
        default=0,
    )

    # This is the direct replacement for the failing logic.
    #
    # DO NOT use:
    #     df["some_column"].replace(np.nan, 0)
    #
    # because the column may be object/string dtype.

    ratio = numeric_series(
        df,
        "loaded_to_metadata_foreground_ratio",
        default=np.nan,
    )

    retention = numeric_series(
        df,
        "foreground_retention_ratio",
        default=np.nan,
    )

    out[
        "metadata_foreground_voxels"
    ] = metadata_fg

    out[
        "loaded_foreground_voxels"
    ] = loaded_fg

    out[
        "processed_foreground_voxels"
    ] = processed_fg

    out[
        "loaded_minus_metadata"
    ] = (
        loaded_fg - metadata_fg
    )

    out[
        "loaded_to_metadata_ratio"
    ] = ratio

    out[
        "foreground_retention_ratio"
    ] = retention

    out[
        "metadata_loaded_exact_match"
    ] = (
        loaded_fg == metadata_fg
    )

    out[
        "loaded_foreground_present"
    ] = (
        loaded_fg > 0
    )

    out[
        "processed_foreground_present"
    ] = (
        processed_fg > 0
    )

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------

    exact_match = int(
        out[
            "metadata_loaded_exact_match"
        ].sum()
    )

    nonzero_loaded = int(
        out[
            "loaded_foreground_present"
        ].sum()
    )

    destroyed = int(
        (
            (loaded_fg > 0)
            & (processed_fg == 0)
        ).sum()
    )

    line(
        "Consistency rows",
        len(out),
    )

    line(
        "Exact metadata/loaded matches",
        exact_match,
    )

    line(
        "Nonzero loaded foreground cases",
        nonzero_loaded,
    )

    line(
        "Foreground destroyed by preprocessing",
        destroyed,
    )

    return out


# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

def build_class_distribution(
    successful: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for class_id in range(NUM_CLASSES):
        raw_col = (
            f"loaded_class_{class_id}_voxels"
        )

        processed_col = (
            f"processed_class_{class_id}_voxels"
        )

        if raw_col in successful.columns:
            raw = numeric_series(
                successful,
                raw_col,
                default=0,
            )
        else:
            raw = pd.Series(
                np.zeros(
                    len(successful),
                    dtype=np.float64,
                ),
                index=successful.index,
            )

        if processed_col in successful.columns:
            processed = numeric_series(
                successful,
                processed_col,
                default=0,
            )
        else:
            processed = pd.Series(
                np.zeros(
                    len(successful),
                    dtype=np.float64,
                ),
                index=successful.index,
            )

        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[
                    class_id
                ],
                "total_loaded_voxels": int(
                    raw.sum()
                ),
                "mean_loaded_voxels_per_case": float(
                    raw.mean()
                )
                if len(raw)
                else 0.0,
                "median_loaded_voxels_per_case": float(
                    raw.median()
                )
                if len(raw)
                else 0.0,
                "min_loaded_voxels_per_case": int(
                    raw.min()
                )
                if len(raw)
                else 0,
                "max_loaded_voxels_per_case": int(
                    raw.max()
                )
                if len(raw)
                else 0,
                "cases_with_class": int(
                    (raw > 0).sum()
                ),
                "case_frequency": (
                    float(
                        (raw > 0).mean()
                    )
                    if len(raw)
                    else 0.0
                ),
                "foreground_voxel_share": 0.0,
                "total_processed_voxels": int(
                    processed.sum()
                ),
                "mean_processed_voxels_per_case": float(
                    processed.mean()
                )
                if len(processed)
                else 0.0,
                "cases_with_processed_class": int(
                    (processed > 0).sum()
                ),
            }
        )

    class_df = pd.DataFrame(
        rows
    )

    foreground_total = float(
        class_df.loc[
            class_df["class_id"] > 0,
            "total_loaded_voxels",
        ].sum()
    )

    if foreground_total > 0:
        foreground_mask = (
            class_df["class_id"] > 0
        )

        class_df.loc[
            foreground_mask,
            "foreground_voxel_share",
        ] = (
            class_df.loc[
                foreground_mask,
                "total_loaded_voxels",
            ]
            / foreground_total
        )

    return class_df


# =============================================================================
# DIAGNOSIS
# =============================================================================

def determine_diagnosis(
    successful: pd.DataFrame,
) -> str:
    if successful.empty:
        return "NO_SUCCESSFUL_CASES"

    loaded_fg = numeric_series(
        successful,
        "loaded_foreground_voxels",
        default=0,
    )

    processed_fg = numeric_series(
        successful,
        "processed_foreground_voxels",
        default=0,
    )

    retention = numeric_series(
        successful,
        "foreground_retention_ratio",
        default=np.nan,
    )

    empty_loaded = int(
        (loaded_fg == 0).sum()
    )

    destroyed = int(
        (
            (loaded_fg > 0)
            & (processed_fg == 0)
        ).sum()
    )

    finite_retention = retention[
        np.isfinite(
            retention.to_numpy()
        )
    ]

    mean_retention = (
        float(
            finite_retention.mean()
        )
        if len(finite_retention)
        else float("nan")
    )

    if empty_loaded == len(successful):
        return (
            "ALL_LOADED_TRAINING_MASKS_HAVE_ZERO_FOREGROUND"
        )

    if empty_loaded > 0:
        return (
            "SOME_LOADED_TRAINING_MASKS_HAVE_ZERO_FOREGROUND"
        )

    if destroyed > 0:
        return (
            "PREPROCESSING_DESTROYS_FOREGROUND_LABELS"
        )

    if (
        np.isfinite(mean_retention)
        and mean_retention < 0.25
    ):
        return (
            "SEVERE_FOREGROUND_LABEL_LOSS_DURING_PREPROCESSING"
        )

    return (
        "TRAINING_FOREGROUND_LABELS_PRESENT; "
        "IMBALANCE_IS_REAL_BUT_NOT_EMPTY_LABEL_COLLAPSE"
    )


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    summary: Dict[str, Any],
    class_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
) -> None:
    with REPORT_TXT.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "PHASE 4 - PART 31\n"
        )

        f.write(
            "RSNA-ONLY FULL TRAINING-COHORT "
            "CLASS IMBALANCE AUDIT\n"
        )

        f.write("\n")

        f.write(
            "Evaluation / audit only.\n"
        )

        f.write(
            "No training is performed.\n"
        )

        f.write(
            "No model weights are modified.\n"
        )

        f.write(
            "No optimizer is created.\n"
        )

        f.write(
            "SPIDER is not used.\n"
        )

        f.write(
            "RSNA test set is not used.\n"
        )

        f.write("\n")

        f.write(
            "PART 31 DIAGNOSIS\n"
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
            "FULL-COHORT SUMMARY\n"
        )

        for key, value in summary.items():
            f.write(
                f"{key}: {value}\n"
            )

        f.write("\n")

        f.write(
            "PER-CLASS DISTRIBUTION\n"
        )

        if not class_df.empty:
            f.write(
                class_df.to_string(
                    index=False
                )
            )

        f.write("\n\n")

        f.write(
            "METADATA / LOADED MASK CONSISTENCY\n"
        )

        if not consistency_df.empty:
            f.write(
                consistency_df.describe(
                    include="all"
                ).to_string()
            )

        f.write("\n\n")

        f.write(
            "CRASH FIX\n"
        )

        f.write(
            "The previous TypeError came from applying "
            "Series.replace(np.nan, 0) to an object/string "
            "series. Numeric values are now normalized using "
            "pd.to_numeric(errors='coerce').fillna(...). "
            "No unsafe object-series replacement is used.\n"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    try:
        header("PHASE 4 - PART 31")

        print(
            "RSNA-ONLY FULL TRAINING-COHORT "
            "CLASS IMBALANCE AUDIT"
        )

        print()
        print(
            "Evaluation / audit only."
        )
        print(
            "No training is performed."
        )
        print(
            "No model weights are modified."
        )
        print(
            "No optimizer is created."
        )
        print(
            "No optimizer step is performed."
        )
        print(
            "SPIDER is not used."
        )
        print(
            "RSNA test set is not used."
        )

        print()
        print("Purpose:")
        print(
            "Determine whether the complete Part 15 "
            "training cohort contains severe "
            "foreground/class imbalance."
        )

        print()
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
            "PART 15 TRAIN COHORT",
            PART15_TRAIN_COHORT,
        )
        line(
            "PART 8 TRAIN MANIFEST",
            PART8_MANIFEST,
        )
        line(
            "OUTPUT DIRECTORY",
            OUTPUT_DIR,
        )

        validate_paths()

        # ------------------------------------------------------------------
        # ENVIRONMENT
        # ------------------------------------------------------------------

        header("PYTORCH ENVIRONMENT")

        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        line(
            "PyTorch version",
            torch.__version__,
        )

        line(
            "CUDA available",
            torch.cuda.is_available(),
        )

        line(
            "Device",
            device,
        )

        if torch.cuda.is_available():
            line(
                "GPU",
                torch.cuda.get_device_name(0),
            )

            props = (
                torch.cuda.get_device_properties(0)
            )

            line(
                "GPU memory",
                f"{props.total_memory / (1024 ** 3):.2f} GB",
            )

        line(
            "Classes",
            NUM_CLASSES,
        )

        line(
            "Foreground classes",
            FOREGROUND_CLASSES,
        )

        # ------------------------------------------------------------------
        # PART 11
        # ------------------------------------------------------------------

        part11 = import_part11()

        # ------------------------------------------------------------------
        # PART 9
        # ------------------------------------------------------------------

        part9 = load_part9(
            part11
        )

        # ------------------------------------------------------------------
        # LOAD COHORT
        # ------------------------------------------------------------------

        header(
            "LOADING EXACT PART 15 TRAINING COHORT"
        )

        cohort = pd.read_csv(
            PART15_TRAIN_COHORT
        )

        line(
            "Training cohort rows",
            len(cohort),
        )

        print(
            "Columns:",
            list(cohort.columns),
        )

        if len(cohort) != EXPECTED_TRAIN_CASES:
            print(
                f"WARNING: expected "
                f"{EXPECTED_TRAIN_CASES} rows but found "
                f"{len(cohort)}."
            )

        # ------------------------------------------------------------------
        # METADATA AUDIT
        # ------------------------------------------------------------------

        header(
            "METADATA FOREGROUND AUDIT"
        )

        metadata_fg = numeric_series(
            cohort,
            "foreground_voxels",
            default=0,
        )

        line(
            "Metadata zero-foreground cases",
            int(
                (metadata_fg == 0).sum()
            ),
        )

        line(
            "Metadata nonzero-foreground cases",
            int(
                (metadata_fg > 0).sum()
            ),
        )

        if len(metadata_fg):
            line(
                "Metadata minimum foreground voxels",
                int(
                    metadata_fg.min()
                ),
            )

            line(
                "Metadata maximum foreground voxels",
                int(
                    metadata_fg.max()
                ),
            )

            line(
                "Metadata median foreground voxels",
                float(
                    metadata_fg.median()
                ),
            )

        # ------------------------------------------------------------------
        # FULL REAL MASK AUDIT
        # ------------------------------------------------------------------

        header(
            "STARTING COMPLETE REAL-MASK AUDIT"
        )

        print()
        print(
            "Loading real masks through "
            "Part 11 -> Part 9..."
        )

        results: List[Dict[str, Any]] = []

        total = len(cohort)

        for position, (_, row) in enumerate(
            cohort.iterrows(),
            start=1,
        ):
            study = safe_string(
                row.get(
                    "study_id",
                    "",
                )
            )

            series = safe_string(
                row.get(
                    "series_id",
                    "",
                )
            )

            result = audit_case(
                position,
                row,
                part11,
                part9,
            )

            results.append(
                result
            )

            if result["status"] == "SUCCESS":
                fg = safe_int(
                    result.get(
                        "loaded_foreground_voxels",
                        0,
                    )
                )

                fraction = safe_float(
                    result.get(
                        "loaded_foreground_fraction",
                        0.0,
                    )
                )

                classes = safe_string(
                    result.get(
                        "loaded_foreground_classes",
                        "",
                    )
                )

                print(
                    f"[{position:03d}/{total}] "
                    f"study={study} "
                    f"series={series} | "
                    f"fg={fg} | "
                    f"fg_frac={fraction:.8f} | "
                    f"classes=[{classes}]"
                )

            else:
                print(
                    f"[{position:03d}/{total}] "
                    f"study={study} "
                    f"series={series} | "
                    f"FAILED | "
                    f"{result.get('error', '')}"
                )

        results_df = pd.DataFrame(
            results
        )

        successful = results_df[
            results_df["status"] == "SUCCESS"
        ].copy()

        failed = results_df[
            results_df["status"] != "SUCCESS"
        ].copy()

        # ------------------------------------------------------------------
        # FULL COHORT SUMMARY
        # ------------------------------------------------------------------

        header(
            "COMPUTING FULL-COHORT SUMMARY"
        )

        loaded_fg = numeric_series(
            successful,
            "loaded_foreground_voxels",
            default=0,
        )

        loaded_fraction = numeric_series(
            successful,
            "loaded_foreground_fraction",
            default=0,
        )

        loaded_total = numeric_series(
            successful,
            "loaded_total_voxels",
            default=0,
        )

        loaded_background = numeric_series(
            successful,
            "loaded_background_voxels",
            default=0,
        )

        if len(successful):
            total_fg = int(
                loaded_fg.sum()
            )

            total_bg = int(
                loaded_background.sum()
            )

            total_voxels = int(
                loaded_total.sum()
            )

            if total_fg:
                bg_fg_ratio = (
                    total_bg / total_fg
                )
            else:
                bg_fg_ratio = float(
                    "inf"
                )

            summary = {
                "phase": "Phase 4 - Part 31",
                "cohort_rows": int(
                    len(cohort)
                ),
                "successfully_loaded_cases": int(
                    len(successful)
                ),
                "failed_cases": int(
                    len(failed)
                ),
                "foreground_voxels_total": total_fg,
                "foreground_voxels_min": int(
                    loaded_fg.min()
                ),
                "foreground_voxels_max": int(
                    loaded_fg.max()
                ),
                "foreground_voxels_mean": float(
                    loaded_fg.mean()
                ),
                "foreground_voxels_median": float(
                    loaded_fg.median()
                ),
                "foreground_fraction_min": float(
                    loaded_fraction.min()
                ),
                "foreground_fraction_max": float(
                    loaded_fraction.max()
                ),
                "foreground_fraction_mean": float(
                    loaded_fraction.mean()
                ),
                "foreground_fraction_median": float(
                    loaded_fraction.median()
                ),
                "background_voxels_total": total_bg,
                "total_voxels": total_voxels,
                "overall_background_foreground_ratio": bg_fg_ratio,
                "zero_foreground_cases": int(
                    (loaded_fg == 0).sum()
                ),
                "nonzero_foreground_cases": int(
                    (loaded_fg > 0).sum()
                ),
            }

        else:
            summary = {
                "phase": "Phase 4 - Part 31",
                "cohort_rows": int(
                    len(cohort)
                ),
                "successfully_loaded_cases": 0,
                "failed_cases": int(
                    len(failed)
                ),
                "foreground_voxels_total": 0,
                "foreground_voxels_min": None,
                "foreground_voxels_max": None,
                "foreground_voxels_mean": None,
                "foreground_voxels_median": None,
                "foreground_fraction_min": None,
                "foreground_fraction_max": None,
                "foreground_fraction_mean": None,
                "foreground_fraction_median": None,
                "background_voxels_total": 0,
                "total_voxels": 0,
                "overall_background_foreground_ratio": None,
                "zero_foreground_cases": 0,
                "nonzero_foreground_cases": 0,
            }

        for key, value in summary.items():
            if key not in {
                "phase",
            }:
                line(
                    key,
                    value,
                )

        # ------------------------------------------------------------------
        # CLASS DISTRIBUTION
        # ------------------------------------------------------------------

        header(
            "PER-CLASS FOREGROUND DISTRIBUTION"
        )

        class_df = build_class_distribution(
            successful
        )

        if not class_df.empty:
            print(
                class_df.to_string(
                    index=False
                )
            )

        # ------------------------------------------------------------------
        # CASE-LEVEL DISTRIBUTION
        # ------------------------------------------------------------------

        header(
            "CASE-LEVEL FOREGROUND DISTRIBUTION"
        )

        if len(successful):
            case_distribution = pd.DataFrame(
                {
                    "foreground_voxels": numeric_series(
                        successful,
                        "loaded_foreground_voxels",
                        0,
                    ),
                    "foreground_fraction": numeric_series(
                        successful,
                        "loaded_foreground_fraction",
                        0,
                    ),
                    "num_foreground_classes": numeric_series(
                        successful,
                        "loaded_num_foreground_classes",
                        0,
                    ),
                }
            )

            print(
                case_distribution.describe().to_string()
            )

        # ------------------------------------------------------------------
        # SPARSITY BINS
        # ------------------------------------------------------------------

        header(
            "FOREGROUND SPARSITY BINS"
        )

        if len(successful):
            frac = numeric_series(
                successful,
                "loaded_foreground_fraction",
                0,
            )

            bins = [
                (
                    "0 - <0.001%",
                    frac.eq(0),
                ),
                (
                    "0.001% - <0.01%",
                    frac.ge(0.00001)
                    & frac.lt(0.0001),
                ),
                (
                    "0.01% - <0.1%",
                    frac.ge(0.0001)
                    & frac.lt(0.001),
                ),
                (
                    "0.1% - <1%",
                    frac.ge(0.001)
                    & frac.lt(0.01),
                ),
                (
                    "1% - <10%",
                    frac.ge(0.01)
                    & frac.lt(0.10),
                ),
                (
                    ">=10%",
                    frac.ge(0.10),
                ),
            ]

            for label, mask in bins:
                print(
                    f"{label:<38}: "
                    f"{int(mask.sum())}"
                )

        # ------------------------------------------------------------------
        # CONSISTENCY
        # ------------------------------------------------------------------

        consistency_df = consistency_audit(
            successful
        )

        # ------------------------------------------------------------------
        # DIAGNOSIS
        # ------------------------------------------------------------------

        header(
            "PART 31 DIAGNOSIS"
        )

        diagnosis = determine_diagnosis(
            successful
        )

        summary[
            "diagnosis"
        ] = diagnosis

        print(
            f"Diagnosis: {diagnosis}"
        )

        # ------------------------------------------------------------------
        # CHECKPOINT HASH
        # ------------------------------------------------------------------

        if USE_CHECKPOINT_HASH:
            header(
                "PART 15 CHECKPOINT INTEGRITY"
            )

            checkpoint_hash = (
                sha256_file(
                    PART15_CHECKPOINT
                )
            )

            line(
                "Checkpoint SHA-256",
                checkpoint_hash,
            )

            summary[
                "part15_checkpoint_sha256"
            ] = checkpoint_hash

        # ------------------------------------------------------------------
        # SAVE
        # ------------------------------------------------------------------

        header(
            "SAVING PART 31 RESULTS"
        )

        results_df.to_csv(
            CASE_CSV,
            index=False,
        )

        class_df.to_csv(
            CLASS_CSV,
            index=False,
        )

        consistency_df.to_csv(
            CONSISTENCY_CSV,
            index=False,
        )

        summary.update(
            {
                "project_root": str(
                    PROJECT_ROOT
                ),
                "rsna_root": str(
                    RSNA_ROOT
                ),
                "part11_source": str(
                    PART11_SOURCE
                ),
                "part15_train_cohort": str(
                    PART15_TRAIN_COHORT
                ),
                "part8_train_manifest": str(
                    PART8_MANIFEST
                ),
                "case_metrics_csv": str(
                    CASE_CSV
                ),
                "class_distribution_csv": str(
                    CLASS_CSV
                ),
                "consistency_csv": str(
                    CONSISTENCY_CSV
                ),
                "report_txt": str(
                    REPORT_TXT
                ),
                "no_training": True,
                "weights_modified": False,
                "optimizer_created": False,
                "optimizer_step_performed": False,
                "spider_used": False,
                "rsna_test_set_used": False,
            }
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
            class_df,
            consistency_df,
        )

        line(
            "Case metrics",
            CASE_CSV,
        )

        line(
            "Class distribution",
            CLASS_CSV,
        )

        line(
            "Consistency audit",
            CONSISTENCY_CSV,
        )

        line(
            "Summary JSON",
            SUMMARY_JSON,
        )

        line(
            "Text report",
            REPORT_TXT,
        )

        header(
            "PART 31 COMPLETE"
        )

        print(
            "✓ Part 31 completed without training."
        )

        print(
            "✓ No checkpoint was modified."
        )

        print(
            "✓ No optimizer was created."
        )

        print(
            "✓ RSNA test set was not used."
        )

        print(
            "✓ Previous pandas replace() TypeError is fixed."
        )

    except Exception as exc:
        header(
            "PART 31 ERROR"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        raise


if __name__ == "__main__":
    main()
