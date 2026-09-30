"""
PHASE 4 - PART 32
RSNA-ONLY METADATA / RAW-MASK / LOADED-MASK CONSISTENCY AUDIT

Evaluation / audit only.
No training is performed.
No model weights are modified.
No optimizer is created.
No optimizer step is performed.
SPIDER is not used.
RSNA test set is not used.

Purpose
-------
Part 31 established:

    - 500/500 Part 15 training cases load successfully
    - 0 loaded cases have zero foreground
    - preprocessing destroys foreground in 0 cases
    - foreground is extremely sparse
    - metadata foreground_voxels and loaded foreground voxels have
      0 exact matches

Part 32 determines WHY those metadata and loaded-mask foreground
counts differ.

The audit compares, where available:

    1. Part 15 cohort metadata
    2. Part 8 training manifest metadata
    3. raw pseudo-mask information exposed by the manifest
    4. Part 11 loaded mask
    5. Part 11 preprocessed mask
    6. shape / label / foreground-count transformations

The script is deliberately defensive around CSV object/string columns.
It does NOT use:

    series.replace(np.nan, 0)

on arbitrary pandas object columns.

Instead, numeric values are normalized with:

    pd.to_numeric(..., errors="coerce").fillna(...)

No checkpoint or source implementation is modified.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART15_TRAIN_COHORT = (
    PART15_DIR
    / "part15_train_cohort.csv"
)

PART15_CHECKPOINT = (
    PART15_DIR
    / "checkpoints"
    / "best_model.pth"
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part32_metadata_raw_loaded_mask_consistency_audit"
)

REPORT_DIR = OUTPUT_DIR / "reports"

CASE_CSV = (
    OUTPUT_DIR
    / "part32_case_consistency.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "phase4_part32_consistency_summary.json"
)

REPORT_TXT = (
    REPORT_DIR
    / "phase4_part32_consistency_report.txt"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_CLASSES = 6
FOREGROUND_CLASSES = list(range(1, NUM_CLASSES))

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

EXPECTED_CASES = 500


# =============================================================================
# PRINT HELPERS
# =============================================================================

def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def line(label: str, value: Any) -> None:
    print(f"{label:<42}: {value}")


# =============================================================================
# SAFE VALUE HELPERS
# =============================================================================

def safe_string(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    if value is None:
        return default

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    if value is None:
        return default

    if isinstance(value, (int, np.integer)):
        return int(value)

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return int(value)

    except Exception:
        return default


def numeric_series(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """
    Safe numeric normalization.

    This avoids the Part 31 pandas error:

        TypeError: replace() argument 1 must be str, not float
    """
    if column not in df.columns:
        return pd.Series(
            np.full(
                len(df),
                default,
                dtype=np.float64,
            ),
            index=df.index,
        )

    return (
        pd.to_numeric(
            df[column],
            errors="coerce",
        )
        .fillna(default)
    )


# =============================================================================
# CSV / COLUMN DISCOVERY
# =============================================================================

def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    """
    Find a column using exact names first and normalized names second.
    """
    if df.empty:
        return None

    exact = {
        str(c): str(c)
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]

    normalized = {
        re.sub(
            r"[^a-z0-9]",
            "",
            str(c).lower(),
        ): str(c)
        for c in df.columns
    }

    for candidate in candidates:
        key = re.sub(
            r"[^a-z0-9]",
            "",
            candidate.lower(),
        )

        if key in normalized:
            return normalized[key]

    return None


def describe_columns(
    name: str,
    df: pd.DataFrame,
) -> None:
    print()
    print(f"{name} columns:")
    for column in df.columns:
        print(f"  - {column}")


# =============================================================================
# PARSE LIST-LIKE METADATA
# =============================================================================

def parse_list_like(
    value: Any,
) -> List[Any]:
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, (list, tuple, set)):
        return list(value)

    text = str(value).strip()

    if not text:
        return []

    # Python-style list / tuple / JSON list.
    if text.startswith("[") or text.startswith("("):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)

        except Exception:
            pass

    text = (
        text
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )

    if "," in text:
        return [
            token.strip()
            for token in text.split(",")
            if token.strip()
        ]

    if ";" in text:
        return [
            token.strip()
            for token in text.split(";")
            if token.strip()
        ]

    return [text]


def parse_class_ids(
    value: Any,
) -> List[int]:
    result: List[int] = []

    for item in parse_list_like(value):
        try:
            class_id = int(float(str(item).strip()))

            if 0 <= class_id < NUM_CLASSES:
                result.append(class_id)

        except Exception:
            continue

    return sorted(set(result))


# =============================================================================
# PATH / STRING RESOLUTION
# =============================================================================

def resolve_possible_path(
    value: Any,
    base_dirs: List[Path],
) -> Optional[Path]:
    text = safe_string(value)

    if not text:
        return None

    # Remove accidental quotes.
    text = text.strip("\"'")

    candidate = Path(text)

    candidates: List[Path] = []

    if candidate.is_absolute():
        candidates.append(candidate)

    else:
        for base in base_dirs:
            candidates.append(
                base / candidate
            )

    for path in candidates:
        try:
            if path.exists():
                return path.resolve()
        except Exception:
            continue

    return None


# =============================================================================
# MASK STATISTICS
# =============================================================================

def normalize_mask_array(
    mask: Any,
) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)

    while array.ndim > 3:
        # Common [1, D, H, W] representation.
        if array.shape[0] == 1:
            array = np.squeeze(
                array,
                axis=0,
            )
            continue

        # Common [D, H, W, 1] representation.
        if array.shape[-1] == 1:
            array = np.squeeze(
                array,
                axis=-1,
            )
            continue

        break

    if array.ndim != 3:
        raise ValueError(
            f"Expected 3D mask, got shape {array.shape}"
        )

    return array


def mask_statistics(
    mask: Any,
) -> Dict[str, Any]:
    array = normalize_mask_array(
        mask
    )

    finite = np.isfinite(
        array
    )

    if not finite.all():
        raise ValueError(
            "Mask contains non-finite values."
        )

    rounded = np.rint(
        array
    )

    if not np.array_equal(
        array.astype(np.float64),
        rounded.astype(np.float64),
    ):
        raise ValueError(
            "Mask contains non-integer labels."
        )

    labels = rounded.astype(
        np.int64
    )

    unique = np.unique(
        labels
    )

    invalid = [
        int(x)
        for x in unique
        if int(x) < 0
        or int(x) >= NUM_CLASSES
    ]

    if invalid:
        raise ValueError(
            f"Invalid class IDs: {invalid}"
        )

    class_voxels = {
        class_id: int(
            np.count_nonzero(
                labels == class_id
            )
        )
        for class_id in range(NUM_CLASSES)
    }

    foreground = int(
        sum(
            class_voxels[c]
            for c in FOREGROUND_CLASSES
        )
    )

    total = int(
        labels.size
    )

    background = int(
        class_voxels[0]
    )

    return {
        "shape": tuple(
            int(x)
            for x in labels.shape
        ),
        "total_voxels": total,
        "foreground_voxels": foreground,
        "background_voxels": background,
        "foreground_fraction": (
            foreground / total
            if total
            else 0.0
        ),
        "unique_labels": [
            int(x)
            for x in unique.tolist()
        ],
        "class_voxels": class_voxels,
    }


# =============================================================================
# RAW MASK FILE INSPECTION
# =============================================================================

def load_mask_from_file(
    path: Path,
) -> Optional[np.ndarray]:
    """
    Best-effort reader for common numpy / torch pseudo-mask files.

    Supported:
        .npy
        .npz
        .pt
        .pth

    For NPZ files, all arrays are inspected and the first plausible
    3D integer-like array is selected.

    This is an audit helper only.
    """
    suffix = path.suffix.lower()

    try:
        if suffix == ".npy":
            return np.load(
                path,
                allow_pickle=False,
            )

        if suffix == ".npz":
            data = np.load(
                path,
                allow_pickle=False,
            )

            try:
                for key in data.files:
                    candidate = data[key]

                    try:
                        arr = normalize_mask_array(
                            candidate
                        )

                        if np.isfinite(arr).all():
                            rounded = np.rint(arr)

                            if np.array_equal(
                                arr,
                                rounded,
                            ):
                                return arr

                    except Exception:
                        continue

            finally:
                data.close()

            return None

        if suffix in {
            ".pt",
            ".pth",
        }:
            obj = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            if isinstance(
                obj,
                torch.Tensor,
            ):
                return obj.detach().cpu().numpy()

            if isinstance(
                obj,
                dict,
            ):
                # Prefer obvious mask/label keys.
                preferred = [
                    "mask",
                    "masks",
                    "label",
                    "labels",
                    "target",
                    "targets",
                    "pseudo_mask",
                    "pseudo_masks",
                ]

                for key in preferred:
                    if key in obj:
                        value = obj[key]

                        if isinstance(
                            value,
                            torch.Tensor,
                        ):
                            return (
                                value
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        try:
                            return np.asarray(
                                value
                            )
                        except Exception:
                            pass

                # Fall back to dictionary values.
                for value in obj.values():
                    if isinstance(
                        value,
                        torch.Tensor,
                    ):
                        try:
                            arr = (
                                value
                                .detach()
                                .cpu()
                                .numpy()
                            )

                            normalize_mask_array(
                                arr
                            )

                            return arr

                        except Exception:
                            continue

            return None

    except Exception:
        return None

    return None


# =============================================================================
# DISCOVER RAW MASK COLUMN
# =============================================================================

def discover_mask_column(
    df: pd.DataFrame,
) -> Optional[str]:
    candidates = [
        "mask_path",
        "pseudo_mask_path",
        "pseudomask_path",
        "label_path",
        "target_path",
        "segmentation_path",
        "annotation_path",
        "mask_file",
        "pseudo_mask_file",
        "pseudomask_file",
        "label_file",
        "target_file",
        "mask",
        "pseudo_mask",
        "pseudomask",
    ]

    column = find_column(
        df,
        candidates,
    )

    if column:
        return column

    # Conservative heuristic: only columns whose name explicitly contains
    # mask/pseudo/label and path/file are considered.
    for column in df.columns:
        name = str(column).lower()

        if (
            ("mask" in name or "label" in name)
            and (
                "path" in name
                or "file" in name
            )
        ):
            return str(column)

    return None


# =============================================================================
# MERGE COHORT AND MANIFEST
# =============================================================================

def merge_metadata(
    cohort: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge using the strongest available identifiers.

    Preference:
        study_id + series_id

    Fall back to:
        study_id

    If neither pair exists, the cohort is returned with empty manifest
    fields rather than inventing a join.
    """
    left = cohort.copy()
    right = manifest.copy()

    study_left = find_column(
        left,
        ["study_id", "study", "studyId"],
    )

    series_left = find_column(
        left,
        ["series_id", "series", "seriesId"],
    )

    study_right = find_column(
        right,
        ["study_id", "study", "studyId"],
    )

    series_right = find_column(
        right,
        ["series_id", "series", "seriesId"],
    )

    if (
        study_left
        and series_left
        and study_right
        and series_right
    ):
        left["_join_study"] = (
            left[study_left]
            .astype(str)
            .str.strip()
        )

        left["_join_series"] = (
            left[series_left]
            .astype(str)
            .str.strip()
        )

        right["_join_study"] = (
            right[study_right]
            .astype(str)
            .str.strip()
        )

        right["_join_series"] = (
            right[series_right]
            .astype(str)
            .str.strip()
        )

        right = right.drop_duplicates(
            subset=[
                "_join_study",
                "_join_series",
            ],
            keep="first",
        )

        merged = left.merge(
            right,
            how="left",
            on=[
                "_join_study",
                "_join_series",
            ],
            suffixes=(
                "",
                "_manifest",
            ),
            indicator=True,
        )

        return merged

    if study_left and study_right:
        left["_join_study"] = (
            left[study_left]
            .astype(str)
            .str.strip()
        )

        right["_join_study"] = (
            right[study_right]
            .astype(str)
            .str.strip()
        )

        right = right.drop_duplicates(
            subset=[
                "_join_study",
            ],
            keep="first",
        )

        return left.merge(
            right,
            how="left",
            on="_join_study",
            suffixes=(
                "",
                "_manifest",
            ),
            indicator=True,
        )

    # No reliable join.
    result = left.copy()

    result["_merge"] = "no_join_keys"

    return result


# =============================================================================
# PART 11 IMPORT
# =============================================================================

def import_part11():
    header(
        "IMPORTING VALIDATED PART 11"
    )

    spec = (
        importlib.util.spec_from_file_location(
            "part11_corrected_part32",
            str(PART11_SOURCE),
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Could not import Part 11: "
            f"{PART11_SOURCE}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        "part11_corrected_part32"
    ] = module

    spec.loader.exec_module(
        module
    )

    required = [
        "load_part9_module",
        "load_tensor_case",
        "preprocess_case",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise AttributeError(
            "Part 11 missing API: "
            + ", ".join(missing)
        )

    print(
        "✓ Corrected Part 11 imported."
    )

    line(
        "load_tensor_case",
        inspect.signature(
            module.load_tensor_case
        ),
    )

    line(
        "preprocess_case",
        inspect.signature(
            module.preprocess_case
        ),
    )

    return module


# =============================================================================
# PART 9 IMPORT
# =============================================================================

def load_part9(
    part11,
):
    header(
        "LOADING PART 9 THROUGH PART 11"
    )

    if hasattr(
        part11,
        "load_part9_module",
    ):
        module = (
            part11.load_part9_module()
        )

        print(
            "✓ Part 9 loader imported through Part 11."
        )

        return module

    fallback = (
        SRC_DIR
        / "segmentation_rsna_part9_3d_dataset_loader.py"
    )

    if not fallback.exists():
        raise RuntimeError(
            "Part 9 could not be obtained."
        )

    spec = (
        importlib.util.spec_from_file_location(
            "part9_for_part32",
            str(fallback),
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Unable to import Part 9: {fallback}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        "part9_for_part32"
    ] = module

    spec.loader.exec_module(
        module
    )

    print(
        "✓ Part 9 loader imported."
    )

    return module


# =============================================================================
# PART 11 LOADER CALL
# =============================================================================

def call_load_tensor_case(
    part11,
    row: pd.Series,
    part9: Any,
):
    fn = part11.load_tensor_case

    signature = inspect.signature(
        fn
    )

    params = list(
        signature.parameters.values()
    )

    # Current validated Part 11 API:
    # load_tensor_case(row, part9)
    if len(params) == 2:
        return fn(
            row,
            part9,
        )

    kwargs: Dict[str, Any] = {}

    for parameter in params:
        name = parameter.name

        if name in {
            "row",
            "case_row",
            "case",
            "record",
        }:
            kwargs[name] = row

        elif name in {
            "part9",
            "part9_module",
            "loader",
            "module",
        }:
            kwargs[name] = part9

        elif (
            parameter.default
            is inspect.Parameter.empty
        ):
            raise TypeError(
                "Unsupported required "
                f"load_tensor_case parameter: {name}"
            )

    return fn(
        **kwargs
    )


# =============================================================================
# PART 11 PREPROCESS CALL
# =============================================================================

def call_preprocess_case(
    part11,
    image: np.ndarray,
    mask: np.ndarray,
):
    fn = part11.preprocess_case

    signature = inspect.signature(
        fn
    )

    params = list(
        signature.parameters.values()
    )

    # Current validated Part 11 API:
    # preprocess_case(image, mask)
    if len(params) == 2:
        return fn(
            image,
            mask,
        )

    kwargs: Dict[str, Any] = {}

    for parameter in params:
        name = parameter.name

        if name in {
            "image",
            "image_np",
            "image_array",
            "img",
        }:
            kwargs[name] = image

        elif name in {
            "mask",
            "mask_np",
            "mask_array",
            "label",
            "target",
        }:
            kwargs[name] = mask

        elif (
            parameter.default
            is inspect.Parameter.empty
        ):
            raise TypeError(
                "Unsupported required "
                f"preprocess_case parameter: {name}"
            )

    return fn(
        **kwargs
    )


# =============================================================================
# CHECKPOINT HASH
# =============================================================================

def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# =============================================================================
# CASE AUDIT
# =============================================================================

def audit_case(
    index: int,
    row: pd.Series,
    part11,
    part9,
    manifest_mask_column: Optional[str],
) -> Dict[str, Any]:

    study_id = safe_string(
        row.get(
            "study_id",
            row.get(
                "study_id_manifest",
                "",
            ),
        )
    )

    series_id = safe_string(
        row.get(
            "series_id",
            row.get(
                "series_id_manifest",
                "",
            ),
        )
    )

    result: Dict[str, Any] = {
        "case_index": index,
        "study_id": study_id,
        "series_id": series_id,
        "status": "FAILED",
        "error": "",
    }

    try:
        # ---------------------------------------------------------------
        # Part 15 metadata
        # ---------------------------------------------------------------

        metadata_fg = safe_int(
            row.get(
                "foreground_voxels",
                0,
            )
        )

        metadata_classes = parse_class_ids(
            row.get(
                "present_class_ids",
                "",
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
                metadata_classes,
            )
        )

        result[
            "metadata_mask_labels"
        ] = safe_string(
            row.get(
                "mask_labels",
                "",
            )
        )

        # ---------------------------------------------------------------
        # Manifest metadata
        # ---------------------------------------------------------------

        manifest_fg_column = find_column(
            pd.DataFrame(
                [row]
            ),
            [
                "foreground_voxels_manifest",
                "foreground_voxels",
                "mask_foreground_voxels",
                "pseudo_mask_foreground_voxels",
            ],
        )

        if manifest_fg_column:
            result[
                "manifest_foreground_voxels"
            ] = safe_int(
                row.get(
                    manifest_fg_column,
                    0,
                )
            )
        else:
            result[
                "manifest_foreground_voxels"
            ] = 0

        manifest_classes_column = find_column(
            pd.DataFrame(
                [row]
            ),
            [
                "present_class_ids_manifest",
                "present_class_ids",
                "class_ids",
                "foreground_classes",
            ],
        )

        if manifest_classes_column:
            result[
                "manifest_present_classes"
            ] = ",".join(
                map(
                    str,
                    parse_class_ids(
                        row.get(
                            manifest_classes_column,
                            "",
                        )
                    ),
                )
            )
        else:
            result[
                "manifest_present_classes"
            ] = ""

        # ---------------------------------------------------------------
        # Raw mask path discovery
        # ---------------------------------------------------------------

        raw_path_value = ""

        if manifest_mask_column:
            raw_path_value = safe_string(
                row.get(
                    manifest_mask_column,
                    "",
                )
            )

        raw_path = resolve_possible_path(
            raw_path_value,
            [
                PROJECT_ROOT,
                RSNA_ROOT,
                PART8_DIR,
                PART8_DIR / "manifests",
                PART8_DIR / "masks",
                PROJECT_ROOT / "outputs",
                PROJECT_ROOT / "outputs" / "segmentation",
            ],
        )

        result[
            "raw_mask_reference"
        ] = raw_path_value

        result[
            "raw_mask_file_found"
        ] = bool(
            raw_path
        )

        result[
            "raw_mask_path"
        ] = (
            str(raw_path)
            if raw_path
            else ""
        )

        # ---------------------------------------------------------------
        # Real Part 11 loading
        # ---------------------------------------------------------------

        image, loaded_mask, info = (
            call_load_tensor_case(
                part11,
                row,
                part9,
            )
        )

        loaded_stats = mask_statistics(
            loaded_mask
        )

        result[
            "loaded_mask_shape"
        ] = str(
            loaded_stats[
                "shape"
            ]
        )

        result[
            "loaded_foreground_voxels"
        ] = loaded_stats[
            "foreground_voxels"
        ]

        result[
            "loaded_background_voxels"
        ] = loaded_stats[
            "background_voxels"
        ]

        result[
            "loaded_foreground_fraction"
        ] = loaded_stats[
            "foreground_fraction"
        ]

        result[
            "loaded_unique_labels"
        ] = ",".join(
            map(
                str,
                loaded_stats[
                    "unique_labels"
                ],
            )
        )

        result[
            "loaded_present_classes"
        ] = ",".join(
            map(
                str,
                [
                    c
                    for c in FOREGROUND_CLASSES
                    if loaded_stats[
                        "class_voxels"
                    ][c] > 0
                ],
            )
        )

        for class_id in range(
            NUM_CLASSES
        ):
            result[
                f"loaded_class_{class_id}_voxels"
            ] = loaded_stats[
                "class_voxels"
            ][class_id]

        # ---------------------------------------------------------------
        # Raw mask file audit, if discoverable
        # ---------------------------------------------------------------

        if raw_path:
            raw_mask = load_mask_from_file(
                raw_path
            )

            if raw_mask is not None:
                raw_stats = mask_statistics(
                    raw_mask
                )

                result[
                    "raw_mask_read_success"
                ] = True

                result[
                    "raw_mask_shape"
                ] = str(
                    raw_stats[
                        "shape"
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

                result[
                    "raw_unique_labels"
                ] = ",".join(
                    map(
                        str,
                        raw_stats[
                            "unique_labels"
                        ],
                    )
                )

                result[
                    "raw_loaded_exact_match"
                ] = bool(
                    raw_stats[
                        "foreground_voxels"
                    ]
                    == loaded_stats[
                        "foreground_voxels"
                    ]
                )

                result[
                    "raw_metadata_exact_match"
                ] = bool(
                    raw_stats[
                        "foreground_voxels"
                    ]
                    == metadata_fg
                )

            else:
                result[
                    "raw_mask_read_success"
                ] = False

        else:
            result[
                "raw_mask_read_success"
            ] = False

        # ---------------------------------------------------------------
        # Preprocessing audit
        # ---------------------------------------------------------------

        if isinstance(
            image,
            torch.Tensor,
        ):
            image_np = (
                image
                .detach()
                .cpu()
                .numpy()
            )
        else:
            image_np = np.asarray(
                image
            )

        loaded_mask_np = normalize_mask_array(
            loaded_mask
        )

        # IMPORTANT: Part 11 load_tensor_case() returns tensors from the
        # validated Part 11 data path.  In this project those tensors may
        # already be shaped [C, D, H, W].  Calling preprocess_case() again
        # would make it [N, C, D, H, W] and Part 11's 3-D interpolate call
        # then fails with:
        #   input spatial dimensions [1, 64, 96, 96]
        #   output size (64, 96, 96)
        #
        # Therefore only invoke preprocess_case() when the loader actually
        # returned unprocessed 3-D arrays.  For the validated tensor path,
        # the loaded mask itself is the final/preprocessed mask and must not
        # be preprocessed a second time.
        if (
            np.asarray(image_np).ndim == 3
            and np.asarray(loaded_mask_np).ndim == 3
        ):
            processed = call_preprocess_case(
                part11,
                image_np,
                loaded_mask_np,
            )

            if isinstance(
                processed,
                tuple,
            ):
                if len(processed) >= 2:
                    processed_mask = processed[1]
                else:
                    processed_mask = loaded_mask_np
            else:
                processed_mask = loaded_mask_np

            result[
                "preprocess_call_mode"
            ] = "called_on_3d_loader_output"
        else:
            processed_mask = loaded_mask_np
            result[
                "preprocess_call_mode"
            ] = "skipped_duplicate_preprocessing_loaded_tensor"

        processed_stats = mask_statistics(
            processed_mask
        )

        result[
            "processed_mask_shape"
        ] = str(
            processed_stats[
                "shape"
            ]
        )

        result[
            "processed_foreground_voxels"
        ] = processed_stats[
            "foreground_voxels"
        ]

        result[
            "processed_foreground_fraction"
        ] = processed_stats[
            "foreground_fraction"
        ]

        result[
            "processed_unique_labels"
        ] = ",".join(
            map(
                str,
                processed_stats[
                    "unique_labels"
                ],
            )
        )

        result[
            "processed_present_classes"
        ] = ",".join(
            map(
                str,
                [
                    c
                    for c in FOREGROUND_CLASSES
                    if processed_stats[
                        "class_voxels"
                    ][c] > 0
                ],
            )
        )

        for class_id in range(
            NUM_CLASSES
        ):
            result[
                f"processed_class_{class_id}_voxels"
            ] = processed_stats[
                "class_voxels"
            ][class_id]

        # ---------------------------------------------------------------
        # Comparisons
        # ---------------------------------------------------------------

        loaded_fg = loaded_stats[
            "foreground_voxels"
        ]

        processed_fg = processed_stats[
            "foreground_voxels"
        ]

        result[
            "metadata_loaded_difference"
        ] = (
            loaded_fg
            - metadata_fg
        )

        result[
            "metadata_loaded_exact_match"
        ] = bool(
            loaded_fg
            == metadata_fg
        )

        result[
            "loaded_processed_difference"
        ] = (
            processed_fg
            - loaded_fg
        )

        result[
            "loaded_processed_exact_match"
        ] = bool(
            processed_fg
            == loaded_fg
        )

        result[
            "foreground_destroyed_by_preprocessing"
        ] = bool(
            loaded_fg > 0
            and processed_fg == 0
        )

        if loaded_fg > 0:
            result[
                "processed_to_loaded_ratio"
            ] = (
                processed_fg
                / loaded_fg
            )
        else:
            result[
                "processed_to_loaded_ratio"
            ] = np.nan

        # ---------------------------------------------------------------
        # Loader info
        # ---------------------------------------------------------------

        if isinstance(
            info,
            dict,
        ):
            for key, value in info.items():
                # Keep only scalar / small metadata.
                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                ):
                    result[
                        f"loader_{key}"
                    ] = safe_string(
                        value
                    )

        result[
            "status"
        ] = "SUCCESS"

        return result

    except Exception as exc:
        result[
            "error"
        ] = repr(exc)

        return result


# =============================================================================
# DIAGNOSIS
# =============================================================================

def diagnose(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    success = df[
        df["status"] == "SUCCESS"
    ].copy()

    if success.empty:
        return {
            "diagnosis": "NO_SUCCESSFUL_CASES",
            "successful_cases": 0,
        }

    metadata_fg = numeric_series(
        success,
        "metadata_foreground_voxels",
        0,
    )

    loaded_fg = numeric_series(
        success,
        "loaded_foreground_voxels",
        0,
    )

    processed_fg = numeric_series(
        success,
        "processed_foreground_voxels",
        0,
    )

    exact_metadata_loaded = int(
        (
            loaded_fg
            == metadata_fg
        ).sum()
    )

    exact_loaded_processed = int(
        (
            loaded_fg
            == processed_fg
        ).sum()
    )

    zero_metadata = int(
        (
            metadata_fg == 0
        ).sum()
    )

    zero_loaded = int(
        (
            loaded_fg == 0
        ).sum()
    )

    zero_processed = int(
        (
            processed_fg == 0
        ).sum()
    )

    result: Dict[str, Any] = {
        "successful_cases": int(
            len(success)
        ),
        "metadata_loaded_exact_matches": exact_metadata_loaded,
        "metadata_loaded_exact_match_rate": (
            exact_metadata_loaded
            / len(success)
        ),
        "loaded_processed_exact_matches": exact_loaded_processed,
        "loaded_processed_exact_match_rate": (
            exact_loaded_processed
            / len(success)
        ),
        "metadata_zero_foreground_cases": zero_metadata,
        "loaded_zero_foreground_cases": zero_loaded,
        "processed_zero_foreground_cases": zero_processed,
        "mean_metadata_foreground": float(
            metadata_fg.mean()
        ),
        "mean_loaded_foreground": float(
            loaded_fg.mean()
        ),
        "mean_processed_foreground": float(
            processed_fg.mean()
        ),
        "median_metadata_foreground": float(
            metadata_fg.median()
        ),
        "median_loaded_foreground": float(
            loaded_fg.median()
        ),
        "median_processed_foreground": float(
            processed_fg.median()
        ),
    }

    # Raw file statistics, when available.
    if "raw_foreground_voxels" in success.columns:
        raw_fg = numeric_series(
            success,
            "raw_foreground_voxels",
            0,
        )

        raw_read = success.get(
            "raw_mask_read_success",
            pd.Series(
                False,
                index=success.index,
            ),
        )

        raw_read = (
            raw_read
            .astype(bool)
        )

        readable = success[
            raw_read
        ]

        if len(readable):
            readable_raw_fg = numeric_series(
                readable,
                "raw_foreground_voxels",
                0,
            )

            readable_loaded_fg = numeric_series(
                readable,
                "loaded_foreground_voxels",
                0,
            )

            result[
                "raw_masks_readable_cases"
            ] = int(
                len(readable)
            )

            result[
                "raw_loaded_exact_matches"
            ] = int(
                (
                    readable_raw_fg
                    == readable_loaded_fg
                ).sum()
            )

            result[
                "raw_metadata_exact_matches"
            ] = int(
                (
                    readable_raw_fg
                    == numeric_series(
                        readable,
                        "metadata_foreground_voxels",
                        0,
                    )
                ).sum()
            )

        else:
            result[
                "raw_masks_readable_cases"
            ] = 0

    # ---------------------------------------------------------------
    # Primary diagnosis
    # ---------------------------------------------------------------

    if zero_loaded == len(success):
        diagnosis = (
            "LOADED_MASKS_HAVE_NO_FOREGROUND"
        )

    elif zero_processed == len(success):
        diagnosis = (
            "PREPROCESSING_DESTROYS_ALL_FOREGROUND"
        )

    elif exact_loaded_processed == len(success):
        if exact_metadata_loaded == len(success):
            diagnosis = (
                "METADATA_AND_LOADED_MASKS_ARE_EXACTLY_CONSISTENT"
            )
        else:
            diagnosis = (
                "LOADED_AND_PREPROCESSED_MASKS_MATCH; "
                "METADATA_COUNT_COMES_FROM_A_DIFFERENT_STAGE"
            )

    elif exact_loaded_processed < len(success):
        diagnosis = (
            "LOADED_MASK_CHANGES_DURING_PREPROCESSING"
        )

    else:
        diagnosis = (
            "UNRESOLVED_METADATA_LOADED_DISCREPANCY"
        )

    result[
        "diagnosis"
    ] = diagnosis

    # ---------------------------------------------------------------
    # More specific explanation
    # ---------------------------------------------------------------

    if (
        exact_loaded_processed == len(success)
        and exact_metadata_loaded == 0
    ):
        explanation = (
            "The loaded mask is preserved exactly by preprocessing, "
            "but the Part 15 foreground_voxels metadata does not equal "
            "the foreground voxel count in the actual Part 11 loaded "
            "mask. The discrepancy therefore occurs before or outside "
            "preprocess_case()."
        )

    elif (
        exact_metadata_loaded == len(success)
    ):
        explanation = (
            "Part 15 metadata and Part 11 loaded masks have identical "
            "foreground counts for every audited case."
        )

    elif (
        exact_loaded_processed == len(success)
    ):
        explanation = (
            "Part 11 preprocessing preserves the foreground voxel "
            "count exactly. Any remaining discrepancy is upstream of "
            "preprocessing."
        )

    else:
        explanation = (
            "The loaded mask changes during preprocessing for at least "
            "one case. The next investigation should inspect the "
            "resize/crop/interpolation behavior."
        )

    result[
        "explanation"
    ] = explanation

    return result


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    summary: Dict[str, Any],
    case_df: pd.DataFrame,
) -> None:
    with REPORT_TXT.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 4 - PART 32\n"
        )

        f.write(
            "RSNA-ONLY METADATA / RAW-MASK / "
            "LOADED-MASK CONSISTENCY AUDIT\n\n"
        )

        f.write(
            "No training was performed.\n"
        )

        f.write(
            "No model weights were modified.\n"
        )

        f.write(
            "No optimizer was created.\n"
        )

        f.write(
            "No optimizer step was performed.\n"
        )

        f.write(
            "SPIDER was not used.\n"
        )

        f.write(
            "RSNA test set was not used.\n\n"
        )

        f.write(
            "PART 32 DIAGNOSIS\n"
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
            "SUMMARY\n"
        )

        for key, value in summary.items():
            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            "\nCASE DATASET\n"
        )

        if not case_df.empty:
            preferred = [
                "case_index",
                "study_id",
                "series_id",
                "metadata_foreground_voxels",
                "manifest_foreground_voxels",
                "raw_foreground_voxels",
                "loaded_foreground_voxels",
                "processed_foreground_voxels",
                "metadata_loaded_difference",
                "loaded_processed_difference",
                "metadata_loaded_exact_match",
                "loaded_processed_exact_match",
                "foreground_destroyed_by_preprocessing",
                "raw_mask_file_found",
                "raw_mask_read_success",
                "status",
            ]

            columns = [
                c
                for c in preferred
                if c in case_df.columns
            ]

            f.write(
                case_df[
                    columns
                ].to_string(
                    index=False
                )
            )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    try:
        header(
            "PHASE 4 - PART 32"
        )

        print(
            "RSNA-ONLY METADATA / RAW-MASK / "
            "LOADED-MASK CONSISTENCY AUDIT"
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
        print(
            "Purpose:"
        )
        print(
            "Determine why Part 15 foreground metadata "
            "does not exactly match the actual Part 11 "
            "loaded-mask foreground count."
        )

        line(
            "PROJECT ROOT",
            PROJECT_ROOT,
        )

        line(
            "RSNA DATASET",
            RSNA_ROOT,
        )

        line(
            "PART 15 TRAIN COHORT",
            PART15_TRAIN_COHORT,
        )

        line(
            "PART 8 TRAIN MANIFEST",
            PART8_TRAIN_MANIFEST,
        )

        line(
            "PART 11 SOURCE",
            PART11_SOURCE,
        )

        line(
            "OUTPUT DIRECTORY",
            OUTPUT_DIR,
        )

        # -----------------------------------------------------------------
        # PATH VALIDATION
        # -----------------------------------------------------------------

        header(
            "PATH VALIDATION"
        )

        required = {
            "RSNA root": RSNA_ROOT,
            "Part 11 source": PART11_SOURCE,
            "Part 15 train cohort": PART15_TRAIN_COHORT,
            "Part 8 train manifest": PART8_TRAIN_MANIFEST,
            "Part 15 checkpoint": PART15_CHECKPOINT,
        }

        missing = []

        for name, path in required.items():
            exists = path.exists()

            line(
                name,
                "FOUND"
                if exists
                else "MISSING",
            )

            if not exists:
                missing.append(name)

        if missing:
            raise FileNotFoundError(
                "Missing required Part 32 input(s):\n"
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

        # -----------------------------------------------------------------
        # ENVIRONMENT
        # -----------------------------------------------------------------

        header(
            "PYTORCH / GPU ENVIRONMENT"
        )

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
                torch.cuda.get_device_properties(
                    0
                )
            )

            line(
                "GPU memory",
                f"{props.total_memory / (1024 ** 3):.2f} GB",
            )

        line(
            "Classes",
            NUM_CLASSES,
        )

        # -----------------------------------------------------------------
        # IMPORT
        # -----------------------------------------------------------------

        part11 = import_part11()

        part9 = load_part9(
            part11
        )

        # -----------------------------------------------------------------
        # LOAD CSVs
        # -----------------------------------------------------------------

        header(
            "LOADING PART 15 AND PART 8 METADATA"
        )

        cohort = pd.read_csv(
            PART15_TRAIN_COHORT
        )

        manifest = pd.read_csv(
            PART8_TRAIN_MANIFEST
        )

        line(
            "Part 15 cohort rows",
            len(cohort),
        )

        line(
            "Part 8 manifest rows",
            len(manifest),
        )

        describe_columns(
            "Part 15 cohort",
            cohort,
        )

        describe_columns(
            "Part 8 manifest",
            manifest,
        )

        # -----------------------------------------------------------------
        # MASK COLUMN
        # -----------------------------------------------------------------

        header(
            "DISCOVERING RAW MASK REFERENCE"
        )

        manifest_mask_column = (
            discover_mask_column(
                manifest
            )
        )

        line(
            "Manifest mask column",
            manifest_mask_column
            or "NOT FOUND",
        )

        if manifest_mask_column:
            examples = (
                manifest[
                    manifest_mask_column
                ]
                .dropna()
                .astype(str)
                .head(5)
                .tolist()
            )

            print(
                "Example references:"
            )

            for value in examples:
                print(
                    f"  {value}"
                )

        # -----------------------------------------------------------------
        # MERGE
        # -----------------------------------------------------------------

        header(
            "MERGING PART 15 COHORT WITH PART 8 MANIFEST"
        )

        merged = merge_metadata(
            cohort,
            manifest,
        )

        if "_merge" in merged.columns:
            merge_counts = (
                merged[
                    "_merge"
                ]
                .value_counts(
                    dropna=False
                )
                .to_dict()
            )

            line(
                "Merge status",
                merge_counts,
            )

        line(
            "Merged rows",
            len(merged),
        )

        # -----------------------------------------------------------------
        # FULL AUDIT
        # -----------------------------------------------------------------

        header(
            "RUNNING FULL 500-CASE CONSISTENCY AUDIT"
        )

        results: List[
            Dict[str, Any]
        ] = []

        total = len(
            merged
        )

        for position, (_, row) in enumerate(
            merged.iterrows(),
            start=1,
        ):

            result = audit_case(
                position,
                row,
                part11,
                part9,
                manifest_mask_column,
            )

            results.append(
                result
            )

            if (
                result["status"]
                == "SUCCESS"
            ):
                metadata_fg = safe_int(
                    result.get(
                        "metadata_foreground_voxels",
                        0,
                    )
                )

                loaded_fg = safe_int(
                    result.get(
                        "loaded_foreground_voxels",
                        0,
                    )
                )

                processed_fg = safe_int(
                    result.get(
                        "processed_foreground_voxels",
                        0,
                    )
                )

                exact = (
                    "YES"
                    if result.get(
                        "metadata_loaded_exact_match",
                        False,
                    )
                    else "NO"
                )

                print(
                    f"[{position:03d}/{total}] "
                    f"study={result['study_id']} "
                    f"series={result['series_id']} | "
                    f"metadata={metadata_fg} | "
                    f"loaded={loaded_fg} | "
                    f"processed={processed_fg} | "
                    f"metadata==loaded:{exact}"
                )

            else:
                print(
                    f"[{position:03d}/{total}] "
                    f"FAILED | "
                    f"{result.get('error', '')}"
                )

        case_df = pd.DataFrame(
            results
        )

        # -----------------------------------------------------------------
        # DIAGNOSIS
        # -----------------------------------------------------------------

        header(
            "PART 32 DIAGNOSIS"
        )

        summary = diagnose(
            case_df
        )

        for key, value in summary.items():
            line(
                key,
                value,
            )

        # -----------------------------------------------------------------
        # CHECKPOINT HASH
        # -----------------------------------------------------------------

        checkpoint_hash = sha256_file(
            PART15_CHECKPOINT
        )

        summary[
            "part15_checkpoint_sha256"
        ] = checkpoint_hash

        # -----------------------------------------------------------------
        # SAVE
        # -----------------------------------------------------------------

        header(
            "SAVING PART 32 RESULTS"
        )

        case_df.to_csv(
            CASE_CSV,
            index=False,
        )

        summary.update(
            {
                "phase": "Phase 4 - Part 32",
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
                    PART8_TRAIN_MANIFEST
                ),
                "case_csv": str(
                    CASE_CSV
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
            case_df,
        )

        line(
            "Case audit CSV",
            CASE_CSV,
        )

        line(
            "Summary JSON",
            SUMMARY_JSON,
        )

        line(
            "Text report",
            REPORT_TXT,
        )

        # -----------------------------------------------------------------
        # FINAL
        # -----------------------------------------------------------------

        header(
            "PART 32 COMPLETE"
        )

        print(
            "✓ Full metadata / loaded-mask audit completed."
        )

        print(
            "✓ No training was performed."
        )

        print(
            "✓ No model weights were modified."
        )

        print(
            "✓ No optimizer step was performed."
        )

        print(
            "✓ SPIDER was not used."
        )

        print(
            "✓ RSNA test set was not used."
        )

        print(
            "✓ Safe numeric conversion is used throughout."
        )

    except Exception as exc:
        header(
            "PART 32 ERROR"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        raise


if __name__ == "__main__":
    main()
