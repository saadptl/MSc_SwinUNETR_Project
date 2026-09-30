"""
==============================================================================
PHASE 4 - PART 31
RSNA-ONLY FULL TRAINING-COHORT CLASS IMBALANCE AUDIT
==============================================================================

Evaluation / audit only.

No training is performed.
No model weights are modified.
No optimizer is created.
No optimizer step is performed.
SPIDER is not used.
RSNA test set is not used.

Purpose
-------
Determine whether the Part 15 training pseudo-label distribution contains
severe foreground/class imbalance that could explain the observed training
and prediction behaviour.

Part 31 audits the COMPLETE Part 15 training cohort:

    - 500 training cases
    - foreground voxel counts
    - foreground fraction
    - background fraction
    - class 1..5 voxel counts
    - class presence by case
    - per-class case frequency
    - per-class voxel frequency
    - minimum / maximum / median foreground size
    - background-to-foreground ratios
    - class imbalance ratios
    - consistency between cohort metadata and actually loaded masks
    - zero-foreground cases
    - cases containing multiple foreground classes

The audit uses the validated Part 11 -> Part 9 loading path wherever possible.
"""

from __future__ import annotations

import json
import math
import statistics
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

SRC_DIR = PROJECT_ROOT / "src"

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

PART15_TRAIN_COHORT = PART15_DIR / "part15_train_cohort.csv"

PART8_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
)

PART8_MANIFEST_DIR = PART8_DIR / "manifests"

PART8_TRAIN_MANIFEST = (
    PART8_MANIFEST_DIR
    / "rsna_part8_train_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part31_full_training_cohort_class_imbalance_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 6
BACKGROUND_CLASS = 0
FOREGROUND_CLASSES = list(range(1, NUM_CLASSES))


# =============================================================================
# PRINTING
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def kv(key: str, value: Any) -> None:
    print(f"{key:<38}: {value}")


# =============================================================================
# PATH VALIDATION
# =============================================================================

def validate_paths() -> None:
    section("PATH VALIDATION")

    paths = {
        "RSNA root": RSNA_ROOT,
        "Part 11 source": PART11_SOURCE,
        "Part 15 train cohort": PART15_TRAIN_COHORT,
        "Part 8 train manifest": PART8_TRAIN_MANIFEST,
    }

    missing = []

    for name, path in paths.items():
        state = "FOUND" if path.exists() else "MISSING"
        kv(name, state)

        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required Part 31 input(s):\n"
            + "\n".join(missing)
        )


# =============================================================================
# IMPORT PART 11
# =============================================================================

def import_part11():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "part11_validated",
        str(PART11_SOURCE),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create Part 11 import specification.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# =============================================================================
# IMPORT PART 9
# =============================================================================

def import_part9():
    part9_path = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"

    if not part9_path.exists():
        raise FileNotFoundError(
            f"Part 9 source not found: {part9_path}"
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "part9_validated",
        str(part9_path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create Part 9 import specification.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# =============================================================================
# SAFE INTEGER PARSING
# =============================================================================

def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


# =============================================================================
# PARSE PRESENT CLASS IDS
# =============================================================================

def parse_present_class_ids(value: Any) -> List[int]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        result = []

        for x in value:
            try:
                result.append(int(x))
            except Exception:
                pass

        return sorted(set(result))

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "[]"}:
        return []

    text = (
        text.replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )

    tokens = (
        text.replace(",", " ")
        .replace(";", " ")
        .split()
    )

    result = []

    for token in tokens:
        try:
            result.append(int(float(token)))
        except Exception:
            pass

    return sorted(set(result))


# =============================================================================
# MASK STATISTICS
# =============================================================================

def mask_statistics(mask: np.ndarray) -> Dict[str, Any]:
    mask = np.asarray(mask)

    total_voxels = int(mask.size)

    values, counts = np.unique(mask, return_counts=True)

    distribution = {
        int(v): int(c)
        for v, c in zip(values, counts)
    }

    class_counts = {
        class_id: int(distribution.get(class_id, 0))
        for class_id in range(NUM_CLASSES)
    }

    foreground_voxels = sum(
        class_counts[c]
        for c in FOREGROUND_CLASSES
    )

    background_voxels = class_counts[BACKGROUND_CLASS]

    foreground_fraction = (
        foreground_voxels / total_voxels
        if total_voxels > 0
        else 0.0
    )

    background_fraction = (
        background_voxels / total_voxels
        if total_voxels > 0
        else 0.0
    )

    bg_fg_ratio = (
        background_voxels / foreground_voxels
        if foreground_voxels > 0
        else math.inf
    )

    present_classes = [
        c
        for c in FOREGROUND_CLASSES
        if class_counts[c] > 0
    ]

    return {
        "total_voxels": total_voxels,
        "background_voxels": background_voxels,
        "foreground_voxels": foreground_voxels,
        "foreground_fraction": foreground_fraction,
        "background_fraction": background_fraction,
        "background_foreground_ratio": bg_fg_ratio,
        "present_foreground_classes": present_classes,
        **{
            f"class_{c}_voxels": class_counts[c]
            for c in range(NUM_CLASSES)
        },
    }


# =============================================================================
# METADATA-ONLY AUDIT
# =============================================================================

def audit_cohort_metadata(
    train_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for index, row in train_df.iterrows():

        foreground_voxels = safe_int(
            row.get("foreground_voxels", 0)
        )

        mask_labels = parse_present_class_ids(
            row.get("mask_labels", "")
        )

        present_class_ids = parse_present_class_ids(
            row.get("present_class_ids", "")
        )

        mask_shape = str(
            row.get("mask_shape", "")
        )

        rows.append(
            {
                "row_index": int(index),
                "study_id": row.get("study_id"),
                "series_id": row.get("series_id"),
                "foreground_voxels_metadata": foreground_voxels,
                "mask_labels_metadata": str(mask_labels),
                "present_class_ids_metadata": str(
                    present_class_ids
                ),
                "mask_shape": mask_shape,
                "dataset_eligible": row.get(
                    "dataset_eligible"
                ),
                "pseudo_mask_valid": row.get(
                    "pseudo_mask_valid"
                ),
                "has_pseudomask": row.get(
                    "has_pseudomask"
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# REAL LOADED-MASK AUDIT
# =============================================================================

def audit_real_masks(
    train_df: pd.DataFrame,
    part11,
    part9,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:

    rows = []
    errors = []

    total = len(train_df)

    print()
    print("Loading real masks through Part 11 -> Part 9...")
    print()

    for i, (_, row) in enumerate(train_df.iterrows(), start=1):

        study_id = row.get("study_id")
        series_id = row.get("series_id")

        print(
            f"[{i:03d}/{total}] "
            f"study={study_id} series={series_id}",
            end="",
        )

        try:

            image, mask, info = part11.load_tensor_case(
                row,
                part9,
            )

            # The validated API returns tensors.
            mask_np = (
                mask.detach()
                .cpu()
                .numpy()
            )

            # Remove singleton dimensions safely.
            mask_np = np.squeeze(mask_np)

            stats = mask_statistics(mask_np)

            stats.update(
                {
                    "row_index": int(
                        train_df.index[i - 1]
                    ),
                    "study_id": study_id,
                    "series_id": series_id,
                    "load_success": True,
                    "image_shape": str(
                        tuple(image.shape)
                    ),
                    "mask_shape_loaded": str(
                        tuple(mask_np.shape)
                    ),
                }
            )

            rows.append(stats)

            print(
                f" | fg={stats['foreground_voxels']}"
                f" | fg_frac="
                f"{stats['foreground_fraction']:.8f}"
                f" | classes="
                f"{stats['present_foreground_classes']}"
            )

        except Exception as exc:

            error = {
                "row_index": int(
                    train_df.index[i - 1]
                ),
                "study_id": study_id,
                "series_id": series_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

            errors.append(error)

            print(
                f" | ERROR: {repr(exc)}"
            )

    return pd.DataFrame(rows), errors


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_summary(
    real_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> Dict[str, Any]:

    summary: Dict[str, Any] = {}

    summary["cohort_rows"] = int(
        len(metadata_df)
    )

    summary["successfully_loaded_cases"] = int(
        len(real_df)
    )

    summary["failed_cases"] = int(
        summary["cohort_rows"]
        - summary["successfully_loaded_cases"]
    )

    if real_df.empty:
        return summary

    fg = real_df[
        "foreground_voxels"
    ].astype(float)

    fg_fraction = real_df[
        "foreground_fraction"
    ].astype(float)

    bg = real_df[
        "background_voxels"
    ].astype(float)

    summary.update(
        {
            "foreground_voxels_total": int(
                fg.sum()
            ),
            "foreground_voxels_min": int(
                fg.min()
            ),
            "foreground_voxels_max": int(
                fg.max()
            ),
            "foreground_voxels_mean": float(
                fg.mean()
            ),
            "foreground_voxels_median": float(
                fg.median()
            ),
            "foreground_fraction_min": float(
                fg_fraction.min()
            ),
            "foreground_fraction_max": float(
                fg_fraction.max()
            ),
            "foreground_fraction_mean": float(
                fg_fraction.mean()
            ),
            "foreground_fraction_median": float(
                fg_fraction.median()
            ),
            "background_voxels_total": int(
                bg.sum()
            ),
            "total_voxels": int(
                real_df["total_voxels"].sum()
            ),
        }
    )

    total_fg = summary[
        "foreground_voxels_total"
    ]

    total_bg = summary[
        "background_voxels_total"
    ]

    summary["overall_background_foreground_ratio"] = (
        float(total_bg / total_fg)
        if total_fg > 0
        else math.inf
    )

    summary["zero_foreground_cases"] = int(
        (fg == 0).sum()
    )

    summary["nonzero_foreground_cases"] = int(
        (fg > 0).sum()
    )

    for class_id in FOREGROUND_CLASSES:

        col = f"class_{class_id}_voxels"

        total_class_voxels = int(
            real_df[col].sum()
        )

        cases_with_class = int(
            (real_df[col] > 0).sum()
        )

        summary[
            f"class_{class_id}_total_voxels"
        ] = total_class_voxels

        summary[
            f"class_{class_id}_cases"
        ] = cases_with_class

        summary[
            f"class_{class_id}_case_frequency"
        ] = (
            cases_with_class
            / len(real_df)
        )

    return summary


# =============================================================================
# CLASS TABLE
# =============================================================================

def build_class_table(
    real_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    total_foreground = int(
        real_df[
            [
                f"class_{c}_voxels"
                for c in FOREGROUND_CLASSES
            ]
        ].sum().sum()
    )

    for class_id in FOREGROUND_CLASSES:

        col = f"class_{class_id}_voxels"

        total_voxels = int(
            real_df[col].sum()
        )

        cases = int(
            (real_df[col] > 0).sum()
        )

        mean_per_case = float(
            real_df[col].mean()
        )

        median_per_case = float(
            real_df[col].median()
        )

        min_per_case = int(
            real_df[col].min()
        )

        max_per_case = int(
            real_df[col].max()
        )

        foreground_share = (
            total_voxels / total_foreground
            if total_foreground > 0
            else 0.0
        )

        rows.append(
            {
                "class_id": class_id,
                "total_voxels": total_voxels,
                "cases_with_class": cases,
                "case_frequency": (
                    cases / len(real_df)
                ),
                "mean_voxels_per_case": mean_per_case,
                "median_voxels_per_case": (
                    median_per_case
                ),
                "min_voxels_per_case": min_per_case,
                "max_voxels_per_case": max_per_case,
                "foreground_voxel_share": (
                    foreground_share
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CASE DISTRIBUTION TABLE
# =============================================================================

def build_case_distribution(
    real_df: pd.DataFrame,
) -> pd.DataFrame:

    df = real_df.copy()

    df["num_foreground_classes"] = df[
        "present_foreground_classes"
    ].apply(len)

    df["foreground_to_background_ratio"] = np.where(
        df["background_voxels"] > 0,
        df["foreground_voxels"]
        / df["background_voxels"],
        0.0,
    )

    return df[
        [
            "row_index",
            "study_id",
            "series_id",
            "total_voxels",
            "background_voxels",
            "foreground_voxels",
            "foreground_fraction",
            "background_foreground_ratio",
            "num_foreground_classes",
            "present_foreground_classes",
            *[
                f"class_{c}_voxels"
                for c in FOREGROUND_CLASSES
            ],
        ]
    ].copy()


# =============================================================================
# CONSISTENCY AUDIT
# =============================================================================

def consistency_audit(
    metadata_df: pd.DataFrame,
    real_df: pd.DataFrame,
) -> pd.DataFrame:

    merged = metadata_df.merge(
        real_df,
        on=[
            "row_index",
            "study_id",
            "series_id",
        ],
        how="left",
    )

    merged[
        "foreground_voxel_difference"
    ] = (
        merged[
            "foreground_voxels"
            .replace(np.nan, 0)
        ]
        -
        merged[
            "foreground_voxels_metadata"
        ]
    )

    merged[
        "metadata_foreground_matches_loaded"
    ] = (
        merged[
            "foreground_voxels"
        ].fillna(-1)
        ==
        merged[
            "foreground_voxels_metadata"
        ].fillna(-2)
    )

    return merged[
        [
            "row_index",
            "study_id",
            "series_id",
            "foreground_voxels_metadata",
            "foreground_voxels",
            "foreground_voxel_difference",
            "metadata_foreground_matches_loaded",
            "mask_labels_metadata",
            "present_foreground_classes",
        ]
    ]


# =============================================================================
# DIAGNOSTIC INTERPRETATION
# =============================================================================

def generate_interpretation(
    summary: Dict[str, Any],
    class_table: pd.DataFrame,
) -> Dict[str, Any]:

    interpretation: Dict[str, Any] = {}

    loaded = summary.get(
        "successfully_loaded_cases",
        0,
    )

    zero_fg = summary.get(
        "zero_foreground_cases",
        0,
    )

    median_fg_fraction = summary.get(
        "foreground_fraction_median",
        0.0,
    )

    mean_fg_fraction = summary.get(
        "foreground_fraction_mean",
        0.0,
    )

    overall_ratio = summary.get(
        "overall_background_foreground_ratio",
        math.inf,
    )

    interpretation[
        "foreground_loading_integrity"
    ] = (
        "PASS: all successfully loaded cases "
        "contain foreground."
        if loaded > 0 and zero_fg == 0
        else
        "WARNING: one or more successfully loaded "
        "cases contain zero foreground."
    )

    interpretation[
        "foreground_sparsity"
    ] = {
        "mean_fraction": mean_fg_fraction,
        "median_fraction": median_fg_fraction,
        "assessment": (
            "EXTREMELY SPARSE"
            if median_fg_fraction < 0.001
            else
            "SPARSE"
            if median_fg_fraction < 0.01
            else
            "MODERATE"
        ),
    }

    interpretation[
        "background_foreground_balance"
    ] = {
        "overall_background_foreground_ratio":
            overall_ratio,
        "assessment": (
            "SEVERE BACKGROUND DOMINANCE"
            if overall_ratio > 1000
            else
            "HIGH BACKGROUND DOMINANCE"
            if overall_ratio > 100
            else
            "MODERATE BACKGROUND DOMINANCE"
        ),
    }

    if not class_table.empty:

        rarest = class_table.sort_values(
            "total_voxels"
        ).iloc[0]

        most_common = class_table.sort_values(
            "total_voxels",
            ascending=False,
        ).iloc[0]

        interpretation[
            "rarest_foreground_class"
        ] = int(
            rarest["class_id"]
        )

        interpretation[
            "rarest_class_voxels"
        ] = int(
            rarest["total_voxels"]
        )

        interpretation[
            "most_common_foreground_class"
        ] = int(
            most_common["class_id"]
        )

        interpretation[
            "most_common_class_voxels"
        ] = int(
            most_common["total_voxels"]
        )

    interpretation[
        "overall_diagnostic_conclusion"
    ] = (
        "Part 30 established that real model patches contain foreground. "
        "Part 31 should therefore be interpreted primarily as a class "
        "imbalance / foreground sparsity audit. If severe imbalance is "
        "confirmed across the complete 500-case cohort, the result supports "
        "investigating loss weighting, foreground-aware sampling, and "
        "pseudo-label density before changing the model architecture."
    )

    return interpretation


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    section("PHASE 4 - PART 31")

    print(
        "RSNA-ONLY FULL TRAINING-COHORT "
        "CLASS IMBALANCE AUDIT"
    )

    print()
    print("Evaluation / audit only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("No optimizer step is performed.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")

    print()
    print("Purpose:")
    print(
        "Determine whether the complete Part 15 training cohort "
        "contains severe foreground/class imbalance."
    )

    kv("PROJECT ROOT", PROJECT_ROOT)
    kv("RSNA DATASET", RSNA_ROOT)
    kv("PART 11 SOURCE", PART11_SOURCE)
    kv("PART 15 TRAIN COHORT", PART15_TRAIN_COHORT)
    kv("PART 8 TRAIN MANIFEST", PART8_TRAIN_MANIFEST)
    kv("OUTPUT DIRECTORY", OUTPUT_DIR)

    validate_paths()

    section("PYTORCH ENVIRONMENT")

    kv(
        "PyTorch version",
        torch.__version__,
    )

    kv(
        "CUDA available",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")

        kv("Device", device)
        kv(
            "GPU",
            torch.cuda.get_device_name(0),
        )

        props = torch.cuda.get_device_properties(0)

        kv(
            "GPU memory",
            f"{props.total_memory / 1024**3:.2f} GB",
        )

    kv("Classes", NUM_CLASSES)
    kv("Foreground classes", FOREGROUND_CLASSES)

    section("IMPORTING VALIDATED PART 11")

    part11 = import_part11()

    print("✓ Corrected Part 11 imported.")

    print(
        "load_tensor_case:",
        getattr(
            part11.load_tensor_case,
            "__annotations__",
            {},
        ),
    )

    section("LOADING PART 9")

    part9 = import_part9()

    print("✓ Part 9 loader imported.")
    kv(
        "Part 9 source",
        SRC_DIR
        / "segmentation_rsna_part9_3d_dataset_loader.py",
    )

    section("LOADING EXACT PART 15 TRAINING COHORT")

    train_df = pd.read_csv(
        PART15_TRAIN_COHORT
    )

    kv(
        "Training cohort rows",
        len(train_df),
    )

    print(
        "Columns:",
        list(train_df.columns),
    )

    if len(train_df) != 500:
        print()
        print(
            "WARNING: Part 15 cohort does not contain "
            "exactly 500 rows."
        )

    section("METADATA FOREGROUND AUDIT")

    metadata_df = audit_cohort_metadata(
        train_df
    )

    metadata_fg = metadata_df[
        "foreground_voxels_metadata"
    ].astype(int)

    kv(
        "Metadata zero-foreground cases",
        int((metadata_fg == 0).sum()),
    )

    kv(
        "Metadata nonzero-foreground cases",
        int((metadata_fg > 0).sum()),
    )

    kv(
        "Metadata minimum foreground voxels",
        int(metadata_fg.min()),
    )

    kv(
        "Metadata maximum foreground voxels",
        int(metadata_fg.max()),
    )

    kv(
        "Metadata median foreground voxels",
        float(metadata_fg.median()),
    )

    section("STARTING COMPLETE REAL-MASK AUDIT")

    real_df, errors = audit_real_masks(
        train_df,
        part11,
        part9,
    )

    section("COMPUTING FULL-COHORT SUMMARY")

    summary = compute_summary(
        real_df,
        metadata_df,
    )

    for key, value in summary.items():
        kv(key, value)

    section("PER-CLASS FOREGROUND DISTRIBUTION")

    class_table = build_class_table(
        real_df
    )

    if not class_table.empty:

        display_table = class_table.copy()

        for col in [
            "case_frequency",
            "foreground_voxel_share",
        ]:
            display_table[col] = display_table[
                col
            ].map(
                lambda x: f"{x:.6f}"
            )

        print(
            display_table.to_string(
                index=False
            )
        )

    section("CASE-LEVEL FOREGROUND DISTRIBUTION")

    case_distribution = build_case_distribution(
        real_df
    )

    print(
        case_distribution[
            [
                "foreground_voxels",
                "foreground_fraction",
                "num_foreground_classes",
            ]
        ].describe().to_string()
    )

    section("FOREGROUND SPARSITY BINS")

    if not real_df.empty:

        bins = [
            -np.inf,
            0.00001,
            0.0001,
            0.001,
            0.01,
            0.1,
            np.inf,
        ]

        labels = [
            "0 - <0.001%",
            "0.001% - <0.01%",
            "0.01% - <0.1%",
            "0.1% - <1%",
            "1% - <10%",
            ">=10%",
        ]

        fractions_percent = (
            real_df[
                "foreground_fraction"
            ]
            * 100.0
        )

        distribution = pd.cut(
            fractions_percent,
            bins=bins,
            labels=labels,
            right=False,
        ).value_counts(
            sort=False
        )

        for label, count in distribution.items():

            kv(
                str(label),
                int(count),
            )

    section("METADATA VS LOADED MASK CONSISTENCY")

    consistency_df = consistency_audit(
        metadata_df,
        real_df,
    )

    matches = int(
        consistency_df[
            "metadata_foreground_matches_loaded"
        ].sum()
    )

    total_compared = int(
        len(consistency_df)
    )

    kv(
        "Cases compared",
        total_compared,
    )

    kv(
        "Exact foreground-count matches",
        matches,
    )

    kv(
        "Mismatches",
        total_compared - matches,
    )

    section("DIAGNOSTIC INTERPRETATION")

    interpretation = generate_interpretation(
        summary,
        class_table,
    )

    print(
        json.dumps(
            interpretation,
            indent=2,
            default=str,
        )
    )

    section("WRITING AUDIT OUTPUTS")

    metadata_path = (
        OUTPUT_DIR
        / "part31_metadata_foreground_audit.csv"
    )

    real_path = (
        OUTPUT_DIR
        / "part31_real_loaded_mask_statistics.csv"
    )

    class_path = (
        OUTPUT_DIR
        / "part31_class_distribution.csv"
    )

    case_path = (
        OUTPUT_DIR
        / "part31_case_foreground_distribution.csv"
    )

    consistency_path = (
        OUTPUT_DIR
        / "part31_metadata_loaded_consistency.csv"
    )

    errors_path = (
        OUTPUT_DIR
        / "part31_loading_errors.json"
    )

    summary_path = (
        OUTPUT_DIR
        / "part31_summary.json"
    )

    interpretation_path = (
        OUTPUT_DIR
        / "part31_interpretation.json"
    )

    metadata_df.to_csv(
        metadata_path,
        index=False,
    )

    real_df.to_csv(
        real_path,
        index=False,
    )

    class_table.to_csv(
        class_path,
        index=False,
    )

    case_distribution.to_csv(
        case_path,
        index=False,
    )

    consistency_df.to_csv(
        consistency_path,
        index=False,
    )

    with open(
        errors_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            errors,
            f,
            indent=2,
            default=str,
        )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    with open(
        interpretation_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            interpretation,
            f,
            indent=2,
            default=str,
        )

    print("✓ Metadata audit written:")
    print(metadata_path)

    print("✓ Real-mask statistics written:")
    print(real_path)

    print("✓ Class distribution written:")
    print(class_path)

    print("✓ Case distribution written:")
    print(case_path)

    print("✓ Consistency audit written:")
    print(consistency_path)

    print("✓ Loading errors written:")
    print(errors_path)

    print("✓ Summary written:")
    print(summary_path)

    print("✓ Interpretation written:")
    print(interpretation_path)

    section("PART 31 COMPLETE")

    print(
        "No training was performed."
    )

    print(
        "No checkpoint was modified."
    )

    print(
        "No optimizer step was performed."
    )

    print(
        "The complete Part 15 training cohort was audited."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        section("PART 31 ERROR")

        print(
            type(exc).__name__ + ":",
            str(exc),
        )

        traceback.print_exc()

        raise