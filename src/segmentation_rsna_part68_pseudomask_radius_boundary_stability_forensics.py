"""
PHASE 4 - PART 68
RSNA-ONLY PSEUDOMASK RADIUS / BOUNDARY / CORE STABILITY FORENSICS

Purpose
-------
Determine whether increasing pseudo-mask dilation mainly adds useful
foreground supervision or uncertain/noisy boundary voxels.

This is an evaluation-only experiment.

NO TRAINING
NO OPTIMIZER
NO MODEL WEIGHT MODIFICATION
NO SPIDER
NO RSNA TEST SET

Compared regions:
    R0 = point/core mask
    R1 = 1-voxel dilation
    R2 = 2-voxel dilation
    R3 = 3-voxel dilation

Additional R2 decomposition:
    R2_CORE  = erosion of R2 by one voxel
    R2_HALO  = R2 - R2_CORE

The experiment also measures annotation agreement with each region.
"""

from __future__ import annotations

import gc
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import pydicom

from scipy import ndimage


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"
)

SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part68_pseudomask_radius_boundary_stability_forensics"
)

REPORT_DIR = OUTPUT_DIR / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# DATA PATHS
# ============================================================================

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

TRAIN_COHORT = PART15_DIR / "part15_train_cohort.csv"
VAL_COHORT = PART15_DIR / "part15_validation_cohort.csv"

PART6_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part6_pseudomask_generation"
)

PSEUDOMASK_DIR = PART6_DIR / "pseudo_masks"


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

MAX_VALIDATION_CASES = 100

LOCAL_RADII = [0, 1, 2, 3]

# For the R2 boundary analysis.
R2_EROSION_ITERATIONS = 1

# Labels
LABELS = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 82)
    print(title)
    print("=" * 82)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_cuda_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# PART 9 / PART 11 IMPORT
# ============================================================================

def load_part11() -> Any:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    import segmentation_rsna_part11_controlled_pilot_training as part11

    return part11


def load_part9() -> Any:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    import segmentation_rsna_part9_3d_dataset_loader as part9

    return part9


# ============================================================================
# IDENTIFIER HELPERS
# ============================================================================

def first_value(
    row: pd.Series,
    names: List[str],
) -> Any:

    for name in names:
        if name in row.index:
            value = row[name]

            if pd.notna(value):
                return value

    return None


def as_id(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, float):
        if np.isnan(value):
            return ""

        if value.is_integer():
            return str(int(value))

    return str(value).strip()


# ============================================================================
# PSEUDOMASK RESOLUTION
# ============================================================================

def resolve_pseudomask_path(row: pd.Series) -> Path:

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

    if not study_id or not series_id:
        raise RuntimeError(
            "Cannot resolve pseudo-mask: missing study_id or series_id."
        )

    candidates = [
        PSEUDOMASK_DIR / f"{study_id}_{series_id}.npz",
        PSEUDOMASK_DIR / f"{study_id}__{series_id}.npz",
        PSEUDOMASK_DIR / f"{series_id}.npz",
    ]

    for path in candidates:

        if path.exists():
            return path

    matches = list(
        PSEUDOMASK_DIR.rglob(
            f"*{series_id}*.npz"
        )
    )

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        "Could not resolve pseudo-mask for "
        f"study={study_id}, series={series_id}."
    )


# ============================================================================
# PSEUDOMASK LOADING
# ============================================================================

def load_pseudomask(
    row: pd.Series,
) -> np.ndarray:

    path = resolve_pseudomask_path(row)

    data = np.load(
        path,
        allow_pickle=True,
    )

    keys = list(data.keys())

    if not keys:
        raise RuntimeError(
            f"Pseudo-mask archive is empty: {path}"
        )

    # Prefer common mask names.
    preferred = [
        "mask",
        "pseudomask",
        "pseudo_mask",
        "labels",
        "label",
        "arr_0",
    ]

    selected_key = None

    for key in preferred:
        if key in data:
            selected_key = key
            break

    if selected_key is None:
        selected_key = keys[0]

    mask = np.asarray(
        data[selected_key]
    )

    mask = np.squeeze(mask)

    if mask.ndim != 3:
        raise RuntimeError(
            f"Expected 3-D pseudo-mask, got {mask.shape} "
            f"from {path}"
        )

    return mask.astype(
        np.int16,
        copy=False,
    )


# ============================================================================
# LOCAL DILATION
# ============================================================================

def offsets(radius: int) -> List[Tuple[int, int, int]]:

    if radius <= 0:
        return [(0, 0, 0)]

    result = []

    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):

                distance = (
                    dz * dz
                    + dy * dy
                    + dx * dx
                )

                if distance <= radius * radius:
                    result.append(
                        (dz, dy, dx)
                    )

    return result


def dilate_class_mask(
    binary_mask: np.ndarray,
    radius: int,
) -> np.ndarray:

    if radius == 0:
        return binary_mask.astype(bool)

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=1,
    )

    # Repeated binary dilation gives a consistent
    # voxel-neighbourhood expansion.
    return ndimage.binary_dilation(
        binary_mask,
        structure=structure,
        iterations=radius,
    )


def make_radius_mask(
    base_mask: np.ndarray,
    radius: int,
) -> np.ndarray:

    output = np.zeros_like(
        base_mask,
        dtype=np.int16,
    )

    for class_id in range(1, 6):

        class_region = (
            base_mask == class_id
        )

        if not np.any(class_region):
            continue

        expanded = dilate_class_mask(
            class_region,
            radius,
        )

        # Preserve the original class.
        output[
            expanded & (output == 0)
        ] = class_id

    return output


# ============================================================================
# R2 CORE / HALO
# ============================================================================

def build_r2_core_and_halo(
    r2_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    core = np.zeros_like(
        r2_mask,
        dtype=np.int16,
    )

    for class_id in range(1, 6):

        region = (
            r2_mask == class_id
        )

        if not np.any(region):
            continue

        eroded = ndimage.binary_erosion(
            region,
            structure=ndimage.generate_binary_structure(
                rank=3,
                connectivity=1,
            ),
            iterations=R2_EROSION_ITERATIONS,
        )

        core[
            eroded
        ] = class_id

    halo = (
        (r2_mask > 0)
        & (core == 0)
    )

    return core, halo


# ============================================================================
# CASE STATISTICS
# ============================================================================

def region_statistics(
    mask: np.ndarray,
    region_name: str,
) -> Dict[str, Any]:

    total_voxels = int(mask.size)

    fg = mask > 0

    foreground_voxels = int(
        np.count_nonzero(fg)
    )

    components = 0

    if foreground_voxels > 0:

        _, components = ndimage.label(
            fg
        )

    class_counts = {}

    for class_id in range(1, 6):

        class_counts[
            f"class_{class_id}_voxels"
        ] = int(
            np.count_nonzero(
                mask == class_id
            )
        )

    result = {
        "region": region_name,
        "total_voxels": total_voxels,
        "foreground_voxels": foreground_voxels,
        "foreground_fraction": (
            foreground_voxels / total_voxels
            if total_voxels
            else 0.0
        ),
        "connected_components": int(
            components
        ),
    }

    result.update(class_counts)

    return result


# ============================================================================
# REGION COMPARISON
# ============================================================================

def compare_regions(
    masks: Dict[str, np.ndarray],
) -> Dict[str, Any]:

    r0 = masks["R0"]
    r1 = masks["R1"]
    r2 = masks["R2"]
    r3 = masks["R3"]

    r0_fg = r0 > 0
    r1_fg = r1 > 0
    r2_fg = r2 > 0
    r3_fg = r3 > 0

    r2_count = int(
        np.count_nonzero(r2_fg)
    )

    r2_core_count = int(
        np.count_nonzero(
            masks["R2_CORE"] > 0
        )
    )

    r2_halo_count = int(
        np.count_nonzero(
            masks["R2_HALO"]
        )
    )

    result = {
        "R0_fg": int(
            np.count_nonzero(r0_fg)
        ),
        "R1_fg": int(
            np.count_nonzero(r1_fg)
        ),
        "R2_fg": int(
            np.count_nonzero(r2_fg)
        ),
        "R3_fg": int(
            np.count_nonzero(r3_fg)
        ),

        "R0_to_R2_retention": (
            float(
                np.count_nonzero(
                    r0_fg & r2_fg
                )
            )
            / max(
                1,
                np.count_nonzero(r0_fg),
            )
        ),

        "R1_to_R2_retention": (
            float(
                np.count_nonzero(
                    r1_fg & r2_fg
                )
            )
            / max(
                1,
                np.count_nonzero(r1_fg),
            )
        ),

        "R2_core_voxels": r2_core_count,

        "R2_halo_voxels": r2_halo_count,

        "R2_core_fraction": (
            r2_core_count
            / max(1, r2_count)
        ),

        "R2_halo_fraction": (
            r2_halo_count
            / max(1, r2_count)
        ),

        "R2_inside_R3_fraction": (
            float(
                np.count_nonzero(
                    r2_fg & r3_fg
                )
            )
            / max(
                1,
                np.count_nonzero(r2_fg),
            )
        ),
    }

    return result


# ============================================================================
# ANNOTATION EXTRACTION
# ============================================================================

def extract_annotation_points(
    row: pd.Series,
) -> List[Tuple[int, int, int, int]]:

    """
    Attempts to extract annotation coordinates from common
    RSNA manifest column layouts.

    Returns:
        [(z, y, x, class_id), ...]
    """

    points = []

    columns = set(
        str(c)
        for c in row.index
    )

    # Common direct column naming patterns.
    possible_sets = [
        (
            ["z", "slice", "instance"],
            ["y", "coord_y"],
            ["x", "coord_x"],
            ["label", "class_id", "level_label"],
        ),
        (
            ["point_z"],
            ["point_y"],
            ["point_x"],
            ["class_id"],
        ),
    ]

    for z_names, y_names, x_names, c_names in possible_sets:

        z_value = first_value(
            row,
            z_names,
        )

        y_value = first_value(
            row,
            y_names,
        )

        x_value = first_value(
            row,
            x_names,
        )

        class_value = first_value(
            row,
            c_names,
        )

        if (
            z_value is not None
            and y_value is not None
            and x_value is not None
            and class_value is not None
        ):

            try:
                z = int(round(float(z_value)))
                y = int(round(float(y_value)))
                x = int(round(float(x_value)))
                c = int(round(float(class_value)))

                if 1 <= c <= 5:
                    points.append(
                        (z, y, x, c)
                    )

            except Exception:
                pass

            if points:
                return points

    return points


# ============================================================================
# POINT AGREEMENT
# ============================================================================

def point_region_label(
    mask: np.ndarray,
    point: Tuple[int, int, int, int],
) -> str:

    z, y, x, expected_class = point

    if not (
        0 <= z < mask.shape[0]
        and 0 <= y < mask.shape[1]
        and 0 <= x < mask.shape[2]
    ):
        return "OUTSIDE"

    actual = int(
        mask[z, y, x]
    )

    if actual == expected_class:
        return "CORRECT_CLASS"

    if actual == 0:
        return "BACKGROUND"

    return "OTHER_FOREGROUND"


def annotation_agreement(
    mask: np.ndarray,
    points: List[Tuple[int, int, int, int]],
) -> Dict[str, Any]:

    counts = {
        "CORRECT_CLASS": 0,
        "OTHER_FOREGROUND": 0,
        "BACKGROUND": 0,
        "OUTSIDE": 0,
    }

    for point in points:

        result = point_region_label(
            mask,
            point,
        )

        counts[result] += 1

    total = len(points)

    return {
        "annotations": total,
        "correct_class": counts["CORRECT_CLASS"],
        "other_foreground": counts["OTHER_FOREGROUND"],
        "background": counts["BACKGROUND"],
        "outside": counts["OUTSIDE"],
        "correct_fraction": (
            counts["CORRECT_CLASS"] / total
            if total
            else 0.0
        ),
        "inside_fraction": (
            (
                total
                - counts["OUTSIDE"]
            ) / total
            if total
            else 0.0
        ),
    }


# ============================================================================
# PER-CLASS AGREEMENT
# ============================================================================

def per_class_agreement(
    mask: np.ndarray,
    points: List[Tuple[int, int, int, int]],
    region_name: str,
) -> List[Dict[str, Any]]:

    rows = []

    for class_id in range(1, 6):

        class_points = [
            p
            for p in points
            if p[3] == class_id
        ]

        counts = {
            "CORRECT_CLASS": 0,
            "OTHER_FOREGROUND": 0,
            "BACKGROUND": 0,
            "OUTSIDE": 0,
        }

        for point in class_points:

            result = point_region_label(
                mask,
                point,
            )

            counts[result] += 1

        total = len(class_points)

        rows.append(
            {
                "region": region_name,
                "class_id": class_id,
                "class_name": LABELS[class_id],
                "annotations": total,
                "correct_class": counts["CORRECT_CLASS"],
                "other_foreground": counts["OTHER_FOREGROUND"],
                "background": counts["BACKGROUND"],
                "outside": counts["OUTSIDE"],
                "correct_fraction": (
                    counts["CORRECT_CLASS"] / total
                    if total
                    else 0.0
                ),
            }
        )

    return rows


# ============================================================================
# VALIDATION COHORT
# ============================================================================

def load_validation_cohort() -> pd.DataFrame:

    if not VAL_COHORT.exists():

        raise FileNotFoundError(
            f"Validation cohort not found:\n{VAL_COHORT}"
        )

    df = pd.read_csv(
        VAL_COHORT
    )

    if df.empty:
        raise RuntimeError(
            "Validation cohort is empty."
        )

    if len(df) > MAX_VALIDATION_CASES:

        df = df.sample(
            n=MAX_VALIDATION_CASES,
            random_state=SEED,
        )

    return df.reset_index(
        drop=True
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    banner(
        "PHASE 4 - PART 68"
    )

    print(
        "RSNA-ONLY PSEUDOMASK RADIUS / BOUNDARY / CORE STABILITY FORENSICS"
    )

    print()
    print(
        "Evaluation only."
    )
    print(
        "No training."
    )
    print(
        "No optimizer."
    )
    print(
        "No model weights modified."
    )
    print(
        "SPIDER not used."
    )
    print(
        "RSNA test set not used."
    )
    print()

    set_seed(
        SEED
    )

    reset_cuda_memory()

    # Load modules so this experiment remains tied to the
    # validated project environment.
    part11 = load_part11()
    part9 = load_part9()

    print(
        f"Part 11 loaded: {part11.__name__}"
    )

    print(
        f"Part 9 loaded: {part9.__name__}"
    )

    print()

    df = load_validation_cohort()

    print(
        f"Validation cases selected: {len(df)}"
    )

    print(
        f"Pseudo-mask directory: {PSEUDOMASK_DIR}"
    )

    print()

    case_rows = []
    region_rows = []
    agreement_rows = []
    per_class_rows = []

    successful = 0
    failed = 0

    for index, row in df.iterrows():

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

        print(
            f"[{index + 1:03d}/{len(df):03d}] "
            f"study={study_id} "
            f"series={series_id}"
        )

        try:

            base_mask = load_pseudomask(
                row
            )

            # ------------------------------------------------------------
            # Build radius masks.
            # ------------------------------------------------------------

            masks = {}

            for radius in LOCAL_RADII:

                masks[
                    f"R{radius}"
                ] = make_radius_mask(
                    base_mask,
                    radius,
                )

            # ------------------------------------------------------------
            # R2 core / halo.
            # ------------------------------------------------------------

            r2_core, r2_halo = (
                build_r2_core_and_halo(
                    masks["R2"]
                )
            )

            masks["R2_CORE"] = r2_core
            masks["R2_HALO"] = (
                r2_halo.astype(
                    np.int16
                )
            )

            # ------------------------------------------------------------
            # Region statistics.
            # ------------------------------------------------------------

            for region_name in [
                "R0",
                "R1",
                "R2",
                "R3",
                "R2_CORE",
                "R2_HALO",
            ]:

                stats = region_statistics(
                    masks[region_name],
                    region_name,
                )

                stats.update(
                    {
                        "case_index": index,
                        "study_id": study_id,
                        "series_id": series_id,
                    }
                )

                region_rows.append(
                    stats
                )

            # ------------------------------------------------------------
            # Region transition statistics.
            # ------------------------------------------------------------

            comparison = compare_regions(
                masks
            )

            comparison.update(
                {
                    "case_index": index,
                    "study_id": study_id,
                    "series_id": series_id,
                }
            )

            case_rows.append(
                comparison
            )

            # ------------------------------------------------------------
            # Annotation points.
            # ------------------------------------------------------------

            points = extract_annotation_points(
                row
            )

            # ------------------------------------------------------------
            # Agreement by radius.
            # ------------------------------------------------------------

            for region_name in [
                "R0",
                "R1",
                "R2",
                "R3",
                "R2_CORE",
                "R2_HALO",
            ]:

                agreement = annotation_agreement(
                    masks[region_name],
                    points,
                )

                agreement.update(
                    {
                        "case_index": index,
                        "study_id": study_id,
                        "series_id": series_id,
                        "region": region_name,
                    }
                )

                agreement_rows.append(
                    agreement
                )

                class_rows = (
                    per_class_agreement(
                        masks[region_name],
                        points,
                        region_name,
                    )
                )

                for item in class_rows:

                    item.update(
                        {
                            "case_index": index,
                            "study_id": study_id,
                            "series_id": series_id,
                        }
                    )

                    per_class_rows.append(
                        item
                    )

            successful += 1

        except Exception as exc:

            failed += 1

            print(
                f"    ERROR: {type(exc).__name__}: {exc}"
            )

    # =========================================================================
    # DATAFRAMES
    # =========================================================================

    region_df = pd.DataFrame(
        region_rows
    )

    comparison_df = pd.DataFrame(
        case_rows
    )

    agreement_df = pd.DataFrame(
        agreement_rows
    )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    # =========================================================================
    # SAVE CASE-LEVEL RESULTS
    # =========================================================================

    region_path = (
        REPORT_DIR
        / "part68_region_statistics.csv"
    )

    comparison_path = (
        REPORT_DIR
        / "part68_radius_transition_statistics.csv"
    )

    agreement_path = (
        REPORT_DIR
        / "part68_annotation_agreement_by_region.csv"
    )

    class_path = (
        REPORT_DIR
        / "part68_per_class_region_agreement.csv"
    )

    region_df.to_csv(
        region_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    agreement_df.to_csv(
        agreement_path,
        index=False,
    )

    per_class_df.to_csv(
        class_path,
        index=False,
    )

    # =========================================================================
    # AGGREGATE SUMMARY
    # =========================================================================

    summary = {
        "phase": "PHASE 4 - PART 68",
        "experiment": (
            "RSNA pseudomask radius / boundary / core stability forensics"
        ),
        "evaluation_only": True,
        "training_performed": False,
        "optimizer_step_performed": False,
        "model_weights_modified": False,
        "spider_used": False,
        "rsna_test_set_used": False,
        "seed": SEED,
        "validation_cases_requested": MAX_VALIDATION_CASES,
        "validation_cases_successful": successful,
        "validation_cases_failed": failed,
    }

    # -------------------------------------------------------------------------
    # Aggregate region statistics.
    # -------------------------------------------------------------------------

    if not region_df.empty:

        region_summary = (
            region_df
            .groupby("region")
            [
                [
                    "foreground_voxels",
                    "foreground_fraction",
                    "connected_components",
                    "class_1_voxels",
                    "class_2_voxels",
                    "class_3_voxels",
                    "class_4_voxels",
                    "class_5_voxels",
                ]
            ]
            .mean()
            .reset_index()
        )

        summary[
            "mean_region_statistics"
        ] = region_summary.to_dict(
            orient="records"
        )

    # -------------------------------------------------------------------------
    # Aggregate annotation agreement.
    # -------------------------------------------------------------------------

    if not agreement_df.empty:

        agreement_summary = (
            agreement_df
            .groupby("region")
            [
                [
                    "annotations",
                    "correct_class",
                    "other_foreground",
                    "background",
                    "outside",
                    "correct_fraction",
                    "inside_fraction",
                ]
            ]
            .mean()
            .reset_index()
        )

        summary[
            "mean_annotation_agreement"
        ] = agreement_summary.to_dict(
            orient="records"
        )

    # -------------------------------------------------------------------------
    # R2 boundary summary.
    # -------------------------------------------------------------------------

    if not comparison_df.empty:

        summary[
            "R2_boundary_analysis"
        ] = {
            "mean_R2_core_fraction": float(
                comparison_df[
                    "R2_core_fraction"
                ].mean()
            ),
            "mean_R2_halo_fraction": float(
                comparison_df[
                    "R2_halo_fraction"
                ].mean()
            ),
            "mean_R0_to_R2_retention": float(
                comparison_df[
                    "R0_to_R2_retention"
                ].mean()
            ),
            "mean_R1_to_R2_retention": float(
                comparison_df[
                    "R1_to_R2_retention"
                ].mean()
            ),
        }

    # =========================================================================
    # DIAGNOSIS
    # =========================================================================

    diagnosis = (
        "PART68_INCONCLUSIVE"
    )

    if (
        not agreement_df.empty
        and not comparison_df.empty
    ):

        r2_agreement = agreement_df[
            agreement_df["region"] == "R2"
        ]

        core_agreement = agreement_df[
            agreement_df["region"] == "R2_CORE"
        ]

        halo_agreement = agreement_df[
            agreement_df["region"] == "R2_HALO"
        ]

        if (
            not r2_agreement.empty
            and not core_agreement.empty
            and not halo_agreement.empty
        ):

            r2_correct = float(
                r2_agreement[
                    "correct_fraction"
                ].mean()
            )

            core_correct = float(
                core_agreement[
                    "correct_fraction"
                ].mean()
            )

            halo_correct = float(
                halo_agreement[
                    "correct_fraction"
                ].mean()
            )

            halo_fraction = float(
                comparison_df[
                    "R2_halo_fraction"
                ].mean()
            )

            if (
                core_correct > r2_correct
                and halo_correct < core_correct
                and halo_fraction > 0.20
            ):

                diagnosis = (
                    "R2_HALO_CONTAINS_WEAKER_SUPERVISION_SIGNAL"
                )

            elif (
                abs(core_correct - r2_correct)
                < 0.03
            ):

                diagnosis = (
                    "R2_CORE_AND_FULL_R2_HAVE_SIMILAR_POINT_AGREEMENT"
                )

            else:

                diagnosis = (
                    "PSEUDOMASK_RADIUS_EFFECT_IS_MIXED"
                )

    summary[
        "diagnosis"
    ] = diagnosis

    summary[
        "interpretation"
    ] = (
        "Part 68 determines whether larger pseudo-mask radii "
        "increase useful supervision or primarily add uncertain "
        "boundary voxels. It does not train a model."
    )

    # =========================================================================
    # SAVE JSON
    # =========================================================================

    summary_path = (
        REPORT_DIR
        / "part68_summary.json"
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

    # =========================================================================
    # HUMAN-READABLE REPORT
    # =========================================================================

    report_lines = [
        "PHASE 4 - PART 68",
        "RSNA-ONLY PSEUDOMASK RADIUS / BOUNDARY / CORE STABILITY FORENSICS",
        "",
        "Evaluation only.",
        "No training performed.",
        "No optimizer step performed.",
        "No model weights modified.",
        "SPIDER not used.",
        "RSNA test set not used.",
        "",
        "SUMMARY",
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        "",
        "FILES",
        str(region_path),
        str(comparison_path),
        str(agreement_path),
        str(class_path),
        str(summary_path),
    ]

    report_path = (
        REPORT_DIR
        / "part68_pseudomask_radius_boundary_stability_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # =========================================================================
    # FINAL CONSOLE OUTPUT
    # =========================================================================

    banner(
        "PART 68 COMPLETE"
    )

    print(
        f"Successful cases : {successful}"
    )

    print(
        f"Failed cases     : {failed}"
    )

    print()

    print(
        f"Diagnosis: {diagnosis}"
    )

    print()

    print(
        f"Saved: {region_path}"
    )

    print(
        f"Saved: {comparison_path}"
    )

    print(
        f"Saved: {agreement_path}"
    )

    print(
        f"Saved: {class_path}"
    )

    print(
        f"Saved: {summary_path}"
    )

    print(
        f"Saved: {report_path}"
    )

    print()


if __name__ == "__main__":
    main()