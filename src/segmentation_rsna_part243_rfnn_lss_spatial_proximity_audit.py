"""
Part 2.43
RFNN-LSS Spatial Proximity Audit

Purpose
-------
Determine whether RFNN annotations are physically/spatially close to
other disease annotations, especially LSS and SCS.

This is ANALYSIS ONLY.

No training.
No checkpoint modification.
No dashboard modification.

Exact validation cohort:
    Part 2.20B

Expected:
    25 validation series
    188 annotated points
    45 RFNN points

Main questions:
    1. How close are RFNN and LSS points?
    2. How close are RFNN and SCS points?
    3. How close are RFNN and LFNN points?
    4. Are RFNN points inside radius 2/4/6 of LSS points?
    5. Does proximity explain the Part 2.42 local winner?
    6. Does the relationship vary by spinal level?
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part243_rfnn_lss_spatial_proximity_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

sys.path.insert(
    0,
    str(SRC_DIR),
)

import segmentation_rsna_part220b_geometry_corrected_training as part220b


# ============================================================================
# CONFIGURATION
# ============================================================================

RADII = [2, 4, 6]

TARGET_CLASSES = {
    "RFNN": "Right Neural Foraminal Narrowing",
    "LFNN": "Left Neural Foraminal Narrowing",
    "LSS": "Left Subarticular Stenosis",
    "RSS": "Right Subarticular Stenosis",
    "SCS": "Spinal Canal Stenosis",
}

LEVELS = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(value):
    try:
        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:
        return np.nan


def euclidean_distance(
    a,
    b,
):
    """
    Euclidean distance between two 3-D points.
    """

    return float(
        math.sqrt(
            sum(
                (
                    float(a[i])
                    - float(b[i])
                ) ** 2
                for i in range(3)
            )
        )
    )


def summarize_values(
    values,
):
    """
    Return standard descriptive statistics.
    """

    values = [
        float(v)
        for v in values
        if np.isfinite(v)
    ]

    if not values:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    array = np.asarray(
        values,
        dtype=float,
    )

    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


# ============================================================================
# EXACT PART 2.20B VALIDATION COHORT
# ============================================================================

def build_exact_validation_cohort(
    manifest,
):
    print()
    print("=" * 80)
    print("BUILDING EXACT PART 2.20B VALIDATION COHORT")
    print("=" * 80)

    selected = (
        part220b.select_validation_series(
            manifest
        )
    )

    if not isinstance(
        selected,
        pd.DataFrame,
    ):
        raise TypeError(
            "Part 2.20B selector must return "
            "a pandas DataFrame."
        )

    if len(selected) != 25:
        raise RuntimeError(
            "Expected 25 validation series, "
            f"found {len(selected)}."
        )

    validation_keys = {
        (
            str(row["study_id"]),
            str(row["series_id"]),
        )
        for _, row in selected.iterrows()
    }

    validation_manifest = manifest[
        manifest.apply(
            lambda row: (
                str(row["study_id"]),
                str(row["series_id"]),
            ) in validation_keys,
            axis=1,
        )
    ].copy()

    validation_cases = (
        part220b.build_case_index(
            validation_manifest
        )
    )

    case_order = {
        (
            str(row["study_id"]),
            str(row["series_id"]),
        ): index
        for index, row in selected.iterrows()
    }

    validation_cases = sorted(
        validation_cases,
        key=lambda case: case_order.get(
            (
                str(case["study_id"]),
                str(case["series_id"]),
            ),
            10000,
        ),
    )

    records = []

    for case in validation_cases:

        study_id = str(
            case["study_id"]
        )

        series_id = str(
            case["series_id"]
        )

        case_points = case["points"]

        if isinstance(
            case_points,
            pd.DataFrame,
        ):
            point_df = (
                case_points
                .reset_index(drop=True)
            )
        else:
            point_df = (
                pd.DataFrame(case_points)
                .reset_index(drop=True)
            )

        for point_index, (_, row) in enumerate(
            point_df.iterrows()
        ):

            records.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "point_index": int(
                        point_index
                    ),
                    "class_id": int(
                        row["class_id"]
                    ),
                    "class_name": str(
                        row["class_name"]
                    ),
                    "level": str(
                        row["level"]
                    ),
                    "native_z": safe_float(
                        row["native_z"]
                    ),
                    "native_y": safe_float(
                        row["native_y"]
                    ),
                    "native_x": safe_float(
                        row["native_x"]
                    ),
                }
            )

    validation_points = pd.DataFrame(
        records
    )

    expected_counts = {
        "Spinal Canal Stenosis": 25,
        "Left Neural Foraminal Narrowing": 45,
        "Right Neural Foraminal Narrowing": 45,
        "Left Subarticular Stenosis": 42,
        "Right Subarticular Stenosis": 31,
    }

    actual_counts = (
        validation_points[
            "class_name"
        ]
        .value_counts()
        .to_dict()
    )

    for disease, expected in (
        expected_counts.items()
    ):

        actual = int(
            actual_counts.get(
                disease,
                0,
            )
        )

        if actual != expected:
            raise RuntimeError(
                f"Cohort mismatch for "
                f"{disease}: expected "
                f"{expected}, found {actual}."
            )

    if len(validation_points) != 188:
        raise RuntimeError(
            "Expected 188 annotated points, "
            f"found {len(validation_points)}."
        )

    print()
    print("EXACT COHORT VERIFIED")
    print("-" * 80)
    print("Validation series : 25")
    print("Validation studies: 25")
    print("Annotated points  : 188")
    print("SCS               : 25")
    print("LFNN              : 45")
    print("RFNN              : 45")
    print("LSS               : 42")
    print("RSS               : 31")
    print("-" * 80)

    return (
        validation_cases,
        validation_points,
    )


# ============================================================================
# BUILD TRANSFORMED POINT TABLE
# ============================================================================

def build_transformed_point_table(
    validation_cases,
):
    """
    Re-run the exact Part 2.20B geometry transformation so that both
    native and canonical/model-grid coordinates are available.
    """

    records = []

    print()
    print("=" * 80)
    print("BUILDING GEOMETRY-CORRECTED POINT TABLE")
    print("=" * 80)

    for case_number, case in enumerate(
        validation_cases,
        start=1,
    ):

        study_id = str(
            case["study_id"]
        )

        series_id = str(
            case["series_id"]
        )

        case_points = case["points"]

        if isinstance(
            case_points,
            pd.DataFrame,
        ):
            point_df = (
                case_points
                .reset_index(drop=True)
            )
        else:
            point_df = (
                pd.DataFrame(case_points)
                .reset_index(drop=True)
            )

        (
            image,
            transformed_points,
            geometry,
        ) = part220b.load_case(
            study_id,
            series_id,
            point_df,
        )

        if len(point_df) != len(
            transformed_points
        ):
            raise RuntimeError(
                f"Point mapping mismatch for "
                f"{study_id}/{series_id}: "
                f"{len(point_df)} original vs "
                f"{len(transformed_points)} transformed."
            )

        for point_index, point in enumerate(
            transformed_points
        ):

            source = point_df.iloc[
                point_index
            ]

            records.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "point_index": int(
                        point_index
                    ),
                    "class_id": int(
                        source["class_id"]
                    ),
                    "class_name": str(
                        source["class_name"]
                    ),
                    "level": str(
                        source["level"]
                    ),
                    "native_z": safe_float(
                        source["native_z"]
                    ),
                    "native_y": safe_float(
                        source["native_y"]
                    ),
                    "native_x": safe_float(
                        source["native_x"]
                    ),
                    "model_z": safe_float(
                        point["z"]
                    ),
                    "model_y": safe_float(
                        point["y"]
                    ),
                    "model_x": safe_float(
                        point["x"]
                    ),
                    "patient_x": safe_float(
                        point.get(
                            "patient_x",
                            np.nan,
                        )
                    ),
                    "patient_y": safe_float(
                        point.get(
                            "patient_y",
                            np.nan,
                        )
                    ),
                    "patient_z": safe_float(
                        point.get(
                            "patient_z",
                            np.nan,
                        )
                    ),
                }
            )

        print(
            f"[{case_number:02d}/25] "
            f"{study_id} / {series_id} "
            f"-> {len(transformed_points)} points"
        )

    point_table = pd.DataFrame(
        records
    )

    if len(point_table) != 188:
        raise RuntimeError(
            "Expected 188 transformed points, "
            f"found {len(point_table)}."
        )

    return point_table


# ============================================================================
# SAME-SERIES PAIRING
# ============================================================================

def build_pair_records(
    point_table,
):
    """
    For every RFNN point, find points of each other disease in the
    same study + series + spinal level.

    Distances are calculated in:
        1. native voxel coordinates
        2. canonical/model coordinates
        3. patient physical coordinates when available
    """

    print()
    print("=" * 80)
    print("BUILDING RFNN DISEASE PROXIMITY PAIRS")
    print("=" * 80)

    rfnn = point_table[
        point_table["class_name"]
        == TARGET_CLASSES["RFNN"]
    ].copy()

    pair_records = []

    for _, rfnn_row in rfnn.iterrows():

        study_id = str(
            rfnn_row["study_id"]
        )

        series_id = str(
            rfnn_row["series_id"]
        )

        level = str(
            rfnn_row["level"]
        )

        same_context = point_table[
            (point_table["study_id"] == study_id)
            & (point_table["series_id"] == series_id)
            & (point_table["level"] == level)
            & (
                point_table["class_name"]
                != TARGET_CLASSES["RFNN"]
            )
        ].copy()

        for target_name in TARGET_CLASSES.values():

            if target_name == TARGET_CLASSES["RFNN"]:
                continue

            candidates = same_context[
                same_context["class_name"]
                == target_name
            ].copy()

            if candidates.empty:

                pair_records.append(
                    {
                        "study_id": study_id,
                        "series_id": series_id,
                        "level": level,
                        "rfnn_point_index": int(
                            rfnn_row["point_index"]
                        ),
                        "comparison_class": target_name,
                        "comparison_point_index": np.nan,
                        "native_distance": np.nan,
                        "model_distance": np.nan,
                        "patient_distance": np.nan,
                        "model_distance_z": np.nan,
                        "model_distance_y": np.nan,
                        "model_distance_x": np.nan,
                    }
                )

                continue

            candidate_records = []

            for _, candidate in candidates.iterrows():

                native_distance = euclidean_distance(
                    (
                        rfnn_row["native_z"],
                        rfnn_row["native_y"],
                        rfnn_row["native_x"],
                    ),
                    (
                        candidate["native_z"],
                        candidate["native_y"],
                        candidate["native_x"],
                    ),
                )

                model_dz = (
                    float(rfnn_row["model_z"])
                    - float(candidate["model_z"])
                )

                model_dy = (
                    float(rfnn_row["model_y"])
                    - float(candidate["model_y"])
                )

                model_dx = (
                    float(rfnn_row["model_x"])
                    - float(candidate["model_x"])
                )

                model_distance = math.sqrt(
                    model_dz ** 2
                    + model_dy ** 2
                    + model_dx ** 2
                )

                patient_values = [
                    rfnn_row["patient_x"],
                    rfnn_row["patient_y"],
                    rfnn_row["patient_z"],
                    candidate["patient_x"],
                    candidate["patient_y"],
                    candidate["patient_z"],
                ]

                if all(
                    np.isfinite(v)
                    for v in patient_values
                ):

                    patient_distance = euclidean_distance(
                        (
                            rfnn_row["patient_x"],
                            rfnn_row["patient_y"],
                            rfnn_row["patient_z"],
                        ),
                        (
                            candidate["patient_x"],
                            candidate["patient_y"],
                            candidate["patient_z"],
                        ),
                    )

                else:

                    patient_distance = np.nan

                candidate_records.append(
                    {
                        "study_id": study_id,
                        "series_id": series_id,
                        "level": level,
                        "rfnn_point_index": int(
                            rfnn_row["point_index"]
                        ),
                        "comparison_class": target_name,
                        "comparison_point_index": int(
                            candidate["point_index"]
                        ),
                        "native_distance": native_distance,
                        "model_distance": model_distance,
                        "patient_distance": patient_distance,
                        "model_distance_z": abs(
                            model_dz
                        ),
                        "model_distance_y": abs(
                            model_dy
                        ),
                        "model_distance_x": abs(
                            model_dx
                        ),
                    }
                )

            # Keep the nearest same-class point.
            candidate_records.sort(
                key=lambda item: (
                    item["model_distance"]
                    if np.isfinite(
                        item["model_distance"]
                    )
                    else float("inf")
                )
            )

            pair_records.append(
                candidate_records[0]
            )

    pairs = pd.DataFrame(
        pair_records
    )

    expected_rows = (
        45 * 4
    )

    if len(pairs) != expected_rows:
        raise RuntimeError(
            "Expected "
            f"{expected_rows} RFNN comparison rows, "
            f"found {len(pairs)}."
        )

    print(
        f"RFNN points analyzed : 45"
    )

    print(
        f"Comparison rows      : {len(pairs)}"
    )

    return pairs


# ============================================================================
# LOCAL RADIUS OVERLAP ANALYSIS
# ============================================================================

def add_radius_overlap_columns(
    pairs,
):
    """
    Determine whether the nearest comparison point lies within
    model-grid radius 2, 4, or 6.
    """

    result = pairs.copy()

    for radius in RADII:

        result[
            f"within_radius_{radius}"
        ] = (
            result[
                "model_distance"
            ]
            <= float(radius)
        )

    return result


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    point_table,
    pairs,
):
    print()
    print("=" * 80)
    print("BUILDING PART 2.43 SUMMARY")
    print("=" * 80)

    # ========================================================================
    # DISTANCE SUMMARY
    # ========================================================================

    distance_rows = []

    for comparison_class in [
        TARGET_CLASSES["LFNN"],
        TARGET_CLASSES["LSS"],
        TARGET_CLASSES["RSS"],
        TARGET_CLASSES["SCS"],
    ]:

        subset = pairs[
            pairs["comparison_class"]
            == comparison_class
        ].copy()

        for coordinate_name in [
            "native_distance",
            "model_distance",
            "patient_distance",
        ]:

            stats = summarize_values(
                subset[
                    coordinate_name
                ].tolist()
            )

            distance_rows.append(
                {
                    "comparison_class": comparison_class,
                    "coordinate_space": coordinate_name,
                    **stats,
                }
            )

    distance_summary = pd.DataFrame(
        distance_rows
    )

    # ========================================================================
    # RADIUS SUMMARY
    # ========================================================================

    radius_rows = []

    for comparison_class in [
        TARGET_CLASSES["LFNN"],
        TARGET_CLASSES["LSS"],
        TARGET_CLASSES["RSS"],
        TARGET_CLASSES["SCS"],
    ]:

        subset = pairs[
            pairs["comparison_class"]
            == comparison_class
        ].copy()

        row = {
            "comparison_class": comparison_class,
            "count": len(subset),
        }

        for radius in RADII:

            column = (
                f"within_radius_{radius}"
            )

            row[
                f"within_radius_{radius}_count"
            ] = int(
                subset[column].sum()
            )

            row[
                f"within_radius_{radius}_fraction"
            ] = float(
                subset[column].mean()
            )

        radius_rows.append(
            row
        )

    radius_summary = pd.DataFrame(
        radius_rows
    )

    # ========================================================================
    # LEVEL SUMMARY
    # ========================================================================

    level_rows = []

    for level in LEVELS:

        for comparison_class in [
            TARGET_CLASSES["LFNN"],
            TARGET_CLASSES["LSS"],
            TARGET_CLASSES["RSS"],
            TARGET_CLASSES["SCS"],
        ]:

            subset = pairs[
                (pairs["level"] == level)
                & (
                    pairs["comparison_class"]
                    == comparison_class
                )
            ].copy()

            if subset.empty:
                continue

            stats = summarize_values(
                subset[
                    "model_distance"
                ].tolist()
            )

            row = {
                "level": level,
                "comparison_class": comparison_class,
                **stats,
            }

            for radius in RADII:

                column = (
                    f"within_radius_{radius}"
                )

                row[
                    f"within_radius_{radius}_fraction"
                ] = float(
                    subset[column].mean()
                )

            level_rows.append(
                row
            )

    level_summary = pd.DataFrame(
        level_rows
    )

    # ========================================================================
    # RFNN TO LSS DIRECT TABLE
    # ========================================================================

    rfnn_lss = pairs[
        pairs["comparison_class"]
        == TARGET_CLASSES["LSS"]
    ].copy()

    rfnn_scs = pairs[
        pairs["comparison_class"]
        == TARGET_CLASSES["SCS"]
    ].copy()

    rfnn_lfnn = pairs[
        pairs["comparison_class"]
        == TARGET_CLASSES["LFNN"]
    ].copy()

    # ========================================================================
    # SAVE
    # ========================================================================

    pairs_path = (
        OUTPUT_DIR
        / "part243_rfnn_disease_proximity_pairs.csv"
    )

    distance_path = (
        OUTPUT_DIR
        / "part243_distance_summary.csv"
    )

    radius_path = (
        OUTPUT_DIR
        / "part243_radius_overlap_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part243_level_proximity_summary.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part243_spatial_proximity_audit_summary.json"
    )

    pairs.to_csv(
        pairs_path,
        index=False,
    )

    distance_summary.to_csv(
        distance_path,
        index=False,
    )

    radius_summary.to_csv(
        radius_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    summary = {
        "validation_cases": 25,
        "validation_points": 188,
        "rfnn_points": 45,
        "comparison_classes": {
            "LFNN": 45,
            "LSS": 45,
            "RSS": 45,
            "SCS": 45,
        },
        "rfnn_lss_model_distance": summarize_values(
            rfnn_lss[
                "model_distance"
            ].tolist()
        ),
        "rfnn_scs_model_distance": summarize_values(
            rfnn_scs[
                "model_distance"
            ].tolist()
        ),
        "rfnn_lfnn_model_distance": summarize_values(
            rfnn_lfnn[
                "model_distance"
            ].tolist()
        ),
        "rfnn_lss_within_radius_2_fraction": float(
            rfnn_lss[
                "within_radius_2"
            ].mean()
        ),
        "rfnn_lss_within_radius_4_fraction": float(
            rfnn_lss[
                "within_radius_4"
            ].mean()
        ),
        "rfnn_lss_within_radius_6_fraction": float(
            rfnn_lss[
                "within_radius_6"
            ].mean()
        ),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ========================================================================
    # TERMINAL REPORT
    # ========================================================================

    print()
    print("=" * 80)
    print("RFNN → LSS SPATIAL PROXIMITY")
    print("=" * 80)

    print(
        distance_summary[
            distance_summary[
                "comparison_class"
            ]
            == TARGET_CLASSES["LSS"]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("RFNN → SCS SPATIAL PROXIMITY")
    print("=" * 80)

    print(
        distance_summary[
            distance_summary[
                "comparison_class"
            ]
            == TARGET_CLASSES["SCS"]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("RFNN → LFNN SPATIAL PROXIMITY")
    print("=" * 80)

    print(
        distance_summary[
            distance_summary[
                "comparison_class"
            ]
            == TARGET_CLASSES["LFNN"]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("RADIUS OVERLAP")
    print("=" * 80)

    print(
        radius_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("LEVEL-WISE PROXIMITY")
    print("=" * 80)

    print(
        level_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.43 COMPLETE")
    print("=" * 80)

    print(
        "Analysis only."
    )

    print(
        "Part 2.27 checkpoint unchanged."
    )

    print(
        "Dashboard unchanged."
    )

    print()
    print("Outputs:")

    print(
        pairs_path
    )

    print(
        distance_path
    )

    print(
        radius_path
    )

    print(
        level_path
    )

    print(
        summary_path
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.43")
    print("RFNN-LSS SPATIAL PROXIMITY AUDIT")
    print("=" * 80)

    print()
    print("ANALYSIS ONLY")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    manifest = (
        part220b.load_manifest()
    )

    print()
    print(
        f"Full manifest rows: "
        f"{len(manifest)}"
    )

    (
        validation_cases,
        validation_points,
    ) = build_exact_validation_cohort(
        manifest
    )

    point_table = (
        build_transformed_point_table(
            validation_cases
        )
    )

    pairs = build_pair_records(
        point_table
    )

    pairs = add_radius_overlap_columns(
        pairs
    )

    build_summary(
        point_table,
        pairs,
    )


if __name__ == "__main__":
    main()