"""
Part 2.44
RFNN Cross-Series Physical-Space Proximity Audit

Purpose
-------
Part 2.43 showed that LSS/SCS/RSS annotations are generally not present
in the SAME MRI SERIES as the RFNN annotation.

Therefore, Part 2.43 cannot test RFNN-to-LSS/SCS physical proximity.

Part 2.44 performs the correct comparison:

    same study
    + same spinal level
    + different/same series allowed
    + patient physical coordinates

This avoids comparing voxel coordinates from different MRI acquisitions.

Questions:
    1. How physically close are RFNN and LSS annotations?
    2. How physically close are RFNN and SCS annotations?
    3. How physically close are RFNN and RSS annotations?
    4. How physically close are RFNN and LFNN annotations?
    5. Does proximity differ by spinal level?
    6. Are RFNN points physically close enough to explain
       the Part 2.42 LSS/SCS local dominance?

ANALYSIS ONLY.

No training.
No checkpoint modification.
No dashboard modification.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part244_rfnn_cross_series_physical_proximity_audit"
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

CLASS_NAMES = {
    "SCS": "Spinal Canal Stenosis",
    "LFNN": "Left Neural Foraminal Narrowing",
    "RFNN": "Right Neural Foraminal Narrowing",
    "LSS": "Left Subarticular Stenosis",
    "RSS": "Right Subarticular Stenosis",
}

COMPARISON_ORDER = [
    "LFNN",
    "LSS",
    "RSS",
    "SCS",
]

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


def physical_distance(
    a,
    b,
):
    """
    Euclidean distance in patient physical coordinates.
    """

    values = [
        a[0],
        a[1],
        a[2],
        b[0],
        b[1],
        b[2],
    ]

    if not all(
        np.isfinite(v)
        for v in values
    ):
        return np.nan

    return float(
        math.sqrt(
            (
                a[0] - b[0]
            ) ** 2
            + (
                a[1] - b[1]
            ) ** 2
            + (
                a[2] - b[2]
            ) ** 2
        )
    )


def summarize(
    values,
):
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
        "mean": float(
            np.mean(array)
        ),
        "median": float(
            np.median(array)
        ),
        "std": float(
            np.std(array)
        ),
        "min": float(
            np.min(array)
        ),
        "max": float(
            np.max(array)
        ),
    }


# ============================================================================
# EXACT PART 2.20B COHORT
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
                f"{study_id}/{series_id}."
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

    point_table = pd.DataFrame(
        records
    )

    if len(point_table) != 188:
        raise RuntimeError(
            "Expected 188 transformed points, "
            f"found {len(point_table)}."
        )

    expected_counts = {
        CLASS_NAMES["SCS"]: 25,
        CLASS_NAMES["LFNN"]: 45,
        CLASS_NAMES["RFNN"]: 45,
        CLASS_NAMES["LSS"]: 42,
        CLASS_NAMES["RSS"]: 31,
    }

    actual_counts = (
        point_table[
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
                f"Expected {expected} "
                f"{disease} points, "
                f"found {actual}."
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

    return point_table


# ============================================================================
# CROSS-SERIES PHYSICAL PROXIMITY
# ============================================================================

def build_cross_series_pairs(
    point_table,
):
    """
    For every RFNN point, find the nearest point of every other disease
    within the SAME STUDY and SAME SPINAL LEVEL.

    Different series are allowed.

    Only patient physical coordinates are compared.
    """

    print()
    print("=" * 80)
    print("BUILDING CROSS-SERIES PHYSICAL PROXIMITY")
    print("=" * 80)

    rfnn = point_table[
        point_table["class_name"]
        == CLASS_NAMES["RFNN"]
    ].copy()

    pair_records = []

    for _, rfnn_row in rfnn.iterrows():

        study_id = str(
            rfnn_row["study_id"]
        )

        level = str(
            rfnn_row["level"]
        )

        for comparison_code in (
            COMPARISON_ORDER
        ):

            comparison_name = CLASS_NAMES[
                comparison_code
            ]

            candidates = point_table[
                (point_table["study_id"] == study_id)
                & (
                    point_table["level"]
                    == level
                )
                & (
                    point_table["class_name"]
                    == comparison_name
                )
            ].copy()

            if candidates.empty:

                pair_records.append(
                    {
                        "study_id": study_id,
                        "rfnn_series_id": str(
                            rfnn_row["series_id"]
                        ),
                        "level": level,
                        "rfnn_point_index": int(
                            rfnn_row["point_index"]
                        ),
                        "comparison_class": comparison_name,
                        "comparison_series_id": np.nan,
                        "comparison_point_index": np.nan,
                        "same_series": False,
                        "physical_distance": np.nan,
                    }
                )

                continue

            candidate_records = []

            for _, candidate in candidates.iterrows():

                distance = physical_distance(
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

                candidate_records.append(
                    {
                        "study_id": study_id,
                        "rfnn_series_id": str(
                            rfnn_row["series_id"]
                        ),
                        "level": level,
                        "rfnn_point_index": int(
                            rfnn_row["point_index"]
                        ),
                        "comparison_class": comparison_name,
                        "comparison_series_id": str(
                            candidate["series_id"]
                        ),
                        "comparison_point_index": int(
                            candidate["point_index"]
                        ),
                        "same_series": (
                            str(
                                candidate[
                                    "series_id"
                                ]
                            )
                            == str(
                                rfnn_row[
                                    "series_id"
                                ]
                            )
                        ),
                        "physical_distance": distance,
                    }
                )

            candidate_records = [
                record
                for record in candidate_records
                if np.isfinite(
                    record["physical_distance"]
                )
            ]

            if not candidate_records:

                pair_records.append(
                    {
                        "study_id": study_id,
                        "rfnn_series_id": str(
                            rfnn_row["series_id"]
                        ),
                        "level": level,
                        "rfnn_point_index": int(
                            rfnn_row["point_index"]
                        ),
                        "comparison_class": comparison_name,
                        "comparison_series_id": np.nan,
                        "comparison_point_index": np.nan,
                        "same_series": False,
                        "physical_distance": np.nan,
                    }
                )

                continue

            candidate_records.sort(
                key=lambda item: item[
                    "physical_distance"
                ]
            )

            pair_records.append(
                candidate_records[0]
            )

    pairs = pd.DataFrame(
        pair_records
    )

    expected_rows = 45 * 4

    if len(pairs) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"found {len(pairs)}."
        )

    print(
        f"RFNN points analyzed : {len(rfnn)}"
    )

    print(
        f"Comparison rows      : {len(pairs)}"
    )

    return pairs


# ============================================================================
# RADIUS ANALYSIS IN PHYSICAL SPACE
# ============================================================================

def add_physical_radius_flags(
    pairs,
):
    """
    Physical distances are in the same physical coordinate system as
    the Part 2.20B geometry transformation.

    We report 2 mm, 4 mm and 6 mm thresholds as physical-distance
    reference thresholds.

    These are NOT model voxel radii.
    """

    result = pairs.copy()

    for threshold in [
        2.0,
        4.0,
        6.0,
        10.0,
        15.0,
        20.0,
    ]:

        result[
            f"within_{int(threshold)}mm"
        ] = (
            result[
                "physical_distance"
            ]
            <= threshold
        )

    return result


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    pairs,
):
    print()
    print("=" * 80)
    print("BUILDING PART 2.44 SUMMARY")
    print("=" * 80)

    summary_rows = []

    for comparison_code in (
        COMPARISON_ORDER
    ):

        comparison_name = CLASS_NAMES[
            comparison_code
        ]

        subset = pairs[
            pairs["comparison_class"]
            == comparison_name
        ].copy()

        stats = summarize(
            subset[
                "physical_distance"
            ].tolist()
        )

        row = {
            "comparison_class": comparison_name,
            **stats,
        }

        for threshold in [
            2,
            4,
            6,
            10,
            15,
            20,
        ]:

            column = (
                f"within_{threshold}mm"
            )

            row[
                f"within_{threshold}mm_count"
            ] = int(
                subset[column].sum()
            )

            row[
                f"within_{threshold}mm_fraction"
            ] = float(
                subset[column].mean()
            )

        row[
            "same_series_count"
        ] = int(
            subset[
                "same_series"
            ].sum()
        )

        row[
            "different_series_count"
        ] = int(
            (
                ~subset[
                    "same_series"
                ]
            ).sum()
        )

        summary_rows.append(
            row
        )

    disease_summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================================
    # LEVEL SUMMARY
    # ========================================================================

    level_rows = []

    for level in LEVELS:

        for comparison_code in (
            COMPARISON_ORDER
        ):

            comparison_name = CLASS_NAMES[
                comparison_code
            ]

            subset = pairs[
                (pairs["level"] == level)
                & (
                    pairs["comparison_class"]
                    == comparison_name
                )
            ].copy()

            if subset.empty:
                continue

            stats = summarize(
                subset[
                    "physical_distance"
                ].tolist()
            )

            row = {
                "level": level,
                "comparison_class": comparison_name,
                **stats,
            }

            for threshold in [
                2,
                4,
                6,
                10,
                15,
                20,
            ]:

                row[
                    f"within_{threshold}mm_fraction"
                ] = float(
                    subset[
                        f"within_{threshold}mm"
                    ].mean()
                )

            level_rows.append(
                row
            )

    level_summary = pd.DataFrame(
        level_rows
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    pairs_path = (
        OUTPUT_DIR
        / "part244_cross_series_physical_pairs.csv"
    )

    disease_path = (
        OUTPUT_DIR
        / "part244_physical_proximity_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part244_level_physical_proximity_summary.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part244_cross_series_physical_proximity_audit_summary.json"
    )

    pairs.to_csv(
        pairs_path,
        index=False,
    )

    disease_summary.to_csv(
        disease_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    json_summary = {
        "validation_studies": 25,
        "validation_series": 25,
        "validation_points": 188,
        "rfnn_points": 45,
        "comparison_points": {
            "LFNN": 45,
            "LSS": 45,
            "RSS": 45,
            "SCS": 45,
        },
    }

    for comparison_code in (
        COMPARISON_ORDER
    ):

        name = CLASS_NAMES[
            comparison_code
        ]

        subset = disease_summary[
            disease_summary[
                "comparison_class"
            ]
            == name
        ]

        if subset.empty:
            continue

        row = subset.iloc[0]

        json_summary[
            comparison_code
        ] = {
            "mean_mm": safe_float(
                row["mean"]
            ),
            "median_mm": safe_float(
                row["median"]
            ),
            "min_mm": safe_float(
                row["min"]
            ),
            "max_mm": safe_float(
                row["max"]
            ),
            "within_6mm_fraction": safe_float(
                row[
                    "within_6mm_fraction"
                ]
            ),
            "within_10mm_fraction": safe_float(
                row[
                    "within_10mm_fraction"
                ]
            ),
            "within_20mm_fraction": safe_float(
                row[
                    "within_20mm_fraction"
                ]
            ),
        }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_summary,
            file,
            indent=2,
        )

    # ========================================================================
    # TERMINAL REPORT
    # ========================================================================

    print()
    print("=" * 80)
    print("CROSS-SERIES RFNN PHYSICAL PROXIMITY")
    print("=" * 80)

    print(
        disease_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("LEVEL-WISE PHYSICAL PROXIMITY")
    print("=" * 80)

    print(
        level_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.44 COMPLETE")
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
        disease_path
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
    print("PART 2.44")
    print("RFNN CROSS-SERIES PHYSICAL-SPACE PROXIMITY AUDIT")
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

    point_table = (
        build_exact_validation_cohort(
            manifest
        )
    )

    pairs = (
        build_cross_series_pairs(
            point_table
        )
    )

    pairs = (
        add_physical_radius_flags(
            pairs
        )
    )

    build_summary(
        pairs
    )


if __name__ == "__main__":
    main()