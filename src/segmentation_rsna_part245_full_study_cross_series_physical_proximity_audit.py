"""
PART 2.45
Full-Study Cross-Series Physical-Space Proximity Audit

Purpose
-------
Part 2.44 was limited to the 25 selected validation series.

Part 2.45 fixes that limitation.

We retain the EXACT 25 validation STUDIES selected by Part 2.20B,
but retrieve ALL annotated series belonging to those 25 studies.

For every RFNN annotation, we find the nearest annotation of:

    LFNN
    LSS
    RSS
    SCS

within:

    same study
    same spinal level

regardless of MRI series.

Distances are calculated only in patient physical coordinates.

This is ANALYSIS ONLY.

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
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part245_full_study_cross_series_physical_proximity_audit"
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

PHYSICAL_THRESHOLDS_MM = [
    2,
    4,
    6,
    10,
    15,
    20,
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
        np.isfinite(value)
        for value in values
    ):
        return np.nan

    return float(
        math.sqrt(
            (a[0] - b[0]) ** 2
            + (a[1] - b[1]) ** 2
            + (a[2] - b[2]) ** 2
        )
    )


def summarize(
    values,
):
    """
    Descriptive statistics for a numeric list.
    """

    values = [
        float(value)
        for value in values
        if np.isfinite(value)
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
# GET EXACT 25 VALIDATION STUDIES
# ============================================================================

def get_exact_validation_studies(
    manifest,
):
    """
    Reproduce the Part 2.20B validation selector.

    IMPORTANT:
    We keep the 25 STUDIES, but do NOT keep only the 25 selected series.
    """

    print()
    print("=" * 80)
    print("SELECTING EXACT PART 2.20B VALIDATION STUDIES")
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
            "Expected exactly 25 selected validation "
            f"series, found {len(selected)}."
        )

    validation_studies = sorted(
        selected[
            "study_id"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(validation_studies) != 25:

        raise RuntimeError(
            "Expected 25 validation studies, "
            f"found {len(validation_studies)}."
        )

    print()
    print(
        f"Selected validation series : "
        f"{len(selected)}"
    )

    print(
        f"Selected validation studies: "
        f"{len(validation_studies)}"
    )

    return validation_studies


# ============================================================================
# BUILD FULL ANNOTATION TABLE FOR THE 25 STUDIES
# ============================================================================

def build_full_study_manifest(
    manifest,
    validation_studies,
):
    """
    Retrieve ALL annotated manifest rows belonging to the 25 validation
    studies.

    We do NOT restrict by series_id here.
    """

    validation_study_set = set(
        str(study_id)
        for study_id in validation_studies
    )

    study_manifest = manifest[
        manifest[
            "study_id"
        ]
        .astype(str)
        .isin(
            validation_study_set
        )
    ].copy()

    study_manifest[
        "study_id"
    ] = study_manifest[
        "study_id"
    ].astype(str)

    study_manifest[
        "series_id"
    ] = study_manifest[
        "series_id"
    ].astype(str)

    print()
    print("=" * 80)
    print("FULL STUDY ANNOTATION MANIFEST")
    print("=" * 80)

    print(
        f"Validation studies          : "
        f"{study_manifest['study_id'].nunique()}"
    )

    print(
        f"Annotated series            : "
        f"{study_manifest['series_id'].nunique()}"
    )

    print(
        f"Annotation rows              : "
        f"{len(study_manifest)}"
    )

    disease_counts = (
        study_manifest[
            "class_name"
        ]
        .value_counts()
    )

    print()
    print("Disease annotation counts:")

    for code in [
        "SCS",
        "LFNN",
        "RFNN",
        "LSS",
        "RSS",
    ]:

        disease_name = CLASS_NAMES[
            code
        ]

        print(
            f"  {code:5s} "
            f"{disease_name:40s}: "
            f"{int(disease_counts.get(disease_name, 0))}"
        )

    return study_manifest


# ============================================================================
# GEOMETRY TRANSFORM ALL ANNOTATED SERIES
# ============================================================================

def build_full_physical_point_table(
    study_manifest,
):
    """
    Transform every annotated series belonging to the 25 validation studies.

    The Part 2.20B load_case() function performs the canonical geometry
    conversion and returns patient_x/y/z for every annotation.
    """

    print()
    print("=" * 80)
    print("BUILDING FULL PATIENT-PHYSICAL ANNOTATION TABLE")
    print("=" * 80)

    unique_series = (
        study_manifest[
            [
                "study_id",
                "series_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "study_id",
                "series_id",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Unique annotated series to process: "
        f"{len(unique_series)}"
    )

    all_records = []

    failed_series = []

    for series_number, (_, series_row) in enumerate(
        unique_series.iterrows(),
        start=1,
    ):

        study_id = str(
            series_row["study_id"]
        )

        series_id = str(
            series_row["series_id"]
        )

        series_points = study_manifest[
            (
                study_manifest[
                    "study_id"
                ]
                == study_id
            )
            & (
                study_manifest[
                    "series_id"
                ]
                == series_id
            )
        ].copy()

        series_points = (
            series_points
            .reset_index(drop=True)
        )

        try:

            (
                image,
                transformed_points,
                geometry,
            ) = part220b.load_case(
                study_id,
                series_id,
                series_points,
            )

            if len(series_points) != len(
                transformed_points
            ):

                raise RuntimeError(
                    "Original/transformed point "
                    "count mismatch."
                )

            for point_index, point in enumerate(
                transformed_points
            ):

                source = series_points.iloc[
                    point_index
                ]

                all_records.append(
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

            print(
                f"[{series_number:04d}/"
                f"{len(unique_series):04d}] "
                f"{study_id} / {series_id} "
                f"-> {len(transformed_points)} points"
            )

        except Exception as exc:

            failed_series.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "error": str(exc),
                }
            )

            print(
                f"[{series_number:04d}/"
                f"{len(unique_series):04d}] "
                f"{study_id} / {series_id} "
                f"FAILED: {exc}"
            )

    point_table = pd.DataFrame(
        all_records
    )

    failed_table = pd.DataFrame(
        failed_series
    )

    print()
    print(
        f"Successfully transformed points: "
        f"{len(point_table)}"
    )

    print(
        f"Failed series: "
        f"{len(failed_table)}"
    )

    if len(point_table) == 0:

        raise RuntimeError(
            "No transformed points were produced."
        )

    return (
        point_table,
        failed_table,
    )


# ============================================================================
# CROSS-SERIES PHYSICAL PROXIMITY
# ============================================================================

def build_rfnn_proximity_pairs(
    point_table,
):
    """
    For every RFNN annotation, find the nearest annotation of each
    comparison disease in the SAME STUDY + SAME LEVEL.

    Series may be different.

    Distance is measured in patient physical coordinates.
    """

    print()
    print("=" * 80)
    print("BUILDING FULL-STUDY RFNN PHYSICAL PROXIMITY PAIRS")
    print("=" * 80)

    rfnn = point_table[
        point_table[
            "class_name"
        ]
        == CLASS_NAMES["RFNN"]
    ].copy()

    if len(rfnn) == 0:

        raise RuntimeError(
            "No RFNN points found."
        )

    pair_records = []

    for rfnn_number, (_, rfnn_row) in enumerate(
        rfnn.iterrows(),
        start=1,
    ):

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
                (
                    point_table[
                        "study_id"
                    ]
                    == study_id
                )
                & (
                    point_table[
                        "level"
                    ]
                    == level
                )
                & (
                    point_table[
                        "class_name"
                    ]
                    == comparison_name
                )
            ].copy()

            candidate_records = []

            for _, candidate in candidates.iterrows():

                distance = physical_distance(
                    (
                        rfnn_row[
                            "patient_x"
                        ],
                        rfnn_row[
                            "patient_y"
                        ],
                        rfnn_row[
                            "patient_z"
                        ],
                    ),
                    (
                        candidate[
                            "patient_x"
                        ],
                        candidate[
                            "patient_y"
                        ],
                        candidate[
                            "patient_z"
                        ],
                    ),
                )

                if not np.isfinite(
                    distance
                ):
                    continue

                candidate_records.append(
                    {
                        "study_id": study_id,
                        "rfnn_series_id": str(
                            rfnn_row[
                                "series_id"
                            ]
                        ),
                        "rfnn_point_index": int(
                            rfnn_row[
                                "point_index"
                            ]
                        ),
                        "level": level,
                        "comparison_class": comparison_name,
                        "comparison_series_id": str(
                            candidate[
                                "series_id"
                            ]
                        ),
                        "comparison_point_index": int(
                            candidate[
                                "point_index"
                            ]
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
                        "physical_distance_mm": float(
                            distance
                        ),
                    }
                )

            if candidate_records:

                candidate_records.sort(
                    key=lambda item:
                    item[
                        "physical_distance_mm"
                    ]
                )

                pair_records.append(
                    candidate_records[0]
                )

            else:

                pair_records.append(
                    {
                        "study_id": study_id,
                        "rfnn_series_id": str(
                            rfnn_row[
                                "series_id"
                            ]
                        ),
                        "rfnn_point_index": int(
                            rfnn_row[
                                "point_index"
                            ]
                        ),
                        "level": level,
                        "comparison_class": comparison_name,
                        "comparison_series_id": np.nan,
                        "comparison_point_index": np.nan,
                        "same_series": False,
                        "physical_distance_mm": np.nan,
                    }
                )

        if (
            rfnn_number == 1
            or rfnn_number % 10 == 0
            or rfnn_number == len(rfnn)
        ):

            print(
                f"Processed RFNN points: "
                f"{rfnn_number}/{len(rfnn)}"
            )

    pairs = pd.DataFrame(
        pair_records
    )

    expected_rows = (
        len(rfnn)
        * len(COMPARISON_ORDER)
    )

    if len(pairs) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows} "
            f"pair rows, found {len(pairs)}."
        )

    return pairs


# ============================================================================
# ADD THRESHOLD FLAGS
# ============================================================================

def add_threshold_flags(
    pairs,
):

    result = pairs.copy()

    for threshold in PHYSICAL_THRESHOLDS_MM:

        result[
            f"within_{threshold}mm"
        ] = (
            result[
                "physical_distance_mm"
            ]
            <= float(threshold)
        )

    return result


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    point_table,
    pairs,
    failed_series,
):
    print()
    print("=" * 80)
    print("BUILDING PART 2.45 SUMMARY")
    print("=" * 80)

    disease_rows = []

    for comparison_code in (
        COMPARISON_ORDER
    ):

        comparison_name = CLASS_NAMES[
            comparison_code
        ]

        subset = pairs[
            pairs[
                "comparison_class"
            ]
            == comparison_name
        ].copy()

        stats = summarize(
            subset[
                "physical_distance_mm"
            ].tolist()
        )

        row = {
            "comparison_class": comparison_name,
            **stats,
        }

        for threshold in PHYSICAL_THRESHOLDS_MM:

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

        disease_rows.append(
            row
        )

    disease_summary = pd.DataFrame(
        disease_rows
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
                (
                    pairs[
                        "level"
                    ]
                    == level
                )
                & (
                    pairs[
                        "comparison_class"
                    ]
                    == comparison_name
                )
            ].copy()

            if subset.empty:
                continue

            stats = summarize(
                subset[
                    "physical_distance_mm"
                ].tolist()
            )

            row = {
                "level": level,
                "comparison_class": comparison_name,
                **stats,
            }

            for threshold in PHYSICAL_THRESHOLDS_MM:

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
    # SERIES TYPE SUMMARY
    # ========================================================================

    series_rows = []

    for comparison_code in (
        COMPARISON_ORDER
    ):

        comparison_name = CLASS_NAMES[
            comparison_code
        ]

        subset = pairs[
            pairs[
                "comparison_class"
            ]
            == comparison_name
        ].copy()

        if subset.empty:
            continue

        series_rows.append(
            {
                "comparison_class": comparison_name,
                "same_series_count": int(
                    subset[
                        "same_series"
                    ].sum()
                ),
                "different_series_count": int(
                    (
                        ~subset[
                            "same_series"
                        ]
                    ).sum()
                ),
                "same_series_fraction": float(
                    subset[
                        "same_series"
                    ].mean()
                ),
            }
        )

    series_summary = pd.DataFrame(
        series_rows
    )

    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================

    pairs_path = (
        OUTPUT_DIR
        / "part245_full_study_rfnn_physical_pairs.csv"
    )

    disease_path = (
        OUTPUT_DIR
        / "part245_full_study_physical_proximity_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part245_full_study_level_proximity_summary.csv"
    )

    series_path = (
        OUTPUT_DIR
        / "part245_same_vs_different_series_summary.csv"
    )

    failed_path = (
        OUTPUT_DIR
        / "part245_failed_series.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part245_full_study_physical_proximity_audit_summary.json"
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

    series_summary.to_csv(
        series_path,
        index=False,
    )

    failed_series.to_csv(
        failed_path,
        index=False,
    )

    json_summary = {
        "validation_studies": int(
            point_table[
                "study_id"
            ].nunique()
        ),
        "annotated_series": int(
            point_table[
                [
                    "study_id",
                    "series_id",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "transformed_points": int(
            len(point_table)
        ),
        "rfnn_points": int(
            (
                point_table[
                    "class_name"
                ]
                == CLASS_NAMES["RFNN"]
            ).sum()
        ),
        "failed_series": int(
            len(failed_series)
        ),
    }

    for comparison_code in (
        COMPARISON_ORDER
    ):

        comparison_name = CLASS_NAMES[
            comparison_code
        ]

        subset = disease_summary[
            disease_summary[
                "comparison_class"
            ]
            == comparison_name
        ]

        if subset.empty:
            continue

        row = subset.iloc[0]

        json_summary[
            comparison_code
        ] = {
            "count": int(
                row["count"]
            ),
            "mean_mm": safe_float(
                row["mean"]
            ),
            "median_mm": safe_float(
                row["median"]
            ),
            "std_mm": safe_float(
                row["std"]
            ),
            "min_mm": safe_float(
                row["min"]
            ),
            "max_mm": safe_float(
                row["max"]
            ),
            "within_2mm_fraction": safe_float(
                row[
                    "within_2mm_fraction"
                ]
            ),
            "within_4mm_fraction": safe_float(
                row[
                    "within_4mm_fraction"
                ]
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
            "same_series_count": int(
                row[
                    "same_series_count"
                ]
            ),
            "different_series_count": int(
                row[
                    "different_series_count"
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
    print("FULL-STUDY RFNN PHYSICAL PROXIMITY")
    print("=" * 80)

    print(
        disease_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("SAME SERIES vs DIFFERENT SERIES")
    print("=" * 80)

    print(
        series_summary.to_string(
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
    print("TRANSFORMATION QUALITY")
    print("=" * 80)

    print(
        f"Validation studies : "
        f"{point_table['study_id'].nunique()}"
    )

    print(
        f"Annotated series   : "
        f"{point_table[['study_id', 'series_id']].drop_duplicates().shape[0]}"
    )

    print(
        f"Transformed points : "
        f"{len(point_table)}"
    )

    print(
        f"Failed series      : "
        f"{len(failed_series)}"
    )

    print()
    print("=" * 80)
    print("PART 2.45 COMPLETE")
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
        series_path
    )

    print(
        failed_path
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
    print("PART 2.45")
    print("FULL-STUDY CROSS-SERIES PHYSICAL-SPACE PROXIMITY AUDIT")
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

    validation_studies = (
        get_exact_validation_studies(
            manifest
        )
    )

    study_manifest = (
        build_full_study_manifest(
            manifest,
            validation_studies,
        )
    )

    (
        point_table,
        failed_series,
    ) = build_full_physical_point_table(
        study_manifest
    )

    pairs = (
        build_rfnn_proximity_pairs(
            point_table
        )
    )

    pairs = add_threshold_flags(
        pairs
    )

    build_summary(
        point_table,
        pairs,
        failed_series,
    )


if __name__ == "__main__":
    main()