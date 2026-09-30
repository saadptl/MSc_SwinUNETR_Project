"""
PART 2.52
DICOM PHYSICAL-AXIS DISPLACEMENT AUDIT

Purpose
-------
Decompose the Part 2.51 physical displacement vectors into the
actual DICOM acquisition directions:

    1. Row direction
    2. Column direction
    3. Slice-normal direction

This avoids interpreting patient X/Y/Z directly as anatomical
directions.

Input:
    Part 2.51 physical evidence records

Expected:
    45 RFNN points
    5 criteria per RFNN point
    225 total records

Analysis only.

No training.
No checkpoint modification.
No dashboard modification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

PART251_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part251_physical_space_evidence_attribution"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part252_dicom_axis_displacement_audit"
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
# INPUT
# ============================================================================

PART251_RECORDS = (
    PART251_DIR
    / "part251_physical_evidence_records.csv"
)


# ============================================================================
# EXPECTED CRITERIA
# ============================================================================

EXPECTED_CRITERIA = {
    "max_rfnn_probability",
    "max_rfnn_logit",
    "max_rfnn_minus_background",
    "max_rfnn_minus_lss",
    "max_rfnn_minus_scs",
}


# ============================================================================
# VECTOR HELPERS
# ============================================================================

def normalize_vector(
    vector,
):

    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        vector
    )

    if norm == 0:

        raise RuntimeError(
            "Cannot normalize a zero vector."
        )

    return vector / norm


def get_geometry_axes(
    geometry,
):

    required = [
        "row_direction",
        "column_direction",
        "normal_direction",
    ]

    for key in required:

        if key not in geometry:

            raise RuntimeError(
                f"Geometry is missing required field: "
                f"{key}"
            )

    row_direction = normalize_vector(
        geometry["row_direction"]
    )

    column_direction = normalize_vector(
        geometry["column_direction"]
    )

    normal_direction = normalize_vector(
        geometry["normal_direction"]
    )

    return (
        row_direction,
        column_direction,
        normal_direction,
    )


# ============================================================================
# LOAD PART 2.51
# ============================================================================

def load_part251_records():

    print()
    print("=" * 80)
    print("LOADING PART 2.51 RECORDS")
    print("=" * 80)

    if not PART251_RECORDS.exists():

        raise FileNotFoundError(
            "Part 2.51 records file not found:\n"
            f"{PART251_RECORDS}"
        )

    df = pd.read_csv(
        PART251_RECORDS
    )

    print(
        f"Part 2.51 records: {len(df)}"
    )

    # ------------------------------------------------------------------------
    # Expected total
    # ------------------------------------------------------------------------

    if len(df) != 225:

        raise RuntimeError(
            "Expected 225 Part 2.51 records "
            f"but found {len(df)}."
        )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Part 2.51 contains only RFNN evidence records.
    # It does NOT need a class_name column.
    # ------------------------------------------------------------------------

    required_columns = {
        "study_id",
        "series_id",
        "point_index",
        "level",
        "criterion",
        "physical_dx_mm",
        "physical_dy_mm",
        "physical_dz_mm",
        "physical_distance_mm",
        "winner_class",
        "rfnn_probability",
        "background_probability",
        "lss_probability",
        "scs_probability",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "Part 2.51 CSV is missing required columns:\n"
            + "\n".join(
                sorted(
                    missing_columns
                )
            )
        )

    # ------------------------------------------------------------------------
    # Validate criteria
    # ------------------------------------------------------------------------

    actual_criteria = set(
        df["criterion"].unique()
    )

    if actual_criteria != EXPECTED_CRITERIA:

        raise RuntimeError(
            "Unexpected Part 2.51 criteria.\n"
            f"Expected: {sorted(EXPECTED_CRITERIA)}\n"
            f"Found:    {sorted(actual_criteria)}"
        )

    # ------------------------------------------------------------------------
    # Validate unique RFNN points
    # ------------------------------------------------------------------------

    point_keys = (
        df[
            [
                "study_id",
                "series_id",
                "point_index",
            ]
        ]
        .drop_duplicates()
    )

    if len(point_keys) != 45:

        raise RuntimeError(
            "Expected 45 unique RFNN points "
            f"but found {len(point_keys)}."
        )

    # ------------------------------------------------------------------------
    # Every RFNN point must contain exactly five criteria.
    # ------------------------------------------------------------------------

    point_counts = (
        df.groupby(
            [
                "study_id",
                "series_id",
                "point_index",
            ]
        )["criterion"]
        .nunique()
    )

    if not (
        point_counts == 5
    ).all():

        bad_points = (
            point_counts[
                point_counts != 5
            ]
        )

        raise RuntimeError(
            "Some RFNN points do not contain "
            "exactly five criteria:\n"
            f"{bad_points.to_string()}"
        )

    print(
        "Unique RFNN points: "
        f"{len(point_keys)}"
    )

    print(
        "Criteria per RFNN point: 5"
    )

    print(
        "Expected records: "
        "45 × 5 = 225"
    )

    return df


# ============================================================================
# LOAD VALIDATION GEOMETRY
# ============================================================================

def load_validation_geometry():

    print()
    print("=" * 80)
    print("LOADING EXACT PART 2.20B VALIDATION GEOMETRY")
    print("=" * 80)

    manifest = (
        part220b.load_manifest()
    )

    print(
        f"Full manifest rows: "
        f"{len(manifest)}"
    )

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
            "Part 2.20B validation selection "
            "must return a DataFrame."
        )

    if len(selected) != 25:

        raise RuntimeError(
            "Expected 25 validation series "
            f"but found {len(selected)}."
        )

    geometry_map = {}

    for case_number, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):

        study_id = str(
            row["study_id"]
        )

        series_id = str(
            row["series_id"]
        )

        key = (
            study_id,
            series_id,
        )

        # --------------------------------------------------------------------
        # Find one manifest row for this series.
        # Geometry is series-level.
        # --------------------------------------------------------------------

        series_manifest = manifest[
            (
                manifest["study_id"]
                .astype(str)
                == study_id
            )
            &
            (
                manifest["series_id"]
                .astype(str)
                == series_id
            )
        ].copy()

        if series_manifest.empty:

            raise RuntimeError(
                f"No manifest rows found for "
                f"{study_id}/{series_id}"
            )

        first_row = (
            series_manifest
            .iloc[0]
        )

        point_df = pd.DataFrame(
            [first_row]
        )

        (
            case_image,
            case_points,
            geometry,
        ) = part220b.load_case(
            study_id,
            series_id,
            point_df,
        )

        # Prevent unused-object warnings.
        del case_image
        del case_points

        # --------------------------------------------------------------------
        # Validate geometry fields.
        # --------------------------------------------------------------------

        get_geometry_axes(
            geometry
        )

        geometry_map[key] = geometry

        print(
            f"[{case_number:02d}/25] "
            f"Study={study_id} | "
            f"Series={series_id}"
        )

    if len(
        geometry_map
    ) != 25:

        raise RuntimeError(
            "Expected geometry for all 25 validation series "
            f"but found {len(geometry_map)}."
        )

    print()
    print(
        f"Validation geometries loaded: "
        f"{len(geometry_map)}/25"
    )

    return geometry_map


# ============================================================================
# DICOM AXIS DECOMPOSITION
# ============================================================================

def decompose_displacement(
    displacement,
    geometry,
):

    (
        row_direction,
        column_direction,
        normal_direction,
    ) = get_geometry_axes(
        geometry
    )

    displacement = np.asarray(
        displacement,
        dtype=np.float64,
    )

    row_component = float(
        np.dot(
            displacement,
            row_direction,
        )
    )

    column_component = float(
        np.dot(
            displacement,
            column_direction,
        )
    )

    normal_component = float(
        np.dot(
            displacement,
            normal_direction,
        )
    )

    return (
        row_component,
        column_component,
        normal_component,
    )


# ============================================================================
# ANALYZE RECORDS
# ============================================================================

def analyze_records(
    records,
    geometry_map,
):

    output_records = []

    for _, row in records.iterrows():

        study_id = str(
            row["study_id"]
        )

        series_id = str(
            row["series_id"]
        )

        key = (
            study_id,
            series_id,
        )

        if key not in geometry_map:

            raise RuntimeError(
                f"Missing geometry for "
                f"{study_id}/{series_id}"
            )

        geometry = geometry_map[
            key
        ]

        # --------------------------------------------------------------------
        # Patient-space displacement vector.
        #
        # These are physical coordinates from Part 2.51:
        #
        #       [patient X, patient Y, patient Z]
        #
        # --------------------------------------------------------------------

        displacement = np.array(
            [
                float(
                    row["physical_dx_mm"]
                ),
                float(
                    row["physical_dy_mm"]
                ),
                float(
                    row["physical_dz_mm"]
                ),
            ],
            dtype=np.float64,
        )

        (
            row_component,
            column_component,
            normal_component,
        ) = decompose_displacement(
            displacement,
            geometry,
        )

        magnitude = float(
            np.linalg.norm(
                displacement
            )
        )

        row_abs = abs(
            row_component
        )

        column_abs = abs(
            column_component
        )

        normal_abs = abs(
            normal_component
        )

        # --------------------------------------------------------------------
        # Determine dominant physical DICOM axis.
        # --------------------------------------------------------------------

        axis_values = {
            "row": row_abs,
            "column": column_abs,
            "slice_normal": normal_abs,
        }

        dominant_axis = max(
            axis_values,
            key=axis_values.get,
        )

        # --------------------------------------------------------------------
        # DICOM spacing.
        # --------------------------------------------------------------------

        row_spacing = float(
            geometry[
                "row_spacing"
            ]
        )

        column_spacing = float(
            geometry[
                "column_spacing"
            ]
        )

        slice_spacing = float(
            geometry[
                "slice_spacing"
            ]
        )

        # --------------------------------------------------------------------
        # Fractions of total displacement magnitude.
        # --------------------------------------------------------------------

        if magnitude > 0:

            row_fraction = (
                row_abs
                / magnitude
            )

            column_fraction = (
                column_abs
                / magnitude
            )

            normal_fraction = (
                normal_abs
                / magnitude
            )

        else:

            row_fraction = 0.0
            column_fraction = 0.0
            normal_fraction = 0.0

        output_records.append(
            {
                "study_id":
                    study_id,

                "series_id":
                    series_id,

                "point_index":
                    int(
                        row["point_index"]
                    ),

                "level":
                    str(
                        row["level"]
                    ),

                "criterion":
                    str(
                        row["criterion"]
                    ),

                # ------------------------------------------------------------
                # Original patient-space vector
                # ------------------------------------------------------------

                "physical_dx_mm":
                    float(
                        displacement[0]
                    ),

                "physical_dy_mm":
                    float(
                        displacement[1]
                    ),

                "physical_dz_mm":
                    float(
                        displacement[2]
                    ),

                "physical_distance_mm":
                    magnitude,

                # ------------------------------------------------------------
                # DICOM-axis components
                # ------------------------------------------------------------

                "row_component_mm":
                    row_component,

                "column_component_mm":
                    column_component,

                "slice_normal_component_mm":
                    normal_component,

                "abs_row_component_mm":
                    row_abs,

                "abs_column_component_mm":
                    column_abs,

                "abs_slice_normal_component_mm":
                    normal_abs,

                # ------------------------------------------------------------
                # DICOM spacing
                # ------------------------------------------------------------

                "row_spacing_mm":
                    row_spacing,

                "column_spacing_mm":
                    column_spacing,

                "slice_spacing_mm":
                    slice_spacing,

                # ------------------------------------------------------------
                # Relative contributions
                # ------------------------------------------------------------

                "row_fraction":
                    row_fraction,

                "column_fraction":
                    column_fraction,

                "normal_fraction":
                    normal_fraction,

                "dominant_axis":
                    dominant_axis,

                # ------------------------------------------------------------
                # Model evidence
                # ------------------------------------------------------------

                "winner_class":
                    str(
                        row["winner_class"]
                    ),

                "rfnn_probability":
                    float(
                        row["rfnn_probability"]
                    ),

                "background_probability":
                    float(
                        row["background_probability"]
                    ),

                "lss_probability":
                    float(
                        row["lss_probability"]
                    ),

                "scs_probability":
                    float(
                        row["scs_probability"]
                    ),
            }
        )

    result = pd.DataFrame(
        output_records
    )

    return result


# ============================================================================
# GLOBAL SUMMARY
# ============================================================================

def build_global_summary(
    df,
):

    rows = []

    for criterion in sorted(
        df["criterion"].unique()
    ):

        subset = df[
            df["criterion"]
            == criterion
        ]

        distance = subset[
            "physical_distance_mm"
        ].astype(float)

        row_abs = subset[
            "abs_row_component_mm"
        ].astype(float)

        column_abs = subset[
            "abs_column_component_mm"
        ].astype(float)

        normal_abs = subset[
            "abs_slice_normal_component_mm"
        ].astype(float)

        dominant_counts = (
            subset[
                "dominant_axis"
            ]
            .value_counts()
        )

        rows.append(
            {
                "criterion":
                    criterion,

                "count":
                    len(subset),

                "mean_distance_mm":
                    float(
                        distance.mean()
                    ),

                "median_distance_mm":
                    float(
                        distance.median()
                    ),

                "min_distance_mm":
                    float(
                        distance.min()
                    ),

                "max_distance_mm":
                    float(
                        distance.max()
                    ),

                "mean_abs_row_mm":
                    float(
                        row_abs.mean()
                    ),

                "mean_abs_column_mm":
                    float(
                        column_abs.mean()
                    ),

                "mean_abs_slice_normal_mm":
                    float(
                        normal_abs.mean()
                    ),

                "median_abs_slice_normal_mm":
                    float(
                        normal_abs.median()
                    ),

                "mean_normal_fraction":
                    float(
                        subset[
                            "normal_fraction"
                        ].mean()
                    ),

                "median_normal_fraction":
                    float(
                        subset[
                            "normal_fraction"
                        ].median()
                    ),

                "dominant_row":
                    int(
                        dominant_counts.get(
                            "row",
                            0,
                        )
                    ),

                "dominant_column":
                    int(
                        dominant_counts.get(
                            "column",
                            0,
                        )
                    ),

                "dominant_slice_normal":
                    int(
                        dominant_counts.get(
                            "slice_normal",
                            0,
                        )
                    ),

                "mean_row_component_mm":
                    float(
                        subset[
                            "row_component_mm"
                        ].mean()
                    ),

                "mean_column_component_mm":
                    float(
                        subset[
                            "column_component_mm"
                        ].mean()
                    ),

                "mean_slice_normal_component_mm":
                    float(
                        subset[
                            "slice_normal_component_mm"
                        ].mean()
                    ),

                "negative_normal_fraction":
                    float(
                        (
                            subset[
                                "slice_normal_component_mm"
                            ]
                            < 0
                        ).mean()
                    ),

                "positive_normal_fraction":
                    float(
                        (
                            subset[
                                "slice_normal_component_mm"
                            ]
                            > 0
                        ).mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# LEVEL SUMMARY
# ============================================================================

def build_level_summary(
    df,
):

    rows = []

    for level in sorted(
        df["level"].unique()
    ):

        for criterion in sorted(
            df["criterion"].unique()
        ):

            subset = df[
                (
                    df["level"]
                    == level
                )
                &
                (
                    df["criterion"]
                    == criterion
                )
            ]

            if subset.empty:

                continue

            distance = subset[
                "physical_distance_mm"
            ].astype(float)

            rows.append(
                {
                    "level":
                        level,

                    "criterion":
                        criterion,

                    "count":
                        len(subset),

                    "mean_distance_mm":
                        float(
                            distance.mean()
                        ),

                    "median_distance_mm":
                        float(
                            distance.median()
                        ),

                    "mean_abs_row_mm":
                        float(
                            subset[
                                "abs_row_component_mm"
                            ].mean()
                        ),

                    "mean_abs_column_mm":
                        float(
                            subset[
                                "abs_column_component_mm"
                            ].mean()
                        ),

                    "mean_abs_slice_normal_mm":
                        float(
                            subset[
                                "abs_slice_normal_component_mm"
                            ].mean()
                        ),

                    "mean_normal_fraction":
                        float(
                            subset[
                                "normal_fraction"
                            ].mean()
                        ),

                    "dominant_slice_normal":
                        int(
                            (
                                subset[
                                    "dominant_axis"
                                ]
                                == "slice_normal"
                            ).sum()
                        ),

                    "dominant_row":
                        int(
                            (
                                subset[
                                    "dominant_axis"
                                ]
                                == "row"
                            ).sum()
                        ),

                    "dominant_column":
                        int(
                            (
                                subset[
                                    "dominant_axis"
                                ]
                                == "column"
                            ).sum()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# WINNER SUMMARY
# ============================================================================

def build_winner_summary(
    df,
):

    rows = []

    for criterion in sorted(
        df["criterion"].unique()
    ):

        subset = df[
            df["criterion"]
            == criterion
        ]

        for winner in sorted(
            subset[
                "winner_class"
            ].unique()
        ):

            winner_subset = subset[
                subset[
                    "winner_class"
                ]
                == winner
            ]

            rows.append(
                {
                    "criterion":
                        criterion,

                    "winner_class":
                        winner,

                    "count":
                        len(
                            winner_subset
                        ),

                    "fraction":
                        float(
                            len(
                                winner_subset
                            )
                            / len(subset)
                        ),

                    "mean_distance_mm":
                        float(
                            winner_subset[
                                "physical_distance_mm"
                            ].mean()
                        ),

                    "mean_abs_row_mm":
                        float(
                            winner_subset[
                                "abs_row_component_mm"
                            ].mean()
                        ),

                    "mean_abs_column_mm":
                        float(
                            winner_subset[
                                "abs_column_component_mm"
                            ].mean()
                        ),

                    "mean_abs_slice_normal_mm":
                        float(
                            winner_subset[
                                "abs_slice_normal_component_mm"
                            ].mean()
                        ),

                    "mean_normal_fraction":
                        float(
                            winner_subset[
                                "normal_fraction"
                            ].mean()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# SAVE + PRINT
# ============================================================================

def save_outputs(
    records_df,
    global_df,
    level_df,
    winner_df,
):

    records_path = (
        OUTPUT_DIR
        / "part252_dicom_axis_displacement_records.csv"
    )

    global_path = (
        OUTPUT_DIR
        / "part252_dicom_axis_displacement_global_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part252_dicom_axis_displacement_level_summary.csv"
    )

    winner_path = (
        OUTPUT_DIR
        / "part252_dicom_axis_displacement_winner_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part252_dicom_axis_displacement_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    global_df.to_csv(
        global_path,
        index=False,
    )

    level_df.to_csv(
        level_path,
        index=False,
    )

    winner_df.to_csv(
        winner_path,
        index=False,
    )

    json_summary = {
        "records":
            int(
                len(records_df)
            ),

        "unique_rfnn_points":
            int(
                records_df[
                    [
                        "study_id",
                        "series_id",
                        "point_index",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),

        "criteria":
            sorted(
                records_df[
                    "criterion"
                ]
                .unique()
                .tolist()
            ),

        "global_summary":
            global_df.to_dict(
                orient="records"
            ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_summary,
            file,
            indent=2,
        )

    # ========================================================================
    # PRINT GLOBAL
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.52 GLOBAL DICOM-AXIS DISPLACEMENT")
    print("=" * 80)

    print(
        global_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # PRINT WINNER
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.52 WINNER / DICOM-AXIS SUMMARY")
    print("=" * 80)

    print(
        winner_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # PRINT LEVEL
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.52 LEVEL ANALYSIS")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # OUTPUT PATHS
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.52 COMPLETE")
    print("=" * 80)

    print(
        "Analysis only."
    )

    print(
        "Part 2.27 checkpoint unchanged."
    )

    print(
        "Part 2.51 unchanged."
    )

    print(
        "Dashboard unchanged."
    )

    print()
    print("Outputs:")

    print(records_path)
    print(global_path)
    print(level_path)
    print(winner_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.52")
    print("DICOM PHYSICAL-AXIS DISPLACEMENT AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    # ------------------------------------------------------------------------
    # Part 2.51
    # ------------------------------------------------------------------------

    records = (
        load_part251_records()
    )

    # ------------------------------------------------------------------------
    # Exact validation geometry
    # ------------------------------------------------------------------------

    geometry_map = (
        load_validation_geometry()
    )

    # ------------------------------------------------------------------------
    # Decompose
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("DECOMPOSING PHYSICAL DISPLACEMENTS")
    print("=" * 80)

    result_df = analyze_records(
        records,
        geometry_map,
    )

    if len(result_df) != 225:

        raise RuntimeError(
            "Expected 225 Part 2.52 records "
            f"but generated {len(result_df)}."
        )

    unique_points = (
        result_df[
            [
                "study_id",
                "series_id",
                "point_index",
            ]
        ]
        .drop_duplicates()
    )

    if len(unique_points) != 45:

        raise RuntimeError(
            "Expected 45 unique RFNN points "
            f"but generated {len(unique_points)}."
        )

    print(
        f"Generated records: "
        f"{len(result_df)}"
    )

    print(
        f"Unique RFNN points: "
        f"{len(unique_points)}"
    )

    # ------------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------------

    global_df = (
        build_global_summary(
            result_df
        )
    )

    level_df = (
        build_level_summary(
            result_df
        )
    )

    winner_df = (
        build_winner_summary(
            result_df
        )
    )

    save_outputs(
        result_df,
        global_df,
        level_df,
        winner_df,
    )


if __name__ == "__main__":
    main()