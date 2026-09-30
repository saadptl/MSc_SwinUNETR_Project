"""
PART 2.50
GEOMETRY ROUND-TRIP VALIDATION

Purpose
-------
Validate that the Part 2.20B physical-space geometry transformation
is internally consistent.

For each validation point:

    patient point
        ->
    patient_point_to_canonical()
        ->
    canonical point
        ->
    numerical inverse
        ->
    reconstructed patient point

Then calculate:

    physical round-trip error in mm
    canonical round-trip error

This is an ANALYSIS-ONLY experiment.

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part250_geometry_roundtrip_validation"
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

MAX_ITERATIONS = 20

CANONICAL_TOLERANCE = 1e-4

FINITE_DIFFERENCE_MM = 0.5

MAX_STEP_MM = 10.0

DAMPING = 0.8


# ============================================================================
# HELPERS
# ============================================================================

def as_float_array(
    values,
):

    return np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)


# ============================================================================
# EXACT VALIDATION COHORT
# ============================================================================

def build_validation_cases(
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
            "Expected validation selection "
            "to be a DataFrame."
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
            )
            in validation_keys,
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
        key=lambda case:
        case_order.get(
            (
                str(case["study_id"]),
                str(case["series_id"]),
            ),
            10000,
        ),
    )

    if len(validation_cases) != 25:

        raise RuntimeError(
            "Validation case count mismatch."
        )

    print()
    print("EXACT COHORT VERIFIED")
    print("-" * 80)
    print("Validation series : 25")
    print("Validation studies: 25")
    print("-" * 80)

    return validation_cases


# ============================================================================
# FORWARD TRANSFORMATION
# ============================================================================

def patient_to_canonical(
    patient_point,
    geometry,
):

    result = (
        part220b.patient_point_to_canonical(
            patient_point,
            geometry,
        )
    )

    result = as_float_array(
        result
    )

    if result.size < 3:

        raise RuntimeError(
            "patient_point_to_canonical() "
            "did not return three coordinates."
        )

    return result[:3]


# ============================================================================
# NUMERICAL INVERSE
# ============================================================================

def canonical_to_patient(
    target_canonical,
    initial_patient,
    geometry,
):
    """
    Numerically invert the exact Part 2.20B
    patient_point_to_canonical() transformation.

    Unknown:
        patient X,Y,Z

    Target:
        canonical Z,Y,X
    """

    target = as_float_array(
        target_canonical
    )[:3]

    patient = as_float_array(
        initial_patient
    )[:3]

    def forward(
        point,
    ):

        return patient_to_canonical(
            point,
            geometry,
        )

    converged = False

    iteration_count = 0

    for iteration in range(
        MAX_ITERATIONS
    ):

        iteration_count = (
            iteration + 1
        )

        current = forward(
            patient
        )

        residual = (
            target
            - current
        )

        residual_norm = float(
            np.linalg.norm(
                residual
            )
        )

        if residual_norm <= (
            CANONICAL_TOLERANCE
        ):

            converged = True

            break

        jacobian = np.zeros(
            (3, 3),
            dtype=np.float64,
        )

        for axis in range(3):

            plus = patient.copy()

            minus = patient.copy()

            plus[axis] += (
                FINITE_DIFFERENCE_MM
            )

            minus[axis] -= (
                FINITE_DIFFERENCE_MM
            )

            plus_value = forward(
                plus
            )

            minus_value = forward(
                minus
            )

            jacobian[
                :,
                axis
            ] = (
                plus_value
                - minus_value
            ) / (
                2.0
                * FINITE_DIFFERENCE_MM
            )

        delta = np.linalg.lstsq(
            jacobian,
            residual,
            rcond=None,
        )[0]

        delta_norm = float(
            np.linalg.norm(
                delta
            )
        )

        if not np.isfinite(
            delta_norm
        ):

            raise RuntimeError(
                "Non-finite inverse step."
            )

        if delta_norm > MAX_STEP_MM:

            delta = (
                delta
                * (
                    MAX_STEP_MM
                    / delta_norm
                )
            )

        patient = (
            patient
            + DAMPING * delta
        )

    final_canonical = forward(
        patient
    )

    final_residual = (
        target
        - final_canonical
    )

    final_canonical_error = float(
        np.linalg.norm(
            final_residual
        )
    )

    physical_error = float(
        np.linalg.norm(
            patient
            - as_float_array(
                initial_patient
            )[:3]
        )
    )

    if final_canonical_error > 0.05:

        raise RuntimeError(
            "Inverse did not converge. "
            f"Canonical error="
            f"{final_canonical_error:.6f}"
        )

    return (
        patient,
        final_canonical_error,
        physical_error,
        iteration_count,
        converged,
    )


# ============================================================================
# CASE PROCESSING
# ============================================================================

def process_case(
    case,
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
            pd.DataFrame(
                case_points
            )
            .reset_index(drop=True)
        )

    image, transformed_points, geometry = (
        part220b.load_case(
            study_id,
            series_id,
            point_df,
        )
    )

    if len(transformed_points) != len(
        point_df
    ):

        raise RuntimeError(
            "Point transformation count mismatch."
        )

    records = []

    for point_index, point in enumerate(
        transformed_points
    ):

        original_patient = np.array(
            [
                float(
                    point["patient_x"]
                ),
                float(
                    point["patient_y"]
                ),
                float(
                    point["patient_z"]
                ),
            ],
            dtype=np.float64,
        )

        original_canonical = np.array(
            [
                float(
                    point["z"]
                ),
                float(
                    point["y"]
                ),
                float(
                    point["x"]
                ),
            ],
            dtype=np.float64,
        )

        # --------------------------------------------------------------
        # Forward check
        # --------------------------------------------------------------

        forward_canonical = (
            patient_to_canonical(
                original_patient,
                geometry,
            )
        )

        forward_error = float(
            np.linalg.norm(
                forward_canonical
                - original_canonical
            )
        )

        # --------------------------------------------------------------
        # Inverse check
        # --------------------------------------------------------------

        (
            reconstructed_patient,
            inverse_canonical_error,
            physical_roundtrip_error,
            iterations,
            converged,
        ) = canonical_to_patient(
            original_canonical,
            original_patient,
            geometry,
        )

        # --------------------------------------------------------------
        # Final forward reconstruction
        # --------------------------------------------------------------

        reconstructed_canonical = (
            patient_to_canonical(
                reconstructed_patient,
                geometry,
            )
        )

        final_canonical_error = float(
            np.linalg.norm(
                reconstructed_canonical
                - original_canonical
            )
        )

        patient_delta = (
            reconstructed_patient
            - original_patient
        )

        records.append(
            {
                "study_id":
                    study_id,

                "series_id":
                    series_id,

                "point_index":
                    int(point_index),

                "class_id":
                    int(
                        point["class_id"]
                    ),

                "class_name":
                    str(
                        point["class_name"]
                    ),

                "level":
                    str(
                        point["level"]
                    ),

                "original_patient_x":
                    float(
                        original_patient[0]
                    ),

                "original_patient_y":
                    float(
                        original_patient[1]
                    ),

                "original_patient_z":
                    float(
                        original_patient[2]
                    ),

                "original_canonical_z":
                    float(
                        original_canonical[0]
                    ),

                "original_canonical_y":
                    float(
                        original_canonical[1]
                    ),

                "original_canonical_x":
                    float(
                        original_canonical[2]
                    ),

                "forward_canonical_z":
                    float(
                        forward_canonical[0]
                    ),

                "forward_canonical_y":
                    float(
                        forward_canonical[1]
                    ),

                "forward_canonical_x":
                    float(
                        forward_canonical[2]
                    ),

                "forward_canonical_error":
                    forward_error,

                "reconstructed_patient_x":
                    float(
                        reconstructed_patient[0]
                    ),

                "reconstructed_patient_y":
                    float(
                        reconstructed_patient[1]
                    ),

                "reconstructed_patient_z":
                    float(
                        reconstructed_patient[2]
                    ),

                "patient_dx_mm":
                    float(
                        patient_delta[0]
                    ),

                "patient_dy_mm":
                    float(
                        patient_delta[1]
                    ),

                "patient_dz_mm":
                    float(
                        patient_delta[2]
                    ),

                "physical_roundtrip_error_mm":
                    physical_roundtrip_error,

                "inverse_canonical_error":
                    inverse_canonical_error,

                "final_canonical_error":
                    final_canonical_error,

                "inverse_iterations":
                    int(iterations),

                "converged":
                    bool(converged),
            }
        )

    return records


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    records,
):

    df = pd.DataFrame(
        records
    )

    if df.empty:

        raise RuntimeError(
            "No round-trip records generated."
        )

    print()
    print("=" * 80)
    print("PART 2.50 GLOBAL GEOMETRY ROUND-TRIP")
    print("=" * 80)

    physical = df[
        "physical_roundtrip_error_mm"
    ].astype(float)

    canonical = df[
        "final_canonical_error"
    ].astype(float)

    forward = df[
        "forward_canonical_error"
    ].astype(float)

    summary = {
        "validation_cases":
            int(
                df[
                    "study_id"
                ].nunique()
            ),

        "validation_series":
            int(
                df[
                    [
                        "study_id",
                        "series_id",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),

        "points":
            int(len(df)),

        "converged_points":
            int(
                df[
                    "converged"
                ].sum()
            ),

        "mean_physical_roundtrip_error_mm":
            float(
                physical.mean()
            ),

        "median_physical_roundtrip_error_mm":
            float(
                physical.median()
            ),

        "max_physical_roundtrip_error_mm":
            float(
                physical.max()
            ),

        "within_0_1mm_fraction":
            float(
                (
                    physical <= 0.1
                ).mean()
            ),

        "within_0_5mm_fraction":
            float(
                (
                    physical <= 0.5
                ).mean()
            ),

        "within_1mm_fraction":
            float(
                (
                    physical <= 1.0
                ).mean()
            ),

        "mean_forward_canonical_error":
            float(
                forward.mean()
            ),

        "max_forward_canonical_error":
            float(
                forward.max()
            ),

        "mean_final_canonical_error":
            float(
                canonical.mean()
            ),

        "max_final_canonical_error":
            float(
                canonical.max()
            ),
    }

    print(
        f"Validation studies : "
        f"{summary['validation_cases']}"
    )

    print(
        f"Validation series  : "
        f"{summary['validation_series']}"
    )

    print(
        f"Points             : "
        f"{summary['points']}"
    )

    print(
        f"Converged points   : "
        f"{summary['converged_points']}"
    )

    print()

    print(
        "Mean physical round-trip error : "
        f"{summary['mean_physical_roundtrip_error_mm']:.6f} mm"
    )

    print(
        "Median physical round-trip error: "
        f"{summary['median_physical_roundtrip_error_mm']:.6f} mm"
    )

    print(
        "Maximum physical round-trip error: "
        f"{summary['max_physical_roundtrip_error_mm']:.6f} mm"
    )

    print()

    print(
        "Within 0.1 mm : "
        f"{summary['within_0_1mm_fraction']:.4f}"
    )

    print(
        "Within 0.5 mm : "
        f"{summary['within_0_5mm_fraction']:.4f}"
    )

    print(
        "Within 1.0 mm : "
        f"{summary['within_1mm_fraction']:.4f}"
    )

    print()

    print(
        "Mean forward canonical error: "
        f"{summary['mean_forward_canonical_error']:.8f}"
    )

    print(
        "Max forward canonical error : "
        f"{summary['max_forward_canonical_error']:.8f}"
    )

    print(
        "Mean final canonical error  : "
        f"{summary['mean_final_canonical_error']:.8f}"
    )

    print(
        "Max final canonical error   : "
        f"{summary['max_final_canonical_error']:.8f}"
    )

    # ========================================================================
    # DISEASE SUMMARY
    # ========================================================================

    disease_rows = []

    for disease in sorted(
        df["class_name"].unique()
    ):

        subset = df[
            df["class_name"]
            == disease
        ]

        errors = subset[
            "physical_roundtrip_error_mm"
        ].astype(float)

        disease_rows.append(
            {
                "class_name":
                    disease,

                "count":
                    len(subset),

                "mean_error_mm":
                    float(
                        errors.mean()
                    ),

                "median_error_mm":
                    float(
                        errors.median()
                    ),

                "max_error_mm":
                    float(
                        errors.max()
                    ),

                "within_1mm_fraction":
                    float(
                        (
                            errors <= 1
                        ).mean()
                    ),
            }
        )

    disease_df = pd.DataFrame(
        disease_rows
    )

    print()
    print("=" * 80)
    print("DISEASE-WISE ROUND-TRIP ERROR")
    print("=" * 80)

    print(
        disease_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    records_path = (
        OUTPUT_DIR
        / "part250_geometry_roundtrip_records.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part250_geometry_roundtrip_summary.json"
    )

    disease_path = (
        OUTPUT_DIR
        / "part250_geometry_roundtrip_disease_summary.csv"
    )

    df.to_csv(
        records_path,
        index=False,
    )

    disease_df.to_csv(
        disease_path,
        index=False,
    )

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

    print()
    print("=" * 80)
    print("PART 2.50 COMPLETE")
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

    print(records_path)
    print(summary_path)
    print(disease_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.50")
    print("GEOMETRY ROUND-TRIP VALIDATION")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    manifest = (
        part220b.load_manifest()
    )

    print(
        f"\nFull manifest rows: "
        f"{len(manifest)}"
    )

    validation_cases = (
        build_validation_cases(
            manifest
        )
    )

    all_records = []

    successful_cases = 0

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

        try:

            records = process_case(
                case
            )

            all_records.extend(
                records
            )

            successful_cases += 1

            print(
                f"[{case_number:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id} | "
                f"Points={len(records)}"
            )

        except Exception as exc:

            print(
                f"[{case_number:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id} "
                f"FAILED: {exc}"
            )

    print()
    print(
        f"Successful cases: "
        f"{successful_cases}/25"
    )

    print(
        f"Total point records: "
        f"{len(all_records)}"
    )

    if len(all_records) != 188:

        raise RuntimeError(
            "Expected 188 point records, "
            f"found {len(all_records)}."
        )

    build_summary(
        all_records
    )


if __name__ == "__main__":
    main()