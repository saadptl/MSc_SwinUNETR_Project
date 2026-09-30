"""
PART 2.49
RFNN PHYSICAL-SPACE DISPLACEMENT AUDIT

Purpose
-------
Convert the RFNN evidence displacement from Part 2.48
(model-grid coordinates) into physical patient-space coordinates.

This version uses the ACTUAL geometry dictionary returned by
Part 2.20B.

Geometry fields used:

    row_direction
    column_direction
    normal_direction
    row_spacing
    column_spacing
    slice_spacing
    origin

No guessed Part 2.20B reverse API is used.

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
import torch


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part249_rfnn_physical_space_displacement_audit"
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

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
    / "part227_best_macro_disease.pth"
)

LEVELS = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]

LOCAL_RADIUS = 6


# ============================================================================
# BASIC HELPERS
# ============================================================================

def safe_float(value):

    try:

        value = float(value)

        if not np.isfinite(value):

            return np.nan

        return value

    except Exception:

        return np.nan


# ============================================================================
# MODEL
# ============================================================================

def build_model():

    print()
    print("=" * 80)
    print("LOADING PART 2.27 CHECKPOINT")
    print("=" * 80)

    model = part220b.build_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print(
        f"Missing keys    : {len(missing)}"
    )

    print(
        f"Unexpected keys : {len(unexpected)}"
    )

    print(
        "Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    if missing:

        raise RuntimeError(
            "Checkpoint contains missing keys."
        )

    if unexpected:

        raise RuntimeError(
            "Checkpoint contains unexpected keys."
        )

    return model


# ============================================================================
# EXACT VALIDATION COHORT
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
            "Part 2.20B validation selector "
            "must return a DataFrame."
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

    records = []

    for case in validation_cases:

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

        for point_index, (_, row) in enumerate(
            point_df.iterrows()
        ):

            records.append(
                {
                    "study_id": str(
                        case["study_id"]
                    ),
                    "series_id": str(
                        case["series_id"]
                    ),
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
            "Expected 188 validation points, "
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

    return validation_cases


# ============================================================================
# GEOMETRY
# ============================================================================

def canonical_to_patient_using_geometry(
    geometry,
    z,
    y,
    x,
    initial_patient_point,
):
    """
    Numerically invert the exact Part 2.20B
    patient_point_to_canonical() transformation.

    Target:
        canonical (z, y, x)

    Unknown:
        patient (x, y, z)

    This avoids assuming that patient XYZ is aligned
    with canonical ZYX and avoids independently
    interpolating patient_min/patient_max.

    The initial patient point is the annotated RFNN
    physical coordinate, so the solver starts very
    close to the expected anatomical location.
    """

    target = np.array(
        [
            float(z),
            float(y),
            float(x),
        ],
        dtype=np.float64,
    )

    patient = np.array(
        initial_patient_point,
        dtype=np.float64,
    )

    if patient.shape != (3,):

        raise RuntimeError(
            "Initial patient point must contain "
            "exactly three coordinates."
        )

    # ----------------------------------------------------------------------
    # Numerical inverse settings
    # ----------------------------------------------------------------------

    max_iterations = 20

    tolerance = 1e-4

    finite_difference_mm = 0.5

    max_step_mm = 10.0

    damping = 0.8

    def forward(
        patient_point,
    ):

        result = (
            part220b.patient_point_to_canonical(
                patient_point,
                geometry,
            )
        )

        result = np.asarray(
            result,
            dtype=np.float64,
        ).reshape(-1)

        if result.size < 3:

            raise RuntimeError(
                "patient_point_to_canonical() "
                "did not return three coordinates."
            )

        return result[:3]

    # ----------------------------------------------------------------------
    # Gauss-Newton / local Jacobian inversion
    # ----------------------------------------------------------------------

    for iteration in range(
        max_iterations
    ):

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

        if residual_norm < tolerance:

            break

        # --------------------------------------------------------------
        # Numerical Jacobian:
        #
        # canonical coordinates
        #       /
        #      /
        # patient physical coordinates
        #
        # J[i,j] = d canonical_i / d patient_j
        # --------------------------------------------------------------

        jacobian = np.zeros(
            (3, 3),
            dtype=np.float64,
        )

        for axis in range(3):

            plus = patient.copy()

            minus = patient.copy()

            plus[axis] += (
                finite_difference_mm
            )

            minus[axis] -= (
                finite_difference_mm
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
                * finite_difference_mm
            )

        try:

            delta = np.linalg.lstsq(
                jacobian,
                residual,
                rcond=None,
            )[0]

        except np.linalg.LinAlgError:

            raise RuntimeError(
                "Could not solve the local "
                "geometry Jacobian."
            )

        delta_norm = float(
            np.linalg.norm(
                delta
            )
        )

        if not np.isfinite(
            delta_norm
        ):

            raise RuntimeError(
                "Non-finite physical-space "
                "inverse step."
            )

        # --------------------------------------------------------------
        # Prevent unstable large jumps.
        # --------------------------------------------------------------

        if delta_norm > max_step_mm:

            delta = (
                delta
                * (
                    max_step_mm
                    / delta_norm
                )
            )

        patient = (
            patient
            + damping * delta
        )

    # ----------------------------------------------------------------------
    # Final verification
    # ----------------------------------------------------------------------

    final_canonical = forward(
        patient
    )

    final_residual = (
        target
        - final_canonical
    )

    final_residual_norm = float(
        np.linalg.norm(
            final_residual
        )
    )

    if final_residual_norm > 0.05:

        raise RuntimeError(
            "Physical-space inverse did not "
            "converge sufficiently. "
            f"Canonical residual="
            f"{final_residual_norm:.6f}"
        )

    return patient


# ============================================================================
# LOCAL VOXELS
# ============================================================================

def build_local_voxel_records(
    logits,
    probabilities,
    center_z,
    center_y,
    center_x,
):

    _, depth, height, width = (
        logits.shape
    )

    center_z = int(
        np.clip(
            round(center_z),
            0,
            depth - 1,
        )
    )

    center_y = int(
        np.clip(
            round(center_y),
            0,
            height - 1,
        )
    )

    center_x = int(
        np.clip(
            round(center_x),
            0,
            width - 1,
        )
    )

    z_start = max(
        0,
        center_z - LOCAL_RADIUS,
    )

    z_end = min(
        depth,
        center_z + LOCAL_RADIUS + 1,
    )

    y_start = max(
        0,
        center_y - LOCAL_RADIUS,
    )

    y_end = min(
        height,
        center_y + LOCAL_RADIUS + 1,
    )

    x_start = max(
        0,
        center_x - LOCAL_RADIUS,
    )

    x_end = min(
        width,
        center_x + LOCAL_RADIUS + 1,
    )

    records = []

    for z in range(
        z_start,
        z_end,
    ):

        for y in range(
            y_start,
            y_end,
        ):

            for x in range(
                x_start,
                x_end,
            ):

                local_logits = (
                    logits[
                        :,
                        z,
                        y,
                        x,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                local_probs = (
                    probabilities[
                        :,
                        z,
                        y,
                        x,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                dz = (
                    z
                    - center_z
                )

                dy = (
                    y
                    - center_y
                )

                dx = (
                    x
                    - center_x
                )

                records.append(
                    {
                        "z": z,
                        "y": y,
                        "x": x,
                        "dz": dz,
                        "dy": dy,
                        "dx": dx,
                        "distance_voxels":
                            float(
                                np.sqrt(
                                    dz * dz
                                    + dy * dy
                                    + dx * dx
                                )
                            ),
                        "rfnn_probability":
                            float(
                                local_probs[3]
                            ),
                        "rfnn_logit":
                            float(
                                local_logits[3]
                            ),
                        "rfnn_minus_background":
                            float(
                                local_logits[3]
                                - local_logits[0]
                            ),
                        "rfnn_minus_lss":
                            float(
                                local_logits[3]
                                - local_logits[4]
                            ),
                        "rfnn_minus_scs":
                            float(
                                local_logits[3]
                                - local_logits[1]
                            ),
                    }
                )

    return records


# ============================================================================
# MAXIMUM
# ============================================================================

def find_maximum(
    voxel_df,
    column,
):

    index = (
        voxel_df[column]
        .astype(float)
        .idxmax()
    )

    return voxel_df.loc[
        index
    ]


# ============================================================================
# CASE
# ============================================================================

def analyze_case(
    model,
    case,
    device,
):

    study_id = str(
        case["study_id"]
    )

    series_id = str(
        case["series_id"]
    )

    case_rows = case["points"]

    if not isinstance(
        case_rows,
        pd.DataFrame,
    ):

        case_rows = pd.DataFrame(
            case_rows
        )

    case_rows = (
        case_rows
        .reset_index(drop=True)
    )

    image, transformed_points, geometry = (
        part220b.load_case(
            study_id,
            series_id,
            case_rows,
        )
    )

    if not torch.is_tensor(
        image
    ):

        image = torch.tensor(
            image,
            dtype=torch.float32,
        )

    if image.ndim == 3:

        image = (
            image
            .unsqueeze(0)
            .unsqueeze(0)
        )

    elif image.ndim == 4:

        image = (
            image
            .unsqueeze(0)
        )

    else:

        raise RuntimeError(
            f"Unexpected image shape: "
            f"{tuple(image.shape)}"
        )

    image = (
        image
        .float()
        .to(device)
    )

    with torch.no_grad():

        output = model(
            image
        )

        logits = output[0]

        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

    rfnn_indices = (
        case_rows.index[
            case_rows[
                "class_name"
            ]
            == "Right Neural Foraminal Narrowing"
        ]
        .tolist()
    )

    records = []

    for point_index in rfnn_indices:

        transformed_point = (
            transformed_points[
                point_index
            ]
        )

        model_z = safe_float(
            transformed_point["z"]
        )

        model_y = safe_float(
            transformed_point["y"]
        )

        model_x = safe_float(
            transformed_point["x"]
        )

        annotation_patient = np.array(
            [
                safe_float(
                    transformed_point[
                        "patient_x"
                    ]
                ),
                safe_float(
                    transformed_point[
                        "patient_y"
                    ]
                ),
                safe_float(
                    transformed_point[
                        "patient_z"
                    ]
                ),
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                annotation_patient
            )
        ):

            continue

        voxel_records = (
            build_local_voxel_records(
                logits,
                probabilities,
                model_z,
                model_y,
                model_x,
            )
        )

        voxel_df = pd.DataFrame(
            voxel_records
        )

        criteria = {
            "max_rfnn_probability":
                "rfnn_probability",

            "max_rfnn_logit":
                "rfnn_logit",

            "max_rfnn_minus_background":
                "rfnn_minus_background",

            "max_rfnn_minus_lss":
                "rfnn_minus_lss",

            "max_rfnn_minus_scs":
                "rfnn_minus_scs",
        }

        for criterion, column in (
            criteria.items()
        ):

            maximum = find_maximum(
                voxel_df,
                column,
            )

            max_z = float(
                maximum["z"]
            )

            max_y = float(
                maximum["y"]
            )

            max_x = float(
                maximum["x"]
            )

            maximum_patient = (
            canonical_to_patient_using_geometry(
                geometry,
                max_z,
                max_y,
                max_x,
                annotation_patient,
            )
            )

            physical_delta = (
                maximum_patient
                - annotation_patient
            )

            physical_distance = float(
                np.linalg.norm(
                    physical_delta
                )
            )

            records.append(
                {
                    "study_id":
                        study_id,

                    "series_id":
                        series_id,

                    "point_index":
                        int(
                            point_index
                        ),

                    "level":
                        str(
                            case_rows.iloc[
                                point_index
                            ]["level"]
                        ),

                    "criterion":
                        criterion,

                    "value":
                        float(
                            maximum[
                                column
                            ]
                        ),

                    "annotation_model_z":
                        model_z,

                    "annotation_model_y":
                        model_y,

                    "annotation_model_x":
                        model_x,

                    "maximum_model_z":
                        max_z,

                    "maximum_model_y":
                        max_y,

                    "maximum_model_x":
                        max_x,

                    "model_dz":
                        float(
                            maximum["dz"]
                        ),

                    "model_dy":
                        float(
                            maximum["dy"]
                        ),

                    "model_dx":
                        float(
                            maximum["dx"]
                        ),

                    "model_distance_voxels":
                        float(
                            maximum[
                                "distance_voxels"
                            ]
                        ),

                    "annotation_patient_x":
                        float(
                            annotation_patient[0]
                        ),

                    "annotation_patient_y":
                        float(
                            annotation_patient[1]
                        ),

                    "annotation_patient_z":
                        float(
                            annotation_patient[2]
                        ),

                    "maximum_patient_x":
                        float(
                            maximum_patient[0]
                        ),

                    "maximum_patient_y":
                        float(
                            maximum_patient[1]
                        ),

                    "maximum_patient_z":
                        float(
                            maximum_patient[2]
                        ),

                    "physical_dx_mm":
                        float(
                            physical_delta[0]
                        ),

                    "physical_dy_mm":
                        float(
                            physical_delta[1]
                        ),

                    "physical_dz_mm":
                        float(
                            physical_delta[2]
                        ),

                    "physical_distance_mm":
                        physical_distance,
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
            "No physical displacement records "
            "were generated."
        )

    expected = 45 * 5

    if len(df) != expected:

        raise RuntimeError(
            f"Expected {expected} records, "
            f"found {len(df)}."
        )

    global_rows = []

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

        dx = subset[
            "physical_dx_mm"
        ].astype(float)

        dy = subset[
            "physical_dy_mm"
        ].astype(float)

        dz = subset[
            "physical_dz_mm"
        ].astype(float)

        global_rows.append(
            {
                "criterion": criterion,
                "count": len(subset),

                "mean_distance_mm":
                    float(distance.mean()),

                "median_distance_mm":
                    float(distance.median()),

                "min_distance_mm":
                    float(distance.min()),

                "max_distance_mm":
                    float(distance.max()),

                "within_2mm_fraction":
                    float(
                        (
                            distance <= 2
                        ).mean()
                    ),

                "within_5mm_fraction":
                    float(
                        (
                            distance <= 5
                        ).mean()
                    ),

                "within_10mm_fraction":
                    float(
                        (
                            distance <= 10
                        ).mean()
                    ),

                "mean_dx_mm":
                    float(dx.mean()),

                "mean_dy_mm":
                    float(dy.mean()),

                "mean_dz_mm":
                    float(dz.mean()),

                "mean_abs_dx_mm":
                    float(
                        np.abs(dx).mean()
                    ),

                "mean_abs_dy_mm":
                    float(
                        np.abs(dy).mean()
                    ),

                "mean_abs_dz_mm":
                    float(
                        np.abs(dz).mean()
                    ),

                "negative_dx_fraction":
                    float(
                        (dx < 0).mean()
                    ),

                "positive_dx_fraction":
                    float(
                        (dx > 0).mean()
                    ),

                "negative_dy_fraction":
                    float(
                        (dy < 0).mean()
                    ),

                "positive_dy_fraction":
                    float(
                        (dy > 0).mean()
                    ),

                "negative_dz_fraction":
                    float(
                        (dz < 0).mean()
                    ),

                "positive_dz_fraction":
                    float(
                        (dz > 0).mean()
                    ),
            }
        )

    global_df = pd.DataFrame(
        global_rows
    )

    level_rows = []

    for level in LEVELS:

        for criterion in sorted(
            df["criterion"].unique()
        ):

            subset = df[
                (
                    df["level"]
                    == level
                )
                & (
                    df["criterion"]
                    == criterion
                )
            ]

            if subset.empty:
                continue

            distance = subset[
                "physical_distance_mm"
            ].astype(float)

            level_rows.append(
                {
                    "level": level,
                    "criterion": criterion,
                    "count": len(subset),

                    "mean_distance_mm":
                        float(
                            distance.mean()
                        ),

                    "median_distance_mm":
                        float(
                            distance.median()
                        ),

                    "within_2mm_fraction":
                        float(
                            (
                                distance <= 2
                            ).mean()
                        ),

                    "within_5mm_fraction":
                        float(
                            (
                                distance <= 5
                            ).mean()
                        ),

                    "within_10mm_fraction":
                        float(
                            (
                                distance <= 10
                            ).mean()
                        ),

                    "mean_dx_mm":
                        float(
                            subset[
                                "physical_dx_mm"
                            ].mean()
                        ),

                    "mean_dy_mm":
                        float(
                            subset[
                                "physical_dy_mm"
                            ].mean()
                        ),

                    "mean_dz_mm":
                        float(
                            subset[
                                "physical_dz_mm"
                            ].mean()
                        ),
                }
            )

    level_df = pd.DataFrame(
        level_rows
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    records_path = (
        OUTPUT_DIR
        / "part249_rfnn_physical_displacement_records.csv"
    )

    global_path = (
        OUTPUT_DIR
        / "part249_rfnn_physical_displacement_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part249_rfnn_level_physical_displacement_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part249_rfnn_physical_space_displacement_audit_summary.json"
    )

    df.to_csv(
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

    json_summary = {
        "validation_cases": 25,
        "validation_points": 188,
        "rfnn_points": 45,
        "records": len(df),
        "criteria": {},
    }

    for _, row in global_df.iterrows():

        criterion = str(
            row["criterion"]
        )

        json_summary[
            "criteria"
        ][criterion] = {
            "mean_distance_mm":
                float(
                    row[
                        "mean_distance_mm"
                    ]
                ),

            "median_distance_mm":
                float(
                    row[
                        "median_distance_mm"
                    ]
                ),

            "within_5mm_fraction":
                float(
                    row[
                        "within_5mm_fraction"
                    ]
                ),

            "within_10mm_fraction":
                float(
                    row[
                        "within_10mm_fraction"
                    ]
                ),

            "mean_dx_mm":
                float(
                    row[
                        "mean_dx_mm"
                    ]
                ),

            "mean_dy_mm":
                float(
                    row[
                        "mean_dy_mm"
                    ]
                ),

            "mean_dz_mm":
                float(
                    row[
                        "mean_dz_mm"
                    ]
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
    # PRINT
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.49 GLOBAL PHYSICAL-SPACE DISPLACEMENT")
    print("=" * 80)

    print(
        global_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.49 LEVEL ANALYSIS")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.49 COMPLETE")
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
    print(global_path)
    print(level_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.49")
    print("RFNN PHYSICAL-SPACE DISPLACEMENT AUDIT")
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

    validation_cases = (
        build_exact_validation_cohort(
            manifest
        )
    )

    model = build_model()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device          : "
        f"{device}"
    )

    model = (
        model
        .to(device)
        .eval()
    )

    print()
    print("=" * 80)
    print("RUNNING PHYSICAL-SPACE RFNN DISPLACEMENT ANALYSIS")
    print("=" * 80)

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

            records = analyze_case(
                model,
                case,
                device,
            )

            all_records.extend(
                records
            )

            if len(records) > 0:

                successful_cases += 1

            print(
                f"[{case_number:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id} | "
                f"RFNN physical records="
                f"{len(records)}"
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
        f"Cases containing RFNN results: "
        f"{successful_cases}/25"
    )

    print(
        f"Physical displacement records: "
        f"{len(all_records)}"
    )

    if len(all_records) != 225:

        raise RuntimeError(
            "Part 2.49 did not reproduce the "
            "expected 45 RFNN points × 5 criteria "
            f"(225 records). Found "
            f"{len(all_records)}."
        )

    build_summary(
        all_records
    )


if __name__ == "__main__":
    main()