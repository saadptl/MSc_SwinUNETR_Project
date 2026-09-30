"""
PART 2.53
RFNN IN-PLANE DIRECTIONAL / ANATOMICAL CONTEXT AUDIT

Purpose
-------
Investigate the in-plane physical direction in which the model's strongest
RFNN-related evidence is displaced from the annotated RFNN point.

Part 2.52 showed that the displacement is predominantly in the DICOM
row/column plane rather than along the slice-normal direction.

This analysis therefore samples MRI/model evidence along the DICOM
row and column directions around each RFNN annotation.

For each RFNN point and each Part 2.51 evidence criterion:

    - annotation location
    - evidence maximum location
    - physical displacement
    - DICOM row component
    - DICOM column component
    - slice-normal component
    - directional angle within the acquisition plane
    - MRI intensity at annotation
    - MRI intensity at maximum
    - intensity difference
    - model RFNN probability at annotation
    - model RFNN probability at maximum
    - competing class probabilities

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
import torch


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

PART252_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part252_dicom_axis_displacement_audit"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part253_rfnn_inplane_directional_context_audit"
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
# INPUTS
# ============================================================================

PART251_RECORDS = (
    PART251_DIR
    / "part251_physical_evidence_records.csv"
)

PART252_RECORDS = (
    PART252_DIR
    / "part252_dicom_axis_displacement_records.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
    / "part227_best_macro_disease.pth"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

CRITERIA = {
    "max_rfnn_probability",
    "max_rfnn_logit",
    "max_rfnn_minus_background",
    "max_rfnn_minus_lss",
    "max_rfnn_minus_scs",
}

# Model-grid neighborhood around annotation.
LOCAL_RADIUS = 6

# Physical directional sampling distances.
PROFILE_DISTANCES_MM = [
    -20.0,
    -15.0,
    -10.0,
    -5.0,
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
]

# Numerical inverse settings.
MAX_INVERSE_ITERATIONS = 20

INVERSE_TOLERANCE = 1e-4

FINITE_DIFFERENCE_MM = 0.5

MAX_INVERSE_STEP_MM = 10.0

INVERSE_DAMPING = 0.8


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
            "Cannot normalize zero vector."
        )

    return vector / norm


def get_geometry_axes(
    geometry,
):

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
# LOAD PART 2.51
# ============================================================================

def load_part251():

    if not PART251_RECORDS.exists():

        raise FileNotFoundError(
            "Part 2.51 records not found:\n"
            f"{PART251_RECORDS}"
        )

    df = pd.read_csv(
        PART251_RECORDS
    )

    if len(df) != 225:

        raise RuntimeError(
            "Expected 225 Part 2.51 records, "
            f"found {len(df)}."
        )

    actual = set(
        df["criterion"].unique()
    )

    if actual != CRITERIA:

        raise RuntimeError(
            "Part 2.51 criterion mismatch."
        )

    return df


# ============================================================================
# LOAD PART 2.52
# ============================================================================

def load_part252():

    if not PART252_RECORDS.exists():

        raise FileNotFoundError(
            "Part 2.52 records not found:\n"
            f"{PART252_RECORDS}"
        )

    df = pd.read_csv(
        PART252_RECORDS
    )

    if len(df) != 225:

        raise RuntimeError(
            "Expected 225 Part 2.52 records, "
            f"found {len(df)}."
        )

    actual = set(
        df["criterion"].unique()
    )

    if actual != CRITERIA:

        raise RuntimeError(
            "Part 2.52 criterion mismatch."
        )

    return df


# ============================================================================
# EXACT VALIDATION CASES
# ============================================================================

def build_validation_cases(
    manifest,
):

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
            "Expected 25 validation series."
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

    cases = (
        part220b.build_case_index(
            validation_manifest
        )
    )

    order = {
        (
            str(row["study_id"]),
            str(row["series_id"]),
        ): index
        for index, row in selected.iterrows()
    }

    cases = sorted(
        cases,
        key=lambda case:
        order.get(
            (
                str(case["study_id"]),
                str(case["series_id"]),
            ),
            10000,
        ),
    )

    if len(cases) != 25:

        raise RuntimeError(
            "Validation case count mismatch."
        )

    return cases


# ============================================================================
# NUMERICAL INVERSE
# ============================================================================

def canonical_to_patient(
    target_canonical,
    initial_patient,
    geometry,
):

    target = np.asarray(
        target_canonical,
        dtype=np.float64,
    ).reshape(-1)[:3]

    patient = np.asarray(
        initial_patient,
        dtype=np.float64,
    ).reshape(-1)[:3].copy()

    def forward(
        point,
    ):

        result = (
            part220b.patient_point_to_canonical(
                point,
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
                "did not return 3 coordinates."
            )

        return result[:3]

    for _ in range(
        MAX_INVERSE_ITERATIONS
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

        if residual_norm <= (
            INVERSE_TOLERANCE
        ):

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

        if delta_norm > MAX_INVERSE_STEP_MM:

            delta = (
                delta
                * (
                    MAX_INVERSE_STEP_MM
                    / delta_norm
                )
            )

        patient = (
            patient
            + INVERSE_DAMPING * delta
        )

    final = forward(
        patient
    )

    error = float(
        np.linalg.norm(
            target
            - final
        )
    )

    if error > 0.05:

        raise RuntimeError(
            "Canonical-to-patient inverse failed. "
            f"Error={error:.6f}"
        )

    return patient


# ============================================================================
# PATIENT -> CANONICAL
# ============================================================================

def patient_to_canonical(
    patient,
    geometry,
):

    result = (
        part220b.patient_point_to_canonical(
            patient,
            geometry,
        )
    )

    result = np.asarray(
        result,
        dtype=np.float64,
    ).reshape(-1)

    if result.size < 3:

        raise RuntimeError(
            "Patient-to-canonical transformation "
            "returned fewer than 3 values."
        )

    return result[:3]


# ============================================================================
# MODEL INFERENCE
# ============================================================================

def run_model(
    model,
    image,
    device,
):

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

    return (
        logits,
        probabilities,
    )


# ============================================================================
# SAMPLE MODEL AT PATIENT POINT
# ============================================================================

def sample_model_at_patient_point(
    patient_point,
    geometry,
    logits,
    probabilities,
):

    canonical = (
        patient_to_canonical(
            patient_point,
            geometry,
        )
    )

    z = int(
        round(
            float(
                canonical[0]
            )
        )
    )

    y = int(
        round(
            float(
                canonical[1]
            )
        )
    )

    x = int(
        round(
            float(
                canonical[2]
            )
        )
    )

    _, depth, height, width = (
        logits.shape
    )

    if (
        z < 0
        or z >= depth
        or y < 0
        or y >= height
        or x < 0
        or x >= width
    ):

        return None

    voxel_logits = (
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

    voxel_probabilities = (
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

    prediction = int(
        np.argmax(
            voxel_probabilities
        )
    )

    return {
        "canonical_z":
            float(canonical[0]),

        "canonical_y":
            float(canonical[1]),

        "canonical_x":
            float(canonical[2]),

        "background_logit":
            float(voxel_logits[0]),

        "scs_logit":
            float(voxel_logits[1]),

        "lfnn_logit":
            float(voxel_logits[2]),

        "rfnn_logit":
            float(voxel_logits[3]),

        "lss_logit":
            float(voxel_logits[4]),

        "rss_logit":
            float(voxel_logits[5]),

        "background_probability":
            float(voxel_probabilities[0]),

        "scs_probability":
            float(voxel_probabilities[1]),

        "lfnn_probability":
            float(voxel_probabilities[2]),

        "rfnn_probability":
            float(voxel_probabilities[3]),

        "lss_probability":
            float(voxel_probabilities[4]),

        "rss_probability":
            float(voxel_probabilities[5]),

        "predicted_class_id":
            prediction,

        "predicted_class":
            CLASS_NAMES[prediction],
    }


# ============================================================================
# IMAGE INTENSITY SAMPLING
# ============================================================================

def sample_image_at_patient_point(
    patient_point,
    geometry,
    image,
):

    canonical = (
        patient_to_canonical(
            patient_point,
            geometry,
        )
    )

    z = int(
        round(
            float(
                canonical[0]
            )
        )
    )

    y = int(
        round(
            float(
                canonical[1]
            )
        )
    )

    x = int(
        round(
            float(
                canonical[2]
            )
        )
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    if image_array.ndim == 4:

        image_array = (
            image_array[0]
        )

    if image_array.ndim != 3:

        raise RuntimeError(
            "Unexpected image shape for "
            f"intensity sampling: "
            f"{image_array.shape}"
        )

    depth, height, width = (
        image_array.shape
    )

    if (
        z < 0
        or z >= depth
        or y < 0
        or y >= height
        or x < 0
        or x >= width
    ):

        return None

    return {
        "canonical_z":
            float(canonical[0]),

        "canonical_y":
            float(canonical[1]),

        "canonical_x":
            float(canonical[2]),

        "image_intensity":
            float(
                image_array[
                    z,
                    y,
                    x,
                ]
            ),
    }


# ============================================================================
# CASE ANALYSIS
# ============================================================================

def analyze_case(
    model,
    case,
    part251_case_records,
    part252_case_records,
    device,
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

    (
        image,
        transformed_points,
        geometry,
    ) = part220b.load_case(
        study_id,
        series_id,
        point_df,
    )

    logits, probabilities = (
        run_model(
            model,
            image,
            device,
        )
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    if image_array.ndim == 4:

        image_array = image_array[0]

    records = []

    # ------------------------------------------------------------------------
    # Only RFNN annotations.
    # ------------------------------------------------------------------------

    rfnn_indices = (
        point_df.index[
            point_df[
                "class_name"
            ]
            == "Right Neural Foraminal Narrowing"
        ]
        .tolist()
    )

    for point_index in rfnn_indices:

        point = transformed_points[
            point_index
        ]

        annotation_patient = np.array(
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

        (
            row_direction,
            column_direction,
            normal_direction,
        ) = get_geometry_axes(
            geometry
        )

        # --------------------------------------------------------------------
        # Match the five Part 2.51 evidence maxima.
        # --------------------------------------------------------------------

        p251 = part251_case_records[
            (
                part251_case_records[
                    "point_index"
                ]
                == point_index
            )
        ].copy()

        p252 = part252_case_records[
            (
                part252_case_records[
                    "point_index"
                ]
                == point_index
            )
        ].copy()

        for criterion in sorted(
            CRITERIA
        ):

            p251_row = p251[
                p251[
                    "criterion"
                ]
                == criterion
            ]

            p252_row = p252[
                p252[
                    "criterion"
                ]
                == criterion
            ]

            if len(p251_row) != 1:

                raise RuntimeError(
                    f"Expected one Part 2.51 row for "
                    f"{study_id}/{series_id}/"
                    f"{point_index}/{criterion}"
                )

            if len(p252_row) != 1:

                raise RuntimeError(
                    f"Expected one Part 2.52 row for "
                    f"{study_id}/{series_id}/"
                    f"{point_index}/{criterion}"
                )

            p251_row = (
                p251_row
                .iloc[0]
            )

            p252_row = (
                p252_row
                .iloc[0]
            )

            # ---------------------------------------------------------------
            # Evidence maximum patient-space position.
            # ---------------------------------------------------------------

            max_patient = np.array(
                [
                    float(
                        p251_row[
                            "maximum_patient_x"
                        ]
                    ),
                    float(
                        p251_row[
                            "maximum_patient_y"
                        ]
                    ),
                    float(
                        p251_row[
                            "maximum_patient_z"
                        ]
                    ),
                ],
                dtype=np.float64,
            )

            displacement = (
                max_patient
                - annotation_patient
            )

            # ---------------------------------------------------------------
            # DICOM decomposition.
            # ---------------------------------------------------------------

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

            inplane_distance = float(
                np.sqrt(
                    row_component ** 2
                    + column_component ** 2
                )
            )

            physical_distance = float(
                np.linalg.norm(
                    displacement
                )
            )

            # ---------------------------------------------------------------
            # Direction angle.
            #
            # 0 degrees = +row direction
            # 90 degrees = +column direction
            # -90 degrees = -column direction
            # 180/-180 = -row direction
            # ---------------------------------------------------------------

            if inplane_distance > 0:

                angle_deg = math.degrees(
                    math.atan2(
                        column_component,
                        row_component,
                    )
                )

            else:

                angle_deg = 0.0

            # ---------------------------------------------------------------
            # Intensity at annotation and evidence maximum.
            # ---------------------------------------------------------------

            annotation_image = (
                sample_image_at_patient_point(
                    annotation_patient,
                    geometry,
                    image_array,
                )
            )

            maximum_image = (
                sample_image_at_patient_point(
                    max_patient,
                    geometry,
                    image_array,
                )
            )

            # ---------------------------------------------------------------
            # Model output at annotation.
            # ---------------------------------------------------------------

            annotation_model = (
                sample_model_at_patient_point(
                    annotation_patient,
                    geometry,
                    logits,
                    probabilities,
                )
            )

            maximum_model = (
                sample_model_at_patient_point(
                    max_patient,
                    geometry,
                    logits,
                    probabilities,
                )
            )

            if (
                annotation_image is None
                or maximum_image is None
                or annotation_model is None
                or maximum_model is None
            ):

                continue

            # ---------------------------------------------------------------
            # Directional physical profile.
            # ---------------------------------------------------------------

            for signed_distance in (
                PROFILE_DISTANCES_MM
            ):

                profile_patient = (
                    annotation_patient
                    + signed_distance
                    * row_direction
                )

                profile_image = (
                    sample_image_at_patient_point(
                        profile_patient,
                        geometry,
                        image_array,
                    )
                )

                profile_model = (
                    sample_model_at_patient_point(
                        profile_patient,
                        geometry,
                        logits,
                        probabilities,
                    )
                )

                if (
                    profile_image is None
                    or profile_model is None
                ):

                    continue

                records.append(
                    {
                        "record_type":
                            "row_profile",

                        "study_id":
                            study_id,

                        "series_id":
                            series_id,

                        "point_index":
                            int(point_index),

                        "level":
                            str(
                                point["level"]
                            ),

                        "criterion":
                            criterion,

                        "profile_distance_mm":
                            float(
                                signed_distance
                            ),

                        "profile_axis":
                            "row",

                        "profile_image_intensity":
                            float(
                                profile_image[
                                    "image_intensity"
                                ]
                            ),

                        "profile_rfnn_probability":
                            float(
                                profile_model[
                                    "rfnn_probability"
                                ]
                            ),

                        "profile_background_probability":
                            float(
                                profile_model[
                                    "background_probability"
                                ]
                            ),

                        "profile_scs_probability":
                            float(
                                profile_model[
                                    "scs_probability"
                                ]
                            ),

                        "profile_lss_probability":
                            float(
                                profile_model[
                                    "lss_probability"
                                ]
                            ),

                        "profile_predicted_class":
                            profile_model[
                                "predicted_class"
                            ],
                    }
                )

                profile_patient = (
                    annotation_patient
                    + signed_distance
                    * column_direction
                )

                profile_image = (
                    sample_image_at_patient_point(
                        profile_patient,
                        geometry,
                        image_array,
                    )
                )

                profile_model = (
                    sample_model_at_patient_point(
                        profile_patient,
                        geometry,
                        logits,
                        probabilities,
                    )
                )

                if (
                    profile_image is None
                    or profile_model is None
                ):

                    continue

                records.append(
                    {
                        "record_type":
                            "column_profile",

                        "study_id":
                            study_id,

                        "series_id":
                            series_id,

                        "point_index":
                            int(point_index),

                        "level":
                            str(
                                point["level"]
                            ),

                        "criterion":
                            criterion,

                        "profile_distance_mm":
                            float(
                                signed_distance
                            ),

                        "profile_axis":
                            "column",

                        "profile_image_intensity":
                            float(
                                profile_image[
                                    "image_intensity"
                                ]
                            ),

                        "profile_rfnn_probability":
                            float(
                                profile_model[
                                    "rfnn_probability"
                                ]
                            ),

                        "profile_background_probability":
                            float(
                                profile_model[
                                    "background_probability"
                                ]
                            ),

                        "profile_scs_probability":
                            float(
                                profile_model[
                                    "scs_probability"
                                ]
                            ),

                        "profile_lss_probability":
                            float(
                                profile_model[
                                    "lss_probability"
                                ]
                            ),

                        "profile_predicted_class":
                            profile_model[
                                "predicted_class"
                            ],
                    }
                )

            # ---------------------------------------------------------------
            # Main evidence displacement record.
            # ---------------------------------------------------------------

            records.append(
                {
                    "record_type":
                        "evidence_maximum",

                    "study_id":
                        study_id,

                    "series_id":
                        series_id,

                    "point_index":
                        int(point_index),

                    "level":
                        str(
                            point["level"]
                        ),

                    "criterion":
                        criterion,

                    "profile_distance_mm":
                        np.nan,

                    "profile_axis":
                        "evidence_maximum",

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
                            max_patient[0]
                        ),

                    "maximum_patient_y":
                        float(
                            max_patient[1]
                        ),

                    "maximum_patient_z":
                        float(
                            max_patient[2]
                        ),

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
                        physical_distance,

                    "row_component_mm":
                        row_component,

                    "column_component_mm":
                        column_component,

                    "slice_normal_component_mm":
                        normal_component,

                    "inplane_distance_mm":
                        inplane_distance,

                    "inplane_angle_deg":
                        float(
                            angle_deg
                        ),

                    "annotation_image_intensity":
                        float(
                            annotation_image[
                                "image_intensity"
                            ]
                        ),

                    "maximum_image_intensity":
                        float(
                            maximum_image[
                                "image_intensity"
                            ]
                        ),

                    "intensity_difference":
                        float(
                            maximum_image[
                                "image_intensity"
                            ]
                            - annotation_image[
                                "image_intensity"
                            ]
                        ),

                    "annotation_rfnn_probability":
                        float(
                            annotation_model[
                                "rfnn_probability"
                            ]
                        ),

                    "maximum_rfnn_probability":
                        float(
                            maximum_model[
                                "rfnn_probability"
                            ]
                        ),

                    "annotation_background_probability":
                        float(
                            annotation_model[
                                "background_probability"
                            ]
                        ),

                    "maximum_background_probability":
                        float(
                            maximum_model[
                                "background_probability"
                            ]
                        ),

                    "annotation_predicted_class":
                        annotation_model[
                            "predicted_class"
                        ],

                    "maximum_predicted_class":
                        maximum_model[
                            "predicted_class"
                        ],
                }
            )

    return records


# ============================================================================
# MAIN SUMMARY
# ============================================================================

def build_summaries(
    records,
):

    df = pd.DataFrame(
        records
    )

    if df.empty:

        raise RuntimeError(
            "No Part 2.53 records generated."
        )

    maxima = df[
        df["record_type"]
        == "evidence_maximum"
    ].copy()

    profiles = df[
        df["record_type"]
        != "evidence_maximum"
    ].copy()

    expected_maxima = 225

    if len(maxima) != expected_maxima:

        raise RuntimeError(
            "Expected 225 evidence-maximum records, "
            f"found {len(maxima)}."
        )

    print()
    print("=" * 80)
    print("PART 2.53 EVIDENCE-MAXIMUM SUMMARY")
    print("=" * 80)

    summary_rows = []

    for criterion in sorted(
        maxima["criterion"].unique()
    ):

        subset = maxima[
            maxima["criterion"]
            == criterion
        ]

        summary_rows.append(
            {
                "criterion":
                    criterion,

                "count":
                    len(subset),

                "mean_physical_distance_mm":
                    float(
                        subset[
                            "physical_distance_mm"
                        ].mean()
                    ),

                "mean_inplane_distance_mm":
                    float(
                        subset[
                            "inplane_distance_mm"
                        ].mean()
                    ),

                "mean_abs_row_component_mm":
                    float(
                        subset[
                            "row_component_mm"
                        ].abs().mean()
                    ),

                "mean_abs_column_component_mm":
                    float(
                        subset[
                            "column_component_mm"
                        ].abs().mean()
                    ),

                "mean_abs_normal_component_mm":
                    float(
                        subset[
                            "slice_normal_component_mm"
                        ].abs().mean()
                    ),

                "mean_inplane_angle_deg":
                    float(
                        subset[
                            "inplane_angle_deg"
                        ].mean()
                    ),

                "mean_annotation_intensity":
                    float(
                        subset[
                            "annotation_image_intensity"
                        ].mean()
                    ),

                "mean_maximum_intensity":
                    float(
                        subset[
                            "maximum_image_intensity"
                        ].mean()
                    ),

                "mean_intensity_difference":
                    float(
                        subset[
                            "intensity_difference"
                        ].mean()
                    ),

                "mean_annotation_rfnn_probability":
                    float(
                        subset[
                            "annotation_rfnn_probability"
                        ].mean()
                    ),

                "mean_maximum_rfnn_probability":
                    float(
                        subset[
                            "maximum_rfnn_probability"
                        ].mean()
                    ),
            }
        )

    global_df = pd.DataFrame(
        summary_rows
    )

    print(
        global_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # WINNER AT EVIDENCE MAXIMUM
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.53 PREDICTED CLASS AT EVIDENCE MAXIMUM")
    print("=" * 80)

    winner_rows = []

    for criterion in sorted(
        maxima["criterion"].unique()
    ):

        subset = maxima[
            maxima["criterion"]
            == criterion
        ]

        counts = (
            subset[
                "maximum_predicted_class"
            ]
            .value_counts()
        )

        for class_name in CLASS_NAMES.values():

            winner_rows.append(
                {
                    "criterion":
                        criterion,

                    "predicted_class":
                        class_name,

                    "count":
                        int(
                            counts.get(
                                class_name,
                                0,
                            )
                        ),

                    "fraction":
                        float(
                            counts.get(
                                class_name,
                                0,
                            )
                            / len(subset)
                        ),
                }
            )

    winner_df = pd.DataFrame(
        winner_rows
    )

    print(
        winner_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # DIRECTION SUMMARY
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.53 IN-PLANE DIRECTION SUMMARY")
    print("=" * 80)

    direction_rows = []

    for criterion in sorted(
        maxima["criterion"].unique()
    ):

        subset = maxima[
            maxima["criterion"]
            == criterion
        ]

        angle = subset[
            "inplane_angle_deg"
        ]

        direction_rows.append(
            {
                "criterion":
                    criterion,

                "mean_angle_deg":
                    float(
                        angle.mean()
                    ),

                "median_angle_deg":
                    float(
                        angle.median()
                    ),

                "std_angle_deg":
                    float(
                        angle.std(
                            ddof=0
                        )
                    ),

                "positive_row_fraction":
                    float(
                        (
                            subset[
                                "row_component_mm"
                            ]
                            > 0
                        ).mean()
                    ),

                "negative_row_fraction":
                    float(
                        (
                            subset[
                                "row_component_mm"
                            ]
                            < 0
                        ).mean()
                    ),

                "positive_column_fraction":
                    float(
                        (
                            subset[
                                "column_component_mm"
                            ]
                            > 0
                        ).mean()
                    ),

                "negative_column_fraction":
                    float(
                        (
                            subset[
                                "column_component_mm"
                            ]
                            < 0
                        ).mean()
                    ),
            }
        )

    direction_df = pd.DataFrame(
        direction_rows
    )

    print(
        direction_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # PROFILE SUMMARY
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.53 DIRECTIONAL PROFILE SUMMARY")
    print("=" * 80)

    if not profiles.empty:

        profile_summary = (
            profiles.groupby(
                [
                    "criterion",
                    "profile_axis",
                    "profile_distance_mm",
                ],
                as_index=False,
            )
            .agg(
                mean_image_intensity=(
                    "profile_image_intensity",
                    "mean",
                ),
                mean_rfnn_probability=(
                    "profile_rfnn_probability",
                    "mean",
                ),
                mean_background_probability=(
                    "profile_background_probability",
                    "mean",
                ),
                mean_scs_probability=(
                    "profile_scs_probability",
                    "mean",
                ),
                mean_lss_probability=(
                    "profile_lss_probability",
                    "mean",
                ),
            )
        )

        print(
            profile_summary.to_string(
                index=False
            )
        )

    else:

        profile_summary = pd.DataFrame()

        print(
            "No profile records generated."
        )

    # ========================================================================
    # SAVE
    # ========================================================================

    records_path = (
        OUTPUT_DIR
        / "part253_directional_context_records.csv"
    )

    global_path = (
        OUTPUT_DIR
        / "part253_evidence_maximum_summary.csv"
    )

    winner_path = (
        OUTPUT_DIR
        / "part253_evidence_maximum_winner_summary.csv"
    )

    direction_path = (
        OUTPUT_DIR
        / "part253_inplane_direction_summary.csv"
    )

    profile_path = (
        OUTPUT_DIR
        / "part253_directional_profile_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part253_directional_context_summary.json"
    )

    df.to_csv(
        records_path,
        index=False,
    )

    global_df.to_csv(
        global_path,
        index=False,
    )

    winner_df.to_csv(
        winner_path,
        index=False,
    )

    direction_df.to_csv(
        direction_path,
        index=False,
    )

    profile_summary.to_csv(
        profile_path,
        index=False,
    )

    json_summary = {
        "validation_studies":
            25,

        "validation_series":
            25,

        "rfnn_points":
            45,

        "evidence_maximum_records":
            int(
                len(maxima)
            ),

        "profile_records":
            int(
                len(profiles)
            ),

        "criteria":
            sorted(
                maxima[
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

    print()
    print("=" * 80)
    print("PART 2.53 COMPLETE")
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
        "Part 2.52 unchanged."
    )

    print(
        "Dashboard unchanged."
    )

    print()
    print("Outputs:")

    print(records_path)
    print(global_path)
    print(winner_path)
    print(direction_path)
    print(profile_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.53")
    print("RFNN IN-PLANE DIRECTIONAL / ANATOMICAL CONTEXT AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    # ------------------------------------------------------------------------
    # Load inputs.
    # ------------------------------------------------------------------------

    part251 = load_part251()

    part252 = load_part252()

    # ------------------------------------------------------------------------
    # Manifest and exact validation cohort.
    # ------------------------------------------------------------------------

    manifest = (
        part220b.load_manifest()
    )

    print()
    print(
        f"Full manifest rows: "
        f"{len(manifest)}"
    )

    validation_cases = (
        build_validation_cases(
            manifest
        )
    )

    print(
        f"Validation cases: "
        f"{len(validation_cases)}"
    )

    # ------------------------------------------------------------------------
    # Model.
    # ------------------------------------------------------------------------

    model = build_model()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = (
        model
        .to(device)
        .eval()
    )

    # ------------------------------------------------------------------------
    # Analyze.
    # ------------------------------------------------------------------------

    all_records = []

    successful_cases = 0

    print()
    print("=" * 80)
    print("RUNNING PART 2.53")
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

        case251 = part251[
            (
                part251["study_id"]
                .astype(str)
                == study_id
            )
            &
            (
                part251["series_id"]
                .astype(str)
                == series_id
            )
        ].copy()

        case252 = part252[
            (
                part252["study_id"]
                .astype(str)
                == study_id
            )
            &
            (
                part252["series_id"]
                .astype(str)
                == series_id
            )
        ].copy()

        try:

            records = analyze_case(
                model,
                case,
                case251,
                case252,
                device,
            )

            all_records.extend(
                records
            )

            successful_cases += 1

            maxima_count = sum(
                1
                for record in records
                if record[
                    "record_type"
                ]
                == "evidence_maximum"
            )

            profile_count = sum(
                1
                for record in records
                if record[
                    "record_type"
                ]
                != "evidence_maximum"
            )

            print(
                f"[{case_number:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id} | "
                f"Maxima={maxima_count} | "
                f"Profiles={profile_count}"
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
        f"Total records: "
        f"{len(all_records)}"
    )

    build_summaries(
        all_records
    )


if __name__ == "__main__":
    main()