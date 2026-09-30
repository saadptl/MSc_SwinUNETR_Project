"""
PART 2.54
PAIRED LFNN / RFNN DISPLACEMENT SYMMETRY AUDIT

Purpose
-------
Compare Left Neural Foraminal Narrowing (LFNN) and
Right Neural Foraminal Narrowing (RFNN) on the SAME:

    study
    series
    spinal level

This controls much of the imaging context and asks whether the model
shows a systematic spatial asymmetry between the left and right
foraminal annotations.

For each LFNN/RFNN point:

    1. Locate annotation in canonical physical-space coordinates.
    2. Find strongest local class-probability evidence within radius 6.
    3. Convert evidence maximum back to patient physical space.
    4. Decompose displacement into DICOM row / column / normal axes.
    5. Compare LFNN and RFNN displacement.
    6. Compare class-vs-background evidence.
    7. Compare local MRI intensity.
    8. Compare paired spatial behavior.

Expected:
    25 validation series
    45 LFNN points
    45 RFNN points
    45 paired LFNN/RFNN observations

Analysis only.

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part254_paired_lfnn_rfnn_displacement_symmetry_audit"
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

LFNN_NAME = (
    "Left Neural Foraminal Narrowing"
)

RFNN_NAME = (
    "Right Neural Foraminal Narrowing"
)

BACKGROUND_ID = 0

LFNN_ID = 2

RFNN_ID = 3

LOCAL_RADIUS = 6

MAX_INVERSE_ITERATIONS = 20

INVERSE_TOLERANCE = 1e-4

FINITE_DIFFERENCE_MM = 0.5

MAX_INVERSE_STEP_MM = 10.0

INVERSE_DAMPING = 0.8


CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
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
# CANONICAL -> PATIENT NUMERICAL INVERSE
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
                "did not return three coordinates."
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
            f"Final canonical error={error:.6f}"
        )

    return patient, error


# ============================================================================
# MODEL
# ============================================================================

def build_model():

    print()
    print("=" * 80)
    print("LOADING PART 2.27 CHECKPOINT")
    print("=" * 80)

    model = (
        part220b.build_model()
    )

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
            "Checkpoint has missing keys."
        )

    if unexpected:

        raise RuntimeError(
            "Checkpoint has unexpected keys."
        )

    return model


# ============================================================================
# EXACT VALIDATION COHORT
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
            "Expected exactly 25 validation series."
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
# SAMPLE VOXEL
# ============================================================================

def sample_voxel(
    tensor,
    z,
    y,
    x,
):

    if tensor.ndim == 5:

        tensor = tensor[0]

    if tensor.ndim == 4:

        # C,Z,Y,X
        channels = tensor.shape[0]

        if (
            z < 0
            or z >= tensor.shape[1]
            or y < 0
            or y >= tensor.shape[2]
            or x < 0
            or x >= tensor.shape[3]
        ):

            return None

        return (
            tensor[
                :,
                z,
                y,
                x,
            ]
            .detach()
            .cpu()
            .numpy()
        )

    if tensor.ndim == 3:

        if (
            z < 0
            or z >= tensor.shape[0]
            or y < 0
            or y >= tensor.shape[1]
            or x < 0
            or x >= tensor.shape[2]
        ):

            return None

        return float(
            tensor[
                z,
                y,
                x,
            ]
        )

    raise RuntimeError(
        f"Unexpected tensor shape: {tensor.shape}"
    )


# ============================================================================
# POINT -> CANONICAL
# ============================================================================

def point_to_canonical(
    point,
    geometry,
):

    patient = np.array(
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

    canonical = (
        part220b.patient_point_to_canonical(
            patient,
            geometry,
        )
    )

    canonical = np.asarray(
        canonical,
        dtype=np.float64,
    ).reshape(-1)

    return (
        patient,
        canonical[:3],
    )


# ============================================================================
# LOCAL CLASS EVIDENCE
# ============================================================================

def find_local_class_maximum(
    probabilities,
    center_canonical,
    class_id,
    radius,
):

    if probabilities.ndim == 5:

        probabilities = probabilities[0]

    if probabilities.ndim != 4:

        raise RuntimeError(
            "Expected probability tensor "
            "with shape C,Z,Y,X."
        )

    _, depth, height, width = (
        probabilities.shape
    )

    center_z = int(
        round(
            float(
                center_canonical[0]
            )
        )
    )

    center_y = int(
        round(
            float(
                center_canonical[1]
            )
        )
    )

    center_x = int(
        round(
            float(
                center_canonical[2]
            )
        )
    )

    z0 = max(
        0,
        center_z - radius,
    )

    z1 = min(
        depth - 1,
        center_z + radius,
    )

    y0 = max(
        0,
        center_y - radius,
    )

    y1 = min(
        height - 1,
        center_y + radius,
    )

    x0 = max(
        0,
        center_x - radius,
    )

    x1 = min(
        width - 1,
        center_x + radius,
    )

    local = (
        probabilities[
            class_id,
            z0:z1 + 1,
            y0:y1 + 1,
            x0:x1 + 1,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    local_index = np.unravel_index(
        np.argmax(local),
        local.shape,
    )

    max_z = (
        z0
        + int(
            local_index[0]
        )
    )

    max_y = (
        y0
        + int(
            local_index[1]
        )
    )

    max_x = (
        x0
        + int(
            local_index[2]
        )
    )

    return {
        "z": max_z,
        "y": max_y,
        "x": max_x,

        "probability":
            float(
                local[
                    local_index
                ]
            ),

        "center_z":
            center_z,

        "center_y":
            center_y,

        "center_x":
            center_x,

        "distance_voxels":
            float(
                np.sqrt(
                    (
                        max_z
                        - center_canonical[0]
                    ) ** 2
                    +
                    (
                        max_y
                        - center_canonical[1]
                    ) ** 2
                    +
                    (
                        max_x
                        - center_canonical[2]
                    ) ** 2
                )
            ),
    }


# ============================================================================
# SAMPLE IMAGE
# ============================================================================

def sample_image(
    image,
    canonical,
):

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    if image_array.ndim == 4:

        image_array = image_array[0]

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

        return np.nan

    return float(
        image_array[
            z,
            y,
            x,
        ]
    )


# ============================================================================
# DICOM DISPLACEMENT
# ============================================================================

def displacement_components(
    annotation_patient,
    maximum_patient,
    geometry,
):

    displacement = (
        maximum_patient
        - annotation_patient
    )

    (
        row_direction,
        column_direction,
        normal_direction,
    ) = get_geometry_axes(
        geometry
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

    inplane_distance = float(
        np.sqrt(
            row_component ** 2
            +
            column_component ** 2
        )
    )

    physical_distance = float(
        np.linalg.norm(
            displacement
        )
    )

    if inplane_distance > 0:

        angle = math.degrees(
            math.atan2(
                column_component,
                row_component,
            )
        )

    else:

        angle = 0.0

    return (
        displacement,
        row_component,
        column_component,
        normal_component,
        inplane_distance,
        physical_distance,
        angle,
    )


# ============================================================================
# ANALYZE ONE POINT
# ============================================================================

def analyze_point(
    point,
    logits,
    probabilities,
    image,
    geometry,
    class_id,
    class_name,
    study_id,
    series_id,
):

    annotation_patient, annotation_canonical = (
        point_to_canonical(
            point,
            geometry,
        )
    )

    evidence = (
        find_local_class_maximum(
            probabilities,
            annotation_canonical,
            class_id,
            LOCAL_RADIUS,
        )
    )

    maximum_canonical = np.array(
        [
            float(
                evidence["z"]
            ),
            float(
                evidence["y"]
            ),
            float(
                evidence["x"]
            ),
        ],
        dtype=np.float64,
    )

    maximum_patient, inverse_error = (
        canonical_to_patient(
            maximum_canonical,
            annotation_patient,
            geometry,
        )
    )

    (
        displacement,
        row_component,
        column_component,
        normal_component,
        inplane_distance,
        physical_distance,
        angle,
    ) = displacement_components(
        annotation_patient,
        maximum_patient,
        geometry,
    )

    # ------------------------------------------------------------------------
    # Annotation voxel probabilities
    # ------------------------------------------------------------------------

    az = int(
        round(
            float(
                annotation_canonical[0]
            )
        )
    )

    ay = int(
        round(
            float(
                annotation_canonical[1]
            )
        )
    )

    ax = int(
        round(
            float(
                annotation_canonical[2]
            )
        )
    )

    annotation_prob = sample_voxel(
        probabilities,
        az,
        ay,
        ax,
    )

    annotation_logit = sample_voxel(
        logits,
        az,
        ay,
        ax,
    )

    mz = evidence["z"]

    my = evidence["y"]

    mx = evidence["x"]

    maximum_prob = sample_voxel(
        probabilities,
        mz,
        my,
        mx,
    )

    maximum_logit = sample_voxel(
        logits,
        mz,
        my,
        mx,
    )

    if (
        annotation_prob is None
        or annotation_logit is None
        or maximum_prob is None
        or maximum_logit is None
    ):

        raise RuntimeError(
            "Could not sample annotation/evidence voxel."
        )

    annotation_intensity = sample_image(
        image,
        annotation_canonical,
    )

    maximum_intensity = sample_image(
        image,
        maximum_canonical,
    )

    # ------------------------------------------------------------------------
    # Local evidence margins
    # ------------------------------------------------------------------------

    annotation_class_probability = float(
        annotation_prob[class_id]
    )

    maximum_class_probability = float(
        maximum_prob[class_id]
    )

    annotation_background_probability = float(
        annotation_prob[
            BACKGROUND_ID
        ]
    )

    maximum_background_probability = float(
        maximum_prob[
            BACKGROUND_ID
        ]
    )

    annotation_margin = (
        annotation_class_probability
        - annotation_background_probability
    )

    maximum_margin = (
        maximum_class_probability
        - maximum_background_probability
    )

    annotation_predicted_class = int(
        np.argmax(
            annotation_prob
        )
    )

    maximum_predicted_class = int(
        np.argmax(
            maximum_prob
        )
    )

    return {
        "study_id":
            study_id,

        "series_id":
            series_id,

        "level":
            str(
                point["level"]
            ),

        "class_name":
            class_name,

        "class_id":
            class_id,

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

        "annotation_canonical_z":
            float(
                annotation_canonical[0]
            ),

        "annotation_canonical_y":
            float(
                annotation_canonical[1]
            ),

        "annotation_canonical_x":
            float(
                annotation_canonical[2]
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

        "maximum_canonical_z":
            float(
                maximum_canonical[0]
            ),

        "maximum_canonical_y":
            float(
                maximum_canonical[1]
            ),

        "maximum_canonical_x":
            float(
                maximum_canonical[2]
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
            angle,

        "canonical_distance_voxels":
            float(
                evidence[
                    "distance_voxels"
                ]
            ),

        "inverse_error":
            float(
                inverse_error
            ),

        "annotation_image_intensity":
            float(
                annotation_intensity
            ),

        "maximum_image_intensity":
            float(
                maximum_intensity
            ),

        "intensity_difference":
            float(
                maximum_intensity
                - annotation_intensity
            ),

        "annotation_class_probability":
            annotation_class_probability,

        "maximum_class_probability":
            maximum_class_probability,

        "annotation_background_probability":
            annotation_background_probability,

        "maximum_background_probability":
            maximum_background_probability,

        "annotation_class_minus_background":
            annotation_margin,

        "maximum_class_minus_background":
            maximum_margin,

        "annotation_predicted_class":
            CLASS_NAMES[
                annotation_predicted_class
            ],

        "maximum_predicted_class":
            CLASS_NAMES[
                maximum_predicted_class
            ],

        "maximum_predicted_class_id":
            maximum_predicted_class,

        "maximum_scs_probability":
            float(
                maximum_prob[1]
            ),

        "maximum_lfnn_probability":
            float(
                maximum_prob[2]
            ),

        "maximum_rfnn_probability":
            float(
                maximum_prob[3]
            ),

        "maximum_lss_probability":
            float(
                maximum_prob[4]
            ),

        "maximum_rss_probability":
            float(
                maximum_prob[5]
            ),
    }


# ============================================================================
# BUILD PAIRED RECORDS
# ============================================================================

def build_pairs(
    records_df,
):

    # ------------------------------------------------------------------------
    # Same study + same series + same level
    # ------------------------------------------------------------------------

    lfnn = records_df[
        records_df["class_name"]
        == LFNN_NAME
    ].copy()

    rfnn = records_df[
        records_df["class_name"]
        == RFNN_NAME
    ].copy()

    pairs = lfnn.merge(
        rfnn,
        on=[
            "study_id",
            "series_id",
            "level",
        ],
        suffixes=(
            "_lfnn",
            "_rfnn",
        ),
    )

    # ------------------------------------------------------------------------
    # There should be exactly 45 paired observations.
    # ------------------------------------------------------------------------

    if len(pairs) != 45:

        raise RuntimeError(
            "Expected 45 paired LFNN/RFNN observations, "
            f"found {len(pairs)}."
        )

    return pairs


# ============================================================================
# SUMMARY
# ============================================================================

def build_summaries(
    records_df,
):

    pairs = build_pairs(
        records_df
    )

    # ========================================================================
    # GLOBAL PAIRED SUMMARY
    # ========================================================================

    global_rows = []

    metrics = [
        "physical_distance_mm",
        "inplane_distance_mm",
        "row_component_mm",
        "column_component_mm",
        "slice_normal_component_mm",
        "canonical_distance_voxels",
        "maximum_class_probability",
        "maximum_class_minus_background",
        "intensity_difference",
    ]

    for metric in metrics:

        lfnn_values = (
            pairs[
                f"{metric}_lfnn"
            ]
            .astype(float)
        )

        rfnn_values = (
            pairs[
                f"{metric}_rfnn"
            ]
            .astype(float)
        )

        difference = (
            rfnn_values
            - lfnn_values
        )

        global_rows.append(
            {
                "metric":
                    metric,

                "lfnn_mean":
                    float(
                        lfnn_values.mean()
                    ),

                "rfnn_mean":
                    float(
                        rfnn_values.mean()
                    ),

                "rfnn_minus_lfnn_mean":
                    float(
                        difference.mean()
                    ),

                "absolute_difference_mean":
                    float(
                        difference.abs().mean()
                    ),

                "lfnn_median":
                    float(
                        lfnn_values.median()
                    ),

                "rfnn_median":
                    float(
                        rfnn_values.median()
                    ),
            }
        )

    global_df = pd.DataFrame(
        global_rows
    )

    # ========================================================================
    # LEVEL SUMMARY
    # ========================================================================

    level_rows = []

    for level in sorted(
        pairs["level"].unique()
    ):

        subset = pairs[
            pairs["level"]
            == level
        ]

        for metric in [
            "physical_distance_mm",
            "inplane_distance_mm",
            "slice_normal_component_mm",
            "maximum_class_probability",
            "maximum_class_minus_background",
        ]:

            lfnn_values = (
                subset[
                    f"{metric}_lfnn"
                ]
                .astype(float)
            )

            rfnn_values = (
                subset[
                    f"{metric}_rfnn"
                ]
                .astype(float)
            )

            level_rows.append(
                {
                    "level":
                        level,

                    "metric":
                        metric,

                    "count":
                        len(subset),

                    "lfnn_mean":
                        float(
                            lfnn_values.mean()
                        ),

                    "rfnn_mean":
                        float(
                            rfnn_values.mean()
                        ),

                    "rfnn_minus_lfnn":
                        float(
                            (
                                rfnn_values
                                - lfnn_values
                            ).mean()
                        ),
                }
            )

    level_df = pd.DataFrame(
        level_rows
    )

    # ========================================================================
    # DIRECTION SYMMETRY
    # ========================================================================

    direction_rows = []

    for side in [
        "lfnn",
        "rfnn",
    ]:

        subset = pairs

        row_component = subset[
            f"row_component_mm_{side}"
        ].astype(float)

        column_component = subset[
            f"column_component_mm_{side}"
        ].astype(float)

        normal_component = subset[
            f"slice_normal_component_mm_{side}"
        ].astype(float)

        direction_rows.append(
            {
                "side":
                    side.upper(),

                "mean_row_component_mm":
                    float(
                        row_component.mean()
                    ),

                "mean_column_component_mm":
                    float(
                        column_component.mean()
                    ),

                "mean_normal_component_mm":
                    float(
                        normal_component.mean()
                    ),

                "mean_abs_row_component_mm":
                    float(
                        row_component.abs().mean()
                    ),

                "mean_abs_column_component_mm":
                    float(
                        column_component.abs().mean()
                    ),

                "mean_abs_normal_component_mm":
                    float(
                        normal_component.abs().mean()
                    ),

                "positive_row_fraction":
                    float(
                        (
                            row_component
                            > 0
                        ).mean()
                    ),

                "positive_column_fraction":
                    float(
                        (
                            column_component
                            > 0
                        ).mean()
                    ),

                "positive_normal_fraction":
                    float(
                        (
                            normal_component
                            > 0
                        ).mean()
                    ),
            }
        )

    direction_df = pd.DataFrame(
        direction_rows
    )

    # ========================================================================
    # PAIRWISE DIRECTION DIFFERENCE
    # ========================================================================

    pair_direction_df = pd.DataFrame(
        {
            "study_id":
                pairs[
                    "study_id"
                ],

            "series_id":
                pairs[
                    "series_id"
                ],

            "level":
                pairs[
                    "level"
                ],

            "row_difference_rfnn_minus_lfnn":
                (
                    pairs[
                        "row_component_mm_rfnn"
                    ]
                    -
                    pairs[
                        "row_component_mm_lfnn"
                    ]
                ),

            "column_difference_rfnn_minus_lfnn":
                (
                    pairs[
                        "column_component_mm_rfnn"
                    ]
                    -
                    pairs[
                        "column_component_mm_lfnn"
                    ]
                ),

            "normal_difference_rfnn_minus_lfnn":
                (
                    pairs[
                        "slice_normal_component_mm_rfnn"
                    ]
                    -
                    pairs[
                        "slice_normal_component_mm_lfnn"
                    ]
                ),

            "distance_difference_rfnn_minus_lfnn":
                (
                    pairs[
                        "physical_distance_mm_rfnn"
                    ]
                    -
                    pairs[
                        "physical_distance_mm_lfnn"
                    ]
                ),

            "inplane_difference_rfnn_minus_lfnn":
                (
                    pairs[
                        "inplane_distance_mm_rfnn"
                    ]
                    -
                    pairs[
                        "inplane_distance_mm_lfnn"
                    ]
                ),

            "probability_difference_rfnn_minus_lfnn":
                (
                    pairs[
                        "maximum_class_probability_rfnn"
                    ]
                    -
                    pairs[
                        "maximum_class_probability_lfnn"
                    ]
                ),

            "margin_difference_rfnn_minus_lfnn":
                (
                    pairs[
                        "maximum_class_minus_background_rfnn"
                    ]
                    -
                    pairs[
                        "maximum_class_minus_background_lfnn"
                    ]
                ),

            "intensity_difference_rfnn_minus_lfnn":
                (
                    pairs[
                        "intensity_difference_rfnn"
                    ]
                    -
                    pairs[
                        "intensity_difference_lfnn"
                    ]
                ),
        }
    )

    # ========================================================================
    # PRINT
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.54 PAIRED LFNN / RFNN SUMMARY")
    print("=" * 80)

    print(
        global_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.54 DIRECTION SYMMETRY")
    print("=" * 80)

    print(
        direction_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.54 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    records_path = (
        OUTPUT_DIR
        / "part254_paired_displacement_records.csv"
    )

    global_path = (
        OUTPUT_DIR
        / "part254_paired_displacement_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part254_level_symmetry_summary.csv"
    )

    direction_path = (
        OUTPUT_DIR
        / "part254_direction_symmetry_summary.csv"
    )

    pair_direction_path = (
        OUTPUT_DIR
        / "part254_pairwise_direction_differences.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part254_paired_displacement_audit_summary.json"
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

    direction_df.to_csv(
        direction_path,
        index=False,
    )

    pair_direction_df.to_csv(
        pair_direction_path,
        index=False,
    )

    json_summary = {
        "validation_series":
            25,

        "lfnn_points":
            45,

        "rfnn_points":
            45,

        "paired_observations":
            int(
                len(pairs)
            ),

        "global_summary":
            global_df.to_dict(
                orient="records"
            ),

        "direction_summary":
            direction_df.to_dict(
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
    print("PART 2.54 COMPLETE")
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
    print(direction_path)
    print(pair_direction_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.54")
    print("PAIRED LFNN / RFNN DISPLACEMENT SYMMETRY AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    # ------------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------------

    manifest = (
        part220b.load_manifest()
    )

    print()
    print(
        f"Full manifest rows: "
        f"{len(manifest)}"
    )

    # ------------------------------------------------------------------------
    # Exact cohort
    # ------------------------------------------------------------------------

    cases = (
        build_validation_cases(
            manifest
        )
    )

    print(
        f"Validation cases: "
        f"{len(cases)}"
    )

    # ------------------------------------------------------------------------
    # Model
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
    # Analyze all 25 series.
    # ------------------------------------------------------------------------

    all_records = []

    successful_cases = 0

    total_lfnn = 0

    total_rfnn = 0

    print()
    print("=" * 80)
    print("RUNNING PAIRED LFNN / RFNN ANALYSIS")
    print("=" * 80)

    for case_number, case in enumerate(
        cases,
        start=1,
    ):

        study_id = str(
            case["study_id"]
        )

        series_id = str(
            case["series_id"]
        )

        try:

            (
                image,
                points,
                geometry,
            ) = part220b.load_case(
                study_id,
                series_id,
                case["points"],
            )

            logits, probabilities = (
                run_model(
                    model,
                    image,
                    device,
                )
            )

            # ---------------------------------------------------------------
            # Convert transformed point list to DataFrame.
            # ---------------------------------------------------------------

            point_df = pd.DataFrame(
                points
            )

            if point_df.empty:

                raise RuntimeError(
                    "No transformed points."
                )

            lfnn_points = point_df[
                point_df[
                    "class_name"
                ]
                == LFNN_NAME
            ]

            rfnn_points = point_df[
                point_df[
                    "class_name"
                ]
                == RFNN_NAME
            ]

            case_lfnn = 0

            case_rfnn = 0

            # ---------------------------------------------------------------
            # LFNN
            # ---------------------------------------------------------------

            for _, point in (
                lfnn_points.iterrows()
            ):

                result = analyze_point(
                    point=point,
                    logits=logits,
                    probabilities=probabilities,
                    image=image,
                    geometry=geometry,
                    class_id=LFNN_ID,
                    class_name=LFNN_NAME,
                    study_id=study_id,
                    series_id=series_id,
                )

                all_records.append(
                    result
                )

                case_lfnn += 1

            # ---------------------------------------------------------------
            # RFNN
            # ---------------------------------------------------------------

            for _, point in (
                rfnn_points.iterrows()
            ):

                result = analyze_point(
                    point=point,
                    logits=logits,
                    probabilities=probabilities,
                    image=image,
                    geometry=geometry,
                    class_id=RFNN_ID,
                    class_name=RFNN_NAME,
                    study_id=study_id,
                    series_id=series_id,
                )

                all_records.append(
                    result
                )

                case_rfnn += 1

            total_lfnn += case_lfnn

            total_rfnn += case_rfnn

            successful_cases += 1

            print(
                f"[{case_number:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id} | "
                f"LFNN={case_lfnn} | "
                f"RFNN={case_rfnn}"
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
        f"LFNN points analyzed: "
        f"{total_lfnn}"
    )

    print(
        f"RFNN points analyzed: "
        f"{total_rfnn}"
    )

    # ------------------------------------------------------------------------
    # Exact expected totals
    # ------------------------------------------------------------------------

    if successful_cases != 25:

        raise RuntimeError(
            "Expected all 25 validation cases "
            "to succeed."
        )

    if total_lfnn != 45:

        raise RuntimeError(
            "Expected 45 LFNN points "
            f"but found {total_lfnn}."
        )

    if total_rfnn != 45:

        raise RuntimeError(
            "Expected 45 RFNN points "
            f"but found {total_rfnn}."
        )

    records_df = pd.DataFrame(
        all_records
    )

    if len(records_df) != 90:

        raise RuntimeError(
            "Expected 90 LFNN/RFNN records "
            f"but found {len(records_df)}."
        )

    # ------------------------------------------------------------------------
    # Paired analysis
    # ------------------------------------------------------------------------

    build_summaries(
        records_df
    )


if __name__ == "__main__":
    main()