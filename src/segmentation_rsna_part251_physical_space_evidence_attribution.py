"""
PART 2.51
PHYSICAL-SPACE RFNN EVIDENCE ATTRIBUTION

Purpose
-------
For the exact Part 2.20B validation cohort, analyze the physical-space
location of the strongest RFNN-related model evidence.

For every RFNN annotation:

    1. RFNN annotation point
    2. Maximum RFNN probability
    3. Maximum RFNN logit
    4. Maximum RFNN - Background margin
    5. Maximum RFNN - LSS margin
    6. Maximum RFNN - SCS margin

For every maximum, record:

    - model-grid location
    - physical patient-space location
    - physical displacement from RFNN annotation
    - winning predicted class
    - winning probability
    - RFNN probability
    - background probability
    - SCS probability
    - LFNN probability
    - LSS probability
    - RSS probability

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
    / "rsna_part251_physical_space_evidence_attribution"
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

LOCAL_RADIUS = 6

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

LEVELS = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]


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
                    "study_id":
                        str(
                            case["study_id"]
                        ),

                    "series_id":
                        str(
                            case["series_id"]
                        ),

                    "point_index":
                        int(point_index),

                    "class_id":
                        int(
                            row["class_id"]
                        ),

                    "class_name":
                        str(
                            row["class_name"]
                        ),

                    "level":
                        str(
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
# NUMERICAL CANONICAL -> PATIENT INVERSE
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

    max_iterations = 20

    tolerance = 1e-4

    finite_difference_mm = 0.5

    max_step_mm = 10.0

    damping = 0.8

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
                "did not return 3 values."
            )

        return result[:3]

    for _ in range(
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
                "Non-finite geometry inverse step."
            )

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

    final_canonical = forward(
        patient
    )

    final_error = float(
        np.linalg.norm(
            target
            - final_canonical
        )
    )

    if final_error > 0.05:

        raise RuntimeError(
            "Canonical-to-patient inverse did not "
            f"converge. Error={final_error:.6f}"
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

                voxel_probs = (
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
                        voxel_probs
                    )
                )

                records.append(
                    {
                        "z": z,
                        "y": y,
                        "x": x,

                        "dz":
                            z - center_z,

                        "dy":
                            y - center_y,

                        "dx":
                            x - center_x,

                        "distance_voxels":
                            float(
                                np.sqrt(
                                    (
                                        z
                                        - center_z
                                    ) ** 2
                                    + (
                                        y
                                        - center_y
                                    ) ** 2
                                    + (
                                        x
                                        - center_x
                                    ) ** 2
                                )
                            ),

                        "background_logit":
                            float(
                                voxel_logits[0]
                            ),

                        "scs_logit":
                            float(
                                voxel_logits[1]
                            ),

                        "lfnn_logit":
                            float(
                                voxel_logits[2]
                            ),

                        "rfnn_logit":
                            float(
                                voxel_logits[3]
                            ),

                        "lss_logit":
                            float(
                                voxel_logits[4]
                            ),

                        "rss_logit":
                            float(
                                voxel_logits[5]
                            ),

                        "background_probability":
                            float(
                                voxel_probs[0]
                            ),

                        "scs_probability":
                            float(
                                voxel_probs[1]
                            ),

                        "lfnn_probability":
                            float(
                                voxel_probs[2]
                            ),

                        "rfnn_probability":
                            float(
                                voxel_probs[3]
                            ),

                        "lss_probability":
                            float(
                                voxel_probs[4]
                            ),

                        "rss_probability":
                            float(
                                voxel_probs[5]
                            ),

                        "predicted_class_id":
                            prediction,

                        "predicted_class_name":
                            CLASS_NAMES[
                                prediction
                            ],

                        "rfnn_minus_background":
                            float(
                                voxel_logits[3]
                                - voxel_logits[0]
                            ),

                        "rfnn_minus_lss":
                            float(
                                voxel_logits[3]
                                - voxel_logits[4]
                            ),

                        "rfnn_minus_scs":
                            float(
                                voxel_logits[3]
                                - voxel_logits[1]
                            ),
                    }
                )

    return records


# ============================================================================
# CASE ANALYSIS
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
        point_df.index[
            point_df[
                "class_name"
            ]
            == "Right Neural Foraminal Narrowing"
        ]
        .tolist()
    )

    records = []

    for point_index in rfnn_indices:

        point = transformed_points[
            point_index
        ]

        annotation_model = np.array(
            [
                float(point["z"]),
                float(point["y"]),
                float(point["x"]),
            ],
            dtype=np.float64,
        )

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

        voxel_records = (
            build_local_voxel_records(
                logits,
                probabilities,
                annotation_model[0],
                annotation_model[1],
                annotation_model[2],
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

            max_index = (
                voxel_df[column]
                .astype(float)
                .idxmax()
            )

            maximum = (
                voxel_df.loc[
                    max_index
                ]
            )

            max_model = np.array(
                [
                    float(
                        maximum["z"]
                    ),
                    float(
                        maximum["y"]
                    ),
                    float(
                        maximum["x"]
                    ),
                ],
                dtype=np.float64,
            )

            max_patient = (
                canonical_to_patient(
                    max_model,
                    annotation_patient,
                    geometry,
                )
            )

            physical_delta = (
                max_patient
                - annotation_patient
            )

            physical_distance = float(
                np.linalg.norm(
                    physical_delta
                )
            )

            winner_id = int(
                maximum[
                    "predicted_class_id"
                ]
            )

            records.append(
                {
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

                    "criterion_value":
                        float(
                            maximum[
                                column
                            ]
                        ),

                    "annotation_model_z":
                        float(
                            annotation_model[0]
                        ),

                    "annotation_model_y":
                        float(
                            annotation_model[1]
                        ),

                    "annotation_model_x":
                        float(
                            annotation_model[2]
                        ),

                    "maximum_model_z":
                        float(
                            max_model[0]
                        ),

                    "maximum_model_y":
                        float(
                            max_model[1]
                        ),

                    "maximum_model_x":
                        float(
                            max_model[2]
                        ),

                    "model_dz":
                        float(
                            max_model[0]
                            - annotation_model[0]
                        ),

                    "model_dy":
                        float(
                            max_model[1]
                            - annotation_model[1]
                        ),

                    "model_dx":
                        float(
                            max_model[2]
                            - annotation_model[2]
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

                    "winner_class_id":
                        winner_id,

                    "winner_class":
                        CLASS_NAMES[
                            winner_id
                        ],

                    "background_probability":
                        float(
                            maximum[
                                "background_probability"
                            ]
                        ),

                    "scs_probability":
                        float(
                            maximum[
                                "scs_probability"
                            ]
                        ),

                    "lfnn_probability":
                        float(
                            maximum[
                                "lfnn_probability"
                            ]
                        ),

                    "rfnn_probability":
                        float(
                            maximum[
                                "rfnn_probability"
                            ]
                        ),

                    "lss_probability":
                        float(
                            maximum[
                                "lss_probability"
                            ]
                        ),

                    "rss_probability":
                        float(
                            maximum[
                                "rss_probability"
                            ]
                        ),
                }
            )

    return records


# ============================================================================
# SUMMARY
# ============================================================================

def build_summaries(
    records,
):

    df = pd.DataFrame(
        records
    )

    if df.empty:

        raise RuntimeError(
            "No evidence-attribution records generated."
        )

    if len(df) != 225:

        raise RuntimeError(
            "Expected 225 records "
            "(45 RFNN points × 5 criteria), "
            f"found {len(df)}."
        )

    # ========================================================================
    # GLOBAL
    # ========================================================================

    global_rows = []

    for criterion in sorted(
        df["criterion"].unique()
    ):

        subset = df[
            df["criterion"]
            == criterion
        ]

        winner_counts = (
            subset[
                "winner_class"
            ]
            .value_counts()
        )

        row = {
            "criterion":
                criterion,

            "count":
                len(subset),

            "mean_distance_mm":
                float(
                    subset[
                        "physical_distance_mm"
                    ].mean()
                ),

            "median_distance_mm":
                float(
                    subset[
                        "physical_distance_mm"
                    ].median()
                ),

            "within_5mm_fraction":
                float(
                    (
                        subset[
                            "physical_distance_mm"
                        ]
                        <= 5
                    ).mean()
                ),

            "within_10mm_fraction":
                float(
                    (
                        subset[
                            "physical_distance_mm"
                        ]
                        <= 10
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

            "mean_rfnn_probability":
                float(
                    subset[
                        "rfnn_probability"
                    ].mean()
                ),

            "mean_background_probability":
                float(
                    subset[
                        "background_probability"
                    ].mean()
                ),

            "mean_lss_probability":
                float(
                    subset[
                        "lss_probability"
                    ].mean()
                ),

            "mean_scs_probability":
                float(
                    subset[
                        "scs_probability"
                    ].mean()
                ),

            "winner_background":
                int(
                    winner_counts.get(
                        "Background",
                        0,
                    )
                ),

            "winner_scs":
                int(
                    winner_counts.get(
                        "Spinal Canal Stenosis",
                        0,
                    )
                ),

            "winner_lfnn":
                int(
                    winner_counts.get(
                        "Left Neural Foraminal Narrowing",
                        0,
                    )
                ),

            "winner_rfnn":
                int(
                    winner_counts.get(
                        "Right Neural Foraminal Narrowing",
                        0,
                    )
                ),

            "winner_lss":
                int(
                    winner_counts.get(
                        "Left Subarticular Stenosis",
                        0,
                    )
                ),

            "winner_rss":
                int(
                    winner_counts.get(
                        "Right Subarticular Stenosis",
                        0,
                    )
                ),
        }

        global_rows.append(
            row
        )

    global_df = pd.DataFrame(
        global_rows
    )

    # ========================================================================
    # LEVEL
    # ========================================================================

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

            winner_counts = (
                subset[
                    "winner_class"
                ]
                .value_counts()
            )

            level_rows.append(
                {
                    "level":
                        level,

                    "criterion":
                        criterion,

                    "count":
                        len(subset),

                    "mean_distance_mm":
                        float(
                            subset[
                                "physical_distance_mm"
                            ].mean()
                        ),

                    "median_distance_mm":
                        float(
                            subset[
                                "physical_distance_mm"
                            ].median()
                        ),

                    "within_5mm_fraction":
                        float(
                            (
                                subset[
                                    "physical_distance_mm"
                                ]
                                <= 5
                            ).mean()
                        ),

                    "within_10mm_fraction":
                        float(
                            (
                                subset[
                                    "physical_distance_mm"
                                ]
                                <= 10
                            ).mean()
                        ),

                    "winner_background":
                        int(
                            winner_counts.get(
                                "Background",
                                0,
                            )
                        ),

                    "winner_scs":
                        int(
                            winner_counts.get(
                                "Spinal Canal Stenosis",
                                0,
                            )
                        ),

                    "winner_lfnn":
                        int(
                            winner_counts.get(
                                "Left Neural Foraminal Narrowing",
                                0,
                            )
                        ),

                    "winner_rfnn":
                        int(
                            winner_counts.get(
                                "Right Neural Foraminal Narrowing",
                                0,
                            )
                        ),

                    "winner_lss":
                        int(
                            winner_counts.get(
                                "Left Subarticular Stenosis",
                                0,
                            )
                        ),

                    "winner_rss":
                        int(
                            winner_counts.get(
                                "Right Subarticular Stenosis",
                                0,
                            )
                        ),
                }
            )

    level_df = pd.DataFrame(
        level_rows
    )

    # ========================================================================
    # WINNER SUMMARY
    # ========================================================================

    winner_rows = []

    for criterion in sorted(
        df["criterion"].unique()
    ):

        subset = df[
            df["criterion"]
            == criterion
        ]

        counts = (
            subset[
                "winner_class"
            ]
            .value_counts()
        )

        for class_name in CLASS_NAMES.values():

            winner_rows.append(
                {
                    "criterion":
                        criterion,

                    "winner_class":
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

    # ========================================================================
    # SAVE
    # ========================================================================

    records_path = (
        OUTPUT_DIR
        / "part251_physical_evidence_records.csv"
    )

    global_path = (
        OUTPUT_DIR
        / "part251_physical_evidence_global_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part251_physical_evidence_level_summary.csv"
    )

    winner_path = (
        OUTPUT_DIR
        / "part251_physical_evidence_winner_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part251_physical_evidence_attribution_summary.json"
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

    winner_df.to_csv(
        winner_path,
        index=False,
    )

    summary = {
        "validation_studies":
            25,

        "validation_series":
            25,

        "validation_points":
            188,

        "rfnn_points":
            45,

        "records":
            len(df),

        "criteria":
            sorted(
                df[
                    "criterion"
                ].unique()
                .tolist()
            ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ========================================================================
    # PRINT
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.51 GLOBAL PHYSICAL-SPACE EVIDENCE ATTRIBUTION")
    print("=" * 80)

    print(
        global_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.51 WINNER COUNTS")
    print("=" * 80)

    print(
        winner_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.51 LEVEL ANALYSIS")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.51 COMPLETE")
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
    print(winner_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.51")
    print("PHYSICAL-SPACE RFNN EVIDENCE ATTRIBUTION")
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
        f"Device: {device}"
    )

    model = (
        model
        .to(device)
        .eval()
    )

    all_records = []

    successful_cases = 0

    print()
    print("=" * 80)
    print("RUNNING PHYSICAL-SPACE EVIDENCE ATTRIBUTION")
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

        try:

            records = analyze_case(
                model,
                case,
                device,
            )

            all_records.extend(
                records
            )

            successful_cases += 1

            print(
                f"[{case_number:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id} | "
                f"Records={len(records)}"
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
        f"Total evidence records: "
        f"{len(all_records)}"
    )

    if len(all_records) != 225:

        raise RuntimeError(
            "Expected 225 records "
            "(45 RFNN points × 5 criteria), "
            f"found {len(all_records)}."
        )

    build_summaries(
        all_records
    )


if __name__ == "__main__":
    main()