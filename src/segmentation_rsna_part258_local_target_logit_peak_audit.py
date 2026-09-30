"""
PART 2.58
LOCAL TARGET-LOGIT PEAK AUDIT

Purpose
-------
Determine whether LFNN and RFNN have localized target-class logit peaks
around their annotated points.

For every LFNN/RFNN point:

    1. Sample target logit at annotation.
    2. Search radius 2.
    3. Search radius 4.
    4. Search radius 6.
    5. Record maximum target-logit location.
    6. Record target-vs-background margin at that location.
    7. Record competing class.
    8. Record canonical displacement.
    9. Convert displacement to physical-space distance.
   10. Compare LFNN and RFNN.

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
    / "rsna_part258_local_target_logit_peak_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
    / "part227_best_macro_disease.pth"
)

sys.path.insert(
    0,
    str(SRC_DIR),
)

import segmentation_rsna_part220b_geometry_corrected_training as part220b


# ============================================================================
# CONSTANTS
# ============================================================================

LFNN_NAME = "Left Neural Foraminal Narrowing"

RFNN_NAME = "Right Neural Foraminal Narrowing"

EXPECTED_CASES = 25

EXPECTED_LFNN = 45

EXPECTED_RFNN = 45

EXPECTED_PAIRS = 45

BACKGROUND_ID = 0

SCS_ID = 1

LFNN_ID = 2

RFNN_ID = 3

LSS_ID = 4

RSS_ID = 5

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

RADII = [2, 4, 6]


# ============================================================================
# VALIDATION COHORT
# ============================================================================

def build_validation_cases(manifest):

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
            "select_validation_series() must return DataFrame."
        )

    if len(selected) != EXPECTED_CASES:

        raise RuntimeError(
            f"Expected {EXPECTED_CASES} validation series, "
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

    cases = (
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

    cases = sorted(
        cases,
        key=lambda case:
        case_order.get(
            (
                str(case["study_id"]),
                str(case["series_id"]),
            ),
            9999,
        ),
    )

    if len(cases) != EXPECTED_CASES:

        raise RuntimeError(
            "Validation case count mismatch."
        )

    return cases


# ============================================================================
# MODEL
# ============================================================================

def load_model():

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
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

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

    if missing or unexpected:

        raise RuntimeError(
            "Checkpoint did not load cleanly."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = (
        model
        .to(device)
        .eval()
    )

    print(
        f"Device: {device}"
    )

    return model, device


# ============================================================================
# INFERENCE
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
# POINT -> CANONICAL
# ============================================================================

def point_to_canonical(
    point,
    geometry,
):

    patient = np.array(
        [
            float(point["patient_x"]),
            float(point["patient_y"]),
            float(point["patient_z"]),
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

    if canonical.size < 3:

        raise RuntimeError(
            "Canonical coordinate must contain "
            "z, y, x."
        )

    return canonical[:3]


# ============================================================================
# CANONICAL -> PATIENT
# ============================================================================
#
# We deliberately use the validated Part 2.50 round-trip strategy.
#
# Part 2.20B exposes patient -> canonical but does not expose a direct
# canonical -> patient function. Therefore we numerically invert the
# validated patient -> canonical mapping.
#
# ============================================================================

def canonical_to_patient(
    target_canonical,
    initial_patient,
    geometry,
):

    target = np.asarray(
        target_canonical,
        dtype=np.float64,
    )

    current = np.asarray(
        initial_patient,
        dtype=np.float64,
    ).copy()

    max_iterations = 20

    tolerance = 1e-5

    finite_difference_mm = 0.5

    max_step_mm = 10.0

    for _ in range(
        max_iterations
    ):

        current_canonical = np.asarray(
            part220b.patient_point_to_canonical(
                current,
                geometry,
            ),
            dtype=np.float64,
        ).reshape(-1)[:3]

        residual = (
            target
            - current_canonical
        )

        if np.linalg.norm(
            residual
        ) < tolerance:

            break

        jacobian = np.zeros(
            (
                3,
                3,
            ),
            dtype=np.float64,
        )

        for axis in range(3):

            plus = current.copy()

            minus = current.copy()

            plus[axis] += (
                finite_difference_mm
            )

            minus[axis] -= (
                finite_difference_mm
            )

            plus_canonical = np.asarray(
                part220b.patient_point_to_canonical(
                    plus,
                    geometry,
                ),
                dtype=np.float64,
            ).reshape(-1)[:3]

            minus_canonical = np.asarray(
                part220b.patient_point_to_canonical(
                    minus,
                    geometry,
                ),
                dtype=np.float64,
            ).reshape(-1)[:3]

            jacobian[
                :,
                axis
            ] = (
                plus_canonical
                -
                minus_canonical
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

            break

        delta_norm = np.linalg.norm(
            delta
        )

        if delta_norm > max_step_mm:

            delta = (
                delta
                * (
                    max_step_mm
                    / delta_norm
                )
            )

        current += delta

    final_canonical = np.asarray(
        part220b.patient_point_to_canonical(
            current,
            geometry,
        ),
        dtype=np.float64,
    ).reshape(-1)[:3]

    final_error = float(
        np.linalg.norm(
            target
            - final_canonical
        )
    )

    return (
        current,
        final_error,
    )


# ============================================================================
# SEARCH LOCAL TARGET LOGIT PEAK
# ============================================================================

def search_target_peak(
    logits,
    probabilities,
    center,
    class_id,
    radius,
):

    if logits.ndim == 5:

        logits = logits[0]

    if probabilities.ndim == 5:

        probabilities = probabilities[0]

    logits_np = (
        logits
        .detach()
        .cpu()
        .numpy()
    )

    probabilities_np = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    center_z = int(
        round(
            float(center[0])
        )
    )

    center_y = int(
        round(
            float(center[1])
        )
    )

    center_x = int(
        round(
            float(center[2])
        )
    )

    _, depth, height, width = (
        logits_np.shape
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

    local_target_logits = (
        logits_np[
            class_id,
            z0:z1 + 1,
            y0:y1 + 1,
            x0:x1 + 1,
        ]
    )

    local_index = np.unravel_index(
        np.argmax(
            local_target_logits
        ),
        local_target_logits.shape,
    )

    mz = (
        z0
        + int(
            local_index[0]
        )
    )

    my = (
        y0
        + int(
            local_index[1]
        )
    )

    mx = (
        x0
        + int(
            local_index[2]
        )
    )

    maximum_logits = (
        logits_np[
            :,
            mz,
            my,
            mx,
        ]
    )

    maximum_probabilities = (
        probabilities_np[
            :,
            mz,
            my,
            mx,
        ]
    )

    maximum_target_logit = float(
        maximum_logits[
            class_id
        ]
    )

    maximum_target_probability = float(
        maximum_probabilities[
            class_id
        ]
    )

    maximum_background_logit = float(
        maximum_logits[
            BACKGROUND_ID
        ]
    )

    maximum_background_probability = float(
        maximum_probabilities[
            BACKGROUND_ID
        ]
    )

    predicted_class_id = int(
        np.argmax(
            maximum_logits
        )
    )

    return {
        "maximum_z":
            float(mz),

        "maximum_y":
            float(my),

        "maximum_x":
            float(mx),

        "target_logit":
            maximum_target_logit,

        "target_probability":
            maximum_target_probability,

        "background_logit":
            maximum_background_logit,

        "background_probability":
            maximum_background_probability,

        "target_minus_background":
            (
                maximum_target_logit
                -
                maximum_background_logit
            ),

        "predicted_class_id":
            predicted_class_id,

        "predicted_class":
            CLASS_NAMES[
                predicted_class_id
            ],
    }


# ============================================================================
# ANALYZE ONE POINT
# ============================================================================

def analyze_point(
    point,
    side,
    logits,
    probabilities,
    geometry,
    study_id,
    series_id,
):

    if side == "LFNN":

        class_id = LFNN_ID

    elif side == "RFNN":

        class_id = RFNN_ID

    else:

        raise ValueError(
            side
        )

    annotation_canonical = (
        point_to_canonical(
            point,
            geometry,
        )
    )

    annotation_patient = np.array(
        [
            float(point["patient_x"]),
            float(point["patient_y"]),
            float(point["patient_z"]),
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------------
    # Annotation logits
    # ------------------------------------------------------------------------

    logits_np = (
        logits[0]
        .detach()
        .cpu()
        .numpy()
        if logits.ndim == 5
        else logits.detach().cpu().numpy()
    )

    probabilities_np = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    az = int(
        round(
            float(annotation_canonical[0])
        )
    )

    ay = int(
        round(
            float(annotation_canonical[1])
        )
    )

    ax = int(
        round(
            float(annotation_canonical[2])
        )
    )

    annotation_logits = (
        logits_np[
            :,
            az,
            ay,
            ax,
        ]
    )

    annotation_probabilities = (
        probabilities_np[
            :,
            az,
            ay,
            ax,
        ]
    )

    records = []

    for radius in RADII:

        peak = search_target_peak(
            logits,
            probabilities,
            annotation_canonical,
            class_id,
            radius,
        )

        maximum_canonical = np.array(
            [
                peak["maximum_z"],
                peak["maximum_y"],
                peak["maximum_x"],
            ],
            dtype=np.float64,
        )

        canonical_delta = (
            maximum_canonical
            -
            annotation_canonical
        )

        canonical_distance = float(
            np.linalg.norm(
                canonical_delta
            )
        )

        maximum_patient, inverse_error = (
            canonical_to_patient(
                maximum_canonical,
                annotation_patient,
                geometry,
            )
        )

        physical_delta = (
            maximum_patient
            -
            annotation_patient
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

                "level":
                    str(point["level"]),

                "side":
                    side,

                "radius":
                    radius,

                "class_id":
                    class_id,

                "annotation_z":
                    float(
                        annotation_canonical[0]
                    ),

                "annotation_y":
                    float(
                        annotation_canonical[1]
                    ),

                "annotation_x":
                    float(
                        annotation_canonical[2]
                    ),

                "maximum_z":
                    float(
                        maximum_canonical[0]
                    ),

                "maximum_y":
                    float(
                        maximum_canonical[1]
                    ),

                "maximum_x":
                    float(
                        maximum_canonical[2]
                    ),

                "canonical_dz":
                    float(
                        canonical_delta[0]
                    ),

                "canonical_dy":
                    float(
                        canonical_delta[1]
                    ),

                "canonical_dx":
                    float(
                        canonical_delta[2]
                    ),

                "canonical_distance_voxels":
                    canonical_distance,

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

                "inverse_roundtrip_error":
                    inverse_error,

                "annotation_target_logit":
                    float(
                        annotation_logits[
                            class_id
                        ]
                    ),

                "annotation_background_logit":
                    float(
                        annotation_logits[
                            BACKGROUND_ID
                        ]
                    ),

                "annotation_target_probability":
                    float(
                        annotation_probabilities[
                            class_id
                        ]
                    ),

                "annotation_target_minus_background":
                    float(
                        annotation_logits[
                            class_id
                        ]
                        -
                        annotation_logits[
                            BACKGROUND_ID
                        ]
                    ),

                "maximum_target_logit":
                    peak[
                        "target_logit"
                    ],

                "maximum_target_probability":
                    peak[
                        "target_probability"
                    ],

                "maximum_background_logit":
                    peak[
                        "background_logit"
                    ],

                "maximum_background_probability":
                    peak[
                        "background_probability"
                    ],

                "maximum_target_minus_background":
                    peak[
                        "target_minus_background"
                    ],

                "maximum_predicted_class_id":
                    peak[
                        "predicted_class_id"
                    ],

                "maximum_predicted_class":
                    peak[
                        "predicted_class"
                    ],
            }
        )

    return records


# ============================================================================
# RUN ANALYSIS
# ============================================================================

def run_analysis(
    model,
    device,
    cases,
):

    records = []

    successful_cases = 0

    total_lfnn = 0

    total_rfnn = 0

    print()
    print("=" * 80)
    print("RUNNING LOCAL TARGET-LOGIT PEAK ANALYSIS")
    print("=" * 80)

    for index, case in enumerate(
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

            point_df = pd.DataFrame(
                points
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

            for _, point in (
                lfnn_points.iterrows()
            ):

                point_records = analyze_point(
                    point=point,
                    side="LFNN",
                    logits=logits,
                    probabilities=probabilities,
                    geometry=geometry,
                    study_id=study_id,
                    series_id=series_id,
                )

                records.extend(
                    point_records
                )

                case_lfnn += 1

            for _, point in (
                rfnn_points.iterrows()
            ):

                point_records = analyze_point(
                    point=point,
                    side="RFNN",
                    logits=logits,
                    probabilities=probabilities,
                    geometry=geometry,
                    study_id=study_id,
                    series_id=series_id,
                )

                records.extend(
                    point_records
                )

                case_rfnn += 1

            total_lfnn += case_lfnn

            total_rfnn += case_rfnn

            successful_cases += 1

            print(
                f"[{index:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id} | "
                f"LFNN={case_lfnn} | "
                f"RFNN={case_rfnn}"
            )

        except Exception as exc:

            print(
                f"[{index:02d}/25] "
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

    if successful_cases != EXPECTED_CASES:

        raise RuntimeError(
            "Not all validation cases succeeded."
        )

    if total_lfnn != EXPECTED_LFNN:

        raise RuntimeError(
            "Expected 45 LFNN points."
        )

    if total_rfnn != EXPECTED_RFNN:

        raise RuntimeError(
            "Expected 45 RFNN points."
        )

    expected_records = (
        (EXPECTED_LFNN + EXPECTED_RFNN)
        * len(RADII)
    )

    if len(records) != expected_records:

        raise RuntimeError(
            f"Expected {expected_records} records, "
            f"found {len(records)}."
        )

    return pd.DataFrame(
        records
    )


# ============================================================================
# GLOBAL SUMMARY
# ============================================================================

def build_global_summary(
    records_df,
):

    rows = []

    for radius in RADII:

        subset = records_df[
            records_df["radius"]
            == radius
        ]

        for side in [
            "LFNN",
            "RFNN",
        ]:

            side_df = subset[
                subset["side"]
                == side
            ]

            rows.append(
                {
                    "radius":
                        radius,

                    "side":
                        side,

                    "count":
                        len(side_df),

                    "annotation_target_logit":
                        float(
                            side_df[
                                "annotation_target_logit"
                            ].mean()
                        ),

                    "maximum_target_logit":
                        float(
                            side_df[
                                "maximum_target_logit"
                            ].mean()
                        ),

                    "target_logit_gain":
                        float(
                            (
                                side_df[
                                    "maximum_target_logit"
                                ]
                                -
                                side_df[
                                    "annotation_target_logit"
                                ]
                            ).mean()
                        ),

                    "maximum_target_probability":
                        float(
                            side_df[
                                "maximum_target_probability"
                            ].mean()
                        ),

                    "maximum_target_minus_background":
                        float(
                            side_df[
                                "maximum_target_minus_background"
                            ].mean()
                        ),

                    "maximum_predicted_class":
                        (
                            side_df[
                                "maximum_predicted_class"
                            ]
                            .value_counts()
                            .idxmax()
                        ),

                    "target_wins_at_peak_fraction":
                        float(
                            (
                                side_df[
                                    "maximum_predicted_class_id"
                                ]
                                ==
                                side_df[
                                    "class_id"
                                ]
                            ).mean()
                        ),

                    "canonical_distance_voxels":
                        float(
                            side_df[
                                "canonical_distance_voxels"
                            ].mean()
                        ),

                    "physical_distance_mm":
                        float(
                            side_df[
                                "physical_distance_mm"
                            ].mean()
                        ),

                    "within_2mm_fraction":
                        float(
                            (
                                side_df[
                                    "physical_distance_mm"
                                ]
                                <= 2.0
                            ).mean()
                        ),

                    "within_4mm_fraction":
                        float(
                            (
                                side_df[
                                    "physical_distance_mm"
                                ]
                                <= 4.0
                            ).mean()
                        ),

                    "within_6mm_fraction":
                        float(
                            (
                                side_df[
                                    "physical_distance_mm"
                                ]
                                <= 6.0
                            ).mean()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# PAIRED SUMMARY
# ============================================================================

def build_paired_summary(
    records_df,
):

    lfnn = records_df[
        records_df["side"]
        == "LFNN"
    ].copy()

    rfnn = records_df[
        records_df["side"]
        == "RFNN"
    ].copy()

    pairs = lfnn.merge(
        rfnn,
        on=[
            "study_id",
            "series_id",
            "level",
            "radius",
        ],
        suffixes=(
            "_lfnn",
            "_rfnn",
        ),
    )

    if len(pairs) != (
        EXPECTED_PAIRS
        * len(RADII)
    ):

        raise RuntimeError(
            "Paired record count mismatch."
        )

    rows = []

    metrics = [
        (
            "annotation_target_logit",
            "annotation_target_logit_lfnn",
            "annotation_target_logit_rfnn",
        ),
        (
            "maximum_target_logit",
            "maximum_target_logit_lfnn",
            "maximum_target_logit_rfnn",
        ),
        (
            "target_logit_gain",
            (
                "maximum_target_logit_lfnn"
            ),
            (
                "maximum_target_logit_rfnn"
            ),
        ),
        (
            "maximum_target_minus_background",
            "maximum_target_minus_background_lfnn",
            "maximum_target_minus_background_rfnn",
        ),
        (
            "maximum_target_probability",
            "maximum_target_probability_lfnn",
            "maximum_target_probability_rfnn",
        ),
        (
            "physical_distance_mm",
            "physical_distance_mm_lfnn",
            "physical_distance_mm_rfnn",
        ),
        (
            "canonical_distance_voxels",
            "canonical_distance_voxels_lfnn",
            "canonical_distance_voxels_rfnn",
        ),
    ]

    for radius in RADII:

        radius_pairs = pairs[
            pairs["radius"]
            == radius
        ]

        for name, lcol, rcol in metrics:

            if name == "target_logit_gain":

                lvalues = (
                    radius_pairs[
                        "maximum_target_logit_lfnn"
                    ]
                    -
                    radius_pairs[
                        "annotation_target_logit_lfnn"
                    ]
                )

                rvalues = (
                    radius_pairs[
                        "maximum_target_logit_rfnn"
                    ]
                    -
                    radius_pairs[
                        "annotation_target_logit_rfnn"
                    ]
                )

            else:

                lvalues = radius_pairs[
                    lcol
                ]

                rvalues = radius_pairs[
                    rcol
                ]

            difference = (
                rvalues
                -
                lvalues
            )

            rows.append(
                {
                    "radius":
                        radius,

                    "metric":
                        name,

                    "lfnn_mean":
                        float(
                            lvalues.mean()
                        ),

                    "rfnn_mean":
                        float(
                            rvalues.mean()
                        ),

                    "rfnn_minus_lfnn":
                        float(
                            difference.mean()
                        ),

                    "lfnn_median":
                        float(
                            lvalues.median()
                        ),

                    "rfnn_median":
                        float(
                            rvalues.median()
                        ),

                    "rfnn_higher_fraction":
                        float(
                            (
                                difference
                                > 0
                            ).mean()
                        ),

                    "lfnn_higher_fraction":
                        float(
                            (
                                difference
                                < 0
                            ).mean()
                        ),
                }
            )

    return (
        pairs,
        pd.DataFrame(rows),
    )


# ============================================================================
# LEVEL SUMMARY
# ============================================================================

def build_level_summary(
    records_df,
):

    rows = []

    for level in sorted(
        records_df[
            "level"
        ].unique()
    ):

        for radius in RADII:

            subset = records_df[
                (
                    records_df[
                        "level"
                    ]
                    == level
                )
                &
                (
                    records_df[
                        "radius"
                    ]
                    == radius
                )
            ]

            lfnn = subset[
                subset["side"]
                == "LFNN"
            ]

            rfnn = subset[
                subset["side"]
                == "RFNN"
            ]

            rows.append(
                {
                    "level":
                        level,

                    "radius":
                        radius,

                    "lfnn_target_logit":
                        float(
                            lfnn[
                                "maximum_target_logit"
                            ].mean()
                        ),

                    "rfnn_target_logit":
                        float(
                            rfnn[
                                "maximum_target_logit"
                            ].mean()
                        ),

                    "rfnn_minus_lfnn_target_logit":
                        float(
                            (
                                rfnn[
                                    "maximum_target_logit"
                                ].mean()
                                -
                                lfnn[
                                    "maximum_target_logit"
                                ].mean()
                            )
                        ),

                    "lfnn_target_background_margin":
                        float(
                            lfnn[
                                "maximum_target_minus_background"
                            ].mean()
                        ),

                    "rfnn_target_background_margin":
                        float(
                            rfnn[
                                "maximum_target_minus_background"
                            ].mean()
                        ),

                    "rfnn_minus_lfnn_margin":
                        float(
                            (
                                rfnn[
                                    "maximum_target_minus_background"
                                ].mean()
                                -
                                lfnn[
                                    "maximum_target_minus_background"
                                ].mean()
                            )
                        ),

                    "lfnn_peak_distance_mm":
                        float(
                            lfnn[
                                "physical_distance_mm"
                            ].mean()
                        ),

                    "rfnn_peak_distance_mm":
                        float(
                            rfnn[
                                "physical_distance_mm"
                            ].mean()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# SAVE
# ============================================================================

def save_outputs(
    records_df,
    global_df,
    paired_df,
    level_df,
    pairs,
):

    records_path = (
        OUTPUT_DIR
        / "part258_local_target_logit_peak_records.csv"
    )

    global_path = (
        OUTPUT_DIR
        / "part258_local_target_logit_peak_summary.csv"
    )

    paired_path = (
        OUTPUT_DIR
        / "part258_paired_peak_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part258_level_peak_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part258_local_target_logit_peak_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    global_df.to_csv(
        global_path,
        index=False,
    )

    paired_df.to_csv(
        paired_path,
        index=False,
    )

    level_df.to_csv(
        level_path,
        index=False,
    )

    summary = {
        "cases":
            EXPECTED_CASES,

        "lfnn_points":
            EXPECTED_LFNN,

        "rfnn_points":
            EXPECTED_RFNN,

        "paired_observations":
            EXPECTED_PAIRS,

        "radii":
            RADII,

        "global_summary":
            global_df.to_dict(
                orient="records"
            ),

        "paired_summary":
            paired_df.to_dict(
                orient="records"
            ),

        "level_summary":
            level_df.to_dict(
                orient="records"
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

    print()
    print("=" * 80)
    print("PART 2.58 GLOBAL LOCAL TARGET-LOGIT PEAK SUMMARY")
    print("=" * 80)

    print(
        global_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.58 PAIRED LFNN / RFNN PEAK SUMMARY")
    print("=" * 80)

    print(
        paired_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.58 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.58 COMPLETE")
    print("=" * 80)

    print(
        "Analysis only."
    )

    print(
        "Part 2.27 checkpoint unchanged."
    )

    print(
        "Part 2.57 unchanged."
    )

    print(
        "Dashboard unchanged."
    )

    print()
    print("Outputs:")

    print(records_path)
    print(global_path)
    print(paired_path)
    print(level_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.58")
    print("LOCAL TARGET-LOGIT PEAK AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
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

    cases = build_validation_cases(
        manifest
    )

    print(
        f"Validation cases: "
        f"{len(cases)}"
    )

    model, device = load_model()

    records_df = run_analysis(
        model,
        device,
        cases,
    )

    global_df = (
        build_global_summary(
            records_df
        )
    )

    pairs, paired_df = (
        build_paired_summary(
            records_df
        )
    )

    level_df = (
        build_level_summary(
            records_df
        )
    )

    save_outputs(
        records_df,
        global_df,
        paired_df,
        level_df,
        pairs,
    )


if __name__ == "__main__":
    main()