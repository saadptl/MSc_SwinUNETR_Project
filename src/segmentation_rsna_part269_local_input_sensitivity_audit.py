"""
PART 2.69
LOCAL INPUT-TO-OUTPUT SENSITIVITY AUDIT

Purpose
-------
Measure how sensitive the trained Swin-UNETR output is to local MRI
intensity information around paired LFNN and RFNN annotation points.

For every paired LFNN/RFNN point we calculate the baseline output and then
perturb a local 3D neighborhood by several controlled intensity changes.

Perturbations:
    0.00 = baseline
    0.05
    0.10
    0.20

The perturbation is applied as:

    local_patch <- local_patch * (1 - strength)

This is a local attenuation experiment, NOT image synthesis.

For each perturbation we measure:

    RFNN logit
    LFNN logit
    Background logit
    RFNN-background margin
    LFNN-background margin
    RFNN probability
    Background probability

The experiment compares:

    LFNN point
    RFNN point

using the exact Part 2.20B validation cohort.

Analysis only.

NO:
    training
    optimizer
    checkpoint modification
    dashboard modification

Expected:
    9 paired validation series
    45 paired observations
    5 levels per series
    2 point types
    4 non-zero perturbation strengths

Expected non-baseline records:
    45 * 5 * 2 * 4 = 1800
"""


from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part269_local_input_sensitivity_audit"
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

BACKGROUND_ID = 0
LFNN_ID = 2
RFNN_ID = 3

EXPECTED_CASES = 25
EXPECTED_PAIRED_SERIES = 9
EXPECTED_PAIRS = 45

# Local perturbation strengths.
# 0.05 = attenuate local signal by 5%
# 0.10 = attenuate local signal by 10%
# 0.20 = attenuate local signal by 20%
PERTURBATION_STRENGTHS = [
    0.05,
    0.10,
    0.20,
]

# Local radius in model voxels.
#
# radius=2 means a 5x5x5 cube.
# radius=3 means a 7x7x7 cube.
#
# We use radius=2 as the primary controlled local experiment.
PATCH_RADIUS = 2

# Multiple local scales provide a useful robustness check.
PATCH_RADII = [
    2,
    3,
]


# ============================================================================
# VALIDATION COHORT
# ============================================================================

def build_validation_cases(manifest):

    selected = part220b.select_validation_series(
        manifest
    )

    if not isinstance(
        selected,
        pd.DataFrame,
    ):
        raise TypeError(
            "select_validation_series() must return a DataFrame."
        )

    if len(selected) != EXPECTED_CASES:

        raise RuntimeError(
            f"Expected {EXPECTED_CASES} validation cases, "
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

    validation_cases = part220b.build_case_index(
        validation_manifest
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
            9999,
        ),
    )

    return validation_cases


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
        isinstance(checkpoint, dict)
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

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
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
# IMAGE PREPARATION
# ============================================================================

def prepare_image(
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

        image = image.unsqueeze(0)

    image = (
        image
        .float()
        .to(device)
    )

    return image


# ============================================================================
# CANONICAL POINT
# ============================================================================

def get_canonical_point(
    point,
    geometry,
):

    patient_point = np.array(
        [
            float(point["patient_x"]),
            float(point["patient_y"]),
            float(point["patient_z"]),
        ],
        dtype=np.float64,
    )

    canonical = (
        part220b.patient_point_to_canonical(
            patient_point,
            geometry,
        )
    )

    canonical = np.asarray(
        canonical,
        dtype=np.float64,
    ).reshape(-1)

    return canonical[:3]


# ============================================================================
# MODEL-SPACE POINT
# ============================================================================

def canonical_to_index(
    canonical_point,
    shape,
):

    depth = int(shape[-3])
    height = int(shape[-2])
    width = int(shape[-1])

    z = int(
        np.clip(
            round(
                float(canonical_point[0])
            ),
            0,
            depth - 1,
        )
    )

    y = int(
        np.clip(
            round(
                float(canonical_point[1])
            ),
            0,
            height - 1,
        )
    )

    x = int(
        np.clip(
            round(
                float(canonical_point[2])
            ),
            0,
            width - 1,
        )
    )

    return z, y, x


# ============================================================================
# OUTPUT SAMPLING
# ============================================================================

def sample_output_at_point(
    output,
    canonical_point,
):

    index = canonical_to_index(
        canonical_point,
        output.shape,
    )

    z, y, x = index

    logits = (
        output[
            0,
            :,
            z,
            y,
            x,
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    return logits


# ============================================================================
# OUTPUT METRICS
# ============================================================================

def calculate_metrics(
    logits,
):

    tensor = torch.tensor(
        logits,
        dtype=torch.float32,
    )

    probabilities = (
        torch.softmax(
            tensor,
            dim=0,
        )
        .numpy()
    )

    rfnn_logit = float(
        logits[RFNN_ID]
    )

    lfnn_logit = float(
        logits[LFNN_ID]
    )

    background_logit = float(
        logits[BACKGROUND_ID]
    )

    return {
        "rfnn_logit":
            rfnn_logit,

        "lfnn_logit":
            lfnn_logit,

        "background_logit":
            background_logit,

        "rfnn_probability":
            float(
                probabilities[RFNN_ID]
            ),

        "lfnn_probability":
            float(
                probabilities[LFNN_ID]
            ),

        "background_probability":
            float(
                probabilities[
                    BACKGROUND_ID
                ]
            ),

        "rfnn_margin":
            rfnn_logit
            -
            background_logit,

        "lfnn_margin":
            lfnn_logit
            -
            background_logit,

        "rfnn_lfnn_logit_difference":
            rfnn_logit
            -
            lfnn_logit,
    }


# ============================================================================
# LOCAL PATCH BOUNDS
# ============================================================================

def get_patch_bounds(
    center,
    shape,
    radius,
):

    z, y, x = center

    depth = int(shape[-3])
    height = int(shape[-2])
    width = int(shape[-1])

    z0 = max(
        0,
        z - radius,
    )

    z1 = min(
        depth,
        z + radius + 1,
    )

    y0 = max(
        0,
        y - radius,
    )

    y1 = min(
        height,
        y + radius + 1,
    )

    x0 = max(
        0,
        x - radius,
    )

    x1 = min(
        width,
        x + radius + 1,
    )

    return (
        z0,
        z1,
        y0,
        y1,
        x0,
        x1,
    )


# ============================================================================
# LOCAL ATTENUATION
# ============================================================================

def attenuate_local_patch(
    image_tensor,
    center,
    radius,
    strength,
):

    perturbed = (
        image_tensor
        .clone()
    )

    bounds = get_patch_bounds(
        center,
        perturbed.shape,
        radius,
    )

    (
        z0,
        z1,
        y0,
        y1,
        x0,
        x1,
    ) = bounds

    # Controlled multiplicative attenuation.
    #
    # strength=0.10:
    #       original * 0.90
    #
    # strength=0.20:
    #       original * 0.80
    #
    factor = (
        1.0
        -
        float(strength)
    )

    perturbed[
        :,
        :,
        z0:z1,
        y0:y1,
        x0:x1,
    ] = (
        perturbed[
            :,
            :,
            z0:z1,
            y0:y1,
            x0:x1,
        ]
        * factor
    )

    voxel_count = (
        (z1 - z0)
        *
        (y1 - y0)
        *
        (x1 - x0)
    )

    return (
        perturbed,
        voxel_count,
    )


# ============================================================================
# LOCAL PERTURBATION METRICS
# ============================================================================

def evaluate_local_perturbation(
    model,
    image_tensor,
    point,
    radius,
    strength,
):

    center = canonical_to_index(
        point,
        image_tensor.shape,
    )

    perturbed, voxel_count = (
        attenuate_local_patch(
            image_tensor,
            center,
            radius,
            strength,
        )
    )

    with torch.no_grad():

        output = model(
            perturbed
        )

    logits = (
        sample_output_at_point(
            output,
            point,
        )
    )

    metrics = calculate_metrics(
        logits
    )

    metrics[
        "perturbed_voxels"
    ] = int(
        voxel_count
    )

    return metrics


# ============================================================================
# CASE ANALYSIS
# ============================================================================

def analyze_case(
    model,
    device,
    case,
):

    study_id = str(
        case["study_id"]
    )

    series_id = str(
        case["series_id"]
    )

    image, points, geometry = (
        part220b.load_case(
            study_id,
            series_id,
            case["points"],
        )
    )

    point_df = pd.DataFrame(
        points
    )

    lfnn_points = point_df[
        point_df["class_name"]
        == LFNN_NAME
    ].copy()

    rfnn_points = point_df[
        point_df["class_name"]
        == RFNN_NAME
    ].copy()

    if len(lfnn_points) != 5:
        return []

    if len(rfnn_points) != 5:
        return []

    common_levels = sorted(
        set(
            lfnn_points["level"]
        )
        &
        set(
            rfnn_points["level"]
        )
    )

    if len(common_levels) != 5:
        return []

    image_tensor = prepare_image(
        image,
        device,
    )

    # ========================================================================
    # BASELINE OUTPUT
    # ========================================================================

    with torch.no_grad():

        baseline_output = model(
            image_tensor
        )

    records = []

    # ========================================================================
    # LEVELS
    # ========================================================================

    for level in common_levels:

        lrow = lfnn_points[
            lfnn_points["level"]
            == level
        ]

        rrow = rfnn_points[
            rfnn_points["level"]
            == level
        ]

        if len(lrow) != 1:
            continue

        if len(rrow) != 1:
            continue

        lpoint = lrow.iloc[0]
        rpoint = rrow.iloc[0]

        points_by_type = {
            "lfnn":
                get_canonical_point(
                    lpoint,
                    geometry,
                ),

            "rfnn":
                get_canonical_point(
                    rpoint,
                    geometry,
                ),
        }

        # ------------------------------------------------------------
        # Baseline metrics
        # ------------------------------------------------------------

        baseline_metrics = {}

        for point_type, canonical_point in (
            points_by_type.items()
        ):

            logits = (
                sample_output_at_point(
                    baseline_output,
                    canonical_point,
                )
            )

            baseline_metrics[
                point_type
            ] = calculate_metrics(
                logits
            )

        # ------------------------------------------------------------
        # Perturbation radii
        # ------------------------------------------------------------

        for radius in PATCH_RADII:

            for point_type, canonical_point in (
                points_by_type.items()
            ):

                center = canonical_to_index(
                    canonical_point,
                    image_tensor.shape,
                )

                # ----------------------------------------------------
                # Perturbation strengths
                # ----------------------------------------------------

                for strength in (
                    PERTURBATION_STRENGTHS
                ):

                    metrics = (
                        evaluate_local_perturbation(
                            model,
                            image_tensor,
                            canonical_point,
                            radius,
                            strength,
                        )
                    )

                    base = baseline_metrics[
                        point_type
                    ]

                    records.append(
                        {
                            "study_id":
                                study_id,

                            "series_id":
                                series_id,

                            "level":
                                str(level),

                            "point_type":
                                point_type,

                            "radius":
                                int(radius),

                            "patch_side":
                                int(
                                    2 * radius + 1
                                ),

                            "perturbation_strength":
                                float(strength),

                            "attenuation_percent":
                                float(
                                    strength
                                    * 100.0
                                ),

                            "center_z":
                                int(center[0]),

                            "center_y":
                                int(center[1]),

                            "center_x":
                                int(center[2]),

                            "perturbed_voxels":
                                int(
                                    metrics[
                                        "perturbed_voxels"
                                    ]
                                ),

                            "baseline_rfnn_logit":
                                base[
                                    "rfnn_logit"
                                ],

                            "perturbed_rfnn_logit":
                                metrics[
                                    "rfnn_logit"
                                ],

                            "delta_rfnn_logit":
                                metrics[
                                    "rfnn_logit"
                                ]
                                -
                                base[
                                    "rfnn_logit"
                                ],

                            "baseline_lfnn_logit":
                                base[
                                    "lfnn_logit"
                                ],

                            "perturbed_lfnn_logit":
                                metrics[
                                    "lfnn_logit"
                                ],

                            "delta_lfnn_logit":
                                metrics[
                                    "lfnn_logit"
                                ]
                                -
                                base[
                                    "lfnn_logit"
                                ],

                            "baseline_background_logit":
                                base[
                                    "background_logit"
                                ],

                            "perturbed_background_logit":
                                metrics[
                                    "background_logit"
                                ],

                            "delta_background_logit":
                                metrics[
                                    "background_logit"
                                ]
                                -
                                base[
                                    "background_logit"
                                ],

                            "baseline_rfnn_margin":
                                base[
                                    "rfnn_margin"
                                ],

                            "perturbed_rfnn_margin":
                                metrics[
                                    "rfnn_margin"
                                ],

                            "delta_rfnn_margin":
                                metrics[
                                    "rfnn_margin"
                                ]
                                -
                                base[
                                    "rfnn_margin"
                                ],

                            "baseline_lfnn_margin":
                                base[
                                    "lfnn_margin"
                                ],

                            "perturbed_lfnn_margin":
                                metrics[
                                    "lfnn_margin"
                                ],

                            "delta_lfnn_margin":
                                metrics[
                                    "lfnn_margin"
                                ]
                                -
                                base[
                                    "lfnn_margin"
                                ],

                            "baseline_rfnn_probability":
                                base[
                                    "rfnn_probability"
                                ],

                            "perturbed_rfnn_probability":
                                metrics[
                                    "rfnn_probability"
                                ],

                            "delta_rfnn_probability":
                                metrics[
                                    "rfnn_probability"
                                ]
                                -
                                base[
                                    "rfnn_probability"
                                ],

                            "baseline_background_probability":
                                base[
                                    "background_probability"
                                ],

                            "perturbed_background_probability":
                                metrics[
                                    "background_probability"
                                ],

                            "delta_background_probability":
                                metrics[
                                    "background_probability"
                                ]
                                -
                                base[
                                    "background_probability"
                                ],

                            "baseline_rfnn_lfnn_logit_difference":
                                base[
                                    "rfnn_lfnn_logit_difference"
                                ],

                            "perturbed_rfnn_lfnn_logit_difference":
                                metrics[
                                    "rfnn_lfnn_logit_difference"
                                ],

                            "delta_rfnn_lfnn_logit_difference":
                                metrics[
                                    "rfnn_lfnn_logit_difference"
                                ]
                                -
                                base[
                                    "rfnn_lfnn_logit_difference"
                                ],
                        }
                    )

    return records


# ============================================================================
# SUMMARY BY POINT TYPE AND RADIUS
# ============================================================================

def build_summary(
    records_df,
):

    rows = []

    grouped = records_df.groupby(
        [
            "point_type",
            "radius",
            "perturbation_strength",
        ],
        sort=False,
    )

    for (
        point_type,
        radius,
        strength,
    ), group in grouped:

        rows.append(
            {
                "point_type":
                    point_type,

                "radius":
                    int(radius),

                "patch_side":
                    int(
                        2 * radius + 1
                    ),

                "perturbation_strength":
                    float(strength),

                "attenuation_percent":
                    float(
                        strength * 100.0
                    ),

                "records":
                    int(len(group)),

                "mean_delta_rfnn_logit":
                    float(
                        group[
                            "delta_rfnn_logit"
                        ].mean()
                    ),

                "mean_delta_lfnn_logit":
                    float(
                        group[
                            "delta_lfnn_logit"
                        ].mean()
                    ),

                "mean_delta_background_logit":
                    float(
                        group[
                            "delta_background_logit"
                        ].mean()
                    ),

                "mean_delta_rfnn_margin":
                    float(
                        group[
                            "delta_rfnn_margin"
                        ].mean()
                    ),

                "mean_delta_lfnn_margin":
                    float(
                        group[
                            "delta_lfnn_margin"
                        ].mean()
                    ),

                "mean_delta_rfnn_probability":
                    float(
                        group[
                            "delta_rfnn_probability"
                        ].mean()
                    ),

                "mean_delta_background_probability":
                    float(
                        group[
                            "delta_background_probability"
                        ].mean()
                    ),

                "mean_delta_rfnn_lfnn_logit_difference":
                    float(
                        group[
                            "delta_rfnn_lfnn_logit_difference"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# PAIRED LFNN VS RFNN DIFFERENCE
# ============================================================================

def build_paired_difference_summary(
    records_df,
):

    pivot = records_df.pivot_table(
        index=[
            "study_id",
            "series_id",
            "level",
            "radius",
            "perturbation_strength",
        ],
        columns="point_type",
        values=[
            "delta_rfnn_logit",
            "delta_lfnn_logit",
            "delta_background_logit",
            "delta_rfnn_margin",
            "delta_lfnn_margin",
            "delta_rfnn_probability",
        ],
        aggfunc="mean",
    )

    pivot = pivot.reset_index()

    # Flatten MultiIndex columns.
    flattened = []

    for column in pivot.columns:

        if isinstance(
            column,
            tuple,
        ):

            if column[0] in (
                "study_id",
                "series_id",
                "level",
                "radius",
                "perturbation_strength",
            ):

                flattened.append(
                    column[0]
                )

            else:

                flattened.append(
                    "_".join(
                        str(x)
                        for x in column
                        if str(x) != ""
                    )
                )

        else:

            flattened.append(
                column
            )

    pivot.columns = flattened

    # RFNN-point minus LFNN-point.
    metric_names = [
        "delta_rfnn_logit",
        "delta_lfnn_logit",
        "delta_background_logit",
        "delta_rfnn_margin",
        "delta_lfnn_margin",
        "delta_rfnn_probability",
    ]

    for metric in metric_names:

        rcol = (
            f"{metric}_rfnn"
        )

        lcol = (
            f"{metric}_lfnn"
        )

        if (
            rcol in pivot.columns
            and
            lcol in pivot.columns
        ):

            pivot[
                f"rfnn_minus_lfnn_{metric}"
            ] = (
                pivot[rcol]
                -
                pivot[lcol]
            )

    return pivot


# ============================================================================
# LEVEL SUMMARY
# ============================================================================

def build_level_summary(
    records_df,
):

    grouped = records_df.groupby(
        [
            "level",
            "point_type",
            "radius",
            "perturbation_strength",
        ],
        sort=False,
    )

    rows = []

    for (
        level,
        point_type,
        radius,
        strength,
    ), group in grouped:

        rows.append(
            {
                "level":
                    level,

                "point_type":
                    point_type,

                "radius":
                    int(radius),

                "perturbation_strength":
                    float(strength),

                "records":
                    int(len(group)),

                "mean_delta_rfnn_logit":
                    float(
                        group[
                            "delta_rfnn_logit"
                        ].mean()
                    ),

                "mean_delta_background_logit":
                    float(
                        group[
                            "delta_background_logit"
                        ].mean()
                    ),

                "mean_delta_rfnn_margin":
                    float(
                        group[
                            "delta_rfnn_margin"
                        ].mean()
                    ),

                "mean_delta_lfnn_margin":
                    float(
                        group[
                            "delta_lfnn_margin"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(
    records_df,
    summary_df,
    paired_summary,
    level_summary,
):

    records_path = (
        OUTPUT_DIR
        / "part269_local_input_sensitivity_records.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part269_local_input_sensitivity_summary.csv"
    )

    paired_path = (
        OUTPUT_DIR
        / "part269_paired_lfnn_rfnn_sensitivity.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part269_local_input_sensitivity_level_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part269_local_input_sensitivity_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    paired_summary.to_csv(
        paired_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    summary = {
        "analysis":
            "Part 2.69 local input-to-output sensitivity audit",

        "checkpoint":
            str(
                CHECKPOINT_PATH
            ),

        "validation_cases":
            EXPECTED_CASES,

        "paired_validation_series":
            EXPECTED_PAIRED_SERIES,

        "paired_observations":
            EXPECTED_PAIRS,

        "patch_radii":
            PATCH_RADII,

        "perturbation_strengths":
            PERTURBATION_STRENGTHS,

        "record_count":
            int(
                len(records_df)
            ),

        "summary":
            summary_df.to_dict(
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
    print("PART 2.69 LOCAL INPUT SENSITIVITY SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.69 PAIRED RFNN-LFNN DIFFERENCE")
    print("=" * 80)

    print(
        paired_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.69 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.69 COMPLETE")
    print("=" * 80)

    print(
        "Analysis only."
    )

    print(
        "No training."
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
    print(paired_path)
    print(level_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.69")
    print("LOCAL INPUT-TO-OUTPUT SENSITIVITY AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    print()
    print(
        f"Patch radii: "
        f"{PATCH_RADII}"
    )

    print(
        f"Perturbation strengths: "
        f"{PERTURBATION_STRENGTHS}"
    )

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

    model, device = (
        load_model()
    )

    all_records = []

    paired_series = 0

    print()
    print("=" * 80)
    print("RUNNING LOCAL INPUT SENSITIVITY AUDIT")
    print("=" * 80)

    for index, case in enumerate(
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

            case_records = (
                analyze_case(
                    model,
                    device,
                    case,
                )
            )

            if case_records:

                all_records.extend(
                    case_records
                )

                paired_series += 1

                print(
                    f"[{index:02d}/25] "
                    f"Study={study_id} | "
                    f"Series={series_id} | "
                    f"paired levels=5 | "
                    f"records={len(case_records)}"
                )

            else:

                print(
                    f"[{index:02d}/25] "
                    f"Study={study_id} | "
                    f"Series={series_id} | "
                    f"no paired records"
                )

        except Exception as exc:

            print(
                f"[{index:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id} | "
                f"FAILED: {exc}"
            )

    records_df = pd.DataFrame(
        all_records
    )

    print()
    print(
        f"Paired validation series: "
        f"{paired_series}/25"
    )

    print(
        f"Local sensitivity records: "
        f"{len(records_df)}"
    )

    if paired_series != (
        EXPECTED_PAIRED_SERIES
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_PAIRED_SERIES} "
            f"paired series, found {paired_series}."
        )

    expected_records = (
        EXPECTED_PAIRS
        *
        len(PATCH_RADII)
        *
        2
        *
        len(PERTURBATION_STRENGTHS)
    )

    if len(records_df) != (
        expected_records
    ):

        raise RuntimeError(
            f"Expected {expected_records} records, "
            f"found {len(records_df)}."
        )

    unique_pairs = (
        records_df[
            [
                "study_id",
                "series_id",
                "level",
            ]
        ]
        .drop_duplicates()
    )

    if len(unique_pairs) != (
        EXPECTED_PAIRS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_PAIRS} paired observations, "
            f"found {len(unique_pairs)}."
        )

    print()
    print(
        "Local sensitivity validation: PASSED"
    )

    print(
        f"Paired observations: "
        f"{len(unique_pairs)}"
    )

    print(
        f"Patch radii: "
        f"{len(PATCH_RADII)}"
    )

    print(
        f"Point types: 2"
    )

    print(
        f"Perturbation strengths: "
        f"{len(PERTURBATION_STRENGTHS)}"
    )

    print(
        f"Expected records: "
        f"{expected_records}"
    )

    summary_df = build_summary(
        records_df
    )

    paired_summary = (
        build_paired_difference_summary(
            records_df
        )
    )

    level_summary = (
        build_level_summary(
            records_df
        )
    )

    save_outputs(
        records_df,
        summary_df,
        paired_summary,
        level_summary,
    )


if __name__ == "__main__":
    main()