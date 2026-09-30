"""
PART 2.70
LOCAL PATCH COUNTERFACTUAL AUDIT

Purpose
-------
Measure the effect of replacing local MRI information around LFNN/RFNN
annotation points with a locally estimated surrounding-shell intensity.

This is a counterfactual input experiment.

It does NOT:
    - train the model
    - modify the checkpoint
    - modify the dashboard
    - create segmentation ground truth
    - claim clinical correctness

Exact Part 2.20B geometry and validation cohort are used.

Checkpoint:
    Part 2.27 best macro disease checkpoint.

Cohort:
    25 validation cases
    9 paired validation series
    45 paired LFNN/RFNN observations
    5 levels per paired series

Perturbations:
    0.25
    0.50
    1.00

Replacement:
    local surrounding-shell median intensity.

Interpretation:
    If replacing the local content substantially changes the RFNN output,
    the local MRI content contributes to the prediction.

    This alone does NOT establish that the model learned the correct
    pathological feature.
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
    / "rsna_part270_local_patch_counterfactual_audit"
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

BACKGROUND_ID = 0
LFNN_ID = 2
RFNN_ID = 3

LFNN_NAME = "Left Neural Foraminal Narrowing"
RFNN_NAME = "Right Neural Foraminal Narrowing"

EXPECTED_CASES = 25
EXPECTED_PAIRED_SERIES = 9
EXPECTED_PAIRS = 45

PATCH_RADII = [2, 3]

# Fraction of the original local patch replaced.
#
# 0.25 = 25% replacement
# 0.50 = 50% replacement
# 1.00 = complete local replacement
REPLACEMENT_STRENGTHS = [
    0.25,
    0.50,
    1.00,
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
        p.numel()
        for p in model.parameters()
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

    if not torch.is_tensor(image):

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

    return (
        image
        .float()
        .to(device)
    )


# ============================================================================
# GEOMETRY
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


def canonical_to_index(
    canonical_point,
    shape,
):

    z = int(
        np.clip(
            round(
                float(canonical_point[0])
            ),
            0,
            int(shape[-3]) - 1,
        )
    )

    y = int(
        np.clip(
            round(
                float(canonical_point[1])
            ),
            0,
            int(shape[-2]) - 1,
        )
    )

    x = int(
        np.clip(
            round(
                float(canonical_point[2])
            ),
            0,
            int(shape[-1]) - 1,
        )
    )

    return z, y, x


# ============================================================================
# PATCH BOUNDS
# ============================================================================

def patch_bounds(
    center,
    shape,
    radius,
):

    z, y, x = center

    z0 = max(
        0,
        z - radius,
    )

    z1 = min(
        int(shape[-3]),
        z + radius + 1,
    )

    y0 = max(
        0,
        y - radius,
    )

    y1 = min(
        int(shape[-2]),
        y + radius + 1,
    )

    x0 = max(
        0,
        x - radius,
    )

    x1 = min(
        int(shape[-1]),
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
# SHELL VALUE
# ============================================================================

def local_shell_median(
    image,
    center,
    radius,
):

    z, y, x = center

    outer_radius = radius + 2

    (
        z0,
        z1,
        y0,
        y1,
        x0,
        x1,
    ) = patch_bounds(
        center,
        image.shape,
        outer_radius,
    )

    local = (
        image[
            0,
            0,
            z0:z1,
            y0:y1,
            x0:x1,
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .copy()
    )

    inner_bounds = patch_bounds(
        center,
        image.shape,
        radius,
    )

    iz0, iz1, iy0, iy1, ix0, ix1 = (
        inner_bounds
    )

    # Coordinates of the inner patch inside the
    # extracted outer array.
    rz0 = iz0 - z0
    rz1 = iz1 - z0

    ry0 = iy0 - y0
    ry1 = iy1 - y0

    rx0 = ix0 - x0
    rx1 = ix1 - x0

    mask = np.ones(
        local.shape,
        dtype=bool,
    )

    mask[
        rz0:rz1,
        ry0:ry1,
        rx0:rx1,
    ] = False

    shell_values = local[
        mask
    ]

    shell_values = shell_values[
        np.isfinite(
            shell_values
        )
    ]

    if len(shell_values) == 0:

        return float(
            np.nanmedian(local)
        )

    return float(
        np.median(
            shell_values
        )
    )


# ============================================================================
# COUNTERFACTUAL PATCH
# ============================================================================

def replace_local_patch(
    image,
    center,
    radius,
    replacement_strength,
):

    perturbed = (
        image.clone()
    )

    (
        z0,
        z1,
        y0,
        y1,
        x0,
        x1,
    ) = patch_bounds(
        center,
        perturbed.shape,
        radius,
    )

    shell_value = local_shell_median(
        image,
        center,
        radius,
    )

    original = (
        perturbed[
            :,
            :,
            z0:z1,
            y0:y1,
            x0:x1,
        ]
    )

    factor = float(
        replacement_strength
    )

    counterfactual = (
        original
        * (1.0 - factor)
        +
        factor
        * shell_value
    )

    perturbed[
        :,
        :,
        z0:z1,
        y0:y1,
        x0:x1,
    ] = counterfactual

    voxel_count = (
        (z1 - z0)
        *
        (y1 - y0)
        *
        (x1 - x0)
    )

    return (
        perturbed,
        shell_value,
        voxel_count,
    )


# ============================================================================
# OUTPUT METRICS
# ============================================================================

def output_metrics(
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

    bg_logit = float(
        logits[BACKGROUND_ID]
    )

    return {
        "rfnn_logit":
            rfnn_logit,

        "lfnn_logit":
            lfnn_logit,

        "background_logit":
            bg_logit,

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
                probabilities[BACKGROUND_ID]
            ),

        "rfnn_margin":
            rfnn_logit
            -
            bg_logit,

        "lfnn_margin":
            lfnn_logit
            -
            bg_logit,
    }


def sample_point_output(
    output,
    canonical_point,
):

    z, y, x = canonical_to_index(
        canonical_point,
        output.shape,
    )

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

    return output_metrics(
        logits
    )


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

    image_tensor = prepare_image(
        image,
        device,
    )

    point_df = pd.DataFrame(
        points
    )

    lfnn_df = point_df[
        point_df["class_name"]
        == LFNN_NAME
    ].copy()

    rfnn_df = point_df[
        point_df["class_name"]
        == RFNN_NAME
    ].copy()

    if len(lfnn_df) != 5:
        return []

    if len(rfnn_df) != 5:
        return []

    common_levels = sorted(
        set(
            lfnn_df["level"]
        )
        &
        set(
            rfnn_df["level"]
        )
    )

    if len(common_levels) != 5:
        return []

    with torch.no_grad():

        baseline_output = model(
            image_tensor
        )

    records = []

    for level in common_levels:

        lrow = lfnn_df[
            lfnn_df["level"]
            == level
        ]

        rrow = rfnn_df[
            rfnn_df["level"]
            == level
        ]

        if len(lrow) != 1:
            continue

        if len(rrow) != 1:
            continue

        point_rows = {
            "lfnn":
                lrow.iloc[0],

            "rfnn":
                rrow.iloc[0],
        }

        for point_type, point_row in (
            point_rows.items()
        ):

            canonical_point = (
                get_canonical_point(
                    point_row,
                    geometry,
                )
            )

            center = canonical_to_index(
                canonical_point,
                image_tensor.shape,
            )

            baseline = sample_point_output(
                baseline_output,
                canonical_point,
            )

            for radius in PATCH_RADII:

                shell_value = (
                    local_shell_median(
                        image_tensor,
                        center,
                        radius,
                    )
                )

                for strength in (
                    REPLACEMENT_STRENGTHS
                ):

                    (
                        counterfactual,
                        shell_value_actual,
                        voxel_count,
                    ) = replace_local_patch(
                        image_tensor,
                        center,
                        radius,
                        strength,
                    )

                    with torch.no_grad():

                        output = model(
                            counterfactual
                        )

                    perturbed = (
                        sample_point_output(
                            output,
                            canonical_point,
                        )
                    )

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

                            "replacement_strength":
                                float(strength),

                            "replacement_percent":
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

                            "shell_median":
                                float(
                                    shell_value_actual
                                ),

                            "perturbed_voxels":
                                int(
                                    voxel_count
                                ),

                            "baseline_rfnn_logit":
                                baseline[
                                    "rfnn_logit"
                                ],

                            "perturbed_rfnn_logit":
                                perturbed[
                                    "rfnn_logit"
                                ],

                            "delta_rfnn_logit":
                                (
                                    perturbed[
                                        "rfnn_logit"
                                    ]
                                    -
                                    baseline[
                                        "rfnn_logit"
                                    ]
                                ),

                            "baseline_lfnn_logit":
                                baseline[
                                    "lfnn_logit"
                                ],

                            "perturbed_lfnn_logit":
                                perturbed[
                                    "lfnn_logit"
                                ],

                            "delta_lfnn_logit":
                                (
                                    perturbed[
                                        "lfnn_logit"
                                    ]
                                    -
                                    baseline[
                                        "lfnn_logit"
                                    ]
                                ),

                            "baseline_background_logit":
                                baseline[
                                    "background_logit"
                                ],

                            "perturbed_background_logit":
                                perturbed[
                                    "background_logit"
                                ],

                            "delta_background_logit":
                                (
                                    perturbed[
                                        "background_logit"
                                    ]
                                    -
                                    baseline[
                                        "background_logit"
                                    ]
                                ),

                            "baseline_rfnn_margin":
                                baseline[
                                    "rfnn_margin"
                                ],

                            "perturbed_rfnn_margin":
                                perturbed[
                                    "rfnn_margin"
                                ],

                            "delta_rfnn_margin":
                                (
                                    perturbed[
                                        "rfnn_margin"
                                    ]
                                    -
                                    baseline[
                                        "rfnn_margin"
                                    ]
                                ),

                            "baseline_lfnn_margin":
                                baseline[
                                    "lfnn_margin"
                                ],

                            "perturbed_lfnn_margin":
                                perturbed[
                                    "lfnn_margin"
                                ],

                            "delta_lfnn_margin":
                                (
                                    perturbed[
                                        "lfnn_margin"
                                    ]
                                    -
                                    baseline[
                                        "lfnn_margin"
                                    ]
                                ),

                            "baseline_rfnn_probability":
                                baseline[
                                    "rfnn_probability"
                                ],

                            "perturbed_rfnn_probability":
                                perturbed[
                                    "rfnn_probability"
                                ],

                            "delta_rfnn_probability":
                                (
                                    perturbed[
                                        "rfnn_probability"
                                    ]
                                    -
                                    baseline[
                                        "rfnn_probability"
                                    ]
                                ),

                            "baseline_background_probability":
                                baseline[
                                    "background_probability"
                                ],

                            "perturbed_background_probability":
                                perturbed[
                                    "background_probability"
                                ],

                            "delta_background_probability":
                                (
                                    perturbed[
                                        "background_probability"
                                    ]
                                    -
                                    baseline[
                                        "background_probability"
                                    ]
                                ),
                        }
                    )

    return records


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    records_df,
):

    grouped = records_df.groupby(
        [
            "point_type",
            "radius",
            "replacement_strength",
        ],
        sort=False,
    )

    rows = []

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

                "replacement_strength":
                    float(strength),

                "replacement_percent":
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
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# PAIRED DIFFERENCE
# ============================================================================

def build_paired_summary(
    records_df,
):

    pivot = records_df.pivot_table(
        index=[
            "study_id",
            "series_id",
            "level",
            "radius",
            "replacement_strength",
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
    ).reset_index()

    flattened = []

    for column in pivot.columns:

        if isinstance(
            column,
            tuple,
        ):

            if column[1] == "":
                flattened.append(
                    column[0]
                )
            else:
                flattened.append(
                    "_".join(
                        str(v)
                        for v in column
                        if str(v) != ""
                    )
                )

        else:

            flattened.append(
                column
            )

    pivot.columns = flattened

    metrics = [
        "delta_rfnn_logit",
        "delta_lfnn_logit",
        "delta_background_logit",
        "delta_rfnn_margin",
        "delta_lfnn_margin",
        "delta_rfnn_probability",
    ]

    for metric in metrics:

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
            "replacement_strength",
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

                "replacement_strength":
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
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.70")
    print("LOCAL PATCH COUNTERFACTUAL AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    print()
    print(
        "Patch radii: "
        f"{PATCH_RADII}"
    )

    print(
        "Replacement strengths: "
        f"{REPLACEMENT_STRENGTHS}"
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
    print("RUNNING LOCAL PATCH COUNTERFACTUAL AUDIT")
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
        f"Counterfactual records: "
        f"{len(records_df)}"
    )

    expected_records = (
        EXPECTED_PAIRS
        *
        len(PATCH_RADII)
        *
        2
        *
        len(REPLACEMENT_STRENGTHS)
    )

    if paired_series != (
        EXPECTED_PAIRED_SERIES
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_PAIRED_SERIES} "
            f"paired series, found {paired_series}."
        )

    if len(records_df) != (
        expected_records
    ):

        raise RuntimeError(
            f"Expected {expected_records} "
            f"records, found {len(records_df)}."
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
            f"Expected {EXPECTED_PAIRS} "
            f"paired observations, "
            f"found {len(unique_pairs)}."
        )

    print()
    print(
        "Counterfactual validation: PASSED"
    )

    print(
        f"Paired observations: "
        f"{len(unique_pairs)}"
    )

    print(
        f"Expected records: "
        f"{expected_records}"
    )

    summary_df = build_summary(
        records_df
    )

    paired_df = build_paired_summary(
        records_df
    )

    level_df = build_level_summary(
        records_df
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    records_path = (
        OUTPUT_DIR
        / "part270_local_patch_counterfactual_records.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part270_local_patch_counterfactual_summary.csv"
    )

    paired_path = (
        OUTPUT_DIR
        / "part270_paired_counterfactual_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part270_local_patch_counterfactual_level_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part270_local_patch_counterfactual_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
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

    report = {
        "analysis":
            "Part 2.70 local patch counterfactual audit",

        "validation_cases":
            EXPECTED_CASES,

        "paired_validation_series":
            paired_series,

        "paired_observations":
            len(unique_pairs),

        "patch_radii":
            PATCH_RADII,

        "replacement_strengths":
            REPLACEMENT_STRENGTHS,

        "records":
            len(records_df),

        "checkpoint":
            str(CHECKPOINT_PATH),

        "checkpoint_modified":
            False,

        "dashboard_modified":
            False,

        "training_performed":
            False,

        "summary":
            summary_df.to_dict(
                orient="records"
            ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ========================================================================
    # DISPLAY
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.70 COUNTERFACTUAL SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.70 PAIRED RFNN-LFNN SUMMARY")
    print("=" * 80)

    print(
        paired_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.70 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.70 COMPLETE")
    print("=" * 80)

    print(
        "Analysis only."
    )

    print(
        "Part 2.70 checkpoint unchanged."
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


if __name__ == "__main__":
    main()