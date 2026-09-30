"""
PART 2.71
PAIRED CONTRALATERAL PATCH-SWAP COUNTERFACTUAL AUDIT

Purpose
-------
Test whether LFNN/RFNN model responses follow local MRI content
or remain tied primarily to spatial location.

For every paired LFNN/RFNN observation from the exact Part 2.27
validation cohort:

    1. Run the original image.
    2. Extract the LFNN local patch.
    3. Extract the RFNN local patch.
    4. Replace LFNN location with RFNN patch.
    5. Replace RFNN location with LFNN patch.
    6. Re-run the unchanged model.
    7. Measure logit, probability and margin changes.

This is an analysis-only counterfactual experiment.

NO:
    - training
    - checkpoint modification
    - dashboard modification
    - fabricated voxel ground truth
    - clinical claim

Checkpoint:
    Part 2.27 best macro disease checkpoint.

Expected cohort:
    25 validation cases
    9 paired validation series
    45 paired LFNN/RFNN observations

Patch radii:
    2
    3
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
    / "rsna_part271_paired_patch_swap_counterfactual_audit"
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
SCS_ID = 1
LFNN_ID = 2
RFNN_ID = 3
LSS_ID = 4
RSS_ID = 5

LFNN_NAME = "Left Neural Foraminal Narrowing"
RFNN_NAME = "Right Neural Foraminal Narrowing"

EXPECTED_CASES = 25
EXPECTED_PAIRED_SERIES = 9
EXPECTED_PAIRS = 45

PATCH_RADII = [2, 3]


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
# IMAGE
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
# PATCH UTILITIES
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


def extract_patch(
    image,
    center,
    radius,
):

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
        radius,
    )

    patch = (
        image[
            :,
            :,
            z0:z1,
            y0:y1,
            x0:x1,
        ]
        .clone()
    )

    return (
        patch,
        (
            z0,
            z1,
            y0,
            y1,
            x0,
            x1,
        ),
    )


def patch_shape_is_valid(
    patch,
    target_bounds,
):

    target_shape = (
        target_bounds[1] - target_bounds[0],
        target_bounds[3] - target_bounds[2],
        target_bounds[5] - target_bounds[4],
    )

    patch_shape = (
        int(patch.shape[-3]),
        int(patch.shape[-2]),
        int(patch.shape[-1]),
    )

    return patch_shape == target_shape


def paste_patch(
    image,
    patch,
    bounds,
):

    (
        z0,
        z1,
        y0,
        y1,
        x0,
        x1,
    ) = bounds

    result = image.clone()

    result[
        :,
        :,
        z0:z1,
        y0:y1,
        x0:x1,
    ] = patch

    return result


# ============================================================================
# OUTPUT
# ============================================================================

def output_metrics(
    logits,
):

    probabilities = (
        torch.softmax(
            torch.tensor(
                logits,
                dtype=torch.float32,
            ),
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
            rfnn_logit - bg_logit,

        "lfnn_margin":
            lfnn_logit - bg_logit,
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
# SINGLE PAIRED OBSERVATION
# ============================================================================

def analyze_pair(
    model,
    device,
    image_tensor,
    geometry,
    lfnn_point,
    rfnn_point,
    study_id,
    series_id,
    level,
):

    lfnn_canonical = (
        get_canonical_point(
            lfnn_point,
            geometry,
        )
    )

    rfnn_canonical = (
        get_canonical_point(
            rfnn_point,
            geometry,
        )
    )

    lfnn_center = canonical_to_index(
        lfnn_canonical,
        image_tensor.shape,
    )

    rfnn_center = canonical_to_index(
        rfnn_canonical,
        image_tensor.shape,
    )

    records = []

    with torch.no_grad():

        baseline_output = model(
            image_tensor
        )

    baseline_lfnn = sample_point_output(
        baseline_output,
        lfnn_canonical,
    )

    baseline_rfnn = sample_point_output(
        baseline_output,
        rfnn_canonical,
    )

    for radius in PATCH_RADII:

        lfnn_patch, lfnn_bounds = (
            extract_patch(
                image_tensor,
                lfnn_center,
                radius,
            )
        )

        rfnn_patch, rfnn_bounds = (
            extract_patch(
                image_tensor,
                rfnn_center,
                radius,
            )
        )

        # --------------------------------------------------------------------
        # If boundary geometry makes patch shapes different, skip safely.
        # --------------------------------------------------------------------

        if not patch_shape_is_valid(
            lfnn_patch,
            rfnn_bounds,
        ):

            continue

        if not patch_shape_is_valid(
            rfnn_patch,
            lfnn_bounds,
        ):

            continue

        # ====================================================================
        # CONDITION A
        # RFNN PATCH -> LFNN LOCATION
        # ====================================================================

        swapped_lfnn_image = paste_patch(
            image_tensor,
            rfnn_patch,
            lfnn_bounds,
        )

        with torch.no_grad():

            swapped_lfnn_output = model(
                swapped_lfnn_image
            )

        swapped_lfnn_at_lfnn = (
            sample_point_output(
                swapped_lfnn_output,
                lfnn_canonical,
            )
        )

        # ====================================================================
        # CONDITION B
        # LFNN PATCH -> RFNN LOCATION
        # ====================================================================

        swapped_rfnn_image = paste_patch(
            image_tensor,
            lfnn_patch,
            rfnn_bounds,
        )

        with torch.no_grad():

            swapped_rfnn_output = model(
                swapped_rfnn_image
            )

        swapped_rfnn_at_rfnn = (
            sample_point_output(
                swapped_rfnn_output,
                rfnn_canonical,
            )
        )

        # ====================================================================
        # RECORD
        # ====================================================================

        records.append(
            {
                "study_id":
                    study_id,

                "series_id":
                    series_id,

                "level":
                    str(level),

                "radius":
                    int(radius),

                "patch_side":
                    int(2 * radius + 1),

                # ------------------------------------------------------------
                # Baseline RFNN location
                # ------------------------------------------------------------

                "baseline_rfnn_location_rfnn_logit":
                    baseline_rfnn[
                        "rfnn_logit"
                    ],

                "baseline_rfnn_location_lfnn_logit":
                    baseline_rfnn[
                        "lfnn_logit"
                    ],

                "baseline_rfnn_location_background_logit":
                    baseline_rfnn[
                        "background_logit"
                    ],

                "baseline_rfnn_location_rfnn_margin":
                    baseline_rfnn[
                        "rfnn_margin"
                    ],

                "baseline_rfnn_location_lfnn_margin":
                    baseline_rfnn[
                        "lfnn_margin"
                    ],

                "baseline_rfnn_location_rfnn_probability":
                    baseline_rfnn[
                        "rfnn_probability"
                    ],

                # ------------------------------------------------------------
                # RFNN patch moved into LFNN location
                # ------------------------------------------------------------

                "swap_rfnn_patch_to_lfnn_location_rfnn_logit":
                    swapped_lfnn_at_lfnn[
                        "rfnn_logit"
                    ],

                "swap_rfnn_patch_to_lfnn_location_lfnn_logit":
                    swapped_lfnn_at_lfnn[
                        "lfnn_logit"
                    ],

                "swap_rfnn_patch_to_lfnn_location_background_logit":
                    swapped_lfnn_at_lfnn[
                        "background_logit"
                    ],

                "swap_rfnn_patch_to_lfnn_location_rfnn_margin":
                    swapped_lfnn_at_lfnn[
                        "rfnn_margin"
                    ],

                "swap_rfnn_patch_to_lfnn_location_lfnn_margin":
                    swapped_lfnn_at_lfnn[
                        "lfnn_margin"
                    ],

                "swap_rfnn_patch_to_lfnn_location_rfnn_probability":
                    swapped_lfnn_at_lfnn[
                        "rfnn_probability"
                    ],

                # ------------------------------------------------------------
                # Baseline LFNN location
                # ------------------------------------------------------------

                "baseline_lfnn_location_rfnn_logit":
                    baseline_lfnn[
                        "rfnn_logit"
                    ],

                "baseline_lfnn_location_lfnn_logit":
                    baseline_lfnn[
                        "lfnn_logit"
                    ],

                "baseline_lfnn_location_background_logit":
                    baseline_lfnn[
                        "background_logit"
                    ],

                "baseline_lfnn_location_rfnn_margin":
                    baseline_lfnn[
                        "rfnn_margin"
                    ],

                "baseline_lfnn_location_lfnn_margin":
                    baseline_lfnn[
                        "lfnn_margin"
                    ],

                "baseline_lfnn_location_lfnn_probability":
                    baseline_lfnn[
                        "lfnn_probability"
                    ],

                # ------------------------------------------------------------
                # LFNN patch moved into RFNN location
                # ------------------------------------------------------------

                "swap_lfnn_patch_to_rfnn_location_rfnn_logit":
                    swapped_rfnn_at_rfnn[
                        "rfnn_logit"
                    ],

                "swap_lfnn_patch_to_rfnn_location_lfnn_logit":
                    swapped_rfnn_at_rfnn[
                        "lfnn_logit"
                    ],

                "swap_lfnn_patch_to_rfnn_location_background_logit":
                    swapped_rfnn_at_rfnn[
                        "background_logit"
                    ],

                "swap_lfnn_patch_to_rfnn_location_rfnn_margin":
                    swapped_rfnn_at_rfnn[
                        "rfnn_margin"
                    ],

                "swap_lfnn_patch_to_rfnn_location_lfnn_margin":
                    swapped_rfnn_at_rfnn[
                        "lfnn_margin"
                    ],

                "swap_lfnn_patch_to_rfnn_location_lfnn_probability":
                    swapped_rfnn_at_rfnn[
                        "lfnn_probability"
                    ],

                # ------------------------------------------------------------
                # Counterfactual changes
                # ------------------------------------------------------------

                "delta_rfnn_logit_at_lfnn_from_rfnn_patch":
                    (
                        swapped_lfnn_at_lfnn[
                            "rfnn_logit"
                        ]
                        -
                        baseline_lfnn[
                            "rfnn_logit"
                        ]
                    ),

                "delta_lfnn_logit_at_lfnn_from_rfnn_patch":
                    (
                        swapped_lfnn_at_lfnn[
                            "lfnn_logit"
                        ]
                        -
                        baseline_lfnnn[
                            "lfnn_logit"
                        ]
                    )
                    if False else (
                        swapped_lfnn_at_lfnn[
                            "lfnn_logit"
                        ]
                        -
                        baseline_lfnn[
                            "lfnn_logit"
                        ]
                    ),

                "delta_rfnn_margin_at_lfnn_from_rfnn_patch":
                    (
                        swapped_lfnn_at_lfnn[
                            "rfnn_margin"
                        ]
                        -
                        baseline_lfnn[
                            "rfnn_margin"
                        ]
                    ),

                "delta_rfnn_probability_at_lfnn_from_rfnn_patch":
                    (
                        swapped_lfnn_at_lfnn[
                            "rfnn_probability"
                        ]
                        -
                        baseline_lfnn[
                            "rfnn_probability"
                        ]
                    ),

                "delta_rfnn_logit_at_rfnn_from_lfnn_patch":
                    (
                        swapped_rfnn_at_rfnn[
                            "rfnn_logit"
                        ]
                        -
                        baseline_rfnn[
                            "rfnn_logit"
                        ]
                    ),

                "delta_lfnn_logit_at_rfnn_from_lfnn_patch":
                    (
                        swapped_rfnn_at_rfnn[
                            "lfnn_logit"
                        ]
                        -
                        baseline_rfnn[
                            "lfnn_logit"
                        ]
                    ),

                "delta_rfnn_margin_at_rfnn_from_lfnn_patch":
                    (
                        swapped_rfnn_at_rfnn[
                            "rfnn_margin"
                        ]
                        -
                        baseline_rfnn[
                            "rfnn_margin"
                        ]
                    ),

                "delta_rfnn_probability_at_rfnn_from_lfnn_patch":
                    (
                        swapped_rfnn_at_rfnn[
                            "rfnn_probability"
                        ]
                        -
                        baseline_rfnn[
                            "rfnn_probability"
                        ]
                    ),

                # ------------------------------------------------------------
                # Spatial-vs-content diagnostic
                # ------------------------------------------------------------

                "rfnn_margin_baseline_difference":
                    (
                        baseline_rfnn[
                            "rfnn_margin"
                        ]
                        -
                        baseline_lfnn[
                            "rfnn_margin"
                        ]
                    ),

                "rfnn_margin_after_cross_swap_difference":
                    (
                        swapped_lfnn_at_lfnn[
                            "rfnn_margin"
                        ]
                        -
                        swapped_rfnn_at_rfnn[
                            "rfnn_margin"
                        ]
                    ),
            }
        )

    return records


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

        pair_records = analyze_pair(
            model=model,
            device=device,
            image_tensor=image_tensor,
            geometry=geometry,
            lfnn_point=lrow.iloc[0],
            rfnn_point=rrow.iloc[0],
            study_id=study_id,
            series_id=series_id,
            level=level,
        )

        records.extend(
            pair_records
        )

    return records


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    records_df,
):

    grouped = records_df.groupby(
        ["radius"],
        sort=False,
    )

    rows = []

    for group_key, group in grouped:

        if isinstance(group_key, tuple):
            radius = group_key[0]
        else:
            radius = group_key

        rows.append(
            {
                "radius":
                    int(radius),

                "patch_side":
                    int(2 * int(radius) + 1),

                "records":
                    int(len(group)),

                "mean_delta_rfnn_logit_at_lfnn_from_rfnn_patch":
                    float(
                        group[
                            "delta_rfnn_logit_at_lfnn_from_rfnn_patch"
                        ].mean()
                    ),

                "mean_delta_rfnn_margin_at_lfnn_from_rfnn_patch":
                    float(
                        group[
                            "delta_rfnn_margin_at_lfnn_from_rfnn_patch"
                        ].mean()
                    ),

                "mean_delta_rfnn_probability_at_lfnn_from_rfnn_patch":
                    float(
                        group[
                            "delta_rfnn_probability_at_lfnn_from_rfnn_patch"
                        ].mean()
                    ),

                "mean_delta_rfnn_logit_at_rfnn_from_lfnn_patch":
                    float(
                        group[
                            "delta_rfnn_logit_at_rfnn_from_lfnn_patch"
                        ].mean()
                    ),

                "mean_delta_rfnn_margin_at_rfnn_from_lfnn_patch":
                    float(
                        group[
                            "delta_rfnn_margin_at_rfnn_from_lfnn_patch"
                        ].mean()
                    ),

                "mean_delta_rfnn_probability_at_rfnn_from_lfnn_patch":
                    float(
                        group[
                            "delta_rfnn_probability_at_rfnn_from_lfnn_patch"
                        ].mean()
                    ),

                "mean_cross_swap_rfnn_margin_difference":
                    float(
                        group[
                            "rfnn_margin_after_cross_swap_difference"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(rows)


def build_level_summary(
    records_df,
):

    grouped = records_df.groupby(
        [
            "level",
            "radius",
        ],
        sort=False,
    )

    rows = []

    for group_key, group in grouped:

        level, radius = group_key

        rows.append(
            {
                "level":
                    str(level),

                "radius":
                    int(radius),

                "records":
                    int(len(group)),

                "mean_delta_rfnn_margin_at_lfnn_from_rfnn_patch":
                    float(
                        group[
                            "delta_rfnn_margin_at_lfnn_from_rfnn_patch"
                        ].mean()
                    ),

                "mean_delta_rfnn_margin_at_rfnn_from_lfnn_patch":
                    float(
                        group[
                            "delta_rfnn_margin_at_rfnn_from_lfnn_patch"
                        ].mean()
                    ),

                "mean_delta_rfnn_probability_at_lfnn_from_rfnn_patch":
                    float(
                        group[
                            "delta_rfnn_probability_at_lfnn_from_rfnn_patch"
                        ].mean()
                    ),

                "mean_delta_rfnn_probability_at_rfnn_from_lfnn_patch":
                    float(
                        group[
                            "delta_rfnn_probability_at_rfnn_from_lfnn_patch"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.71")
    print("PAIRED CONTRALATERAL PATCH-SWAP COUNTERFACTUAL AUDIT")
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
    print("RUNNING PATCH-SWAP COUNTERFACTUAL AUDIT")
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
        f"Patch-swap records: "
        f"{len(records_df)}"
    )

    expected_records = (
        EXPECTED_PAIRS
        *
        len(PATCH_RADII)
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
            f"paired observations, found "
            f"{len(unique_pairs)}."
        )

    print()
    print(
        "Patch-swap validation: PASSED"
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

    level_df = build_level_summary(
        records_df
    )

    records_path = (
        OUTPUT_DIR
        / "part271_patch_swap_records.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part271_patch_swap_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part271_patch_swap_level_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part271_patch_swap_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    level_df.to_csv(
        level_path,
        index=False,
    )

    report = {
        "analysis":
            "Part 2.71 paired contralateral patch-swap counterfactual audit",

        "validation_cases":
            EXPECTED_CASES,

        "paired_validation_series":
            paired_series,

        "paired_observations":
            len(unique_pairs),

        "patch_radii":
            PATCH_RADII,

        "records":
            len(records_df),

        "checkpoint":
            str(CHECKPOINT_PATH),

        "training_performed":
            False,

        "checkpoint_modified":
            False,

        "dashboard_modified":
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

    print()
    print("=" * 80)
    print("PART 2.71 PATCH-SWAP SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.71 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.71 COMPLETE")
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
    print(level_path)
    print(json_path)


if __name__ == "__main__":
    main()