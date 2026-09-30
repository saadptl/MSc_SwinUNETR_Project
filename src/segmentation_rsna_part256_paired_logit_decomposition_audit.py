"""
PART 2.56
PAIRED LFNN / RFNN LOGIT DECOMPOSITION AUDIT

Purpose
-------
Decompose the LFNN/RFNN foreground failure using raw model logits.

For the exact 25-case validation cohort and Part 2.27 checkpoint:

    LFNN annotation/evidence:
        LFNN logit
        Background logit
        LFNN - Background
        LSS - Background
        SCS - Background
        RFNN - Background

    RFNN annotation/evidence:
        RFNN logit
        Background logit
        RFNN - Background
        LSS - Background
        SCS - Background
        LFNN - Background

The analysis determines whether the RFNN deficit is primarily caused by:

    A. weak RFNN logits,
    B. excessive Background logits,
    C. both,
    D. another foreground competitor.

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
    / "rsna_part256_paired_logit_decomposition_audit"
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

EXPECTED_SERIES = 25

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


# ============================================================================
# LOAD EXACT VALIDATION COHORT
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
            "Expected select_validation_series() "
            "to return a DataFrame."
        )

    if len(selected) != EXPECTED_SERIES:

        raise RuntimeError(
            f"Expected {EXPECTED_SERIES} validation series, "
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

    if len(cases) != EXPECTED_SERIES:

        raise RuntimeError(
            "Validation case count mismatch."
        )

    return cases


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
        and "model_state_dict" in checkpoint
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

    if canonical.size < 3:

        raise RuntimeError(
            "Canonical point does not contain "
            "three coordinates."
        )

    return canonical[:3]


# ============================================================================
# SAMPLE LOGITS
# ============================================================================

def sample_logits(
    logits,
    canonical,
):

    if logits.ndim == 5:

        logits = logits[0]

    if logits.ndim != 4:

        raise RuntimeError(
            f"Unexpected logits shape: {logits.shape}"
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

    channels, depth, height, width = (
        logits.shape
    )

    if not (
        0 <= z < depth
        and
        0 <= y < height
        and
        0 <= x < width
    ):

        raise RuntimeError(
            "Canonical point is outside "
            "model grid."
        )

    values = (
        logits[
            :,
            z,
            y,
            x,
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    return values


# ============================================================================
# SAMPLE PROBABILITIES
# ============================================================================

def sample_probabilities(
    probabilities,
    canonical,
):

    if probabilities.ndim == 5:

        probabilities = probabilities[0]

    if probabilities.ndim != 4:

        raise RuntimeError(
            f"Unexpected probability shape: "
            f"{probabilities.shape}"
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

    channels, depth, height, width = (
        probabilities.shape
    )

    if not (
        0 <= z < depth
        and
        0 <= y < height
        and
        0 <= x < width
    ):

        raise RuntimeError(
            "Canonical point is outside "
            "model grid."
        )

    values = (
        probabilities[
            :,
            z,
            y,
            x,
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    return values


# ============================================================================
# FIND LOCAL MAXIMUM
# ============================================================================

def find_local_maximum(
    logits,
    probabilities,
    center,
    class_id,
    radius=6,
):

    if logits.ndim == 5:

        logits = logits[0]

    if probabilities.ndim == 5:

        probabilities = probabilities[0]

    _, depth, height, width = (
        logits.shape
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

    local_probability = (
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
        np.argmax(
            local_probability
        ),
        local_probability.shape,
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

    local_canonical = np.array(
        [
            mz,
            my,
            mx,
        ],
        dtype=np.float64,
    )

    local_logits = sample_logits(
        logits,
        local_canonical,
    )

    local_probabilities = sample_probabilities(
        probabilities,
        local_canonical,
    )

    return (
        local_canonical,
        local_logits,
        local_probabilities,
    )


# ============================================================================
# ANALYZE ONE POINT
# ============================================================================

def analyze_point(
    point,
    logits,
    probabilities,
    geometry,
    side,
    study_id,
    series_id,
):

    if side == "LFNN":

        class_id = LFNN_ID

    elif side == "RFNN":

        class_id = RFNN_ID

    else:

        raise ValueError(
            f"Unknown side: {side}"
        )

    annotation_canonical = (
        point_to_canonical(
            point,
            geometry,
        )
    )

    (
        maximum_canonical,
        maximum_logits,
        maximum_probabilities,
    ) = find_local_maximum(
        logits,
        probabilities,
        annotation_canonical,
        class_id,
        radius=6,
    )

    annotation_logits = sample_logits(
        logits,
        annotation_canonical,
    )

    annotation_probabilities = (
        sample_probabilities(
            probabilities,
            annotation_canonical,
        )
    )

    # ------------------------------------------------------------------------
    # Annotation raw logits
    # ------------------------------------------------------------------------

    row = {
        "study_id":
            study_id,

        "series_id":
            series_id,

        "level":
            str(
                point["level"]
            ),

        "side":
            side,

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

        # ---------------------------------------------------------------
        # Annotation logits
        # ---------------------------------------------------------------

        "annotation_background_logit":
            float(
                annotation_logits[
                    BACKGROUND_ID
                ]
            ),

        "annotation_scs_logit":
            float(
                annotation_logits[
                    SCS_ID
                ]
            ),

        "annotation_lfnn_logit":
            float(
                annotation_logits[
                    LFNN_ID
                ]
            ),

        "annotation_rfnn_logit":
            float(
                annotation_logits[
                    RFNN_ID
                ]
            ),

        "annotation_lss_logit":
            float(
                annotation_logits[
                    LSS_ID
                ]
            ),

        "annotation_rss_logit":
            float(
                annotation_logits[
                    RSS_ID
                ]
            ),

        # ---------------------------------------------------------------
        # Maximum logits
        # ---------------------------------------------------------------

        "maximum_background_logit":
            float(
                maximum_logits[
                    BACKGROUND_ID
                ]
            ),

        "maximum_scs_logit":
            float(
                maximum_logits[
                    SCS_ID
                ]
            ),

        "maximum_lfnn_logit":
            float(
                maximum_logits[
                    LFNN_ID
                ]
            ),

        "maximum_rfnn_logit":
            float(
                maximum_logits[
                    RFNN_ID
                ]
            ),

        "maximum_lss_logit":
            float(
                maximum_logits[
                    LSS_ID
                ]
            ),

        "maximum_rss_logit":
            float(
                maximum_logits[
                    RSS_ID
                ]
            ),

        # ---------------------------------------------------------------
        # Maximum probabilities
        # ---------------------------------------------------------------

        "maximum_background_probability":
            float(
                maximum_probabilities[
                    BACKGROUND_ID
                ]
            ),

        "maximum_scs_probability":
            float(
                maximum_probabilities[
                    SCS_ID
                ]
            ),

        "maximum_lfnn_probability":
            float(
                maximum_probabilities[
                    LFNN_ID
                ]
            ),

        "maximum_rfnn_probability":
            float(
                maximum_probabilities[
                    RFNN_ID
                ]
            ),

        "maximum_lss_probability":
            float(
                maximum_probabilities[
                    LSS_ID
                ]
            ),

        "maximum_rss_probability":
            float(
                maximum_probabilities[
                    RSS_ID
                ]
            ),
    }

    # =========================================================================
    # Annotation margins
    # =========================================================================

    row[
        "annotation_target_logit"
    ] = float(
        annotation_logits[
            class_id
        ]
    )

    row[
        "annotation_target_probability"
    ] = float(
        annotation_probabilities[
            class_id
        ]
    )

    row[
        "annotation_target_minus_background"
    ] = float(
        annotation_logits[
            class_id
        ]
        -
        annotation_logits[
            BACKGROUND_ID
        ]
    )

    row[
        "annotation_background_logit"
    ] = float(
        annotation_logits[
            BACKGROUND_ID
        ]
    )

    # =========================================================================
    # Maximum margins
    # =========================================================================

    row[
        "maximum_target_logit"
    ] = float(
        maximum_logits[
            class_id
        ]
    )

    row[
        "maximum_target_probability"
    ] = float(
        maximum_probabilities[
            class_id
        ]
    )

    row[
        "maximum_target_minus_background"
    ] = float(
        maximum_logits[
            class_id
        ]
        -
        maximum_logits[
            BACKGROUND_ID
        ]
    )

    row[
        "maximum_background_minus_target"
    ] = float(
        maximum_logits[
            BACKGROUND_ID
        ]
        -
        maximum_logits[
            class_id
        ]
    )

    # =========================================================================
    # Target versus competitors
    # =========================================================================

    row[
        "maximum_target_minus_scs"
    ] = float(
        maximum_logits[
            class_id
        ]
        -
        maximum_logits[
            SCS_ID
        ]
    )

    row[
        "maximum_target_minus_lss"
    ] = float(
        maximum_logits[
            class_id
        ]
        -
        maximum_logits[
            LSS_ID
        ]
    )

    row[
        "maximum_target_minus_rss"
    ] = float(
        maximum_logits[
            class_id
        ]
        -
        maximum_logits[
            RSS_ID
        ]
    )

    opposite_id = (
        RFNN_ID
        if class_id == LFNN_ID
        else LFNN_ID
    )

    row[
        "maximum_target_minus_opposite_foraminal"
    ] = float(
        maximum_logits[
            class_id
        ]
        -
        maximum_logits[
            opposite_id
        ]
    )

    # =========================================================================
    # Predicted class
    # =========================================================================

    row[
        "maximum_predicted_class_id"
    ] = int(
        np.argmax(
            maximum_logits
        )
    )

    row[
        "maximum_predicted_class"
    ] = CLASS_NAMES[
        row[
            "maximum_predicted_class_id"
        ]
    ]

    return row


# ============================================================================
# BUILD PAIRS
# ============================================================================

def build_pairs(
    records,
):

    df = pd.DataFrame(
        records
    )

    if len(df) != 90:

        raise RuntimeError(
            f"Expected 90 records, found {len(df)}."
        )

    lfnn = df[
        df["side"]
        == "LFNN"
    ].copy()

    rfnn = df[
        df["side"]
        == "RFNN"
    ].copy()

    if len(lfnn) != EXPECTED_LFNN:

        raise RuntimeError(
            "Expected 45 LFNN records."
        )

    if len(rfnn) != EXPECTED_RFNN:

        raise RuntimeError(
            "Expected 45 RFNN records."
        )

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

    if len(pairs) != EXPECTED_PAIRS:

        raise RuntimeError(
            "Expected 45 paired observations, "
            f"found {len(pairs)}."
        )

    return df, pairs


# ============================================================================
# LOGIT DECOMPOSITION
# ============================================================================

def build_decomposition_summary(
    pairs,
):

    rows = []

    comparisons = [
        (
            "target_logit",
            "maximum_target_logit_lfnn",
            "maximum_target_logit_rfnn",
        ),
        (
            "background_logit",
            "maximum_background_logit_lfnn",
            "maximum_background_logit_rfnn",
        ),
        (
            "target_minus_background",
            "maximum_target_minus_background_lfnn",
            "maximum_target_minus_background_rfnn",
        ),
        (
            "target_minus_scs",
            "maximum_target_minus_scs_lfnn",
            "maximum_target_minus_scs_rfnn",
        ),
        (
            "target_minus_lss",
            "maximum_target_minus_lss_lfnn",
            "maximum_target_minus_lss_rfnn",
        ),
        (
            "target_minus_rss",
            "maximum_target_minus_rss_lfnn",
            "maximum_target_minus_rss_rfnn",
        ),
        (
            "target_minus_opposite_foraminal",
            "maximum_target_minus_opposite_foraminal_lfnn",
            "maximum_target_minus_opposite_foraminal_rfnn",
        ),
        (
            "target_probability",
            "maximum_target_probability_lfnn",
            "maximum_target_probability_rfnn",
        ),
    ]

    for name, lcol, rcol in comparisons:

        lvalues = pairs[
            lcol
        ].astype(float)

        rvalues = pairs[
            rcol
        ].astype(float)

        difference = (
            rvalues
            - lvalues
        )

        rows.append(
            {
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

                "absolute_pair_difference":
                    float(
                        difference.abs().mean()
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

    return pd.DataFrame(
        rows
    )


# ============================================================================
# RAW LOGIT COMPONENT DECOMPOSITION
# ============================================================================

def build_component_decomposition(
    pairs,
):

    # ------------------------------------------------------------------------
    # RFNN class deficit:
    #
    # If RFNN target logit is lower than LFNN target logit, this contributes
    # negatively.
    #
    # If RFNN background logit is higher, this also contributes negatively.
    # ------------------------------------------------------------------------

    target_logit_change = (
        pairs[
            "maximum_target_logit_rfnn"
        ]
        -
        pairs[
            "maximum_target_logit_lfnn"
        ]
    )

    background_logit_change = (
        pairs[
            "maximum_background_logit_rfnn"
        ]
        -
        pairs[
            "maximum_background_logit_lfnn"
        ]
    )

    observed_margin_change = (
        pairs[
            "maximum_target_minus_background_rfnn"
        ]
        -
        pairs[
            "maximum_target_minus_background_lfnn"
        ]
    )

    rows = pd.DataFrame(
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

            "target_logit_change_rfnn_minus_lfnn":
                target_logit_change,

            "background_logit_change_rfnn_minus_lfnn":
                background_logit_change,

            "observed_margin_change_rfnn_minus_lfnn":
                observed_margin_change,

            "target_logit_deficit":
                -target_logit_change,

            "background_logit_penalty":
                background_logit_change,

            "sum_of_margin_components":
                (
                    target_logit_change
                    -
                    background_logit_change
                ),
        }
    )

    rows[
        "decomposition_error"
    ] = (
        rows[
            "observed_margin_change_rfnn_minus_lfnn"
        ]
        -
        rows[
            "sum_of_margin_components"
        ]
    )

    return rows


# ============================================================================
# COMPETITOR SUMMARY
# ============================================================================

def build_competitor_summary(
    pairs,
):

    rows = []

    competitors = [
        (
            "Background",
            "maximum_background_logit_rfnn",
        ),
        (
            "Spinal Canal Stenosis",
            "maximum_scs_logit_rfnn",
        ),
        (
            "Left Subarticular Stenosis",
            "maximum_lss_logit_rfnn",
        ),
        (
            "Right Subarticular Stenosis",
            "maximum_rss_logit_rfnn",
        ),
        (
            "Left Neural Foraminal Narrowing",
            "maximum_lfnn_logit_rfnn",
        ),
    ]

    target = pairs[
        "maximum_rfnn_logit_rfnn"
    ].astype(float)

    for name, column in competitors:

        competitor = pairs[
            column
        ].astype(float)

        margin = (
            target
            -
            competitor
        )

        rows.append(
            {
                "competitor":
                    name,

                "mean_rfnn_minus_competitor":
                    float(
                        margin.mean()
                    ),

                "median_rfnn_minus_competitor":
                    float(
                        margin.median()
                    ),

                "rfnn_wins_fraction":
                    float(
                        (
                            margin
                            > 0
                        ).mean()
                    ),

                "competitor_wins_fraction":
                    float(
                        (
                            margin
                            < 0
                        ).mean()
                    ),

                "tie_fraction":
                    float(
                        (
                            margin
                            == 0
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
    pairs,
):

    rows = []

    for level in sorted(
        pairs["level"].unique()
    ):

        subset = pairs[
            pairs["level"]
            == level
        ]

        target_change = (
            subset[
                "maximum_target_logit_rfnn"
            ]
            -
            subset[
                "maximum_target_logit_lfnn"
            ]
        )

        background_change = (
            subset[
                "maximum_background_logit_rfnn"
            ]
            -
            subset[
                "maximum_background_logit_lfnn"
            ]
        )

        margin_change = (
            subset[
                "maximum_target_minus_background_rfnn"
            ]
            -
            subset[
                "maximum_target_minus_background_lfnn"
            ]
        )

        rows.append(
            {
                "level":
                    level,

                "count":
                    len(subset),

                "target_logit_change":
                    float(
                        target_change.mean()
                    ),

                "background_logit_change":
                    float(
                        background_change.mean()
                    ),

                "margin_change":
                    float(
                        margin_change.mean()
                    ),

                "rfnn_target_logit":
                    float(
                        subset[
                            "maximum_target_logit_rfnn"
                        ].mean()
                    ),

                "lfnn_target_logit":
                    float(
                        subset[
                            "maximum_target_logit_lfnn"
                        ].mean()
                    ),

                "rfnn_background_logit":
                    float(
                        subset[
                            "maximum_background_logit_rfnn"
                        ].mean()
                    ),

                "lfnn_background_logit":
                    float(
                        subset[
                            "maximum_background_logit_lfnn"
                        ].mean()
                    ),

                "rfnn_margin":
                    float(
                        subset[
                            "maximum_target_minus_background_rfnn"
                        ].mean()
                    ),

                "lfnn_margin":
                    float(
                        subset[
                            "maximum_target_minus_background_lfnn"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# PRINT AND SAVE
# ============================================================================

def save_and_print(
    records_df,
    pairs,
    decomposition_df,
    summary_df,
    competitor_df,
    level_df,
):

    records_path = (
        OUTPUT_DIR
        / "part256_logit_records.csv"
    )

    pairs_path = (
        OUTPUT_DIR
        / "part256_paired_logit_records.csv"
    )

    decomposition_path = (
        OUTPUT_DIR
        / "part256_logit_component_decomposition.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part256_logit_decomposition_summary.csv"
    )

    competitor_path = (
        OUTPUT_DIR
        / "part256_rfnn_competitor_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part256_level_logit_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part256_paired_logit_decomposition_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    pairs.to_csv(
        pairs_path,
        index=False,
    )

    decomposition_df.to_csv(
        decomposition_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    competitor_df.to_csv(
        competitor_path,
        index=False,
    )

    level_df.to_csv(
        level_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PART 2.56 LOGIT DECOMPOSITION SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.56 RAW RFNN COMPETITION")
    print("=" * 80)

    print(
        competitor_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.56 LOGIT COMPONENT DECOMPOSITION")
    print("=" * 80)

    print(
        decomposition_df[
            [
                "target_logit_change_rfnn_minus_lfnn",
                "background_logit_change_rfnn_minus_lfnn",
                "observed_margin_change_rfnn_minus_lfnn",
                "target_logit_deficit",
                "background_logit_penalty",
            ]
        ]
        .describe()
        .to_string()
    )

    print()
    print("=" * 80)
    print("PART 2.56 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # Overall interpretation numbers
    # ------------------------------------------------------------------------

    target_change = (
        decomposition_df[
            "target_logit_change_rfnn_minus_lfnn"
        ]
    )

    background_change = (
        decomposition_df[
            "background_logit_change_rfnn_minus_lfnn"
        ]
    )

    margin_change = (
        decomposition_df[
            "observed_margin_change_rfnn_minus_lfnn"
        ]
    )

    json_summary = {
        "paired_observations":
            int(
                len(pairs)
            ),

        "lfnn_records":
            45,

        "rfnn_records":
            45,

        "mean_target_logit_change_rfnn_minus_lfnn":
            float(
                target_change.mean()
            ),

        "mean_background_logit_change_rfnn_minus_lfnn":
            float(
                background_change.mean()
            ),

        "mean_margin_change_rfnn_minus_lfnn":
            float(
                margin_change.mean()
            ),

        "fraction_target_logit_lower_for_rfnn":
            float(
                (
                    target_change
                    < 0
                ).mean()
            ),

        "fraction_background_logit_higher_for_rfnn":
            float(
                (
                    background_change
                    > 0
                ).mean()
            ),

        "fraction_margin_lower_for_rfnn":
            float(
                (
                    margin_change
                    < 0
                ).mean()
            ),

        "summary":
            summary_df.to_dict(
                orient="records"
            ),

        "competitors":
            competitor_df.to_dict(
                orient="records"
            ),

        "levels":
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
            json_summary,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print("PART 2.56 COMPLETE")
    print("=" * 80)

    print(
        "Analysis only."
    )

    print(
        "Part 2.27 checkpoint unchanged."
    )

    print(
        "Part 2.54 unchanged."
    )

    print(
        "Part 2.55 unchanged."
    )

    print(
        "Dashboard unchanged."
    )

    print()
    print("Outputs:")

    print(records_path)
    print(pairs_path)
    print(decomposition_path)
    print(summary_path)
    print(competitor_path)
    print(level_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.56")
    print("PAIRED LFNN / RFNN LOGIT DECOMPOSITION AUDIT")
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

    records = []

    successful_cases = 0

    total_lfnn = 0

    total_rfnn = 0

    print()
    print("=" * 80)
    print("RUNNING EXACT PAIRED LOGIT ANALYSIS")
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

                record = analyze_point(
                    point=point,
                    logits=logits,
                    probabilities=probabilities,
                    geometry=geometry,
                    side="LFNN",
                    study_id=study_id,
                    series_id=series_id,
                )

                records.append(
                    record
                )

                case_lfnn += 1

            for _, point in (
                rfnn_points.iterrows()
            ):

                record = analyze_point(
                    point=point,
                    logits=logits,
                    probabilities=probabilities,
                    geometry=geometry,
                    side="RFNN",
                    study_id=study_id,
                    series_id=series_id,
                )

                records.append(
                    record
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

    if successful_cases != 25:

        raise RuntimeError(
            "Not all validation cases succeeded."
        )

    if total_lfnn != 45:

        raise RuntimeError(
            "Expected 45 LFNN points."
        )

    if total_rfnn != 45:

        raise RuntimeError(
            "Expected 45 RFNN points."
        )

    records_df, pairs = build_pairs(
        records
    )

    decomposition_df = (
        build_component_decomposition(
            pairs
        )
    )

    summary_df = (
        build_decomposition_summary(
            pairs
        )
    )

    competitor_df = (
        build_competitor_summary(
            pairs
        )
    )

    level_df = (
        build_level_summary(
            pairs
        )
    )

    save_and_print(
        records_df,
        pairs,
        decomposition_df,
        summary_df,
        competitor_df,
        level_df,
    )


if __name__ == "__main__":
    main()