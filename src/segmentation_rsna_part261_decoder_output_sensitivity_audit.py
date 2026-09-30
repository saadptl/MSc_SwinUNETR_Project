"""
PART 2.61
DECODER-TO-OUTPUT SENSITIVITY AUDIT

Purpose
-------
Determine whether decoder-channel activations are associated with the
final RFNN prediction.

This is an ANALYSIS-ONLY experiment.

No:
    - training
    - checkpoint modification
    - dashboard modification
    - pseudo-ground-truth creation

The analysis uses the exact Part 2.20B validation cohort and the
protected Part 2.27 checkpoint.

For each decoder channel we measure:

    1. LFNN activation
    2. RFNN activation
    3. RFNN-LFNN activation difference
    4. Final RFNN target logit
    5. RFNN-background logit margin
    6. Correlation between channel activation and RFNN target logit
    7. Correlation between channel activation and RFNN-background margin
    8. Paired correlation using RFNN-LFNN activation difference

The goal is descriptive diagnostic evidence, not causal attribution.
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
    / "rsna_part261_decoder_output_sensitivity_audit"
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
SCS_ID = 1
LFNN_ID = 2
RFNN_ID = 3
LSS_ID = 4
RSS_ID = 5

EXPECTED_CASES = 25
EXPECTED_PAIRED_SERIES = 9
EXPECTED_PAIRS = 45

DECODER_FEATURES = [
    "decoder4.conv_block.norm2",
    "decoder3.conv_block.norm2",
    "decoder2.conv_block.norm2",
    "decoder1.conv_block.norm2",
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

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
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
        f"Model parameters: {parameter_count:,}"
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
# MODULE LOOKUP
# ============================================================================

def get_module_by_name(
    model,
    name,
):

    current = model

    for part in name.split("."):

        if part.isdigit():

            current = current[
                int(part)
            ]

        else:

            if not hasattr(
                current,
                part,
            ):
                return None

            current = getattr(
                current,
                part,
            )

    return current


# ============================================================================
# REGISTER DECODER HOOKS
# ============================================================================

def register_hooks(model):

    activations = {}
    handles = []

    for feature_name in DECODER_FEATURES:

        module = get_module_by_name(
            model,
            feature_name,
        )

        if module is None:

            raise RuntimeError(
                f"Decoder feature not found: "
                f"{feature_name}"
            )

        def make_hook(name):

            def hook(
                module,
                inputs,
                output,
            ):

                if torch.is_tensor(output):

                    activations[name] = (
                        output.detach()
                        .float()
                        .cpu()
                    )

            return hook

        handle = module.register_forward_hook(
            make_hook(feature_name)
        )

        handles.append(handle)

    return activations, handles


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
# MODEL INPUT
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
# FORWARD PASS
# ============================================================================

def run_model(
    model,
    image,
    device,
):

    image_tensor = prepare_image(
        image,
        device,
    )

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1,
        )

    return (
        output,
        probabilities,
    )


# ============================================================================
# SAMPLE FINAL LOGITS
# ============================================================================

def sample_output_at_point(
    output,
    probabilities,
    canonical_point,
):

    logits = (
        output
        .detach()
        .cpu()
        .numpy()
    )

    probs = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    if logits.ndim == 5:
        logits = logits[0]

    if probs.ndim == 5:
        probs = probs[0]

    depth = logits.shape[1]
    height = logits.shape[2]
    width = logits.shape[3]

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

    return (
        logits[:, z, y, x],
        probs[:, z, y, x],
    )


# ============================================================================
# SAMPLE DECODER VECTOR
# ============================================================================

def sample_decoder_vector(
    feature,
    canonical_point,
):

    if feature is None:
        return None

    array = (
        feature
        .detach()
        .cpu()
        .numpy()
    )

    if array.ndim == 5:
        array = array[0]

    if array.ndim != 4:
        return None

    channels = array.shape[0]

    depth = array.shape[1]
    height = array.shape[2]
    width = array.shape[3]

    # Model grid:
    # Z = 64
    # Y = 96
    # X = 96

    z = int(
        round(
            float(canonical_point[0])
            * (
                max(depth - 1, 1)
                / 63.0
            )
        )
    )

    y = int(
        round(
            float(canonical_point[1])
            * (
                max(height - 1, 1)
                / 95.0
            )
        )
    )

    x = int(
        round(
            float(canonical_point[2])
            * (
                max(width - 1, 1)
                / 95.0
            )
        )
    )

    z = int(
        np.clip(
            z,
            0,
            depth - 1,
        )
    )

    y = int(
        np.clip(
            y,
            0,
            height - 1,
        )
    )

    x = int(
        np.clip(
            x,
            0,
            width - 1,
        )
    )

    vector = array[
        :,
        z,
        y,
        x,
    ]

    return vector.astype(
        np.float64
    )


# ============================================================================
# ONE CASE
# ============================================================================

def analyze_case(
    model,
    device,
    case,
    activations,
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

    activations.clear()

    output, probabilities = run_model(
        model,
        image,
        device,
    )

    records = []

    common_levels = sorted(
        set(
            lfnn_points["level"]
        )
        &
        set(
            rfnn_points["level"]
        )
    )

    for level in common_levels:

        lfnn_rows = lfnn_points[
            lfnn_points["level"]
            == level
        ]

        rfnn_rows = rfnn_points[
            rfnn_points["level"]
            == level
        ]

        if len(lfnn_rows) != 1:
            continue

        if len(rfnn_rows) != 1:
            continue

        lfnn_point = lfnn_rows.iloc[0]
        rfnn_point = rfnn_rows.iloc[0]

        lfnn_canonical = get_canonical_point(
            lfnn_point,
            geometry,
        )

        rfnn_canonical = get_canonical_point(
            rfnn_point,
            geometry,
        )

        lfnn_logits, lfnn_probs = (
            sample_output_at_point(
                output,
                probabilities,
                lfnn_canonical,
            )
        )

        rfnn_logits, rfnn_probs = (
            sample_output_at_point(
                output,
                probabilities,
                rfnn_canonical,
            )
        )

        lfnn_margin = float(
            lfnn_logits[LFNN_ID]
            -
            lfnn_logits[BACKGROUND_ID]
        )

        rfnn_margin = float(
            rfnn_logits[RFNN_ID]
            -
            rfnn_logits[BACKGROUND_ID]
        )

        for feature_name in DECODER_FEATURES:

            feature = activations.get(
                feature_name
            )

            lfnn_vector = (
                sample_decoder_vector(
                    feature,
                    lfnn_canonical,
                )
            )

            rfnn_vector = (
                sample_decoder_vector(
                    feature,
                    rfnn_canonical,
                )
            )

            if (
                lfnn_vector is None
                or
                rfnn_vector is None
            ):
                continue

            channel_count = len(
                lfnn_vector
            )

            for channel in range(
                channel_count
            ):

                lfnn_activation = float(
                    lfnn_vector[channel]
                )

                rfnn_activation = float(
                    rfnn_vector[channel]
                )

                activation_difference = (
                    rfnn_activation
                    -
                    lfnn_activation
                )

                records.append(
                    {
                        "study_id":
                            study_id,

                        "series_id":
                            series_id,

                        "level":
                            str(level),

                        "feature":
                            feature_name,

                        "channel":
                            int(channel),

                        "lfnn_activation":
                            lfnn_activation,

                        "rfnn_activation":
                            rfnn_activation,

                        "rfnn_minus_lfnn_activation":
                            activation_difference,

                        "lfnn_rfnn_activation_ratio":
                            (
                                rfnn_activation
                                /
                                (
                                    abs(
                                        lfnn_activation
                                    )
                                    + 1e-8
                                )
                            ),

                        "lfnn_target_logit":
                            float(
                                lfnn_logits[
                                    LFNN_ID
                                ]
                            ),

                        "rfnn_target_logit":
                            float(
                                rfnn_logits[
                                    RFNN_ID
                                ]
                            ),

                        "lfnn_background_logit":
                            float(
                                lfnn_logits[
                                    BACKGROUND_ID
                                ]
                            ),

                        "rfnn_background_logit":
                            float(
                                rfnn_logits[
                                    BACKGROUND_ID
                                ]
                            ),

                        "lfnn_target_background_margin":
                            lfnn_margin,

                        "rfnn_target_background_margin":
                            rfnn_margin,

                        "rfnn_minus_lfnn_target_logit":
                            float(
                                rfnn_logits[
                                    RFNN_ID
                                ]
                                -
                                lfnn_logits[
                                    LFNN_ID
                                ]
                            ),

                        "rfnn_minus_lfnn_margin":
                            float(
                                rfnn_margin
                                -
                                lfnn_margin
                            ),

                        "lfnn_target_probability":
                            float(
                                lfnn_probs[
                                    LFNN_ID
                                ]
                            ),

                        "rfnn_target_probability":
                            float(
                                rfnn_probs[
                                    RFNN_ID
                                ]
                            ),
                    }
                )

    return records


# ============================================================================
# SAFE CORRELATION
# ============================================================================

def safe_corr(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0:
        return np.nan

    if np.std(y) == 0:
        return np.nan

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


# ============================================================================
# CHANNEL SENSITIVITY SUMMARY
# ============================================================================

def build_channel_sensitivity_summary(
    records_df,
):

    rows = []

    grouped = records_df.groupby(
        [
            "feature",
            "channel",
        ],
        sort=False,
    )

    for (
        feature,
        channel,
    ), group in grouped:

        activation_difference = (
            group[
                "rfnn_minus_lfnn_activation"
            ]
        )

        rfnn_activation = (
            group[
                "rfnn_activation"
            ]
        )

        lfnn_activation = (
            group[
                "lfnn_activation"
            ]
        )

        rfnn_logit = (
            group[
                "rfnn_target_logit"
            ]
        )

        lfnn_logit = (
            group[
                "lfnn_target_logit"
            ]
        )

        rfnn_margin = (
            group[
                "rfnn_target_background_margin"
            ]
        )

        lfnn_margin = (
            group[
                "lfnn_target_background_margin"
            ]
        )

        margin_difference = (
            group[
                "rfnn_minus_lfnn_margin"
            ]
        )

        rows.append(
            {
                "feature":
                    feature,

                "channel":
                    int(channel),

                "pairs":
                    int(len(group)),

                "mean_lfnn_activation":
                    float(
                        lfnn_activation.mean()
                    ),

                "mean_rfnn_activation":
                    float(
                        rfnn_activation.mean()
                    ),

                "mean_rfnn_minus_lfnn_activation":
                    float(
                        activation_difference.mean()
                    ),

                "median_rfnn_minus_lfnn_activation":
                    float(
                        activation_difference.median()
                    ),

                "rfnn_higher_fraction":
                    float(
                        (
                            activation_difference
                            > 0
                        ).mean()
                    ),

                "lfnn_higher_fraction":
                    float(
                        (
                            activation_difference
                            < 0
                        ).mean()
                    ),

                "corr_rfnn_activation_vs_rfnn_logit":
                    safe_corr(
                        rfnn_activation,
                        rfnn_logit,
                    ),

                "corr_rfnn_activation_vs_rfnn_margin":
                    safe_corr(
                        rfnn_activation,
                        rfnn_margin,
                    ),

                "corr_lfnn_activation_vs_lfnn_logit":
                    safe_corr(
                        lfnn_activation,
                        lfnn_logit,
                    ),

                "corr_lfnn_activation_vs_lfnn_margin":
                    safe_corr(
                        lfnn_activation,
                        lfnn_margin,
                    ),

                "corr_activation_difference_vs_margin_difference":
                    safe_corr(
                        activation_difference,
                        margin_difference,
                    ),

                "corr_activation_difference_vs_target_logit_difference":
                    safe_corr(
                        activation_difference,
                        group[
                            "rfnn_minus_lfnn_target_logit"
                        ],
                    ),

                "mean_rfnn_margin":
                    float(
                        rfnn_margin.mean()
                    ),

                "mean_lfnn_margin":
                    float(
                        lfnn_margin.mean()
                    ),

                "mean_margin_difference":
                    float(
                        margin_difference.mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# TOP SENSITIVITY CHANNELS
# ============================================================================

def build_top_sensitivity_channels(
    summary,
):

    outputs = []

    for feature in DECODER_FEATURES:

        subset = summary[
            summary["feature"]
            == feature
        ].copy()

        if len(subset) == 0:
            continue

        # Strongest positive relationship with RFNN logit.
        rfnn_logit = (
            subset
            .sort_values(
                "corr_rfnn_activation_vs_rfnn_logit",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        rfnn_logit[
            "ranking"
        ] = "RFNN_logit_positive_correlation"

        # Strongest positive relationship with RFNN margin.
        rfnn_margin = (
            subset
            .sort_values(
                "corr_rfnn_activation_vs_rfnn_margin",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        rfnn_margin[
            "ranking"
        ] = "RFNN_margin_positive_correlation"

        # Strongest negative relationship with RFNN margin.
        rfnn_margin_negative = (
            subset
            .sort_values(
                "corr_rfnn_activation_vs_rfnn_margin",
                ascending=True,
            )
            .head(10)
            .copy()
        )

        rfnn_margin_negative[
            "ranking"
        ] = "RFNN_margin_negative_correlation"

        # Strongest paired activation-to-margin relationship.
        paired = (
            subset
            .sort_values(
                "corr_activation_difference_vs_margin_difference",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        paired[
            "ranking"
        ] = "paired_activation_margin_positive"

        outputs.extend(
            [
                rfnn_logit,
                rfnn_margin,
                rfnn_margin_negative,
                paired,
            ]
        )

    if not outputs:

        return pd.DataFrame()

    return pd.concat(
        outputs,
        ignore_index=True,
    )


# ============================================================================
# FEATURE-LEVEL SUMMARY
# ============================================================================

def build_feature_summary(
    channel_summary,
):

    rows = []

    for feature in DECODER_FEATURES:

        subset = channel_summary[
            channel_summary[
                "feature"
            ]
            == feature
        ].copy()

        if len(subset) == 0:
            continue

        valid_rfnn_logit = subset[
            "corr_rfnn_activation_vs_rfnn_logit"
        ].dropna()

        valid_rfnn_margin = subset[
            "corr_rfnn_activation_vs_rfnn_margin"
        ].dropna()

        valid_paired = subset[
            "corr_activation_difference_vs_margin_difference"
        ].dropna()

        rows.append(
            {
                "feature":
                    feature,

                "channels":
                    int(len(subset)),

                "mean_activation_difference":
                    float(
                        subset[
                            "mean_rfnn_minus_lfnn_activation"
                        ].mean()
                    ),

                "rfnn_higher_channel_fraction":
                    float(
                        (
                            subset[
                                "mean_rfnn_minus_lfnn_activation"
                            ]
                            > 0
                        ).mean()
                    ),

                "lfnn_higher_channel_fraction":
                    float(
                        (
                            subset[
                                "mean_rfnn_minus_lfnn_activation"
                            ]
                            < 0
                        ).mean()
                    ),

                "mean_rfnn_logit_correlation":
                    float(
                        valid_rfnn_logit.mean()
                    )
                    if len(valid_rfnn_logit)
                    else np.nan,

                "max_rfnn_logit_correlation":
                    float(
                        valid_rfnn_logit.max()
                    )
                    if len(valid_rfnn_logit)
                    else np.nan,

                "min_rfnn_logit_correlation":
                    float(
                        valid_rfnn_logit.min()
                    )
                    if len(valid_rfnn_logit)
                    else np.nan,

                "mean_rfnn_margin_correlation":
                    float(
                        valid_rfnn_margin.mean()
                    )
                    if len(valid_rfnn_margin)
                    else np.nan,

                "max_rfnn_margin_correlation":
                    float(
                        valid_rfnn_margin.max()
                    )
                    if len(valid_rfnn_margin)
                    else np.nan,

                "min_rfnn_margin_correlation":
                    float(
                        valid_rfnn_margin.min()
                    )
                    if len(valid_rfnn_margin)
                    else np.nan,

                "mean_paired_activation_margin_correlation":
                    float(
                        valid_paired.mean()
                    )
                    if len(valid_paired)
                    else np.nan,

                "max_paired_activation_margin_correlation":
                    float(
                        valid_paired.max()
                    )
                    if len(valid_paired)
                    else np.nan,

                "min_paired_activation_margin_correlation":
                    float(
                        valid_paired.min()
                    )
                    if len(valid_paired)
                    else np.nan,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# LEVEL SUMMARY
# ============================================================================

def build_level_summary(
    records_df,
):

    rows = []

    grouped = records_df.groupby(
        [
            "feature",
            "level",
        ],
        sort=False,
    )

    for (
        feature,
        level,
    ), group in grouped:

        rows.append(
            {
                "feature":
                    feature,

                "level":
                    level,

                "pairs":
                    int(
                        group[
                            [
                                "study_id",
                                "series_id",
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),

                "mean_activation_difference":
                    float(
                        group[
                            "rfnn_minus_lfnn_activation"
                        ].mean()
                    ),

                "mean_rfnn_margin":
                    float(
                        group[
                            "rfnn_target_background_margin"
                        ].mean()
                    ),

                "mean_lfnn_margin":
                    float(
                        group[
                            "lfnn_target_background_margin"
                        ].mean()
                    ),

                "mean_margin_difference":
                    float(
                        group[
                            "rfnn_minus_lfnn_margin"
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
    channel_summary,
    top_channels,
    feature_summary,
    level_summary,
):

    records_path = (
        OUTPUT_DIR
        / "part261_decoder_output_records.csv"
    )

    channel_path = (
        OUTPUT_DIR
        / "part261_decoder_channel_sensitivity_summary.csv"
    )

    top_path = (
        OUTPUT_DIR
        / "part261_top_decoder_output_sensitive_channels.csv"
    )

    feature_path = (
        OUTPUT_DIR
        / "part261_decoder_output_feature_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part261_decoder_output_level_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part261_decoder_output_sensitivity_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    channel_summary.to_csv(
        channel_path,
        index=False,
    )

    top_channels.to_csv(
        top_path,
        index=False,
    )

    feature_summary.to_csv(
        feature_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    summary = {
        "validation_cases":
            EXPECTED_CASES,

        "paired_validation_series":
            EXPECTED_PAIRED_SERIES,

        "paired_observations":
            EXPECTED_PAIRS,

        "decoder_features":
            DECODER_FEATURES,

        "records":
            int(
                len(records_df)
            ),

        "feature_summary":
            feature_summary.to_dict(
                orient="records"
            ),

        "top_channels":
            top_channels.to_dict(
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
    print("PART 2.61 DECODER OUTPUT FEATURE SUMMARY")
    print("=" * 80)

    print(
        feature_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.61 TOP OUTPUT-SENSITIVE CHANNELS")
    print("=" * 80)

    print(
        top_channels.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.61 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.61 COMPLETE")
    print("=" * 80)

    print("Analysis only.")
    print("Part 2.27 checkpoint unchanged.")
    print("Part 2.60 unchanged.")
    print("Dashboard unchanged.")

    print()
    print("Outputs:")

    print(records_path)
    print(channel_path)
    print(top_path)
    print(feature_path)
    print(level_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.61")
    print("DECODER-TO-OUTPUT SENSITIVITY AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    manifest = part220b.load_manifest()

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

    activations, handles = register_hooks(
        model
    )

    all_records = []

    paired_series = 0

    try:

        print()
        print("=" * 80)
        print("RUNNING DECODER-TO-OUTPUT SENSITIVITY")
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

                case_records = analyze_case(
                    model,
                    device,
                    case,
                    activations,
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
                        f"paired levels=5"
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

    finally:

        for handle in handles:
            handle.remove()

    records_df = pd.DataFrame(
        all_records
    )

    print()
    print(
        f"Paired validation series: "
        f"{paired_series}/25"
    )

    print(
        f"Output-sensitivity records: "
        f"{len(records_df)}"
    )

    if paired_series != EXPECTED_PAIRED_SERIES:

        raise RuntimeError(
            f"Expected {EXPECTED_PAIRED_SERIES} "
            f"paired validation series, "
            f"found {paired_series}."
        )

    if len(records_df) == 0:

        raise RuntimeError(
            "No decoder output-sensitivity records generated."
        )

    actual_pairs = (
        records_df[
            [
                "study_id",
                "series_id",
                "level",
            ]
        ]
        .drop_duplicates()
    )

    if len(actual_pairs) != EXPECTED_PAIRS:

        raise RuntimeError(
            f"Expected {EXPECTED_PAIRS} paired observations, "
            f"found {len(actual_pairs)}."
        )

    print()
    print(
        "Output-sensitivity record validation: PASSED"
    )

    print(
        f"Paired observations: "
        f"{len(actual_pairs)}"
    )

    print(
        f"Decoder layers: "
        f"{records_df['feature'].nunique()}"
    )

    channel_summary = (
        build_channel_sensitivity_summary(
            records_df
        )
    )

    top_channels = (
        build_top_sensitivity_channels(
            channel_summary
        )
    )

    feature_summary = (
        build_feature_summary(
            channel_summary
        )
    )

    level_summary = (
        build_level_summary(
            records_df
        )
    )

    save_outputs(
        records_df,
        channel_summary,
        top_channels,
        feature_summary,
        level_summary,
    )


if __name__ == "__main__":
    main()