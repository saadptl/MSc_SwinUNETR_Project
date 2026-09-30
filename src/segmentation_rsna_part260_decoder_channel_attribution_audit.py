"""
PART 2.60
DECODER CHANNEL ATTRIBUTION AUDIT

Purpose
-------
Analyze decoder-channel activation differences between paired LFNN/RFNN
points from the same study, series, and spinal level.

This is an ANALYSIS-ONLY experiment.

It does NOT:
    - train a model
    - modify a checkpoint
    - modify the dashboard
    - create segmentation ground truth

It examines:
    1. Decoder feature channels at LFNN points.
    2. Decoder feature channels at RFNN points.
    3. RFNN-LFNN channel activation differences.
    4. Channels most associated with RFNN suppression.
    5. Correlation between decoder channel differences and final RFNN
       target-vs-background margin.
    6. Consistency across spinal levels.

Important:
The decoder activations are descriptive. They do not by themselves prove
causality.
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
    / "rsna_part260_decoder_channel_attribution_audit"
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

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

EXPECTED_CASES = 25

EXPECTED_PAIRED_SERIES = 9

EXPECTED_PAIRS = 45

EXPECTED_RECORDS_PER_LAYER = 45


# ============================================================================
# DECODER LAYERS
# ============================================================================

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
            "select_validation_series() must return DataFrame."
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
# HOOKS
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
            make_hook(
                feature_name
            )
        )

        handles.append(handle)

    return activations, handles


# ============================================================================
# POINT -> CANONICAL
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
# SAMPLE DECODER FEATURE
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

    # The model input is 64 x 96 x 96.
    # Map the canonical coordinate to this feature's spatial size.

    z = int(
        round(
            float(canonical_point[0])
            * (
                max(
                    depth - 1,
                    1,
                )
                / 63.0
            )
        )
    )

    y = int(
        round(
            float(canonical_point[1])
            * (
                max(
                    height - 1,
                    1,
                )
                / 95.0
            )
        )
    )

    x = int(
        round(
            float(canonical_point[2])
            * (
                max(
                    width - 1,
                    1,
                )
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
# FINAL LOGITS
# ============================================================================

def run_model(
    model,
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
# SAMPLE FINAL OUTPUT
# ============================================================================

def sample_final_output(
    logits,
    probabilities,
    canonical_point,
):

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

    if logits_np.ndim == 5:

        logits_np = logits_np[0]

    if probabilities_np.ndim == 5:

        probabilities_np = probabilities_np[0]

    depth = logits_np.shape[1]
    height = logits_np.shape[2]
    width = logits_np.shape[3]

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
        logits_np[
            :,
            z,
            y,
            x,
        ],
        probabilities_np[
            :,
            z,
            y,
            x,
        ],
    )


# ============================================================================
# ANALYZE ONE CASE
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

    (
        image,
        points,
        geometry,
    ) = part220b.load_case(
        study_id,
        series_id,
        case["points"],
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

    # Clear previous forward activations.
    activations.clear()

    logits, probabilities = run_model(
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

        lfnn_logits, _ = sample_final_output(
            logits,
            probabilities,
            lfnn_canonical,
        )

        rfnn_logits, _ = sample_final_output(
            logits,
            probabilities,
            rfnn_canonical,
        )

        lfnn_margin = float(
            lfnn_logits[
                LFNN_ID
            ]
            -
            lfnn_logits[
                BACKGROUND_ID
            ]
        )

        rfnn_margin = float(
            rfnn_logits[
                RFNN_ID
            ]
            -
            rfnn_logits[
                BACKGROUND_ID
            ]
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

            channel_difference = (
                rfnn_vector
                -
                lfnn_vector
            )

            channel_mean = (
                0.5
                * (
                    rfnn_vector
                    +
                    lfnn_vector
                )
            )

            # ------------------------------------------------------------
            # Rank channels by RFNN-LFNN difference.
            #
            # Positive:
            #     RFNN activation > LFNN activation
            #
            # Negative:
            #     RFNN activation < LFNN activation
            # ------------------------------------------------------------

            for channel_id in range(
                len(channel_difference)
            ):

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
                            int(channel_id),

                        "lfnn_activation":
                            float(
                                lfnn_vector[
                                    channel_id
                                ]
                            ),

                        "rfnn_activation":
                            float(
                                rfnn_vector[
                                    channel_id
                                ]
                            ),

                        "rfnn_minus_lfnn_activation":
                            float(
                                channel_difference[
                                    channel_id
                                ]
                            ),

                        "mean_pair_activation":
                            float(
                                channel_mean[
                                    channel_id
                                ]
                            ),

                        "absolute_activation_difference":
                            float(
                                abs(
                                    channel_difference[
                                        channel_id
                                    ]
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

                        "lfnn_target_background_margin":
                            lfnn_margin,

                        "rfnn_target_background_margin":
                            rfnn_margin,

                        "rfnn_minus_lfnn_margin":
                            (
                                rfnn_margin
                                -
                                lfnn_margin
                            ),
                    }
                )

    return records


# ============================================================================
# CHANNEL SUMMARY
# ============================================================================

def build_channel_summary(
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

        differences = group[
            "rfnn_minus_lfnn_activation"
        ]

        rows.append(
            {
                "feature":
                    feature,

                "channel":
                    int(channel),

                "pairs":
                    int(len(group)),

                "mean_rfnn_activation":
                    float(
                        group[
                            "rfnn_activation"
                        ].mean()
                    ),

                "mean_lfnn_activation":
                    float(
                        group[
                            "lfnn_activation"
                        ].mean()
                    ),

                "mean_rfnn_minus_lfnn":
                    float(
                        differences.mean()
                    ),

                "median_rfnn_minus_lfnn":
                    float(
                        differences.median()
                    ),

                "mean_absolute_difference":
                    float(
                        group[
                            "absolute_activation_difference"
                        ].mean()
                    ),

                "rfnn_higher_fraction":
                    float(
                        (
                            differences
                            > 0
                        ).mean()
                    ),

                "lfnn_higher_fraction":
                    float(
                        (
                            differences
                            < 0
                        ).mean()
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
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# TOP CHANNELS
# ============================================================================

def build_top_channels(
    channel_summary,
):

    outputs = []

    for feature in DECODER_FEATURES:

        subset = channel_summary[
            channel_summary[
                "feature"
            ]
            == feature
        ].copy()

        if len(subset) == 0:
            continue

        most_rfnn = (
            subset
            .sort_values(
                "mean_rfnn_minus_lfnn",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        most_lfnn = (
            subset
            .sort_values(
                "mean_rfnn_minus_lfnn",
                ascending=True,
            )
            .head(10)
            .copy()
        )

        most_absolute = (
            subset
            .sort_values(
                "mean_absolute_difference",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        most_rfnn[
            "ranking"
        ] = "RFNN_higher"

        most_lfnn[
            "ranking"
        ] = "LFNN_higher"

        most_absolute[
            "ranking"
        ] = "largest_absolute_difference"

        outputs.append(
            most_rfnn
        )

        outputs.append(
            most_lfnn
        )

        outputs.append(
            most_absolute
        )

    if not outputs:

        return pd.DataFrame()

    return pd.concat(
        outputs,
        ignore_index=True,
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
                            "series_id"
                        ].nunique()
                    ),

                "mean_activation_difference":
                    float(
                        group[
                            "rfnn_minus_lfnn_activation"
                        ].mean()
                    ),

                "mean_absolute_difference":
                    float(
                        group[
                            "absolute_activation_difference"
                        ].mean()
                    ),

                "rfnn_higher_fraction":
                    float(
                        (
                            group[
                                "rfnn_minus_lfnn_activation"
                            ]
                            > 0
                        ).mean()
                    ),

                "lfnn_higher_fraction":
                    float(
                        (
                            group[
                                "rfnn_minus_lfnn_activation"
                            ]
                            < 0
                        ).mean()
                    ),

                "rfnn_margin":
                    float(
                        group[
                            "rfnn_target_background_margin"
                        ].mean()
                    ),

                "lfnn_margin":
                    float(
                        group[
                            "lfnn_target_background_margin"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def build_correlation_summary(
    records_df,
    channel_summary,
):

    rows = []

    for feature in DECODER_FEATURES:

        feature_records = records_df[
            records_df[
                "feature"
            ]
            == feature
        ]

        for channel in sorted(
            feature_records[
                "channel"
            ].unique()
        ):

            subset = feature_records[
                feature_records[
                    "channel"
                ]
                == channel
            ]

            x = subset[
                "rfnn_minus_lfnn_activation"
            ].to_numpy(
                dtype=float
            )

            y = subset[
                "rfnn_minus_lfnn_margin"
            ].to_numpy(
                dtype=float
            )

            if len(x) < 3:

                correlation = np.nan

            elif (
                np.std(x) == 0
                or
                np.std(y) == 0
            ):

                correlation = np.nan

            else:

                correlation = float(
                    np.corrcoef(
                        x,
                        y,
                    )[0, 1]
                )

            rows.append(
                {
                    "feature":
                        feature,

                    "channel":
                        int(channel),

                    "pairs":
                        len(subset),

                    "activation_margin_correlation":
                        correlation,

                    "mean_activation_difference":
                        float(
                            x.mean()
                        ),

                    "mean_margin_difference":
                        float(
                            y.mean()
                        ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================================
# FEATURE-LEVEL SUMMARY
# ============================================================================

def build_feature_summary(
    channel_summary,
    correlation_df,
):

    rows = []

    for feature in DECODER_FEATURES:

        channels = channel_summary[
            channel_summary[
                "feature"
            ]
            == feature
        ]

        correlations = correlation_df[
            correlation_df[
                "feature"
            ]
            == feature
        ]

        valid_correlations = correlations[
            correlations[
                "activation_margin_correlation"
            ].notna()
        ]

        if len(channels) == 0:
            continue

        rows.append(
            {
                "feature":
                    feature,

                "channels":
                    int(
                        len(channels)
                    ),

                "mean_channel_difference":
                    float(
                        channels[
                            "mean_rfnn_minus_lfnn"
                        ].mean()
                    ),

                "median_channel_difference":
                    float(
                        channels[
                            "mean_rfnn_minus_lfnn"
                        ].median()
                    ),

                "mean_absolute_channel_difference":
                    float(
                        channels[
                            "mean_absolute_difference"
                        ].mean()
                    ),

                "channels_rfnn_higher_fraction":
                    float(
                        (
                            channels[
                                "mean_rfnn_minus_lfnn"
                            ]
                            > 0
                        ).mean()
                    ),

                "channels_lfnn_higher_fraction":
                    float(
                        (
                            channels[
                                "mean_rfnn_minus_lfnn"
                            ]
                            < 0
                        ).mean()
                    ),

                "strongest_rfnn_channel_difference":
                    float(
                        channels[
                            "mean_rfnn_minus_lfnn"
                        ].max()
                    ),

                "strongest_lfnn_channel_difference":
                    float(
                        channels[
                            "mean_rfnn_minus_lfnn"
                        ].min()
                    ),

                "mean_channel_margin_correlation":
                    float(
                        valid_correlations[
                            "activation_margin_correlation"
                        ].mean()
                    )
                    if len(valid_correlations)
                    else np.nan,
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
    level_summary,
    correlation_summary,
    feature_summary,
):

    records_path = (
        OUTPUT_DIR
        / "part260_decoder_channel_records.csv"
    )

    channel_path = (
        OUTPUT_DIR
        / "part260_decoder_channel_summary.csv"
    )

    top_path = (
        OUTPUT_DIR
        / "part260_top_decoder_channels.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part260_decoder_level_summary.csv"
    )

    correlation_path = (
        OUTPUT_DIR
        / "part260_decoder_channel_margin_correlations.csv"
    )

    feature_path = (
        OUTPUT_DIR
        / "part260_decoder_feature_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part260_decoder_channel_attribution_audit_summary.json"
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

    level_summary.to_csv(
        level_path,
        index=False,
    )

    correlation_summary.to_csv(
        correlation_path,
        index=False,
    )

    feature_summary.to_csv(
        feature_path,
        index=False,
    )

    summary = {
        "validation_cases":
            EXPECTED_CASES,

        "paired_series":
            EXPECTED_PAIRED_SERIES,

        "paired_observations":
            EXPECTED_PAIRS,

        "decoder_features":
            DECODER_FEATURES,

        "feature_records":
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
    print("PART 2.60 DECODER FEATURE SUMMARY")
    print("=" * 80)

    print(
        feature_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.60 TOP DECODER CHANNELS")
    print("=" * 80)

    print(
        top_channels.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.60 DECODER LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.60 COMPLETE")
    print("=" * 80)

    print(
        "Analysis only."
    )

    print(
        "Part 2.27 checkpoint unchanged."
    )

    print(
        "Part 2.59 unchanged."
    )

    print(
        "Dashboard unchanged."
    )

    print()
    print("Outputs:")

    print(records_path)
    print(channel_path)
    print(top_path)
    print(level_path)
    print(correlation_path)
    print(feature_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.60")
    print("DECODER CHANNEL ATTRIBUTION AUDIT")
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
        print("RUNNING DECODER CHANNEL AUDIT")
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
        f"Feature records: "
        f"{len(records_df)}"
    )

    expected_records = (
        EXPECTED_PAIRS
        * len(DECODER_FEATURES)
    )

    if paired_series != EXPECTED_PAIRED_SERIES:

        raise RuntimeError(
        f"Expected {EXPECTED_PAIRED_SERIES} "
        f"paired validation series, "
        f"found {paired_series}."
    )

    if len(records_df) == 0:

        raise RuntimeError(
        "No decoder-channel records were generated."
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

    actual_pair_count = len(
    actual_pairs
)

    if actual_pair_count != EXPECTED_PAIRS:

        raise RuntimeError(
        f"Expected {EXPECTED_PAIRS} paired observations, "
        f"found {actual_pair_count}."
    )

    actual_features = (
    records_df[
        "feature"
    ]
    .nunique()
)

    if actual_features != len(
    DECODER_FEATURES
):

        raise RuntimeError(
        f"Expected {len(DECODER_FEATURES)} decoder "
        f"features, found {actual_features}."
    )

    print()
    print(
    "Decoder-channel record validation: PASSED"
)

    print(
    f"Paired observations: "
    f"{actual_pair_count}"
)

    print(
    f"Decoder layers: "
    f"{actual_features}"
)

    print(
    f"Channel records: "
    f"{len(records_df)}"
)

    channel_summary = (
        build_channel_summary(
            records_df
        )
    )

    top_channels = (
        build_top_channels(
            channel_summary
        )
    )

    level_summary = (
        build_level_summary(
            records_df
        )
    )

    correlation_summary = (
        build_correlation_summary(
            records_df,
            channel_summary,
        )
    )

    feature_summary = (
        build_feature_summary(
            channel_summary,
            correlation_summary,
        )
    )

    save_outputs(
        records_df,
        channel_summary,
        top_channels,
        level_summary,
        correlation_summary,
        feature_summary,
    )


if __name__ == "__main__":
    main()