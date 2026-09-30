"""
PART 2.64
DECODER CHANNEL SENSITIVITY AUDIT

Purpose
-------
Systematically measure how selected decoder channels affect the RFNN
target logit, background logit, RFNN-vs-background margin, RFNN probability,
and LFNN margin.

This is an ANALYSIS-ONLY experiment.

No:
    - training
    - optimizer
    - checkpoint modification
    - dashboard modification
    - model saving

Protected checkpoint:
    Part 2.27 best macro-disease checkpoint

Validation cohort:
    Exact Part 2.20B validation cohort

Selected channels:
    decoder1: ch2, ch4
    decoder2: ch2, ch3, ch4
    decoder3: ch1, ch2, ch3, ch5
    decoder4: ch33, ch42, ch47

Perturbations:
    1.00 = normal
    0.50 = 50% attenuation
    0.00 = complete ablation

Expected:
    9 paired validation series
    45 paired LFNN/RFNN observations
    12 channels
    3 perturbation levels

Expected perturbed records:
    45 * 12 * 3 = 1620

Baseline is evaluated separately for every paired observation.
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
    / "rsna_part264_decoder_channel_sensitivity_audit"
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

PERTURBATION_LEVELS = [
    1.00,
    0.50,
    0.00,
]

SELECTED_CHANNELS = [
    (
        "decoder1.conv_block.norm2",
        2,
    ),
    (
        "decoder1.conv_block.norm2",
        4,
    ),
    (
        "decoder2.conv_block.norm2",
        2,
    ),
    (
        "decoder2.conv_block.norm2",
        3,
    ),
    (
        "decoder2.conv_block.norm2",
        4,
    ),
    (
        "decoder3.conv_block.norm2",
        1,
    ),
    (
        "decoder3.conv_block.norm2",
        2,
    ),
    (
        "decoder3.conv_block.norm2",
        3,
    ),
    (
        "decoder3.conv_block.norm2",
        5,
    ),
    (
        "decoder4.conv_block.norm2",
        33,
    ),
    (
        "decoder4.conv_block.norm2",
        42,
    ),
    (
        "decoder4.conv_block.norm2",
        47,
    ),
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
# INPUT PREPARATION
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
# OUTPUT SAMPLING
# ============================================================================

def sample_output_at_point(
    output,
    canonical_point,
):

    array = (
        output
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    if array.ndim == 5:

        array = array[0]

    depth = array.shape[1]
    height = array.shape[2]
    width = array.shape[3]

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

    return array[
        :,
        z,
        y,
        x,
    ]


# ============================================================================
# PROBABILITY
# ============================================================================

def probability_from_logits(
    logits,
):

    tensor = torch.tensor(
        logits,
        dtype=torch.float32,
    )

    return (
        torch.softmax(
            tensor,
            dim=0,
        )
        .numpy()
    )


# ============================================================================
# CHANNEL PERTURBATION HOOK
# ============================================================================

class ChannelPerturbationHook:

    def __init__(
        self,
        channel,
        scale,
    ):

        self.channel = int(
            channel
        )

        self.scale = float(
            scale
        )

    def __call__(
        self,
        module,
        inputs,
        output,
    ):

        if not torch.is_tensor(
            output
        ):
            return output

        if output.ndim != 5:

            raise RuntimeError(
                "Expected decoder feature tensor "
                "with 5 dimensions."
            )

        channel_count = output.shape[1]

        if (
            self.channel < 0
            or
            self.channel >= channel_count
        ):

            raise RuntimeError(
                f"Channel {self.channel} invalid "
                f"for feature tensor with "
                f"{channel_count} channels."
            )

        modified = output.clone()

        modified[
            :,
            self.channel,
            ...,
        ] *= self.scale

        return modified


# ============================================================================
# REGISTER HOOK
# ============================================================================

def register_channel_perturbation(
    model,
    feature_name,
    channel,
    scale,
):

    module = get_module_by_name(
        model,
        feature_name,
    )

    if module is None:

        raise RuntimeError(
            f"Could not find feature: "
            f"{feature_name}"
        )

    hook = ChannelPerturbationHook(
        channel,
        scale,
    )

    handle = module.register_forward_hook(
        hook
    )

    return handle


# ============================================================================
# FORWARD
# ============================================================================

def forward_model(
    model,
    image_tensor,
):

    with torch.no_grad():

        return model(
            image_tensor
        )


# ============================================================================
# METRICS AT ONE POINT
# ============================================================================

def calculate_metrics(
    logits,
):

    probabilities = (
        probability_from_logits(
            logits
        )
    )

    rfnn_logit = float(
        logits[RFNN_ID]
    )

    background_logit = float(
        logits[BACKGROUND_ID]
    )

    rfnn_margin = (
        rfnn_logit
        -
        background_logit
    )

    rfnn_probability = float(
        probabilities[RFNN_ID]
    )

    lfnn_margin = float(
        logits[LFNN_ID]
        -
        logits[BACKGROUND_ID]
    )

    return {
        "rfnn_logit":
            rfnn_logit,

        "background_logit":
            background_logit,

        "rfnn_margin":
            rfnn_margin,

        "rfnn_probability":
            rfnn_probability,

        "lfnn_margin":
            lfnn_margin,
    }


# ============================================================================
# ONE PAIRED CASE
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

    image_tensor = prepare_image(
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

    if len(common_levels) != 5:
        return []

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

        # ------------------------------------------------------------
        # BASELINE FOR THIS PAIRED OBSERVATION
        # ------------------------------------------------------------

        baseline_output = (
            forward_model(
                model,
                image_tensor,
            )
        )

        baseline_lfnn_logits = (
            sample_output_at_point(
                baseline_output,
                lfnn_canonical,
            )
        )

        baseline_rfnn_logits = (
            sample_output_at_point(
                baseline_output,
                rfnn_canonical,
            )
        )

        baseline_lfnn_metrics = (
            calculate_metrics(
                baseline_lfnn_logits
            )
        )

        baseline_rfnn_metrics = (
            calculate_metrics(
                baseline_rfnn_logits
            )
        )

        # ------------------------------------------------------------
        # EACH SELECTED CHANNEL
        # ------------------------------------------------------------

        for feature_name, channel in (
            SELECTED_CHANNELS
        ):

            # Validate channel existence once.
            module = get_module_by_name(
                model,
                feature_name,
            )

            if module is None:

                raise RuntimeError(
                    f"Missing feature: "
                    f"{feature_name}"
                )

            # --------------------------------------------------------
            # EACH PERTURBATION LEVEL
            # --------------------------------------------------------

            for scale in (
                PERTURBATION_LEVELS
            ):

                handle = None

                try:

                    handle = (
                        register_channel_perturbation(
                            model,
                            feature_name,
                            channel,
                            scale,
                        )
                    )

                    modified_output = (
                        forward_model(
                            model,
                            image_tensor,
                        )
                    )

                finally:

                    if handle is not None:
                        handle.remove()

                modified_lfnn_logits = (
                    sample_output_at_point(
                        modified_output,
                        lfnn_canonical,
                    )
                )

                modified_rfnn_logits = (
                    sample_output_at_point(
                        modified_output,
                        rfnn_canonical,
                    )
                )

                modified_lfnn_metrics = (
                    calculate_metrics(
                        modified_lfnn_logits
                    )
                )

                modified_rfnn_metrics = (
                    calculate_metrics(
                        modified_rfnn_logits
                    )
                )

                # ----------------------------------------------------
                # RFNN DELTAS
                # ----------------------------------------------------

                delta_rfnn_logit = (
                    modified_rfnn_metrics[
                        "rfnn_logit"
                    ]
                    -
                    baseline_rfnn_metrics[
                        "rfnn_logit"
                    ]
                )

                delta_background = (
                    modified_rfnn_metrics[
                        "background_logit"
                    ]
                    -
                    baseline_rfnn_metrics[
                        "background_logit"
                    ]
                )

                delta_rfnn_margin = (
                    modified_rfnn_metrics[
                        "rfnn_margin"
                    ]
                    -
                    baseline_rfnn_metrics[
                        "rfnn_margin"
                    ]
                )

                delta_rfnn_probability = (
                    modified_rfnn_metrics[
                        "rfnn_probability"
                    ]
                    -
                    baseline_rfnn_metrics[
                        "rfnn_probability"
                    ]
                )

                delta_lfnn_margin = (
                    modified_lfnn_metrics[
                        "lfnn_margin"
                    ]
                    -
                    baseline_lfnn_metrics[
                        "lfnn_margin"
                    ]
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

                        "perturbation_scale":
                            float(scale),

                        "perturbation_percent":
                            float(scale * 100.0),

                        # ----------------------------
                        # RFNN
                        # ----------------------------

                        "baseline_rfnn_logit":
                            baseline_rfnn_metrics[
                                "rfnn_logit"
                            ],

                        "modified_rfnn_logit":
                            modified_rfnn_metrics[
                                "rfnn_logit"
                            ],

                        "delta_rfnn_logit":
                            delta_rfnn_logit,

                        "baseline_rfnn_background":
                            baseline_rfnn_metrics[
                                "background_logit"
                            ],

                        "modified_rfnn_background":
                            modified_rfnn_metrics[
                                "background_logit"
                            ],

                        "delta_rfnn_background":
                            delta_background,

                        "baseline_rfnn_margin":
                            baseline_rfnn_metrics[
                                "rfnn_margin"
                            ],

                        "modified_rfnn_margin":
                            modified_rfnn_metrics[
                                "rfnn_margin"
                            ],

                        "delta_rfnn_margin":
                            delta_rfnn_margin,

                        "baseline_rfnn_probability":
                            baseline_rfnn_metrics[
                                "rfnn_probability"
                            ],

                        "modified_rfnn_probability":
                            modified_rfnn_metrics[
                                "rfnn_probability"
                            ],

                        "delta_rfnn_probability":
                            delta_rfnn_probability,

                        # ----------------------------
                        # LFNN
                        # ----------------------------

                        "baseline_lfnn_margin":
                            baseline_lfnn_metrics[
                                "lfnn_margin"
                            ],

                        "modified_lfnn_margin":
                            modified_lfnn_metrics[
                                "lfnn_margin"
                            ],

                        "delta_lfnn_margin":
                            delta_lfnn_margin,
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

        zero_group = group[
            group[
                "perturbation_scale"
            ]
            == 0.0
        ]

        half_group = group[
            group[
                "perturbation_scale"
            ]
            == 0.5
        ]

        if len(zero_group) == 0:
            continue

        if len(half_group) == 0:
            continue

        rows.append(
            {
                "feature":
                    feature,

                "channel":
                    int(channel),

                "pairs":
                    int(
                        zero_group[
                            [
                                "study_id",
                                "series_id",
                                "level",
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),

                "zero_mean_delta_rfnn_logit":
                    float(
                        zero_group[
                            "delta_rfnn_logit"
                        ].mean()
                    ),

                "zero_mean_delta_rfnn_background":
                    float(
                        zero_group[
                            "delta_rfnn_background"
                        ].mean()
                    ),

                "zero_mean_delta_rfnn_margin":
                    float(
                        zero_group[
                            "delta_rfnn_margin"
                        ].mean()
                    ),

                "zero_mean_delta_rfnn_probability":
                    float(
                        zero_group[
                            "delta_rfnn_probability"
                        ].mean()
                    ),

                "zero_mean_delta_lfnn_margin":
                    float(
                        zero_group[
                            "delta_lfnn_margin"
                        ].mean()
                    ),

                "zero_rfnn_margin_decreased_fraction":
                    float(
                        (
                            zero_group[
                                "delta_rfnn_margin"
                            ]
                            < 0
                        ).mean()
                    ),

                "zero_rfnn_margin_improved_fraction":
                    float(
                        (
                            zero_group[
                                "delta_rfnn_margin"
                            ]
                            > 0
                        ).mean()
                    ),

                "half_mean_delta_rfnn_logit":
                    float(
                        half_group[
                            "delta_rfnn_logit"
                        ].mean()
                    ),

                "half_mean_delta_rfnn_background":
                    float(
                        half_group[
                            "delta_rfnn_background"
                        ].mean()
                    ),

                "half_mean_delta_rfnn_margin":
                    float(
                        half_group[
                            "delta_rfnn_margin"
                        ].mean()
                    ),

                "half_mean_delta_rfnn_probability":
                    float(
                        half_group[
                            "delta_rfnn_probability"
                        ].mean()
                    ),

                "half_mean_delta_lfnn_margin":
                    float(
                        half_group[
                            "delta_lfnn_margin"
                        ].mean()
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    if len(summary) > 0:

        summary[
            "absolute_zero_margin_effect"
        ] = (
            summary[
                "zero_mean_delta_rfnn_margin"
            ].abs()
        )

        summary = summary.sort_values(
            "absolute_zero_margin_effect",
            ascending=False,
        )

    return summary


# ============================================================================
# PERTURBATION SUMMARY
# ============================================================================

def build_perturbation_summary(
    records_df,
):

    rows = []

    grouped = records_df.groupby(
        "perturbation_scale",
        sort=False,
    )

    for (
        scale,
        group,
    ) in grouped:

        rows.append(
            {
                "perturbation_scale":
                    float(scale),

                "perturbation_percent":
                    float(scale * 100.0),

                "records":
                    int(len(group)),

                "mean_delta_rfnn_logit":
                    float(
                        group[
                            "delta_rfnn_logit"
                        ].mean()
                    ),

                "mean_delta_rfnn_background":
                    float(
                        group[
                            "delta_rfnn_background"
                        ].mean()
                    ),

                "mean_delta_rfnn_margin":
                    float(
                        group[
                            "delta_rfnn_margin"
                        ].mean()
                    ),

                "mean_delta_rfnn_probability":
                    float(
                        group[
                            "delta_rfnn_probability"
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
# LEVEL SUMMARY
# ============================================================================

def build_level_summary(
    records_df,
):

    rows = []

    grouped = records_df.groupby(
        [
            "feature",
            "channel",
            "level",
            "perturbation_scale",
        ],
        sort=False,
    )

    for (
        feature,
        channel,
        level,
        scale,
    ), group in grouped:

        rows.append(
            {
                "feature":
                    feature,

                "channel":
                    int(channel),

                "level":
                    level,

                "perturbation_scale":
                    float(scale),

                "records":
                    int(len(group)),

                "mean_delta_rfnn_logit":
                    float(
                        group[
                            "delta_rfnn_logit"
                        ].mean()
                    ),

                "mean_delta_rfnn_background":
                    float(
                        group[
                            "delta_rfnn_background"
                        ].mean()
                    ),

                "mean_delta_rfnn_margin":
                    float(
                        group[
                            "delta_rfnn_margin"
                        ].mean()
                    ),

                "mean_delta_rfnn_probability":
                    float(
                        group[
                            "delta_rfnn_probability"
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
    perturbation_summary,
    level_summary,
):

    records_path = (
        OUTPUT_DIR
        / "part264_decoder_channel_sensitivity_records.csv"
    )

    channel_path = (
        OUTPUT_DIR
        / "part264_decoder_channel_sensitivity_summary.csv"
    )

    perturbation_path = (
        OUTPUT_DIR
        / "part264_perturbation_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part264_decoder_channel_level_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part264_decoder_channel_sensitivity_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    channel_summary.to_csv(
        channel_path,
        index=False,
    )

    perturbation_summary.to_csv(
        perturbation_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    # Top channels according to absolute margin effect.
    top_channels = []

    if len(channel_summary) > 0:

        for _, row in (
            channel_summary
            .head(12)
            .iterrows()
        ):

            top_channels.append(
                {
                    "feature":
                        str(row["feature"]),

                    "channel":
                        int(row["channel"]),

                    "zero_mean_delta_rfnn_margin":
                        float(
                            row[
                                "zero_mean_delta_rfnn_margin"
                            ]
                        ),

                    "zero_mean_delta_rfnn_logit":
                        float(
                            row[
                                "zero_mean_delta_rfnn_logit"
                            ]
                        ),

                    "zero_mean_delta_rfnn_background":
                        float(
                            row[
                                "zero_mean_delta_rfnn_background"
                            ]
                        ),
                }
            )

    summary = {
        "analysis":
            "Part 2.64 decoder channel sensitivity audit",

        "checkpoint":
            str(CHECKPOINT_PATH),

        "validation_cases":
            EXPECTED_CASES,

        "paired_validation_series":
            EXPECTED_PAIRED_SERIES,

        "paired_observations":
            EXPECTED_PAIRS,

        "selected_channels":
            [
                {
                    "feature":
                        feature,

                    "channel":
                        channel,
                }
                for feature, channel
                in SELECTED_CHANNELS
            ],

        "perturbation_levels":
            PERTURBATION_LEVELS,

        "record_count":
            int(len(records_df)),

        "top_channels_by_absolute_zero_margin_effect":
            top_channels,
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

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PART 2.64 CHANNEL SENSITIVITY SUMMARY")
    print("=" * 80)

    print(
        channel_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.64 PERTURBATION SUMMARY")
    print("=" * 80)

    print(
        perturbation_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.64 COMPLETE")
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
    print(
        "Outputs:"
    )

    print(
        records_path
    )

    print(
        channel_path
    )

    print(
        perturbation_path
    )

    print(
        level_path
    )

    print(
        json_path
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.64")
    print("DECODER CHANNEL SENSITIVITY AUDIT")
    print("=" * 80)

    print()
    print(
        "Analysis only."
    )

    print(
        "No training."
    )

    print(
        "No checkpoint modification."
    )

    print(
        "No dashboard modification."
    )

    print()
    print(
        "Selected channels:"
    )

    for feature, channel in (
        SELECTED_CHANNELS
    ):

        print(
            f"  {feature} | channel {channel}"
        )

    print()
    print(
        "Perturbations:"
    )

    print(
        "  1.00 = normal"
    )

    print(
        "  0.50 = 50% attenuation"
    )

    print(
        "  0.00 = complete ablation"
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
    print(
        "RUNNING CHANNEL SENSITIVITY AUDIT"
    )
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
        f"Sensitivity records: "
        f"{len(records_df)}"
    )

    if paired_series != (
        EXPECTED_PAIRED_SERIES
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_PAIRED_SERIES} paired series, "
            f"found {paired_series}."
        )

    expected_records = (
        EXPECTED_PAIRS
        *
        len(SELECTED_CHANNELS)
        *
        len(PERTURBATION_LEVELS)
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
            f"Expected "
            f"{EXPECTED_PAIRS} paired observations, "
            f"found {len(unique_pairs)}."
        )

    print()
    print(
        "Sensitivity record validation: PASSED"
    )

    print(
        f"Paired observations: "
        f"{len(unique_pairs)}"
    )

    print(
        f"Selected channels: "
        f"{len(SELECTED_CHANNELS)}"
    )

    print(
        f"Perturbation levels: "
        f"{len(PERTURBATION_LEVELS)}"
    )

    print(
        f"Expected records: "
        f"{expected_records}"
    )

    channel_summary = (
        build_channel_summary(
            records_df
        )
    )

    perturbation_summary = (
        build_perturbation_summary(
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
        channel_summary,
        perturbation_summary,
        level_summary,
    )


if __name__ == "__main__":
    main()