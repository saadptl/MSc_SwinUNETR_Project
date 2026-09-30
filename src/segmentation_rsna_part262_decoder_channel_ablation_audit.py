"""
PART 2.62
DECODER CHANNEL ABLATION AUDIT

Purpose
-------
Controlled inference-only ablation of selected decoder channels.

The experiment compares normal inference against inference where one
decoder channel is temporarily zeroed.

No:
    - training
    - checkpoint modification
    - dashboard modification
    - pseudo-ground-truth creation

Protected checkpoint:
    Part 2.27 best macro-disease checkpoint

Validation cohort:
    Exact Part 2.20B validation cohort

Paired observations:
    45 LFNN/RFNN pairs

The experiment measures:

    - RFNN target logit
    - background logit
    - RFNN-background margin
    - RFNN probability
    - LFNN target logit
    - change caused by channel ablation

IMPORTANT
---------
This is a sensitivity experiment.

A change in output after ablation indicates that the channel participates
in the computation, but does NOT by itself prove causal anatomical meaning.
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
    / "rsna_part262_decoder_channel_ablation_audit"
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

# These are the most informative channels identified by Part 2.61.
#
# We intentionally do NOT ablate every channel.
# This keeps the experiment focused and interpretable.

TARGET_CHANNELS = [
    (
        "decoder4.conv_block.norm2",
        47,
    ),
    (
        "decoder4.conv_block.norm2",
        42,
    ),
    (
        "decoder4.conv_block.norm2",
        33,
    ),
    (
        "decoder3.conv_block.norm2",
        1,
    ),
    (
        "decoder3.conv_block.norm2",
        5,
    ),
    (
        "decoder3.conv_block.norm2",
        3,
    ),
    (
        "decoder2.conv_block.norm2",
        4,
    ),
    (
        "decoder2.conv_block.norm2",
        3,
    ),
    (
        "decoder2.conv_block.norm2",
        2,
    ),
    (
        "decoder1.conv_block.norm2",
        4,
    ),
    (
        "decoder1.conv_block.norm2",
        2,
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
# OUTPUT AT A POINT
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
# POINT CONVERSION
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
# DECODER CHANNEL ABLATION HOOK
# ============================================================================

class ChannelAblationHook:

    def __init__(
        self,
        channel,
    ):

        self.channel = int(
            channel
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
            return output

        if (
            self.channel < 0
            or
            self.channel >= output.shape[1]
        ):
            raise RuntimeError(
                f"Channel {self.channel} is invalid "
                f"for tensor with "
                f"{output.shape[1]} channels."
            )

        modified = output.clone()

        modified[
            :,
            self.channel,
            ...,
        ] = 0.0

        return modified


# ============================================================================
# REGISTER ABLATION
# ============================================================================

def register_ablation(
    model,
    feature_name,
    channel,
):

    module = get_module_by_name(
        model,
        feature_name,
    )

    if module is None:

        raise RuntimeError(
            f"Feature not found: "
            f"{feature_name}"
        )

    hook_function = (
        ChannelAblationHook(
            channel
        )
    )

    handle = module.register_forward_hook(
        hook_function
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

        output = model(
            image_tensor
        )

    return output


# ============================================================================
# ONE CASE
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
        # BASELINE
        # ------------------------------------------------------------

        baseline_output = (
            forward_model(
                model,
                image_tensor,
            )
        )

        lfnn_logits = (
            sample_output_at_point(
                baseline_output,
                lfnn_canonical,
            )
        )

        rfnn_logits = (
            sample_output_at_point(
                baseline_output,
                rfnn_canonical,
            )
        )

        lfnn_probs = (
            probability_from_logits(
                lfnn_logits
            )
        )

        rfnn_probs = (
            probability_from_logits(
                rfnn_logits
            )
        )

        baseline_lfnn_logit = float(
            lfnn_logits[LFNN_ID]
        )

        baseline_rfnn_logit = float(
            rfnn_logits[RFNN_ID]
        )

        baseline_lfnn_bg = float(
            lfnn_logits[BACKGROUND_ID]
        )

        baseline_rfnn_bg = float(
            rfnn_logits[BACKGROUND_ID]
        )

        baseline_lfnn_margin = (
            baseline_lfnn_logit
            -
            baseline_lfnn_bg
        )

        baseline_rfnn_margin = (
            baseline_rfnn_logit
            -
            baseline_rfnn_bg
        )

        # ------------------------------------------------------------
        # ABLATE SELECTED CHANNELS
        # ------------------------------------------------------------

        for (
            feature_name,
            channel,
        ) in TARGET_CHANNELS:

            handle = None

            try:

                handle = register_ablation(
                    model,
                    feature_name,
                    channel,
                )

                ablated_output = (
                    forward_model(
                        model,
                        image_tensor,
                    )
                )

            finally:

                if handle is not None:
                    handle.remove()

            ablated_lfnn_logits = (
                sample_output_at_point(
                    ablated_output,
                    lfnn_canonical,
                )
            )

            ablated_rfnn_logits = (
                sample_output_at_point(
                    ablated_output,
                    rfnn_canonical,
                )
            )

            ablated_lfnn_probs = (
                probability_from_logits(
                    ablated_lfnn_logits
                )
            )

            ablated_rfnn_probs = (
                probability_from_logits(
                    ablated_rfnn_logits
                )
            )

            ablated_lfnn_logit = float(
                ablated_lfnn_logits[
                    LFNN_ID
                ]
            )

            ablated_rfnn_logit = float(
                ablated_rfnn_logits[
                    RFNN_ID
                ]
            )

            ablated_lfnn_bg = float(
                ablated_lfnn_logits[
                    BACKGROUND_ID
                ]
            )

            ablated_rfnn_bg = float(
                ablated_rfnn_logits[
                    BACKGROUND_ID
                ]
            )

            ablated_lfnn_margin = (
                ablated_lfnn_logit
                -
                ablated_lfnn_bg
            )

            ablated_rfnn_margin = (
                ablated_rfnn_logit
                -
                ablated_rfnn_bg
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

                    "baseline_lfnn_logit":
                        baseline_lfnn_logit,

                    "ablated_lfnn_logit":
                        ablated_lfnn_logit,

                    "delta_lfnn_logit":
                        (
                            ablated_lfnn_logit
                            -
                            baseline_lfnn_logit
                        ),

                    "baseline_rfnn_logit":
                        baseline_rfnn_logit,

                    "ablated_rfnn_logit":
                        ablated_rfnn_logit,

                    "delta_rfnn_logit":
                        (
                            ablated_rfnn_logit
                            -
                            baseline_rfnn_logit
                        ),

                    "baseline_lfnn_background":
                        baseline_lfnn_bg,

                    "ablated_lfnn_background":
                        ablated_lfnn_bg,

                    "delta_lfnn_background":
                        (
                            ablated_lfnn_bg
                            -
                            baseline_lfnn_bg
                        ),

                    "baseline_rfnn_background":
                        baseline_rfnn_bg,

                    "ablated_rfnn_background":
                        ablated_rfnn_bg,

                    "delta_rfnn_background":
                        (
                            ablated_rfnn_bg
                            -
                            baseline_rfnn_bg
                        ),

                    "baseline_lfnn_margin":
                        baseline_lfnn_margin,

                    "ablated_lfnn_margin":
                        ablated_lfnn_margin,

                    "delta_lfnn_margin":
                        (
                            ablated_lfnn_margin
                            -
                            baseline_lfnn_margin
                        ),

                    "baseline_rfnn_margin":
                        baseline_rfnn_margin,

                    "ablated_rfnn_margin":
                        ablated_rfnn_margin,

                    "delta_rfnn_margin":
                        (
                            ablated_rfnn_margin
                            -
                            baseline_rfnn_margin
                        ),

                    "baseline_lfnn_probability":
                        float(
                            lfnn_probs[
                                LFNN_ID
                            ]
                        ),

                    "ablated_lfnn_probability":
                        float(
                            ablated_lfnn_probs[
                                LFNN_ID
                            ]
                        ),

                    "delta_lfnn_probability":
                        float(
                            ablated_lfnn_probs[
                                LFNN_ID
                            ]
                            -
                            lfnn_probs[
                                LFNN_ID
                            ]
                        ),

                    "baseline_rfnn_probability":
                        float(
                            rfnn_probs[
                                RFNN_ID
                            ]
                        ),

                    "ablated_rfnn_probability":
                        float(
                            ablated_rfnn_probs[
                                RFNN_ID
                            ]
                        ),

                    "delta_rfnn_probability":
                        float(
                            ablated_rfnn_probs[
                                RFNN_ID
                            ]
                            -
                            rfnn_probs[
                                RFNN_ID
                            ]
                        ),
                }
            )

    return records


# ============================================================================
# SUMMARY
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

        rows.append(
            {
                "feature":
                    feature,

                "channel":
                    int(channel),

                "pairs":
                    int(len(group)),

                "mean_delta_rfnn_logit":
                    float(
                        group[
                            "delta_rfnn_logit"
                        ].mean()
                    ),

                "median_delta_rfnn_logit":
                    float(
                        group[
                            "delta_rfnn_logit"
                        ].median()
                    ),

                "mean_delta_rfnn_margin":
                    float(
                        group[
                            "delta_rfnn_margin"
                        ].mean()
                    ),

                "median_delta_rfnn_margin":
                    float(
                        group[
                            "delta_rfnn_margin"
                        ].median()
                    ),

                "mean_delta_rfnn_probability":
                    float(
                        group[
                            "delta_rfnn_probability"
                        ].mean()
                    ),

                "mean_delta_rfnn_background":
                    float(
                        group[
                            "delta_rfnn_background"
                        ].mean()
                    ),

                "rfnn_margin_decreased_fraction":
                    float(
                        (
                            group[
                                "delta_rfnn_margin"
                            ]
                            < 0
                        ).mean()
                    ),

                "rfnn_margin_increased_fraction":
                    float(
                        (
                            group[
                                "delta_rfnn_margin"
                            ]
                            > 0
                        ).mean()
                    ),

                "mean_delta_lfnn_margin":
                    float(
                        group[
                            "delta_lfnn_margin"
                        ].mean()
                    ),

                "mean_delta_lfnn_logit":
                    float(
                        group[
                            "delta_lfnn_logit"
                        ].mean()
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    return summary


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
        ],
        sort=False,
    )

    for (
        feature,
        channel,
        level,
    ), group in grouped:

        rows.append(
            {
                "feature":
                    feature,

                "channel":
                    int(channel),

                "level":
                    level,

                "pairs":
                    int(len(group)),

                "mean_delta_rfnn_logit":
                    float(
                        group[
                            "delta_rfnn_logit"
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

                "mean_delta_rfnn_background":
                    float(
                        group[
                            "delta_rfnn_background"
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
    channel_summary,
    level_summary,
):

    records_path = (
        OUTPUT_DIR
        / "part262_decoder_channel_ablation_records.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part262_decoder_channel_ablation_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part262_decoder_channel_ablation_level_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part262_decoder_channel_ablation_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    channel_summary.to_csv(
        summary_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    # Identify channels with the largest absolute RFNN margin effect.
    ranked = (
        channel_summary
        .copy()
    )

    ranked[
        "abs_mean_delta_rfnn_margin"
    ] = (
        ranked[
            "mean_delta_rfnn_margin"
        ].abs()
    )

    ranked = ranked.sort_values(
        "abs_mean_delta_rfnn_margin",
        ascending=False,
    )

    summary = {
        "validation_cases":
            EXPECTED_CASES,

        "paired_validation_series":
            EXPECTED_PAIRED_SERIES,

        "paired_observations":
            EXPECTED_PAIRS,

        "target_channels":
            [
                {
                    "feature": feature,
                    "channel": channel,
                }
                for feature, channel
                in TARGET_CHANNELS
            ],

        "records":
            int(len(records_df)),

        "ranked_by_absolute_rfnn_margin_effect":
            ranked.to_dict(
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

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PART 2.62 CHANNEL ABLATION SUMMARY")
    print("=" * 80)

    display_columns = [
        "feature",
        "channel",
        "pairs",
        "mean_delta_rfnn_logit",
        "mean_delta_rfnn_margin",
        "mean_delta_rfnn_probability",
        "mean_delta_rfnn_background",
        "rfnn_margin_decreased_fraction",
        "mean_delta_lfnn_margin",
    ]

    print(
        ranked[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.62 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.62 COMPLETE")
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

    print(records_path)
    print(summary_path)
    print(level_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.62")
    print("DECODER CHANNEL ABLATION AUDIT")
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

    print(
        f"Selected decoder channels: "
        f"{len(TARGET_CHANNELS)}"
    )

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

    all_records = []

    paired_series = 0

    print()
    print("=" * 80)
    print("RUNNING DECODER CHANNEL ABLATION")
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

    records_df = pd.DataFrame(
        all_records
    )

    print()
    print(
        f"Paired validation series: "
        f"{paired_series}/25"
    )

    print(
        f"Ablation records: "
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
            "No ablation records were generated."
        )

    expected_records = (
        EXPECTED_PAIRS
        * len(TARGET_CHANNELS)
    )

    if len(records_df) != expected_records:

        raise RuntimeError(
            f"Expected {expected_records} records, "
            f"found {len(records_df)}."
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
        "Ablation record validation: PASSED"
    )

    print(
        f"Paired observations: "
        f"{len(actual_pairs)}"
    )

    print(
        f"Channels tested: "
        f"{len(TARGET_CHANNELS)}"
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

    level_summary = (
        build_level_summary(
            records_df
        )
    )

    save_outputs(
        records_df,
        channel_summary,
        level_summary,
    )


if __name__ == "__main__":
    main()