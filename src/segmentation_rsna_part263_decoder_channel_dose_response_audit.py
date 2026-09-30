"""
PART 2.63
DECODER CHANNEL DOSE-RESPONSE AUDIT

Purpose
-------
Test whether progressively attenuating Decoder 1 channel 2 produces a
consistent change in the RFNN output.

Attenuation levels:

    1.00 = normal channel
    0.75 = 75% of original activation
    0.50 = 50%
    0.25 = 25%
    0.00 = complete ablation

Analysis only.

No:
    - training
    - checkpoint modification
    - dashboard modification
    - pseudo-ground-truth creation

Protected checkpoint:
    Part 2.27 best macro-disease checkpoint

Validation cohort:
    Exact Part 2.20B validation cohort

Expected:
    45 LFNN/RFNN paired observations
    5 attenuation levels
    225 records
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
    / "rsna_part263_decoder_channel_dose_response_audit"
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

TARGET_FEATURE = "decoder1.conv_block.norm2"
TARGET_CHANNEL = 2

ATTENUATION_LEVELS = [
    1.00,
    0.75,
    0.50,
    0.25,
    0.00,
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
# OUTPUT AT POINT
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
# CHANNEL ATTENUATION HOOK
# ============================================================================

class ChannelAttenuationHook:

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
        ] = (
            modified[
                :,
                self.channel,
                ...
            ]
            * self.scale
        )

        return modified


# ============================================================================
# REGISTER ATTENUATION
# ============================================================================

def register_attenuation(
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
            f"Feature not found: "
            f"{feature_name}"
        )

    hook_function = (
        ChannelAttenuationHook(
            channel,
            scale,
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

        return model(
            image_tensor
        )


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

        baseline_lfnn_probs = (
            probability_from_logits(
                baseline_lfnn_logits
            )
        )

        baseline_rfnn_probs = (
            probability_from_logits(
                baseline_rfnn_logits
            )
        )

        baseline_lfnn_logit = float(
            baseline_lfnn_logits[
                LFNN_ID
            ]
        )

        baseline_rfnn_logit = float(
            baseline_rfnn_logits[
                RFNN_ID
            ]
        )

        baseline_lfnn_background = float(
            baseline_lfnn_logits[
                BACKGROUND_ID
            ]
        )

        baseline_rfnn_background = float(
            baseline_rfnn_logits[
                BACKGROUND_ID
            ]
        )

        baseline_lfnn_margin = (
            baseline_lfnn_logit
            -
            baseline_lfnn_background
        )

        baseline_rfnn_margin = (
            baseline_rfnn_logit
            -
            baseline_rfnn_background
        )

        # ------------------------------------------------------------
        # ATTENUATION LEVELS
        # ------------------------------------------------------------

        for scale in ATTENUATION_LEVELS:

            handle = None

            try:

                handle = register_attenuation(
                    model,
                    TARGET_FEATURE,
                    TARGET_CHANNEL,
                    scale,
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

            modified_lfnn_probs = (
                probability_from_logits(
                    modified_lfnn_logits
                )
            )

            modified_rfnn_probs = (
                probability_from_logits(
                    modified_rfnn_logits
                )
            )

            modified_lfnn_logit = float(
                modified_lfnn_logits[
                    LFNN_ID
                ]
            )

            modified_rfnn_logit = float(
                modified_rfnn_logits[
                    RFNN_ID
                ]
            )

            modified_lfnn_background = float(
                modified_lfnn_logits[
                    BACKGROUND_ID
                ]
            )

            modified_rfnn_background = float(
                modified_rfnn_logits[
                    BACKGROUND_ID
                ]
            )

            modified_lfnn_margin = (
                modified_lfnn_logit
                -
                modified_lfnn_background
            )

            modified_rfnn_margin = (
                modified_rfnn_logit
                -
                modified_rfnn_background
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
                        TARGET_FEATURE,

                    "channel":
                        TARGET_CHANNEL,

                    "attenuation_scale":
                        scale,

                    "attenuation_percent":
                        scale * 100.0,

                    "baseline_lfnn_logit":
                        baseline_lfnn_logit,

                    "modified_lfnn_logit":
                        modified_lfnn_logit,

                    "delta_lfnn_logit":
                        (
                            modified_lfnn_logit
                            -
                            baseline_lfnn_logit
                        ),

                    "baseline_rfnn_logit":
                        baseline_rfnn_logit,

                    "modified_rfnn_logit":
                        modified_rfnn_logit,

                    "delta_rfnn_logit":
                        (
                            modified_rfnn_logit
                            -
                            baseline_rfnn_logit
                        ),

                    "baseline_lfnn_background":
                        baseline_lfnn_background,

                    "modified_lfnn_background":
                        modified_lfnn_background,

                    "delta_lfnn_background":
                        (
                            modified_lfnn_background
                            -
                            baseline_lfnn_background
                        ),

                    "baseline_rfnn_background":
                        baseline_rfnn_background,

                    "modified_rfnn_background":
                        modified_rfnn_background,

                    "delta_rfnn_background":
                        (
                            modified_rfnn_background
                            -
                            baseline_rfnn_background
                        ),

                    "baseline_lfnn_margin":
                        baseline_lfnn_margin,

                    "modified_lfnn_margin":
                        modified_lfnn_margin,

                    "delta_lfnn_margin":
                        (
                            modified_lfnn_margin
                            -
                            baseline_lfnn_margin
                        ),

                    "baseline_rfnn_margin":
                        baseline_rfnn_margin,

                    "modified_rfnn_margin":
                        modified_rfnn_margin,

                    "delta_rfnn_margin":
                        (
                            modified_rfnn_margin
                            -
                            baseline_rfnn_margin
                        ),

                    "baseline_lfnn_probability":
                        float(
                            baseline_lfnn_probs[
                                LFNN_ID
                            ]
                        ),

                    "modified_lfnn_probability":
                        float(
                            modified_lfnn_probs[
                                LFNN_ID
                            ]
                        ),

                    "delta_lfnn_probability":
                        float(
                            modified_lfnn_probs[
                                LFNN_ID
                            ]
                            -
                            baseline_lfnn_probs[
                                LFNN_ID
                            ]
                        ),

                    "baseline_rfnn_probability":
                        float(
                            baseline_rfnn_probs[
                                RFNN_ID
                            ]
                        ),

                    "modified_rfnn_probability":
                        float(
                            modified_rfnn_probs[
                                RFNN_ID
                            ]
                        ),

                    "delta_rfnn_probability":
                        float(
                            modified_rfnn_probs[
                                RFNN_ID
                            ]
                            -
                            baseline_rfnn_probs[
                                RFNN_ID
                            ]
                        ),
                }
            )

    return records


# ============================================================================
# SUMMARY
# ============================================================================

def build_dose_summary(
    records_df,
):

    rows = []

    grouped = records_df.groupby(
        "attenuation_scale",
        sort=True,
    )

    for scale, group in grouped:

        rows.append(
            {
                "attenuation_scale":
                    float(scale),

                "attenuation_percent":
                    float(scale * 100.0),

                "pairs":
                    int(len(group) / 5),

                "records":
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

                "mean_modified_rfnn_margin":
                    float(
                        group[
                            "modified_rfnn_margin"
                        ].mean()
                    ),

                "mean_modified_rfnn_probability":
                    float(
                        group[
                            "modified_rfnn_probability"
                        ].mean()
                    ),

                "mean_modified_rfnn_background":
                    float(
                        group[
                            "modified_rfnn_background"
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
# MONOTONICITY CHECK
# ============================================================================

def evaluate_monotonicity(
    records_df,
):

    rows = []

    grouped = records_df.groupby(
        [
            "study_id",
            "series_id",
            "level",
        ],
        sort=False,
    )

    for (
        study_id,
        series_id,
        level,
    ), group in grouped:

        group = group.sort_values(
            "attenuation_scale"
        )

        scales = (
            group[
                "attenuation_scale"
            ]
            .to_numpy()
        )

        margins = (
            group[
                "modified_rfnn_margin"
            ]
            .to_numpy()
        )

        probabilities = (
            group[
                "modified_rfnn_probability"
            ]
            .to_numpy()
        )

        # As scale increases, the channel contribution is restored.
        margin_differences = np.diff(
            margins
        )

        probability_differences = np.diff(
            probabilities
        )

        monotonic_margin = (
            np.all(
                margin_differences
                >= -1e-6
            )
            or
            np.all(
                margin_differences
                <= 1e-6
            )
        )

        monotonic_probability = (
            np.all(
                probability_differences
                >= -1e-6
            )
            or
            np.all(
                probability_differences
                <= 1e-6
            )
        )

        rows.append(
            {
                "study_id":
                    study_id,

                "series_id":
                    series_id,

                "level":
                    level,

                "monotonic_rfnn_margin":
                    bool(
                        monotonic_margin
                    ),

                "monotonic_rfnn_probability":
                    bool(
                        monotonic_probability
                    ),

                "minimum_rfnn_margin":
                    float(
                        margins.min()
                    ),

                "maximum_rfnn_margin":
                    float(
                        margins.max()
                    ),

                "minimum_rfnn_probability":
                    float(
                        probabilities.min()
                    ),

                "maximum_rfnn_probability":
                    float(
                        probabilities.max()
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
            "level",
            "attenuation_scale",
        ],
        sort=True,
    )

    for (
        level,
        scale,
    ), group in grouped:

        rows.append(
            {
                "level":
                    level,

                "attenuation_scale":
                    float(scale),

                "pairs":
                    int(len(group) / 1),

                "mean_rfnn_logit":
                    float(
                        group[
                            "modified_rfnn_logit"
                        ].mean()
                    ),

                "mean_rfnn_background":
                    float(
                        group[
                            "modified_rfnn_background"
                        ].mean()
                    ),

                "mean_rfnn_margin":
                    float(
                        group[
                            "modified_rfnn_margin"
                        ].mean()
                    ),

                "mean_rfnn_probability":
                    float(
                        group[
                            "modified_rfnn_probability"
                        ].mean()
                    ),

                "mean_delta_rfnn_margin":
                    float(
                        group[
                            "delta_rfnn_margin"
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
    dose_summary,
    monotonicity,
    level_summary,
):

    records_path = (
        OUTPUT_DIR
        / "part263_decoder_channel_dose_response_records.csv"
    )

    dose_path = (
        OUTPUT_DIR
        / "part263_decoder_channel_dose_response_summary.csv"
    )

    monotonicity_path = (
        OUTPUT_DIR
        / "part263_decoder_channel_dose_monotonicity.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part263_decoder_channel_dose_level_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part263_decoder_channel_dose_response_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    dose_summary.to_csv(
        dose_path,
        index=False,
    )

    monotonicity.to_csv(
        monotonicity_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    summary = {
        "target_feature":
            TARGET_FEATURE,

        "target_channel":
            TARGET_CHANNEL,

        "attenuation_levels":
            ATTENUATION_LEVELS,

        "validation_cases":
            EXPECTED_CASES,

        "paired_validation_series":
            EXPECTED_PAIRED_SERIES,

        "paired_observations":
            EXPECTED_PAIRS,

        "records":
            int(len(records_df)),

        "dose_summary":
            dose_summary.to_dict(
                orient="records"
            ),

        "monotonic_rfnn_margin_fraction":
            float(
                monotonicity[
                    "monotonic_rfnn_margin"
                ].mean()
            ),

        "monotonic_rfnn_probability_fraction":
            float(
                monotonicity[
                    "monotonic_rfnn_probability"
                ].mean()
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
    # Console output
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PART 2.63 DOSE-RESPONSE SUMMARY")
    print("=" * 80)

    print(
        dose_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.63 MONOTONICITY SUMMARY")
    print("=" * 80)

    print(
        "RFNN margin monotonic fraction: "
        f"{monotonicity['monotonic_rfnn_margin'].mean():.6f}"
    )

    print(
        "RFNN probability monotonic fraction: "
        f"{monotonicity['monotonic_rfnn_probability'].mean():.6f}"
    )

    print()
    print("=" * 80)
    print("PART 2.63 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.63 COMPLETE")
    print("=" * 80)

    print("Analysis only.")
    print("No training.")
    print("Part 2.27 checkpoint unchanged.")
    print("Dashboard unchanged.")

    print()
    print("Outputs:")

    print(records_path)
    print(dose_path)
    print(monotonicity_path)
    print(level_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.63")
    print("DECODER CHANNEL DOSE-RESPONSE AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    print()
    print(
        f"Target feature : {TARGET_FEATURE}"
    )

    print(
        f"Target channel : {TARGET_CHANNEL}"
    )

    print(
        "Attenuation levels: "
        + ", ".join(
            f"{value:.2f}"
            for value in ATTENUATION_LEVELS
        )
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
    print("RUNNING DECODER CHANNEL DOSE-RESPONSE")
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
        f"Dose-response records: "
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
            "No dose-response records were generated."
        )

    expected_records = (
        EXPECTED_PAIRS
        * len(ATTENUATION_LEVELS)
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
        "Dose-response record validation: PASSED"
    )

    print(
        f"Paired observations: "
        f"{len(actual_pairs)}"
    )

    print(
        f"Attenuation levels: "
        f"{len(ATTENUATION_LEVELS)}"
    )

    print(
        f"Expected records: "
        f"{expected_records}"
    )

    dose_summary = build_dose_summary(
        records_df
    )

    monotonicity = evaluate_monotonicity(
        records_df
    )

    level_summary = build_level_summary(
        records_df
    )

    save_outputs(
        records_df,
        dose_summary,
        monotonicity,
        level_summary,
    )


if __name__ == "__main__":
    main()