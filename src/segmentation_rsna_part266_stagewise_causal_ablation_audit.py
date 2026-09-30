"""
PART 2.66
STAGE-WISE CAUSAL ABLATION AUDIT

Purpose
-------
Test whether perturbing complete intermediate network stages changes the
final RFNN prediction.

Stages:
    encoder1
    encoder2
    encoder3
    encoder4
    encoder10
    decoder4.conv_block.norm2
    decoder3.conv_block.norm2
    decoder2.conv_block.norm2
    decoder1.conv_block.norm2

Perturbations:
    1.00 = normal
    0.50 = 50% attenuation
    0.00 = complete ablation

For each paired LFNN/RFNN observation we measure:

    RFNN target logit
    Background logit
    RFNN-background margin
    RFNN probability
    LFNN-background margin

Analysis only.

NO:
    training
    optimizer
    checkpoint modification
    dashboard modification

Protected checkpoint:
    Part 2.27 best macro-disease checkpoint

Expected:
    9 paired validation series
    45 paired observations
    9 stages
    3 perturbation levels

Expected records:
    45 * 9 * 3 = 1215
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
    / "rsna_part266_stagewise_causal_ablation_audit"
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

PERTURBATION_LEVELS = [
    1.00,
    0.50,
    0.00,
]

STAGES = [
    "encoder1",
    "encoder2",
    "encoder3",
    "encoder4",
    "encoder10",
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
# INPUT PREPARATION
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
# OUTPUT SAMPLING
# ============================================================================

def sample_output_at_point(
    output,
    canonical_point,
):

    tensor = (
        output
        .detach()
        .float()
    )

    if tensor.ndim != 5:

        raise RuntimeError(
            "Expected output shape "
            "(B,C,D,H,W)."
        )

    _, _, depth, height, width = (
        tensor.shape
    )

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
        tensor[
            0,
            :,
            z,
            y,
            x,
        ]
        .cpu()
        .numpy()
    )


# ============================================================================
# OUTPUT METRICS
# ============================================================================

def calculate_metrics(
    logits,
):

    logits_tensor = torch.tensor(
        logits,
        dtype=torch.float32,
    )

    probabilities = (
        torch.softmax(
            logits_tensor,
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

        "rfnn_margin":
            rfnn_logit - background_logit,

        "lfnn_margin":
            lfnn_logit - background_logit,

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
    }


# ============================================================================
# STAGE PERTURBATION HOOK
# ============================================================================

class StagePerturbationHook:

    def __init__(
        self,
        scale,
    ):

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

            raise RuntimeError(
                "Stage output is not a tensor."
            )

        if output.ndim < 3:

            raise RuntimeError(
                "Unexpected stage output dimensions: "
                f"{tuple(output.shape)}"
            )

        return (
            output
            * self.scale
        )


# ============================================================================
# REGISTER STAGE PERTURBATION
# ============================================================================

def register_stage_perturbation(
    model,
    stage_name,
    scale,
):

    module = get_module_by_name(
        model,
        stage_name,
    )

    if module is None:

        raise RuntimeError(
            f"Could not find stage: "
            f"{stage_name}"
        )

    hook = StagePerturbationHook(
        scale
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

    records = []

    # ========================================================================
    # BASELINE FOR ALL LEVELS
    # ========================================================================

    baseline_output = (
        forward_model(
            model,
            image_tensor,
        )
    )

    baseline_metrics = {}

    for level in common_levels:

        lfnn_row = lfnn_points[
            lfnn_points["level"]
            == level
        ]

        rfnn_row = rfnn_points[
            rfnn_points["level"]
            == level
        ]

        if len(lfnn_row) != 1:
            continue

        if len(rfnn_row) != 1:
            continue

        lfnn_point = lfnn_row.iloc[0]
        rfnn_point = rfnn_row.iloc[0]

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

        baseline_metrics[
            str(level)
        ] = {
            "lfnn_canonical":
                lfnn_canonical,

            "rfnn_canonical":
                rfnn_canonical,

            "lfnn":
                calculate_metrics(
                    lfnn_logits
                ),

            "rfnn":
                calculate_metrics(
                    rfnn_logits
                ),
        }

    # ========================================================================
    # STAGE-BY-STAGE PERTURBATION
    # ========================================================================

    for stage_name in STAGES:

        # ------------------------------------------------------------
        # Validate stage
        # ------------------------------------------------------------

        module = get_module_by_name(
            model,
            stage_name,
        )

        if module is None:

            raise RuntimeError(
                f"Stage not found: "
                f"{stage_name}"
            )

        for scale in PERTURBATION_LEVELS:

            handle = None

            try:

                handle = (
                    register_stage_perturbation(
                        model,
                        stage_name,
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

            # --------------------------------------------------------
            # Compare every paired level
            # --------------------------------------------------------

            for level in common_levels:

                if str(level) not in (
                    baseline_metrics
                ):
                    continue

                base = baseline_metrics[
                    str(level)
                ]

                lfnn_logits = (
                    sample_output_at_point(
                        modified_output,
                        base[
                            "lfnn_canonical"
                        ],
                    )
                )

                rfnn_logits = (
                    sample_output_at_point(
                        modified_output,
                        base[
                            "rfnn_canonical"
                        ],
                    )
                )

                modified_lfnn = (
                    calculate_metrics(
                        lfnn_logits
                    )
                )

                modified_rfnn = (
                    calculate_metrics(
                        rfnn_logits
                    )
                )

                # ----------------------------------------------------
                # RFNN changes
                # ----------------------------------------------------

                delta_rfnn_logit = (
                    modified_rfnn[
                        "rfnn_logit"
                    ]
                    -
                    base["rfnn"][
                        "rfnn_logit"
                    ]
                )

                delta_background = (
                    modified_rfnn[
                        "background_logit"
                    ]
                    -
                    base["rfnn"][
                        "background_logit"
                    ]
                )

                delta_margin = (
                    modified_rfnn[
                        "rfnn_margin"
                    ]
                    -
                    base["rfnn"][
                        "rfnn_margin"
                    ]
                )

                delta_probability = (
                    modified_rfnn[
                        "rfnn_probability"
                    ]
                    -
                    base["rfnn"][
                        "rfnn_probability"
                    ]
                )

                delta_lfnn_margin = (
                    modified_lfnn[
                        "lfnn_margin"
                    ]
                    -
                    base["lfnn"][
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

                        "stage":
                            stage_name,

                        "perturbation_scale":
                            float(scale),

                        "perturbation_percent":
                            float(
                                scale * 100.0
                            ),

                        "baseline_rfnn_logit":
                            base["rfnn"][
                                "rfnn_logit"
                            ],

                        "modified_rfnn_logit":
                            modified_rfnn[
                                "rfnn_logit"
                            ],

                        "delta_rfnn_logit":
                            delta_rfnn_logit,

                        "baseline_background_logit":
                            base["rfnn"][
                                "background_logit"
                            ],

                        "modified_background_logit":
                            modified_rfnn[
                                "background_logit"
                            ],

                        "delta_background_logit":
                            delta_background,

                        "baseline_rfnn_margin":
                            base["rfnn"][
                                "rfnn_margin"
                            ],

                        "modified_rfnn_margin":
                            modified_rfnn[
                                "rfnn_margin"
                            ],

                        "delta_rfnn_margin":
                            delta_margin,

                        "baseline_rfnn_probability":
                            base["rfnn"][
                                "rfnn_probability"
                            ],

                        "modified_rfnn_probability":
                            modified_rfnn[
                                "rfnn_probability"
                            ],

                        "delta_rfnn_probability":
                            delta_probability,

                        "baseline_lfnn_margin":
                            base["lfnn"][
                                "lfnn_margin"
                            ],

                        "modified_lfnn_margin":
                            modified_lfnn[
                                "lfnn_margin"
                            ],

                        "delta_lfnn_margin":
                            delta_lfnn_margin,
                    }
                )

    return records


# ============================================================================
# STAGE SUMMARY
# ============================================================================

def build_stage_summary(
    records_df,
):

    rows = []

    grouped = records_df.groupby(
        "stage",
        sort=False,
    )

    for stage, group in grouped:

        zero = group[
            group[
                "perturbation_scale"
            ]
            == 0.0
        ]

        half = group[
            group[
                "perturbation_scale"
            ]
            == 0.5
        ]

        rows.append(
            {
                "stage":
                    stage,

                "records":
                    int(len(group)),

                "zero_mean_delta_rfnn_logit":
                    float(
                        zero[
                            "delta_rfnn_logit"
                        ].mean()
                    ),

                "zero_mean_delta_background_logit":
                    float(
                        zero[
                            "delta_background_logit"
                        ].mean()
                    ),

                "zero_mean_delta_rfnn_margin":
                    float(
                        zero[
                            "delta_rfnn_margin"
                        ].mean()
                    ),

                "zero_mean_delta_rfnn_probability":
                    float(
                        zero[
                            "delta_rfnn_probability"
                        ].mean()
                    ),

                "zero_mean_delta_lfnn_margin":
                    float(
                        zero[
                            "delta_lfnn_margin"
                        ].mean()
                    ),

                "zero_margin_improved_fraction":
                    float(
                        (
                            zero[
                                "delta_rfnn_margin"
                            ]
                            > 0
                        ).mean()
                    ),

                "zero_margin_decreased_fraction":
                    float(
                        (
                            zero[
                                "delta_rfnn_margin"
                            ]
                            < 0
                        ).mean()
                    ),

                "half_mean_delta_rfnn_logit":
                    float(
                        half[
                            "delta_rfnn_logit"
                        ].mean()
                    ),

                "half_mean_delta_background_logit":
                    float(
                        half[
                            "delta_background_logit"
                        ].mean()
                    ),

                "half_mean_delta_rfnn_margin":
                    float(
                        half[
                            "delta_rfnn_margin"
                        ].mean()
                    ),

                "half_mean_delta_rfnn_probability":
                    float(
                        half[
                            "delta_rfnn_probability"
                        ].mean()
                    ),

                "half_mean_delta_lfnn_margin":
                    float(
                        half[
                            "delta_lfnn_margin"
                        ].mean()
                    ),

                "absolute_zero_margin_effect":
                    float(
                        abs(
                            zero[
                                "delta_rfnn_margin"
                            ].mean()
                        )
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    stage_order = {
        name: index + 1
        for index, name
        in enumerate(STAGES)
    }

    summary[
        "stage_order"
    ] = (
        summary[
            "stage"
        ]
        .map(stage_order)
    )

    summary = summary.sort_values(
        "stage_order"
    )

    return summary


# ============================================================================
# LEVEL SUMMARY
# ============================================================================

def build_level_summary(
    records_df,
):

    zero = records_df[
        records_df[
            "perturbation_scale"
        ]
        == 0.0
    ].copy()

    grouped = zero.groupby(
        [
            "stage",
            "level",
        ],
        sort=False,
    )

    rows = []

    for (
        stage,
        level,
    ), group in grouped:

        rows.append(
            {
                "stage":
                    stage,

                "level":
                    level,

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
# PAIR-LEVEL SUMMARY
# ============================================================================

def build_pair_summary(
    records_df,
):

    zero = records_df[
        records_df[
            "perturbation_scale"
        ]
        == 0.0
    ].copy()

    grouped = zero.groupby(
        "stage",
        sort=False,
    )

    rows = []

    for stage, group in grouped:

        rows.append(
            {
                "stage":
                    stage,

                "records":
                    int(len(group)),

                "rfnn_margin_negative_fraction":
                    float(
                        (
                            group[
                                "modified_rfnn_margin"
                            ]
                            < 0
                        ).mean()
                    ),

                "rfnn_probability_below_0_5_fraction":
                    float(
                        (
                            group[
                                "modified_rfnn_probability"
                            ]
                            < 0.5
                        ).mean()
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
    stage_summary,
    level_summary,
    pair_summary,
):

    records_path = (
        OUTPUT_DIR
        / "part266_stagewise_causal_ablation_records.csv"
    )

    stage_path = (
        OUTPUT_DIR
        / "part266_stagewise_causal_ablation_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part266_stagewise_causal_ablation_level_summary.csv"
    )

    pair_path = (
        OUTPUT_DIR
        / "part266_stagewise_causal_ablation_pair_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part266_stagewise_causal_ablation_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    stage_summary.to_csv(
        stage_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    pair_summary.to_csv(
        pair_path,
        index=False,
    )

    summary = {
        "analysis":
            "Part 2.66 stage-wise causal ablation audit",

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

        "stages":
            STAGES,

        "perturbation_levels":
            PERTURBATION_LEVELS,

        "record_count":
            int(
                len(records_df)
            ),

        "stage_summary":
            stage_summary.to_dict(
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
    print("PART 2.66 STAGE-WISE CAUSAL SUMMARY")
    print("=" * 80)

    print(
        stage_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.66 LEVEL SUMMARY — COMPLETE ABLATION")
    print("=" * 80)

    print(
        level_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.66 COMPLETE")
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
    print(stage_path)
    print(level_path)
    print(pair_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.66")
    print("STAGE-WISE CAUSAL ABLATION AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    print()
    print("Stages:")

    for stage in STAGES:

        print(
            f"  {stage}"
        )

    print()
    print("Perturbations:")

    print("  1.00 = normal")
    print("  0.50 = 50% attenuation")
    print("  0.00 = complete ablation")

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
    print("RUNNING STAGE-WISE CAUSAL ABLATION")
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
        f"Causal ablation records: "
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
        len(STAGES)
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
        "Causal-ablation validation: PASSED"
    )

    print(
        f"Paired observations: "
        f"{len(unique_pairs)}"
    )

    print(
        f"Stages: "
        f"{len(STAGES)}"
    )

    print(
        f"Perturbation levels: "
        f"{len(PERTURBATION_LEVELS)}"
    )

    print(
        f"Expected records: "
        f"{expected_records}"
    )

    stage_summary = (
        build_stage_summary(
            records_df
        )
    )

    level_summary = (
        build_level_summary(
            records_df
        )
    )

    pair_summary = (
        build_pair_summary(
            records_df
        )
    )

    save_outputs(
        records_df,
        stage_summary,
        level_summary,
        pair_summary,
    )


if __name__ == "__main__":
    main()