"""
PART 2.67
TARGETED STAGE DOSE-RESPONSE AUDIT

Purpose
-------
Test whether the Part 2.66 stage-ablation effects are dose-dependent.

Selected stages:
    encoder1
    encoder10
    decoder3.conv_block.norm2
    decoder2.conv_block.norm2
    decoder4.conv_block.norm2

Attenuation:
    1.00 = normal
    0.75 = 25% attenuation
    0.50 = 50% attenuation
    0.25 = 75% attenuation
    0.00 = complete ablation

Measures:
    RFNN logit
    background logit
    RFNN-background margin
    RFNN probability
    LFNN margin

Analysis only.
No training.
No checkpoint modification.
No dashboard modification.

Expected:
    45 paired observations
    5 stages
    5 attenuation levels
    1125 records
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
    / "rsna_part267_targeted_stage_dose_response_audit"
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

DOSE_LEVELS = [
    1.00,
    0.75,
    0.50,
    0.25,
    0.00,
]

TARGET_STAGES = [
    "encoder1",
    "encoder10",
    "decoder3.conv_block.norm2",
    "decoder2.conv_block.norm2",
    "decoder4.conv_block.norm2",
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
        p.numel()
        for p in model.parameters()
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
# INPUT
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

    tensor = (
        output
        .detach()
        .float()
    )

    if tensor.ndim != 5:

        raise RuntimeError(
            "Expected output shape (B,C,D,H,W)."
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
# METRICS
# ============================================================================

def calculate_metrics(
    logits,
):

    tensor = torch.tensor(
        logits,
        dtype=torch.float32,
    )

    probability = torch.softmax(
        tensor,
        dim=0,
    ).numpy()

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
            rfnn_logit
            -
            background_logit,

        "lfnn_margin":
            lfnn_logit
            -
            background_logit,

        "rfnn_probability":
            float(
                probability[RFNN_ID]
            ),

        "background_probability":
            float(
                probability[BACKGROUND_ID]
            ),
    }


# ============================================================================
# PERTURBATION HOOK
# ============================================================================

class DoseHook:

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

        return (
            output
            *
            self.scale
        )


def register_dose_hook(
    model,
    stage,
    scale,
):

    module = get_module_by_name(
        model,
        stage,
    )

    if module is None:

        raise RuntimeError(
            f"Stage not found: {stage}"
        )

    hook = DoseHook(
        scale
    )

    handle = (
        module.register_forward_hook(
            hook
        )
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

    # ------------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------------

    baseline_output = (
        forward_model(
            model,
            image_tensor,
        )
    )

    baseline = {}

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

        lcanonical = (
            get_canonical_point(
                lpoint,
                geometry,
            )
        )

        rcanonical = (
            get_canonical_point(
                rpoint,
                geometry,
            )
        )

        llogits = (
            sample_output_at_point(
                baseline_output,
                lcanonical,
            )
        )

        rlogits = (
            sample_output_at_point(
                baseline_output,
                rcanonical,
            )
        )

        baseline[str(level)] = {
            "lfnn":
                calculate_metrics(
                    llogits
                ),

            "rfnn":
                calculate_metrics(
                    rlogits
                ),

            "lfnn_canonical":
                lcanonical,

            "rfnn_canonical":
                rcanonical,
        }

    records = []

    # ------------------------------------------------------------------------
    # Stage dose response
    # ------------------------------------------------------------------------

    for stage in TARGET_STAGES:

        module = get_module_by_name(
            model,
            stage,
        )

        if module is None:

            raise RuntimeError(
                f"Stage not found: {stage}"
            )

        for scale in DOSE_LEVELS:

            handle = None

            try:

                handle = (
                    register_dose_hook(
                        model,
                        stage,
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

            for level in common_levels:

                key = str(level)

                if key not in baseline:
                    continue

                base = baseline[key]

                llogits = (
                    sample_output_at_point(
                        modified_output,
                        base[
                            "lfnn_canonical"
                        ],
                    )
                )

                rlogits = (
                    sample_output_at_point(
                        modified_output,
                        base[
                            "rfnn_canonical"
                        ],
                    )
                )

                lm = calculate_metrics(
                    llogits
                )

                rm = calculate_metrics(
                    rlogits
                )

                records.append(
                    {
                        "study_id":
                            study_id,

                        "series_id":
                            series_id,

                        "level":
                            key,

                        "stage":
                            stage,

                        "dose":
                            float(scale),

                        "dose_percent":
                            float(
                                scale * 100.0
                            ),

                        "baseline_rfnn_logit":
                            base["rfnn"][
                                "rfnn_logit"
                            ],

                        "modified_rfnn_logit":
                            rm[
                                "rfnn_logit"
                            ],

                        "delta_rfnn_logit":
                            rm[
                                "rfnn_logit"
                            ]
                            -
                            base["rfnn"][
                                "rfnn_logit"
                            ],

                        "baseline_background_logit":
                            base["rfnn"][
                                "background_logit"
                            ],

                        "modified_background_logit":
                            rm[
                                "background_logit"
                            ],

                        "delta_background_logit":
                            rm[
                                "background_logit"
                            ]
                            -
                            base["rfnn"][
                                "background_logit"
                            ],

                        "baseline_rfnn_margin":
                            base["rfnn"][
                                "rfnn_margin"
                            ],

                        "modified_rfnn_margin":
                            rm[
                                "rfnn_margin"
                            ],

                        "delta_rfnn_margin":
                            rm[
                                "rfnn_margin"
                            ]
                            -
                            base["rfnn"][
                                "rfnn_margin"
                            ],

                        "baseline_rfnn_probability":
                            base["rfnn"][
                                "rfnn_probability"
                            ],

                        "modified_rfnn_probability":
                            rm[
                                "rfnn_probability"
                            ],

                        "delta_rfnn_probability":
                            rm[
                                "rfnn_probability"
                            ]
                            -
                            base["rfnn"][
                                "rfnn_probability"
                            ],

                        "baseline_lfnn_margin":
                            base["lfnn"][
                                "lfnn_margin"
                            ],

                        "modified_lfnn_margin":
                            lm[
                                "lfnn_margin"
                            ],

                        "delta_lfnn_margin":
                            lm[
                                "lfnn_margin"
                            ]
                            -
                            base["lfnn"][
                                "lfnn_margin"
                            ],
                    }
                )

    return records


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    records_df,
):

    rows = []

    grouped = records_df.groupby(
        [
            "stage",
            "dose",
        ],
        sort=False,
    )

    for (
        stage,
        dose,
    ), group in grouped:

        rows.append(
            {
                "stage":
                    stage,

                "dose":
                    float(dose),

                "dose_percent":
                    float(
                        dose * 100.0
                    ),

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

                "rfnn_margin_decreased_fraction":
                    float(
                        (
                            group[
                                "delta_rfnn_margin"
                            ]
                            < 0
                        ).mean()
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    stage_order = {
        stage: index
        for index, stage
        in enumerate(
            TARGET_STAGES,
            start=1,
        )
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
        [
            "stage_order",
            "dose",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return summary


# ============================================================================
# COMPLETE-ABLATION SUMMARY
# ============================================================================

def build_zero_summary(
    records_df,
):

    zero = records_df[
        records_df["dose"] == 0.0
    ].copy()

    rows = []

    for stage, group in (
        zero.groupby(
            "stage",
            sort=False,
        )
    ):

        rows.append(
            {
                "stage":
                    stage,

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
# SAVE
# ============================================================================

def save_outputs(
    records_df,
    summary_df,
    zero_summary,
):

    records_path = (
        OUTPUT_DIR
        / "part267_stage_dose_response_records.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part267_stage_dose_response_summary.csv"
    )

    zero_path = (
        OUTPUT_DIR
        / "part267_stage_complete_ablation_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part267_stage_dose_response_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    zero_summary.to_csv(
        zero_path,
        index=False,
    )

    output = {
        "analysis":
            "Part 2.67 targeted stage dose-response audit",

        "checkpoint":
            str(
                CHECKPOINT_PATH
            ),

        "paired_observations":
            EXPECTED_PAIRS,

        "target_stages":
            TARGET_STAGES,

        "dose_levels":
            DOSE_LEVELS,

        "records":
            int(
                len(records_df)
            ),

        "complete_ablation":
            zero_summary.to_dict(
                orient="records"
            ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print("PART 2.67 DOSE-RESPONSE SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.67 COMPLETE-ABLATION SUMMARY")
    print("=" * 80)

    print(
        zero_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.67 COMPLETE")
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
    print(zero_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.67")
    print("TARGETED STAGE DOSE-RESPONSE AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    print()
    print("Target stages:")

    for stage in TARGET_STAGES:

        print(
            f"  {stage}"
        )

    print()
    print("Dose levels:")

    for dose in DOSE_LEVELS:

        print(
            f"  {dose:.2f}"
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
    print("RUNNING TARGETED DOSE-RESPONSE AUDIT")
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
        f"Dose-response records: "
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
        len(TARGET_STAGES)
        *
        len(DOSE_LEVELS)
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
        "Dose-response validation: PASSED"
    )

    print(
        f"Paired observations: "
        f"{len(unique_pairs)}"
    )

    print(
        f"Target stages: "
        f"{len(TARGET_STAGES)}"
    )

    print(
        f"Dose levels: "
        f"{len(DOSE_LEVELS)}"
    )

    print(
        f"Expected records: "
        f"{expected_records}"
    )

    summary_df = build_summary(
        records_df
    )

    zero_summary = (
        build_zero_summary(
            records_df
        )
    )

    save_outputs(
        records_df,
        summary_df,
        zero_summary,
    )


if __name__ == "__main__":
    main()