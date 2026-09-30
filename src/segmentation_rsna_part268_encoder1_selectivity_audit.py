"""
PART 2.68
ENCODER1 RFNN-SELECTIVITY AUDIT

Purpose
-------
Determine whether the unusual Encoder1 dose-response is specifically
associated with the RFNN anatomical point or is caused by broader
multi-class/background changes.

This experiment does NOT train anything.

It evaluates Encoder1 at:

    1.00 = normal
    0.75
    0.50
    0.25
    0.00 = complete ablation

For BOTH paired anatomical points:

    LFNN point
    RFNN point

it records:

    RFNN logit
    LFNN logit
    Background logit
    RFNN probability
    LFNN probability
    Background probability
    RFNN-background margin
    LFNN-background margin
    RFNN-LFNN logit difference

Analysis only.
No checkpoint modification.
No dashboard modification.

Expected:
    45 paired observations
    5 doses
    2 anatomical points

Expected records:
    45 * 5 * 2 = 450
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
    / "rsna_part268_encoder1_selectivity_audit"
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

DOSES = [
    1.00,
    0.75,
    0.50,
    0.25,
    0.00,
]

STAGE = "encoder1"


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
# POINT MAPPING
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
            background_logit,

        "lfnn_margin":
            lfnn_logit
            -
            background_logit,

        "rfnn_lfnn_logit_difference":
            rfnn_logit
            -
            lfnn_logit,
    }


# ============================================================================
# ENCODER1 HOOK
# ============================================================================

class Encoder1DoseHook:

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
                "Encoder1 output is not a tensor."
            )

        return (
            output
            *
            self.scale
        )


def register_encoder1_hook(
    model,
    scale,
):

    module = get_module_by_name(
        model,
        STAGE,
    )

    if module is None:

        raise RuntimeError(
            "encoder1 module not found."
        )

    hook = Encoder1DoseHook(
        scale
    )

    return module.register_forward_hook(
        hook
    )


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
    # Point locations
    # ------------------------------------------------------------------------

    point_locations = {}

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

        point_locations[
            str(level)
        ] = {
            "lfnn":
                get_canonical_point(
                    lrow.iloc[0],
                    geometry,
                ),

            "rfnn":
                get_canonical_point(
                    rrow.iloc[0],
                    geometry,
                ),
        }

    records = []

    # ------------------------------------------------------------------------
    # Encoder1 dose response
    # ------------------------------------------------------------------------

    for dose in DOSES:

        handle = None

        try:

            handle = (
                register_encoder1_hook(
                    model,
                    dose,
                )
            )

            output = (
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

            if key not in point_locations:
                continue

            for point_type in (
                "lfnn",
                "rfnn",
            ):

                logits = (
                    sample_output_at_point(
                        output,
                        point_locations[
                            key
                        ][
                            point_type
                        ],
                    )
                )

                metrics = (
                    calculate_metrics(
                        logits
                    )
                )

                records.append(
                    {
                        "study_id":
                            study_id,

                        "series_id":
                            series_id,

                        "level":
                            key,

                        "point_type":
                            point_type,

                        "dose":
                            float(dose),

                        "dose_percent":
                            float(
                                dose * 100.0
                            ),

                        "rfnn_logit":
                            metrics[
                                "rfnn_logit"
                            ],

                        "lfnn_logit":
                            metrics[
                                "lfnn_logit"
                            ],

                        "background_logit":
                            metrics[
                                "background_logit"
                            ],

                        "rfnn_probability":
                            metrics[
                                "rfnn_probability"
                            ],

                        "lfnn_probability":
                            metrics[
                                "lfnn_probability"
                            ],

                        "background_probability":
                            metrics[
                                "background_probability"
                            ],

                        "rfnn_margin":
                            metrics[
                                "rfnn_margin"
                            ],

                        "lfnn_margin":
                            metrics[
                                "lfnn_margin"
                            ],

                        "rfnn_lfnn_logit_difference":
                            metrics[
                                "rfnn_lfnn_logit_difference"
                            ],
                    }
                )

    return records


# ============================================================================
# BASELINE DELTA SUMMARY
# ============================================================================

def add_baseline_deltas(
    records_df,
):

    baseline = records_df[
        records_df["dose"] == 1.0
    ].copy()

    baseline = baseline[
        [
            "study_id",
            "series_id",
            "level",
            "point_type",
            "rfnn_logit",
            "lfnn_logit",
            "background_logit",
            "rfnn_probability",
            "lfnn_probability",
            "background_probability",
            "rfnn_margin",
            "lfnn_margin",
            "rfnn_lfnn_logit_difference",
        ]
    ].copy()

    rename_map = {
        column:
            f"baseline_{column}"
        for column in baseline.columns
        if column not in (
            "study_id",
            "series_id",
            "level",
            "point_type",
        )
    }

    baseline = baseline.rename(
        columns=rename_map
    )

    merged = records_df.merge(
        baseline,
        on=[
            "study_id",
            "series_id",
            "level",
            "point_type",
        ],
        how="left",
        validate="many_to_one",
    )

    merged[
        "delta_rfnn_logit"
    ] = (
        merged["rfnn_logit"]
        -
        merged[
            "baseline_rfnn_logit"
        ]
    )

    merged[
        "delta_lfnn_logit"
    ] = (
        merged["lfnn_logit"]
        -
        merged[
            "baseline_lfnn_logit"
        ]
    )

    merged[
        "delta_background_logit"
    ] = (
        merged["background_logit"]
        -
        merged[
            "baseline_background_logit"
        ]
    )

    merged[
        "delta_rfnn_margin"
    ] = (
        merged["rfnn_margin"]
        -
        merged[
            "baseline_rfnn_margin"
        ]
    )

    merged[
        "delta_lfnn_margin"
    ] = (
        merged["lfnn_margin"]
        -
        merged[
            "baseline_lfnn_margin"
        ]
    )

    merged[
        "delta_rfnn_probability"
    ] = (
        merged["rfnn_probability"]
        -
        merged[
            "baseline_rfnn_probability"
        ]
    )

    merged[
        "delta_background_probability"
    ] = (
        merged[
            "background_probability"
        ]
        -
        merged[
            "baseline_background_probability"
        ]
    )

    merged[
        "delta_rfnn_lfnn_logit_difference"
    ] = (
        merged[
            "rfnn_lfnn_logit_difference"
        ]
        -
        merged[
            "baseline_rfnn_lfnn_logit_difference"
        ]
    )

    return merged


# ============================================================================
# SELECTIVITY SUMMARY
# ============================================================================

def build_selectivity_summary(
    df,
):

    nonbaseline = df[
        df["dose"] < 1.0
    ].copy()

    rows = []

    for dose, group in (
        nonbaseline.groupby(
            "dose",
            sort=False,
        )
    ):

        lgroup = group[
            group["point_type"]
            == "lfnn"
        ]

        rgroup = group[
            group["point_type"]
            == "rfnn"
        ]

        rows.append(
            {
                "dose":
                    float(dose),

                "dose_percent":
                    float(
                        dose * 100.0
                    ),

                "records":
                    int(len(group)),

                "rfnn_point_mean_delta_rfnn_margin":
                    float(
                        rgroup[
                            "delta_rfnn_margin"
                        ].mean()
                    ),

                "lfnn_point_mean_delta_rfnn_margin":
                    float(
                        lgroup[
                            "delta_rfnn_margin"
                        ].mean()
                    ),

                "rfnn_point_mean_delta_lfnn_margin":
                    float(
                        rgroup[
                            "delta_lfnn_margin"
                        ].mean()
                    ),

                "lfnn_point_mean_delta_lfnn_margin":
                    float(
                        lgroup[
                            "delta_lfnn_margin"
                        ].mean()
                    ),

                "rfnn_point_mean_delta_rfnn_logit":
                    float(
                        rgroup[
                            "delta_rfnn_logit"
                        ].mean()
                    ),

                "lfnn_point_mean_delta_rfnn_logit":
                    float(
                        lgroup[
                            "delta_rfnn_logit"
                        ].mean()
                    ),

                "rfnn_point_mean_delta_background":
                    float(
                        rgroup[
                            "delta_background_logit"
                        ].mean()
                    ),

                "lfnn_point_mean_delta_background":
                    float(
                        lgroup[
                            "delta_background_logit"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# COMPLETE ABLATION POINT SUMMARY
# ============================================================================

def build_point_summary(
    df,
):

    zero = df[
        df["dose"] == 0.0
    ].copy()

    rows = []

    for point_type, group in (
        zero.groupby(
            "point_type",
            sort=False,
        )
    ):

        rows.append(
            {
                "point_type":
                    point_type,

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
# LEVEL SUMMARY
# ============================================================================

def build_level_summary(
    df,
):

    zero = df[
        df["dose"] == 0.0
    ].copy()

    rows = []

    for (
        level,
        point_type,
    ), group in zero.groupby(
        [
            "level",
            "point_type",
        ],
        sort=False,
    ):

        rows.append(
            {
                "level":
                    level,

                "point_type":
                    point_type,

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
    selectivity_summary,
    point_summary,
    level_summary,
):

    records_path = (
        OUTPUT_DIR
        / "part268_encoder1_selectivity_records.csv"
    )

    selectivity_path = (
        OUTPUT_DIR
        / "part268_encoder1_selectivity_summary.csv"
    )

    point_path = (
        OUTPUT_DIR
        / "part268_encoder1_complete_ablation_point_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part268_encoder1_complete_ablation_level_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part268_encoder1_selectivity_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    selectivity_summary.to_csv(
        selectivity_path,
        index=False,
    )

    point_summary.to_csv(
        point_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    summary = {
        "analysis":
            "Part 2.68 Encoder1 RFNN-selectivity audit",

        "checkpoint":
            str(
                CHECKPOINT_PATH
            ),

        "paired_observations":
            EXPECTED_PAIRS,

        "doses":
            DOSES,

        "records":
            int(
                len(records_df)
            ),

        "selectivity_summary":
            selectivity_summary.to_dict(
                orient="records"
            ),

        "complete_ablation":
            point_summary.to_dict(
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
    print("PART 2.68 SELECTIVITY SUMMARY")
    print("=" * 80)

    print(
        selectivity_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.68 COMPLETE-ABLATION POINT SUMMARY")
    print("=" * 80)

    print(
        point_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.68 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.68 COMPLETE")
    print("=" * 80)

    print("Analysis only.")
    print("No training.")
    print("Part 2.27 checkpoint unchanged.")
    print("Dashboard unchanged.")

    print()
    print("Outputs:")

    print(records_path)
    print(selectivity_path)
    print(point_path)
    print(level_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.68")
    print("ENCODER1 RFNN-SELECTIVITY AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    print()
    print("Encoder1 doses:")

    for dose in DOSES:

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
    print("RUNNING ENCODER1 SELECTIVITY AUDIT")
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
        f"Raw records: "
        f"{len(records_df)}"
    )

    if paired_series != (
        EXPECTED_PAIRED_SERIES
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_PAIRED_SERIES} "
            f"paired series, found {paired_series}."
        )

    expected_records = (
        EXPECTED_PAIRS
        *
        len(DOSES)
        *
        2
    )

    if len(records_df) != (
        expected_records
    ):

        raise RuntimeError(
            f"Expected {expected_records} records, "
            f"found {len(records_df)}."
        )

    records_df = add_baseline_deltas(
        records_df
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
        "Encoder1 selectivity validation: PASSED"
    )

    print(
        f"Paired observations: "
        f"{len(unique_pairs)}"
    )

    print(
        f"Doses: "
        f"{len(DOSES)}"
    )

    print(
        f"Point types: 2"
    )

    print(
        f"Expected records: "
        f"{expected_records}"
    )

    selectivity_summary = (
        build_selectivity_summary(
            records_df
        )
    )

    point_summary = (
        build_point_summary(
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
        selectivity_summary,
        point_summary,
        level_summary,
    )


if __name__ == "__main__":
    main()