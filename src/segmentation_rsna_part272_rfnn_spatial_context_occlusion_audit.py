"""
PART 2.72
RFNN SPATIAL-CONTEXT OCCLUSION AUDIT

Purpose
-------
Measure how RFNN predictions change when progressively larger local
neighborhoods around the RFNN annotation are replaced by a neutral
local intensity value.

Radii:
    0   = original image
    2
    4
    6
    8
    10

For each RFNN point we record:
    - RFNN logit
    - background logit
    - RFNN-background margin
    - RFNN probability
    - LFNN logit
    - LSS logit
    - RSS logit
    - SCS logit
    - predicted class
    - RFNN probability change
    - RFNN margin change

This is an analysis-only experiment.

NO:
    - training
    - checkpoint modification
    - dashboard modification
    - fabricated voxel ground truth

Checkpoint:
    Part 2.27 best macro disease checkpoint.

Expected:
    25 validation cases
    9 paired validation series
    45 RFNN observations
    6 radii
    270 records
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
    / "rsna_part272_rfnn_spatial_context_occlusion_audit"
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

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

RFNN_NAME = "Right Neural Foraminal Narrowing"

OCCLUSION_RADII = [
    0,
    2,
    4,
    6,
    8,
    10,
]

EXPECTED_CASES = 25
EXPECTED_PAIRED_SERIES = 9
EXPECTED_RFNN_POINTS = 45
EXPECTED_RECORDS = 45 * 6


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
# IMAGE PREPARATION
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
# OCCLUSION
# ============================================================================

def create_occluded_image(
    image,
    center,
    radius,
):

    if radius == 0:

        return image.clone()

    z, y, x = center

    z0 = max(
        0,
        z - radius,
    )

    z1 = min(
        int(image.shape[-3]),
        z + radius + 1,
    )

    y0 = max(
        0,
        y - radius,
    )

    y1 = min(
        int(image.shape[-2]),
        y + radius + 1,
    )

    x0 = max(
        0,
        x - radius,
    )

    x1 = min(
        int(image.shape[-1]),
        x + radius + 1,
    )

    result = image.clone()

    # Use the global neutral intensity of this normalized volume.
    neutral_value = torch.median(
        image
    )

    result[
        :,
        :,
        z0:z1,
        y0:y1,
        x0:x1,
    ] = neutral_value

    return result


# ============================================================================
# MODEL OUTPUT
# ============================================================================

def get_point_metrics(
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

    predicted_class = int(
        np.argmax(
            probabilities
        )
    )

    return {
        "background_logit":
            float(logits[BACKGROUND_ID]),

        "scs_logit":
            float(logits[SCS_ID]),

        "lfnn_logit":
            float(logits[LFNN_ID]),

        "rfnn_logit":
            float(logits[RFNN_ID]),

        "lss_logit":
            float(logits[LSS_ID]),

        "rss_logit":
            float(logits[RSS_ID]),

        "background_probability":
            float(
                probabilities[
                    BACKGROUND_ID
                ]
            ),

        "scs_probability":
            float(
                probabilities[
                    SCS_ID
                ]
            ),

        "lfnn_probability":
            float(
                probabilities[
                    LFNN_ID
                ]
            ),

        "rfnn_probability":
            float(
                probabilities[
                    RFNN_ID
                ]
            ),

        "lss_probability":
            float(
                probabilities[
                    LSS_ID
                ]
            ),

        "rss_probability":
            float(
                probabilities[
                    RSS_ID
                ]
            ),

        "rfnn_margin":
            float(
                logits[RFNN_ID]
                -
                logits[BACKGROUND_ID]
            ),

        "predicted_class":
            predicted_class,

        "predicted_class_name":
            CLASS_NAMES[
                predicted_class
            ],
    }


# ============================================================================
# SINGLE CASE
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

    rfnn_df = point_df[
        point_df["class_name"]
        == RFNN_NAME
    ].copy()

    if len(rfnn_df) != 5:

        return []

    records = []

    with torch.no_grad():

        baseline_output = model(
            image_tensor
        )

    for _, point in rfnn_df.iterrows():

        level = str(
            point["level"]
        )

        canonical_point = (
            get_canonical_point(
                point,
                geometry,
            )
        )

        center = canonical_to_index(
            canonical_point,
            image_tensor.shape,
        )

        baseline = get_point_metrics(
            baseline_output,
            canonical_point,
        )

        for radius in OCCLUSION_RADII:

            occluded_image = (
                create_occluded_image(
                    image_tensor,
                    center,
                    radius,
                )
            )

            if radius == 0:

                output = (
                    baseline_output
                )

            else:

                with torch.no_grad():

                    output = model(
                        occluded_image
                    )

            metrics = get_point_metrics(
                output,
                canonical_point,
            )

            records.append(
                {
                    "study_id":
                        study_id,

                    "series_id":
                        series_id,

                    "level":
                        level,

                    "radius":
                        int(radius),

                    "patch_side":
                        int(
                            2 * radius + 1
                        ),

                    "rfnn_logit":
                        metrics[
                            "rfnn_logit"
                        ],

                    "background_logit":
                        metrics[
                            "background_logit"
                        ],

                    "scs_logit":
                        metrics[
                            "scs_logit"
                        ],

                    "lfnn_logit":
                        metrics[
                            "lfnn_logit"
                        ],

                    "lss_logit":
                        metrics[
                            "lss_logit"
                        ],

                    "rss_logit":
                        metrics[
                            "rss_logit"
                        ],

                    "rfnn_probability":
                        metrics[
                            "rfnn_probability"
                        ],

                    "background_probability":
                        metrics[
                            "background_probability"
                        ],

                    "scs_probability":
                        metrics[
                            "scs_probability"
                        ],

                    "lfnn_probability":
                        metrics[
                            "lfnn_probability"
                        ],

                    "lss_probability":
                        metrics[
                            "lss_probability"
                        ],

                    "rss_probability":
                        metrics[
                            "rss_probability"
                        ],

                    "rfnn_margin":
                        metrics[
                            "rfnn_margin"
                        ],

                    "predicted_class":
                        metrics[
                            "predicted_class"
                        ],

                    "predicted_class_name":
                        metrics[
                            "predicted_class_name"
                        ],

                    "delta_rfnn_logit":
                        (
                            metrics["rfnn_logit"]
                            -
                            baseline["rfnn_logit"]
                        ),

                    "delta_background_logit":
                        (
                            metrics[
                                "background_logit"
                            ]
                            -
                            baseline[
                                "background_logit"
                            ]
                        ),

                    "delta_rfnn_margin":
                        (
                            metrics[
                                "rfnn_margin"
                            ]
                            -
                            baseline[
                                "rfnn_margin"
                            ]
                        ),

                    "delta_rfnn_probability":
                        (
                            metrics[
                                "rfnn_probability"
                            ]
                            -
                            baseline[
                                "rfnn_probability"
                            ]
                        ),

                    "delta_lfnn_logit":
                        (
                            metrics[
                                "lfnn_logit"
                            ]
                            -
                            baseline[
                                "lfnn_logit"
                            ]
                        ),

                    "baseline_rfnn_logit":
                        baseline[
                            "rfnn_logit"
                        ],

                    "baseline_background_logit":
                        baseline[
                            "background_logit"
                        ],

                    "baseline_rfnn_margin":
                        baseline[
                            "rfnn_margin"
                        ],

                    "baseline_rfnn_probability":
                        baseline[
                            "rfnn_probability"
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

    grouped = records_df.groupby(
        ["radius"],
        sort=False,
    )

    rows = []

    for group_key, group in grouped:

        if isinstance(
            group_key,
            tuple,
        ):

            radius = group_key[0]

        else:

            radius = group_key

        rows.append(
            {
                "radius":
                    int(radius),

                "patch_side":
                    int(
                        2 * int(radius) + 1
                    ),

                "records":
                    int(len(group)),

                "mean_rfnn_logit":
                    float(
                        group[
                            "rfnn_logit"
                        ].mean()
                    ),

                "mean_background_logit":
                    float(
                        group[
                            "background_logit"
                        ].mean()
                    ),

                "mean_rfnn_margin":
                    float(
                        group[
                            "rfnn_margin"
                        ].mean()
                    ),

                "mean_rfnn_probability":
                    float(
                        group[
                            "rfnn_probability"
                        ].mean()
                    ),

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

                "mean_delta_lfnn_logit":
                    float(
                        group[
                            "delta_lfnn_logit"
                        ].mean()
                    ),

                "rfnn_predicted_fraction":
                    float(
                        (
                            group[
                                "predicted_class"
                            ]
                            == RFNN_ID
                        ).mean()
                    ),

                "background_predicted_fraction":
                    float(
                        (
                            group[
                                "predicted_class"
                            ]
                            == BACKGROUND_ID
                        ).mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


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

    for (
        level,
        radius,
    ), group in grouped:

        rows.append(
            {
                "level":
                    str(level),

                "radius":
                    int(radius),

                "records":
                    int(len(group)),

                "mean_rfnn_logit":
                    float(
                        group[
                            "rfnn_logit"
                        ].mean()
                    ),

                "mean_background_logit":
                    float(
                        group[
                            "background_logit"
                        ].mean()
                    ),

                "mean_rfnn_margin":
                    float(
                        group[
                            "rfnn_margin"
                        ].mean()
                    ),

                "mean_rfnn_probability":
                    float(
                        group[
                            "rfnn_probability"
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

                "background_fraction":
                    float(
                        (
                            group[
                                "predicted_class"
                            ]
                            == BACKGROUND_ID
                        ).mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.72")
    print("RFNN SPATIAL-CONTEXT OCCLUSION AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    print()
    print(
        f"Occlusion radii: "
        f"{OCCLUSION_RADII}"
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
    print("RUNNING RFNN SPATIAL-CONTEXT OCCLUSION")
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

            if len(case_records) == 30:

                all_records.extend(
                    case_records
                )

                paired_series += 1

                print(
                    f"[{index:02d}/25] "
                    f"Study={study_id} | "
                    f"Series={series_id} | "
                    f"RFNN levels=5 | "
                    f"records={len(case_records)}"
                )

            else:

                print(
                    f"[{index:02d}/25] "
                    f"Study={study_id} | "
                    f"Series={series_id} | "
                    f"RFNN records={len(case_records)}"
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
        f"RFNN validation series: "
        f"{paired_series}/25"
    )

    print(
        f"Occlusion records: "
        f"{len(records_df)}"
    )

    if paired_series != (
        EXPECTED_PAIRED_SERIES
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_PAIRED_SERIES} "
            f"RFNN series, found {paired_series}."
        )

    if len(records_df) != (
        EXPECTED_RECORDS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} "
            f"records, found {len(records_df)}."
        )

    unique_points = (
        records_df[
            [
                "study_id",
                "series_id",
                "level",
            ]
        ]
        .drop_duplicates()
    )

    if len(unique_points) != (
        EXPECTED_RFNN_POINTS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_RFNN_POINTS} "
            f"RFNN observations, found "
            f"{len(unique_points)}."
        )

    print()
    print(
        "Occlusion validation: PASSED"
    )

    print(
        f"RFNN observations: "
        f"{len(unique_points)}"
    )

    print(
        f"Expected records: "
        f"{EXPECTED_RECORDS}"
    )

    summary_df = build_summary(
        records_df
    )

    level_df = build_level_summary(
        records_df
    )

    records_path = (
        OUTPUT_DIR
        / "part272_rfnn_occlusion_records.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part272_rfnn_occlusion_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part272_rfnn_occlusion_level_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part272_rfnn_occlusion_summary.json"
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
            "Part 2.72 RFNN spatial-context occlusion audit",

        "validation_cases":
            EXPECTED_CASES,

        "rfnn_validation_series":
            paired_series,

        "rfnn_observations":
            len(unique_points),

        "occlusion_radii":
            OCCLUSION_RADII,

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
    print("PART 2.72 SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.72 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.72 COMPLETE")
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