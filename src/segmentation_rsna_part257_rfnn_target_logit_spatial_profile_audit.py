"""
PART 2.57
RFNN TARGET-LOGIT SPATIAL PROFILE AUDIT

Purpose
-------
Determine whether RFNN target evidence is spatially recoverable around
the annotated LFNN/RFNN pair.

Exact Part 2.20B validation cohort:
    25 studies
    25 series
    45 LFNN points
    45 RFNN points
    45 paired LFNN/RFNN observations

For every paired observation, evaluate five spatial positions:

    LFNN
    P25
    MID
    P75
    RFNN

At each position measure:

    RFNN logit
    Background logit
    RFNN - Background
    LFNN logit
    SCS logit
    LSS logit
    RFNN probability
    Background probability
    RFNN probability margin

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
    / "rsna_part257_rfnn_target_logit_spatial_profile_audit"
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

EXPECTED_CASES = 25

EXPECTED_LFNN = 45

EXPECTED_RFNN = 45

EXPECTED_PAIRS = 45

BACKGROUND_ID = 0

SCS_ID = 1

LFNN_ID = 2

RFNN_ID = 3

LSS_ID = 4

RSS_ID = 5

PROFILE_NAMES = [
    "LFNN",
    "P25",
    "MID",
    "P75",
    "RFNN",
]


# ============================================================================
# VALIDATION COHORT
# ============================================================================

def build_validation_cases(manifest):

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
            "select_validation_series() must return DataFrame."
        )

    if len(selected) != EXPECTED_CASES:

        raise RuntimeError(
            f"Expected {EXPECTED_CASES} validation series, "
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

    validation_cases = (
        part220b.build_case_index(
            validation_manifest
        )
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
        key=lambda case:
        case_order.get(
            (
                str(case["study_id"]),
                str(case["series_id"]),
            ),
            9999,
        ),
    )

    if len(validation_cases) != EXPECTED_CASES:

        raise RuntimeError(
            "Validation case count mismatch."
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
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

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
# MODEL INFERENCE
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
# POINT -> CANONICAL
# ============================================================================

def point_to_canonical(
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

    if canonical.size < 3:

        raise RuntimeError(
            "Canonical point does not contain 3 coordinates."
        )

    return canonical[:3]


# ============================================================================
# INTERPOLATED CANONICAL POINT
# ============================================================================

def interpolate_points(
    left_point,
    right_point,
):

    left = point_to_canonical(
        left_point["point"],
        left_point["geometry"],
    )

    right = point_to_canonical(
        right_point["point"],
        right_point["geometry"],
    )

    # Both points belong to the same series, therefore the same geometry.
    return left, right


def make_profile_coordinates(
    lfnn_point,
    rfnn_point,
    geometry,
):

    lfnn = point_to_canonical(
        lfnn_point,
        geometry,
    )

    rfnn = point_to_canonical(
        rfnn_point,
        geometry,
    )

    positions = {
        "LFNN": lfnn,
        "P25": (
            lfnn
            + 0.25
            * (
                rfnn
                - lfnn
            )
        ),
        "MID": (
            lfnn
            + 0.50
            * (
                rfnn
                - lfnn
            )
        ),
        "P75": (
            lfnn
            + 0.75
            * (
                rfnn
                - lfnn
            )
        ),
        "RFNN": rfnn,
    }

    return positions


# ============================================================================
# TRILINEAR SAMPLING
# ============================================================================

def trilinear_sample(
    volume,
    coordinate,
):

    """
    Sample a 3D volume at floating-point z,y,x coordinates.

    This avoids rounding the profile positions to integer voxels.
    """

    if volume.ndim != 3:

        raise RuntimeError(
            f"Expected 3D volume, got {volume.shape}"
        )

    depth, height, width = volume.shape

    z = float(coordinate[0])
    y = float(coordinate[1])
    x = float(coordinate[2])

    z = np.clip(
        z,
        0.0,
        depth - 1.0,
    )

    y = np.clip(
        y,
        0.0,
        height - 1.0,
    )

    x = np.clip(
        x,
        0.0,
        width - 1.0,
    )

    z0 = int(np.floor(z))
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))

    z1 = min(
        z0 + 1,
        depth - 1,
    )

    y1 = min(
        y0 + 1,
        height - 1,
    )

    x1 = min(
        x0 + 1,
        width - 1,
    )

    dz = z - z0
    dy = y - y0
    dx = x - x0

    c000 = volume[
        z0, y0, x0
    ]

    c001 = volume[
        z0, y0, x1
    ]

    c010 = volume[
        z0, y1, x0
    ]

    c011 = volume[
        z0, y1, x1
    ]

    c100 = volume[
        z1, y0, x0
    ]

    c101 = volume[
        z1, y0, x1
    ]

    c110 = volume[
        z1, y1, x0
    ]

    c111 = volume[
        z1, y1, x1
    ]

    c00 = (
        c000 * (1.0 - dx)
        + c001 * dx
    )

    c01 = (
        c010 * (1.0 - dx)
        + c011 * dx
    )

    c10 = (
        c100 * (1.0 - dx)
        + c101 * dx
    )

    c11 = (
        c110 * (1.0 - dx)
        + c111 * dx
    )

    c0 = (
        c00 * (1.0 - dy)
        + c01 * dy
    )

    c1 = (
        c10 * (1.0 - dy)
        + c11 * dy
    )

    value = (
        c0 * (1.0 - dz)
        + c1 * dz
    )

    return float(value)


# ============================================================================
# SAMPLE MODEL
# ============================================================================

def sample_profile(
    logits,
    probabilities,
    coordinate,
):

    if logits.ndim == 5:

        logits = logits[0]

    if probabilities.ndim == 5:

        probabilities = probabilities[0]

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

    values = {}

    for class_id, name in [
        (BACKGROUND_ID, "background"),
        (SCS_ID, "scs"),
        (LFNN_ID, "lfnn"),
        (RFNN_ID, "rfnn"),
        (LSS_ID, "lss"),
        (RSS_ID, "rss"),
    ]:

        values[
            f"{name}_logit"
        ] = trilinear_sample(
            logits_np[
                class_id
            ],
            coordinate,
        )

        values[
            f"{name}_probability"
        ] = trilinear_sample(
            probabilities_np[
                class_id
            ],
            coordinate,
        )

    values[
        "rfnn_minus_background_logit"
    ] = (
        values[
            "rfnn_logit"
        ]
        -
        values[
            "background_logit"
        ]
    )

    values[
        "rfnn_minus_lss_logit"
    ] = (
        values[
            "rfnn_logit"
        ]
        -
        values[
            "lss_logit"
        ]
    )

    values[
        "rfnn_minus_scs_logit"
    ] = (
        values[
            "rfnn_logit"
        ]
        -
        values[
            "scs_logit"
        ]
    )

    values[
        "rfnn_minus_lfnn_logit"
    ] = (
        values[
            "rfnn_logit"
        ]
        -
        values[
            "lfnn_logit"
        ]
    )

    return values


# ============================================================================
# ENTROPY
# ============================================================================

def calculate_entropy(
    probabilities,
):

    values = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    values = np.clip(
        values,
        1e-12,
        1.0,
    )

    return float(
        -np.sum(
            values
            * np.log(
                values
            )
        )
    )


# ============================================================================
# BUILD PAIRS WITH SAME LEVEL
# ============================================================================

def build_case_pairs(
    points,
):

    point_df = pd.DataFrame(
        points
    )

    lfnn = point_df[
        point_df["class_name"]
        == LFNN_NAME
    ].copy()

    rfnn = point_df[
        point_df["class_name"]
        == RFNN_NAME
    ].copy()

    pairs = lfnn.merge(
        rfnn,
        on=[
            "level",
        ],
        suffixes=(
            "_lfnn",
            "_rfnn",
        ),
    )

    return pairs


# ============================================================================
# RUN ONE CASE
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
        point_df["class_name"]
        == LFNN_NAME
    ]

    rfnn_points = point_df[
        point_df["class_name"]
        == RFNN_NAME
    ]

    if len(lfnn_points) == 0:

        return []

    if len(rfnn_points) == 0:

        return []

    records = []

    for level in sorted(
        set(
            lfnn_points["level"]
        )
        &
        set(
            rfnn_points["level"]
        )
    ):

        lrow = lfnn_points[
            lfnn_points["level"]
            == level
        ].iloc[0]

        rrow = rfnn_points[
            rfnn_points["level"]
            == level
        ].iloc[0]

        positions = make_profile_coordinates(
            lrow,
            rrow,
            geometry,
        )

        for profile_name in PROFILE_NAMES:

            coordinate = positions[
                profile_name
            ]

            values = sample_profile(
                logits,
                probabilities,
                coordinate,
            )

            probability_vector = np.array(
                [
                    values[
                        "background_probability"
                    ],
                    values[
                        "scs_probability"
                    ],
                    values[
                        "lfnn_probability"
                    ],
                    values[
                        "rfnn_probability"
                    ],
                    values[
                        "lss_probability"
                    ],
                    values[
                        "rss_probability"
                    ],
                ],
                dtype=np.float64,
            )

            record = {
                "study_id":
                    study_id,

                "series_id":
                    series_id,

                "level":
                    str(level),

                "position":
                    profile_name,

                "z":
                    float(
                        coordinate[0]
                    ),

                "y":
                    float(
                        coordinate[1]
                    ),

                "x":
                    float(
                        coordinate[2]
                    ),

                "entropy":
                    calculate_entropy(
                        probability_vector
                    ),
            }

            record.update(
                values
            )

            records.append(
                record
            )

    return records


# ============================================================================
# MEAN PROFILE
# ============================================================================

def build_mean_profile(
    records_df,
):

    grouped = (
        records_df
        .groupby(
            "position",
            sort=False,
        )
    )

    rows = []

    for position in PROFILE_NAMES:

        if position not in grouped.groups:

            continue

        subset = grouped.get_group(
            position
        )

        rows.append(
            {
                "position":
                    position,

                "count":
                    len(subset),

                "mean_rfnn_logit":
                    float(
                        subset[
                            "rfnn_logit"
                        ].mean()
                    ),

                "mean_background_logit":
                    float(
                        subset[
                            "background_logit"
                        ].mean()
                    ),

                "mean_rfnn_minus_background":
                    float(
                        subset[
                            "rfnn_minus_background_logit"
                        ].mean()
                    ),

                "mean_lfnn_logit":
                    float(
                        subset[
                            "lfnn_logit"
                        ].mean()
                    ),

                "mean_scs_logit":
                    float(
                        subset[
                            "scs_logit"
                        ].mean()
                    ),

                "mean_lss_logit":
                    float(
                        subset[
                            "lss_logit"
                        ].mean()
                    ),

                "mean_rfnn_probability":
                    float(
                        subset[
                            "rfnn_probability"
                        ].mean()
                    ),

                "mean_background_probability":
                    float(
                        subset[
                            "background_probability"
                        ].mean()
                    ),

                "mean_entropy":
                    float(
                        subset[
                            "entropy"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# MONOTONICITY
# ============================================================================

def build_monotonicity(
    records_df,
):

    rows = []

    metrics = [
        "rfnn_logit",
        "background_logit",
        "rfnn_minus_background_logit",
        "rfnn_probability",
        "background_probability",
    ]

    position_order = {
        name: index
        for index, name
        in enumerate(
            PROFILE_NAMES
        )
    }

    ordered = (
        records_df
        .assign(
            position_order=
                records_df[
                    "position"
                ].map(
                    position_order
                )
        )
        .sort_values(
            [
                "study_id",
                "series_id",
                "level",
                "position_order",
            ]
        )
    )

    for metric in metrics:

        non_decreasing = 0

        non_increasing = 0

        valid = 0

        for _, group in (
            ordered.groupby(
                [
                    "study_id",
                    "series_id",
                    "level",
                ]
            )
        ):

            values = (
                group[
                    metric
                ]
                .to_numpy(
                    dtype=float
                )
            )

            if len(values) != 5:

                continue

            valid += 1

            differences = np.diff(
                values
            )

            if np.all(
                differences
                >= -1e-9
            ):

                non_decreasing += 1

            if np.all(
                differences
                <= 1e-9
            ):

                non_increasing += 1

        rows.append(
            {
                "metric":
                    metric,

                "valid_profiles":
                    valid,

                "non_decreasing_count":
                    non_decreasing,

                "non_increasing_count":
                    non_increasing,

                "non_decreasing_fraction":
                    (
                        non_decreasing
                        / valid
                        if valid
                        else np.nan
                    ),

                "non_increasing_fraction":
                    (
                        non_increasing
                        / valid
                        if valid
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    records_df,
    mean_profile,
    monotonicity,
):

    rfnn_lf = mean_profile.iloc[0]

    rfnn_rf = mean_profile.iloc[-1]

    mid = mean_profile[
        mean_profile["position"]
        == "MID"
    ]

    if len(mid):

        mid_row = mid.iloc[0]

    else:

        mid_row = None

    summary = {
        "profile_records":
            int(
                len(records_df)
            ),

        "expected_profile_records":
            EXPECTED_PAIRS * 5,

        "paired_observations":
            EXPECTED_PAIRS,

        "mean_rfnn_logit_LFNN_endpoint":
            float(
                rfnn_lf[
                    "mean_rfnn_logit"
                ]
            ),

        "mean_rfnn_logit_RFNN_endpoint":
            float(
                rfnn_rf[
                    "mean_rfnn_logit"
                ]
            ),

        "RFNN_endpoint_minus_LFNN_endpoint_rfnn_logit":
            float(
                rfnn_rf[
                    "mean_rfnn_logit"
                ]
                -
                rfnn_lf[
                    "mean_rfnn_logit"
                ]
            ),

        "mean_rfnn_background_margin_LFNN_endpoint":
            float(
                rfnn_lf[
                    "mean_rfnn_minus_background"
                ]
            ),

        "mean_rfnn_background_margin_RFNN_endpoint":
            float(
                rfnn_rf[
                    "mean_rfnn_minus_background"
                ]
            ),

        "mean_rfnn_probability_LFNN_endpoint":
            float(
                rfnn_lf[
                    "mean_rfnn_probability"
                ]
            ),

        "mean_rfnn_probability_RFNN_endpoint":
            float(
                rfnn_rf[
                    "mean_rfnn_probability"
                ]
            ),

        "monotonicity":
            monotonicity.to_dict(
                orient="records"
            ),
    }

    if mid_row is not None:

        summary[
            "mean_rfnn_logit_MID"
        ] = float(
            mid_row[
                "mean_rfnn_logit"
            ]
        )

        summary[
            "mean_rfnn_probability_MID"
        ] = float(
            mid_row[
                "mean_rfnn_probability"
            ]
        )

        summary[
            "mean_rfnn_background_margin_MID"
        ] = float(
            mid_row[
                "mean_rfnn_minus_background"
            ]
        )

    return summary


# ============================================================================
# SAVE
# ============================================================================

def save_outputs(
    records_df,
    mean_profile,
    monotonicity,
    summary,
):

    records_path = (
        OUTPUT_DIR
        / "part257_spatial_logit_profile_records.csv"
    )

    mean_path = (
        OUTPUT_DIR
        / "part257_mean_rfnn_logit_profile.csv"
    )

    monotonicity_path = (
        OUTPUT_DIR
        / "part257_logit_profile_monotonicity.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part257_rfnn_target_logit_spatial_profile_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    mean_profile.to_csv(
        mean_path,
        index=False,
    )

    monotonicity.to_csv(
        monotonicity_path,
        index=False,
    )

    with open(
        summary_path,
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
    print("PART 2.57 MEAN RFNN LOGIT SPATIAL PROFILE")
    print("=" * 80)

    print(
        mean_profile.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.57 LOGIT PROFILE MONOTONICITY")
    print("=" * 80)

    print(
        monotonicity.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.57 COMPLETE")
    print("=" * 80)

    print(
        "Analysis only."
    )

    print(
        "Part 2.27 checkpoint unchanged."
    )

    print(
        "Part 2.56 unchanged."
    )

    print(
        "Dashboard unchanged."
    )

    print()
    print("Outputs:")

    print(records_path)
    print(mean_path)
    print(monotonicity_path)
    print(summary_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.57")
    print("RFNN TARGET-LOGIT SPATIAL PROFILE AUDIT")
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

    model, device = load_model()

    all_records = []

    successful_cases = 0

    print()
    print("=" * 80)
    print("RUNNING RFNN TARGET-LOGIT SPATIAL ANALYSIS")
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

            records = analyze_case(
                model,
                device,
                case,
            )

            all_records.extend(
                records
            )

            successful_cases += 1

            pair_count = (
                len(records)
                // 5
            )

            print(
                f"[{index:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id} | "
                f"Paired levels={pair_count}"
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

    if successful_cases != EXPECTED_CASES:

        raise RuntimeError(
            "Not all validation cases succeeded."
        )

    records_df = pd.DataFrame(
        all_records
    )

    expected_records = (
        EXPECTED_PAIRS * 5
    )

    print(
        f"Profile records: "
        f"{len(records_df)}"
    )

    if len(records_df) != expected_records:

        raise RuntimeError(
            f"Expected {expected_records} "
            f"profile records, found "
            f"{len(records_df)}."
        )

    mean_profile = (
        build_mean_profile(
            records_df
        )
    )

    monotonicity = (
        build_monotonicity(
            records_df
        )
    )

    summary = build_summary(
        records_df,
        mean_profile,
        monotonicity,
    )

    save_outputs(
        records_df,
        mean_profile,
        monotonicity,
        summary,
    )


if __name__ == "__main__":
    main()