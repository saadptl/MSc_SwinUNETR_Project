"""
PART 2.47
RFNN POINT-VS-NEIGHBORHOOD EVIDENCE DECAY AUDIT

Purpose
-------
Determine how RFNN evidence changes as the spatial neighborhood around
the annotated RFNN point becomes larger.

For every RFNN annotation, measure:

    Point
    Radius 2
    Radius 4
    Radius 6

For each location, calculate:

    RFNN probability
    RFNN logit
    RFNN - Background logit
    RFNN - LSS logit
    RFNN - SCS logit
    RFNN - LFNN logit
    RFNN - RSS logit

Then calculate evidence changes:

    Point -> Radius 2
    Radius 2 -> Radius 4
    Radius 4 -> Radius 6
    Point -> Radius 6

The analysis is performed for:

    Overall RFNN
    L1/L2
    L2/L3
    L3/L4
    L4/L5
    L5/S1

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
    / "rsna_part247_rfnn_point_neighborhood_decay_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

sys.path.insert(
    0,
    str(SRC_DIR),
)

import segmentation_rsna_part220b_geometry_corrected_training as part220b


# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
    / "part227_best_macro_disease.pth"
)

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

LEVELS = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]

RADII = [
    2,
    4,
    6,
]

COMPARISONS = [
    "background",
    "lss",
    "scs",
    "lfnn",
    "rss",
]


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(value):

    try:

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:

        return np.nan


def clamp_coordinate(
    value,
    maximum,
):

    return int(
        max(
            0,
            min(
                int(round(float(value))),
                int(maximum) - 1,
            ),
        )
    )


def local_cube(
    tensor,
    z,
    y,
    x,
    radius,
):
    """
    tensor shape:

        [C, Z, Y, X]
    """

    _, depth, height, width = (
        tensor.shape
    )

    z = clamp_coordinate(
        z,
        depth,
    )

    y = clamp_coordinate(
        y,
        height,
    )

    x = clamp_coordinate(
        x,
        width,
    )

    z0 = max(
        0,
        z - radius,
    )

    z1 = min(
        depth,
        z + radius + 1,
    )

    y0 = max(
        0,
        y - radius,
    )

    y1 = min(
        height,
        y + radius + 1,
    )

    x0 = max(
        0,
        x - radius,
    )

    x1 = min(
        width,
        x + radius + 1,
    )

    return tensor[
        :,
        z0:z1,
        y0:y1,
        x0:x1,
    ]


def point_values(
    tensor,
    z,
    y,
    x,
):
    """
    Get class values at one model-grid point.
    """

    _, depth, height, width = (
        tensor.shape
    )

    z = clamp_coordinate(
        z,
        depth,
    )

    y = clamp_coordinate(
        y,
        height,
    )

    x = clamp_coordinate(
        x,
        width,
    )

    return (
        tensor[
            :,
            z,
            y,
            x,
        ]
        .detach()
        .cpu()
        .numpy()
    )


def local_max_values(
    tensor,
    z,
    y,
    x,
    radius,
):
    """
    Maximum class value within the local cube.
    """

    patch = local_cube(
        tensor,
        z,
        y,
        x,
        radius,
    )

    flat = patch.reshape(
        patch.shape[0],
        -1,
    )

    return (
        flat.max(
            dim=1
        )
        .values
        .detach()
        .cpu()
        .numpy()
    )


def local_mean_values(
    tensor,
    z,
    y,
    x,
    radius,
):
    """
    Mean class value within the local cube.
    """

    patch = local_cube(
        tensor,
        z,
        y,
        x,
        radius,
    )

    flat = patch.reshape(
        patch.shape[0],
        -1,
    )

    return (
        flat.mean(
            dim=1
        )
        .detach()
        .cpu()
        .numpy()
    )


def build_model():

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
        and "model_state_dict"
        in checkpoint
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
        f"Missing keys    : "
        f"{len(missing)}"
    )

    print(
        f"Unexpected keys : "
        f"{len(unexpected)}"
    )

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    if missing:
        raise RuntimeError(
            "Checkpoint contains missing keys."
        )

    if unexpected:
        raise RuntimeError(
            "Checkpoint contains unexpected keys."
        )

    return model


# ============================================================================
# EXACT PART 2.20B VALIDATION COHORT
# ============================================================================

def build_exact_validation_cohort(
    manifest,
):

    print()
    print("=" * 80)
    print("BUILDING EXACT PART 2.20B VALIDATION COHORT")
    print("=" * 80)

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
            "Part 2.20B selector must return "
            "a pandas DataFrame."
        )

    if len(selected) != 25:

        raise RuntimeError(
            "Expected 25 validation series, "
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
            )
            in validation_keys,
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
            10000,
        ),
    )

    records = []

    for case in validation_cases:

        study_id = str(
            case["study_id"]
        )

        series_id = str(
            case["series_id"]
        )

        case_points = case["points"]

        if isinstance(
            case_points,
            pd.DataFrame,
        ):

            point_df = (
                case_points
                .reset_index(drop=True)
            )

        else:

            point_df = (
                pd.DataFrame(
                    case_points
                )
                .reset_index(drop=True)
            )

        for point_index, (_, row) in enumerate(
            point_df.iterrows()
        ):

            records.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "point_index": int(
                        point_index
                    ),
                    "class_id": int(
                        row["class_id"]
                    ),
                    "class_name": str(
                        row["class_name"]
                    ),
                    "level": str(
                        row["level"]
                    ),
                }
            )

    validation_points = pd.DataFrame(
        records
    )

    expected_counts = {
        "Spinal Canal Stenosis": 25,
        "Left Neural Foraminal Narrowing": 45,
        "Right Neural Foraminal Narrowing": 45,
        "Left Subarticular Stenosis": 42,
        "Right Subarticular Stenosis": 31,
    }

    actual_counts = (
        validation_points[
            "class_name"
        ]
        .value_counts()
        .to_dict()
    )

    for disease, expected in (
        expected_counts.items()
    ):

        actual = int(
            actual_counts.get(
                disease,
                0,
            )
        )

        if actual != expected:

            raise RuntimeError(
                f"Cohort mismatch for "
                f"{disease}: expected "
                f"{expected}, found {actual}."
            )

    if len(validation_points) != 188:

        raise RuntimeError(
            "Expected 188 validation points, "
            f"found {len(validation_points)}."
        )

    print()
    print("EXACT COHORT VERIFIED")
    print("-" * 80)
    print("Validation series : 25")
    print("Validation studies: 25")
    print("Annotated points  : 188")
    print("SCS               : 25")
    print("LFNN              : 45")
    print("RFNN              : 45")
    print("LSS               : 42")
    print("RSS               : 31")
    print("-" * 80)

    return (
        validation_cases,
        validation_points,
    )


# ============================================================================
# CASE ANALYSIS
# ============================================================================

def analyze_case(
    model,
    case,
    case_rows,
    device,
):

    study_id = str(
        case["study_id"]
    )

    series_id = str(
        case["series_id"]
    )

    image, transformed_points, geometry = (
        part220b.load_case(
            study_id,
            series_id,
            case_rows,
        )
    )

    if len(case_rows) != len(
        transformed_points
    ):

        raise RuntimeError(
            f"Point mapping mismatch for "
            f"{study_id}/{series_id}."
        )

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

        image = (
            image
            .unsqueeze(0)
        )

    else:

        raise RuntimeError(
            f"Unexpected image shape: "
            f"{tuple(image.shape)}"
        )

    image = (
        image
        .float()
        .to(device)
    )

    with torch.no_grad():

        logits = model(
            image
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        logits = logits[0]

    records = []

    rfnn_mask = (
        case_rows[
            "class_name"
        ]
        == "Right Neural Foraminal Narrowing"
    )

    rfnn_indices = (
        case_rows.index[
            rfnn_mask
        ]
        .tolist()
    )

    for point_index in rfnn_indices:

        point = transformed_points[
            point_index
        ]

        z = safe_float(
            point["z"]
        )

        y = safe_float(
            point["y"]
        )

        x = safe_float(
            point["x"]
        )

        if not all(
            np.isfinite(value)
            for value in [
                z,
                y,
                x,
            ]
        ):

            continue

        record = {
            "study_id": study_id,
            "series_id": series_id,
            "point_index": int(
                point_index
            ),
            "level": str(
                case_rows.iloc[
                    point_index
                ]["level"]
            ),
            "model_z": z,
            "model_y": y,
            "model_x": x,
        }

        # ====================================================================
        # POINT
        # ====================================================================

        point_logits = point_values(
            logits,
            z,
            y,
            x,
        )

        point_probabilities = point_values(
            probabilities,
            z,
            y,
            x,
        )

        record[
            "point_rfnn_logit"
        ] = float(
            point_logits[3]
        )

        record[
            "point_rfnn_probability"
        ] = float(
            point_probabilities[3]
        )

        record[
            "point_rfnn_minus_background"
        ] = float(
            point_logits[3]
            - point_logits[0]
        )

        record[
            "point_rfnn_minus_lss"
        ] = float(
            point_logits[3]
            - point_logits[4]
        )

        record[
            "point_rfnn_minus_scs"
        ] = float(
            point_logits[3]
            - point_logits[1]
        )

        record[
            "point_rfnn_minus_lfnn"
        ] = float(
            point_logits[3]
            - point_logits[2]
        )

        record[
            "point_rfnn_minus_rss"
        ] = float(
            point_logits[3]
            - point_logits[5]
        )

        # ====================================================================
        # LOCAL RADII
        # ====================================================================

        for radius in RADII:

            max_logits = local_max_values(
                logits,
                z,
                y,
                x,
                radius,
            )

            mean_logits = local_mean_values(
                logits,
                z,
                y,
                x,
                radius,
            )

            max_probabilities = (
                local_max_values(
                    probabilities,
                    z,
                    y,
                    x,
                    radius,
                )
            )

            mean_probabilities = (
                local_mean_values(
                    probabilities,
                    z,
                    y,
                    x,
                    radius,
                )
            )

            # ---------------------------------------------------------------
            # RFNN probability
            # ---------------------------------------------------------------

            record[
                f"r{radius}_rfnn_max_probability"
            ] = float(
                max_probabilities[3]
            )

            record[
                f"r{radius}_rfnn_mean_probability"
            ] = float(
                mean_probabilities[3]
            )

            record[
                f"r{radius}_rfnn_max_logit"
            ] = float(
                max_logits[3]
            )

            record[
                f"r{radius}_rfnn_mean_logit"
            ] = float(
                mean_logits[3]
            )

            # ---------------------------------------------------------------
            # RFNN probability competitors
            # ---------------------------------------------------------------

            for class_id, name in [
                (0, "background"),
                (1, "scs"),
                (2, "lfnn"),
                (4, "lss"),
                (5, "rss"),
            ]:

                record[
                    f"r{radius}_{name}_max_probability"
                ] = float(
                    max_probabilities[
                        class_id
                    ]
                )

                record[
                    f"r{radius}_{name}_mean_probability"
                ] = float(
                    mean_probabilities[
                        class_id
                    ]
                )

            # ---------------------------------------------------------------
            # Logit margins using LOCAL MAX
            # ---------------------------------------------------------------

            record[
                f"r{radius}_rfnn_minus_background_max"
            ] = float(
                max_logits[3]
                - max_logits[0]
            )

            record[
                f"r{radius}_rfnn_minus_lss_max"
            ] = float(
                max_logits[3]
                - max_logits[4]
            )

            record[
                f"r{radius}_rfnn_minus_scs_max"
            ] = float(
                max_logits[3]
                - max_logits[1]
            )

            record[
                f"r{radius}_rfnn_minus_lfnn_max"
            ] = float(
                max_logits[3]
                - max_logits[2]
            )

            record[
                f"r{radius}_rfnn_minus_rss_max"
            ] = float(
                max_logits[3]
                - max_logits[5]
            )

            # ---------------------------------------------------------------
            # Logit margins using LOCAL MEAN
            # ---------------------------------------------------------------

            record[
                f"r{radius}_rfnn_minus_background_mean"
            ] = float(
                mean_logits[3]
                - mean_logits[0]
            )

            record[
                f"r{radius}_rfnn_minus_lss_mean"
            ] = float(
                mean_logits[3]
                - mean_logits[4]
            )

            record[
                f"r{radius}_rfnn_minus_scs_mean"
            ] = float(
                mean_logits[3]
                - mean_logits[1]
            )

            record[
                f"r{radius}_rfnn_minus_lfnn_mean"
            ] = float(
                mean_logits[3]
                - mean_logits[2]
            )

            record[
                f"r{radius}_rfnn_minus_rss_mean"
            ] = float(
                mean_logits[3]
                - mean_logits[5]
            )

        records.append(
            record
        )

    return records


# ============================================================================
# BUILD DECAY TABLE
# ============================================================================

def build_decay_table(
    records,
):

    df = pd.DataFrame(
        records
    )

    if len(df) != 45:

        raise RuntimeError(
            "Expected 45 RFNN records, "
            f"found {len(df)}."
        )

    decay_rows = []

    # ========================================================================
    # RFNN PROBABILITY
    # ========================================================================

    for metric_name, columns in {
        "RFNN probability": [
            "point_rfnn_probability",
            "r2_rfnn_max_probability",
            "r4_rfnn_max_probability",
            "r6_rfnn_max_probability",
        ],
        "RFNN logit": [
            "point_rfnn_logit",
            "r2_rfnn_max_logit",
            "r4_rfnn_max_logit",
            "r6_rfnn_max_logit",
        ],
        "RFNN-BG max logit": [
            "point_rfnn_minus_background",
            "r2_rfnn_minus_background_max",
            "r4_rfnn_minus_background_max",
            "r6_rfnn_minus_background_max",
        ],
        "RFNN-LSS max logit": [
            "point_rfnn_minus_lss",
            "r2_rfnn_minus_lss_max",
            "r4_rfnn_minus_lss_max",
            "r6_rfnn_minus_lss_max",
        ],
        "RFNN-SCS max logit": [
            "point_rfnn_minus_scs",
            "r2_rfnn_minus_scs_max",
            "r4_rfnn_minus_scs_max",
            "r6_rfnn_minus_scs_max",
        ],
        "RFNN-LFNN max logit": [
            "point_rfnn_minus_lfnn",
            "r2_rfnn_minus_lfnn_max",
            "r4_rfnn_minus_lfnn_max",
            "r6_rfnn_minus_lfnn_max",
        ],
        "RFNN-RSS max logit": [
            "point_rfnn_minus_rss",
            "r2_rfnn_minus_rss_max",
            "r4_rfnn_minus_rss_max",
            "r6_rfnn_minus_rss_max",
        ],
    }.items():

        values = [
            df[column]
            .astype(float)
            for column in columns
        ]

        point_values_array = values[0]
        r2_values = values[1]
        r4_values = values[2]
        r6_values = values[3]

        transitions = [
            (
                "Point_to_Radius2",
                point_values_array,
                r2_values,
            ),
            (
                "Radius2_to_Radius4",
                r2_values,
                r4_values,
            ),
            (
                "Radius4_to_Radius6",
                r4_values,
                r6_values,
            ),
            (
                "Point_to_Radius6",
                point_values_array,
                r6_values,
            ),
        ]

        for transition_name, start, end in (
            transitions
        ):

            delta = end - start

            decay_rows.append(
                {
                    "metric": metric_name,
                    "transition": transition_name,
                    "count": int(
                        delta.notna().sum()
                    ),
                    "mean_start": float(
                        start.mean()
                    ),
                    "mean_end": float(
                        end.mean()
                    ),
                    "mean_delta": float(
                        delta.mean()
                    ),
                    "median_delta": float(
                        delta.median()
                    ),
                    "std_delta": float(
                        delta.std()
                    ),
                    "negative_delta_fraction": float(
                        (delta < 0).mean()
                    ),
                    "positive_delta_fraction": float(
                        (delta > 0).mean()
                    ),
                }
            )

    return pd.DataFrame(
        decay_rows
    )


# ============================================================================
# LEVEL SUMMARY
# ============================================================================

def build_level_summary(
    records,
):

    df = pd.DataFrame(
        records
    )

    rows = []

    for level in LEVELS:

        subset = df[
            df["level"] == level
        ].copy()

        if subset.empty:
            continue

        row = {
            "level": level,
            "count": len(subset),
        }

        metrics = {
            "RFNN_probability_point":
                "point_rfnn_probability",

            "RFNN_probability_r2":
                "r2_rfnn_max_probability",

            "RFNN_probability_r4":
                "r4_rfnn_max_probability",

            "RFNN_probability_r6":
                "r6_rfnn_max_probability",

            "RFNN_BG_point":
                "point_rfnn_minus_background",

            "RFNN_BG_r2":
                "r2_rfnn_minus_background_max",

            "RFNN_BG_r4":
                "r4_rfnn_minus_background_max",

            "RFNN_BG_r6":
                "r6_rfnn_minus_background_max",

            "RFNN_LSS_point":
                "point_rfnn_minus_lss",

            "RFNN_LSS_r2":
                "r2_rfnn_minus_lss_max",

            "RFNN_LSS_r4":
                "r4_rfnn_minus_lss_max",

            "RFNN_LSS_r6":
                "r6_rfnn_minus_lss_max",

            "RFNN_SCS_point":
                "point_rfnn_minus_scs",

            "RFNN_SCS_r2":
                "r2_rfnn_minus_scs_max",

            "RFNN_SCS_r4":
                "r4_rfnn_minus_scs_max",

            "RFNN_SCS_r6":
                "r6_rfnn_minus_scs_max",
        }

        for output_name, column in (
            metrics.items()
        ):

            row[output_name] = float(
                subset[column].mean()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    records,
):

    print()
    print("=" * 80)
    print("BUILDING PART 2.47 SUMMARY")
    print("=" * 80)

    records_df = pd.DataFrame(
        records
    )

    if len(records_df) != 45:

        raise RuntimeError(
            "Expected exactly 45 RFNN records."
        )

    decay_df = build_decay_table(
        records
    )

    level_df = build_level_summary(
        records
    )

    # ========================================================================
    # DIRECT STAGE SUMMARY
    # ========================================================================

    stage_rows = []

    stages = [
        (
            "Point",
            "point_rfnn_probability",
            "point_rfnn_minus_background",
            "point_rfnn_minus_lss",
            "point_rfnn_minus_scs",
        ),
        (
            "Radius2",
            "r2_rfnn_max_probability",
            "r2_rfnn_minus_background_max",
            "r2_rfnn_minus_lss_max",
            "r2_rfnn_minus_scs_max",
        ),
        (
            "Radius4",
            "r4_rfnn_max_probability",
            "r4_rfnn_minus_background_max",
            "r4_rfnn_minus_lss_max",
            "r4_rfnn_minus_scs_max",
        ),
        (
            "Radius6",
            "r6_rfnn_max_probability",
            "r6_rfnn_minus_background_max",
            "r6_rfnn_minus_lss_max",
            "r6_rfnn_minus_scs_max",
        ),
    ]

    for (
        stage,
        probability_column,
        background_column,
        lss_column,
        scs_column,
    ) in stages:

        stage_rows.append(
            {
                "stage": stage,
                "count": len(records_df),
                "mean_rfnn_probability": float(
                    records_df[
                        probability_column
                    ].mean()
                ),
                "median_rfnn_probability": float(
                    records_df[
                        probability_column
                    ].median()
                ),
                "mean_rfnn_minus_background": float(
                    records_df[
                        background_column
                    ].mean()
                ),
                "positive_rfnn_minus_background_fraction": float(
                    (
                        records_df[
                            background_column
                        ]
                        > 0
                    ).mean()
                ),
                "mean_rfnn_minus_lss": float(
                    records_df[
                        lss_column
                    ].mean()
                ),
                "positive_rfnn_minus_lss_fraction": float(
                    (
                        records_df[
                            lss_column
                        ]
                        > 0
                    ).mean()
                ),
                "mean_rfnn_minus_scs": float(
                    records_df[
                        scs_column
                    ].mean()
                ),
                "positive_rfnn_minus_scs_fraction": float(
                    (
                        records_df[
                            scs_column
                        ]
                        > 0
                    ).mean()
                ),
            }
        )

    stage_df = pd.DataFrame(
        stage_rows
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    records_path = (
        OUTPUT_DIR
        / "part247_rfnn_point_neighborhood_records.csv"
    )

    stage_path = (
        OUTPUT_DIR
        / "part247_rfnn_stage_summary.csv"
    )

    decay_path = (
        OUTPUT_DIR
        / "part247_rfnn_evidence_decay_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part247_rfnn_level_evidence_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part247_rfnn_point_neighborhood_decay_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    stage_df.to_csv(
        stage_path,
        index=False,
    )

    decay_df.to_csv(
        decay_path,
        index=False,
    )

    level_df.to_csv(
        level_path,
        index=False,
    )

    json_summary = {
        "validation_cases": 25,
        "validation_points": 188,
        "rfnn_points": 45,
        "stages": {},
    }

    for _, row in stage_df.iterrows():

        json_summary[
            "stages"
        ][
            row["stage"]
        ] = {
            "mean_rfnn_probability":
                float(
                    row[
                        "mean_rfnn_probability"
                    ]
                ),
            "mean_rfnn_minus_background":
                float(
                    row[
                        "mean_rfnn_minus_background"
                    ]
                ),
            "positive_rfnn_minus_background_fraction":
                float(
                    row[
                        "positive_rfnn_minus_background_fraction"
                    ]
                ),
            "mean_rfnn_minus_lss":
                float(
                    row[
                        "mean_rfnn_minus_lss"
                    ]
                ),
            "mean_rfnn_minus_scs":
                float(
                    row[
                        "mean_rfnn_minus_scs"
                    ]
                ),
        }

    for _, row in decay_df.iterrows():

        key = (
            f"{row['metric']}__"
            f"{row['transition']}"
        )

        json_summary[
            key
        ] = {
            "mean_delta":
                float(
                    row[
                        "mean_delta"
                    ]
                ),
            "median_delta":
                float(
                    row[
                        "median_delta"
                    ]
                ),
            "negative_delta_fraction":
                float(
                    row[
                        "negative_delta_fraction"
                    ]
                ),
            "positive_delta_fraction":
                float(
                    row[
                        "positive_delta_fraction"
                    ]
                ),
        }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_summary,
            file,
            indent=2,
        )

    # ========================================================================
    # TERMINAL OUTPUT
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2.47 STAGE SUMMARY")
    print("=" * 80)

    print(
        stage_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.47 EVIDENCE DECAY")
    print("=" * 80)

    print(
        decay_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.47 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.47 COMPLETE")
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

    print(
        records_path
    )

    print(
        stage_path
    )

    print(
        decay_path
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
    print("PART 2.47")
    print("RFNN POINT-VS-NEIGHBORHOOD EVIDENCE DECAY AUDIT")
    print("=" * 80)

    print()
    print("ANALYSIS ONLY")
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

    (
        validation_cases,
        validation_points,
    ) = build_exact_validation_cohort(
        manifest
    )

    model = build_model()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device          : "
        f"{device}"
    )

    model = (
        model
        .to(device)
        .eval()
    )

    print()
    print("=" * 80)
    print("RUNNING RFNN POINT-NEIGHBORHOOD DECAY ANALYSIS")
    print("=" * 80)

    all_records = []

    for case_number, case in enumerate(
        validation_cases,
        start=1,
    ):

        study_id = str(
            case["study_id"]
        )

        series_id = str(
            case["series_id"]
        )

        case_rows = case["points"]

        if not isinstance(
            case_rows,
            pd.DataFrame,
        ):

            case_rows = pd.DataFrame(
                case_rows
            )

        case_rows = (
            case_rows
            .reset_index(drop=True)
        )

        try:

            records = analyze_case(
                model,
                case,
                case_rows,
                device,
            )

            all_records.extend(
                records
            )

            print(
                f"[{case_number:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id}"
            )

            print(
                f"  RFNN points analyzed: "
                f"{len(records)}"
            )

        except Exception as exc:

            print(
                f"[{case_number:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id} "
                f"FAILED: {exc}"
            )

    build_summary(
        all_records
    )


if __name__ == "__main__":
    main()