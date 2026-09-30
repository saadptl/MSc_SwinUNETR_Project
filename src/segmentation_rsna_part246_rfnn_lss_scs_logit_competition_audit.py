"""
PART 2.46
RFNN-LSS-SCS LOGIT COMPETITION AUDIT

Purpose
-------
Quantify class-logit competition around annotated RFNN points.

For every RFNN annotation, measure model logits/probabilities at:

    1. RFNN point
    2. local radius 2
    3. local radius 4
    4. local radius 6

Classes:

    Background
    Spinal Canal Stenosis (SCS)
    Left Neural Foraminal Narrowing (LFNN)
    Right Neural Foraminal Narrowing (RFNN)
    Left Subarticular Stenosis (LSS)
    Right Subarticular Stenosis (RSS)

Main comparisons:

    RFNN - Background
    RFNN - LSS
    RFNN - SCS
    RFNN - LFNN
    RFNN - RSS

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
    / "rsna_part246_rfnn_lss_scs_logit_competition_audit"
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

CLASS_CODES = {
    "Background": 0,
    "SCS": 1,
    "LFNN": 2,
    "RFNN": 3,
    "LSS": 4,
    "RSS": 5,
}

RADII = [
    2,
    4,
    6,
]

PROFILE_POSITIONS = [
    "RFNN",
    "P25",
    "MID",
    "P75",
    "RFNN_OPPOSITE",
]


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(
    value,
):
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


def local_cube_values(
    tensor,
    z,
    y,
    x,
    radius,
):
    """
    Extract a local cube from a tensor with shape:

        [C, Z, Y, X]

    Returns:

        [C, local_Z, local_Y, local_X]
    """

    _, depth, height, width = tensor.shape

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


def local_probability_summary(
    probabilities,
    z,
    y,
    x,
    radius,
):
    """
    Calculate local class maxima and mean probabilities.
    """

    patch = local_cube_values(
        probabilities,
        z,
        y,
        x,
        radius,
    )

    patch_flat = patch.reshape(
        patch.shape[0],
        -1,
    )

    max_values = patch_flat.max(
        dim=1
    ).values

    mean_values = patch_flat.mean(
        dim=1
    )

    return (
        max_values.detach().cpu().numpy(),
        mean_values.detach().cpu().numpy(),
    )


def point_probabilities(
    probabilities,
    z,
    y,
    x,
):
    """
    Get six class probabilities at a point.
    """

    _, depth, height, width = (
        probabilities.shape
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
        probabilities[
            :,
            z,
            y,
            x,
        ]
        .detach()
        .cpu()
        .numpy()
    )


def point_logits(
    logits,
    z,
    y,
    x,
):
    """
    Get six class logits at a point.
    """

    _, depth, height, width = (
        logits.shape
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
        logits[
            :,
            z,
            y,
            x,
        ]
        .detach()
        .cpu()
        .numpy()
    )


def build_model():
    """
    Construct the exact canonical Part 2.27 SwinUNETR model.
    """

    model = part220b.build_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    if isinstance(
        checkpoint,
        dict,
    ) and "model_state_dict" in checkpoint:

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

    print()
    print("=" * 80)
    print("LOADING PART 2.27 CHECKPOINT")
    print("=" * 80)

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
        print(
            "WARNING: missing checkpoint keys."
        )

    if unexpected:
        print(
            "WARNING: unexpected checkpoint keys."
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
                    "native_z": safe_float(
                        row["native_z"]
                    ),
                    "native_y": safe_float(
                        row["native_y"]
                    ),
                    "native_x": safe_float(
                        row["native_x"]
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
# SINGLE CASE ANALYSIS
# ============================================================================

def analyze_case(
    model,
    case,
    case_rows,
    device,
):
    """
    Run inference once and analyze all RFNN points in the case.
    """

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

    image_tensor = image

    if not torch.is_tensor(
        image_tensor
    ):

        image_tensor = torch.tensor(
            image_tensor,
            dtype=torch.float32,
        )

    if image_tensor.ndim == 3:

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .unsqueeze(0)
        )

    elif image_tensor.ndim == 4:

        image_tensor = (
            image_tensor
            .unsqueeze(0)
        )

    else:

        raise RuntimeError(
            f"Unexpected image shape: "
            f"{tuple(image_tensor.shape)}"
        )

    image_tensor = (
        image_tensor
        .float()
        .to(device)
    )

    with torch.no_grad():

        logits = model(
            image_tensor
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

        point_position = int(
            point_index
        )

        point = transformed_points[
            point_position
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

        point_logit_values = point_logits(
            logits,
            z,
            y,
            x,
        )

        point_probability_values = (
            point_probabilities(
                probabilities,
                z,
                y,
                x,
            )
        )

        # ------------------------------------------------------------
        # Point-level margins
        # ------------------------------------------------------------

        point_record = {
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

        for class_id, class_name in (
            CLASS_NAMES.items()
        ):

            point_record[
                f"point_logit_{class_name}"
            ] = float(
                point_logit_values[
                    class_id
                ]
            )

            point_record[
                f"point_probability_{class_name}"
            ] = float(
                point_probability_values[
                    class_id
                ]
            )

        point_record[
            "point_rfnn_minus_background"
        ] = float(
            point_logit_values[3]
            - point_logit_values[0]
        )

        point_record[
            "point_rfnn_minus_lss"
        ] = float(
            point_logit_values[3]
            - point_logit_values[4]
        )

        point_record[
            "point_rfnn_minus_scs"
        ] = float(
            point_logit_values[3]
            - point_logit_values[1]
        )

        point_record[
            "point_rfnn_minus_lfnn"
        ] = float(
            point_logit_values[3]
            - point_logit_values[2]
        )

        point_record[
            "point_rfnn_minus_rss"
        ] = float(
            point_logit_values[3]
            - point_logit_values[5]
        )

        # ------------------------------------------------------------
        # Local neighborhoods
        # ------------------------------------------------------------

        for radius in RADII:

            max_values, mean_values = (
                local_probability_summary(
                    probabilities,
                    z,
                    y,
                    x,
                    radius,
                )
            )

            local_logits = (
                local_cube_values(
                    logits,
                    z,
                    y,
                    x,
                    radius,
                )
            )

            local_logits_flat = (
                local_logits
                .reshape(
                    local_logits.shape[0],
                    -1,
                )
            )

            max_logits = (
                local_logits_flat.max(
                    dim=1
                ).values
                .detach()
                .cpu()
                .numpy()
            )

            mean_logits = (
                local_logits_flat.mean(
                    dim=1
                ).detach()
                .cpu()
                .numpy()
            )

            for class_id, class_name in (
                CLASS_NAMES.items()
            ):

                point_record[
                    f"r{radius}_max_probability_{class_name}"
                ] = float(
                    max_values[
                        class_id
                    ]
                )

                point_record[
                    f"r{radius}_mean_probability_{class_name}"
                ] = float(
                    mean_values[
                        class_id
                    ]
                )

                point_record[
                    f"r{radius}_max_logit_{class_name}"
                ] = float(
                    max_logits[
                        class_id
                    ]
                )

                point_record[
                    f"r{radius}_mean_logit_{class_name}"
                ] = float(
                    mean_logits[
                        class_id
                    ]
                )

            # --------------------------------------------------------
            # Local max-logit competition
            # --------------------------------------------------------

            point_record[
                f"r{radius}_max_rfnn_minus_background"
            ] = float(
                max_logits[3]
                - max_logits[0]
            )

            point_record[
                f"r{radius}_max_rfnn_minus_lss"
            ] = float(
                max_logits[3]
                - max_logits[4]
            )

            point_record[
                f"r{radius}_max_rfnn_minus_scs"
            ] = float(
                max_logits[3]
                - max_logits[1]
            )

            point_record[
                f"r{radius}_max_rfnn_minus_lfnn"
            ] = float(
                max_logits[3]
                - max_logits[2]
            )

            point_record[
                f"r{radius}_max_rfnn_minus_rss"
            ] = float(
                max_logits[3]
                - max_logits[5]
            )

            # --------------------------------------------------------
            # Local mean-logit competition
            # --------------------------------------------------------

            point_record[
                f"r{radius}_mean_rfnn_minus_background"
            ] = float(
                mean_logits[3]
                - mean_logits[0]
            )

            point_record[
                f"r{radius}_mean_rfnn_minus_lss"
            ] = float(
                mean_logits[3]
                - mean_logits[4]
            )

            point_record[
                f"r{radius}_mean_rfnn_minus_scs"
            ] = float(
                mean_logits[3]
                - mean_logits[1]
            )

            point_record[
                f"r{radius}_mean_rfnn_minus_lfnn"
            ] = float(
                mean_logits[3]
                - mean_logits[2]
            )

            point_record[
                f"r{radius}_mean_rfnn_minus_rss"
            ] = float(
                mean_logits[3]
                - mean_logits[5]
            )

        records.append(
            point_record
        )

    return records


# ============================================================================
# SUMMARY BUILDING
# ============================================================================

def build_summaries(
    records,
):
    print()
    print("=" * 80)
    print("BUILDING PART 2.46 SUMMARY")
    print("=" * 80)

    df = pd.DataFrame(
        records
    )

    if len(df) != 45:

        raise RuntimeError(
            "Expected 45 RFNN records, "
            f"found {len(df)}."
        )

    # ========================================================================
    # POINT SUMMARY
    # ========================================================================

    point_rows = []

    point_margin_columns = [
        "point_rfnn_minus_background",
        "point_rfnn_minus_lss",
        "point_rfnn_minus_scs",
        "point_rfnn_minus_lfnn",
        "point_rfnn_minus_rss",
    ]

    for column in point_margin_columns:

        values = df[
            column
        ].astype(float)

        point_rows.append(
            {
                "scope": "RFNN_POINT",
                "comparison": column,
                "count": int(
                    values.notna().sum()
                ),
                "mean": float(
                    values.mean()
                ),
                "median": float(
                    values.median()
                ),
                "std": float(
                    values.std()
                ),
                "min": float(
                    values.min()
                ),
                "max": float(
                    values.max()
                ),
                "positive_fraction": float(
                    (values > 0).mean()
                ),
                "negative_fraction": float(
                    (values < 0).mean()
                ),
            }
        )

    point_summary = pd.DataFrame(
        point_rows
    )

    # ========================================================================
    # RADIUS SUMMARY
    # ========================================================================

    radius_rows = []

    comparisons = {
        "background": "background",
        "lss": "lss",
        "scs": "scs",
        "lfnn": "lfnn",
        "rss": "rss",
    }

    for radius in RADII:

        for comparison_code, comparison_name in (
            comparisons.items()
        ):

            max_column = (
                f"r{radius}_max_rfnn_minus_"
                f"{comparison_name}"
            )

            mean_column = (
                f"r{radius}_mean_rfnn_minus_"
                f"{comparison_name}"
            )

            max_values = df[
                max_column
            ].astype(float)

            mean_values = df[
                mean_column
            ].astype(float)

            radius_rows.append(
                {
                    "radius": radius,
                    "comparison": comparison_name,
                    "max_margin_mean": float(
                        max_values.mean()
                    ),
                    "max_margin_median": float(
                        max_values.median()
                    ),
                    "max_margin_std": float(
                        max_values.std()
                    ),
                    "max_margin_positive_fraction": float(
                        (max_values > 0).mean()
                    ),
                    "mean_margin_mean": float(
                        mean_values.mean()
                    ),
                    "mean_margin_median": float(
                        mean_values.median()
                    ),
                    "mean_margin_std": float(
                        mean_values.std()
                    ),
                    "mean_margin_positive_fraction": float(
                        (mean_values > 0).mean()
                    ),
                }
            )

    radius_summary = pd.DataFrame(
        radius_rows
    )

    # ========================================================================
    # LEVEL SUMMARY
    # ========================================================================

    level_rows = []

    for level in [
        "L1/L2",
        "L2/L3",
        "L3/L4",
        "L4/L5",
        "L5/S1",
    ]:

        subset = df[
            df["level"] == level
        ].copy()

        if subset.empty:
            continue

        row = {
            "level": level,
            "count": len(subset),
        }

        for comparison in [
            "background",
            "lss",
            "scs",
            "lfnn",
            "rss",
        ]:

            column = (
                f"point_rfnn_minus_"
                f"{comparison}"
            )

            row[
                f"mean_rfnn_minus_{comparison}"
            ] = float(
                subset[column].mean()
            )

            row[
                f"positive_fraction_{comparison}"
            ] = float(
                (
                    subset[column] > 0
                ).mean()
            )

        level_rows.append(
            row
        )

    level_summary = pd.DataFrame(
        level_rows
    )

    # ========================================================================
    # LOCAL WINNER ANALYSIS
    # ========================================================================

    winner_rows = []

    for radius in RADII:

        for _, row in df.iterrows():

            class_prob_columns = {
                class_id: (
                    f"r{radius}_max_probability_"
                    f"{class_name}"
                )
                for class_id, class_name in (
                    CLASS_NAMES.items()
                )
            }

            values = {
                class_id: float(
                    row[column]
                )
                for class_id, column in (
                    class_prob_columns.items()
                )
            }

            winner_id = max(
                values,
                key=values.get,
            )

            winner_rows.append(
                {
                    "study_id": row[
                        "study_id"
                    ],
                    "series_id": row[
                        "series_id"
                    ],
                    "point_index": int(
                        row["point_index"]
                    ),
                    "level": row[
                        "level"
                    ],
                    "radius": radius,
                    "winner_class": CLASS_NAMES[
                        winner_id
                    ],
                    "winner_probability": values[
                        winner_id
                    ],
                    "rfnn_probability": values[
                        3
                    ],
                    "lss_probability": values[
                        4
                    ],
                    "scs_probability": values[
                        1
                    ],
                    "background_probability": values[
                        0
                    ],
                }
            )

    winner_df = pd.DataFrame(
        winner_rows
    )

    winner_summary = (
        winner_df
        .groupby(
            [
                "radius",
                "winner_class",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    records_path = (
        OUTPUT_DIR
        / "part246_rfnn_logit_competition_records.csv"
    )

    point_path = (
        OUTPUT_DIR
        / "part246_point_margin_summary.csv"
    )

    radius_path = (
        OUTPUT_DIR
        / "part246_radius_margin_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part246_level_margin_summary.csv"
    )

    winner_records_path = (
        OUTPUT_DIR
        / "part246_local_winner_records.csv"
    )

    winner_summary_path = (
        OUTPUT_DIR
        / "part246_local_winner_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part246_rfnn_logit_competition_audit_summary.json"
    )

    df.to_csv(
        records_path,
        index=False,
    )

    point_summary.to_csv(
        point_path,
        index=False,
    )

    radius_summary.to_csv(
        radius_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    winner_df.to_csv(
        winner_records_path,
        index=False,
    )

    winner_summary.to_csv(
        winner_summary_path,
        index=False,
    )

    json_summary = {
        "validation_cases": 25,
        "validation_points": 188,
        "rfnn_points": 45,
    }

    for comparison in [
        "background",
        "lss",
        "scs",
        "lfnn",
        "rss",
    ]:

        column = (
            f"point_rfnn_minus_"
            f"{comparison}"
        )

        values = df[
            column
        ].astype(float)

        json_summary[
            f"point_rfnn_minus_{comparison}"
        ] = {
            "mean": float(
                values.mean()
            ),
            "median": float(
                values.median()
            ),
            "positive_fraction": float(
                (values > 0).mean()
            ),
            "negative_fraction": float(
                (values < 0).mean()
            ),
        }

    for radius in RADII:

        radius_subset = radius_summary[
            radius_summary[
                "radius"
            ]
            == radius
        ]

        json_summary[
            f"radius_{radius}"
        ] = {}

        for _, row in (
            radius_subset.iterrows()
        ):

            comparison = row[
                "comparison"
            ]

            json_summary[
                f"radius_{radius}"
            ][
                comparison
            ] = {
                "max_margin_mean": float(
                    row[
                        "max_margin_mean"
                    ]
                ),
                "max_margin_positive_fraction": float(
                    row[
                        "max_margin_positive_fraction"
                    ]
                ),
                "mean_margin_mean": float(
                    row[
                        "mean_margin_mean"
                    ]
                ),
                "mean_margin_positive_fraction": float(
                    row[
                        "mean_margin_positive_fraction"
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
    print("PART 2.46 POINT-LEVEL RFNN LOGIT MARGINS")
    print("=" * 80)

    print(
        point_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.46 LOCAL RFNN LOGIT COMPETITION")
    print("=" * 80)

    print(
        radius_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.46 LEVEL ANALYSIS")
    print("=" * 80)

    print(
        level_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.46 LOCAL WINNERS")
    print("=" * 80)

    print(
        winner_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.46 COMPLETE")
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
        point_path
    )

    print(
        radius_path
    )

    print(
        level_path
    )

    print(
        winner_records_path
    )

    print(
        winner_summary_path
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
    print("PART 2.46")
    print("RFNN-LSS-SCS LOGIT COMPETITION AUDIT")
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
    print("RUNNING RFNN LOGIT COMPETITION ANALYSIS")
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

    build_summaries(
        all_records
    )


if __name__ == "__main__":
    main()