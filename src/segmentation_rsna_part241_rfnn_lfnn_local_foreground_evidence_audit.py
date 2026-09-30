"""
Part 2.41
RFNN vs LFNN Local Foreground Evidence Audit

Purpose
-------
Determine whether RFNN points fail because:

1. RFNN evidence is absent around the annotated point, OR
2. RFNN evidence exists nearby but misses the exact point, OR
3. the whole local RFNN probability field is suppressed.

This is an ANALYSIS-ONLY experiment.

No training.
No checkpoint modification.
No dashboard modification.

Exact validation cohort:
    Part 2.20B

Checkpoint:
    Part 2.27 best macro-disease checkpoint

Expected:
    25 validation cases
    188 annotated points
    45 LFNN points
    45 RFNN points
    45 paired LFNN/RFNN observations
"""

from __future__ import annotations

import json
import math
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
    / "rsna_part241_rfnn_lfnn_local_foreground_evidence_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT = (
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
# CONFIGURATION
# ============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BACKGROUND_CLASS = 0
LFNN_CLASS = 2
RFNN_CLASS = 3

LEVELS = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]

RADII = [2, 4, 6]


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


def clamp_index(value, maximum):
    return int(
        max(
            0,
            min(
                int(round(float(value))),
                maximum - 1,
            ),
        )
    )


def make_patch(
    tensor,
    z,
    y,
    x,
    radius,
):
    """
    Extract a cubic patch around a model-grid point.
    """

    depth, height, width = tensor.shape[-3:]

    z = clamp_index(z, depth)
    y = clamp_index(y, height)
    x = clamp_index(x, width)

    z0 = max(0, z - radius)
    z1 = min(depth, z + radius + 1)

    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)

    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)

    return tensor[
        z0:z1,
        y0:y1,
        x0:x1,
    ]


def distance_to_maximum(
    probability_volume,
    z,
    y,
    x,
    radius,
):
    """
    Distance from annotated point to the strongest probability
    voxel inside the requested local radius.
    """

    patch = make_patch(
        probability_volume,
        z,
        y,
        x,
        radius,
    )

    if patch.numel() == 0:
        return float(radius + 1)

    flat_index = int(
        torch.argmax(
            patch
        ).item()
    )

    local_coords = np.unravel_index(
        flat_index,
        tuple(
            patch.shape
        ),
    )

    depth, height, width = probability_volume.shape

    zc = clamp_index(z, depth)
    yc = clamp_index(y, height)
    xc = clamp_index(x, width)

    z0 = max(0, zc - radius)
    y0 = max(0, yc - radius)
    x0 = max(0, xc - radius)

    best_z = z0 + int(local_coords[0])
    best_y = y0 + int(local_coords[1])
    best_x = x0 + int(local_coords[2])

    return float(
        math.sqrt(
            (best_z - zc) ** 2
            + (best_y - yc) ** 2
            + (best_x - xc) ** 2
        )
    )


def distance_to_foreground(
    prediction,
    z,
    y,
    x,
    max_radius=12,
):
    """
    Distance from annotation to nearest predicted foreground voxel.
    """

    depth, height, width = prediction.shape

    zc = clamp_index(z, depth)
    yc = clamp_index(y, height)
    xc = clamp_index(x, width)

    if (
        prediction[
            zc,
            yc,
            xc,
        ].item()
        != BACKGROUND_CLASS
    ):
        return 0.0

    z0 = max(
        0,
        zc - max_radius,
    )

    z1 = min(
        depth,
        zc + max_radius + 1,
    )

    y0 = max(
        0,
        yc - max_radius,
    )

    y1 = min(
        height,
        yc + max_radius + 1,
    )

    x0 = max(
        0,
        xc - max_radius,
    )

    x1 = min(
        width,
        xc + max_radius + 1,
    )

    patch = prediction[
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    coords = torch.nonzero(
        patch != BACKGROUND_CLASS,
        as_tuple=False,
    )

    if coords.numel() == 0:
        return float(
            max_radius + 1
        )

    best = float(
        max_radius + 1
    )

    for coord in coords:

        zz = z0 + int(
            coord[0].item()
        )

        yy = y0 + int(
            coord[1].item()
        )

        xx = x0 + int(
            coord[2].item()
        )

        distance = math.sqrt(
            (zz - zc) ** 2
            + (yy - yc) ** 2
            + (xx - xc) ** 2
        )

        best = min(
            best,
            distance,
        )

    return best


# ============================================================================
# EXACT VALIDATION COHORT
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
            lambda row:
            (
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

        point_data = case["points"]

        if isinstance(
            point_data,
            pd.DataFrame,
        ):
            point_df = (
                point_data
                .reset_index(drop=True)
            )
        else:
            point_df = (
                pd.DataFrame(point_data)
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
            "Expected 188 annotated points, "
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
# MODEL
# ============================================================================

def load_model():

    print()
    print("=" * 80)
    print("LOADING PART 2.27 CHECKPOINT")
    print("=" * 80)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    model = part220b.build_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
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

    elif (
        isinstance(
            checkpoint,
            dict,
        )
        and "state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
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

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint did not load cleanly."
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: {parameter_count:,}"
    )

    print(
        f"Device          : {DEVICE}"
    )

    return model


# ============================================================================
# CASE ANALYSIS
# ============================================================================

@torch.no_grad()
def analyze_case(
    model,
    case,
):

    study_id = str(
        case["study_id"]
    )

    series_id = str(
        case["series_id"]
    )

    case_points = case[
        "points"
    ]

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
            pd.DataFrame(case_points)
            .reset_index(drop=True)
        )

    image_np, transformed_points, geometry = (
        part220b.load_case(
            study_id,
            series_id,
            point_df,
        )
    )

    image_tensor = torch.as_tensor(
        image_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    if image_tensor.ndim == 3:
        model_input = image_tensor[
            None,
            None,
        ]
    else:
        model_input = image_tensor[
            None
        ]

    output = model(
        model_input
    )

    if isinstance(
        output,
        (tuple, list),
    ):
        output = output[0]

    logits = output[0]

    probabilities = torch.softmax(
        logits,
        dim=0,
    )

    prediction = torch.argmax(
        probabilities,
        dim=0,
    )

    records = []

    for point_index, point in enumerate(
        transformed_points
    ):

        source = point_df.iloc[
            point_index
        ]

        class_id = int(
            source["class_id"]
        )

        class_name = str(
            source["class_name"]
        )

        level = str(
            source["level"]
        )

        z = clamp_index(
            point["z"],
            probabilities.shape[1],
        )

        y = clamp_index(
            point["y"],
            probabilities.shape[2],
        )

        x = clamp_index(
            point["x"],
            probabilities.shape[3],
        )

        record = {
            "study_id": study_id,
            "series_id": series_id,
            "point_index": int(
                point_index
            ),
            "class_id": class_id,
            "class_name": class_name,
            "level": level,
            "z": z,
            "y": y,
            "x": x,
        }

        # ================================================================
        # Point probability
        # ================================================================

        record[
            "point_target_probability"
        ] = float(
            probabilities[
                class_id,
                z,
                y,
                x,
            ].item()
        )

        record[
            "point_background_probability"
        ] = float(
            probabilities[
                BACKGROUND_CLASS,
                z,
                y,
                x,
            ].item()
        )

        record[
            "point_rfnn_probability"
        ] = float(
            probabilities[
                RFNN_CLASS,
                z,
                y,
                x,
            ].item()
        )

        record[
            "point_lfnn_probability"
        ] = float(
            probabilities[
                LFNN_CLASS,
                z,
                y,
                x,
            ].item()
        )

        # ================================================================
        # Local evidence
        # ================================================================

        for radius in RADII:

            target_patch = make_patch(
                probabilities[
                    class_id
                ],
                z,
                y,
                x,
                radius,
            )

            rfnn_patch = make_patch(
                probabilities[
                    RFNN_CLASS
                ],
                z,
                y,
                x,
                radius,
            )

            lfnn_patch = make_patch(
                probabilities[
                    LFNN_CLASS
                ],
                z,
                y,
                x,
                radius,
            )

            background_patch = make_patch(
                probabilities[
                    BACKGROUND_CLASS
                ],
                z,
                y,
                x,
                radius,
            )

            foreground_patch = torch.cat(
                [
                    probabilities[
                        1:2
                    ],
                    probabilities[
                        2:3
                    ],
                    probabilities[
                        3:4
                    ],
                    probabilities[
                        4:5
                    ],
                    probabilities[
                        5:6
                    ],
                ],
                dim=0,
            ).max(
                dim=0
            ).values

            prediction_patch = make_patch(
                prediction,
                z,
                y,
                x,
                radius,
            )

            prefix = (
                f"r{radius}"
            )

            record[
                f"{prefix}_target_max"
            ] = float(
                target_patch.max().item()
            )

            record[
                f"{prefix}_target_mean"
            ] = float(
                target_patch.mean().item()
            )

            record[
                f"{prefix}_rfnn_max"
            ] = float(
                rfnn_patch.max().item()
            )

            record[
                f"{prefix}_rfnn_mean"
            ] = float(
                rfnn_patch.mean().item()
            )

            record[
                f"{prefix}_lfnn_max"
            ] = float(
                lfnn_patch.max().item()
            )

            record[
                f"{prefix}_lfnn_mean"
            ] = float(
                lfnn_patch.mean().item()
            )

            record[
                f"{prefix}_background_max"
            ] = float(
                background_patch.max().item()
            )

            record[
                f"{prefix}_background_mean"
            ] = float(
                background_patch.mean().item()
            )

            record[
                f"{prefix}_foreground_max"
            ] = float(
                foreground_patch.max().item()
            )

            record[
                f"{prefix}_foreground_mean"
            ] = float(
                foreground_patch.mean().item()
            )

            record[
                f"{prefix}_predicted_foreground_ratio"
            ] = float(
                (
                    prediction_patch
                    != BACKGROUND_CLASS
                )
                .float()
                .mean()
                .item()
            )

            record[
                f"{prefix}_distance_to_target_max"
            ] = distance_to_maximum(
                target_patch,
                radius,
                radius,
                radius,
                radius,
            )

            record[
                f"{prefix}_distance_to_rfnn_max"
            ] = distance_to_maximum(
                rfnn_patch,
                radius,
                radius,
                radius,
                radius,
            )

        record[
            "distance_to_predicted_foreground"
        ] = distance_to_foreground(
            prediction,
            z,
            y,
            x,
            max_radius=12,
        )

        record[
            "predicted_class"
        ] = int(
            prediction[
                z,
                y,
                x,
            ].item()
        )

        records.append(
            record
        )

    return records


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    records_df,
):

    print()
    print("=" * 80)
    print("BUILDING PART 2.41 SUMMARY")
    print("=" * 80)

    rfnn = records_df[
        records_df["class_name"]
        == "Right Neural Foraminal Narrowing"
    ].copy()

    lfnn = records_df[
        records_df["class_name"]
        == "Left Neural Foraminal Narrowing"
    ].copy()

    if len(rfnn) != 45:
        raise RuntimeError(
            f"Expected 45 RFNN records, "
            f"found {len(rfnn)}."
        )

    if len(lfnn) != 45:
        raise RuntimeError(
            f"Expected 45 LFNN records, "
            f"found {len(lfnn)}."
        )

    # ------------------------------------------------------------------------
    # Disease summary
    # ------------------------------------------------------------------------

    disease_summary = (
        records_df
        .groupby(
            "class_name",
            dropna=False,
        )
        .agg(
            count=(
                "point_index",
                "count",
            ),
            mean_point_probability=(
                "point_target_probability",
                "mean",
            ),
            mean_r2_target_max=(
                "r2_target_max",
                "mean",
            ),
            mean_r4_target_max=(
                "r4_target_max",
                "mean",
            ),
            mean_r6_target_max=(
                "r6_target_max",
                "mean",
            ),
            mean_r2_foreground_max=(
                "r2_foreground_max",
                "mean",
            ),
            mean_r4_foreground_max=(
                "r4_foreground_max",
                "mean",
            ),
            mean_r6_foreground_max=(
                "r6_foreground_max",
                "mean",
            ),
            mean_distance_to_foreground=(
                "distance_to_predicted_foreground",
                "mean",
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------------
    # RFNN level summary
    # ------------------------------------------------------------------------

    level_summary = (
        rfnn
        .groupby(
            "level",
            dropna=False,
        )
        .agg(
            count=(
                "point_index",
                "count",
            ),
            mean_point_rfnn_probability=(
                "point_rfnn_probability",
                "mean",
            ),
            mean_r2_rfnn_max=(
                "r2_rfnn_max",
                "mean",
            ),
            mean_r4_rfnn_max=(
                "r4_rfnn_max",
                "mean",
            ),
            mean_r6_rfnn_max=(
                "r6_rfnn_max",
                "mean",
            ),
            mean_r2_foreground_max=(
                "r2_foreground_max",
                "mean",
            ),
            mean_r4_foreground_max=(
                "r4_foreground_max",
                "mean",
            ),
            mean_r6_foreground_max=(
                "r6_foreground_max",
                "mean",
            ),
            mean_distance_to_foreground=(
                "distance_to_predicted_foreground",
                "mean",
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------------
    # Paired LFNN/RFNN
    # ------------------------------------------------------------------------

    paired = (
        rfnn[
            [
                "study_id",
                "series_id",
                "level",
                "point_rfnn_probability",
                "point_background_probability",
                "r2_rfnn_max",
                "r4_rfnn_max",
                "r6_rfnn_max",
                "r2_foreground_max",
                "r4_foreground_max",
                "r6_foreground_max",
                "distance_to_predicted_foreground",
            ]
        ]
        .rename(
            columns={
                "point_rfnn_probability":
                    "rfnn_point_probability",
                "point_background_probability":
                    "rfnn_background_probability",
                "r2_rfnn_max":
                    "rfnn_r2_max",
                "r4_rfnn_max":
                    "rfnn_r4_max",
                "r6_rfnn_max":
                    "rfnn_r6_max",
                "r2_foreground_max":
                    "rfnn_r2_foreground_max",
                "r4_foreground_max":
                    "rfnn_r4_foreground_max",
                "r6_foreground_max":
                    "rfnn_r6_foreground_max",
                "distance_to_predicted_foreground":
                    "rfnn_distance_to_foreground",
            }
        )
        .merge(
            lfnn[
                [
                    "study_id",
                    "series_id",
                    "level",
                    "point_lfnn_probability",
                    "point_background_probability",
                    "r2_lfnn_max",
                    "r4_lfnn_max",
                    "r6_lfnn_max",
                    "r2_foreground_max",
                    "r4_foreground_max",
                    "r6_foreground_max",
                    "distance_to_predicted_foreground",
                ]
            ].rename(
                columns={
                    "point_lfnn_probability":
                        "lfnn_point_probability",
                    "point_background_probability":
                        "lfnn_background_probability",
                    "r2_lfnn_max":
                        "lfnn_r2_max",
                    "r4_lfnn_max":
                        "lfnn_r4_max",
                    "r6_lfnn_max":
                        "lfnn_r6_max",
                    "r2_foreground_max":
                        "lfnn_r2_foreground_max",
                    "r4_foreground_max":
                        "lfnn_r4_foreground_max",
                    "r6_foreground_max":
                        "lfnn_r6_foreground_max",
                    "distance_to_predicted_foreground":
                        "lfnn_distance_to_foreground",
                }
            ),
            on=[
                "study_id",
                "series_id",
                "level",
            ],
            how="inner",
        )
    )

    if len(paired) != 45:
        raise RuntimeError(
            "Expected 45 paired LFNN/RFNN "
            f"observations, found {len(paired)}."
        )

    # ------------------------------------------------------------------------
    # Differences
    # ------------------------------------------------------------------------

    for radius in RADII:

        paired[
            f"rfnn_minus_lfnn_r{radius}_max"
        ] = (
            paired[
                f"rfnn_r{radius}_max"
            ]
            -
            paired[
                f"lfnn_r{radius}_max"
            ]
        )

        paired[
            f"rfnn_minus_lfnn_r{radius}_foreground_max"
        ] = (
            paired[
                f"rfnn_r{radius}_foreground_max"
            ]
            -
            paired[
                f"lfnn_r{radius}_foreground_max"
            ]
        )

    paired[
        "rfnn_minus_lfnn_point_probability"
    ] = (
        paired[
            "rfnn_point_probability"
        ]
        -
        paired[
            "lfnn_point_probability"
        ]
    )

    paired[
        "rfnn_minus_lfnn_distance"
    ] = (
        paired[
            "rfnn_distance_to_foreground"
        ]
        -
        paired[
            "lfnn_distance_to_foreground"
        ]
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    records_path = (
        OUTPUT_DIR
        / "part241_all_local_evidence_records.csv"
    )

    disease_path = (
        OUTPUT_DIR
        / "part241_disease_local_evidence_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part241_rfnn_level_local_evidence_summary.csv"
    )

    paired_path = (
        OUTPUT_DIR
        / "part241_paired_lfnn_rfnn_local_evidence.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part241_local_evidence_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    disease_summary.to_csv(
        disease_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    summary = {
        "validation_cases": 25,
        "validation_points": 188,
        "rfnn_points": 45,
        "lfnn_points": 45,
        "paired_observations": 45,
        "rfnn_point_probability_mean": safe_float(
            rfnn[
                "point_rfnn_probability"
            ].mean()
        ),
        "rfnn_r2_max_mean": safe_float(
            rfnn[
                "r2_rfnn_max"
            ].mean()
        ),
        "rfnn_r4_max_mean": safe_float(
            rfnn[
                "r4_rfnn_max"
            ].mean()
        ),
        "rfnn_r6_max_mean": safe_float(
            rfnn[
                "r6_rfnn_max"
            ].mean()
        ),
        "rfnn_r2_foreground_max_mean": safe_float(
            rfnn[
                "r2_foreground_max"
            ].mean()
        ),
        "rfnn_r4_foreground_max_mean": safe_float(
            rfnn[
                "r4_foreground_max"
            ].mean()
        ),
        "rfnn_r6_foreground_max_mean": safe_float(
            rfnn[
                "r6_foreground_max"
            ].mean()
        ),
        "rfnn_distance_to_foreground_mean": safe_float(
            rfnn[
                "distance_to_predicted_foreground"
            ].mean()
        ),
        "paired_r2_max_difference_mean": safe_float(
            paired[
                "rfnn_minus_lfnn_r2_max"
            ].mean()
        ),
        "paired_r4_max_difference_mean": safe_float(
            paired[
                "rfnn_minus_lfnn_r4_max"
            ].mean()
        ),
        "paired_r6_max_difference_mean": safe_float(
            paired[
                "rfnn_minus_lfnn_r6_max"
            ].mean()
        ),
        "paired_point_probability_difference_mean": safe_float(
            paired[
                "rfnn_minus_lfnn_point_probability"
            ].mean()
        ),
        "paired_distance_difference_mean": safe_float(
            paired[
                "rfnn_minus_lfnn_distance"
            ].mean()
        ),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PART 2.41 GLOBAL RFNN LOCAL EVIDENCE")
    print("=" * 80)

    print(
        f"RFNN points                 : {len(rfnn)}"
    )

    print(
        f"RFNN point probability      : "
        f"{rfnn['point_rfnn_probability'].mean():.6f}"
    )

    print(
        f"RFNN radius-2 maximum       : "
        f"{rfnn['r2_rfnn_max'].mean():.6f}"
    )

    print(
        f"RFNN radius-4 maximum       : "
        f"{rfnn['r4_rfnn_max'].mean():.6f}"
    )

    print(
        f"RFNN radius-6 maximum       : "
        f"{rfnn['r6_rfnn_max'].mean():.6f}"
    )

    print(
        f"RFNN radius-2 FG maximum    : "
        f"{rfnn['r2_foreground_max'].mean():.6f}"
    )

    print(
        f"RFNN radius-4 FG maximum    : "
        f"{rfnn['r4_foreground_max'].mean():.6f}"
    )

    print(
        f"RFNN radius-6 FG maximum    : "
        f"{rfnn['r6_foreground_max'].mean():.6f}"
    )

    print(
        f"Distance to predicted FG    : "
        f"{rfnn['distance_to_predicted_foreground'].mean():.6f}"
    )

    print()
    print("=" * 80)
    print("PAIRED LFNN vs RFNN LOCAL EVIDENCE")
    print("=" * 80)

    print(
        f"Paired observations         : "
        f"{len(paired)}"
    )

    print(
        f"Point probability diff      : "
        f"{paired['rfnn_minus_lfnn_point_probability'].mean():.6f}"
    )

    print(
        f"Radius-2 maximum diff       : "
        f"{paired['rfnn_minus_lfnn_r2_max'].mean():.6f}"
    )

    print(
        f"Radius-4 maximum diff       : "
        f"{paired['rfnn_minus_lfnn_r4_max'].mean():.6f}"
    )

    print(
        f"Radius-6 maximum diff       : "
        f"{paired['rfnn_minus_lfnn_r6_max'].mean():.6f}"
    )

    print(
        f"Foreground radius-2 diff    : "
        f"{paired['rfnn_minus_lfnn_r2_foreground_max'].mean():.6f}"
    )

    print(
        f"Foreground radius-4 diff    : "
        f"{paired['rfnn_minus_lfnn_r4_foreground_max'].mean():.6f}"
    )

    print(
        f"Foreground radius-6 diff    : "
        f"{paired['rfnn_minus_lfnn_r6_foreground_max'].mean():.6f}"
    )

    print(
        f"Distance-to-FG difference   : "
        f"{paired['rfnn_minus_lfnn_distance'].mean():.6f}"
    )

    print()
    print("=" * 80)
    print("RFNN LEVEL ANALYSIS")
    print("=" * 80)

    print(
        level_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.41 COMPLETE")
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
        disease_path
    )

    print(
        level_path
    )

    print(
        paired_path
    )

    print(
        summary_path
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.41")
    print("RFNN vs LFNN LOCAL FOREGROUND EVIDENCE AUDIT")
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
        f"Full manifest rows: {len(manifest)}"
    )

    (
        validation_cases,
        validation_points,
    ) = build_exact_validation_cohort(
        manifest
    )

    model = load_model()

    print()
    print("=" * 80)
    print("RUNNING LOCAL FOREGROUND EVIDENCE ANALYSIS")
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

        print()
        print(
            f"[{case_number:02d}/25] "
            f"Study={study_id} | "
            f"Series={series_id}"
        )

        records = analyze_case(
            model,
            case,
        )

        all_records.extend(
            records
        )

        print(
            f"  Analysis successful."
        )

        print(
            f"  Annotated points: "
            f"{len(records)}"
        )

    records_df = pd.DataFrame(
        all_records
    )

    if len(records_df) != 188:
        raise RuntimeError(
            "Expected 188 records, "
            f"found {len(records_df)}."
        )

    build_summary(
        records_df
    )


if __name__ == "__main__":
    main()