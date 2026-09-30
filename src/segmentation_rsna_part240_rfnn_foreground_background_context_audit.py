"""
Part 2.40
RFNN Foreground-vs-Background Context Diagnostic Audit

Purpose
-------
Investigate why the Part 2.27 model systematically suppresses
Right Neural Foraminal Narrowing (RFNN) points toward background.

This is an ANALYSIS-ONLY experiment.

It does NOT:
    - train the model
    - modify any checkpoint
    - modify the dashboard
    - create voxel ground truth
    - claim that point supervision is manual segmentation

Validation cohort:
    EXACT Part 2.20B validation cohort

Checkpoint:
    Part 2.27 best macro-disease checkpoint

Expected validation:
    25 studies
    25 series
    188 annotated points
    45 RFNN points
    45 LFNN points
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part240_rfnn_foreground_background_context_audit"
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


# ============================================================================
# IMPORT EXACT PART 2.20B PIPELINE
# ============================================================================

import segmentation_rsna_part220b_geometry_corrected_training as part220b


# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_SHAPE = (
    64,
    96,
    96,
)

RFNN_CLASS = 3
LFNN_CLASS = 2
BACKGROUND_CLASS = 0

LEVEL_ORDER = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]

# Neighborhood radii in model voxels.
NEIGHBORHOOD_RADII = [
    2,
    4,
    6,
]

EPS = 1e-8


# ============================================================================
# SAFE HELPERS
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


def normalize_vector(values):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if maximum - minimum < EPS:
        return np.zeros_like(values)

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# ============================================================================
# EXACT VALIDATION COHORT
# ============================================================================

def build_exact_validation_cohort(manifest):
    """
    Reproduce the exact Part 2.20B validation cohort.

    Returns
    -------
    validation_cases
        List returned by Part 2.20B build_case_index().
    validation_points
        DataFrame containing the original annotated point rows.
    """

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

    print(
        f"Validation series selected: "
        f"{len(selected)}"
    )

    if len(selected) != 25:
        raise RuntimeError(
            "Expected exactly 25 validation "
            f"series, found {len(selected)}."
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
                pd.DataFrame(case_points)
                .reset_index(drop=True)
            )

        for point_index, (_, row) in enumerate(
            point_df.iterrows()
        ):

            records.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "point_index": int(point_index),
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

    print(
        f"Validation cases: "
        f"{len(validation_cases)}"
    )

    print(
        f"Validation point rows: "
        f"{len(validation_points)}"
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
                f"Validation cohort mismatch "
                f"for {disease}: "
                f"expected {expected}, "
                f"found {actual}."
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
# MODEL LOADING
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

    if isinstance(
        checkpoint,
        dict,
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

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
        f"Checkpoint:\n{CHECKPOINT}"
    )

    print(
        f"Missing keys    : {len(missing)}"
    )

    print(
        f"Unexpected keys : {len(unexpected)}"
    )

    if missing:
        print(missing[:10])

    if unexpected:
        print(unexpected[:10])

    if len(missing) != 0 or len(unexpected) != 0:
        raise RuntimeError(
            "Checkpoint did not load cleanly."
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Model parameters: {parameter_count:,}"
    )

    print(
        f"Device          : {DEVICE}"
    )

    return model


# ============================================================================
# POINT EXTRACTION
# ============================================================================

def get_transformed_points(
    study_id,
    series_id,
    point_df,
):
    """
    Run the exact Part 2.20B geometry transformation.
    """

    image, transformed_points, geometry = (
        part220b.load_case(
            study_id,
            series_id,
            point_df,
        )
    )

    return (
        image,
        transformed_points,
        geometry,
    )


# ============================================================================
# LOCAL NEIGHBORHOOD STATISTICS
# ============================================================================

def extract_neighborhood(
    tensor,
    z,
    y,
    x,
    radius,
):
    """
    Extract a cubic neighborhood around a point.

    tensor:
        [D,H,W]
    """

    depth, height, width = (
        tensor.shape
    )

    zc = clamp_index(
        z,
        depth,
    )

    yc = clamp_index(
        y,
        height,
    )

    xc = clamp_index(
        x,
        width,
    )

    z0 = max(
        0,
        zc - radius,
    )

    z1 = min(
        depth,
        zc + radius + 1,
    )

    y0 = max(
        0,
        yc - radius,
    )

    y1 = min(
        height,
        yc + radius + 1,
    )

    x0 = max(
        0,
        xc - radius,
    )

    x1 = min(
        width,
        xc + radius + 1,
    )

    return tensor[
        z0:z1,
        y0:y1,
        x0:x1,
    ]


def gradient_magnitude(
    image,
):
    """
    Simple 3D finite-difference gradient magnitude.
    """

    dz = torch.zeros_like(
        image
    )

    dy = torch.zeros_like(
        image
    )

    dx = torch.zeros_like(
        image
    )

    dz[1:-1] = (
        image[2:]
        - image[:-2]
    ) * 0.5

    dy[:, 1:-1] = (
        image[:, 2:]
        - image[:, :-2]
    ) * 0.5

    dx[:, :, 1:-1] = (
        image[:, :, 2:]
        - image[:, :, :-2]
    ) * 0.5

    return torch.sqrt(
        dz * dz
        + dy * dy
        + dx * dx
        + 1e-8
    )


# ============================================================================
# DISTANCE TO PREDICTED FOREGROUND
# ============================================================================

def distance_to_foreground(
    prediction,
    point,
    max_radius=12,
):
    """
    Search outward from a point for predicted foreground.

    Returns the minimum Euclidean voxel distance
    to any non-background prediction.

    If none is found within max_radius,
    returns max_radius + 1.
    """

    z, y, x = point

    depth, height, width = (
        prediction.shape
    )

    z = clamp_index(
        z,
        depth,
    )

    y = clamp_index(
        y,
        height,
    )

    x = clamp_index(
        x,
        width,
    )

    if prediction[
        z,
        y,
        x
    ] != BACKGROUND_CLASS:

        return 0.0

    best = (
        max_radius
        + 1.0
    )

    z0 = max(
        0,
        z - max_radius,
    )

    z1 = min(
        depth,
        z + max_radius + 1,
    )

    y0 = max(
        0,
        y - max_radius,
    )

    y1 = min(
        height,
        y + max_radius + 1,
    )

    x0 = max(
        0,
        x - max_radius,
    )

    x1 = min(
        width,
        x + max_radius + 1,
    )

    coords = torch.nonzero(
        prediction[
            z0:z1,
            y0:y1,
            x0:x1
        ] != BACKGROUND_CLASS,
        as_tuple=False,
    )

    for coord in coords:

        zz = int(
            coord[0]
        ) + z0

        yy = int(
            coord[1]
        ) + y0

        xx = int(
            coord[2]
        ) + x0

        distance = math.sqrt(
            (zz - z) ** 2
            + (yy - y) ** 2
            + (xx - x) ** 2
        )

        best = min(
            best,
            distance,
        )

    return float(best)


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
        get_transformed_points(
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
        model_input = (
            image_tensor[
                None,
                None,
            ]
        )

    elif image_tensor.ndim == 4:
        model_input = (
            image_tensor[
                None
            ]
        )

    else:
        raise RuntimeError(
            f"Unexpected image shape: "
            f"{tuple(image_tensor.shape)}"
        )

    output = model(
        model_input
    )

    if isinstance(
        output,
        (tuple, list),
    ):
        output = output[0]

    if output.ndim != 5:
        raise RuntimeError(
            f"Unexpected model output shape: "
            f"{tuple(output.shape)}"
        )

    logits = output[0]

    probabilities = torch.softmax(
        logits,
        dim=0,
    )

    prediction = torch.argmax(
        probabilities,
        dim=0,
    )

    grad = gradient_magnitude(
        image_tensor
    )

    records = []

    for point_index, point in enumerate(
        transformed_points
    ):

        if point_index >= len(
            point_df
        ):
            break

        source_row = point_df.iloc[
            point_index
        ]

        class_id = int(
            source_row[
                "class_id"
            ]
        )

        class_name = str(
            source_row[
                "class_name"
            ]
        )

        level = str(
            source_row[
                "level"
            ]
        )

        z = clamp_index(
            point["z"],
            MODEL_SHAPE[0],
        )

        y = clamp_index(
            point["y"],
            MODEL_SHAPE[1],
        )

        x = clamp_index(
            point["x"],
            MODEL_SHAPE[2],
        )

        point_probability = (
            probabilities[
                class_id,
                z,
                y,
                x,
            ]
            .item()
        )

        background_probability = (
            probabilities[
                BACKGROUND_CLASS,
                z,
                y,
                x,
            ]
            .item()
        )

        target_logit = (
            logits[
                class_id,
                z,
                y,
                x,
            ]
            .item()
        )

        background_logit = (
            logits[
                BACKGROUND_CLASS,
                z,
                y,
                x,
            ]
            .item()
        )

        margin = (
            target_logit
            - background_logit
        )

        predicted_class = int(
            prediction[
                z,
                y,
                x,
            ]
            .item()
        )

        distance_fg = (
            distance_to_foreground(
                prediction,
                (z, y, x),
                max_radius=12,
            )
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
            "target_probability": (
                point_probability
            ),
            "background_probability": (
                background_probability
            ),
            "target_background_margin": (
                margin
            ),
            "predicted_class": (
                predicted_class
            ),
            "predicted_class_name": (
                str(
                    [
                        "Background",
                        "Spinal Canal Stenosis",
                        "Left Neural Foraminal Narrowing",
                        "Right Neural Foraminal Narrowing",
                        "Left Subarticular Stenosis",
                        "Right Subarticular Stenosis",
                    ][
                        predicted_class
                    ]
                )
            ),
            "distance_to_predicted_foreground": (
                distance_fg
            ),
        }

        # ------------------------------------------------------------
        # Local context
        # ------------------------------------------------------------

        for radius in NEIGHBORHOOD_RADII:

            image_patch = extract_neighborhood(
                image_tensor,
                z,
                y,
                x,
                radius,
            )

            gradient_patch = extract_neighborhood(
                grad,
                z,
                y,
                x,
                radius,
            )

            prediction_patch = extract_neighborhood(
                prediction,
                z,
                y,
                x,
                radius,
            )

            target_probability_patch = (
                extract_neighborhood(
                    probabilities[
                        class_id
                    ],
                    z,
                    y,
                    x,
                    radius,
                )
            )

            background_probability_patch = (
                extract_neighborhood(
                    probabilities[
                        BACKGROUND_CLASS
                    ],
                    z,
                    y,
                    x,
                    radius,
                )
            )

            foreground_mask = (
                prediction_patch
                != BACKGROUND_CLASS
            )

            prefix = (
                f"r{radius}"
            )

            record[
                f"{prefix}_image_mean"
            ] = float(
                image_patch.mean().item()
            )

            record[
                f"{prefix}_image_std"
            ] = float(
                image_patch.std().item()
            )

            record[
                f"{prefix}_image_min"
            ] = float(
                image_patch.min().item()
            )

            record[
                f"{prefix}_image_max"
            ] = float(
                image_patch.max().item()
            )

            record[
                f"{prefix}_gradient_mean"
            ] = float(
                gradient_patch.mean().item()
            )

            record[
                f"{prefix}_gradient_std"
            ] = float(
                gradient_patch.std().item()
            )

            record[
                f"{prefix}_foreground_ratio"
            ] = float(
                foreground_mask.float()
                .mean()
                .item()
            )

            record[
                f"{prefix}_target_probability_mean"
            ] = float(
                target_probability_patch
                .mean()
                .item()
            )

            record[
                f"{prefix}_background_probability_mean"
            ] = float(
                background_probability_patch
                .mean()
                .item()
            )

            record[
                f"{prefix}_target_minus_background_mean"
            ] = float(
                (
                    target_probability_patch
                    - background_probability_patch
                )
                .mean()
                .item()
            )

        records.append(
            record
        )

    return records


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def build_summaries(
    records_df,
):
    print()
    print("=" * 80)
    print("BUILDING PART 2.40 SUMMARIES")
    print("=" * 80)

    rfnn = records_df[
        records_df[
            "class_name"
        ]
        == "Right Neural Foraminal Narrowing"
    ].copy()

    lfnn = records_df[
        records_df[
            "class_name"
        ]
        == "Left Neural Foraminal Narrowing"
    ].copy()

    print(
        f"RFNN records: {len(rfnn)}"
    )

    print(
        f"LFNN records: {len(lfnn)}"
    )

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

    # ------------------------------------------------------------
    # RFNN level summary
    # ------------------------------------------------------------

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
            mean_margin=(
                "target_background_margin",
                "mean",
            ),
            median_margin=(
                "target_background_margin",
                "median",
            ),
            mean_target_probability=(
                "target_probability",
                "mean",
            ),
            mean_background_probability=(
                "background_probability",
                "mean",
            ),
            mean_distance_to_foreground=(
                "distance_to_predicted_foreground",
                "mean",
            ),
            predicted_foreground_rate=(
                "predicted_class",
                lambda x: float(
                    (x != BACKGROUND_CLASS)
                    .mean()
                ),
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # RFNN series summary
    # ------------------------------------------------------------

    series_summary = (
        rfnn
        .groupby(
            [
                "study_id",
                "series_id",
            ],
            dropna=False,
        )
        .agg(
            count=(
                "point_index",
                "count",
            ),
            mean_margin=(
                "target_background_margin",
                "mean",
            ),
            median_margin=(
                "target_background_margin",
                "median",
            ),
            mean_target_probability=(
                "target_probability",
                "mean",
            ),
            mean_background_probability=(
                "background_probability",
                "mean",
            ),
            mean_distance_to_foreground=(
                "distance_to_predicted_foreground",
                "mean",
            ),
            predicted_foreground_rate=(
                "predicted_class",
                lambda x: float(
                    (x != BACKGROUND_CLASS)
                    .mean()
                ),
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # RFNN vs LFNN paired summary
    # ------------------------------------------------------------

    paired = (
        rfnn[
            [
                "study_id",
                "series_id",
                "level",
                "target_background_margin",
                "target_probability",
                "background_probability",
                "distance_to_predicted_foreground",
            ]
        ]
        .rename(
            columns={
                "target_background_margin":
                    "rfnn_margin",
                "target_probability":
                    "rfnn_probability",
                "background_probability":
                    "rfnn_background_probability",
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
                    "target_background_margin",
                    "target_probability",
                    "background_probability",
                    "distance_to_predicted_foreground",
                ]
            ].rename(
                columns={
                    "target_background_margin":
                        "lfnn_margin",
                    "target_probability":
                        "lfnn_probability",
                    "background_probability":
                        "lfnn_background_probability",
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

    paired[
        "margin_difference_rfnn_minus_lfnn"
    ] = (
        paired["rfnn_margin"]
        - paired["lfnn_margin"]
    )

    paired[
        "probability_difference_rfnn_minus_lfnn"
    ] = (
        paired["rfnn_probability"]
        - paired["lfnn_probability"]
    )

    paired[
        "background_difference_rfnn_minus_lfnn"
    ] = (
        paired["rfnn_background_probability"]
        - paired["lfnn_background_probability"]
    )

    paired[
        "distance_difference_rfnn_minus_lfnn"
    ] = (
        paired["rfnn_distance_to_foreground"]
        - paired["lfnn_distance_to_foreground"]
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    records_path = (
        OUTPUT_DIR
        / "part240_all_rfnn_context_records.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part240_rfnn_level_summary.csv"
    )

    series_path = (
        OUTPUT_DIR
        / "part240_rfnn_series_summary.csv"
    )

    paired_path = (
        OUTPUT_DIR
        / "part240_paired_lfnn_rfnn_context.csv"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    series_summary.to_csv(
        series_path,
        index=False,
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Global RFNN summary
    # ------------------------------------------------------------

    global_summary = {
        "rfnn_points": int(
            len(rfnn)
        ),
        "lfnn_points": int(
            len(lfnn)
        ),
        "rfnn_mean_margin": safe_float(
            rfnn[
                "target_background_margin"
            ].mean()
        ),
        "rfnn_median_margin": safe_float(
            rfnn[
                "target_background_margin"
            ].median()
        ),
        "rfnn_positive_margin_rate": safe_float(
            (
                rfnn[
                    "target_background_margin"
                ]
                > 0
            ).mean()
        ),
        "rfnn_mean_target_probability": safe_float(
            rfnn[
                "target_probability"
            ].mean()
        ),
        "rfnn_mean_background_probability": safe_float(
            rfnn[
                "background_probability"
            ].mean()
        ),
        "rfnn_mean_distance_to_predicted_foreground": safe_float(
            rfnn[
                "distance_to_predicted_foreground"
            ].mean()
        ),
        "rfnn_predicted_foreground_rate": safe_float(
            (
                rfnn[
                    "predicted_class"
                ]
                != BACKGROUND_CLASS
            ).mean()
        ),
        "paired_observations": int(
            len(paired)
        ),
        "paired_mean_margin_difference": safe_float(
            paired[
                "margin_difference_rfnn_minus_lfnn"
            ].mean()
        ),
        "paired_mean_probability_difference": safe_float(
            paired[
                "probability_difference_rfnn_minus_lfnn"
            ].mean()
        ),
        "paired_mean_background_probability_difference": safe_float(
            paired[
                "background_difference_rfnn_minus_lfnn"
            ].mean()
        ),
        "paired_mean_distance_difference": safe_float(
            paired[
                "distance_difference_rfnn_minus_lfnn"
            ].mean()
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "part240_rfnn_context_audit_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            global_summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Terminal report
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("RFNN GLOBAL SUMMARY")
    print("=" * 80)

    print(
        f"RFNN points                     : "
        f"{len(rfnn)}"
    )

    print(
        f"Mean RFNN-BG margin             : "
        f"{rfnn['target_background_margin'].mean():.6f}"
    )

    print(
        f"Median RFNN-BG margin           : "
        f"{rfnn['target_background_margin'].median():.6f}"
    )

    predicted_foreground_rate = (
    rfnn["predicted_class"]
    != BACKGROUND_CLASS
    ).mean()

    print(
    f"Predicted foreground rate        : "
    f"{predicted_foreground_rate:.6f}"
    )

    print(
        f"Mean RFNN probability            : "
        f"{rfnn['target_probability'].mean():.6f}"
    )

    print(
        f"Mean background probability      : "
        f"{rfnn['background_probability'].mean():.6f}"
    )

    print(
        f"Mean distance to predicted FG    : "
        f"{rfnn['distance_to_predicted_foreground'].mean():.6f}"
    )

    predicted_foreground_rate = (
    rfnn["predicted_class"]
    != BACKGROUND_CLASS
    ).mean()

    print(
    f"Predicted foreground rate        : "
    f"{predicted_foreground_rate:.6f}"
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
    print("RFNN SERIES ANALYSIS")
    print("=" * 80)

    series_sorted = (
        series_summary
        .sort_values(
            "mean_margin"
        )
    )

    print(
        "\nWorst RFNN series:\n"
    )

    print(
        series_sorted
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nBest RFNN series:\n"
    )

    print(
        series_sorted
        .tail(10)
        .sort_values(
            "mean_margin",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PAIRED LFNN vs RFNN")
    print("=" * 80)

    print(
        f"Paired observations             : "
        f"{len(paired)}"
    )

    print(
        f"Mean RFNN-LFNN margin difference: "
        f"{paired['margin_difference_rfnn_minus_lfnn'].mean():.6f}"
    )

    print(
        f"Mean RFNN-LFNN probability diff  : "
        f"{paired['probability_difference_rfnn_minus_lfnn'].mean():.6f}"
    )

    print(
        f"Mean RFNN-LFNN background diff   : "
        f"{paired['background_difference_rfnn_minus_lfnn'].mean():.6f}"
    )

    print(
        f"Mean RFNN-LFNN FG distance diff : "
        f"{paired['distance_difference_rfnn_minus_lfnn'].mean():.6f}"
    )

    print()
    print("=" * 80)
    print("PART 2.40 COMPLETE")
    print("=" * 80)

    print(
        f"RFNN records analyzed : {len(rfnn)}"
    )

    print(
        f"LFNN records analyzed : {len(lfnn)}"
    )

    print(
        f"Paired observations    : {len(paired)}"
    )

    print()
    print("Outputs:")

    print(
        f"  {records_path}"
    )

    print(
        f"  {level_path}"
    )

    print(
        f"  {series_path}"
    )

    print(
        f"  {paired_path}"
    )

    print(
        f"  {summary_path}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This experiment did not train the model."
    )

    print(
        "The Part 2.27 checkpoint was not modified."
    )

    print(
        "No dashboard files were modified."
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.40")
    print("RFNN FOREGROUND-vs-BACKGROUND CONTEXT DIAGNOSTIC")
    print("=" * 80)

    print()
    print("This is an ANALYSIS-ONLY experiment.")
    print("No training will be performed.")
    print("No checkpoint will be modified.")
    print("No dashboard files will be modified.")

    # ------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING EXACT PART 2.20B MANIFEST")
    print("=" * 80)

    manifest = (
        part220b.load_manifest()
    )

    print(
        f"Full manifest rows: "
        f"{len(manifest)}"
    )

    # ------------------------------------------------------------
    # Validation cohort
    # ------------------------------------------------------------

    (
        validation_cases,
        validation_points,
    ) = build_exact_validation_cohort(
        manifest
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = load_model()

    # ------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING RFNN CONTEXT ANALYSIS")
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

        try:

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

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            raise

    # ------------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------------

    records_df = pd.DataFrame(
        all_records
    )

    if len(records_df) != 188:
        raise RuntimeError(
            "Expected exactly 188 point "
            f"records, found {len(records_df)}."
        )

    # ------------------------------------------------------------
    # Build summaries
    # ------------------------------------------------------------

    build_summaries(
        records_df
    )


if __name__ == "__main__":
    main()