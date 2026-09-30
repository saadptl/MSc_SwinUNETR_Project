"""
Part 2.42
RFNN Local Foreground Class Attribution Audit

Purpose
-------
Determine which disease class produces the strongest local foreground
activation around annotated RFNN points.

Questions:
    1. What class wins exactly at the RFNN annotation?
    2. What class has the strongest probability within radius 2?
    3. What class has the strongest probability within radius 4?
    4. What class has the strongest probability within radius 6?
    5. How far is that maximum from the RFNN annotation?
    6. Is the strong local foreground activation RFNN, LFNN, LSS,
       RSS, or SCS?
    7. Does the same pattern occur for paired LFNN points?

This is ANALYSIS ONLY.

No training.
No checkpoint modification.
No dashboard modification.
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
    / "rsna_part242_rfnn_local_foreground_class_attribution_audit"
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
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

BACKGROUND_CLASS = 0
SCS_CLASS = 1
LFNN_CLASS = 2
RFNN_CLASS = 3
LSS_CLASS = 4
RSS_CLASS = 5

FOREGROUND_CLASSES = [
    SCS_CLASS,
    LFNN_CLASS,
    RFNN_CLASS,
    LSS_CLASS,
    RSS_CLASS,
]

RADII = [2, 4, 6]

LEVELS = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]


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


# ============================================================================
# LOCAL PATCH
# ============================================================================

def extract_patch(
    volume,
    z,
    y,
    x,
    radius,
):
    """
    Extract a local 3-D patch around a model-grid point.

    Returns
    -------
    patch
    origin
        Origin of the patch in the original volume.
    """

    depth = volume.shape[-3]
    height = volume.shape[-2]
    width = volume.shape[-1]

    zc = clamp_index(z, depth)
    yc = clamp_index(y, height)
    xc = clamp_index(x, width)

    z0 = max(0, zc - radius)
    z1 = min(depth, zc + radius + 1)

    y0 = max(0, yc - radius)
    y1 = min(height, yc + radius + 1)

    x0 = max(0, xc - radius)
    x1 = min(width, xc + radius + 1)

    patch = volume[
        ...,
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    return patch, (z0, y0, x0)


# ============================================================================
# LOCAL FOREGROUND CLASS WINNER
# ============================================================================

def local_class_maximum(
    probabilities,
    z,
    y,
    x,
    radius,
):
    """
    Find the strongest foreground class activation in a local neighborhood.

    Returns
    -------
    winner_class
    winner_probability
    distance
    winner_z
    winner_y
    winner_x
    """

    depth = probabilities.shape[1]
    height = probabilities.shape[2]
    width = probabilities.shape[3]

    zc = clamp_index(z, depth)
    yc = clamp_index(y, height)
    xc = clamp_index(x, width)

    z0 = max(0, zc - radius)
    z1 = min(depth, zc + radius + 1)

    y0 = max(0, yc - radius)
    y1 = min(height, yc + radius + 1)

    x0 = max(0, xc - radius)
    x1 = min(width, xc + radius + 1)

    foreground = probabilities[
        FOREGROUND_CLASSES,
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    n_classes = len(FOREGROUND_CLASSES)

    flat = foreground.reshape(
        n_classes,
        -1,
    )

    best_index = int(
        torch.argmax(flat).item()
    )

    class_position = (
        best_index // flat.shape[1]
    )

    voxel_position = (
        best_index % flat.shape[1]
    )

    winner_class = FOREGROUND_CLASSES[
        class_position
    ]

    winner_probability = float(
        flat[
            class_position,
            voxel_position,
        ].item()
    )

    local_shape = (
        z1 - z0,
        y1 - y0,
        x1 - x0,
    )

    local_coords = np.unravel_index(
        voxel_position,
        local_shape,
    )

    winner_z = (
        z0 + int(local_coords[0])
    )

    winner_y = (
        y0 + int(local_coords[1])
    )

    winner_x = (
        x0 + int(local_coords[2])
    )

    distance = math.sqrt(
        (winner_z - zc) ** 2
        + (winner_y - yc) ** 2
        + (winner_x - xc) ** 2
    )

    return (
        int(winner_class),
        winner_probability,
        distance,
        winner_z,
        winner_y,
        winner_x,
    )


# ============================================================================
# CLASS-SPECIFIC LOCAL MAXIMUM
# ============================================================================

def class_specific_local_maximum(
    probabilities,
    class_id,
    z,
    y,
    x,
    radius,
):
    """
    Find maximum probability of one specific class in a local neighborhood.
    """

    patch, origin = extract_patch(
        probabilities[class_id],
        z,
        y,
        x,
        radius,
    )

    if patch.numel() == 0:
        return (
            0.0,
            float(radius + 1),
            int(z),
            int(y),
            int(x),
        )

    flat_index = int(
        torch.argmax(patch).item()
    )

    local_coords = np.unravel_index(
        flat_index,
        tuple(patch.shape),
    )

    best_z = (
        origin[0]
        + int(local_coords[0])
    )

    best_y = (
        origin[1]
        + int(local_coords[1])
    )

    best_x = (
        origin[2]
        + int(local_coords[2])
    )

    zc = int(round(z))
    yc = int(round(y))
    xc = int(round(x))

    distance = math.sqrt(
        (best_z - zc) ** 2
        + (best_y - yc) ** 2
        + (best_x - xc) ** 2
    )

    maximum = float(
        patch.max().item()
    )

    return (
        maximum,
        distance,
        best_z,
        best_y,
        best_x,
    )


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
        key=lambda case: case_order.get(
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

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
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

    elif image_tensor.ndim == 4:

        model_input = image_tensor[
            None
        ]

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

    records = []

    for point_index, point in enumerate(
        transformed_points
    ):

        source = point_df.iloc[
            point_index
        ]

        target_class = int(
            source["class_id"]
        )

        target_name = str(
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

        point_probs = probabilities[
            :,
            z,
            y,
            x,
        ]

        point_winner = int(
            torch.argmax(
                point_probs
            ).item()
        )

        record = {
            "study_id": study_id,
            "series_id": series_id,
            "point_index": int(point_index),
            "target_class": target_class,
            "target_class_name": target_name,
            "level": level,
            "z": z,
            "y": y,
            "x": x,
            "point_winner_class": point_winner,
            "point_winner_name": CLASS_NAMES[
                point_winner
            ],
            "point_winner_probability": float(
                point_probs[
                    point_winner
                ].item()
            ),
            "point_target_probability": float(
                point_probs[
                    target_class
                ].item()
            ),
        }

        # ------------------------------------------------------------
        # Local radius analysis
        # ------------------------------------------------------------

        for radius in RADII:

            (
                winner_class,
                winner_probability,
                winner_distance,
                winner_z,
                winner_y,
                winner_x,
            ) = local_class_maximum(
                probabilities,
                z,
                y,
                x,
                radius,
            )

            prefix = f"r{radius}"

            record[
                f"{prefix}_winner_class"
            ] = winner_class

            record[
                f"{prefix}_winner_name"
            ] = CLASS_NAMES[
                winner_class
            ]

            record[
                f"{prefix}_winner_probability"
            ] = winner_probability

            record[
                f"{prefix}_winner_distance"
            ] = winner_distance

            record[
                f"{prefix}_winner_z"
            ] = winner_z

            record[
                f"{prefix}_winner_y"
            ] = winner_y

            record[
                f"{prefix}_winner_x"
            ] = winner_x

            # --------------------------------------------------------
            # Maximum probability for each foreground class
            # --------------------------------------------------------

            class_short_names = {
                1: "scs",
                2: "lfnn",
                3: "rfnn",
                4: "lss",
                5: "rss",
            }

            for class_id in FOREGROUND_CLASSES:

                (
                    maximum,
                    distance,
                    best_z,
                    best_y,
                    best_x,
                ) = class_specific_local_maximum(
                    probabilities,
                    class_id,
                    z,
                    y,
                    x,
                    radius,
                )

                short_name = class_short_names[
                    class_id
                ]

                record[
                    f"{prefix}_{short_name}_max"
                ] = maximum

                record[
                    f"{prefix}_{short_name}_distance"
                ] = distance

                record[
                    f"{prefix}_{short_name}_z"
                ] = best_z

                record[
                    f"{prefix}_{short_name}_y"
                ] = best_y

                record[
                    f"{prefix}_{short_name}_x"
                ] = best_x

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
    print("BUILDING PART 2.42 SUMMARY")
    print("=" * 80)

    rfnn = records_df[
        records_df[
            "target_class_name"
        ]
        == "Right Neural Foraminal Narrowing"
    ].copy()

    lfnn = records_df[
        records_df[
            "target_class_name"
        ]
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

    # ========================================================================
    # LOCAL WINNER TABLES
    # ========================================================================

    winner_tables = {}

    for radius in RADII:

        column = (
            f"r{radius}_winner_name"
        )

        winner_tables[radius] = (
            rfnn[column]
            .value_counts()
            .rename_axis(
                "winner_class"
            )
            .reset_index(
                name="count"
            )
        )

    # ========================================================================
    # RFNN LEVEL SUMMARY
    # ========================================================================

    level_rows = []

    for level in LEVELS:

        subset = rfnn[
            rfnn["level"] == level
        ]

        if len(subset) == 0:
            continue

        row = {
            "level": level,
            "count": len(subset),
        }

        for radius in RADII:

            winner_column = (
                f"r{radius}_winner_name"
            )

            for class_id, short_name in [
                (SCS_CLASS, "scs"),
                (LFNN_CLASS, "lfnn"),
                (RFNN_CLASS, "rfnn"),
                (LSS_CLASS, "lss"),
                (RSS_CLASS, "rss"),
            ]:

                class_name = CLASS_NAMES[
                    class_id
                ]

                row[
                    f"r{radius}_{short_name}_winner_count"
                ] = int(
                    (
                        subset[
                            winner_column
                        ]
                        == class_name
                    ).sum()
                )

        level_rows.append(
            row
        )

    level_summary = pd.DataFrame(
        level_rows
    )

    # ========================================================================
    # PAIRED LFNN/RFNN
    # ========================================================================

    rfnn_columns = [
        "study_id",
        "series_id",
        "level",
        "point_target_probability",
    ]

    paired_rfnn = rfnn[
        rfnn_columns
    ].copy()

    paired_rfnn = paired_rfnn.rename(
        columns={
            "point_target_probability":
                "rfnn_point_probability",
        }
    )

    for radius in RADII:

        paired_rfnn[
            f"rfnn_r{radius}_winner"
        ] = rfnn[
            f"r{radius}_winner_name"
        ].values

        paired_rfnn[
            f"rfnn_r{radius}_winner_probability"
        ] = rfnn[
            f"r{radius}_winner_probability"
        ].values

        paired_rfnn[
            f"rfnn_r{radius}_winner_distance"
        ] = rfnn[
            f"r{radius}_winner_distance"
        ].values

    lfnn_columns = [
        "study_id",
        "series_id",
        "level",
        "point_target_probability",
    ]

    paired_lfnn = lfnn[
        lfnn_columns
    ].copy()

    paired_lfnn = paired_lfnn.rename(
        columns={
            "point_target_probability":
                "lfnn_point_probability",
        }
    )

    for radius in RADII:

        paired_lfnn[
            f"lfnn_r{radius}_winner"
        ] = lfnn[
            f"r{radius}_winner_name"
        ].values

        paired_lfnn[
            f"lfnn_r{radius}_winner_probability"
        ] = lfnn[
            f"r{radius}_winner_probability"
        ].values

        paired_lfnn[
            f"lfnn_r{radius}_winner_distance"
        ] = lfnn[
            f"r{radius}_winner_distance"
        ].values

    paired = paired_rfnn.merge(
        paired_lfnn,
        on=[
            "study_id",
            "series_id",
            "level",
        ],
        how="inner",
    )

    if len(paired) != 45:
        raise RuntimeError(
            "Expected 45 paired LFNN/RFNN "
            f"observations, found {len(paired)}."
        )

    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================

    records_path = (
        OUTPUT_DIR
        / "part242_all_local_class_attribution_records.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part242_rfnn_level_class_attribution_summary.csv"
    )

    paired_path = (
        OUTPUT_DIR
        / "part242_paired_lfnn_rfnn_class_attribution.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part242_local_class_attribution_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
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
        "local_winner_counts_r2": winner_tables[
            2
        ].to_dict(
            orient="records"
        ),
        "local_winner_counts_r4": winner_tables[
            4
        ].to_dict(
            orient="records"
        ),
        "local_winner_counts_r6": winner_tables[
            6
        ].to_dict(
            orient="records"
        ),
    }

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

    # ========================================================================
    # TERMINAL REPORT
    # ========================================================================

    print()
    print("=" * 80)
    print("RFNN LOCAL WINNER — RADIUS 2")
    print("=" * 80)

    print(
        winner_tables[2].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("RFNN LOCAL WINNER — RADIUS 4")
    print("=" * 80)

    print(
        winner_tables[4].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("RFNN LOCAL WINNER — RADIUS 6")
    print("=" * 80)

    print(
        winner_tables[6].to_string(
            index=False
        )
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
    print("PAIRED LFNN vs RFNN")
    print("=" * 80)

    for radius in RADII:

        rfnn_winner_count = int(
            (
                paired[
                    f"rfnn_r{radius}_winner"
                ]
                == "Right Neural Foraminal Narrowing"
            ).sum()
        )

        lfnn_winner_count = int(
            (
                paired[
                    f"lfnn_r{radius}_winner"
                ]
                == "Left Neural Foraminal Narrowing"
            ).sum()
        )

        print()
        print(
            f"Radius {radius}:"
        )

        print(
            f"  RFNN local RFNN winner: "
            f"{rfnn_winner_count}/45"
        )

        print(
            f"  LFNN local LFNN winner: "
            f"{lfnn_winner_count}/45"
        )

    print()
    print("=" * 80)
    print("PART 2.42 COMPLETE")
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
    print("PART 2.42")
    print("RFNN LOCAL FOREGROUND CLASS ATTRIBUTION AUDIT")
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

    model = load_model()

    print()
    print("=" * 80)
    print("RUNNING LOCAL CLASS ATTRIBUTION ANALYSIS")
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
            "  Analysis successful."
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