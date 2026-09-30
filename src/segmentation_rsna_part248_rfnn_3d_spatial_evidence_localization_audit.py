"""
PART 2.48
RFNN 3D SPATIAL EVIDENCE LOCALIZATION AUDIT

Purpose
-------
Determine where RFNN evidence occurs around annotated RFNN points.

For every RFNN annotation, inspect a 13 x 13 x 13 voxel neighborhood
(centered on the annotation):

    z +/- 6
    y +/- 6
    x +/- 6

For every valid local voxel calculate:

    RFNN probability
    RFNN logit
    Background probability
    LSS probability
    SCS probability
    RFNN - Background logit
    RFNN - LSS logit
    RFNN - SCS logit

Then identify:

    1. Maximum RFNN probability voxel
    2. Maximum RFNN logit voxel
    3. Maximum RFNN-background margin voxel
    4. Maximum RFNN-LSS margin voxel
    5. Maximum RFNN-SCS margin voxel

For each maximum, report displacement from the annotated point:

    dz
    dy
    dx
    Euclidean voxel distance

The analysis is also summarized by lumbar level.

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
    / "rsna_part248_rfnn_3d_spatial_evidence_localization_audit"
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

LOCAL_RADIUS = 6


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


def clamp_point(
    z,
    y,
    x,
    shape,
):

    depth = shape[0]
    height = shape[1]
    width = shape[2]

    z = int(
        max(
            0,
            min(
                int(round(z)),
                depth - 1,
            ),
        )
    )

    y = int(
        max(
            0,
            min(
                int(round(y)),
                height - 1,
            ),
        )
    )

    x = int(
        max(
            0,
            min(
                int(round(x)),
                width - 1,
            ),
        )
    )

    return z, y, x


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
# LOCALIZATION
# ============================================================================

def analyze_rfnn_point(
    logits,
    probabilities,
    transformed_point,
):

    z = safe_float(
        transformed_point["z"]
    )

    y = safe_float(
        transformed_point["y"]
    )

    x = safe_float(
        transformed_point["x"]
    )

    if not all(
        np.isfinite(value)
        for value in [
            z,
            y,
            x,
        ]
    ):

        return None, []

    _, depth, height, width = (
        logits.shape
    )

    center_z, center_y, center_x = (
        clamp_point(
            z,
            y,
            x,
            (
                depth,
                height,
                width,
            ),
        )
    )

    local_records = []

    z_start = max(
        0,
        center_z - LOCAL_RADIUS,
    )

    z_end = min(
        depth,
        center_z + LOCAL_RADIUS + 1,
    )

    y_start = max(
        0,
        center_y - LOCAL_RADIUS,
    )

    y_end = min(
        height,
        center_y + LOCAL_RADIUS + 1,
    )

    x_start = max(
        0,
        center_x - LOCAL_RADIUS,
    )

    x_end = min(
        width,
        center_x + LOCAL_RADIUS + 1,
    )

    for local_z in range(
        z_start,
        z_end,
    ):

        for local_y in range(
            y_start,
            y_end,
        ):

            for local_x in range(
                x_start,
                x_end,
            ):

                logits_here = (
                    logits[
                        :,
                        local_z,
                        local_y,
                        local_x,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                probabilities_here = (
                    probabilities[
                        :,
                        local_z,
                        local_y,
                        local_x,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                dz = (
                    local_z
                    - center_z
                )

                dy = (
                    local_y
                    - center_y
                )

                dx = (
                    local_x
                    - center_x
                )

                distance = float(
                    np.sqrt(
                        dz * dz
                        + dy * dy
                        + dx * dx
                    )
                )

                local_records.append(
                    {
                        "z": int(
                            local_z
                        ),
                        "y": int(
                            local_y
                        ),
                        "x": int(
                            local_x
                        ),
                        "dz": int(dz),
                        "dy": int(dy),
                        "dx": int(dx),
                        "distance_voxels":
                            distance,
                        "rfnn_logit":
                            float(
                                logits_here[3]
                            ),
                        "rfnn_probability":
                            float(
                                probabilities_here[3]
                            ),
                        "background_probability":
                            float(
                                probabilities_here[0]
                            ),
                        "lss_probability":
                            float(
                                probabilities_here[4]
                            ),
                        "scs_probability":
                            float(
                                probabilities_here[1]
                            ),
                        "lfnn_probability":
                            float(
                                probabilities_here[2]
                            ),
                        "rss_probability":
                            float(
                                probabilities_here[5]
                            ),
                        "rfnn_minus_background":
                            float(
                                logits_here[3]
                                - logits_here[0]
                            ),
                        "rfnn_minus_lss":
                            float(
                                logits_here[3]
                                - logits_here[4]
                            ),
                        "rfnn_minus_scs":
                            float(
                                logits_here[3]
                                - logits_here[1]
                            ),
                        "rfnn_minus_lfnn":
                            float(
                                logits_here[3]
                                - logits_here[2]
                            ),
                        "rfnn_minus_rss":
                            float(
                                logits_here[3]
                                - logits_here[5]
                            ),
                    }
                )

    local_df = pd.DataFrame(
        local_records
    )

    if local_df.empty:

        return None, []

    # ------------------------------------------------------------------------
    # Identify maxima
    # ------------------------------------------------------------------------

    criteria = {
        "max_rfnn_probability":
            "rfnn_probability",

        "max_rfnn_logit":
            "rfnn_logit",

        "max_rfnn_minus_background":
            "rfnn_minus_background",

        "max_rfnn_minus_lss":
            "rfnn_minus_lss",

        "max_rfnn_minus_scs":
            "rfnn_minus_scs",
    }

    summary = {}

    for name, column in criteria.items():

        index = (
            local_df[column]
            .astype(float)
            .idxmax()
        )

        row = local_df.loc[
            index
        ]

        summary[
            name
        ] = {
            "value": float(
                row[column]
            ),
            "dz": int(
                row["dz"]
            ),
            "dy": int(
                row["dy"]
            ),
            "dx": int(
                row["dx"]
            ),
            "distance_voxels":
                float(
                    row[
                        "distance_voxels"
                    ]
                ),
        }

    return summary, local_records


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

        output = model(
            image
        )

        logits = output[0]

        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

    rfnn_indices = (
        case_rows.index[
            case_rows[
                "class_name"
            ]
            == "Right Neural Foraminal Narrowing"
        ]
        .tolist()
    )

    summary_records = []

    voxel_records = []

    for point_index in rfnn_indices:

        transformed_point = (
            transformed_points[
                point_index
            ]
        )

        summary, local_records = (
            analyze_rfnn_point(
                logits,
                probabilities,
                transformed_point,
            )
        )

        if summary is None:

            continue

        level = str(
            case_rows.iloc[
                point_index
            ]["level"]
        )

        base = {
            "study_id": study_id,
            "series_id": series_id,
            "point_index": int(
                point_index
            ),
            "level": level,
            "center_z": safe_float(
                transformed_point["z"]
            ),
            "center_y": safe_float(
                transformed_point["y"]
            ),
            "center_x": safe_float(
                transformed_point["x"]
            ),
        }

        for criterion, values in (
            summary.items()
        ):

            base[
                f"{criterion}_value"
            ] = values[
                "value"
            ]

            base[
                f"{criterion}_dz"
            ] = values[
                "dz"
            ]

            base[
                f"{criterion}_dy"
            ] = values[
                "dy"
            ]

            base[
                f"{criterion}_dx"
            ] = values[
                "dx"
            ]

            base[
                f"{criterion}_distance"
            ] = values[
                "distance_voxels"
            ]

        summary_records.append(
            base
        )

        for voxel in local_records:

            voxel_record = {
                "study_id": study_id,
                "series_id": series_id,
                "point_index": int(
                    point_index
                ),
                "level": level,
            }

            voxel_record.update(
                voxel
            )

            voxel_records.append(
                voxel_record
            )

    return (
        summary_records,
        voxel_records,
    )


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    summary_records,
    voxel_records,
):

    print()
    print("=" * 80)
    print("BUILDING PART 2.48 SUMMARY")
    print("=" * 80)

    summary_df = pd.DataFrame(
        summary_records
    )

    voxel_df = pd.DataFrame(
        voxel_records
    )

    if len(summary_df) != 45:

        raise RuntimeError(
            "Expected 45 RFNN summary records, "
            f"found {len(summary_df)}."
        )

    expected_voxels_min = (
        45 * 13 * 13 * 13
    )

    if len(voxel_df) < expected_voxels_min:

        raise RuntimeError(
            "Unexpectedly low voxel count: "
            f"{len(voxel_df)}."
        )

    # ========================================================================
    # GLOBAL LOCALIZATION SUMMARY
    # ========================================================================

    criteria = [
        "max_rfnn_probability",
        "max_rfnn_logit",
        "max_rfnn_minus_background",
        "max_rfnn_minus_lss",
        "max_rfnn_minus_scs",
    ]

    global_rows = []

    for criterion in criteria:

        distance_column = (
            f"{criterion}_distance"
        )

        dz_column = (
            f"{criterion}_dz"
        )

        dy_column = (
            f"{criterion}_dy"
        )

        dx_column = (
            f"{criterion}_dx"
        )

        value_column = (
            f"{criterion}_value"
        )

        distances = summary_df[
            distance_column
        ].astype(float)

        dz_values = summary_df[
            dz_column
        ].astype(float)

        dy_values = summary_df[
            dy_column
        ].astype(float)

        dx_values = summary_df[
        dx_column
        ].astype(float)

        values = summary_df[
            value_column
        ].astype(float)

        global_rows.append(
            {
                "criterion": criterion,
                "count": len(
                    summary_df
                ),
                "mean_value": float(
                    values.mean()
                ),
                "median_value": float(
                    values.median()
                ),
                "mean_distance": float(
                    distances.mean()
                ),
                "median_distance": float(
                    distances.median()
                ),
                "min_distance": float(
                    distances.min()
                ),
                "max_distance": float(
                    distances.max()
                ),
                "within_1_voxel_fraction":
                    float(
                        (
                            distances
                            <= 1.0
                        ).mean()
                    ),
                "within_2_voxel_fraction":
                    float(
                        (
                            distances
                            <= 2.0
                        ).mean()
                    ),
                "within_4_voxel_fraction":
                    float(
                        (
                            distances
                            <= 4.0
                        ).mean()
                    ),
                "mean_abs_dz": float(
                    np.abs(
                        dz_values
                    ).mean()
                ),
                "mean_abs_dy": float(
                    np.abs(
                        dy_values
                    ).mean()
                ),
                "mean_abs_dx": float(
                    np.abs(
                        dx_values
                    ).mean()
                ),
                "mean_dz": float(
                    dz_values.mean()
                ),
                "mean_dy": float(
                    dy_values.mean()
                ),
                "mean_dx": float(
                    dx_values.mean()
                ),
            }
        )

    global_df = pd.DataFrame(
        global_rows
    )

    # ========================================================================
    # LEVEL SUMMARY
    # ========================================================================

    level_rows = []

    for level in LEVELS:

        subset = summary_df[
            summary_df[
                "level"
            ]
            == level
        ]

        if subset.empty:

            continue

        row = {
            "level": level,
            "count": len(
                subset
            ),
        }

        for criterion in criteria:

            distance_column = (
                f"{criterion}_distance"
            )

            value_column = (
                f"{criterion}_value"
            )

            row[
                f"{criterion}_mean_distance"
            ] = float(
                subset[
                    distance_column
                ].mean()
            )

            row[
                f"{criterion}_median_distance"
            ] = float(
                subset[
                    distance_column
                ].median()
            )

            row[
                f"{criterion}_within_2_fraction"
            ] = float(
                (
                    subset[
                        distance_column
                    ]
                    <= 2
                ).mean()
            )

            row[
                f"{criterion}_mean_value"
            ] = float(
                subset[
                    value_column
                ].mean()
            )

        level_rows.append(
            row
        )

    level_df = pd.DataFrame(
        level_rows
    )

    # ========================================================================
    # DIRECTIONAL SUMMARY
    # ========================================================================

    directional_rows = []

    for criterion in criteria:

        for axis in [
            "dz",
            "dy",
            "dx",
        ]:

            column = (
                f"{criterion}_{axis}"
            )

            values = summary_df[
                column
            ].astype(float)

            directional_rows.append(
                {
                    "criterion": criterion,
                    "axis": axis,
                    "mean": float(
                        values.mean()
                    ),
                    "median": float(
                        values.median()
                    ),
                    "std": float(
                        values.std()
                    ),
                    "negative_fraction": float(
                        (values < 0).mean()
                    ),
                    "positive_fraction": float(
                        (values > 0).mean()
                    ),
                    "zero_fraction": float(
                        (values == 0).mean()
                    ),
                }
            )

    directional_df = pd.DataFrame(
        directional_rows
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    summary_path = (
        OUTPUT_DIR
        / "part248_rfnn_spatial_maxima_records.csv"
    )

    voxel_path = (
        OUTPUT_DIR
        / "part248_rfnn_local_voxel_records.csv"
    )

    global_path = (
        OUTPUT_DIR
        / "part248_rfnn_spatial_localization_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part248_rfnn_level_spatial_localization_summary.csv"
    )

    directional_path = (
        OUTPUT_DIR
        / "part248_rfnn_directional_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part248_rfnn_3d_spatial_evidence_localization_audit_summary.json"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    voxel_df.to_csv(
        voxel_path,
        index=False,
    )

    global_df.to_csv(
        global_path,
        index=False,
    )

    level_df.to_csv(
        level_path,
        index=False,
    )

    directional_df.to_csv(
        directional_path,
        index=False,
    )

    json_summary = {
        "validation_cases": 25,
        "validation_points": 188,
        "rfnn_points": 45,
        "local_radius": 6,
        "local_cube_size": 13,
        "criteria": {},
    }

    for _, row in global_df.iterrows():

        criterion = row[
            "criterion"
        ]

        json_summary[
            "criteria"
        ][criterion] = {
            "mean_value":
                float(
                    row[
                        "mean_value"
                    ]
                ),
            "median_value":
                float(
                    row[
                        "median_value"
                    ]
                ),
            "mean_distance":
                float(
                    row[
                        "mean_distance"
                    ]
                ),
            "median_distance":
                float(
                    row[
                        "median_distance"
                    ]
                ),
            "within_2_voxel_fraction":
                float(
                    row[
                        "within_2_voxel_fraction"
                    ]
                ),
            "mean_dz":
                float(
                    row[
                        "mean_dz"
                    ]
                ),
            "mean_dy":
                float(
                    row[
                        "mean_dy"
                    ]
                ),
            "mean_dx":
                float(
                    row[
                        "mean_dx"
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
    print("PART 2.48 GLOBAL RFNN SPATIAL LOCALIZATION")
    print("=" * 80)

    print(
        global_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.48 LEVEL ANALYSIS")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.48 DIRECTIONAL ANALYSIS")
    print("=" * 80)

    print(
        directional_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.48 COMPLETE")
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
        summary_path
    )

    print(
        voxel_path
    )

    print(
        global_path
    )

    print(
        level_path
    )

    print(
        directional_path
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
    print("PART 2.48")
    print("RFNN 3D SPATIAL EVIDENCE LOCALIZATION AUDIT")
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
    print("RUNNING 3D RFNN SPATIAL EVIDENCE LOCALIZATION")
    print("=" * 80)

    all_summary_records = []
    all_voxel_records = []

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

            summary_records, voxel_records = (
                analyze_case(
                    model,
                    case,
                    case_rows,
                    device,
                )
            )

            all_summary_records.extend(
                summary_records
            )

            all_voxel_records.extend(
                voxel_records
            )

            print(
                f"[{case_number:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id}"
            )

            print(
                f"  RFNN points analyzed: "
                f"{len(summary_records)}"
            )

            print(
                f"  Local voxels recorded: "
                f"{len(voxel_records)}"
            )

        except Exception as exc:

            print(
                f"[{case_number:02d}/25] "
                f"Study={study_id} | "
                f"Series={series_id} "
                f"FAILED: {exc}"
            )

    build_summary(
        all_summary_records,
        all_voxel_records,
    )


if __name__ == "__main__":
    main()