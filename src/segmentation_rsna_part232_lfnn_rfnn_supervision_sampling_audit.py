"""
PART 2.32
LFNN/RFNN SUPERVISION & SAMPLING AUDIT

Purpose
-------
Audit the training supervision available for:

    LFNN = class 2
    RFNN = class 3

Questions:
1. Are LFNN and RFNN point counts balanced?
2. Are they balanced across spinal levels?
3. Are they balanced across studies/series?
4. How often do LFNN and RFNN occur in the same series?
5. Are their canonical/native coordinates distributed differently?
6. Are RFNN points closer to volume boundaries?
7. How many neighboring annotations are around each point?
8. Does the training sampler expose the two classes equally?

Analysis only.

NO:
    - model training
    - checkpoint modification
    - dashboard modification
    - Part104 modification
    - Part2.27 modification
    - fabricated voxel ground truth
"""

from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src import (
    segmentation_rsna_part220b_geometry_corrected_training as p220b
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part232_lfnn_rfnn_supervision_sampling_audit"
)

TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CLASSES
# ============================================================

LFNN = 2
RFNN = 3

LFNN_NAME = "Left Neural Foraminal Narrowing"
RFNN_NAME = "Right Neural Foraminal Narrowing"


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):

    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def numeric_series(
    df,
    column,
):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def print_section(title):

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "PART 2.32 — LFNN/RFNN "
        "SUPERVISION & SAMPLING AUDIT"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    # --------------------------------------------------------
    # Load Part 2.13 manifest
    # --------------------------------------------------------

    manifest_path = (
        PROJECT_ROOT
        / "outputs"
        / "segmentation"
        / "rsna_part213_point_supervision_manifest"
        / "part213_point_manifest.csv"
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            "Part 2.13 manifest not found:\n"
            f"{manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    print(
        f"Manifest rows: {len(manifest)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "study_id",
        "series_id",
        "class_id",
        "class_name",
        "level",
        "native_z",
        "native_y",
        "native_x",
        "model_z_float",
        "model_y_float",
        "model_x_float",
        "native_shape",
        "model_shape",
    ]

    missing = [
        c
        for c in required
        if c not in manifest.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
        )

    # --------------------------------------------------------
    # Normalize IDs
    # --------------------------------------------------------

    manifest["study_id"] = (
        manifest["study_id"]
        .astype(str)
    )

    manifest["series_id"] = (
        manifest["series_id"]
        .astype(str)
    )

    manifest["class_id"] = (
        pd.to_numeric(
            manifest["class_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    # --------------------------------------------------------
    # LFNN / RFNN subset
    # --------------------------------------------------------

    foraminal = manifest[
        manifest["class_id"].isin(
            [LFNN, RFNN]
        )
    ].copy()

    print(
        f"LFNN/RFNN points: "
        f"{len(foraminal)}"
    )

    # ========================================================
    # 1. GLOBAL CLASS COUNTS
    # ========================================================

    print_section(
        "1. GLOBAL LFNN/RFNN COUNTS"
    )

    global_counts = (
        foraminal
        .groupby(
            [
                "class_id",
                "class_name",
            ]
        )
        .size()
        .reset_index(
            name="points"
        )
    )

    global_counts[
        "percentage"
    ] = (
        global_counts["points"]
        / len(foraminal)
        * 100.0
    )

    print(
        global_counts.to_string(
            index=False
        )
    )

    global_counts.to_csv(
        TABLE_DIR
        / "part232_global_class_counts.csv",
        index=False,
    )

    # ========================================================
    # 2. LEVEL BALANCE
    # ========================================================

    print_section(
        "2. LEVEL-WISE LFNN/RFNN BALANCE"
    )

    level_counts = (
        foraminal
        .groupby(
            [
                "level",
                "class_name",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    if LFNN_NAME not in level_counts.columns:
        level_counts[LFNN_NAME] = 0

    if RFNN_NAME not in level_counts.columns:
        level_counts[RFNN_NAME] = 0

    level_counts[
        "RFNN_minus_LFNN"
    ] = (
        level_counts[RFNN_NAME]
        - level_counts[LFNN_NAME]
    )

    level_counts[
        "RFNN_to_LFNN_ratio"
    ] = (
        level_counts[RFNN_NAME]
        / level_counts[LFNN_NAME]
    )

    print(
        level_counts.to_string(
            index=False
        )
    )

    level_counts.to_csv(
        TABLE_DIR
        / "part232_level_balance.csv",
        index=False,
    )

    # ========================================================
    # 3. STUDY / SERIES COUNTS
    # ========================================================

    print_section(
        "3. STUDY/SERIES DISTRIBUTION"
    )

    series_class_counts = (
        foraminal
        .groupby(
            [
                "study_id",
                "series_id",
                "class_name",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    if LFNN_NAME not in series_class_counts.columns:
        series_class_counts[LFNN_NAME] = 0

    if RFNN_NAME not in series_class_counts.columns:
        series_class_counts[RFNN_NAME] = 0

    series_class_counts[
        "has_lfnn"
    ] = (
        series_class_counts[LFNN_NAME]
        > 0
    )

    series_class_counts[
        "has_rfnn"
    ] = (
        series_class_counts[RFNN_NAME]
        > 0
    )

    series_class_counts[
        "paired_series"
    ] = (
        series_class_counts["has_lfnn"]
        &
        series_class_counts["has_rfnn"]
    )

    print(
        f"Unique study/series pairs: "
        f"{len(series_class_counts)}"
    )

    print(
        f"Series with LFNN: "
        f"{series_class_counts['has_lfnn'].sum()}"
    )

    print(
        f"Series with RFNN: "
        f"{series_class_counts['has_rfnn'].sum()}"
    )

    print(
        f"Series with BOTH: "
        f"{series_class_counts['paired_series'].sum()}"
    )

    series_class_counts.to_csv(
        TABLE_DIR
        / "part232_series_distribution.csv",
        index=False,
    )

    # ========================================================
    # 4. PAIRED LEVEL SUPERVISION
    # ========================================================

    print_section(
        "4. SAME-SERIES SAME-LEVEL LFNN/RFNN PAIRS"
    )

    level_series = (
        foraminal
        .groupby(
            [
                "study_id",
                "series_id",
                "level",
            ]
        )["class_id"]
        .agg(
            lambda x: set(
                x.astype(int)
            )
        )
        .reset_index()
    )

    level_series[
        "has_lfnn"
    ] = level_series[
        "class_id"
    ].apply(
        lambda x: LFNN in x
    )

    level_series[
        "has_rfnn"
    ] = level_series[
        "class_id"
    ].apply(
        lambda x: RFNN in x
    )

    level_series[
        "paired"
    ] = (
        level_series["has_lfnn"]
        &
        level_series["has_rfnn"]
    )

    print(
        f"Same-series same-level groups: "
        f"{len(level_series)}"
    )

    print(
        f"Paired LFNN/RFNN groups: "
        f"{level_series['paired'].sum()}"
    )

    level_series.to_csv(
        TABLE_DIR
        / "part232_paired_level_groups.csv",
        index=False,
    )

    # ========================================================
    # 5. SPATIAL DISTRIBUTION
    # ========================================================

    print_section(
        "5. SPATIAL DISTRIBUTION"
    )

    spatial_rows = []

    for class_id, class_name in [
        (LFNN, LFNN_NAME),
        (RFNN, RFNN_NAME),
    ]:

        subset = foraminal[
            foraminal["class_id"]
            == class_id
        ].copy()

        row = {
            "class_id":
                class_id,

            "class_name":
                class_name,

            "points":
                len(subset),
        }

        for column in [
            "native_z",
            "native_y",
            "native_x",
            "model_z_float",
            "model_y_float",
            "model_x_float",
        ]:

            values = numeric_series(
                subset,
                column,
            ).dropna()

            row[
                f"{column}_mean"
            ] = values.mean()

            row[
                f"{column}_std"
            ] = values.std()

            row[
                f"{column}_min"
            ] = values.min()

            row[
                f"{column}_max"
            ] = values.max()

        spatial_rows.append(
            row
        )

    spatial_df = pd.DataFrame(
        spatial_rows
    )

    print(
        spatial_df.to_string(
            index=False
        )
    )

    spatial_df.to_csv(
        TABLE_DIR
        / "part232_spatial_distribution.csv",
        index=False,
    )

    # ========================================================
    # 6. BOUNDARY PROXIMITY
    # ========================================================

    print_section(
        "6. MODEL-GRID BOUNDARY PROXIMITY"
    )

    boundary_rows = []

    for class_id, class_name in [
        (LFNN, LFNN_NAME),
        (RFNN, RFNN_NAME),
    ]:

        subset = foraminal[
            foraminal["class_id"]
            == class_id
        ].copy()

        distances = []

        for _, row in subset.iterrows():

            z = safe_float(
                row["model_z_float"]
            )

            y = safe_float(
                row["model_y_float"]
            )

            x = safe_float(
                row["model_x_float"]
            )

            if any(
                pd.isna(v)
                for v in [
                    z,
                    y,
                    x,
                ]
            ):
                continue

            # Model grid is known from Part 2.13:
            # z: 0-63
            # y: 0-87
            # x: 0-73

            distance = min(
                z,
                63.0 - z,
                y,
                87.0 - y,
                x,
                73.0 - x,
            )

            distances.append(
                distance
            )

        distances = np.asarray(
            distances,
            dtype=np.float32,
        )

        boundary_rows.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    class_name,

                "points":
                    len(distances),

                "mean_boundary_distance":
                    float(
                        distances.mean()
                    ),

                "median_boundary_distance":
                    float(
                        np.median(
                            distances
                        )
                    ),

                "min_boundary_distance":
                    float(
                        distances.min()
                    ),

                "points_within_2_voxels":
                    int(
                        np.sum(
                            distances <= 2
                        )
                    ),

                "points_within_4_voxels":
                    int(
                        np.sum(
                            distances <= 4
                        )
                    ),
            }
        )

    boundary_df = pd.DataFrame(
        boundary_rows
    )

    print(
        boundary_df.to_string(
            index=False
        )
    )

    boundary_df.to_csv(
        TABLE_DIR
        / "part232_boundary_proximity.csv",
        index=False,
    )

    # ========================================================
    # 7. NEIGHBORING POINT DENSITY
    # ========================================================

    print_section(
        "7. NEIGHBORING ANNOTATION DENSITY"
    )

    coordinates = (
        foraminal[
            [
                "study_id",
                "series_id",
                "level",
                "class_id",
                "model_z_float",
                "model_y_float",
                "model_x_float",
            ]
        ].copy()
    )

    for column in [
        "model_z_float",
        "model_y_float",
        "model_x_float",
    ]:

        coordinates[column] = pd.to_numeric(
            coordinates[column],
            errors="coerce",
        )

    neighbor_rows = []

    for idx, row in coordinates.iterrows():

        same_series = coordinates[
            (
                coordinates["study_id"]
                == row["study_id"]
            )
            &
            (
                coordinates["series_id"]
                == row["series_id"]
            )
        ].drop(
            index=idx,
            errors="ignore",
        )

        if same_series.empty:

            neighbor_rows.append(
                {
                    "index": idx,
                    "class_id": row["class_id"],
                    "neighbors_within_5": 0,
                    "neighbors_within_10": 0,
                    "neighbors_within_15": 0,
                }
            )

            continue

        dz = (
            same_series["model_z_float"]
            - row["model_z_float"]
        )

        dy = (
            same_series["model_y_float"]
            - row["model_y_float"]
        )

        dx = (
            same_series["model_x_float"]
            - row["model_x_float"]
        )

        distance = np.sqrt(
            dz ** 2
            + dy ** 2
            + dx ** 2
        )

        neighbor_rows.append(
            {
                "index": idx,

                "class_id":
                    int(row["class_id"]),

                "neighbors_within_5":
                    int(
                        np.sum(
                            distance <= 5
                        )
                    ),

                "neighbors_within_10":
                    int(
                        np.sum(
                            distance <= 10
                        )
                    ),

                "neighbors_within_15":
                    int(
                        np.sum(
                            distance <= 15
                        )
                    ),
            }
        )

    neighbor_df = pd.DataFrame(
        neighbor_rows
    )

    neighbor_summary = (
        neighbor_df
        .groupby(
            "class_id"
        )[
            [
                "neighbors_within_5",
                "neighbors_within_10",
                "neighbors_within_15",
            ]
        ]
        .mean()
        .reset_index()
    )

    neighbor_summary[
        "class_name"
    ] = (
        neighbor_summary[
            "class_id"
        ]
        .map(
            {
                LFNN: LFNN_NAME,
                RFNN: RFNN_NAME,
            }
        )
    )

    print(
        neighbor_summary.to_string(
            index=False
        )
    )

    neighbor_summary.to_csv(
        TABLE_DIR
        / "part232_neighbor_density.csv",
        index=False,
    )

    # ========================================================
    # 8. CLASS × LEVEL MATRIX
    # ========================================================

    print_section(
        "8. CLASS × LEVEL MATRIX"
    )

    class_level = (
        foraminal
        .groupby(
            [
                "level",
                "class_id",
                "class_name",
            ]
        )
        .size()
        .reset_index(
            name="points"
        )
    )

    print(
        class_level.to_string(
            index=False
        )
    )

    class_level.to_csv(
        TABLE_DIR
        / "part232_class_level_matrix.csv",
        index=False,
    )

    # ========================================================
    # 9. SAMPLING EXPOSURE AUDIT
    # ========================================================

    print_section(
        "9. SAMPLING EXPOSURE AUDIT"
    )

    # Part 2.27 configuration from the actual experiment:
    # RFNN extra sampling weight = 2.0
    # RSS extra sampling weight = 1.5
    #
    # LFNN had no additional positive sampling weight.

    sampling_rows = [

        {
            "class_id": 2,
            "class_name":
                LFNN_NAME,
            "point_count":
                int(
                    (
                        foraminal["class_id"]
                        == LFNN
                    ).sum()
                ),
            "extra_case_sampling_weight":
                1.0,
        },

        {
            "class_id": 3,
            "class_name":
                RFNN_NAME,
            "point_count":
                int(
                    (
                        foraminal["class_id"]
                        == RFNN
                    ).sum()
                ),
            "extra_case_sampling_weight":
                2.0,
        },
    ]

    sampling_df = pd.DataFrame(
        sampling_rows
    )

    sampling_df[
        "weighted_point_exposure"
    ] = (
        sampling_df["point_count"]
        *
        sampling_df[
            "extra_case_sampling_weight"
        ]
    )

    sampling_df[
        "weighted_exposure_percentage"
    ] = (
        sampling_df[
            "weighted_point_exposure"
        ]
        /
        sampling_df[
            "weighted_point_exposure"
        ].sum()
        * 100.0
    )

    print(
        sampling_df.to_string(
            index=False
        )
    )

    sampling_df.to_csv(
        TABLE_DIR
        / "part232_sampling_exposure.csv",
        index=False,
    )

    # ========================================================
    # 10. REPORT
    # ========================================================

    report = []

    report.append(
        "PART 2.32 — LFNN/RFNN "
        "SUPERVISION & SAMPLING AUDIT"
    )

    report.append("")

    report.append(
        f"Total manifest rows: "
        f"{len(manifest)}"
    )

    report.append(
        f"LFNN/RFNN rows: "
        f"{len(foraminal)}"
    )

    report.append("")

    report.append(
        "GLOBAL COUNTS:"
    )

    report.append(
        global_counts.to_string(
            index=False
        )
    )

    report.append("")

    report.append(
        "LEVEL BALANCE:"
    )

    report.append(
        level_counts.to_string(
            index=False
        )
    )

    report.append("")

    report.append(
        "SERIES DISTRIBUTION:"
    )

    report.append(
        f"Unique series: "
        f"{len(series_class_counts)}"
    )

    report.append(
        f"Series with LFNN: "
        f"{series_class_counts['has_lfnn'].sum()}"
    )

    report.append(
        f"Series with RFNN: "
        f"{series_class_counts['has_rfnn'].sum()}"
    )

    report.append(
        f"Series with both: "
        f"{series_class_counts['paired_series'].sum()}"
    )

    report.append("")

    report.append(
        "PAIRED SAME-SERIES/SAME-LEVEL GROUPS:"
    )

    report.append(
        f"{level_series['paired'].sum()}"
    )

    report.append("")

    report.append(
        "SPATIAL DISTRIBUTION:"
    )

    report.append(
        spatial_df.to_string(
            index=False
        )
    )

    report.append("")

    report.append(
        "BOUNDARY PROXIMITY:"
    )

    report.append(
        boundary_df.to_string(
            index=False
        )
    )

    report.append("")

    report.append(
        "NEIGHBOR DENSITY:"
    )

    report.append(
        neighbor_summary.to_string(
            index=False
        )
    )

    report.append("")

    report.append(
        "SAMPLING EXPOSURE:"
    )

    report.append(
        sampling_df.to_string(
            index=False
        )
    )

    report.append("")

    report.append(
        "IMPORTANT:"
    )

    report.append(
        "This is an analysis-only audit."
    )

    report.append(
        "No training was performed."
    )

    report.append(
        "No checkpoint was modified."
    )

    report.append(
        "No dashboard was modified."
    )

    report.append(
        "No voxel-wise ground truth was fabricated."
    )

    report.append("")

    report.append(
        "RSNA annotations remain point/localization "
        "annotations, not manual voxel segmentation masks."
    )

    report_path = (
        REPORT_DIR
        / "part232_report.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL
    # ========================================================

    print_section(
        "PART 2.32 COMPLETE"
    )

    print(
        f"Total manifest rows: "
        f"{len(manifest)}"
    )

    print(
        f"LFNN/RFNN rows: "
        f"{len(foraminal)}"
    )

    print(
        f"Paired same-series/same-level groups: "
        f"{level_series['paired'].sum()}"
    )

    print(
        f"Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Training performed: NO"
    )

    print(
        "Checkpoint modified: NO"
    )

    print(
        "Dashboard modified: NO"
    )


if __name__ == "__main__":
    main()
