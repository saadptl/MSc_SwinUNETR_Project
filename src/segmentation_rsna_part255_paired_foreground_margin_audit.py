"""
PART 2.55
PAIRED LFNN / RFNN FOREGROUND-BACKGROUND MARGIN AUDIT

Purpose
-------
Use the exact Part 2.54 paired LFNN/RFNN records to determine why RFNN
loses foreground-vs-background competition.

Analysis only.

No training.
No checkpoint modification.
No dashboard modification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part254_paired_lfnn_rfnn_displacement_symmetry_audit"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part255_paired_foreground_margin_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INPUT_FILE = (
    INPUT_DIR
    / "part254_paired_displacement_records.csv"
)


# ============================================================================
# CONSTANTS
# ============================================================================

LFNN = "Left Neural Foraminal Narrowing"

RFNN = "Right Neural Foraminal Narrowing"

EXPECTED_PAIRS = 45


# ============================================================================
# LOAD
# ============================================================================

def load_part254():

    print()
    print("=" * 80)
    print("LOADING PART 2.54 PAIRED RECORDS")
    print("=" * 80)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Part 2.54 records not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Part 2.54 records: {len(df)}"
    )

    if len(df) != 90:

        raise RuntimeError(
            "Expected 90 records "
            f"(45 LFNN + 45 RFNN), "
            f"found {len(df)}."
        )

    required = {
        "study_id",
        "series_id",
        "level",
        "class_name",
        "physical_distance_mm",
        "inplane_distance_mm",
        "row_component_mm",
        "column_component_mm",
        "slice_normal_component_mm",
        "canonical_distance_voxels",
        "maximum_class_probability",
        "maximum_class_minus_background",
        "intensity_difference",
        "maximum_background_probability",
        "maximum_scs_probability",
        "maximum_lss_probability",
        "maximum_rfnn_probability",
        "maximum_lfnn_probability",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Missing required Part 2.54 columns:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    lfnn_count = int(
        (
            df["class_name"]
            == LFNN
        ).sum()
    )

    rfnn_count = int(
        (
            df["class_name"]
            == RFNN
        ).sum()
    )

    print(
        f"LFNN records: {lfnn_count}"
    )

    print(
        f"RFNN records: {rfnn_count}"
    )

    if lfnn_count != 45:

        raise RuntimeError(
            "Expected 45 LFNN records."
        )

    if rfnn_count != 45:

        raise RuntimeError(
            "Expected 45 RFNN records."
        )

    return df


# ============================================================================
# BUILD EXACT PAIRS
# ============================================================================

def build_pairs(df):

    lfnn = df[
        df["class_name"]
        == LFNN
    ].copy()

    rfnn = df[
        df["class_name"]
        == RFNN
    ].copy()

    pairs = lfnn.merge(
        rfnn,
        on=[
            "study_id",
            "series_id",
            "level",
        ],
        suffixes=(
            "_lfnn",
            "_rfnn",
        ),
    )

    print()
    print("=" * 80)
    print("BUILDING PAIRED LFNN / RFNN OBSERVATIONS")
    print("=" * 80)

    print(
        f"Paired observations: {len(pairs)}"
    )

    if len(pairs) != EXPECTED_PAIRS:

        raise RuntimeError(
            f"Expected {EXPECTED_PAIRS} paired observations "
            f"but found {len(pairs)}."
        )

    return pairs


# ============================================================================
# CALCULATE COMPETITION MARGINS
# ============================================================================

def add_competition_metrics(
    pairs,
):

    result = pairs.copy()

    # ------------------------------------------------------------------------
    # The class probability at the evidence maximum.
    # ------------------------------------------------------------------------

    result[
        "lfnn_minus_background"
    ] = (
        result[
            "maximum_class_probability_lfnn"
        ]
        -
        result[
            "maximum_background_probability_lfnn"
        ]
    )

    result[
        "rfnn_minus_background"
    ] = (
        result[
            "maximum_class_probability_rfnn"
        ]
        -
        result[
            "maximum_background_probability_rfnn"
        ]
    )

    # ------------------------------------------------------------------------
    # Explicit cross-foraminal competition.
    # ------------------------------------------------------------------------

    result[
        "lfnn_minus_rfnn"
    ] = (
        result[
            "maximum_lfnn_probability_lfnn"
        ]
        -
        result[
            "maximum_rfnn_probability_lfnn"
        ]
    )

    result[
        "rfnn_minus_lfnn"
    ] = (
        result[
            "maximum_rfnn_probability_rfnn"
        ]
        -
        result[
            "maximum_lfnn_probability_rfnn"
        ]
    )

    # ------------------------------------------------------------------------
    # LSS / SCS competition.
    # ------------------------------------------------------------------------

    result[
        "lfnn_minus_lss"
    ] = (
        result[
            "maximum_lfnn_probability_lfnn"
        ]
        -
        result[
            "maximum_lss_probability_lfnn"
        ]
    )

    result[
        "rfnn_minus_lss"
    ] = (
        result[
            "maximum_rfnn_probability_rfnn"
        ]
        -
        result[
            "maximum_lss_probability_rfnn"
        ]
    )

    result[
        "lfnn_minus_scs"
    ] = (
        result[
            "maximum_lfnn_probability_lfnn"
        ]
        -
        result[
            "maximum_scs_probability_lfnn"
        ]
    )

    result[
        "rfnn_minus_scs"
    ] = (
        result[
            "maximum_rfnn_probability_rfnn"
        ]
        -
        result[
            "maximum_scs_probability_rfnn"
        ]
    )

    # ------------------------------------------------------------------------
    # Foreground probability.
    #
    # 1 - background.
    # ------------------------------------------------------------------------

    result[
        "lfnn_foreground_probability"
    ] = (
        1.0
        -
        result[
            "maximum_background_probability_lfnn"
        ]
    )

    result[
        "rfnn_foreground_probability"
    ] = (
        1.0
        -
        result[
            "maximum_background_probability_rfnn"
        ]
    )

    # ------------------------------------------------------------------------
    # RFNN-LFNN paired differences.
    # ------------------------------------------------------------------------

    result[
        "rfnn_background_margin_minus_lfnn"
    ] = (
        result[
            "rfnn_minus_background"
        ]
        -
        result[
            "lfnn_minus_background"
        ]
    )

    result[
        "rfnn_foreground_minus_lfnn_foreground"
    ] = (
        result[
            "rfnn_foreground_probability"
        ]
        -
        result[
            "lfnn_foreground_probability"
        ]
    )

    result[
        "rfnn_probability_minus_lfnn_probability"
    ] = (
        result[
            "maximum_class_probability_rfnn"
        ]
        -
        result[
            "maximum_class_probability_lfnn"
        ]
    )

    # ------------------------------------------------------------------------
    # Margin categories.
    # ------------------------------------------------------------------------

    result[
        "lfnn_foreground_wins"
    ] = (
        result[
            "lfnn_minus_background"
        ]
        > 0
    )

    result[
        "rfnn_foreground_wins"
    ] = (
        result[
            "rfnn_minus_background"
        ]
        > 0
    )

    result[
        "rfnn_background_wins"
    ] = (
        result[
            "rfnn_minus_background"
        ]
        < 0
    )

    return result


# ============================================================================
# GLOBAL SUMMARY
# ============================================================================

def build_global_summary(
    pairs,
):

    metrics = [
        (
            "maximum_class_probability",
            "maximum_class_probability_lfnn",
            "maximum_class_probability_rfnn",
        ),
        (
            "class_minus_background",
            "lfnn_minus_background",
            "rfnn_minus_background",
        ),
        (
            "foreground_probability",
            "lfnn_foreground_probability",
            "rfnn_foreground_probability",
        ),
        (
            "class_minus_lss",
            "lfnn_minus_lss",
            "rfnn_minus_lss",
        ),
        (
            "class_minus_scs",
            "lfnn_minus_scs",
            "rfnn_minus_scs",
        ),
        (
            "class_minus_opposite_foraminal",
            "lfnn_minus_rfnn",
            "rfnn_minus_lfnn",
        ),
        (
            "physical_distance_mm",
            "physical_distance_mm_lfnn",
            "physical_distance_mm_rfnn",
        ),
        (
            "inplane_distance_mm",
            "inplane_distance_mm_lfnn",
            "inplane_distance_mm_rfnn",
        ),
        (
            "intensity_difference",
            "intensity_difference_lfnn",
            "intensity_difference_rfnn",
        ),
    ]

    rows = []

    for name, lcol, rcol in metrics:

        left = pairs[
            lcol
        ].astype(float)

        right = pairs[
            rcol
        ].astype(float)

        difference = (
            right
            - left
        )

        rows.append(
            {
                "metric":
                    name,

                "lfnn_mean":
                    float(
                        left.mean()
                    ),

                "rfnn_mean":
                    float(
                        right.mean()
                    ),

                "rfnn_minus_lfnn":
                    float(
                        difference.mean()
                    ),

                "absolute_pair_difference":
                    float(
                        difference.abs().mean()
                    ),

                "lfnn_median":
                    float(
                        left.median()
                    ),

                "rfnn_median":
                    float(
                        right.median()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# COMPETITION SUMMARY
# ============================================================================

def build_competition_summary(
    pairs,
):

    rows = []

    for side in [
        "lfnn",
        "rfnn",
    ]:

        if side == "lfnn":

            margin = pairs[
                "lfnn_minus_background"
            ]

            class_probability = pairs[
                "maximum_class_probability_lfnn"
            ]

            background = pairs[
                "maximum_background_probability_lfnn"
            ]

            lss = pairs[
                "maximum_lss_probability_lfnn"
            ]

            scs = pairs[
                "maximum_scs_probability_lfnn"
            ]

            opposite = pairs[
                "maximum_rfnn_probability_lfnn"
            ]

        else:

            margin = pairs[
                "rfnn_minus_background"
            ]

            class_probability = pairs[
                "maximum_class_probability_rfnn"
            ]

            background = pairs[
                "maximum_background_probability_rfnn"
            ]

            lss = pairs[
                "maximum_lss_probability_rfnn"
            ]

            scs = pairs[
                "maximum_scs_probability_rfnn"
            ]

            opposite = pairs[
                "maximum_lfnn_probability_rfnn"
            ]

        rows.append(
            {
                "side":
                    side.upper(),

                "count":
                    len(margin),

                "mean_class_probability":
                    float(
                        class_probability.mean()
                    ),

                "mean_background_probability":
                    float(
                        background.mean()
                    ),

                "mean_foreground_probability":
                    float(
                        (
                            1.0
                            - background
                        ).mean()
                    ),

                "mean_class_background_margin":
                    float(
                        margin.mean()
                    ),

                "positive_margin_fraction":
                    float(
                        (
                            margin
                            > 0
                        ).mean()
                    ),

                "negative_margin_fraction":
                    float(
                        (
                            margin
                            < 0
                        ).mean()
                    ),

                "mean_lss_probability":
                    float(
                        lss.mean()
                    ),

                "mean_scs_probability":
                    float(
                        scs.mean()
                    ),

                "mean_opposite_foraminal_probability":
                    float(
                        opposite.mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# PAIRWISE MARGIN SUMMARY
# ============================================================================

def build_pairwise_summary(
    pairs,
):

    rows = []

    margin_columns = [
        (
            "background_margin",
            "lfnn_minus_background",
            "rfnn_minus_background",
        ),
        (
            "lss_margin",
            "lfnn_minus_lss",
            "rfnn_minus_lss",
        ),
        (
            "scs_margin",
            "lfnn_minus_scs",
            "rfnn_minus_scs",
        ),
        (
            "opposite_foraminal_margin",
            "lfnn_minus_rfnn",
            "rfnn_minus_lfnn",
        ),
    ]

    for name, lcol, rcol in margin_columns:

        left = pairs[
            lcol
        ].astype(float)

        right = pairs[
            rcol
        ].astype(float)

        diff = (
            right
            - left
        )

        rows.append(
            {
                "competition":
                    name,

                "lfnn_mean":
                    float(
                        left.mean()
                    ),

                "rfnn_mean":
                    float(
                        right.mean()
                    ),

                "rfnn_minus_lfnn":
                    float(
                        diff.mean()
                    ),

                "rfnn_better_fraction":
                    float(
                        (
                            diff
                            > 0
                        ).mean()
                    ),

                "lfnn_better_fraction":
                    float(
                        (
                            diff
                            < 0
                        ).mean()
                    ),

                "tie_fraction":
                    float(
                        (
                            diff
                            == 0
                        ).mean()
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
    pairs,
):

    rows = []

    for level in sorted(
        pairs["level"].unique()
    ):

        subset = pairs[
            pairs["level"]
            == level
        ]

        l_margin = subset[
            "lfnn_minus_background"
        ]

        r_margin = subset[
            "rfnn_minus_background"
        ]

        l_prob = subset[
            "maximum_class_probability_lfnn"
        ]

        r_prob = subset[
            "maximum_class_probability_rfnn"
        ]

        rows.append(
            {
                "level":
                    level,

                "count":
                    len(subset),

                "lfnn_class_probability":
                    float(
                        l_prob.mean()
                    ),

                "rfnn_class_probability":
                    float(
                        r_prob.mean()
                    ),

                "probability_gap_rfnn_minus_lfnn":
                    float(
                        (
                            r_prob
                            - l_prob
                        ).mean()
                    ),

                "lfnn_background_margin":
                    float(
                        l_margin.mean()
                    ),

                "rfnn_background_margin":
                    float(
                        r_margin.mean()
                    ),

                "margin_gap_rfnn_minus_lfnn":
                    float(
                        (
                            r_margin
                            - l_margin
                        ).mean()
                    ),

                "lfnn_positive_margin_fraction":
                    float(
                        (
                            l_margin
                            > 0
                        ).mean()
                    ),

                "rfnn_positive_margin_fraction":
                    float(
                        (
                            r_margin
                            > 0
                        ).mean()
                    ),

                "rfnn_background_wins_fraction":
                    float(
                        (
                            r_margin
                            < 0
                        ).mean()
                    ),

                "lfnn_mean_distance_mm":
                    float(
                        subset[
                            "physical_distance_mm_lfnn"
                        ].mean()
                    ),

                "rfnn_mean_distance_mm":
                    float(
                        subset[
                            "physical_distance_mm_rfnn"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# CORRELATION WITH DISPLACEMENT
# ============================================================================

def build_correlation_summary(
    pairs,
):

    rows = []

    variables = [
        (
            "physical_distance_mm",
            "physical_distance_mm_rfnn",
        ),
        (
            "inplane_distance_mm",
            "inplane_distance_mm_rfnn",
        ),
        (
            "row_component_mm",
            "row_component_mm_rfnn",
        ),
        (
            "column_component_mm",
            "column_component_mm_rfnn",
        ),
        (
            "slice_normal_component_mm",
            "slice_normal_component_mm_rfnn",
        ),
        (
            "canonical_distance_voxels",
            "canonical_distance_voxels_rfnn",
        ),
        (
            "intensity_difference",
            "intensity_difference_rfnn",
        ),
    ]

    target = pairs[
        "rfnn_minus_background"
    ].astype(float)

    for name, column in variables:

        values = pairs[
            column
        ].astype(float)

        if (
            values.nunique() > 1
            and target.nunique() > 1
        ):

            pearson = float(
                values.corr(
                    target,
                    method="pearson",
                )
            )

            spearman = float(
                values.corr(
                    target,
                    method="spearman",
                )
            )

        else:

            pearson = np.nan

            spearman = np.nan

        rows.append(
            {
                "variable":
                    name,

                "target":
                    "RFNN_class_minus_background",

                "pearson_r":
                    pearson,

                "spearman_r":
                    spearman,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# SAVE
# ============================================================================

def save_outputs(
    pairs,
    global_df,
    competition_df,
    pairwise_df,
    level_df,
    correlation_df,
):

    pairs_path = (
        OUTPUT_DIR
        / "part255_paired_margin_records.csv"
    )

    global_path = (
        OUTPUT_DIR
        / "part255_global_margin_summary.csv"
    )

    competition_path = (
        OUTPUT_DIR
        / "part255_foreground_competition_summary.csv"
    )

    pairwise_path = (
        OUTPUT_DIR
        / "part255_pairwise_competition_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part255_level_margin_summary.csv"
    )

    correlation_path = (
        OUTPUT_DIR
        / "part255_displacement_margin_correlations.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part255_paired_foreground_margin_audit_summary.json"
    )

    pairs.to_csv(
        pairs_path,
        index=False,
    )

    global_df.to_csv(
        global_path,
        index=False,
    )

    competition_df.to_csv(
        competition_path,
        index=False,
    )

    pairwise_df.to_csv(
        pairwise_path,
        index=False,
    )

    level_df.to_csv(
        level_path,
        index=False,
    )

    correlation_df.to_csv(
        correlation_path,
        index=False,
    )

    json_summary = {
        "paired_observations":
            int(
                len(pairs)
            ),

        "lfnn_points":
            45,

        "rfnn_points":
            45,

        "global_summary":
            global_df.to_dict(
                orient="records"
            ),

        "foreground_competition":
            competition_df.to_dict(
                orient="records"
            ),

        "pairwise_competition":
            pairwise_df.to_dict(
                orient="records"
            ),

        "level_summary":
            level_df.to_dict(
                orient="records"
            ),

        "correlation_summary":
            correlation_df.to_dict(
                orient="records"
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

    print()
    print("=" * 80)
    print("PART 2.55 GLOBAL FOREGROUND / BACKGROUND SUMMARY")
    print("=" * 80)

    print(
        global_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.55 FOREGROUND COMPETITION")
    print("=" * 80)

    print(
        competition_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.55 PAIRWISE COMPETITION")
    print("=" * 80)

    print(
        pairwise_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.55 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.55 DISPLACEMENT / MARGIN CORRELATIONS")
    print("=" * 80)

    print(
        correlation_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.55 COMPLETE")
    print("=" * 80)

    print(
        "Analysis only."
    )

    print(
        "Part 2.27 checkpoint unchanged."
    )

    print(
        "Part 2.54 unchanged."
    )

    print(
        "Dashboard unchanged."
    )

    print()
    print("Outputs:")

    print(pairs_path)
    print(global_path)
    print(competition_path)
    print(pairwise_path)
    print(level_path)
    print(correlation_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.55")
    print("PAIRED LFNN / RFNN FOREGROUND-BACKGROUND MARGIN AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    df = load_part254()

    pairs = build_pairs(
        df
    )

    pairs = add_competition_metrics(
        pairs
    )

    global_df = (
        build_global_summary(
            pairs
        )
    )

    competition_df = (
        build_competition_summary(
            pairs
        )
    )

    pairwise_df = (
        build_pairwise_summary(
            pairs
        )
    )

    level_df = (
        build_level_summary(
            pairs
        )
    )

    correlation_df = (
        build_correlation_summary(
            pairs
        )
    )

    save_outputs(
        pairs,
        global_df,
        competition_df,
        pairwise_df,
        level_df,
        correlation_df,
    )


if __name__ == "__main__":
    main()