"""
Part 2.39 — Foreground-vs-Background Logit Margin Audit

Purpose
-------
Analyze whether each disease class has sufficient foreground evidence
relative to the Background class at its annotated point.

This is an ANALYSIS-ONLY audit.

It uses:
    - Exact Part 2.20B validation cohort
    - Exact Part 2.20B physical/canonical preprocessing
    - Part 2.27 best checkpoint
    - Point-supervision annotations only
    - No voxel-level ground-truth masks

It does NOT:
    - train the model
    - modify any checkpoint
    - modify the dashboard
    - create fake segmentation masks

Main analysis
-------------
For every annotated validation point:

    target_logit - background_logit

is calculated.

The audit compares:

    SCS  vs Background
    LFNN vs Background
    RFNN vs Background
    LSS  vs Background
    RSS  vs Background

Special attention is given to the LFNN/RFNN paired observations.

Expected validation data:
    25 studies
    25 series
    188 annotated points
    45 LFNN points
    45 RFNN points
    45 paired LFNN/RFNN observations

Output
------
outputs/
    segmentation/
        rsna_part239_foreground_background_logit_margin_audit/
            part239_all_point_foreground_background_records.csv
            part239_lfnn_rfnn_paired_comparison.csv
            part239_class_foreground_background_summary.csv
            part239_level_foreground_background_summary.csv
            part239_foreground_background_audit_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part239_foreground_background_logit_margin_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
    / "part227_best_macro_disease.pth"
)


# ============================================================
# 2. IMPORT PART 2.20B
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import segmentation_rsna_part220b_geometry_corrected_training as part220b


# ============================================================
# 3. CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Background",
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]

CLASS_ID_TO_NAME = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

DISEASE_CLASS_IDS = [1, 2, 3, 4, 5]

DISEASE_SHORT = {
    1: "SCS",
    2: "LFNN",
    3: "RFNN",
    4: "LSS",
    5: "RSS",
}

EXPECTED_POINT_COUNTS = {
    "Spinal Canal Stenosis": 25,
    "Left Neural Foraminal Narrowing": 45,
    "Right Neural Foraminal Narrowing": 45,
    "Left Subarticular Stenosis": 42,
    "Right Subarticular Stenosis": 31,
}


# ============================================================
# 4. UTILITY FUNCTIONS
# ============================================================

def safe_float(value):
    """Convert a value to float safely."""
    try:
        return float(value)
    except Exception:
        return np.nan


def percentile_or_nan(values, percentile):
    """Return percentile or NaN for an empty collection."""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return np.nan

    return float(np.percentile(values, percentile))


def summarize_margin(values):
    """Return descriptive statistics for a margin array."""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "max": np.nan,
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
            "positive_rate": np.nan,
        }

    positive = int(np.sum(values > 0))
    negative = int(np.sum(values < 0))
    zero = int(np.sum(values == 0))

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": percentile_or_nan(values, 25),
        "p75": percentile_or_nan(values, 75),
        "max": float(np.max(values)),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "positive_rate": float(positive / values.size),
    }


def get_probability(logits):
    """Convert class logits into probabilities."""
    return torch.softmax(logits, dim=0)


# ============================================================
# 5. CHECKPOINT LOADING
# ============================================================

def load_model():
    print("\n" + "=" * 80)
    print("LOADING PART 2.27 CHECKPOINT")
    print("=" * 80)

    print(f"Checkpoint:")
    print(CHECKPOINT_PATH)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Part 2.27 checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    model = part220b.build_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    print(f"Missing keys    : {len(missing)}")
    print(f"Unexpected keys : {len(unexpected)}")

    if missing:
        print("WARNING — missing checkpoint keys:")
        print(missing[:20])

    if unexpected:
        print("WARNING — unexpected checkpoint keys:")
        print(unexpected[:20])

    model = model.to(DEVICE)
    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")
    print(f"Device          : {DEVICE}")

    return model


# ============================================================
# 6. EXACT VALIDATION COHORT
# ============================================================

def build_validation_point_table(manifest):
    """
    Reproduce the EXACT Part 2.20B validation cohort.

    Important:
    select_validation_series() returns a pandas DataFrame.
    Do NOT convert it with list(), because that would return
    the DataFrame column names.

    This follows the same validation construction used by
    Part 2.23 reproduction audit:

        1. select_validation_series(manifest)
        2. construct (study_id, series_id) validation keys
        3. filter the full manifest
        4. build_case_index(validation_manifest)
        5. preserve selected ordering

    Expected:
        25 validation cases
        188 annotated points
    """

    print("\n" + "=" * 80)
    print("BUILDING EXACT PART 2.20B VALIDATION COHORT")
    print("=" * 80)

    # ------------------------------------------------------------
    # EXACT Part 2.20B validation selection
    # ------------------------------------------------------------

    selected = part220b.select_validation_series(
        manifest
    )

    # IMPORTANT:
    # selected MUST remain a DataFrame.
    if not isinstance(selected, pd.DataFrame):
        raise TypeError(
            "Part 2.20B select_validation_series() was expected "
            "to return a pandas DataFrame, but returned: "
            f"{type(selected)}"
        )

    print(
        f"Part 2.20B validation series selected: "
        f"{len(selected)}"
    )

    print(
        "Part 2.20B validation studies selected: "
        f"{selected['study_id'].astype(str).nunique()}"
    )

    if len(selected) != 25:
        raise RuntimeError(
            "The Part 2.20B selector did not return the "
            "expected 25 validation series. "
            f"Found {len(selected)}."
        )

    # ------------------------------------------------------------
    # EXACT validation keys
    # ------------------------------------------------------------

    validation_keys = {
        (
            str(row["study_id"]),
            str(row["series_id"]),
        )
        for _, row in selected.iterrows()
    }

    print(
        f"Unique validation study/series keys: "
        f"{len(validation_keys)}"
    )

    if len(validation_keys) != 25:
        raise RuntimeError(
            "Expected 25 unique validation study/series "
            f"keys, found {len(validation_keys)}."
        )

    # ------------------------------------------------------------
    # EXACT validation manifest
    # ------------------------------------------------------------

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

    print(
        f"Validation manifest rows: "
        f"{len(validation_manifest)}"
    )

    # ------------------------------------------------------------
    # EXACT Part 2.20B case construction
    # ------------------------------------------------------------

    validation_cases = (
        part220b.build_case_index(
            validation_manifest
        )
    )

    print(
        f"Validation cases constructed: "
        f"{len(validation_cases)}"
    )

    if len(validation_cases) != 25:
        raise RuntimeError(
            "Part 2.20B build_case_index() did not produce "
            "the expected 25 validation cases. "
            f"Found {len(validation_cases)}."
        )

    # ------------------------------------------------------------
    # Preserve EXACT selected ordering.
    # This is copied from the successful Part 2.23 logic.
    # ------------------------------------------------------------

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
            10_000,
        ),
    )

    # ------------------------------------------------------------
    # Build point table directly from the case objects.
    #
    # This is safer than reconstructing points from arbitrary
    # series selection because build_case_index() already contains
    # the exact Part 2.20B point grouping.
    # ------------------------------------------------------------

    records = []

    for case in validation_cases:

        study_id = str(
            case["study_id"]
        )

        series_id = str(
            case["series_id"]
        )

        case_points = case["points"]

        # case["points"] is expected to be a DataFrame.
        if isinstance(case_points, pd.DataFrame):

            case_points_df = (
                case_points
                .reset_index(drop=True)
            )

        else:

            # Defensive conversion if the implementation returns
            # another tabular structure.
            case_points_df = pd.DataFrame(
                case_points
            ).reset_index(drop=True)

        for point_index, (_, row) in enumerate(
            case_points_df.iterrows()
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

                    "model_z_float": safe_float(
                        row["model_z_float"]
                    ),

                    "model_y_float": safe_float(
                        row["model_y_float"]
                    ),

                    "model_x_float": safe_float(
                        row["model_x_float"]
                    ),
                }
            )

    validation_points = pd.DataFrame(
        records
    )

    print(
        f"\nValidation point rows: "
        f"{len(validation_points)}"
    )

    # ------------------------------------------------------------
    # Disease count verification
    # ------------------------------------------------------------

    expected_order = [
        "Spinal Canal Stenosis",
        "Left Neural Foraminal Narrowing",
        "Right Neural Foraminal Narrowing",
        "Left Subarticular Stenosis",
        "Right Subarticular Stenosis",
    ]

    if not validation_points.empty:

        counts = (
            validation_points["class_name"]
            .value_counts()
            .reindex(
                expected_order,
                fill_value=0,
            )
        )

        print("\nDisease counts:")

        for disease, count in counts.items():

            print(
                f"  {disease:35s}: "
                f"{int(count)}"
            )

    # ------------------------------------------------------------
    # CRITICAL exact cohort check
    # ------------------------------------------------------------

    expected_counts = {
        "Spinal Canal Stenosis": 25,
        "Left Neural Foraminal Narrowing": 45,
        "Right Neural Foraminal Narrowing": 45,
        "Left Subarticular Stenosis": 42,
        "Right Subarticular Stenosis": 31,
    }

    actual_counts = (
        validation_points["class_name"]
        .value_counts()
        .to_dict()
    )

    for disease, expected in expected_counts.items():

        actual = int(
            actual_counts.get(
                disease,
                0,
            )
        )

        if actual != expected:

            raise RuntimeError(
                f"Validation cohort mismatch for "
                f"{disease}: expected {expected}, "
                f"found {actual}."
            )

    if len(validation_points) != 188:

        raise RuntimeError(
            "Exact Part 2.20B validation cohort mismatch: "
            f"expected 188 annotated points, "
            f"found {len(validation_points)}."
        )

    print("\nEXACT COHORT VERIFIED")
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

    return validation_cases, validation_points


# ============================================================
# 7. ANALYZE ONE CASE
# ============================================================

@torch.no_grad()
def analyze_case(
    model,
    study_id,
    series_id,
    case_rows,
):
    """
    Run the model once for one validation series and calculate
    foreground-vs-background margins at every annotated point.
    """

    canonical, transformed_points, geometry = (
        part220b.load_case(
            study_id,
            series_id,
            case_rows,
        )
    )

    if canonical is None:
        raise RuntimeError(
            f"Canonical image is None for "
            f"{study_id}/{series_id}"
        )

    image = canonical

    if not torch.is_tensor(image):
        image = torch.from_numpy(
            np.asarray(image)
        )

    image = image.float()

    # Expected shape is [D,H,W].
    if image.ndim == 3:
        image_tensor = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 4:
        image_tensor = image.unsqueeze(0)
    elif image.ndim == 5:
        image_tensor = image
    else:
        raise RuntimeError(
            f"Unexpected canonical image shape: "
            f"{tuple(image.shape)}"
        )

    image_tensor = image_tensor.to(DEVICE)

    output = model(image_tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if output.ndim != 5:
        raise RuntimeError(
            f"Unexpected model output shape: "
            f"{tuple(output.shape)}"
        )

    # --------------------------------------------------------
    # Verify positional mapping.
    # --------------------------------------------------------

    original_rows = case_rows.reset_index(drop=True)

    if len(original_rows) != len(transformed_points):
        raise RuntimeError(
            f"Point mapping mismatch for "
            f"{study_id}/{series_id}: "
            f"{len(original_rows)} original rows vs "
            f"{len(transformed_points)} transformed points"
        )

    records = []

    for point_index in range(len(original_rows)):

        original = original_rows.iloc[point_index]
        transformed = transformed_points[point_index]

        class_id = int(original["class_id"])
        class_name = str(original["class_name"])
        level = str(original["level"])

        z = int(
            np.clip(
                round(float(transformed["z"])),
                0,
                output.shape[2] - 1,
            )
        )

        y = int(
            np.clip(
                round(float(transformed["y"])),
                0,
                output.shape[3] - 1,
            )
        )

        x = int(
            np.clip(
                round(float(transformed["x"])),
                0,
                output.shape[4] - 1,
            )
        )

        point_logits = output[
            0,
            :,
            z,
            y,
            x,
        ]

        probabilities = get_probability(point_logits)

        logits_np = (
            point_logits
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

        predicted_class_id = int(
            np.argmax(logits_np)
        )

        background_logit = float(
            logits_np[0]
        )

        target_logit = float(
            logits_np[class_id]
        )

        target_background_margin = (
            target_logit - background_logit
        )

        target_probability = float(
            probabilities_np[class_id]
        )

        background_probability = float(
            probabilities_np[0]
        )

        target_wins_background = (
            target_logit > background_logit
        )

        record = {
            "study_id": str(study_id),
            "series_id": str(series_id),
            "point_index": int(point_index),

            "class_id": class_id,
            "class_name": class_name,
            "disease_short": DISEASE_SHORT.get(
                class_id,
                class_name,
            ),
            "level": level,

            "native_z": safe_float(
                original["native_z"]
            ),
            "native_y": safe_float(
                original["native_y"]
            ),
            "native_x": safe_float(
                original["native_x"]
            ),

            "model_z_float": safe_float(
                original["model_z_float"]
            ),
            "model_y_float": safe_float(
                original["model_y_float"]
            ),
            "model_x_float": safe_float(
                original["model_x_float"]
            ),

            "sample_z": z,
            "sample_y": y,
            "sample_x": x,

            "background_logit": background_logit,
            "target_logit": target_logit,

            "target_background_logit_margin":
                float(target_background_margin),

            "target_probability":
                target_probability,

            "background_probability":
                background_probability,

            "target_wins_background":
                bool(target_wins_background),

            "predicted_class_id":
                predicted_class_id,

            "predicted_class_name":
                CLASS_ID_TO_NAME[
                    predicted_class_id
                ],
        }

        # ----------------------------------------------------
        # Store every foreground-vs-background margin.
        # This lets us compare all disease classes at their
        # annotated points.
        # ----------------------------------------------------

        for disease_id in DISEASE_CLASS_IDS:

            short_name = DISEASE_SHORT[disease_id]

            disease_logit = float(
                logits_np[disease_id]
            )

            disease_probability = float(
                probabilities_np[disease_id]
            )

            record[
                f"{short_name}_logit"
            ] = disease_logit

            record[
                f"{short_name}_probability"
            ] = disease_probability

            record[
                f"{short_name}_minus_background_logit"
            ] = (
                disease_logit
                - background_logit
            )

            record[
                f"{short_name}_wins_background"
            ] = bool(
                disease_logit > background_logit
            )

        records.append(record)

    return records


# ============================================================
# 8. CLASS-WISE SUMMARY
# ============================================================

def create_class_summary(records_df):

    rows = []

    for class_id in DISEASE_CLASS_IDS:

        class_name = CLASS_ID_TO_NAME[class_id]
        short_name = DISEASE_SHORT[class_id]

        subset = records_df[
            records_df["class_id"] == class_id
        ].copy()

        margins = subset[
            f"{short_name}_minus_background_logit"
        ].to_numpy(dtype=float)

        target_prob = subset[
            "target_probability"
        ].to_numpy(dtype=float)

        background_prob = subset[
            "background_probability"
        ].to_numpy(dtype=float)

        stats = summarize_margin(margins)

        predicted_target_count = int(
            np.sum(
                subset["predicted_class_id"].to_numpy()
                == class_id
            )
        )

        rows.append(
            {
                "class_id": class_id,
                "disease": class_name,
                "short_name": short_name,

                "count": stats["count"],

                "mean_target_minus_background_logit":
                    stats["mean"],

                "median_target_minus_background_logit":
                    stats["median"],

                "std_target_minus_background_logit":
                    stats["std"],

                "min_target_minus_background_logit":
                    stats["min"],

                "p25_target_minus_background_logit":
                    stats["p25"],

                "p75_target_minus_background_logit":
                    stats["p75"],

                "max_target_minus_background_logit":
                    stats["max"],

                "positive_margin_count":
                    stats["positive_count"],

                "negative_margin_count":
                    stats["negative_count"],

                "zero_margin_count":
                    stats["zero_count"],

                "foreground_over_background_win_rate":
                    stats["positive_rate"],

                "mean_target_probability":
                    float(np.mean(target_prob))
                    if len(target_prob)
                    else np.nan,

                "mean_background_probability":
                    float(np.mean(background_prob))
                    if len(background_prob)
                    else np.nan,

                "predicted_as_target_count":
                    predicted_target_count,

                "predicted_as_target_rate":
                    float(
                        predicted_target_count
                        / len(subset)
                    )
                    if len(subset)
                    else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 9. LEVEL-WISE SUMMARY
# ============================================================

def create_level_summary(records_df):

    rows = []

    for class_id in DISEASE_CLASS_IDS:

        class_name = CLASS_ID_TO_NAME[class_id]
        short_name = DISEASE_SHORT[class_id]

        class_subset = records_df[
            records_df["class_id"] == class_id
        ]

        for level in sorted(
            class_subset["level"].unique()
        ):

            subset = class_subset[
                class_subset["level"] == level
            ]

            margins = subset[
                f"{short_name}_minus_background_logit"
            ].to_numpy(dtype=float)

            stats = summarize_margin(margins)

            rows.append(
                {
                    "class_id": class_id,
                    "disease": class_name,
                    "short_name": short_name,
                    "level": level,
                    "count": stats["count"],
                    "mean_margin": stats["mean"],
                    "median_margin": stats["median"],
                    "std_margin": stats["std"],
                    "min_margin": stats["min"],
                    "max_margin": stats["max"],
                    "positive_count":
                        stats["positive_count"],
                    "positive_rate":
                        stats["positive_rate"],
                    "mean_target_probability":
                        float(
                            subset[
                                "target_probability"
                            ].mean()
                        ),
                    "mean_background_probability":
                        float(
                            subset[
                                "background_probability"
                            ].mean()
                        ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 10. LFNN/RFNN PAIRED COMPARISON
# ============================================================

def create_lfnn_rfnn_pairs(records_df):

    lfnn = records_df[
        records_df["class_id"] == 2
    ].copy()

    rfnn = records_df[
        records_df["class_id"] == 3
    ].copy()

    merge_keys = [
        "study_id",
        "series_id",
        "level",
    ]

    paired = lfnn.merge(
        rfnn,
        on=merge_keys,
        how="inner",
        suffixes=("_lfnn", "_rfnn"),
    )

    if paired.empty:
        return pd.DataFrame()

    output = pd.DataFrame()

    output["study_id"] = paired[
        "study_id"
    ]

    output["series_id"] = paired[
        "series_id"
    ]

    output["level"] = paired[
        "level"
    ]

    output[
        "lfnn_target_minus_background_logit"
    ] = paired[
        "LFNN_minus_background_logit_lfnn"
    ]

    output[
        "rfnn_target_minus_background_logit"
    ] = paired[
        "RFNN_minus_background_logit_rfnn"
    ]

    output[
        "rfnn_minus_lfnn_margin_difference"
    ] = (
        output[
            "rfnn_target_minus_background_logit"
        ]
        -
        output[
            "lfnn_target_minus_background_logit"
        ]
    )

    output[
        "lfnn_target_probability"
    ] = paired[
        "target_probability_lfnn"
    ]

    output[
        "rfnn_target_probability"
    ] = paired[
        "target_probability_rfnn"
    ]

    output[
        "lfnn_background_probability"
    ] = paired[
        "background_probability_lfnn"
    ]

    output[
        "rfnn_background_probability"
    ] = paired[
        "background_probability_rfnn"
    ]

    output[
        "lfnn_target_wins_background"
    ] = paired[
        "LFNN_wins_background_lfnn"
    ]

    output[
        "rfnn_target_wins_background"
    ] = paired[
        "RFNN_wins_background_rfnn"
    ]

    return output


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("PART 2.39")
    print("FOREGROUND-vs-BACKGROUND LOGIT MARGIN AUDIT")
    print("=" * 80)

    print("\nThis is an ANALYSIS-ONLY experiment.")
    print("No training will be performed.")
    print("No checkpoint will be modified.")
    print("No dashboard files will be modified.")

    # --------------------------------------------------------
    # Load exact manifest.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("LOADING EXACT PART 2.20B MANIFEST")
    print("=" * 80)

    manifest = part220b.load_manifest()

    print(f"Full manifest rows: {len(manifest)}")

    # --------------------------------------------------------
    # Validation cohort.
    # --------------------------------------------------------

    validation_cases, validation_points = (
    build_validation_point_table(
        manifest
    )
)

    expected_total = sum(
        EXPECTED_POINT_COUNTS.values()
    )

    if len(validation_points) != expected_total:
        print(
            "\nWARNING:"
            f" expected {expected_total} validation points "
            f"but found {len(validation_points)}."
        )

    # --------------------------------------------------------
    # Load checkpoint.
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Process each validation series.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RUNNING FOREGROUND/BACKGROUND AUDIT")
    print("=" * 80)

    all_records = []

    successful_cases = 0
    failed_cases = []

    for case_number, item in enumerate(
        validation_cases,
        start=1,
    ):

        if isinstance(item, dict):
            study_id = str(item["study_id"])
            series_id = str(item["series_id"])
        else:
            study_id = str(item[0])
            series_id = str(item[1])

        print(
            f"\n[{case_number:02d}/{len(validation_cases):02d}] "
            f"Study={study_id} | Series={series_id}"
        )

        case_rows = validation_points[
            (validation_points["study_id"] == study_id)
            & (validation_points["series_id"] == series_id)
        ].copy()

        if case_rows.empty:
            print("  No validation points — skipped.")
            continue

        # Remove audit-only columns before sending back to
        # Part 2.20B. Keep the original manifest columns.
        original_case_rows = manifest[
            (manifest["study_id"].astype(str) == study_id)
            & (manifest["series_id"].astype(str) == series_id)
        ].copy()

        original_case_rows = (
            original_case_rows.reset_index(drop=True)
        )

        try:

            records = analyze_case(
                model=model,
                study_id=study_id,
                series_id=series_id,
                case_rows=original_case_rows,
            )

            all_records.extend(records)

            successful_cases += 1

            print(
                f"  Image/model analysis successful."
            )
            print(
                f"  Annotated points: {len(records)}"
            )

        except Exception as exc:

            failed_cases.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "error": repr(exc),
                }
            )

            print(
                f"  FAILED: {repr(exc)}"
            )

    # --------------------------------------------------------
    # Build final DataFrame.
    # --------------------------------------------------------

    records_df = pd.DataFrame(all_records)

    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    print(
        f"Validation cases expected : "
        f"{len(validation_cases)}"
    )

    print(
        f"Validation cases succeeded: "
        f"{successful_cases}"
    )

    print(
        f"Validation cases failed   : "
        f"{len(failed_cases)}"
    )

    print(
        f"Point records generated    : "
        f"{len(records_df)}"
    )

    # --------------------------------------------------------
    # Save failed cases if any.
    # --------------------------------------------------------

    if failed_cases:

        failed_df = pd.DataFrame(
            failed_cases
        )

        failed_df.to_csv(
            OUTPUT_DIR / "part239_failed_cases.csv",
            index=False,
        )

        print(
            "\nWARNING: failed cases were saved to:"
        )

        print(
            OUTPUT_DIR
            / "part239_failed_cases.csv"
        )

    # --------------------------------------------------------
    # Validate expected counts.
    # --------------------------------------------------------

    if not records_df.empty:

        actual_counts = (
            records_df["class_name"]
            .value_counts()
            .to_dict()
        )

        print("\nActual point counts:")

        for disease, expected in (
            EXPECTED_POINT_COUNTS.items()
        ):

            actual = int(
                actual_counts.get(
                    disease,
                    0,
                )
            )

            status = (
                "OK"
                if actual == expected
                else "CHECK"
            )

            print(
                f"  {disease:35s} "
                f"expected={expected:3d} "
                f"actual={actual:3d} "
                f"[{status}]"
            )

    # --------------------------------------------------------
    # Save all point-level records.
    # --------------------------------------------------------

    all_records_path = (
        OUTPUT_DIR
        / "part239_all_point_foreground_background_records.csv"
    )

    records_df.to_csv(
        all_records_path,
        index=False,
    )

    # --------------------------------------------------------
    # Class summary.
    # --------------------------------------------------------

    class_summary = create_class_summary(
        records_df
    )

    class_summary_path = (
        OUTPUT_DIR
        / "part239_class_foreground_background_summary.csv"
    )

    class_summary.to_csv(
        class_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Level summary.
    # --------------------------------------------------------

    level_summary = create_level_summary(
        records_df
    )

    level_summary_path = (
        OUTPUT_DIR
        / "part239_level_foreground_background_summary.csv"
    )

    level_summary.to_csv(
        level_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # LFNN/RFNN paired comparison.
    # --------------------------------------------------------

    paired_df = create_lfnn_rfnn_pairs(
        records_df
    )

    paired_path = (
        OUTPUT_DIR
        / "part239_lfnn_rfnn_paired_comparison.csv"
    )

    paired_df.to_csv(
        paired_path,
        index=False,
    )

    # ========================================================
    # 12. PRINT MAIN RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("FOREGROUND-VS-BACKGROUND SUMMARY")
    print("=" * 80)

    if not class_summary.empty:

        display_columns = [
            "short_name",
            "count",
            "mean_target_minus_background_logit",
            "median_target_minus_background_logit",
            "positive_margin_count",
            "negative_margin_count",
            "foreground_over_background_win_rate",
            "mean_target_probability",
            "mean_background_probability",
            "predicted_as_target_rate",
        ]

        print(
            class_summary[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    # ========================================================
    # 13. LFNN/RFNN PAIRED SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("PAIRED LFNN/RFNN FOREGROUND-BACKGROUND SUMMARY")
    print("=" * 80)

    if not paired_df.empty:

        lfnn_margin = paired_df[
            "lfnn_target_minus_background_logit"
        ].to_numpy(dtype=float)

        rfnn_margin = paired_df[
            "rfnn_target_minus_background_logit"
        ].to_numpy(dtype=float)

        difference = paired_df[
            "rfnn_minus_lfnn_margin_difference"
        ].to_numpy(dtype=float)

        print(
            f"Paired observations: "
            f"{len(paired_df)}"
        )

        print("\nLFNN target - Background:")
        print(
            f"  Mean   : {np.mean(lfnn_margin):.6f}"
        )
        print(
            f"  Median : {np.median(lfnn_margin):.6f}"
        )
        print(
            f"  Positive: "
            f"{np.sum(lfnn_margin > 0)}/"
            f"{len(lfnn_margin)}"
        )
        print(
            f"  Win rate: "
            f"{np.mean(lfnn_margin > 0):.6f}"
        )

        print("\nRFNN target - Background:")
        print(
            f"  Mean   : {np.mean(rfnn_margin):.6f}"
        )
        print(
            f"  Median : {np.median(rfnn_margin):.6f}"
        )
        print(
            f"  Positive: "
            f"{np.sum(rfnn_margin > 0)}/"
            f"{len(rfnn_margin)}"
        )
        print(
            f"  Win rate: "
            f"{np.mean(rfnn_margin > 0):.6f}"
        )

        print(
            "\nRFNN margin minus LFNN margin:"
        )

        print(
            f"  Mean   : {np.mean(difference):.6f}"
        )

        print(
            f"  Median : {np.median(difference):.6f}"
        )

        print(
            f"  RFNN higher: "
            f"{np.sum(difference > 0)}/"
            f"{len(difference)}"
        )

        print(
            f"  LFNN higher: "
            f"{np.sum(difference < 0)}/"
            f"{len(difference)}"
        )

    else:

        print(
            "No paired LFNN/RFNN observations found."
        )

    # ========================================================
    # 14. DISEASE-WISE INTERPRETATION DATA
    # ========================================================

    print("\n" + "=" * 80)
    print("DISEASE-WISE FOREGROUND/BACKGROUND MARGIN")
    print("=" * 80)

    for _, row in class_summary.iterrows():

        print(
            f"\n{row['short_name']} "
            f"({row['disease']})"
        )

        print(
            f"  Points                 : "
            f"{int(row['count'])}"
        )

        print(
            f"  Mean target-BG margin  : "
            f"{row['mean_target_minus_background_logit']:.6f}"
        )

        print(
            f"  Median target-BG margin: "
            f"{row['median_target_minus_background_logit']:.6f}"
        )

        print(
            f"  Positive margin        : "
            f"{int(row['positive_margin_count'])}/"
            f"{int(row['count'])}"
        )

        print(
            f"  FG-over-BG win rate    : "
            f"{row['foreground_over_background_win_rate']:.6f}"
        )

        print(
            f"  Mean target probability: "
            f"{row['mean_target_probability']:.6f}"
        )

        print(
            f"  Mean background prob   : "
            f"{row['mean_background_probability']:.6f}"
        )

    # ========================================================
    # 15. BUILD JSON SUMMARY
    # ========================================================

    json_summary = {
        "experiment": "Part 2.39",
        "title": (
            "Foreground-vs-Background "
            "Logit Margin Audit"
        ),

        "analysis_only": True,

        "checkpoint": str(
            CHECKPOINT_PATH
        ),

        "device": str(DEVICE),

        "validation_cases_expected": int(
            len(validation_cases)
        ),

        "validation_cases_successful": int(
            successful_cases
        ),

        "validation_cases_failed": int(
            len(failed_cases)
        ),

        "validation_point_records": int(
            len(records_df)
        ),

        "expected_point_counts":
            EXPECTED_POINT_COUNTS,

        "class_summary":
            class_summary.to_dict(
                orient="records"
            ),

        "paired_lfnn_rfnn_count": int(
            len(paired_df)
        ),
    }

    if not paired_df.empty:

        lfnn_margin = paired_df[
            "lfnn_target_minus_background_logit"
        ].to_numpy(dtype=float)

        rfnn_margin = paired_df[
            "rfnn_target_minus_background_logit"
        ].to_numpy(dtype=float)

        json_summary[
            "paired_lfnn_mean_target_background_margin"
        ] = float(
            np.mean(lfnn_margin)
        )

        json_summary[
            "paired_rfnn_mean_target_background_margin"
        ] = float(
            np.mean(rfnn_margin)
        )

        json_summary[
            "paired_lfnn_positive_margin_rate"
        ] = float(
            np.mean(lfnn_margin > 0)
        )

        json_summary[
            "paired_rfnn_positive_margin_rate"
        ] = float(
            np.mean(rfnn_margin > 0)
        )

    summary_path = (
        OUTPUT_DIR
        / "part239_foreground_background_audit_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_summary,
            f,
            indent=2,
        )

    # ========================================================
    # 16. FINAL OUTPUT LOCATIONS
    # ========================================================

    print("\n" + "=" * 80)
    print("PART 2.39 COMPLETE")
    print("=" * 80)

    print(
        f"Successful cases : "
        f"{successful_cases}/{len(validation_cases)}"
    )

    print(
        f"Point records    : "
        f"{len(records_df)}"
    )

    print("\nOutputs:")

    print(
        f"  {all_records_path}"
    )

    print(
        f"  {paired_path}"
    )

    print(
        f"  {class_summary_path}"
    )

    print(
        f"  {level_summary_path}"
    )

    print(
        f"  {summary_path}"
    )

    if failed_cases:
        print(
            f"  {OUTPUT_DIR / 'part239_failed_cases.csv'}"
        )

    print("\nIMPORTANT:")
    print(
        "This experiment did not train the model "
        "and did not modify the Part 2.27 checkpoint."
    )


if __name__ == "__main__":
    main()