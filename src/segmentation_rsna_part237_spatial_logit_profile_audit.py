"""
PART 2.37 - LFNN/RFNN SPATIAL LOGIT PROFILE AUDIT

Analysis-only experiment.

Purpose
-------
For the exact Part 2.20B validation cohort, identify paired LFNN/RFNN
points from the same study + series + spinal level.

For every paired observation, sample five positions along the
canonical/model-space line:

    LFNN -> P25 -> MID -> P75 -> RFNN

At every position, record the final SwinUNETR output logits and
probabilities for all six segmentation classes:

    0 = Background
    1 = Spinal Canal Stenosis
    2 = Left Neural Foraminal Narrowing
    3 = Right Neural Foraminal Narrowing
    4 = Left Subarticular Stenosis
    5 = Right Subarticular Stenosis

Also calculate:

    - predicted class
    - background probability
    - LFNN probability
    - RFNN probability
    - RFNN-LFNN probability margin
    - RFNN-background probability margin
    - entropy

This experiment:
    - does NOT train
    - does NOT modify the checkpoint
    - does NOT modify the dashboard
    - does NOT create voxel ground truth
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import torch


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part237_spatial_logit_profile_audit"
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
    str(PROJECT_ROOT / "src"),
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_SHAPE = (64, 96, 96)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

PROFILE_POSITIONS = [
    ("LFNN", 0.00),
    ("P25", 0.25),
    ("MID", 0.50),
    ("P75", 0.75),
    ("RFNN", 1.00),
]

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

LFNN_CLASS_ID = 2
RFNN_CLASS_ID = 3
BACKGROUND_CLASS_ID = 0


# =============================================================================
# IMPORT PART 2.20B
# =============================================================================

def import_part220b():
    candidates = [
        "segmentation_rsna_part220b_geometry_corrected_training",
        "segmentation_rsna_part220b_geometry_corrected_point_supervised_training",
    ]

    errors = []

    for module_name in candidates:
        try:
            module = import_module(module_name)

            print(
                "[OK] Imported Part 2.20B module:",
                module_name,
            )

            return module

        except Exception as exc:
            errors.append(
                (
                    module_name,
                    repr(exc),
                )
            )

    print(
        "\nCould not import Part 2.20B."
    )

    for module_name, error in errors:
        print(
            f"  {module_name}: {error}"
        )

    raise ImportError(
        "Part 2.20B module could not be imported."
    )


part220b = import_part220b()


# =============================================================================
# UTILITIES
# =============================================================================

def section(title):
    print(
        "\n"
        + "=" * 80
    )

    print(title)

    print(
        "=" * 80
    )


def clean_id(value):
    if pd.isna(value):
        return ""

    try:
        numeric = float(value)

        if numeric.is_integer():
            return str(
                int(numeric)
            )

    except Exception:
        pass

    return str(value).strip()


# =============================================================================
# MODEL
# =============================================================================

def build_model():
    """
    Use the exact Part 2.20B model builder.
    """

    return part220b.build_model()


def load_checkpoint(model):

    section(
        "LOADING PART 2.27 BEST CHECKPOINT"
    )

    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT
    )

    print(
        "Device:",
        DEVICE,
    )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):
            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif (
            "state_dict"
            in checkpoint
        ):
            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print(
        "Missing keys:",
        len(missing),
    )

    print(
        "Unexpected keys:",
        len(unexpected),
    )

    if missing:
        print(
            "Missing:",
            missing,
        )

    if unexpected:
        print(
            "Unexpected:",
            unexpected,
        )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "[OK] Model loaded."
    )

    print(
        "Parameter count:",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )

    return model


# =============================================================================
# VALIDATION COHORT
# =============================================================================

def recover_validation_points():

    section(
        "LOADING EXACT PART 2.20B VALIDATION COHORT"
    )

    manifest = (
        part220b.load_manifest()
    )

    print(
        "Full manifest rows:",
        len(manifest),
    )

    validation_series = (
        part220b.select_validation_series(
            manifest
        )
    )

    if isinstance(
        validation_series,
        pd.DataFrame,
    ):
        val_df = (
            validation_series.copy()
        )

    else:
        val_df = pd.DataFrame(
            validation_series
        )

    print(
        "Validation series returned:",
        len(val_df),
    )

    def find_column(
        df,
        candidates,
    ):

        mapping = {
            str(c).lower(): c
            for c in df.columns
        }

        for candidate in candidates:

            if (
                candidate.lower()
                in mapping
            ):
                return mapping[
                    candidate.lower()
                ]

        return None

    study_col = find_column(
        val_df,
        ["study_id"],
    )

    series_col = find_column(
        val_df,
        ["series_id"],
    )

    if (
        study_col is None
        or series_col is None
    ):
        raise KeyError(
            "study_id/series_id missing "
            "from validation cohort."
        )

    validation_keys = {
        (
            clean_id(row[study_col]),
            clean_id(row[series_col]),
        )
        for _, row
        in val_df.iterrows()
    }

    manifest_study_col = find_column(
        manifest,
        ["study_id"],
    )

    manifest_series_col = find_column(
        manifest,
        ["series_id"],
    )

    if (
        manifest_study_col is None
        or manifest_series_col is None
    ):
        raise KeyError(
            "study_id/series_id missing "
            "from full manifest."
        )

    mask = manifest.apply(
        lambda row:
        (
            clean_id(
                row[
                    manifest_study_col
                ]
            ),
            clean_id(
                row[
                    manifest_series_col
                ]
            ),
        )
        in validation_keys,
        axis=1,
    )

    point_rows = (
        manifest.loc[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "Recovered validation point rows:",
        len(point_rows),
    )

    print(
        "Validation studies:",
        point_rows[
            "study_id"
        ].nunique(),
    )

    print(
        "Validation series:",
        point_rows[
            "series_id"
        ].nunique(),
    )

    foraminal_mask = (
        point_rows[
            "class_name"
        ]
        .astype(str)
        .str.contains(
            "Neural Foraminal",
            case=False,
            na=False,
        )
    )

    foraminal = (
        point_rows.loc[
            foraminal_mask
        ]
        .copy()
    )

    print(
        "Validation LFNN/RFNN rows:",
        len(foraminal),
    )

    return (
        point_rows,
        foraminal,
    )


# =============================================================================
# BUILD LFNN/RFNN PAIRS
# =============================================================================

def build_pairs(foraminal):

    pairs = []

    grouped = foraminal.groupby(
        [
            "study_id",
            "series_id",
            "level",
        ],
        dropna=False,
    )

    for (
        study_id,
        series_id,
        level,
    ), group in grouped:

        left = group[
            group[
                "class_name"
            ]
            .astype(str)
            .str.contains(
                "Left Neural",
                case=False,
                na=False,
            )
        ].copy()

        right = group[
            group[
                "class_name"
            ]
            .astype(str)
            .str.contains(
                "Right Neural",
                case=False,
                na=False,
            )
        ].copy()

        if (
            len(left) == 0
            or len(right) == 0
        ):
            continue

        left = (
            left.sort_values(
                [
                    "native_z",
                    "native_y",
                    "native_x",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        right = (
            right.sort_values(
                [
                    "native_z",
                    "native_y",
                    "native_x",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        n = min(
            len(left),
            len(right),
        )

        for index in range(n):

            pairs.append(
                {
                    "pair_id":
                        len(pairs),

                    "study_id":
                        clean_id(
                            study_id
                        ),

                    "series_id":
                        clean_id(
                            series_id
                        ),

                    "level":
                        str(level),

                    "lfnn_row":
                        left.iloc[
                            index
                        ].to_dict(),

                    "rfnn_row":
                        right.iloc[
                            index
                        ].to_dict(),
                }
            )

    return pairs


# =============================================================================
# TRANSFORMED POINT LOOKUP
# =============================================================================

def transformed_point_lookup(
    transformed_points
):

    lookup = {}

    for point in transformed_points:

        key = (
            str(
                point.get(
                    "class_name",
                    "",
                )
            ),
            str(
                point.get(
                    "level",
                    "",
                )
            ),
        )

        lookup.setdefault(
            key,
            [],
        ).append(point)

    return lookup


def get_transformed_point(
    lookup,
    row,
):

    key = (
        str(
            row["class_name"]
        ),
        str(
            row["level"]
        ),
    )

    candidates = lookup.get(
        key,
        [],
    )

    if not candidates:
        raise KeyError(
            f"Transformed point missing: "
            f"{key}"
        )

    return candidates[0]


# =============================================================================
# POINT / MODEL UTILITIES
# =============================================================================

def model_point_from_transformed(
    point
):

    return np.array(
        [
            float(
                point["z"]
            ),
            float(
                point["y"]
            ),
            float(
                point["x"]
            ),
        ],
        dtype=np.float32,
    )


def interpolate_profile(
    lfnn_point,
    rfnn_point,
):

    a = np.asarray(
        lfnn_point,
        dtype=np.float32,
    )

    b = np.asarray(
        rfnn_point,
        dtype=np.float32,
    )

    positions = []

    for (
        name,
        fraction,
    ) in PROFILE_POSITIONS:

        point = (
            (1.0 - fraction) * a
            + fraction * b
        )

        positions.append(
            {
                "position_name":
                    name,

                "position_fraction":
                    float(fraction),

                "z":
                    float(point[0]),

                "y":
                    float(point[1]),

                "x":
                    float(point[2]),
            }
        )

    return positions


# =============================================================================
# OUTPUT LOGIT EXTRACTION
# =============================================================================

def get_logits_at_point(
    logits,
    point,
):

    if logits.ndim == 5:
        logits = logits[0]

    if logits.ndim != 4:
        raise ValueError(
            "Expected logits with shape "
            "C x D x H x W."
        )

    channels, depth, height, width = (
        logits.shape
    )

    z = int(
        round(
            float(point[0])
        )
    )

    y = int(
        round(
            float(point[1])
        )
    )

    x = int(
        round(
            float(point[2])
        )
    )

    z = max(
        0,
        min(
            depth - 1,
            z,
        ),
    )

    y = max(
        0,
        min(
            height - 1,
            y,
        ),
    )

    x = max(
        0,
        min(
            width - 1,
            x,
        ),
    )

    voxel_logits = (
        logits[
            :,
            z,
            y,
            x,
        ]
    )

    probabilities = torch.softmax(
        voxel_logits,
        dim=0,
    )

    logits_np = (
        voxel_logits
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    probs_np = (
        probabilities
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    predicted_class = int(
        np.argmax(
            probs_np
        )
    )

    return (
        logits_np,
        probs_np,
        predicted_class,
        (
            z,
            y,
            x,
        ),
    )


# =============================================================================
# ENTROPY
# =============================================================================

def calculate_entropy(
    probabilities
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    return float(
        -np.sum(
            probabilities
            * np.log(
                probabilities
            )
        )
    )


# =============================================================================
# MODEL INFERENCE
# =============================================================================

@torch.no_grad()
def run_model(
    model,
    image,
):

    if not torch.is_tensor(
        image
    ):
        image = torch.as_tensor(
            image,
            dtype=torch.float32,
        )

    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.ndim == 4:
        image = image.unsqueeze(0)

    image = image.to(
        DEVICE,
        dtype=torch.float32,
    )

    output = model(
        image
    )

    if isinstance(
        output,
        (
            tuple,
            list,
        ),
    ):
        output = output[0]

    return output


# =============================================================================
# MAIN
# =============================================================================

def main():

    section(
        "PART 2.37 - LFNN/RFNN "
        "SPATIAL LOGIT PROFILE AUDIT"
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Output:",
        OUTPUT_DIR,
    )

    print(
        "Model shape:",
        MODEL_SHAPE,
    )

    print(
        "Profile positions:",
        [
            name
            for name, _
            in PROFILE_POSITIONS
        ],
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = build_model()

    model = load_checkpoint(
        model
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    (
        point_rows,
        foraminal,
    ) = recover_validation_points()

    pairs = build_pairs(
        foraminal
    )

    section(
        "PAIRING SUMMARY"
    )

    candidate_series = {
        (
            pair["study_id"],
            pair["series_id"],
        )
        for pair in pairs
    }

    print(
        "Candidate paired series:",
        len(candidate_series),
    )

    print(
        "Paired LFNN/RFNN observations:",
        len(pairs),
    )

    # -------------------------------------------------------------------------
    # Containers
    # -------------------------------------------------------------------------

    profile_records = []

    successful_cases = set()

    prediction_records = []

    # -------------------------------------------------------------------------
    # Unique cases
    # -------------------------------------------------------------------------

    unique_cases = sorted(
        candidate_series
    )

    for case_index, (
        study_id,
        series_id,
    ) in enumerate(
        unique_cases,
        start=1,
    ):

        print(
            f"\n[{case_index}/{len(unique_cases)}] "
            f"Study={study_id} "
            f"Series={series_id}"
        )

        case_rows = point_rows[
            (
                point_rows[
                    "study_id"
                ].map(clean_id)
                == study_id
            )
            &
            (
                point_rows[
                    "series_id"
                ].map(clean_id)
                == series_id
            )
        ].copy()

        try:

            image, transformed_points, geometry = (
                part220b.load_case(
                    study_id,
                    series_id,
                    case_rows,
                )
            )

            print(
                "  Image shape:",
                tuple(
                    image.shape
                ),
            )

            print(
                "  Original point rows:",
                len(case_rows),
            )

            print(
                "  Transformed points:",
                len(
                    transformed_points
                ),
            )

            lookup = (
                transformed_point_lookup(
                    transformed_points
                )
            )

            # -------------------------------------------------------------
            # Forward pass
            # -------------------------------------------------------------

            logits = run_model(
                model,
                image,
            )

            if logits.ndim == 5:
                output_shape = tuple(
                    logits.shape
                )
            else:
                output_shape = tuple(
                    logits.shape
                )

            print(
                "  Output shape:",
                output_shape,
            )

            # -------------------------------------------------------------
            # Cases' pairs
            # -------------------------------------------------------------

            case_pairs = [
                pair
                for pair in pairs
                if (
                    pair["study_id"]
                    == study_id
                    and pair["series_id"]
                    == series_id
                )
            ]

            for pair in case_pairs:

                lrow = pair[
                    "lfnn_row"
                ]

                rrow = pair[
                    "rfnn_row"
                ]

                lpoint_dict = (
                    get_transformed_point(
                        lookup,
                        lrow,
                    )
                )

                rpoint_dict = (
                    get_transformed_point(
                        lookup,
                        rrow,
                    )
                )

                lpoint = (
                    model_point_from_transformed(
                        lpoint_dict
                    )
                )

                rpoint = (
                    model_point_from_transformed(
                        rpoint_dict
                    )
                )

                pair_distance = float(
                    np.linalg.norm(
                        rpoint - lpoint
                    )
                )

                profile = (
                    interpolate_profile(
                        lpoint,
                        rpoint,
                    )
                )

                # ---------------------------------------------------------
                # Profile
                # ---------------------------------------------------------

                for position in profile:

                    model_point = np.array(
                        [
                            position["z"],
                            position["y"],
                            position["x"],
                        ],
                        dtype=np.float32,
                    )

                    (
                        logits_np,
                        probs_np,
                        predicted_class,
                        voxel,
                    ) = get_logits_at_point(
                        logits,
                        model_point,
                    )

                    entropy = (
                        calculate_entropy(
                            probs_np
                        )
                    )

                    bg_prob = float(
                        probs_np[
                            BACKGROUND_CLASS_ID
                        ]
                    )

                    lfnn_prob = float(
                        probs_np[
                            LFNN_CLASS_ID
                        ]
                    )

                    rfnn_prob = float(
                        probs_np[
                            RFNN_CLASS_ID
                        ]
                    )

                    rfnn_lfnn_margin = (
                        rfnn_prob
                        - lfnn_prob
                    )

                    rfnn_bg_margin = (
                        rfnn_prob
                        - bg_prob
                    )

                    # -----------------------------------------------------
                    # Record
                    # -----------------------------------------------------

                    record = {
                        "pair_id":
                            pair[
                                "pair_id"
                            ],

                        "study_id":
                            study_id,

                        "series_id":
                            series_id,

                        "level":
                            str(
                                pair[
                                    "level"
                                ]
                            ),

                        "position_name":
                            position[
                                "position_name"
                            ],

                        "position_fraction":
                            position[
                                "position_fraction"
                            ],

                        "model_z":
                            float(
                                model_point[0]
                            ),

                        "model_y":
                            float(
                                model_point[1]
                            ),

                        "model_x":
                            float(
                                model_point[2]
                            ),

                        "sampled_z":
                            int(
                                voxel[0]
                            ),

                        "sampled_y":
                            int(
                                voxel[1]
                            ),

                        "sampled_x":
                            int(
                                voxel[2]
                            ),

                        "lfnn_rfnn_distance":
                            pair_distance,

                        "predicted_class":
                            predicted_class,

                        "predicted_class_name":
                            CLASS_NAMES.get(
                                predicted_class,
                                f"Class_{predicted_class}",
                            ),

                        "background_probability":
                            bg_prob,

                        "scs_probability":
                            float(
                                probs_np[1]
                            ),

                        "lfnn_probability":
                            lfnn_prob,

                        "rfnn_probability":
                            rfnn_prob,

                        "lss_probability":
                            float(
                                probs_np[4]
                            ),

                        "rss_probability":
                            float(
                                probs_np[5]
                            ),

                        "rfnn_minus_lfnn_probability":
                            rfnn_lfnn_margin,

                        "rfnn_minus_background_probability":
                            rfnn_bg_margin,

                        "entropy":
                            entropy,

                        "background_logit":
                            float(
                                logits_np[0]
                            ),

                        "scs_logit":
                            float(
                                logits_np[1]
                            ),

                        "lfnn_logit":
                            float(
                                logits_np[2]
                            ),

                        "rfnn_logit":
                            float(
                                logits_np[3]
                            ),

                        "lss_logit":
                            float(
                                logits_np[4]
                            ),

                        "rss_logit":
                            float(
                                logits_np[5]
                            ),
                    }

                    profile_records.append(
                        record
                    )

                # ---------------------------------------------------------
                # Endpoint prediction record
                # ---------------------------------------------------------

                (
                    l_logits,
                    l_probs,
                    l_pred,
                    l_voxel,
                ) = get_logits_at_point(
                    logits,
                    lpoint,
                )

                (
                    r_logits,
                    r_probs,
                    r_pred,
                    r_voxel,
                ) = get_logits_at_point(
                    logits,
                    rpoint,
                )

                prediction_records.append(
                    {
                        "pair_id":
                            pair[
                                "pair_id"
                            ],

                        "study_id":
                            study_id,

                        "series_id":
                            series_id,

                        "level":
                            str(
                                pair[
                                    "level"
                                ]
                            ),

                        "lfnn_sampled_z":
                            l_voxel[0],

                        "lfnn_sampled_y":
                            l_voxel[1],

                        "lfnn_sampled_x":
                            l_voxel[2],

                        "rfnn_sampled_z":
                            r_voxel[0],

                        "rfnn_sampled_y":
                            r_voxel[1],

                        "rfnn_sampled_x":
                            r_voxel[2],

                        "pair_distance":
                            pair_distance,

                        "lfnn_predicted_class":
                            l_pred,

                        "rfnn_predicted_class":
                            r_pred,

                        "lfnn_true_class_probability":
                            float(
                                l_probs[
                                    LFNN_CLASS_ID
                                ]
                            ),

                        "rfnn_true_class_probability":
                            float(
                                r_probs[
                                    RFNN_CLASS_ID
                                ]
                            ),

                        "lfnn_rfnn_probability_difference":
                            float(
                                l_probs[
                                    LFNN_CLASS_ID
                                ]
                                -
                                r_probs[
                                    RFNN_CLASS_ID
                                ]
                            ),

                        "lfnn_background_probability":
                            float(
                                l_probs[
                                    BACKGROUND_CLASS_ID
                                ]
                            ),

                        "rfnn_background_probability":
                            float(
                                r_probs[
                                    BACKGROUND_CLASS_ID
                                ]
                            ),
                    }
                )

            successful_cases.add(
                (
                    study_id,
                    series_id,
                )
            )

        except Exception as exc:

            print(
                "  [ERROR]",
                repr(exc),
            )

    # =============================================================================
    # DATAFRAME
    # =============================================================================

    profile_df = pd.DataFrame(
        profile_records
    )

    prediction_df = pd.DataFrame(
        prediction_records
    )

    if profile_df.empty:
        raise RuntimeError(
            "No spatial logit profile records generated."
        )

    # =============================================================================
    # SUMMARY
    # =============================================================================

    section(
        "SPATIAL LOGIT PROFILE SUMMARY"
    )

    summary = (
        profile_df
        .groupby(
            [
                "position_name",
                "position_fraction",
            ],
            as_index=False,
        )
        .agg(
            pairs=(
                "pair_id",
                "nunique",
            ),

            mean_background_probability=(
                "background_probability",
                "mean",
            ),

            mean_scs_probability=(
                "scs_probability",
                "mean",
            ),

            mean_lfnn_probability=(
                "lfnn_probability",
                "mean",
            ),

            mean_rfnn_probability=(
                "rfnn_probability",
                "mean",
            ),

            mean_lss_probability=(
                "lss_probability",
                "mean",
            ),

            mean_rss_probability=(
                "rss_probability",
                "mean",
            ),

            mean_rfnn_minus_lfnn=(
                "rfnn_minus_lfnn_probability",
                "mean",
            ),

            mean_rfnn_minus_background=(
                "rfnn_minus_background_probability",
                "mean",
            ),

            mean_entropy=(
                "entropy",
                "mean",
            ),
        )
    )

    # =============================================================================
    # PAIRWISE PROFILE SUMMARY
    # =============================================================================

    pair_profile_summary = (
        profile_df
        .pivot_table(
            index=[
                "pair_id",
                "study_id",
                "series_id",
                "level",
            ],
            columns="position_name",
            values=[
                "background_probability",
                "lfnn_probability",
                "rfnn_probability",
                "rfnn_minus_lfnn_probability",
                "rfnn_minus_background_probability",
                "entropy",
            ],
            aggfunc="mean",
        )
        .reset_index()
    )

    pair_profile_summary.columns = [
        "_".join(
            [
                str(part)
                for part in column
                if str(part) != ""
            ]
        )
        for column in (
            pair_profile_summary.columns
            if isinstance(
                pair_profile_summary.columns,
                pd.MultiIndex,
            )
            else [
                (column,)
                for column
                in pair_profile_summary.columns
            ]
        )
    ]

    # =============================================================================
    # MONOTONICITY ANALYSIS
    # =============================================================================

    monotonic_records = []

    for pair_id, group in (
        profile_df.groupby(
            "pair_id"
        )
    ):

        ordered = (
            group.sort_values(
                "position_fraction"
            )
        )

        rfnn_probability = (
            ordered[
                "rfnn_probability"
            ]
            .to_numpy(
                dtype=float
            )
        )

        lfnn_probability = (
            ordered[
                "lfnn_probability"
            ]
            .to_numpy(
                dtype=float
            )
        )

        background_probability = (
            ordered[
                "background_probability"
            ]
            .to_numpy(
                dtype=float
            )
        )

        # RFNN probability non-decreasing?
        rfnn_non_decreasing = bool(
            np.all(
                np.diff(
                    rfnn_probability
                )
                >= -1e-8
            )
        )

        # LFNN probability non-increasing?
        lfnn_non_increasing = bool(
            np.all(
                np.diff(
                    lfnn_probability
                )
                <= 1e-8
            )
        )

        # Background probability non-increasing?
        bg_non_increasing = bool(
            np.all(
                np.diff(
                    background_probability
                )
                <= 1e-8
            )
        )

        monotonic_records.append(
            {
                "pair_id":
                    pair_id,

                "rfnn_probability_non_decreasing":
                    rfnn_non_decreasing,

                "lfnn_probability_non_increasing":
                    lfnn_non_increasing,

                "background_probability_non_increasing":
                    bg_non_increasing,
            }
        )

    monotonic_df = pd.DataFrame(
        monotonic_records
    )

    # =============================================================================
    # PRINT MAIN RESULT
    # =============================================================================

    print(
        "\nMean probability by spatial position:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nMonotonic profile counts:"
    )

    print(
        "RFNN probability non-decreasing:",
        int(
            monotonic_df[
                "rfnn_probability_non_decreasing"
            ].sum()
        ),
        "/",
        len(monotonic_df),
    )

    print(
        "LFNN probability non-increasing:",
        int(
            monotonic_df[
                "lfnn_probability_non_increasing"
            ].sum()
        ),
        "/",
        len(monotonic_df),
    )

    print(
        "Background probability non-increasing:",
        int(
            monotonic_df[
                "background_probability_non_increasing"
            ].sum()
        ),
        "/",
        len(monotonic_df),
    )

    # =============================================================================
    # SAVE
    # =============================================================================

    profile_path = (
        OUTPUT_DIR
        / "part237_spatial_logit_profile_records.csv"
    )

    prediction_path = (
        OUTPUT_DIR
        / "part237_endpoint_prediction_records.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part237_spatial_logit_profile_summary.csv"
    )

    pair_summary_path = (
        OUTPUT_DIR
        / "part237_pairwise_spatial_logit_profiles.csv"
    )

    monotonic_path = (
        OUTPUT_DIR
        / "part237_monotonicity_audit.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part237_spatial_logit_profile_audit_summary.json"
    )

    profile_df.to_csv(
        profile_path,
        index=False,
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    pair_profile_summary.to_csv(
        pair_summary_path,
        index=False,
    )

    monotonic_df.to_csv(
        monotonic_path,
        index=False,
    )

    # =============================================================================
    # JSON
    # =============================================================================

    json_summary = {
        "part": "2.37",

        "title":
            "LFNN/RFNN Spatial Logit Profile Audit",

        "purpose":
            (
                "Analysis-only audit of final SwinUNETR "
                "logits/probabilities along the LFNN-to-RFNN "
                "canonical-space direction."
            ),

        "checkpoint":
            str(
                CHECKPOINT
            ),

        "device":
            str(
                DEVICE
            ),

        "model_shape":
            list(
                MODEL_SHAPE
            ),

        "successful_cases":
            len(
                successful_cases
            ),

        "paired_observations":
            len(
                pairs
            ),

        "profile_positions":
            [
                {
                    "name":
                        name,

                    "fraction":
                        fraction,
                }

                for name, fraction
                in PROFILE_POSITIONS
            ],

        "profile_records":
            len(
                profile_df
            ),

        "expected_profile_records":
            (
                len(pairs)
                * len(PROFILE_POSITIONS)
            ),

        "prediction_records":
            len(
                prediction_df
            ),

        "monotonicity": {
            "rfnn_probability_non_decreasing":
                int(
                    monotonic_df[
                        "rfnn_probability_non_decreasing"
                    ].sum()
                ),

            "lfnn_probability_non_increasing":
                int(
                    monotonic_df[
                        "lfnn_probability_non_increasing"
                    ].sum()
                ),

            "background_probability_non_increasing":
                int(
                    monotonic_df[
                        "background_probability_non_increasing"
                    ].sum()
                ),

            "total_pairs":
                len(
                    monotonic_df
                ),
        },

        "files": {
            "profile_records":
                str(
                    profile_path
                ),

            "prediction_records":
                str(
                    prediction_path
                ),

            "summary":
                str(
                    summary_path
                ),

            "pairwise_profiles":
                str(
                    pair_summary_path
                ),

            "monotonicity":
                str(
                    monotonic_path
                ),
        },

        "training_performed":
            False,

        "checkpoint_modified":
            False,

        "dashboard_modified":
            False,
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

    # =============================================================================
    # FINAL
    # =============================================================================

    section(
        "PART 2.37 COMPLETE"
    )

    print(
        "Successful cases:",
        len(
            successful_cases
        ),
    )

    print(
        "Paired observations:",
        len(
            pairs
        ),
    )

    print(
        "Profile records:",
        len(
            profile_df
        ),
    )

    print(
        "Expected profile records:",
        len(pairs)
        * len(
            PROFILE_POSITIONS
        ),
    )

    print(
        "Prediction records:",
        len(
            prediction_df
        ),
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {profile_path}"
    )

    print(
        f"  {prediction_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {pair_summary_path}"
    )

    print(
        f"  {monotonic_path}"
    )

    print(
        f"  {json_path}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This experiment is analysis-only."
    )

    print(
        "No training/checkpoint/dashboard changes were made."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()