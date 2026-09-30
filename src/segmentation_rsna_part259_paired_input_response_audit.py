"""
PART 2.59
PAIRED LFNN / RFNN INPUT-RESPONSE AUDIT

Purpose
-------
Compare paired LFNN and RFNN points from the same:

    study
    series
    spinal level

The goal is to determine whether the RFNN failure is already visible in
the local MRI input or whether the major divergence appears later in the
Swin-UNETR representation.

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
    / "rsna_part259_paired_input_response_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
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
# CONSTANTS
# ============================================================================

LFNN_NAME = "Left Neural Foraminal Narrowing"

RFNN_NAME = "Right Neural Foraminal Narrowing"

BACKGROUND_ID = 0
SCS_ID = 1
LFNN_ID = 2
RFNN_ID = 3
LSS_ID = 4
RSS_ID = 5

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

EXPECTED_CASES = 25
EXPECTED_PAIRS = 45


# ============================================================================
# VALIDATION COHORT
# ============================================================================

def build_validation_cases(manifest):

    selected = part220b.select_validation_series(
        manifest
    )

    if not isinstance(selected, pd.DataFrame):

        raise TypeError(
            "select_validation_series() must return a DataFrame."
        )

    if len(selected) != EXPECTED_CASES:

        raise RuntimeError(
            f"Expected {EXPECTED_CASES} validation series, "
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

    validation_cases = part220b.build_case_index(
        validation_manifest
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
            9999,
        ),
    )

    if len(validation_cases) != EXPECTED_CASES:

        raise RuntimeError(
            "Validation case count mismatch."
        )

    return validation_cases


# ============================================================================
# MODEL
# ============================================================================

def load_model():

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
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"Missing keys    : {len(missing)}"
    )

    print(
        f"Unexpected keys : {len(unexpected)}"
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: {parameter_count:,}"
    )

    if missing or unexpected:

        raise RuntimeError(
            "Checkpoint did not load cleanly."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = (
        model
        .to(device)
        .eval()
    )

    print(
        f"Device: {device}"
    )

    return model, device


# ============================================================================
# MODEL FORWARD
# ============================================================================

def run_model(
    model,
    image,
    device,
):

    if not torch.is_tensor(image):

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

        image = image.unsqueeze(0)

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

    return logits, probabilities


# ============================================================================
# FEATURE HOOKS
# ============================================================================

TARGET_FEATURES = [
    "encoder1.layer.norm2",
    "encoder2.layer.norm2",
    "encoder3.layer.norm2",
    "encoder4.layer.norm2",
    "encoder10.layer.norm2",
    "decoder4.conv_block.norm2",
    "decoder3.conv_block.norm2",
    "decoder2.conv_block.norm2",
    "decoder1.conv_block.norm2",
]


def get_module_by_name(
    model,
    name,
):

    current = model

    for part in name.split("."):

        if part.isdigit():

            current = current[
                int(part)
            ]

        else:

            if not hasattr(
                current,
                part,
            ):

                return None

            current = getattr(
                current,
                part,
            )

    return current


def register_feature_hooks(model):

    activations = {}

    hooks = []

    for feature_name in TARGET_FEATURES:

        module = get_module_by_name(
            model,
            feature_name,
        )

        if module is None:

            print(
                f"WARNING: feature not found: "
                f"{feature_name}"
            )

            continue

        def make_hook(name):

            def hook(
                module,
                inputs,
                output,
            ):

                if torch.is_tensor(output):

                    activations[name] = (
                        output.detach()
                        .float()
                        .cpu()
                    )

            return hook

        handle = module.register_forward_hook(
            make_hook(feature_name)
        )

        hooks.append(handle)

    return activations, hooks


# ============================================================================
# CANONICAL POINT
# ============================================================================

def get_canonical_point(
    point,
    geometry,
):

    patient_point = np.array(
        [
            float(point["patient_x"]),
            float(point["patient_y"]),
            float(point["patient_z"]),
        ],
        dtype=np.float64,
    )

    canonical = (
        part220b.patient_point_to_canonical(
            patient_point,
            geometry,
        )
    )

    canonical = np.asarray(
        canonical,
        dtype=np.float64,
    ).reshape(-1)

    return canonical[:3]


# ============================================================================
# SAFE FEATURE SAMPLING
# ============================================================================

def sample_feature_at_point(
    feature,
    point,
):

    if feature is None:

        return None

    array = (
        feature
        .detach()
        .cpu()
        .numpy()
    )

    if array.ndim == 5:

        array = array[0]

    # ------------------------------------------------------------
    # Standard channel-first tensor:
    # C x Z x Y x X
    # ------------------------------------------------------------

    if array.ndim == 4:

        channels = array.shape[0]

        depth = array.shape[1]
        height = array.shape[2]
        width = array.shape[3]

        z = int(
            round(
                float(point[0])
                * (
                    max(
                        depth - 1,
                        1,
                    )
                    /
                    max(
                        63,
                        1,
                    )
                )
            )
        )

        y = int(
            round(
                float(point[1])
                * (
                    max(
                        height - 1,
                        1,
                    )
                    /
                    max(
                        95,
                        1,
                    )
                )
            )
        )

        x = int(
            round(
                float(point[2])
                * (
                    max(
                        width - 1,
                        1,
                    )
                    /
                    max(
                        95,
                        1,
                    )
                )
            )
        )

        z = np.clip(
            z,
            0,
            depth - 1,
        )

        y = np.clip(
            y,
            0,
            height - 1,
        )

        x = np.clip(
            x,
            0,
            width - 1,
        )

        vector = array[
            :,
            z,
            y,
            x,
        ]

        return vector.astype(
            np.float64
        )

    return None


# ============================================================================
# FEATURE RESAMPLING POINT
# ============================================================================

def scale_point_to_feature(
    point,
    feature,
):

    array = feature

    if torch.is_tensor(array):

        shape = tuple(
            array.shape
        )

    else:

        shape = tuple(
            np.asarray(array).shape
        )

    if len(shape) == 5:

        _, channels, depth, height, width = shape

    elif len(shape) == 4:

        channels, depth, height, width = shape

    else:

        return None

    return np.array(
        [
            float(point[0])
            * (
                max(
                    depth - 1,
                    1,
                )
                /
                63.0
            ),

            float(point[1])
            * (
                max(
                    height - 1,
                    1,
                )
                /
                95.0
            ),

            float(point[2])
            * (
                max(
                    width - 1,
                    1,
                )
                /
                95.0
            ),
        ],
        dtype=np.float64,
    )


def sample_feature_scaled(
    feature,
    canonical_point,
):

    if feature is None:

        return None

    array = feature.numpy()

    if array.ndim == 5:

        array = array[0]

    if array.ndim != 4:

        return None

    channels = array.shape[0]

    depth = array.shape[1]
    height = array.shape[2]
    width = array.shape[3]

    z = int(
        round(
            canonical_point[0]
            * (
                max(depth - 1, 1)
                / 63.0
            )
        )
    )

    y = int(
        round(
            canonical_point[1]
            * (
                max(height - 1, 1)
                / 95.0
            )
        )
    )

    x = int(
        round(
            canonical_point[2]
            * (
                max(width - 1, 1)
                / 95.0
            )
        )
    )

    z = int(
        np.clip(
            z,
            0,
            depth - 1,
        )
    )

    y = int(
        np.clip(
            y,
            0,
            height - 1,
        )
    )

    x = int(
        np.clip(
            x,
            0,
            width - 1,
        )
    )

    return array[
        :,
        z,
        y,
        x,
    ].astype(
        np.float64
    )


# ============================================================================
# LOCAL MRI PATCH STATISTICS
# ============================================================================

def local_patch_statistics(
    image,
    point,
    radius=2,
):

    array = (
        image
        .detach()
        .cpu()
        .numpy()
        if torch.is_tensor(image)
        else np.asarray(image)
    )

    array = np.squeeze(
        array
    )

    depth, height, width = (
        array.shape
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

    patch = array[
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    if patch.size == 0:

        return {
            "intensity_mean": np.nan,
            "intensity_std": np.nan,
            "intensity_min": np.nan,
            "intensity_max": np.nan,
            "gradient_mean": np.nan,
            "gradient_std": np.nan,
        }

    gradients = np.gradient(
        patch.astype(
            np.float32
        )
    )

    gradient_magnitude = np.sqrt(
        gradients[0] ** 2
        +
        gradients[1] ** 2
        +
        gradients[2] ** 2
    )

    return {
        "intensity_mean":
            float(
                patch.mean()
            ),

        "intensity_std":
            float(
                patch.std()
            ),

        "intensity_min":
            float(
                patch.min()
            ),

        "intensity_max":
            float(
                patch.max()
            ),

        "gradient_mean":
            float(
                gradient_magnitude.mean()
            ),

        "gradient_std":
            float(
                gradient_magnitude.std()
            ),
    }


# ============================================================================
# LOGIT STATISTICS
# ============================================================================

def extract_logits_at_point(
    logits,
    probabilities,
    canonical_point,
):

    logits_np = (
        logits
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

    if logits_np.ndim == 5:

        logits_np = logits_np[0]

    if probabilities_np.ndim == 5:

        probabilities_np = probabilities_np[0]

    depth = logits_np.shape[1]
    height = logits_np.shape[2]
    width = logits_np.shape[3]

    z = int(
        round(
            float(canonical_point[0])
        )
    )

    y = int(
        round(
            float(canonical_point[1])
        )
    )

    x = int(
        round(
            float(canonical_point[2])
        )
    )

    z = int(
        np.clip(
            z,
            0,
            depth - 1,
        )
    )

    y = int(
        np.clip(
            y,
            0,
            height - 1,
        )
    )

    x = int(
        np.clip(
            x,
            0,
            width - 1,
        )
    )

    return (
        logits_np[
            :,
            z,
            y,
            x,
        ],
        probabilities_np[
            :,
            z,
            y,
            x,
        ],
    )


# ============================================================================
# FEATURE VECTOR COMPARISON
# ============================================================================

def compare_vectors(
    left,
    right,
):

    if left is None or right is None:

        return {
            "cosine":
                np.nan,

            "l2":
                np.nan,

            "mean_abs_difference":
                np.nan,

            "left_norm":
                np.nan,

            "right_norm":
                np.nan,
        }

    left = np.asarray(
        left,
        dtype=np.float64,
    )

    right = np.asarray(
        right,
        dtype=np.float64,
    )

    left_norm = float(
        np.linalg.norm(
            left
        )
    )

    right_norm = float(
        np.linalg.norm(
            right
        )
    )

    denominator = (
        left_norm
        *
        right_norm
    )

    if denominator > 0:

        cosine = float(
            np.dot(
                left,
                right,
            )
            /
            denominator
        )

    else:

        cosine = np.nan

    return {
        "cosine":
            cosine,

        "l2":
            float(
                np.linalg.norm(
                    left - right
                )
            ),

        "mean_abs_difference":
            float(
                np.mean(
                    np.abs(
                        left - right
                    )
                )
            ),

        "left_norm":
            left_norm,

        "right_norm":
            right_norm,
    }


# ============================================================================
# ANALYZE CASE
# ============================================================================

def analyze_case(
    model,
    device,
    case,
    hooks,
    activations,
):

    study_id = str(
        case["study_id"]
    )

    series_id = str(
        case["series_id"]
    )

    (
        image,
        points,
        geometry,
    ) = part220b.load_case(
        study_id,
        series_id,
        case["points"],
    )

    point_df = pd.DataFrame(
        points
    )

    lfnn_points = point_df[
        point_df["class_name"]
        == LFNN_NAME
    ].copy()

    rfnn_points = point_df[
        point_df["class_name"]
        == RFNN_NAME
    ].copy()

    if len(lfnn_points) != 5:
        return []

    if len(rfnn_points) != 5:
        return []

    # Clear stale hooks.
    activations.clear()

    logits, probabilities = run_model(
        model,
        image,
        device,
    )

    records = []

    for level in sorted(
        set(
            lfnn_points["level"]
        )
        &
        set(
            rfnn_points["level"]
        )
    ):

        lfnn_row = lfnn_points[
            lfnn_points["level"]
            == level
        ]

        rfnn_row = rfnn_points[
            rfnn_points["level"]
            == level
        ]

        if len(lfnn_row) != 1:
            continue

        if len(rfnn_row) != 1:
            continue

        lfnn = lfnn_row.iloc[0]
        rfnn = rfnn_row.iloc[0]

        lfnn_canonical = get_canonical_point(
            lfnn,
            geometry,
        )

        rfnn_canonical = get_canonical_point(
            rfnn,
            geometry,
        )

        lfnn_logits, lfnn_probs = (
            extract_logits_at_point(
                logits,
                probabilities,
                lfnn_canonical,
            )
        )

        rfnn_logits, rfnn_probs = (
            extract_logits_at_point(
                logits,
                probabilities,
                rfnn_canonical,
            )
        )

        lfnn_input = (
            local_patch_statistics(
                image,
                lfnn_canonical,
                radius=2,
            )
        )

        rfnn_input = (
            local_patch_statistics(
                image,
                rfnn_canonical,
                radius=2,
            )
        )

        for feature_name in TARGET_FEATURES:

            feature = activations.get(
                feature_name
            )

            lfnn_feature = (
                sample_feature_scaled(
                    feature,
                    lfnn_canonical,
                )
            )

            rfnn_feature = (
                sample_feature_scaled(
                    feature,
                    rfnn_canonical,
                )
            )

            feature_comparison = (
                compare_vectors(
                    lfnn_feature,
                    rfnn_feature,
                )
            )

            records.append(
                {
                    "study_id":
                        study_id,

                    "series_id":
                        series_id,

                    "level":
                        str(level),

                    "feature":
                        feature_name,

                    "lfnn_intensity_mean":
                        lfnn_input[
                            "intensity_mean"
                        ],

                    "rfnn_intensity_mean":
                        rfnn_input[
                            "intensity_mean"
                        ],

                    "intensity_difference_rfnn_minus_lfnn":
                        (
                            rfnn_input[
                                "intensity_mean"
                            ]
                            -
                            lfnn_input[
                                "intensity_mean"
                            ]
                        ),

                    "lfnn_intensity_std":
                        lfnn_input[
                            "intensity_std"
                        ],

                    "rfnn_intensity_std":
                        rfnn_input[
                            "intensity_std"
                        ],

                    "lfnn_gradient_mean":
                        lfnn_input[
                            "gradient_mean"
                        ],

                    "rfnn_gradient_mean":
                        rfnn_input[
                            "gradient_mean"
                        ],

                    "lfnn_target_logit":
                        float(
                            lfnn_logits[
                                LFNN_ID
                            ]
                        ),

                    "rfnn_target_logit":
                        float(
                            rfnn_logits[
                                RFNN_ID
                            ]
                        ),

                    "lfnn_background_logit":
                        float(
                            lfnn_logits[
                                BACKGROUND_ID
                            ]
                        ),

                    "rfnn_background_logit":
                        float(
                            rfnn_logits[
                                BACKGROUND_ID
                            ]
                        ),

                    "lfnn_target_background_margin":
                        float(
                            lfnn_logits[
                                LFNN_ID
                            ]
                            -
                            lfnn_logits[
                                BACKGROUND_ID
                            ]
                        ),

                    "rfnn_target_background_margin":
                        float(
                            rfnn_logits[
                                RFNN_ID
                            ]
                            -
                            rfnn_logits[
                                BACKGROUND_ID
                            ]
                        ),

                    "lfnn_target_probability":
                        float(
                            lfnn_probs[
                                LFNN_ID
                            ]
                        ),

                    "rfnn_target_probability":
                        float(
                            rfnn_probs[
                                RFNN_ID
                            ]
                        ),

                    "rfnn_minus_lfnn_target_logit":
                        float(
                            rfnn_logits[
                                RFNN_ID
                            ]
                            -
                            lfnn_logits[
                                LFNN_ID
                            ]
                        ),

                    "rfnn_minus_lfnn_margin":
                        float(
                            (
                                rfnn_logits[
                                    RFNN_ID
                                ]
                                -
                                rfnn_logits[
                                    BACKGROUND_ID
                                ]
                            )
                            -
                            (
                                lfnn_logits[
                                    LFNN_ID
                                ]
                                -
                                lfnn_logits[
                                    BACKGROUND_ID
                                ]
                            )
                        ),

                    "feature_cosine":
                        feature_comparison[
                            "cosine"
                        ],

                    "feature_l2":
                        feature_comparison[
                            "l2"
                        ],

                    "feature_mean_abs_difference":
                        feature_comparison[
                            "mean_abs_difference"
                        ],

                    "lfnn_feature_norm":
                        feature_comparison[
                            "left_norm"
                        ],

                    "rfnn_feature_norm":
                        feature_comparison[
                            "right_norm"
                        ],
                }
            )

    return records


# ============================================================================
# GLOBAL SUMMARY
# ============================================================================

def build_global_summary(
    records_df,
):

    rows = []

    for feature in TARGET_FEATURES:

        subset = records_df[
            records_df["feature"]
            == feature
        ]

        if len(subset) == 0:
            continue

        rows.append(
            {
                "feature":
                    feature,

                "pairs":
                    len(subset),

                "mean_intensity_difference":
                    float(
                        subset[
                            "intensity_difference_rfnn_minus_lfnn"
                        ].mean()
                    ),

                "mean_gradient_lfnn":
                    float(
                        subset[
                            "lfnn_gradient_mean"
                        ].mean()
                    ),

                "mean_gradient_rfnn":
                    float(
                        subset[
                            "rfnn_gradient_mean"
                        ].mean()
                    ),

                "mean_lfnn_target_logit":
                    float(
                        subset[
                            "lfnn_target_logit"
                        ].mean()
                    ),

                "mean_rfnn_target_logit":
                    float(
                        subset[
                            "rfnn_target_logit"
                        ].mean()
                    ),

                "rfnn_minus_lfnn_target_logit":
                    float(
                        subset[
                            "rfnn_minus_lfnn_target_logit"
                        ].mean()
                    ),

                "mean_lfnn_margin":
                    float(
                        subset[
                            "lfnn_target_background_margin"
                        ].mean()
                    ),

                "mean_rfnn_margin":
                    float(
                        subset[
                            "rfnn_target_background_margin"
                        ].mean()
                    ),

                "rfnn_minus_lfnn_margin":
                    float(
                        subset[
                            "rfnn_minus_lfnn_margin"
                        ].mean()
                    ),

                "mean_feature_cosine":
                    float(
                        subset[
                            "feature_cosine"
                        ].mean()
                    ),

                "median_feature_cosine":
                    float(
                        subset[
                            "feature_cosine"
                        ].median()
                    ),

                "mean_feature_l2":
                    float(
                        subset[
                            "feature_l2"
                        ].mean()
                    ),

                "mean_feature_abs_difference":
                    float(
                        subset[
                            "feature_mean_abs_difference"
                        ].mean()
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
    records_df,
):

    rows = []

    for feature in TARGET_FEATURES:

        for level in sorted(
            records_df["level"].unique()
        ):

            subset = records_df[
                (
                    records_df[
                        "feature"
                    ]
                    == feature
                )
                &
                (
                    records_df[
                        "level"
                    ]
                    == level
                )
            ]

            if len(subset) == 0:
                continue

            rows.append(
                {
                    "feature":
                        feature,

                    "level":
                        level,

                    "pairs":
                        len(subset),

                    "rfnn_minus_lfnn_target_logit":
                        float(
                            subset[
                                "rfnn_minus_lfnn_target_logit"
                            ].mean()
                        ),

                    "rfnn_minus_lfnn_margin":
                        float(
                            subset[
                                "rfnn_minus_lfnn_margin"
                            ].mean()
                        ),

                    "feature_cosine":
                        float(
                            subset[
                                "feature_cosine"
                            ].mean()
                        ),

                    "feature_l2":
                        float(
                            subset[
                                "feature_l2"
                            ].mean()
                        ),

                    "intensity_difference":
                        float(
                            subset[
                                "intensity_difference_rfnn_minus_lfnn"
                            ].mean()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# SAVE
# ============================================================================

def save_outputs(
    records_df,
    global_df,
    level_df,
):

    records_path = (
        OUTPUT_DIR
        / "part259_paired_input_response_records.csv"
    )

    global_path = (
        OUTPUT_DIR
        / "part259_feature_response_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part259_level_feature_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part259_paired_input_response_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
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

    summary = {
        "validation_cases":
            EXPECTED_CASES,

        "paired_observations":
            EXPECTED_PAIRS,

        "feature_layers":
            TARGET_FEATURES,

        "records":
            int(
                len(records_df)
            ),

        "global_summary":
            global_df.to_dict(
                orient="records"
            ),

        "level_summary":
            level_df.to_dict(
                orient="records"
            ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print("PART 2.59 FEATURE / RESPONSE SUMMARY")
    print("=" * 80)

    print(
        global_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.59 LEVEL SUMMARY")
    print("=" * 80)

    print(
        level_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.59 COMPLETE")
    print("=" * 80)

    print(
        "Analysis only."
    )

    print(
        "Part 2.27 checkpoint unchanged."
    )

    print(
        "Part 2.58 unchanged."
    )

    print(
        "Dashboard unchanged."
    )

    print()
    print("Outputs:")

    print(records_path)
    print(global_path)
    print(level_path)
    print(json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("PART 2.59")
    print("PAIRED LFNN / RFNN INPUT-RESPONSE AUDIT")
    print("=" * 80)

    print()
    print("Analysis only.")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")

    manifest = part220b.load_manifest()

    print(
        f"Full manifest rows: "
        f"{len(manifest)}"
    )

    cases = build_validation_cases(
        manifest
    )

    print(
        f"Validation cases: "
        f"{len(cases)}"
    )

    model, device = load_model()

    activations, hooks = (
        register_feature_hooks(
            model
        )
    )

    all_records = []

    successful_cases = 0

    paired_cases = 0

    print()
    print("=" * 80)
    print("RUNNING PAIRED INPUT-RESPONSE ANALYSIS")
    print("=" * 80)

    try:

        for index, case in enumerate(
            cases,
            start=1,
        ):

            study_id = str(
                case["study_id"]
            )

            series_id = str(
                case["series_id"]
            )

            try:

                case_records = analyze_case(
                    model,
                    device,
                    case,
                    hooks,
                    activations,
                )

                all_records.extend(
                    case_records
                )

                if case_records:

                    successful_cases += 1
                    paired_cases += 1

                    print(
                        f"[{index:02d}/25] "
                        f"Study={study_id} | "
                        f"Series={series_id} | "
                        f"paired levels=5"
                    )

                else:

                    print(
                        f"[{index:02d}/25] "
                        f"Study={study_id} | "
                        f"Series={series_id} | "
                        f"no paired records"
                    )

            except Exception as exc:

                print(
                    f"[{index:02d}/25] "
                    f"Study={study_id} | "
                    f"Series={series_id} | "
                    f"FAILED: {exc}"
                )

    finally:

        for hook in hooks:

            hook.remove()

    print()
    print(
        f"Successful paired cases: "
        f"{paired_cases}/25"
    )

    records_df = pd.DataFrame(
        all_records
    )

    expected_records = (
        EXPECTED_PAIRS
        * len(TARGET_FEATURES)
    )

    print(
        f"Feature records: "
        f"{len(records_df)}"
    )

    if paired_cases != 9:

        raise RuntimeError(
            f"Expected 9 paired validation series, "
            f"found {paired_cases}."
        )

    if len(records_df) != 405:

        raise RuntimeError(
            f"Expected 405 feature records "
            f"(45 paired observations x 9 feature layers), "
            f"found {len(records_df)}."
        )

    if len(records_df) != expected_records:

        raise RuntimeError(
            f"Expected {expected_records} feature records, "
            f"found {len(records_df)}."
        )

    global_df = build_global_summary(
        records_df
    )

    level_df = build_level_summary(
        records_df
    )

    save_outputs(
        records_df,
        global_df,
        level_df,
    )


if __name__ == "__main__":
    main()