"""
PART 2.65
STAGE-WISE RFNN/LFNN ATTRIBUTION AUDIT

Purpose
-------
Compare LFNN and RFNN responses through the Swin-UNETR network at:

    1. Encoder stages
    2. Bottleneck
    3. Decoder stages
    4. Final segmentation logits

This is an ANALYSIS-ONLY experiment.

No:
    - training
    - optimizer
    - checkpoint modification
    - dashboard modification

Protected checkpoint:
    Part 2.27 best macro-disease checkpoint

Validation cohort:
    Exact Part 2.20B validation cohort

Expected:
    25 validation cases
    9 paired validation series
    45 paired LFNN/RFNN observations

The script dynamically discovers feature tensors from the model so that
it does not assume fixed channel counts.
"""

from __future__ import annotations

import json
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
    / "rsna_part265_stagewise_rfnn_lfnn_attribution_audit"
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
LFNN_ID = 2
RFNN_ID = 3

EXPECTED_CASES = 25
EXPECTED_PAIRED_SERIES = 9
EXPECTED_PAIRS = 45


# ============================================================================
# FEATURE MODULES
# ============================================================================

FEATURE_NAMES = [
    "encoder1",
    "encoder2",
    "encoder3",
    "encoder4",
    "encoder10",
    "decoder4.conv_block.norm2",
    "decoder3.conv_block.norm2",
    "decoder2.conv_block.norm2",
    "decoder1.conv_block.norm2",
]


# ============================================================================
# VALIDATION COHORT
# ============================================================================

def build_validation_cases(manifest):

    selected = part220b.select_validation_series(
        manifest
    )

    if not isinstance(
        selected,
        pd.DataFrame,
    ):
        raise TypeError(
            "select_validation_series() must return a DataFrame."
        )

    if len(selected) != EXPECTED_CASES:

        raise RuntimeError(
            f"Expected {EXPECTED_CASES} validation cases, "
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

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
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
# MODULE LOOKUP
# ============================================================================

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


# ============================================================================
# INPUT PREPARATION
# ============================================================================

def prepare_image(
    image,
    device,
):

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

        image = image.unsqueeze(0)

    return (
        image
        .float()
        .to(device)
    )


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
# POINT MAPPING INTO FEATURE SPACE
# ============================================================================

def map_canonical_to_feature(
    canonical_point,
    input_shape,
    feature_shape,
):

    input_z = float(
        input_shape[0]
    )

    input_y = float(
        input_shape[1]
    )

    input_x = float(
        input_shape[2]
    )

    feature_z = float(
        feature_shape[-3]
    )

    feature_y = float(
        feature_shape[-2]
    )

    feature_x = float(
        feature_shape[-1]
    )

    z = (
        float(canonical_point[0])
        / max(input_z - 1.0, 1.0)
        * max(feature_z - 1.0, 1.0)
    )

    y = (
        float(canonical_point[1])
        / max(input_y - 1.0, 1.0)
        * max(feature_y - 1.0, 1.0)
    )

    x = (
        float(canonical_point[2])
        / max(input_x - 1.0, 1.0)
        * max(feature_x - 1.0, 1.0)
    )

    return (
        int(
            np.clip(
                round(z),
                0,
                feature_shape[-3] - 1,
            )
        ),
        int(
            np.clip(
                round(y),
                0,
                feature_shape[-2] - 1,
            )
        ),
        int(
            np.clip(
                round(x),
                0,
                feature_shape[-1] - 1,
            )
        ),
    )


# ============================================================================
# FEATURE SUMMARY AT POINT
# ============================================================================

def summarize_feature_at_point(
    feature,
    canonical_point,
    input_shape,
):

    if not torch.is_tensor(
        feature
    ):

        return None

    tensor = (
        feature
        .detach()
        .float()
    )

    if tensor.ndim != 5:

        return None

    batch = tensor.shape[0]

    if batch != 1:

        return None

    _, channels, depth, height, width = (
        tensor.shape
    )

    z, y, x = (
        map_canonical_to_feature(
            canonical_point,
            input_shape,
            tensor.shape,
        )
    )

    values = (
        tensor[
            0,
            :,
            z,
            y,
            x,
        ]
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    if len(values) == 0:

        return None

    norm = float(
        np.linalg.norm(
            values
        )
    )

    mean = float(
        np.mean(
            values
        )
    )

    std = float(
        np.std(
            values
        )
    )

    absolute_mean = float(
        np.mean(
            np.abs(values)
        )
    )

    positive_fraction = float(
        np.mean(
            values > 0
        )
    )

    maximum = float(
        np.max(
            values
        )
    )

    minimum = float(
        np.min(
            values
        )
    )

    return {
        "channels":
            int(channels),

        "feature_z":
            int(z),

        "feature_y":
            int(y),

        "feature_x":
            int(x),

        "feature_norm":
            norm,

        "feature_mean":
            mean,

        "feature_std":
            std,

        "feature_absolute_mean":
            absolute_mean,

        "feature_positive_fraction":
            positive_fraction,

        "feature_max":
            maximum,

        "feature_min":
            minimum,
    }


# ============================================================================
# FEATURE HOOK COLLECTOR
# ============================================================================

class FeatureCollector:

    def __init__(
        self,
        model,
        feature_names,
    ):

        self.model = model

        self.feature_names = (
            feature_names
        )

        self.features = {}

        self.handles = []

        for name in feature_names:

            module = (
                get_module_by_name(
                    model,
                    name,
                )
            )

            if module is None:

                raise RuntimeError(
                    f"Feature module not found: "
                    f"{name}"
                )

            handle = (
                module.register_forward_hook(
                    self._make_hook(name)
                )
            )

            self.handles.append(
                handle
            )

    def _make_hook(
        self,
        name,
    ):

        def hook(
            module,
            inputs,
            output,
        ):

            self.features[
                name
            ] = output

        return hook

    def clear(self):

        self.features = {}

    def remove(self):

        for handle in self.handles:

            handle.remove()

        self.handles = []


# ============================================================================
# OUTPUT METRICS
# ============================================================================

def calculate_output_metrics(
    logits,
):

    probabilities = (
        torch.softmax(
            torch.tensor(
                logits,
                dtype=torch.float32,
            ),
            dim=0,
        )
        .numpy()
    )

    rfnn_logit = float(
        logits[RFNN_ID]
    )

    lfnn_logit = float(
        logits[LFNN_ID]
    )

    background_logit = float(
        logits[BACKGROUND_ID]
    )

    rfnn_margin = (
        rfnn_logit
        -
        background_logit
    )

    lfnn_margin = (
        lfnn_logit
        -
        background_logit
    )

    return {
        "rfnn_logit":
            rfnn_logit,

        "lfnn_logit":
            lfnn_logit,

        "background_logit":
            background_logit,

        "rfnn_margin":
            rfnn_margin,

        "lfnn_margin":
            lfnn_margin,

        "rfnn_probability":
            float(
                probabilities[RFNN_ID]
            ),

        "lfnn_probability":
            float(
                probabilities[LFNN_ID]
            ),

        "background_probability":
            float(
                probabilities[
                    BACKGROUND_ID
                ]
            ),
    }


# ============================================================================
# OUTPUT AT POINT
# ============================================================================

def sample_output_at_point(
    output,
    canonical_point,
):

    tensor = (
        output
        .detach()
        .float()
    )

    if tensor.ndim != 5:

        raise RuntimeError(
            "Expected model output with shape "
            "(B,C,D,H,W)."
        )

    _, _, depth, height, width = (
        tensor.shape
    )

    z = int(
        np.clip(
            round(
                float(canonical_point[0])
            ),
            0,
            depth - 1,
        )
    )

    y = int(
        np.clip(
            round(
                float(canonical_point[1])
            ),
            0,
            height - 1,
        )
    )

    x = int(
        np.clip(
            round(
                float(canonical_point[2])
            ),
            0,
            width - 1,
        )
    )

    return (
        tensor[
            0,
            :,
            z,
            y,
            x,
        ]
        .cpu()
        .numpy()
    )


# ============================================================================
# PAIRED CASE ANALYSIS
# ============================================================================

def analyze_case(
    model,
    device,
    case,
):

    study_id = str(
        case["study_id"]
    )

    series_id = str(
        case["series_id"]
    )

    image, points, geometry = (
        part220b.load_case(
            study_id,
            series_id,
            case["points"],
        )
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

    common_levels = sorted(
        set(
            lfnn_points["level"]
        )
        &
        set(
            rfnn_points["level"]
        )
    )

    if len(common_levels) != 5:
        return []

    image_tensor = prepare_image(
        image,
        device,
    )

    input_shape = tuple(
        image_tensor.shape[-3:]
    )

    collector = FeatureCollector(
        model,
        FEATURE_NAMES,
    )

    records = []

    try:

        # ------------------------------------------------------------
        # ONE FORWARD PASS FOR THE CASE
        # ------------------------------------------------------------

        collector.clear()

        with torch.no_grad():

            output = model(
                image_tensor
            )

        # ------------------------------------------------------------
        # EACH LEVEL
        # ------------------------------------------------------------

        for level in common_levels:

            lfnn_rows = lfnn_points[
                lfnn_points["level"]
                == level
            ]

            rfnn_rows = rfnn_points[
                rfnn_points["level"]
                == level
            ]

            if len(lfnn_rows) != 1:
                continue

            if len(rfnn_rows) != 1:
                continue

            lfnn_point = (
                lfnn_rows.iloc[0]
            )

            rfnn_point = (
                rfnn_rows.iloc[0]
            )

            lfnn_canonical = (
                get_canonical_point(
                    lfnn_point,
                    geometry,
                )
            )

            rfnn_canonical = (
                get_canonical_point(
                    rfnn_point,
                    geometry,
                )
            )

            lfnn_logits = (
                sample_output_at_point(
                    output,
                    lfnn_canonical,
                )
            )

            rfnn_logits = (
                sample_output_at_point(
                    output,
                    rfnn_canonical,
                )
            )

            lfnn_output_metrics = (
                calculate_output_metrics(
                    lfnn_logits
                )
            )

            rfnn_output_metrics = (
                calculate_output_metrics(
                    rfnn_logits
                )
            )

            # --------------------------------------------------------
            # FEATURE STAGES
            # --------------------------------------------------------

            for feature_name in (
                FEATURE_NAMES
            ):

                feature = (
                    collector.features.get(
                        feature_name
                    )
                )

                if feature is None:

                    raise RuntimeError(
                        f"No captured feature for "
                        f"{feature_name}"
                    )

                lfnn_summary = (
                    summarize_feature_at_point(
                        feature,
                        lfnn_canonical,
                        input_shape,
                    )
                )

                rfnn_summary = (
                    summarize_feature_at_point(
                        feature,
                        rfnn_canonical,
                        input_shape,
                    )
                )

                if (
                    lfnn_summary is None
                    or
                    rfnn_summary is None
                ):

                    continue

                # ----------------------------------------------------
                # PAIRED DIFFERENCES
                # ----------------------------------------------------

                feature_norm_difference = (
                    rfnn_summary[
                        "feature_norm"
                    ]
                    -
                    lfnn_summary[
                        "feature_norm"
                    ]
                )

                feature_mean_difference = (
                    rfnn_summary[
                        "feature_mean"
                    ]
                    -
                    lfnn_summary[
                        "feature_mean"
                    ]
                )

                feature_abs_difference = (
                    rfnn_summary[
                        "feature_absolute_mean"
                    ]
                    -
                    lfnn_summary[
                        "feature_absolute_mean"
                    ]
                )

                feature_positive_difference = (
                    rfnn_summary[
                        "feature_positive_fraction"
                    ]
                    -
                    lfnn_summary[
                        "feature_positive_fraction"
                    ]
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

                        "lfnn_feature_norm":
                            lfnn_summary[
                                "feature_norm"
                            ],

                        "rfnn_feature_norm":
                            rfnn_summary[
                                "feature_norm"
                            ],

                        "delta_feature_norm":
                            feature_norm_difference,

                        "lfnn_feature_mean":
                            lfnn_summary[
                                "feature_mean"
                            ],

                        "rfnn_feature_mean":
                            rfnn_summary[
                                "feature_mean"
                            ],

                        "delta_feature_mean":
                            feature_mean_difference,

                        "lfnn_feature_std":
                            lfnn_summary[
                                "feature_std"
                            ],

                        "rfnn_feature_std":
                            rfnn_summary[
                                "feature_std"
                            ],

                        "lfnn_feature_absolute_mean":
                            lfnn_summary[
                                "feature_absolute_mean"
                            ],

                        "rfnn_feature_absolute_mean":
                            rfnn_summary[
                                "feature_absolute_mean"
                            ],

                        "delta_feature_absolute_mean":
                            feature_abs_difference,

                        "lfnn_positive_fraction":
                            lfnn_summary[
                                "feature_positive_fraction"
                            ],

                        "rfnn_positive_fraction":
                            rfnn_summary[
                                "feature_positive_fraction"
                            ],

                        "delta_positive_fraction":
                            feature_positive_difference,

                        "lfnn_feature_max":
                            lfnn_summary[
                                "feature_max"
                            ],

                        "rfnn_feature_max":
                            rfnn_summary[
                                "feature_max"
                            ],

                        "lfnn_feature_min":
                            lfnn_summary[
                                "feature_min"
                            ],

                        "rfnn_feature_min":
                            rfnn_summary[
                                "feature_min"
                            ],

                        # Final-output response at LFNN point
                        "lfnn_point_rfnn_logit":
                            lfnn_output_metrics[
                                "rfnn_logit"
                            ],

                        "lfnn_point_lfnn_logit":
                            lfnn_output_metrics[
                                "lfnn_logit"
                            ],

                        "lfnn_point_background_logit":
                            lfnn_output_metrics[
                                "background_logit"
                            ],

                        "lfnn_point_rfnn_margin":
                            lfnn_output_metrics[
                                "rfnn_margin"
                            ],

                        "lfnn_point_lfnn_margin":
                            lfnn_output_metrics[
                                "lfnn_margin"
                            ],

                        # Final-output response at RFNN point
                        "rfnn_point_rfnn_logit":
                            rfnn_output_metrics[
                                "rfnn_logit"
                            ],

                        "rfnn_point_lfnn_logit":
                            rfnn_output_metrics[
                                "lfnn_logit"
                            ],

                        "rfnn_point_background_logit":
                            rfnn_output_metrics[
                                "background_logit"
                            ],

                        "rfnn_point_rfnn_margin":
                            rfnn_output_metrics[
                                "rfnn_margin"
                            ],

                        "rfnn_point_lfnn_margin":
                            rfnn_output_metrics[
                                "lfnn_margin"
                            ],

                        "rfnn_point_rfnn_probability":
                            rfnn_output_metrics[
                                "rfnn_probability"
                            ],

                        "rfnn_point_background_probability":
                            rfnn_output_metrics[
                                "background_probability"
                            ],
                    }
                )

    finally:

        collector.remove()

    return records


# ============================================================================
# STAGE SUMMARY
# ============================================================================

def build_stage_summary(
    records_df,
):

    rows = []

    grouped = records_df.groupby(
        "feature",
        sort=False,
    )

    for feature, group in grouped:

        rows.append(
            {
                "feature":
                    feature,

                "records":
                    int(len(group)),

                "pairs":
                    int(
                        group[
                            [
                                "study_id",
                                "series_id",
                                "level",
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),

                "mean_delta_feature_norm":
                    float(
                        group[
                            "delta_feature_norm"
                        ].mean()
                    ),

                "median_delta_feature_norm":
                    float(
                        group[
                            "delta_feature_norm"
                        ].median()
                    ),

                "mean_delta_feature_mean":
                    float(
                        group[
                            "delta_feature_mean"
                        ].mean()
                    ),

                "mean_delta_absolute_activation":
                    float(
                        group[
                            "delta_feature_absolute_mean"
                        ].mean()
                    ),

                "mean_delta_positive_fraction":
                    float(
                        group[
                            "delta_positive_fraction"
                        ].mean()
                    ),

                "mean_rfnn_margin_at_rfnn":
                    float(
                        group[
                            "rfnn_point_rfnn_margin"
                        ].mean()
                    ),

                "mean_lfnn_margin_at_lfnn":
                    float(
                        group[
                            "lfnn_point_lfnn_margin"
                        ].mean()
                    ),

                "mean_rfnn_probability_at_rfnn":
                    float(
                        group[
                            "rfnn_point_rfnn_probability"
                        ].mean()
                    ),

                "mean_background_probability_at_rfnn":
                    float(
                        group[
                            "rfnn_point_background_probability"
                        ].mean()
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    if len(summary) > 0:

        summary[
            "absolute_mean_norm_difference"
        ] = (
            summary[
                "mean_delta_feature_norm"
            ].abs()
        )

    return summary


# ============================================================================
# LEVEL SUMMARY
# ============================================================================

def build_level_summary(
    records_df,
):

    rows = []

    grouped = records_df.groupby(
        [
            "feature",
            "level",
        ],
        sort=False,
    )

    for (
        feature,
        level,
    ), group in grouped:

        rows.append(
            {
                "feature":
                    feature,

                "level":
                    level,

                "records":
                    int(len(group)),

                "mean_delta_feature_norm":
                    float(
                        group[
                            "delta_feature_norm"
                        ].mean()
                    ),

                "mean_delta_feature_mean":
                    float(
                        group[
                            "delta_feature_mean"
                        ].mean()
                    ),

                "mean_delta_absolute_activation":
                    float(
                        group[
                            "delta_feature_absolute_mean"
                        ].mean()
                    ),

                "mean_delta_positive_fraction":
                    float(
                        group[
                            "delta_positive_fraction"
                        ].mean()
                    ),

                "mean_rfnn_margin":
                    float(
                        group[
                            "rfnn_point_rfnn_margin"
                        ].mean()
                    ),

                "mean_rfnn_probability":
                    float(
                        group[
                            "rfnn_point_rfnn_probability"
                        ].mean()
                    ),

                "mean_background_probability":
                    float(
                        group[
                            "rfnn_point_background_probability"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# STAGE ORDER SUMMARY
# ============================================================================

def build_stage_order_summary(
    stage_summary,
):

    stage_order = {
        "encoder1": 1,
        "encoder2": 2,
        "encoder3": 3,
        "encoder4": 4,
        "encoder10": 5,
        "decoder4.conv_block.norm2": 6,
        "decoder3.conv_block.norm2": 7,
        "decoder2.conv_block.norm2": 8,
        "decoder1.conv_block.norm2": 9,
    }

    summary = stage_summary.copy()

    summary[
        "stage_order"
    ] = (
        summary[
            "feature"
        ]
        .map(stage_order)
    )

    return summary.sort_values(
        "stage_order"
    )


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(
    records_df,
    stage_summary,
    level_summary,
    ordered_summary,
):

    records_path = (
        OUTPUT_DIR
        / "part265_stagewise_records.csv"
    )

    stage_path = (
        OUTPUT_DIR
        / "part265_stagewise_summary.csv"
    )

    level_path = (
        OUTPUT_DIR
        / "part265_stagewise_level_summary.csv"
    )

    ordered_path = (
        OUTPUT_DIR
        / "part265_stage_order_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part265_stagewise_audit_summary.json"
    )

    records_df.to_csv(
        records_path,
        index=False,
    )

    stage_summary.to_csv(
        stage_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    ordered_summary.to_csv(
        ordered_path,
        index=False,
    )

    summary = {
        "analysis":
            "Part 2.65 stage-wise RFNN/LFNN attribution audit",

        "checkpoint":
            str(
                CHECKPOINT_PATH
            ),

        "validation_cases":
            EXPECTED_CASES,

        "paired_validation_series":
            EXPECTED_PAIRED_SERIES,

        "paired_observations":
            EXPECTED_PAIRS,

        "features":
            FEATURE_NAMES,

        "record_count":
            int(
                len(records_df)
            ),

        "stage_summary":
            stage_summary.to_dict(
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
    print("PART 2.65 STAGE SUMMARY")
    print("=" * 80)

    print(
        ordered_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PART 2.65 COMPLETE")
    print("=" * 80)

    print(
        "Analysis only."
    )

    print(
        "No training."
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
        stage_path
    )

    print(
        level_path
    )

    print(
        ordered_path
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
    print("PART 2.65")
    print("STAGE-WISE RFNN/LFNN ATTRIBUTION AUDIT")
    print("=" * 80)

    print()
    print(
        "Analysis only."
    )

    print(
        "No training."
    )

    print(
        "No checkpoint modification."
    )

    print(
        "No dashboard modification."
    )

    print()
    print(
        "Feature stages:"
    )

    for name in FEATURE_NAMES:

        print(
            f"  {name}"
        )

    manifest = (
        part220b.load_manifest()
    )

    print()
    print(
        f"Full manifest rows: "
        f"{len(manifest)}"
    )

    validation_cases = (
        build_validation_cases(
            manifest
        )
    )

    print(
        f"Validation cases: "
        f"{len(validation_cases)}"
    )

    model, device = (
        load_model()
    )

    all_records = []

    paired_series = 0

    print()
    print("=" * 80)
    print(
        "RUNNING STAGE-WISE AUDIT"
    )
    print("=" * 80)

    for index, case in enumerate(
        validation_cases,
        start=1,
    ):

        study_id = str(
            case["study_id"]
        )

        series_id = str(
            case["series_id"]
        )

        try:

            case_records = (
                analyze_case(
                    model,
                    device,
                    case,
                )
            )

            if case_records:

                all_records.extend(
                    case_records
                )

                paired_series += 1

                print(
                    f"[{index:02d}/25] "
                    f"Study={study_id} | "
                    f"Series={series_id} | "
                    f"paired levels=5 | "
                    f"records={len(case_records)}"
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

    records_df = pd.DataFrame(
        all_records
    )

    print()
    print(
        f"Paired validation series: "
        f"{paired_series}/25"
    )

    print(
        f"Stage-wise records: "
        f"{len(records_df)}"
    )

    if paired_series != (
        EXPECTED_PAIRED_SERIES
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_PAIRED_SERIES} paired series, "
            f"found {paired_series}."
        )

    expected_records = (
        EXPECTED_PAIRS
        *
        len(FEATURE_NAMES)
    )

    if len(records_df) != (
        expected_records
    ):

        raise RuntimeError(
            f"Expected {expected_records} records, "
            f"found {len(records_df)}."
        )

    unique_pairs = (
        records_df[
            [
                "study_id",
                "series_id",
                "level",
            ]
        ]
        .drop_duplicates()
    )

    if len(unique_pairs) != (
        EXPECTED_PAIRS
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_PAIRS} paired observations, "
            f"found {len(unique_pairs)}."
        )

    print()
    print(
        "Stage-wise record validation: PASSED"
    )

    print(
        f"Paired observations: "
        f"{len(unique_pairs)}"
    )

    print(
        f"Feature stages: "
        f"{len(FEATURE_NAMES)}"
    )

    print(
        f"Expected records: "
        f"{expected_records}"
    )

    stage_summary = (
        build_stage_summary(
            records_df
        )
    )

    level_summary = (
        build_level_summary(
            records_df
        )
    )

    ordered_summary = (
        build_stage_order_summary(
            stage_summary
        )
    )

    save_outputs(
        records_df,
        stage_summary,
        level_summary,
        ordered_summary,
    )


if __name__ == "__main__":
    main()