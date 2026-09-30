"""
Part 2.34
Multi-Scale LFNN/RFNN Feature Representation Audit

Purpose
-------
Determine where LFNN/RFNN representation asymmetry appears
inside the Swin-UNETR network.

Analysis only.

This script:
    1. Loads the Part 2.27 best checkpoint.
    2. Uses the exact Part 2.20B geometry/canonical pipeline.
    3. Uses the same 25-case validation cohort.
    4. Uses the same 45 paired LFNN/RFNN observations.
    5. Captures features from multiple encoder/decoder layers.
    6. Samples those features at the exact transformed points.
    7. Compares LFNN vs RFNN representations.
    8. Saves CSV and JSON reports.

This script DOES NOT:
    - train the model
    - modify any checkpoint
    - modify Part 2.20B
    - modify Part 2.27
    - modify the dashboard
    - create voxel ground truth
    - claim segmentation accuracy
"""

from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from collections import defaultdict

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
    / "rsna_part234_multiscale_feature_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. IMPORT PART 2.20B
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:

    import segmentation_rsna_part220b_geometry_corrected_training as part220b

except ImportError as exc:

    raise ImportError(
        "\nCould not import Part 2.20B.\n\n"
        "Expected file:\n"
        "src\\segmentation_rsna_part220b_geometry_corrected_training.py\n"
    ) from exc


# ============================================================
# 3. CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
    / "part227_best_macro_disease.pth"
)

MODEL_SHAPE = (
    64,
    96,
    96,
)


# ============================================================
# 4. TARGET FEATURE LAYERS
# ============================================================

TARGET_FEATURES = [

    # Encoder
    "encoder1.layer.norm2",
    "encoder2.layer.norm2",
    "encoder3.layer.norm2",
    "encoder4.layer.norm2",
    "encoder10.layer.norm2",

    # Decoder
    "decoder4.conv_block.norm2",
    "decoder3.conv_block.norm2",
    "decoder2.conv_block.norm2",
    "decoder1.conv_block.norm2",
]


# ============================================================
# 5. PRINT HEADER
# ============================================================

def print_header(title: str):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 6. SAFE FLOAT
# ============================================================

def safe_float(value):

    try:

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return np.nan

        return value

    except Exception:

        return np.nan


# ============================================================
# 7. COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=np.float64,
    ).reshape(-1)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


# ============================================================
# 8. FEATURE L2
# ============================================================

def feature_l2(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=np.float64,
    ).reshape(-1)

    return float(
        np.linalg.norm(a - b)
    )


# ============================================================
# 9. MEAN ABSOLUTE DIFFERENCE
# ============================================================

def mean_absolute_difference(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=np.float64,
    ).reshape(-1)

    return float(
        np.mean(
            np.abs(a - b)
        )
    )


# ============================================================
# 10. FEATURE NORM
# ============================================================

def safe_norm(
    value: np.ndarray,
) -> float:

    return float(
        np.linalg.norm(
            np.asarray(
                value,
                dtype=np.float64,
            ).reshape(-1)
        )
    )


# ============================================================
# 11. RESOLVE MODULE
# ============================================================

def resolve_module_path(
    model,
    module_path: str,
):

    current = model

    for part in module_path.split("."):

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


# ============================================================
# 12. CONVERT HOOK OUTPUT TO NUMPY
# ============================================================

def tensor_to_numpy(output):

    if isinstance(
        output,
        torch.Tensor,
    ):

        return (
            output
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    if isinstance(
        output,
        (tuple, list),
    ):

        for item in output:

            if isinstance(
                item,
                torch.Tensor,
            ):

                return (
                    item
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

    return None


# ============================================================
# 13. NORMALIZE FEATURE LAYOUT
# ============================================================

def normalize_feature_layout(
    feature: np.ndarray,
):

    feature = np.asarray(
        feature
    )

    # B,C,D,H,W
    if feature.ndim == 5:

        feature = feature[0]

    # C,D,H,W
    if feature.ndim != 4:

        raise ValueError(
            "Unexpected feature shape: "
            f"{feature.shape}"
        )

    # Standard project encoder/decoder format:
    # C,D,H,W
    if feature.shape[0] <= 256:

        return feature

    # Possible D,H,W,C format.
    if feature.shape[-1] <= 256:

        return np.transpose(
            feature,
            (3, 0, 1, 2),
        )

    raise ValueError(
        "Could not determine feature layout: "
        f"{feature.shape}"
    )


# ============================================================
# 14. SAMPLE FEATURE AT POINT
# ============================================================

def get_point_feature(
    feature: np.ndarray,
    z: float,
    y: float,
    x: float,
):

    feature = normalize_feature_layout(
        feature
    )

    channels, depth, height, width = (
        feature.shape
    )

    # Map canonical model coordinates:
    #
    # 64 x 96 x 96
    #
    # to the current feature resolution.

    fz = (
        z
        * (depth - 1)
        / max(
            MODEL_SHAPE[0] - 1,
            1,
        )
    )

    fy = (
        y
        * (height - 1)
        / max(
            MODEL_SHAPE[1] - 1,
            1,
        )
    )

    fx = (
        x
        * (width - 1)
        / max(
            MODEL_SHAPE[2] - 1,
            1,
        )
    )

    iz = int(round(fz))
    iy = int(round(fy))
    ix = int(round(fx))

    iz = max(
        0,
        min(
            depth - 1,
            iz,
        ),
    )

    iy = max(
        0,
        min(
            height - 1,
            iy,
        ),
    )

    ix = max(
        0,
        min(
            width - 1,
            ix,
        ),
    )

    vector = feature[
        :,
        iz,
        iy,
        ix,
    ]

    location = {

        "feature_z":
            iz,

        "feature_y":
            iy,

        "feature_x":
            ix,

        "feature_shape":
            (
                int(channels),
                int(depth),
                int(height),
                int(width),
            ),
    }

    return (
        vector.astype(
            np.float32
        ),
        location,
    )


# ============================================================
# 15. FEATURE HOOK MANAGER
# ============================================================

class FeatureHookManager:

    def __init__(
        self,
        model,
        target_names,
    ):

        self.model = model

        self.target_names = list(
            target_names
        )

        self.outputs = {}

        self.handles = []

    def _make_hook(
        self,
        name,
    ):

        def hook(
            module,
            inputs,
            output,
        ):

            self.outputs[name] = (
                tensor_to_numpy(
                    output
                )
            )

        return hook

    def register(self):

        print_header(
            "REGISTERING FEATURE HOOKS"
        )

        for name in self.target_names:

            module = resolve_module_path(
                self.model,
                name,
            )

            if module is None:

                print(
                    f"[WARNING] Layer not found: "
                    f"{name}"
                )

                continue

            handle = (
                module.register_forward_hook(
                    self._make_hook(
                        name
                    )
                )
            )

            self.handles.append(
                handle
            )

            print(
                f"[OK] {name}"
            )

        if not self.handles:

            raise RuntimeError(
                "No feature hooks were registered."
            )

    def clear(self):

        self.outputs.clear()

    def remove(self):

        for handle in self.handles:

            handle.remove()

        self.handles.clear()


# ============================================================
# 16. LOAD MODEL
# ============================================================

def load_model():

    print_header(
        "LOADING PART 2.27 BEST CHECKPOINT"
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{CHECKPOINT_PATH}\n"
        )

    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    model = part220b.build_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    # --------------------------------------------------------
    # Resolve checkpoint structure.
    # --------------------------------------------------------

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

        elif (
            "model"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove DataParallel prefix if present.
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in (
        state_dict.items()
    ):

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    print(
        f"Missing keys: {len(missing)}"
    )

    print(
        f"Unexpected keys: {len(unexpected)}"
    )

    if missing:

        print(
            "\nFirst missing keys:"
        )

        for key in missing[:10]:

            print(
                "  ",
                key,
            )

    if unexpected:

        print(
            "\nFirst unexpected keys:"
        )

        for key in unexpected[:10]:

            print(
                "  ",
                key,
            )

    model = model.to(
        DEVICE
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "\n[OK] Model loaded."
    )

    print(
        f"Parameter count: "
        f"{parameter_count:,}"
    )

    return model


# ============================================================
# 17. LOAD EXACT VALIDATION MANIFEST
# ============================================================

def get_validation_manifest():

    print_header(
        "LOADING EXACT PART 2.20B VALIDATION COHORT"
    )

    # --------------------------------------------------------
    # Load complete point-supervision manifest.
    # --------------------------------------------------------

    manifest = (
        part220b.load_manifest()
    )

    print(
        f"Full manifest rows: "
        f"{len(manifest):,}"
    )

    # --------------------------------------------------------
    # Get exact validation cohort.
    # --------------------------------------------------------

    validation_series = (
        part220b.select_validation_series(
            manifest
        )
    )

    print(
        f"Validation series returned: "
        f"{len(validation_series):,}"
    )

    # --------------------------------------------------------
    # If DataFrame, recover complete point rows.
    # --------------------------------------------------------

    if isinstance(
        validation_series,
        pd.DataFrame,
    ):

        required_ids = [
            "study_id",
            "series_id",
        ]

        missing_ids = [
            column
            for column
            in required_ids
            if column
            not in validation_series.columns
        ]

        if missing_ids:

            raise RuntimeError(
                "Validation DataFrame is missing: "
                f"{missing_ids}"
            )

        point_columns = [
            "class_name",
            "level",
            "native_z",
            "native_y",
            "native_x",
        ]

        has_point_columns = all(
            column
            in validation_series.columns
            for column
            in point_columns
        )

        if has_point_columns:

            validation_df = (
                validation_series.copy()
            )

        else:

            validation_keys = (
                validation_series[
                    [
                        "study_id",
                        "series_id",
                    ]
                ]
                .drop_duplicates()
                .copy()
            )

            validation_keys[
                "study_id"
            ] = (
                validation_keys[
                    "study_id"
                ]
                .astype(str)
            )

            validation_keys[
                "series_id"
            ] = (
                validation_keys[
                    "series_id"
                ]
                .astype(str)
            )

            manifest_work = (
                manifest.copy()
            )

            manifest_work[
                "study_id"
            ] = (
                manifest_work[
                    "study_id"
                ]
                .astype(str)
            )

            manifest_work[
                "series_id"
            ] = (
                manifest_work[
                    "series_id"
                ]
                .astype(str)
            )

            validation_df = (
                manifest_work.merge(
                    validation_keys,
                    on=[
                        "study_id",
                        "series_id",
                    ],
                    how="inner",
                )
            )

    else:

        # ----------------------------------------------------
        # Fallback for list/tuple/dict validation IDs.
        # ----------------------------------------------------

        validation_pairs = set()

        for item in validation_series:

            if isinstance(
                item,
                (tuple, list),
            ):

                if len(item) >= 2:

                    validation_pairs.add(
                        (
                            str(item[0]),
                            str(item[1]),
                        )
                    )

            elif isinstance(
                item,
                dict,
            ):

                if (
                    "study_id"
                    in item
                    and
                    "series_id"
                    in item
                ):

                    validation_pairs.add(
                        (
                            str(
                                item[
                                    "study_id"
                                ]
                            ),
                            str(
                                item[
                                    "series_id"
                                ]
                            ),
                        )
                    )

        if not validation_pairs:

            raise RuntimeError(
                "Could not interpret the "
                "validation cohort."
            )

        manifest_work = (
            manifest.copy()
        )

        manifest_work[
            "study_id"
        ] = (
            manifest_work[
                "study_id"
            ]
            .astype(str)
        )

        manifest_work[
            "series_id"
        ] = (
            manifest_work[
                "series_id"
            ]
            .astype(str)
        )

        validation_df = (
            manifest_work[
                manifest_work.apply(
                    lambda row: (
                        str(
                            row[
                                "study_id"
                            ]
                        ),
                        str(
                            row[
                                "series_id"
                            ]
                        ),
                    )
                    in validation_pairs,
                    axis=1,
                )
            ]
            .copy()
        )

    # --------------------------------------------------------
    # Remove duplicate point rows.
    # --------------------------------------------------------

    dedup_columns = [
        "study_id",
        "series_id",
        "class_name",
        "level",
        "native_z",
        "native_y",
        "native_x",
    ]

    available_dedup_columns = [
        column
        for column
        in dedup_columns
        if column
        in validation_df.columns
    ]

    validation_df = (
        validation_df
        .drop_duplicates(
            subset=available_dedup_columns
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Required columns.
    # --------------------------------------------------------

    required_columns = [
        "study_id",
        "series_id",
        "class_name",
        "level",
        "native_z",
        "native_y",
        "native_x",
    ]

    missing = [
        column
        for column
        in required_columns
        if column
        not in validation_df.columns
    ]

    if missing:

        raise RuntimeError(
            "Recovered validation manifest "
            "is missing columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Count LFNN/RFNN rows.
    # --------------------------------------------------------

    foraminal_names = {
        "Left Neural Foraminal Narrowing",
        "Right Neural Foraminal Narrowing",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "LFNN",
        "RFNN",
    }

    foraminal_mask = (
        validation_df[
            "class_name"
        ]
        .astype(str)
        .isin(
            foraminal_names
        )
    )

    foraminal_count = int(
        foraminal_mask.sum()
    )

    print(
        f"Recovered validation point rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Validation studies: "
        f"{validation_df['study_id'].nunique():,}"
    )

    print(
        f"Validation series: "
        f"{validation_df['series_id'].nunique():,}"
    )

    print(
        f"Validation LFNN/RFNN rows: "
        f"{foraminal_count:,}"
    )

    return validation_df


# ============================================================
# 18. BUILD CASE GROUPS
# ============================================================

def build_case_groups(
    validation_df,
):

    required_columns = [
        "study_id",
        "series_id",
        "class_name",
        "level",
        "native_z",
        "native_y",
        "native_x",
    ]

    missing = [
        column
        for column
        in required_columns
        if column
        not in validation_df.columns
    ]

    if missing:

        raise RuntimeError(
            "Validation manifest missing: "
            f"{missing}"
        )

    lfnn_names = {
        "Left Neural Foraminal Narrowing",
        "left_neural_foraminal_narrowing",
        "LFNN",
    }

    rfnn_names = {
        "Right Neural Foraminal Narrowing",
        "right_neural_foraminal_narrowing",
        "RFNN",
    }

    groups = []

    for (
        study_id,
        series_id,
    ), group in validation_df.groupby(
        [
            "study_id",
            "series_id",
        ],
        sort=True,
    ):

        group = group.copy()

        lfnn = group[
            group[
                "class_name"
            ]
            .astype(str)
            .isin(
                lfnn_names
            )
        ].copy()

        rfnn = group[
            group[
                "class_name"
            ]
            .astype(str)
            .isin(
                rfnn_names
            )
        ].copy()

        if (
            lfnn.empty
            or rfnn.empty
        ):

            continue

        groups.append(
            {
                "study_id":
                    str(study_id),

                "series_id":
                    str(series_id),

                "all_rows":
                    group,

                "lfnn":
                    lfnn,

                "rfnn":
                    rfnn,
            }
        )

    print(
        f"LFNN/RFNN candidate series: "
        f"{len(groups):,}"
    )

    return groups


# ============================================================
# 19. BUILD EXACT LFNN/RFNN PAIRS
# ============================================================

def build_pairs(
    groups,
):

    pairs = []

    for group in groups:

        lfnn = group[
            "lfnn"
        ]

        rfnn = group[
            "rfnn"
        ]

        levels = sorted(
            set(
                lfnn[
                    "level"
                ]
                .astype(str)
            )
            &
            set(
                rfnn[
                    "level"
                ]
                .astype(str)
            )
        )

        for level in levels:

            left_rows = (
                lfnn[
                    lfnn[
                        "level"
                    ]
                    .astype(str)
                    == level
                ]
                .sort_values(
                    [
                        "native_z",
                        "native_y",
                        "native_x",
                    ]
                )
                .copy()
            )

            right_rows = (
                rfnn[
                    rfnn[
                        "level"
                    ]
                    .astype(str)
                    == level
                ]
                .sort_values(
                    [
                        "native_z",
                        "native_y",
                        "native_x",
                    ]
                )
                .copy()
            )

            n = min(
                len(left_rows),
                len(right_rows),
            )

            for index in range(n):

                pairs.append(
                    {
                        "study_id":
                            group[
                                "study_id"
                            ],

                        "series_id":
                            group[
                                "series_id"
                            ],

                        "level":
                            level,

                        "lfnn_row":
                            left_rows.iloc[
                                index
                            ],

                        "rfnn_row":
                            right_rows.iloc[
                                index
                            ],
                    }
                )

    print(
        f"Paired LFNN/RFNN observations: "
        f"{len(pairs):,}"
    )

    return pairs


# ============================================================
# 20. RUN ONE CASE
# ============================================================

@torch.no_grad()
def run_case(
    model,
    hook_manager,
    study_id,
    series_id,
    point_rows,
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Part 2.20B receives point_rows and creates its returned
    # point list in the same row order.
    #
    # We preserve that order later.
    # --------------------------------------------------------

    canonical, points, geometry = (
        part220b.load_case(
            study_id,
            series_id,
            point_rows,
        )
    )

    image = np.asarray(
        canonical,
        dtype=np.float32,
    )

    if image.ndim != 3:

        raise ValueError(
            "Unexpected canonical image shape: "
            f"{image.shape}"
        )

    image_tensor = (
        torch.from_numpy(
            image
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    hook_manager.clear()

    logits = model(
        image_tensor
    )

    if isinstance(
        logits,
        (tuple, list),
    ):

        logits = logits[0]

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    probabilities = (
        probabilities
        .detach()
        .cpu()
        .numpy()[0]
    )

    return (
        points,
        hook_manager.outputs.copy(),
        probabilities,
        image.shape,
    )


# ============================================================
# 21. ANALYSE FEATURE PAIR
# ============================================================

def analyse_feature_pair(
    feature,
    lfnn_point,
    rfnn_point,
):

    if feature is None:
        return None

    lfnn_vector, lfnn_location = (
        get_point_feature(
            feature,
            float(
                lfnn_point["z"]
            ),
            float(
                lfnn_point["y"]
            ),
            float(
                lfnn_point["x"]
            ),
        )
    )

    rfnn_vector, rfnn_location = (
        get_point_feature(
            feature,
            float(
                rfnn_point["z"]
            ),
            float(
                rfnn_point["y"]
            ),
            float(
                rfnn_point["x"]
            ),
        )
    )

    return {

        "lfnn_feature_norm":
            safe_norm(
                lfnn_vector
            ),

        "rfnn_feature_norm":
            safe_norm(
                rfnn_vector
            ),

        "feature_cosine_similarity":
            cosine_similarity(
                lfnn_vector,
                rfnn_vector,
            ),

        "feature_l2_distance":
            feature_l2(
                lfnn_vector,
                rfnn_vector,
            ),

        "feature_mean_absolute_difference":
            mean_absolute_difference(
                lfnn_vector,
                rfnn_vector,
            ),

        "lfnn_feature_z":
            lfnn_location[
                "feature_z"
            ],

        "lfnn_feature_y":
            lfnn_location[
                "feature_y"
            ],

        "lfnn_feature_x":
            lfnn_location[
                "feature_x"
            ],

        "rfnn_feature_z":
            rfnn_location[
                "feature_z"
            ],

        "rfnn_feature_y":
            rfnn_location[
                "feature_y"
            ],

        "rfnn_feature_x":
            rfnn_location[
                "feature_x"
            ],

        "feature_shape":
            str(
                lfnn_location[
                    "feature_shape"
                ]
            ),
    }


# ============================================================
# 22. MAIN
# ============================================================

def main():

    print_header(
        "PART 2.34 - MULTI-SCALE FEATURE REPRESENTATION AUDIT"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Verify layers.
    # --------------------------------------------------------

    print_header(
        "VERIFYING TARGET FEATURE LAYERS"
    )

    available_layers = []

    for feature_name in TARGET_FEATURES:

        module = resolve_module_path(
            model,
            feature_name,
        )

        if module is None:

            print(
                f"[MISSING] {feature_name}"
            )

        else:

            print(
                f"[OK] {feature_name} -> "
                f"{module.__class__.__name__}"
            )

            available_layers.append(
                feature_name
            )

    if not available_layers:

        raise RuntimeError(
            "No requested feature layers found."
        )

    # --------------------------------------------------------
    # Hooks.
    # --------------------------------------------------------

    hook_manager = FeatureHookManager(
        model,
        available_layers,
    )

    hook_manager.register()

    # --------------------------------------------------------
    # Validation manifest.
    # --------------------------------------------------------

    validation_df = (
        get_validation_manifest()
    )

    groups = build_case_groups(
        validation_df
    )

    pairs = build_pairs(
        groups
    )

    if not pairs:

        raise RuntimeError(
            "No LFNN/RFNN pairs found."
        )

    # --------------------------------------------------------
    # Group pairs by case.
    # --------------------------------------------------------

    pairs_by_case = (
        defaultdict(list)
    )

    for pair in pairs:

        key = (
            pair["study_id"],
            pair["series_id"],
        )

        pairs_by_case[
            key
        ].append(
            pair
        )

    print_header(
        "PAIRING SUMMARY"
    )

    print(
        f"Validation cases with pairs: "
        f"{len(pairs_by_case):,}"
    )

    print(
        f"Total LFNN/RFNN pairs: "
        f"{len(pairs):,}"
    )

    # --------------------------------------------------------
    # Result storage.
    # --------------------------------------------------------

    feature_records = []

    prediction_records = []

    successful_cases = 0

    # ========================================================
    # PROCESS EACH CASE
    # ========================================================

    for case_number, (
        case_key,
        case_pairs,
    ) in enumerate(
        sorted(
            pairs_by_case.items()
        ),
        start=1,
    ):

        study_id, series_id = (
            case_key
        )

        print(
            f"\n[{case_number}/{len(pairs_by_case)}] "
            f"Study={study_id} "
            f"Series={series_id}"
        )

        try:

            # ------------------------------------------------
            # Collect original manifest rows.
            #
            # Preserve DataFrame index/order.
            # ------------------------------------------------

            row_indices = []

            for pair in case_pairs:

                row_indices.append(
                    pair[
                        "lfnn_row"
                    ].name
                )

                row_indices.append(
                    pair[
                        "rfnn_row"
                    ].name
                )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # sort=False and explicit index selection preserve
            # the exact row order supplied to Part 2.20B.
            # ------------------------------------------------

            point_rows = (
                validation_df.loc[
                    row_indices
                ].copy()
            )

            # ------------------------------------------------
            # Run case.
            # ------------------------------------------------

            (
                transformed_points,
                feature_outputs,
                probabilities,
                image_shape,
            ) = run_case(
                model,
                hook_manager,
                study_id,
                series_id,
                point_rows,
            )

            successful_cases += 1

            print(
                f"  Image shape: "
                f"{image_shape}"
            )

            print(
                f"  Captured features: "
                f"{len(feature_outputs)}"
            )

            print(
                f"  Original point rows: "
                f"{len(point_rows)}"
            )

            print(
                f"  Transformed points: "
                f"{len(transformed_points)}"
            )

            # ------------------------------------------------
            # CRITICAL IDENTITY CHECK.
            # ------------------------------------------------

            if (
                len(point_rows)
                != len(transformed_points)
            ):

                raise RuntimeError(
                    "Part 2.20B point count mismatch: "
                    f"original={len(point_rows)}, "
                    f"transformed={len(transformed_points)}"
                )

            # ------------------------------------------------
            # Exact positional identity map.
            #
            # original DataFrame index
            #       ->
            # transformed point
            # ------------------------------------------------

            transformed_by_manifest_index = {}

            for position, manifest_index in enumerate(
                point_rows.index
            ):

                transformed_by_manifest_index[
                    manifest_index
                ] = transformed_points[
                    position
                ]

            print(
                f"  Point identity mapping: "
                f"{len(transformed_by_manifest_index)}"
            )

            # ------------------------------------------------
            # Process LFNN/RFNN pairs.
            # ------------------------------------------------

            for pair_index, pair in enumerate(
                case_pairs,
                start=1,
            ):

                lfnn_row = pair[
                    "lfnn_row"
                ]

                rfnn_row = pair[
                    "rfnn_row"
                ]

                # --------------------------------------------
                # Exact manifest-index mapping.
                # --------------------------------------------

                lfnn_point = (
                    transformed_by_manifest_index.get(
                        lfnn_row.name
                    )
                )

                rfnn_point = (
                    transformed_by_manifest_index.get(
                        rfnn_row.name
                    )
                )

                if (
                    lfnn_point is None
                    or rfnn_point is None
                ):

                    print(
                        f"  [WARNING] Could not map "
                        f"pair {pair_index}"
                    )

                    continue

                # --------------------------------------------
                # Canonical coordinates.
                # --------------------------------------------

                lz = int(
                    round(
                        float(
                            lfnn_point["z"]
                        )
                    )
                )

                ly = int(
                    round(
                        float(
                            lfnn_point["y"]
                        )
                    )
                )

                lx = int(
                    round(
                        float(
                            lfnn_point["x"]
                        )
                    )
                )

                rz = int(
                    round(
                        float(
                            rfnn_point["z"]
                        )
                    )
                )

                ry = int(
                    round(
                        float(
                            rfnn_point["y"]
                        )
                    )
                )

                rx = int(
                    round(
                        float(
                            rfnn_point["x"]
                        )
                    )
                )

                # --------------------------------------------
                # Clamp to probability volume.
                # --------------------------------------------

                lz = max(
                    0,
                    min(
                        probabilities.shape[1] - 1,
                        lz,
                    ),
                )

                ly = max(
                    0,
                    min(
                        probabilities.shape[2] - 1,
                        ly,
                    ),
                )

                lx = max(
                    0,
                    min(
                        probabilities.shape[3] - 1,
                        lx,
                    ),
                )

                rz = max(
                    0,
                    min(
                        probabilities.shape[1] - 1,
                        rz,
                    ),
                )

                ry = max(
                    0,
                    min(
                        probabilities.shape[2] - 1,
                        ry,
                    ),
                )

                rx = max(
                    0,
                    min(
                        probabilities.shape[3] - 1,
                        rx,
                    ),
                )

                # --------------------------------------------
                # Point probabilities.
                # --------------------------------------------

                l_probs = probabilities[
                    :,
                    lz,
                    ly,
                    lx,
                ]

                r_probs = probabilities[
                    :,
                    rz,
                    ry,
                    rx,
                ]

                l_pred = int(
                    np.argmax(
                        l_probs
                    )
                )

                r_pred = int(
                    np.argmax(
                        r_probs
                    )
                )

                lfnn_class_id = int(
                    lfnn_row[
                        "class_id"
                    ]
                )

                rfnn_class_id = int(
                    rfnn_row[
                        "class_id"
                    ]
                )

                # --------------------------------------------
                # Prediction record.
                # --------------------------------------------

                prediction_records.append(
                    {

                        "study_id":
                            study_id,

                        "series_id":
                            series_id,

                        "level":
                            pair["level"],

                        "lfnn_class_id":
                            lfnn_class_id,

                        "rfnn_class_id":
                            rfnn_class_id,

                        "lfnn_z":
                            float(
                                lfnn_point["z"]
                            ),

                        "lfnn_y":
                            float(
                                lfnn_point["y"]
                            ),

                        "lfnn_x":
                            float(
                                lfnn_point["x"]
                            ),

                        "rfnn_z":
                            float(
                                rfnn_point["z"]
                            ),

                        "rfnn_y":
                            float(
                                rfnn_point["y"]
                            ),

                        "rfnn_x":
                            float(
                                rfnn_point["x"]
                            ),

                        "lfnn_pred_class":
                            l_pred,

                        "rfnn_pred_class":
                            r_pred,

                        "lfnn_background_probability":
                            float(
                                l_probs[0]
                            ),

                        "rfnn_background_probability":
                            float(
                                r_probs[0]
                            ),

                        "lfnn_true_probability":
                            float(
                                l_probs[
                                    lfnn_class_id
                                ]
                            ),

                        "rfnn_true_probability":
                            float(
                                r_probs[
                                    rfnn_class_id
                                ]
                            ),

                        "lfnn_rfnn_margin":
                            float(
                                l_probs[
                                    rfnn_class_id
                                ]
                                -
                                l_probs[0]
                            ),

                        "rfnn_rfnn_margin":
                            float(
                                r_probs[
                                    rfnn_class_id
                                ]
                                -
                                r_probs[0]
                            ),
                    }
                )

                # --------------------------------------------
                # Feature representation analysis.
                # --------------------------------------------

                for feature_name in available_layers:

                    feature = (
                        feature_outputs.get(
                            feature_name
                        )
                    )

                    if feature is None:

                        continue

                    try:

                        stats = (
                            analyse_feature_pair(
                                feature,
                                lfnn_point,
                                rfnn_point,
                            )
                        )

                        if stats is None:
                            continue

                        feature_records.append(
                            {

                                "study_id":
                                    study_id,

                                "series_id":
                                    series_id,

                                "level":
                                    pair["level"],

                                "feature":
                                    feature_name,

                                **stats,
                            }
                        )

                    except Exception as exc:

                        print(
                            f"  [WARNING] Feature "
                            f"analysis failed for "
                            f"{feature_name}: "
                            f"{exc}"
                        )

        except Exception as exc:

            print(
                f"  [ERROR] Case failed: "
                f"{exc}"
            )

            continue

    # --------------------------------------------------------
    # Remove hooks.
    # --------------------------------------------------------

    hook_manager.remove()

    # ========================================================
    # DATAFRAMES
    # ========================================================

    feature_df = pd.DataFrame(
        feature_records
    )

    prediction_df = pd.DataFrame(
        prediction_records
    )

    # ========================================================
    # SAVE RAW FEATURE RESULTS
    # ========================================================

    feature_csv = (
        OUTPUT_DIR
        / "part234_multiscale_feature_records.csv"
    )

    feature_df.to_csv(
        feature_csv,
        index=False,
    )

    # ========================================================
    # SAVE PREDICTION RESULTS
    # ========================================================

    prediction_csv = (
        OUTPUT_DIR
        / "part234_prediction_records.csv"
    )

    prediction_df.to_csv(
        prediction_csv,
        index=False,
    )

    # ========================================================
    # FEATURE SUMMARY
    # ========================================================

    print_header(
        "MULTI-SCALE FEATURE SUMMARY"
    )

    summary_rows = []

    if not feature_df.empty:

        for (
            feature_name,
            group,
        ) in feature_df.groupby(
            "feature",
            sort=False,
        ):

            summary_rows.append(
                {

                    "feature":
                        feature_name,

                    "pairs":
                        len(group),

                    "mean_feature_cosine":
                        group[
                            "feature_cosine_similarity"
                        ].mean(),

                    "median_feature_cosine":
                        group[
                            "feature_cosine_similarity"
                        ].median(),

                    "mean_feature_l2":
                        group[
                            "feature_l2_distance"
                        ].mean(),

                    "median_feature_l2":
                        group[
                            "feature_l2_distance"
                        ].median(),

                    "mean_feature_abs_difference":
                        group[
                            "feature_mean_absolute_difference"
                        ].mean(),

                    "lfnn_feature_norm":
                        group[
                            "lfnn_feature_norm"
                        ].mean(),

                    "rfnn_feature_norm":
                        group[
                            "rfnn_feature_norm"
                        ].mean(),

                    "norm_difference_rfnn_minus_lfnn":
                        (
                            group[
                                "rfnn_feature_norm"
                            ].mean()
                            -
                            group[
                                "lfnn_feature_norm"
                            ].mean()
                        ),
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        OUTPUT_DIR
        / "part234_multiscale_feature_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    if not summary_df.empty:

        print(
            summary_df.to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.6f}",
            )
        )

    else:

        print(
            "No feature summary could be generated."
        )

    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    prediction_summary = {}

    if not prediction_df.empty:

        prediction_summary = {

            "prediction_records":
                int(
                    len(
                        prediction_df
                    )
                ),

            "lfnn_mean_true_probability":
                float(
                    prediction_df[
                        "lfnn_true_probability"
                    ].mean()
                ),

            "rfnn_mean_true_probability":
                float(
                    prediction_df[
                        "rfnn_true_probability"
                    ].mean()
                ),

            "lfnn_mean_background_probability":
                float(
                    prediction_df[
                        "lfnn_background_probability"
                    ].mean()
                ),

            "rfnn_mean_background_probability":
                float(
                    prediction_df[
                        "rfnn_background_probability"
                    ].mean()
                ),

            "lfnn_mean_rfnn_margin":
                float(
                    prediction_df[
                        "lfnn_rfnn_margin"
                    ].mean()
                ),

            "rfnn_mean_rfnn_margin":
                float(
                    prediction_df[
                        "rfnn_rfnn_margin"
                    ].mean()
                ),
        }

    # ========================================================
    # ORDERED FEATURE INTERPRETATION
    # ========================================================

    interpretation = {}

    if not summary_df.empty:

        order_map = {
            feature_name: index
            for index, feature_name
            in enumerate(
                available_layers
            )
        }

        ordered = (
            summary_df.copy()
        )

        ordered["order"] = (
            ordered[
                "feature"
            ].map(
                order_map
            )
        )

        ordered = (
            ordered
            .sort_values(
                "order"
            )
            .reset_index(
                drop=True
            )
        )

        interpretation = {

            "earliest_layer":
                str(
                    ordered.iloc[0][
                        "feature"
                    ]
                ),

            "latest_layer":
                str(
                    ordered.iloc[-1][
                        "feature"
                    ]
                ),

            "earliest_cosine":
                safe_float(
                    ordered.iloc[0][
                        "mean_feature_cosine"
                    ]
                ),

            "latest_cosine":
                safe_float(
                    ordered.iloc[-1][
                        "mean_feature_cosine"
                    ]
                ),

            "earliest_l2":
                safe_float(
                    ordered.iloc[0][
                        "mean_feature_l2"
                    ]
                ),

            "latest_l2":
                safe_float(
                    ordered.iloc[-1][
                        "mean_feature_l2"
                    ]
                ),

            "note":
                (
                    "These measurements describe "
                    "LFNN/RFNN representation differences. "
                    "They do not establish causal attribution."
                ),
        }

    # ========================================================
    # COMPLETE JSON REPORT
    # ========================================================

    report = {

        "experiment":
            "Part 2.34 Multi-Scale Feature Representation Audit",

        "purpose":
            (
                "Determine where LFNN/RFNN representation "
                "differences emerge within Swin-UNETR."
            ),

        "analysis_only":
            True,

        "training_performed":
            False,

        "checkpoint_modified":
            False,

        "dashboard_modified":
            False,

        "checkpoint":
            str(
                CHECKPOINT_PATH
            ),

        "device":
            str(
                DEVICE
            ),

        "model_shape":
            list(
                MODEL_SHAPE
            ),

        "target_features":
            available_layers,

        "validation_manifest_rows":
            int(
                len(
                    validation_df
                )
            ),

        "validation_cases_with_pairs":
            int(
                len(
                    pairs_by_case
                )
            ),

        "paired_observations":
            int(
                len(
                    pairs
                )
            ),

        "successful_cases":
            int(
                successful_cases
            ),

        "feature_records":
            int(
                len(
                    feature_df
                )
            ),

        "prediction_records":
            int(
                len(
                    prediction_df
                )
            ),

        "prediction_summary":
            prediction_summary,

        "feature_interpretation":
            interpretation,

        "outputs":
            {

                "feature_records":
                    str(
                        feature_csv
                    ),

                "prediction_records":
                    str(
                        prediction_csv
                    ),

                "feature_summary":
                    str(
                        summary_csv
                    ),
            },
    }

    json_path = (
        OUTPUT_DIR
        / "part234_multiscale_feature_audit_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            allow_nan=True,
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print_header(
        "PART 2.34 COMPLETE"
    )

    print(
        f"Successful cases: "
        f"{successful_cases}"
    )

    print(
        f"Paired observations: "
        f"{len(pairs):,}"
    )

    print(
        f"Feature records: "
        f"{len(feature_df):,}"
    )

    print(
        f"Prediction records: "
        f"{len(prediction_df):,}"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {feature_csv}"
    )

    print(
        f"  {prediction_csv}"
    )

    print(
        f"  {summary_csv}"
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

    print(
        "\nExpected feature records if all 45 pairs "
        "and 9 layers succeed:"
    )

    print(
        "  45 × 9 = 405"
    )


# ============================================================
# 23. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()