"""
Part 2.35
Decoder Spatial Activation Audit

Analysis-only experiment.

Purpose:
    Investigate whether the LFNN/RFNN decoder representation
    difference found in Part 2.34 is spatially localized.

Uses:
    - Part 2.20B geometry/canonical pipeline
    - Part 2.27 best checkpoint
    - same validation cohort
    - same 45 LFNN/RFNN paired observations
    - decoder4, decoder3, decoder2, decoder1
    - radii 2, 4, 6

Does NOT:
    - train
    - modify checkpoints
    - modify dashboard
    - create voxel ground truth
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
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part235_decoder_spatial_activation_audit"
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

TARGET_FEATURES = [
    "decoder4.conv_block.norm2",
    "decoder3.conv_block.norm2",
    "decoder2.conv_block.norm2",
    "decoder1.conv_block.norm2",
]

RADII = [
    2,
    4,
    6,
]


# ============================================================
# 4. UTILITIES
# ============================================================

def print_header(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_float(value):

    try:

        value = float(value)

        if math.isnan(value):
            return np.nan

        if math.isinf(value):
            return np.nan

        return value

    except Exception:

        return np.nan


def resolve_module_path(model, module_path):

    current = model

    for part in module_path.split("."):

        if part.isdigit():

            current = current[int(part)]

        else:

            if not hasattr(current, part):
                return None

            current = getattr(current, part)

    return current


def tensor_to_numpy(output):

    if isinstance(output, torch.Tensor):

        return (
            output
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    if isinstance(output, (tuple, list)):

        for item in output:

            if isinstance(item, torch.Tensor):

                return (
                    item
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

    return None


# ============================================================
# 5. FEATURE LAYOUT
# ============================================================

def normalize_feature_layout(feature):

    feature = np.asarray(feature)

    # B,C,D,H,W
    if feature.ndim == 5:
        feature = feature[0]

    if feature.ndim != 4:

        raise ValueError(
            "Unexpected feature shape: "
            f"{feature.shape}"
        )

    # C,D,H,W
    if feature.shape[0] <= 256:
        return feature

    # D,H,W,C
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
# 6. FEATURE HOOK MANAGER
# ============================================================

class FeatureHookManager:

    def __init__(self, model, target_names):

        self.model = model

        self.target_names = list(
            target_names
        )

        self.outputs = {}

        self.handles = []

    def make_hook(self, name):

        def hook(module, inputs, output):

            self.outputs[name] = (
                tensor_to_numpy(output)
            )

        return hook

    def register(self):

        print_header(
            "REGISTERING DECODER FEATURE HOOKS"
        )

        for name in self.target_names:

            module = resolve_module_path(
                self.model,
                name,
            )

            if module is None:

                print(
                    f"[MISSING] {name}"
                )

                continue

            handle = (
                module.register_forward_hook(
                    self.make_hook(name)
                )
            )

            self.handles.append(handle)

            print(
                f"[OK] {name} -> "
                f"{module.__class__.__name__}"
            )

        if len(self.handles) == 0:

            raise RuntimeError(
                "No decoder hooks were registered."
            )

    def clear(self):

        self.outputs.clear()

    def remove(self):

        for handle in self.handles:
            handle.remove()

        self.handles.clear()


# ============================================================
# 7. LOAD MODEL
# ============================================================

def load_model():

    print_header(
        "LOADING PART 2.27 BEST CHECKPOINT"
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
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

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint["state_dict"]
            )

        elif "model" in checkpoint:

            state_dict = (
                checkpoint["model"]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state_dict[key] = value

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

    model = model.to(DEVICE)

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "\n[OK] Model loaded."
    )

    print(
        f"Parameter count: {parameter_count:,}"
    )

    return model


# ============================================================
# 8. VALIDATION MANIFEST
# ============================================================

def get_validation_manifest():

    print_header(
        "LOADING EXACT PART 2.20B VALIDATION COHORT"
    )

    manifest = part220b.load_manifest()

    print(
        f"Full manifest rows: {len(manifest):,}"
    )

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
    # Validation series is normally a DataFrame.
    # --------------------------------------------------------

    if isinstance(
        validation_series,
        pd.DataFrame,
    ):

        required_ids = [
            "study_id",
            "series_id",
        ]

        for column in required_ids:

            if column not in validation_series.columns:

                raise RuntimeError(
                    f"Validation cohort missing {column}"
                )

        required_points = [
            "class_name",
            "level",
            "native_z",
            "native_y",
            "native_x",
        ]

        if all(
            column in validation_series.columns
            for column in required_points
        ):

            validation_df = (
                validation_series.copy()
            )

        else:

            keys = (
                validation_series[
                    [
                        "study_id",
                        "series_id",
                    ]
                ]
                .drop_duplicates()
                .copy()
            )

            manifest_work = manifest.copy()

            manifest_work["study_id"] = (
                manifest_work["study_id"]
                .astype(str)
            )

            manifest_work["series_id"] = (
                manifest_work["series_id"]
                .astype(str)
            )

            keys["study_id"] = (
                keys["study_id"]
                .astype(str)
            )

            keys["series_id"] = (
                keys["series_id"]
                .astype(str)
            )

            validation_df = (
                manifest_work.merge(
                    keys,
                    on=[
                        "study_id",
                        "series_id",
                    ],
                    how="inner",
                )
            )

    else:

        # ----------------------------------------------------
        # Fallback for list-like validation output.
        # ----------------------------------------------------

        validation_pairs = set()

        for item in validation_series:

            if isinstance(item, (tuple, list)):

                if len(item) >= 2:

                    validation_pairs.add(
                        (
                            str(item[0]),
                            str(item[1]),
                        )
                    )

            elif isinstance(item, dict):

                if (
                    "study_id" in item
                    and "series_id" in item
                ):

                    validation_pairs.add(
                        (
                            str(item["study_id"]),
                            str(item["series_id"]),
                        )
                    )

        if len(validation_pairs) == 0:

            raise RuntimeError(
                "Could not interpret validation cohort."
            )

        manifest_work = manifest.copy()

        manifest_work["study_id"] = (
            manifest_work["study_id"]
            .astype(str)
        )

        manifest_work["series_id"] = (
            manifest_work["series_id"]
            .astype(str)
        )

        mask = manifest_work.apply(
            lambda row: (
                str(row["study_id"]),
                str(row["series_id"]),
            ) in validation_pairs,
            axis=1,
        )

        validation_df = (
            manifest_work[mask].copy()
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
        for column in required_columns
        if column not in validation_df.columns
    ]

    if missing:

        raise RuntimeError(
            "Recovered validation manifest "
            "is missing columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Deduplicate exact point rows.
    # --------------------------------------------------------

    validation_df = (
        validation_df
        .drop_duplicates(
            subset=required_columns
        )
        .reset_index(drop=True)
    )

    foraminal_names = {
        "Left Neural Foraminal Narrowing",
        "Right Neural Foraminal Narrowing",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "LFNN",
        "RFNN",
    }

    foraminal_mask = (
        validation_df["class_name"]
        .astype(str)
        .isin(foraminal_names)
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
        f"{int(foraminal_mask.sum()):,}"
    )

    return validation_df


# ============================================================
# 9. CASE GROUPS
# ============================================================

def build_case_groups(validation_df):

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

        lfnn = group[
            group["class_name"]
            .astype(str)
            .isin(lfnn_names)
        ].copy()

        rfnn = group[
            group["class_name"]
            .astype(str)
            .isin(rfnn_names)
        ].copy()

        if lfnn.empty or rfnn.empty:
            continue

        groups.append(
            {
                "study_id": str(study_id),
                "series_id": str(series_id),
                "lfnn": lfnn,
                "rfnn": rfnn,
            }
        )

    print(
        f"LFNN/RFNN candidate series: "
        f"{len(groups):,}"
    )

    return groups


# ============================================================
# 10. BUILD PAIRS
# ============================================================

def build_pairs(groups):

    pairs = []

    for group in groups:

        lfnn = group["lfnn"]
        rfnn = group["rfnn"]

        common_levels = sorted(
            set(
                lfnn["level"].astype(str)
            )
            &
            set(
                rfnn["level"].astype(str)
            )
        )

        for level in common_levels:

            left_rows = (
                lfnn[
                    lfnn["level"].astype(str)
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
                    rfnn["level"].astype(str)
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
                        "study_id": group["study_id"],
                        "series_id": group["series_id"],
                        "level": level,
                        "lfnn_row": left_rows.iloc[index],
                        "rfnn_row": right_rows.iloc[index],
                    }
                )

    print(
        f"Paired LFNN/RFNN observations: "
        f"{len(pairs):,}"
    )

    return pairs


# ============================================================
# 11. RUN CASE
# ============================================================

@torch.no_grad()
def run_case(
    model,
    hook_manager,
    study_id,
    series_id,
    point_rows,
):

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
            "Unexpected image shape: "
            f"{image.shape}"
        )

    image_tensor = (
        torch.from_numpy(image)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    hook_manager.clear()

    logits = model(image_tensor)

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
# 12. POINT TO FEATURE COORDINATE
# ============================================================

def point_to_feature_coordinate(
    point,
    feature,
):

    feature = normalize_feature_layout(
        feature
    )

    channels, depth, height, width = (
        feature.shape
    )

    z = float(point["z"])
    y = float(point["y"])
    x = float(point["x"])

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
        min(depth - 1, iz),
    )

    iy = max(
        0,
        min(height - 1, iy),
    )

    ix = max(
        0,
        min(width - 1, ix),
    )

    return (
        iz,
        iy,
        ix,
    )


# ============================================================
# 13. LOCAL ACTIVATION STATISTICS
# ============================================================

def neighborhood_statistics(
    feature,
    point,
    radius,
):

    feature = normalize_feature_layout(
        feature
    )

    channels, depth, height, width = (
        feature.shape
    )

    z, y, x = (
        point_to_feature_coordinate(
            point,
            feature,
        )
    )

    z0 = max(
        0,
        z - radius,
    )

    z1 = min(
        depth - 1,
        z + radius,
    )

    y0 = max(
        0,
        y - radius,
    )

    y1 = min(
        height - 1,
        y + radius,
    )

    x0 = max(
        0,
        x - radius,
    )

    x1 = min(
        width - 1,
        x + radius,
    )

    block = feature[
        :,
        z0:z1 + 1,
        y0:y1 + 1,
        x0:x1 + 1,
    ]

    values = block.reshape(
        channels,
        -1,
    )

    channel_mean = values.mean(axis=1)
    channel_std = values.std(axis=1)
    channel_min = values.min(axis=1)
    channel_max = values.max(axis=1)

    channel_energy = np.sqrt(
        np.mean(
            values ** 2,
            axis=1,
        )
    )

    return {
        "feature_z": int(z),
        "feature_y": int(y),
        "feature_x": int(x),
        "feature_channels": int(channels),
        "feature_depth": int(depth),
        "feature_height": int(height),
        "feature_width": int(width),

        "neighborhood_voxels": int(
            block.shape[1]
            * block.shape[2]
            * block.shape[3]
        ),

        "activation_mean": float(
            channel_mean.mean()
        ),

        "activation_std": float(
            channel_std.mean()
        ),

        "activation_min": float(
            channel_min.mean()
        ),

        "activation_max": float(
            channel_max.mean()
        ),

        "activation_energy": float(
            channel_energy.mean()
        ),

        "activation_abs_mean": float(
            np.mean(
                np.abs(values)
            )
        ),

        "feature_norm": float(
            np.linalg.norm(values)
        ),
    }


# ============================================================
# 14. POINT FEATURE VECTOR
# ============================================================

def point_activation_vector(
    feature,
    point,
):

    feature = normalize_feature_layout(
        feature
    )

    z, y, x = (
        point_to_feature_coordinate(
            point,
            feature,
        )
    )

    vector = feature[
        :,
        z,
        y,
        x,
    ]

    return vector.astype(
        np.float32
    )


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print_header(
        "PART 2.35 - DECODER SPATIAL ACTIVATION AUDIT"
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

    print(
        f"Radii: {RADII}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Verify decoder layers
    # --------------------------------------------------------

    print_header(
        "VERIFYING DECODER FEATURE LAYERS"
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

    if len(available_layers) == 0:

        raise RuntimeError(
            "No decoder layers found."
        )

    # --------------------------------------------------------
    # Hooks
    # --------------------------------------------------------

    hook_manager = FeatureHookManager(
        model,
        available_layers,
    )

    hook_manager.register()

    # --------------------------------------------------------
    # Validation cohort
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

    if len(pairs) == 0:

        raise RuntimeError(
            "No LFNN/RFNN pairs found."
        )

    pairs_by_case = defaultdict(list)

    for pair in pairs:

        key = (
            pair["study_id"],
            pair["series_id"],
        )

        pairs_by_case[key].append(
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
    # Storage
    # --------------------------------------------------------

    records = []

    prediction_records = []

    successful_cases = 0

    # ========================================================
    # CASE LOOP
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

        study_id, series_id = case_key

        print(
            f"\n[{case_number}/{len(pairs_by_case)}] "
            f"Study={study_id} "
            f"Series={series_id}"
        )

        try:

            # ------------------------------------------------
            # Exact original manifest rows.
            # ------------------------------------------------

            row_indices = []

            for pair in case_pairs:

                row_indices.append(
                    pair["lfnn_row"].name
                )

                row_indices.append(
                    pair["rfnn_row"].name
                )

            point_rows = (
                validation_df.loc[
                    row_indices
                ].copy()
            )

            # ------------------------------------------------
            # Exact Part 2.20B processing.
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
                f"  Image shape: {image_shape}"
            )

            print(
                f"  Captured decoder features: "
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

            if (
                len(point_rows)
                != len(transformed_points)
            ):

                raise RuntimeError(
                    "Point count mismatch: "
                    f"original={len(point_rows)}, "
                    f"transformed={len(transformed_points)}"
                )

            # ------------------------------------------------
            # Exact positional identity.
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

            # =================================================
            # PAIR LOOP
            # =================================================

            for pair_index, pair in enumerate(
                case_pairs,
                start=1,
            ):

                lfnn_row = pair["lfnn_row"]
                rfnn_row = pair["rfnn_row"]

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

                # ------------------------------------------------
                # Prediction values at the exact canonical points.
                # ------------------------------------------------

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

                lfnn_class_id = int(
                    lfnn_row["class_id"]
                )

                rfnn_class_id = int(
                    rfnn_row["class_id"]
                )

                prediction_records.append(
                    {
                        "study_id": study_id,
                        "series_id": series_id,
                        "level": pair["level"],

                        "lfnn_true_probability": float(
                            l_probs[
                                lfnn_class_id
                            ]
                        ),

                        "rfnn_true_probability": float(
                            r_probs[
                                rfnn_class_id
                            ]
                        ),

                        "lfnn_background_probability": float(
                            l_probs[0]
                        ),

                        "rfnn_background_probability": float(
                            r_probs[0]
                        ),

                        "lfnn_rfnn_margin": float(
                            l_probs[
                                rfnn_class_id
                            ]
                            - l_probs[0]
                        ),

                        "rfnn_rfnn_margin": float(
                            r_probs[
                                rfnn_class_id
                            ]
                            - r_probs[0]
                        ),
                    }
                )

                # =================================================
                # DECODER FEATURE ANALYSIS
                # =================================================

                for feature_name in available_layers:

                    feature = (
                        feature_outputs.get(
                            feature_name
                        )
                    )

                    if feature is None:
                        continue

                    feature = normalize_feature_layout(
                        feature
                    )

                    # ------------------------------------------------
                    # Point vectors
                    # ------------------------------------------------

                    lfnn_vector = (
                        point_activation_vector(
                            feature,
                            lfnn_point,
                        )
                    )

                    rfnn_vector = (
                        point_activation_vector(
                            feature,
                            rfnn_point,
                        )
                    )

                    denominator = (
                        np.linalg.norm(
                            lfnn_vector
                        )
                        *
                        np.linalg.norm(
                            rfnn_vector
                        )
                    )

                    if denominator <= 1e-12:

                        point_cosine = 0.0

                    else:

                        point_cosine = float(
                            np.dot(
                                lfnn_vector,
                                rfnn_vector,
                            )
                            / denominator
                        )

                    point_l2 = float(
                        np.linalg.norm(
                            lfnn_vector
                            - rfnn_vector
                        )
                    )

                    point_abs_difference = float(
                        np.mean(
                            np.abs(
                                lfnn_vector
                                - rfnn_vector
                            )
                        )
                    )

                    # ------------------------------------------------
                    # Radius analysis
                    # ------------------------------------------------

                    for radius in RADII:

                        lfnn_stats = (
                            neighborhood_statistics(
                                feature,
                                lfnn_point,
                                radius,
                            )
                        )

                        rfnn_stats = (
                            neighborhood_statistics(
                                feature,
                                rfnn_point,
                                radius,
                            )
                        )

                        lfnn_energy = (
                            lfnn_stats[
                                "activation_energy"
                            ]
                        )

                        rfnn_energy = (
                            rfnn_stats[
                                "activation_energy"
                            ]
                        )

                        lfnn_norm = (
                            lfnn_stats[
                                "feature_norm"
                            ]
                        )

                        rfnn_norm = (
                            rfnn_stats[
                                "feature_norm"
                            ]
                        )

                        if abs(lfnn_energy) <= 1e-12:

                            energy_ratio = np.nan

                        else:

                            energy_ratio = (
                                rfnn_energy
                                / abs(
                                    lfnn_energy
                                )
                            )

                        if abs(lfnn_norm) <= 1e-12:

                            norm_ratio = np.nan

                        else:

                            norm_ratio = (
                                rfnn_norm
                                / abs(
                                    lfnn_norm
                                )
                            )

                        records.append(
                            {
                                "study_id": study_id,
                                "series_id": series_id,
                                "level": pair["level"],
                                "feature": feature_name,
                                "radius": int(radius),

                                "lfnn_activation_mean":
                                    lfnn_stats[
                                        "activation_mean"
                                    ],

                                "lfnn_activation_std":
                                    lfnn_stats[
                                        "activation_std"
                                    ],

                                "lfnn_activation_min":
                                    lfnn_stats[
                                        "activation_min"
                                    ],

                                "lfnn_activation_max":
                                    lfnn_stats[
                                        "activation_max"
                                    ],

                                "lfnn_activation_energy":
                                    lfnn_energy,

                                "lfnn_activation_abs_mean":
                                    lfnn_stats[
                                        "activation_abs_mean"
                                    ],

                                "lfnn_feature_norm":
                                    lfnn_norm,

                                "rfnn_activation_mean":
                                    rfnn_stats[
                                        "activation_mean"
                                    ],

                                "rfnn_activation_std":
                                    rfnn_stats[
                                        "activation_std"
                                    ],

                                "rfnn_activation_min":
                                    rfnn_stats[
                                        "activation_min"
                                    ],

                                "rfnn_activation_max":
                                    rfnn_stats[
                                        "activation_max"
                                    ],

                                "rfnn_activation_energy":
                                    rfnn_energy,

                                "rfnn_activation_abs_mean":
                                    rfnn_stats[
                                        "activation_abs_mean"
                                    ],

                                "rfnn_feature_norm":
                                    rfnn_norm,

                                "difference_activation_mean":
                                    (
                                        rfnn_stats[
                                            "activation_mean"
                                        ]
                                        -
                                        lfnn_stats[
                                            "activation_mean"
                                        ]
                                    ),

                                "difference_activation_energy":
                                    (
                                        rfnn_energy
                                        -
                                        lfnn_energy
                                    ),

                                "difference_activation_abs_mean":
                                    (
                                        rfnn_stats[
                                            "activation_abs_mean"
                                        ]
                                        -
                                        lfnn_stats[
                                            "activation_abs_mean"
                                        ]
                                    ),

                                "difference_feature_norm":
                                    (
                                        rfnn_norm
                                        -
                                        lfnn_norm
                                    ),

                                "rfnn_lfnn_energy_ratio":
                                    safe_float(
                                        energy_ratio
                                    ),

                                "rfnn_lfnn_feature_norm_ratio":
                                    safe_float(
                                        norm_ratio
                                    ),

                                "point_cosine_similarity":
                                    point_cosine,

                                "point_l2_distance":
                                    point_l2,

                                "point_mean_abs_difference":
                                    point_abs_difference,

                                "feature_channels":
                                    lfnn_stats[
                                        "feature_channels"
                                    ],

                                "feature_depth":
                                    lfnn_stats[
                                        "feature_depth"
                                    ],

                                "feature_height":
                                    lfnn_stats[
                                        "feature_height"
                                    ],

                                "feature_width":
                                    lfnn_stats[
                                        "feature_width"
                                    ],

                                "neighborhood_voxels":
                                    lfnn_stats[
                                        "neighborhood_voxels"
                                    ],
                            }
                        )

        except Exception as exc:

            print(
                f"  [ERROR] Case failed: {exc}"
            )

            continue

    # --------------------------------------------------------
    # Remove hooks.
    # --------------------------------------------------------

    hook_manager.remove()

    # ========================================================
    # DATAFRAMES
    # ========================================================

    records_df = pd.DataFrame(
        records
    )

    prediction_df = pd.DataFrame(
        prediction_records
    )

    # ========================================================
    # SAVE RAW RECORDS
    # ========================================================

    records_csv = (
        OUTPUT_DIR
        / "part235_decoder_point_records.csv"
    )

    prediction_csv = (
        OUTPUT_DIR
        / "part235_prediction_records.csv"
    )

    records_df.to_csv(
        records_csv,
        index=False,
    )

    prediction_df.to_csv(
        prediction_csv,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_header(
        "DECODER SPATIAL ACTIVATION SUMMARY"
    )

    summary_rows = []

    if not records_df.empty:

        grouped = records_df.groupby(
            [
                "feature",
                "radius",
            ],
            sort=False,
        )

        for (
            feature_name,
            radius,
        ), group in grouped:

            row = {
                "feature":
                    feature_name,

                "radius":
                    int(radius),

                "pairs":
                    int(len(group)),

                "lfnn_activation_mean":
                    float(
                        group[
                            "lfnn_activation_mean"
                        ].mean()
                    ),

                "lfnn_activation_energy":
                    float(
                        group[
                            "lfnn_activation_energy"
                        ].mean()
                    ),

                "lfnn_activation_abs_mean":
                    float(
                        group[
                            "lfnn_activation_abs_mean"
                        ].mean()
                    ),

                "lfnn_feature_norm":
                    float(
                        group[
                            "lfnn_feature_norm"
                        ].mean()
                    ),

                "rfnn_activation_mean":
                    float(
                        group[
                            "rfnn_activation_mean"
                        ].mean()
                    ),

                "rfnn_activation_energy":
                    float(
                        group[
                            "rfnn_activation_energy"
                        ].mean()
                    ),

                "rfnn_activation_abs_mean":
                    float(
                        group[
                            "rfnn_activation_abs_mean"
                        ].mean()
                    ),

                "rfnn_feature_norm":
                    float(
                        group[
                            "rfnn_feature_norm"
                        ].mean()
                    ),

                "difference_activation_mean":
                    float(
                        group[
                            "difference_activation_mean"
                        ].mean()
                    ),

                "difference_activation_energy":
                    float(
                        group[
                            "difference_activation_energy"
                        ].mean()
                    ),

                "difference_activation_abs_mean":
                    float(
                        group[
                            "difference_activation_abs_mean"
                        ].mean()
                    ),

                "difference_feature_norm":
                    float(
                        group[
                            "difference_feature_norm"
                        ].mean()
                    ),

                "rfnn_lfnn_energy_ratio":
                    float(
                        group[
                            "rfnn_lfnn_energy_ratio"
                        ].mean()
                    ),

                "rfnn_lfnn_feature_norm_ratio":
                    float(
                        group[
                            "rfnn_lfnn_feature_norm_ratio"
                        ].mean()
                    ),

                "point_cosine_similarity":
                    float(
                        group[
                            "point_cosine_similarity"
                        ].mean()
                    ),

                "point_l2_distance":
                    float(
                        group[
                            "point_l2_distance"
                        ].mean()
                    ),

                "point_mean_abs_difference":
                    float(
                        group[
                            "point_mean_abs_difference"
                        ].mean()
                    ),
            }

            summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        OUTPUT_DIR
        / "part235_decoder_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    if not summary_df.empty:

        print(
            summary_df.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

    else:

        print(
            "No decoder summary generated."
        )

    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    prediction_summary = {}

    if not prediction_df.empty:

        prediction_summary = {
            "pairs":
                int(
                    len(prediction_df)
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
    # SUPPRESSION SUMMARY
    # ========================================================

    suppression_summary = {}

    if not summary_df.empty:

        strongest_index = (
            summary_df[
                "difference_activation_energy"
            ]
            .idxmin()
        )

        strongest = (
            summary_df.loc[
                strongest_index
            ]
        )

        suppression_summary = {
            "strongest_negative_energy_difference": {
                "feature":
                    str(
                        strongest["feature"]
                    ),

                "radius":
                    int(
                        strongest["radius"]
                    ),

                "difference":
                    float(
                        strongest[
                            "difference_activation_energy"
                        ]
                    ),
            }
        }

    # ========================================================
    # JSON REPORT
    # ========================================================

    report = {
        "experiment":
            "Part 2.35 Decoder Spatial Activation Audit",

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

        "decoder_features":
            available_layers,

        "radii":
            RADII,

        "validation_manifest_rows":
            int(
                len(validation_df)
            ),

        "validation_cases_with_pairs":
            int(
                len(pairs_by_case)
            ),

        "paired_observations":
            int(
                len(pairs)
            ),

        "successful_cases":
            int(
                successful_cases
            ),

        "activation_records":
            int(
                len(records_df)
            ),

        "prediction_records":
            int(
                len(prediction_df)
            ),

        "prediction_summary":
            prediction_summary,

        "suppression_summary":
            suppression_summary,

        "outputs":
            {
                "activation_records":
                    str(
                        records_csv
                    ),

                "prediction_records":
                    str(
                        prediction_csv
                    ),

                "summary":
                    str(
                        summary_csv
                    ),
            },

        "interpretation_note":
            (
                "Negative RFNN-minus-LFNN activation "
                "differences indicate lower measured "
                "decoder activation around RFNN points "
                "relative to paired LFNN points. "
                "These measurements are descriptive and "
                "do not establish causality."
            ),
    }

    json_path = (
        OUTPUT_DIR
        / "part235_decoder_audit_summary.json"
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
    # FINAL
    # ========================================================

    print_header(
        "PART 2.35 COMPLETE"
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
        f"Activation records: "
        f"{len(records_df):,}"
    )

    print(
        f"Prediction records: "
        f"{len(prediction_df):,}"
    )

    print()

    print(
        "Expected activation records:"
    )

    print(
        "45 pairs × 4 decoder layers × 3 radii = 540"
    )

    print()

    print(
        "Saved:"
    )

    print(
        f"  {records_csv}"
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

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This experiment is analysis-only."
    )

    print(
        "No training/checkpoint/dashboard changes were made."
    )


# ============================================================
# 16. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()