"""
PART 2.36 - PAIRED LFNN/RFNN DECODER SPATIAL PROFILE AUDIT

Purpose
-------
Analysis-only experiment.

For the exact Part 2.20B validation cohort, identify paired LFNN/RFNN
points from the same study + series + spinal level.

For every LFNN -> RFNN pair, sample five positions along the straight
canonical-space line:

    0.00 = LFNN
    0.25 = 25% between
    0.50 = midpoint
    0.75 = 75% between
    1.00 = RFNN

At each position, measure decoder activation statistics from:

    decoder4.conv_block.norm2
    decoder3.conv_block.norm2
    decoder2.conv_block.norm2
    decoder1.conv_block.norm2

using feature-map neighborhoods with radii 2, 4 and 6.

This experiment:
    - does NOT train
    - does NOT modify any checkpoint
    - does NOT modify the dashboard
    - does NOT create voxel ground truth
    - uses the Part 2.27 best checkpoint only for inference
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part236_paired_spatial_profile_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
    / "part227_best_macro_disease.pth"
)

sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_FEATURES = [
    "decoder4.conv_block.norm2",
    "decoder3.conv_block.norm2",
    "decoder2.conv_block.norm2",
    "decoder1.conv_block.norm2",
]

RADII = [2, 4, 6]

# Five positions along LFNN -> RFNN.
PROFILE_POSITIONS = [
    ("LFNN", 0.00),
    ("P25", 0.25),
    ("MID", 0.50),
    ("P75", 0.75),
    ("RFNN", 1.00),
]

MODEL_SHAPE = (64, 96, 96)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# IMPORT PART 2.20B
# =============================================================================

def import_part220b():
    """
    Try the known Part 2.20B module names.

    The working project has used slightly different filenames during
    development, so we explicitly try the known variants.
    """

    candidates = [
        "segmentation_rsna_part220b_geometry_corrected_training",
        "segmentation_rsna_part220b_geometry_corrected_point_supervised_training",
    ]

    errors = []

    for name in candidates:
        try:
            module = import_module(name)
            print(f"[OK] Imported Part 2.20B module: {name}")
            return module
        except Exception as exc:
            errors.append((name, repr(exc)))

    print("\nCould not import Part 2.20B.")
    print("Tried:")

    for name, error in errors:
        print(f"  {name}: {error}")

    raise ImportError(
        "Part 2.20B module could not be imported. "
        "Check the actual filename in src."
    )


part220b = import_part220b()


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def clean_id(value):
    """
    Normalize IDs so integer/string representations match reliably.
    """
    if pd.isna(value):
        return ""

    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return str(value).strip()


def point_key(row):
    """
    Exact point identity based on the original Part 2.13 manifest.
    """
    return (
        clean_id(row["study_id"]),
        clean_id(row["series_id"]),
        clean_id(row["condition"]),
        clean_id(row["level"]),
        int(row["class_id"]),
        int(row["instance_number"]),
        int(row["native_z"]),
        int(row["native_y"]),
        int(row["native_x"]),
    )


# =============================================================================
# CHECKPOINT
# =============================================================================

def build_model():
    """
    Use the exact Part 2.20B model builder.
    """
    model = part220b.build_model()
    return model


def load_checkpoint(model):
    section("LOADING PART 2.27 BEST CHECKPOINT")

    print(f"Checkpoint:\n{CHECKPOINT}")
    print(f"Device: {DEVICE}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing:
        print("Missing:", missing)

    if unexpected:
        print("Unexpected:", unexpected)

    model.to(DEVICE)
    model.eval()

    print("[OK] Model loaded.")
    print(
        "Parameter count:",
        sum(p.numel() for p in model.parameters())
    )

    return model


# =============================================================================
# FEATURE HOOKS
# =============================================================================

class FeatureHookManager:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.features = {}
        self.handles = []

        named_modules = dict(model.named_modules())

        for name in feature_names:
            if name not in named_modules:
                raise KeyError(
                    f"Feature layer not found: {name}"
                )

            module = named_modules[name]

            print(
                f"[OK] {name} -> "
                f"{module.__class__.__name__}"
            )

            handle = module.register_forward_hook(
                self._make_hook(name)
            )

            self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]

            self.features[name] = output.detach()

        return hook

    def clear(self):
        self.features = {}

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles = []


# =============================================================================
# FEATURE LAYOUT
# =============================================================================

def normalize_feature_layout(feature):
    """
    Convert feature tensor into:

        C x D x H x W

    Expected decoder output:

        1 x C x D x H x W
    """

    if not torch.is_tensor(feature):
        feature = torch.as_tensor(feature)

    feature = feature.detach()

    if feature.ndim == 5:
        if feature.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1, got {tuple(feature.shape)}"
            )

        feature = feature[0]

    if feature.ndim != 4:
        raise ValueError(
            f"Expected CxDxHxW feature, got {tuple(feature.shape)}"
        )

    return feature


# =============================================================================
# MODEL-GRID -> FEATURE-GRID COORDINATE MAPPING
# =============================================================================

def model_to_feature_coord(point, feature_shape):
    """
    Map a model-grid coordinate:

        z,y,x in (64,96,96)

    to a feature-grid coordinate.

    Coordinates are continuous.
    """

    z, y, x = [float(v) for v in point]

    fd, fh, fw = feature_shape

    md, mh, mw = MODEL_SHAPE

    if md <= 1:
        fz = 0.0
    else:
        fz = z * (fd - 1) / (md - 1)

    if mh <= 1:
        fy = 0.0
    else:
        fy = y * (fh - 1) / (mh - 1)

    if mw <= 1:
        fx = 0.0
    else:
        fx = x * (fw - 1) / (mw - 1)

    return np.array([fz, fy, fx], dtype=np.float32)


# =============================================================================
# NEIGHBORHOOD STATISTICS
# =============================================================================

def feature_neighborhood_stats(
    feature_cdhw: torch.Tensor,
    feature_coord,
    radius: int,
):
    """
    Extract local decoder activation statistics.

    feature_cdhw:
        C x D x H x W

    feature_coord:
        continuous z,y,x coordinate in feature-map space.

    Radius is measured in feature-map voxels.
    """

    feature = feature_cdhw

    c, d, h, w = feature.shape

    z, y, x = [float(v) for v in feature_coord]

    cz = int(round(z))
    cy = int(round(y))
    cx = int(round(x))

    z0 = max(0, cz - radius)
    z1 = min(d, cz + radius + 1)

    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)

    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)

    patch = feature[:, z0:z1, y0:y1, x0:x1]

    if patch.numel() == 0:
        raise RuntimeError(
            "Empty feature neighborhood."
        )

    flat = patch.reshape(c, -1)

    # Per-channel means.
    channel_mean = flat.mean(dim=1)

    # Feature vector formed from spatially averaged channels.
    vector = channel_mean

    vector_norm = torch.linalg.vector_norm(vector)

    activation_mean = patch.mean()
    activation_abs_mean = patch.abs().mean()

    # Mean square activation.
    activation_energy = torch.sqrt(
        torch.mean(patch.float() ** 2)
    )

    activation_std = patch.std(unbiased=False)
    activation_min = patch.min()
    activation_max = patch.max()

    return {
        "activation_mean": float(
            activation_mean.item()
        ),
        "activation_abs_mean": float(
            activation_abs_mean.item()
        ),
        "activation_energy": float(
            activation_energy.item()
        ),
        "feature_norm": float(
            vector_norm.item()
        ),
        "activation_std": float(
            activation_std.item()
        ),
        "activation_min": float(
            activation_min.item()
        ),
        "activation_max": float(
            activation_max.item()
        ),
    }


# =============================================================================
# POINT PROBABILITY
# =============================================================================

def point_prediction(
    logits: torch.Tensor,
    point,
):
    """
    Obtain class probabilities at the nearest model-grid voxel.
    """

    if logits.ndim == 5:
        logits = logits[0]

    if logits.ndim != 4:
        raise ValueError(
            f"Expected CxDxHxW logits, got {tuple(logits.shape)}"
        )

    c, d, h, w = logits.shape

    z, y, x = [
        int(round(float(v)))
        for v in point
    ]

    z = max(0, min(d - 1, z))
    y = max(0, min(h - 1, y))
    x = max(0, min(w - 1, x))

    voxel_logits = logits[:, z, y, x]

    probs = torch.softmax(
        voxel_logits,
        dim=0,
    )

    pred_class = int(
        torch.argmax(probs).item()
    )

    return (
        probs.detach().cpu().numpy(),
        pred_class,
    )


# =============================================================================
# FEATURE POINT SIMILARITY
# =============================================================================

def vector_similarity(a, b):
    """
    Cosine similarity and L2 distance between two activation vectors.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom <= 1e-12:
        cosine = 0.0
    else:
        cosine = float(
            np.dot(a, b) / denom
        )

    l2 = float(
        np.linalg.norm(a - b)
    )

    abs_diff = float(
        np.mean(np.abs(a - b))
    )

    return cosine, l2, abs_diff


# =============================================================================
# VALIDATION MANIFEST RECOVERY
# =============================================================================

def recover_validation_points():
    """
    Reconstruct the exact Part 2.20B validation cohort and recover
    the complete original manifest rows.

    The Part 2.20B validation selector may return only identifiers,
    so we recover the original point rows from the full manifest.
    """

    section(
        "LOADING EXACT PART 2.20B VALIDATION COHORT"
    )

    manifest = part220b.load_manifest()

    print(
        "Full manifest rows:",
        len(manifest),
    )

    validation_series = (
        part220b.select_validation_series(manifest)
    )

    # Convert selector output into a DataFrame of IDs.
    if isinstance(validation_series, pd.DataFrame):
        val_df = validation_series.copy()
    else:
        val_df = pd.DataFrame(validation_series)

    print(
        "Validation series returned:",
        len(val_df),
    )

    # -------------------------------------------------------------------------
    # Identify study/series columns.
    # -------------------------------------------------------------------------

    def find_column(df, candidates):
        lower_map = {
            str(c).lower(): c
            for c in df.columns
        }

        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]

        return None

    study_col = find_column(
        val_df,
        ["study_id"],
    )

    series_col = find_column(
        val_df,
        ["series_id"],
    )

    if study_col is None or series_col is None:
        raise KeyError(
            "Could not find study_id/series_id in "
            "validation-series output."
        )

    validation_keys = set(
        (
            clean_id(row[study_col]),
            clean_id(row[series_col]),
        )
        for _, row in val_df.iterrows()
    )

    # -------------------------------------------------------------------------
    # Recover original manifest rows.
    # -------------------------------------------------------------------------

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
            "Full manifest does not contain "
            "study_id/series_id."
        )

    mask = manifest.apply(
        lambda row: (
            clean_id(row[manifest_study_col]),
            clean_id(row[manifest_series_col]),
        ) in validation_keys,
        axis=1,
    )

    point_rows = manifest.loc[
        mask
    ].copy().reset_index(drop=True)

    print(
        "Recovered validation point rows:",
        len(point_rows),
    )

    print(
        "Validation studies:",
        point_rows["study_id"].nunique(),
    )

    print(
        "Validation series:",
        point_rows["series_id"].nunique(),
    )

    # -------------------------------------------------------------------------
    # Restrict to LFNN/RFNN.
    # -------------------------------------------------------------------------

    disease_mask = point_rows[
        "class_name"
    ].astype(str).str.contains(
        "Neural Foraminal",
        case=False,
        na=False,
    )

    foraminal = point_rows.loc[
        disease_mask
    ].copy()

    print(
        "Validation LFNN/RFNN rows:",
        len(foraminal),
    )

    return manifest, point_rows, foraminal


# =============================================================================
# BUILD EXACT PAIRED OBSERVATIONS
# =============================================================================

def build_pairs(foraminal):
    """
    Pair LFNN and RFNN points using:

        study_id
        series_id
        level

    Each group should contain one LFNN and one RFNN observation.
    """

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

        lfnn = group[
            group["class_name"]
            .astype(str)
            .str.contains(
                "Left Neural",
                case=False,
                na=False,
            )
        ]

        rfnn = group[
            group["class_name"]
            .astype(str)
            .str.contains(
                "Right Neural",
                case=False,
                na=False,
            )
        ]

        if len(lfnn) == 0 or len(rfnn) == 0:
            continue

        # Usually exactly one each.
        # If multiple rows occur, pair by row order.
        n = min(len(lfnn), len(rfnn))

        lfnn = lfnn.sort_values(
            [
                "native_z",
                "native_y",
                "native_x",
            ]
        ).reset_index(drop=True)

        rfnn = rfnn.sort_values(
            [
                "native_z",
                "native_y",
                "native_x",
            ]
        ).reset_index(drop=True)

        for i in range(n):

            pairs.append(
                {
                    "pair_id": len(pairs),
                    "study_id": clean_id(study_id),
                    "series_id": clean_id(series_id),
                    "level": str(level),

                    "lfnn_row": lfnn.iloc[i].to_dict(),
                    "rfnn_row": rfnn.iloc[i].to_dict(),
                }
            )

    return pairs


# =============================================================================
# MAP ORIGINAL POINT ROWS TO TRANSFORMED POINTS
# =============================================================================

def get_transformed_point(
    original_row,
    transformed_points,
):
    """
    Part 2.20B load_case() returns transformed points as dictionaries.

    The transformed points preserve the original point ordering.

    We first attempt to match using class + level.
    """

    class_name = str(
        original_row["class_name"]
    )

    level = str(
        original_row["level"]
    )

    candidates = [
        p
        for p in transformed_points
        if str(p.get("class_name", "")) == class_name
        and str(p.get("level", "")) == level
    ]

    if len(candidates) == 1:
        return candidates[0]

    # If duplicates exist, use the nearest original native point
    # when the transformed point carries patient/native metadata.
    if len(candidates) > 1:
        # Deterministic fallback: first candidate.
        return candidates[0]

    raise KeyError(
        f"Could not map transformed point for "
        f"{class_name} / {level}"
    )


# =============================================================================
# MODEL INFERENCE
# =============================================================================

@torch.no_grad()
def run_model(model, image):
    """
    Forward pass while decoder hooks capture intermediate features.
    """

    if not torch.is_tensor(image):
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

    output = model(image)

    if isinstance(output, (tuple, list)):
        output = output[0]

    return output


# =============================================================================
# PROFILE COORDINATES
# =============================================================================

def interpolate_points(
    lfnn_point,
    rfnn_point,
):
    """
    Return five model-grid points along LFNN -> RFNN.
    """

    a = np.asarray(
        lfnn_point,
        dtype=np.float32,
    )

    b = np.asarray(
        rfnn_point,
        dtype=np.float32,
    )

    results = []

    for name, t in PROFILE_POSITIONS:

        p = (
            (1.0 - t) * a
            + t * b
        )

        results.append(
            {
                "position_name": name,
                "position_fraction": float(t),
                "z": float(p[0]),
                "y": float(p[1]),
                "x": float(p[2]),
            }
        )

    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():

    section(
        "PART 2.36 - PAIRED LFNN/RFNN "
        "DECODER SPATIAL PROFILE AUDIT"
    )

    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)
    print("Output:", OUTPUT_DIR)
    print("Model shape:", MODEL_SHAPE)
    print("Radii:", RADII)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = build_model()
    model = load_checkpoint(model)

    # -------------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------------

    section(
        "REGISTERING DECODER FEATURE HOOKS"
    )

    hook_manager = FeatureHookManager(
        model,
        TARGET_FEATURES,
    )

    # -------------------------------------------------------------------------
    # Validation cohort
    # -------------------------------------------------------------------------

    manifest, point_rows, foraminal = (
        recover_validation_points()
    )

    pairs = build_pairs(foraminal)

    section("PAIRING SUMMARY")

    print(
        "Candidate paired series:",
        len(
            set(
                (
                    p["study_id"],
                    p["series_id"],
                )
                for p in pairs
            )
        ),
    )

    print(
        "Total LFNN/RFNN pairs:",
        len(pairs),
    )

    # -------------------------------------------------------------------------
    # Containers
    # -------------------------------------------------------------------------

    profile_records = []
    prediction_records = []

    successful_cases = set()

    # -------------------------------------------------------------------------
    # Process unique series
    # -------------------------------------------------------------------------

    unique_series = sorted(
        set(
            (
                p["study_id"],
                p["series_id"],
            )
            for p in pairs
        )
    )

    for case_index, (
        study_id,
        series_id,
    ) in enumerate(
        unique_series,
        start=1,
    ):

        print(
            f"\n[{case_index}/{len(unique_series)}] "
            f"Study={study_id} "
            f"Series={series_id}"
        )

        case_rows = point_rows[
            (
                point_rows["study_id"].map(clean_id)
                == study_id
            )
            & (
                point_rows["series_id"].map(clean_id)
                == series_id
            )
        ].copy()

        try:

            # Exact Part 2.20B preprocessing.
            image, transformed_points, geometry = (
                part220b.load_case(
                    study_id,
                    series_id,
                    case_rows,
                )
            )

            print(
                "  Image shape:",
                tuple(image.shape),
            )

            print(
                "  Original point rows:",
                len(case_rows),
            )

            print(
                "  Transformed points:",
                len(transformed_points),
            )

            # -----------------------------------------------------------------
            # Forward pass.
            # -----------------------------------------------------------------

            hook_manager.clear()

            logits = run_model(
                model,
                image,
            )

            captured = hook_manager.features

            if len(captured) != len(
                TARGET_FEATURES
            ):
                raise RuntimeError(
                    "Not all decoder features were captured."
                )

            print(
                "  Captured decoder features:",
                len(captured),
            )

            # -----------------------------------------------------------------
            # Build point mappings.
            # -----------------------------------------------------------------

            transformed_lookup = {}

            for p in transformed_points:

                key = (
                    str(p.get("class_name", "")),
                    str(p.get("level", "")),
                )

                transformed_lookup.setdefault(
                    key,
                    [],
                ).append(p)

            # -----------------------------------------------------------------
            # Process pairs belonging to this case.
            # -----------------------------------------------------------------

            case_pairs = [
                p
                for p in pairs
                if p["study_id"] == study_id
                and p["series_id"] == series_id
            ]

            for pair in case_pairs:

                lrow = pair["lfnn_row"]
                rrow = pair["rfnn_row"]

                # -------------------------------------------------------------
                # Exact transformed points.
                # -------------------------------------------------------------

                l_candidates = transformed_lookup.get(
                    (
                        str(lrow["class_name"]),
                        str(lrow["level"]),
                    ),
                    [],
                )

                r_candidates = transformed_lookup.get(
                    (
                        str(rrow["class_name"]),
                        str(rrow["level"]),
                    ),
                    [],
                )

                if not l_candidates:
                    raise KeyError(
                        "LFNN transformed point missing."
                    )

                if not r_candidates:
                    raise KeyError(
                        "RFNN transformed point missing."
                    )

                # Since each class/level occurs once in the paired
                # observations used by Part 2.34/2.35, use the first.
                lpoint = l_candidates[0]
                rpoint = r_candidates[0]

                l_model = np.array(
                    [
                        float(lpoint["z"]),
                        float(lpoint["y"]),
                        float(lpoint["x"]),
                    ],
                    dtype=np.float32,
                )

                r_model = np.array(
                    [
                        float(rpoint["z"]),
                        float(rpoint["y"]),
                        float(rpoint["x"]),
                    ],
                    dtype=np.float32,
                )

                # -------------------------------------------------------------
                # LFNN -> RFNN profile.
                # -------------------------------------------------------------

                profile = interpolate_points(
                    l_model,
                    r_model,
                )

                pair_distance = float(
                    np.linalg.norm(
                        r_model - l_model
                    )
                )

                # -------------------------------------------------------------
                # Prediction at exact LFNN and RFNN points.
                # -------------------------------------------------------------

                l_probs, l_pred = point_prediction(
                    logits,
                    l_model,
                )

                r_probs, r_pred = point_prediction(
                    logits,
                    r_model,
                )

                prediction_records.append(
                    {
                        "pair_id": pair["pair_id"],
                        "study_id": study_id,
                        "series_id": series_id,
                        "level": str(pair["level"]),

                        "lfnn_model_z": float(l_model[0]),
                        "lfnn_model_y": float(l_model[1]),
                        "lfnn_model_x": float(l_model[2]),

                        "rfnn_model_z": float(r_model[0]),
                        "rfnn_model_y": float(r_model[1]),
                        "rfnn_model_x": float(r_model[2]),

                        "lfnn_rfnn_model_distance":
                            pair_distance,

                        "lfnn_true_probability":
                            float(
                                l_probs[
                                    int(lrow["class_id"])
                                ]
                            ),

                        "rfnn_true_probability":
                            float(
                                r_probs[
                                    int(rrow["class_id"])
                                ]
                            ),

                        "lfnn_pred_class":
                            int(l_pred),

                        "rfnn_pred_class":
                            int(r_pred),
                    }
                )

                # -------------------------------------------------------------
                # Spatial profile.
                # -------------------------------------------------------------

                for feature_name in TARGET_FEATURES:

                    feature = normalize_feature_layout(
                        captured[feature_name]
                    )

                    feature_shape = tuple(
                        feature.shape[1:]
                    )

                    for profile_position in profile:

                        model_point = np.array(
                            [
                                profile_position["z"],
                                profile_position["y"],
                                profile_position["x"],
                            ],
                            dtype=np.float32,
                        )

                        feature_coord = (
                            model_to_feature_coord(
                                model_point,
                                feature_shape,
                            )
                        )

                        # Exact decoder activation vector at the
                        # nearest feature-map voxel.
                        fz = int(
                            round(
                                float(
                                    feature_coord[0]
                                )
                            )
                        )

                        fy = int(
                            round(
                                float(
                                    feature_coord[1]
                                )
                            )
                        )

                        fx = int(
                            round(
                                float(
                                    feature_coord[2]
                                )
                            )
                        )

                        fz = max(
                            0,
                            min(
                                feature_shape[0] - 1,
                                fz,
                            ),
                        )

                        fy = max(
                            0,
                            min(
                                feature_shape[1] - 1,
                                fy,
                            ),
                        )

                        fx = max(
                            0,
                            min(
                                feature_shape[2] - 1,
                                fx,
                            ),
                        )

                        point_vector = (
                            feature[
                                :,
                                fz,
                                fy,
                                fx,
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(
                                np.float64
                            )
                        )

                        # -----------------------------------------------------
                        # Neighborhoods.
                        # -----------------------------------------------------

                        for radius in RADII:

                            stats = (
                                feature_neighborhood_stats(
                                    feature,
                                    feature_coord,
                                    radius,
                                )
                            )

                            profile_records.append(
                                {
                                    "pair_id":
                                        pair["pair_id"],

                                    "study_id":
                                        study_id,

                                    "series_id":
                                        series_id,

                                    "level":
                                        str(
                                            pair["level"]
                                        ),

                                    "feature":
                                        feature_name,

                                    "radius":
                                        int(radius),

                                    "position_name":
                                        profile_position[
                                            "position_name"
                                        ],

                                    "position_fraction":
                                        float(
                                            profile_position[
                                                "position_fraction"
                                            ]
                                        ),

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

                                    "feature_z":
                                        float(
                                            feature_coord[0]
                                        ),

                                    "feature_y":
                                        float(
                                            feature_coord[1]
                                        ),

                                    "feature_x":
                                        float(
                                            feature_coord[2]
                                        ),

                                    "feature_depth":
                                        int(
                                            feature_shape[0]
                                        ),

                                    "feature_height":
                                        int(
                                            feature_shape[1]
                                        ),

                                    "feature_width":
                                        int(
                                            feature_shape[2]
                                        ),

                                    "activation_mean":
                                        stats[
                                            "activation_mean"
                                        ],

                                    "activation_abs_mean":
                                        stats[
                                            "activation_abs_mean"
                                        ],

                                    "activation_energy":
                                        stats[
                                            "activation_energy"
                                        ],

                                    "feature_norm":
                                        stats[
                                            "feature_norm"
                                        ],

                                    "activation_std":
                                        stats[
                                            "activation_std"
                                        ],

                                    "activation_min":
                                        stats[
                                            "activation_min"
                                        ],

                                    "activation_max":
                                        stats[
                                            "activation_max"
                                        ],
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
    # DATAFRAMES
    # =============================================================================

    profile_df = pd.DataFrame(
        profile_records
    )

    prediction_df = pd.DataFrame(
        prediction_records
    )

    if profile_df.empty:
        raise RuntimeError(
            "No profile records were generated."
        )

    if prediction_df.empty:
        raise RuntimeError(
            "No prediction records were generated."
        )

    # =============================================================================
    # NORMALIZED LFNN -> RFNN PROFILE
    # =============================================================================

    section(
        "CALCULATING NORMALIZED SPATIAL PROFILES"
    )

    # Create one reference table for LFNN and RFNN endpoint activation.
    endpoint = profile_df[
        profile_df["position_name"].isin(
            ["LFNN", "RFNN"]
        )
    ].copy()

    endpoint_key = [
        "pair_id",
        "feature",
        "radius",
    ]

    endpoint_energy = endpoint.pivot_table(
        index=endpoint_key,
        columns="position_name",
        values="activation_energy",
        aggfunc="mean",
    ).reset_index()

    endpoint_energy.columns.name = None

    if "LFNN" not in endpoint_energy.columns:
        endpoint_energy["LFNN"] = np.nan

    if "RFNN" not in endpoint_energy.columns:
        endpoint_energy["RFNN"] = np.nan

    endpoint_energy[
        "energy_difference_rfnn_minus_lfnn"
    ] = (
        endpoint_energy["RFNN"]
        - endpoint_energy["LFNN"]
    )

    endpoint_energy[
        "energy_ratio_rfnn_over_lfnn"
    ] = (
        endpoint_energy["RFNN"]
        / endpoint_energy["LFNN"].replace(
            0,
            np.nan,
        )
    )

    endpoint_energy[
        "energy_slope_per_profile_fraction"
    ] = (
        endpoint_energy[
            "energy_difference_rfnn_minus_lfnn"
        ]
        / 1.0
    )

    profile_df = profile_df.merge(
        endpoint_energy[
            endpoint_key
            + [
                "LFNN",
                "RFNN",
                "energy_difference_rfnn_minus_lfnn",
                "energy_ratio_rfnn_over_lfnn",
                "energy_slope_per_profile_fraction",
            ]
        ].rename(
            columns={
                "LFNN":
                    "endpoint_lfnn_energy",
                "RFNN":
                    "endpoint_rfnn_energy",
            }
        ),
        on=endpoint_key,
        how="left",
    )

    # Normalize each spatial profile relative to LFNN.
    profile_df[
        "energy_normalized_to_lfnn"
    ] = (
        profile_df[
            "activation_energy"
        ]
        / profile_df[
            "endpoint_lfnn_energy"
        ].replace(
            0,
            np.nan,
        )
    )

    profile_df[
        "feature_norm_normalized_to_lfnn"
    ] = (
        profile_df[
            "feature_norm"
        ]
        / profile_df[
            "feature_norm"
        ]
        .where(
            profile_df[
                "position_name"
            ] == "LFNN"
        )
        .groupby(
            [
                profile_df["pair_id"],
                profile_df["feature"],
                profile_df["radius"],
            ]
        )
        .transform("first")
    )

    # =============================================================================
    # SUMMARY
    # =============================================================================

    section(
        "SPATIAL PROFILE SUMMARY"
    )

    summary_records = []

    for (
        feature,
        radius,
    ), group in profile_df.groupby(
        [
            "feature",
            "radius",
        ]
    ):

        # -------------------------------------------------------------
        # Mean profile by position.
        # -------------------------------------------------------------

        position_summary = (
            group.groupby(
                [
                    "position_name",
                    "position_fraction",
                ],
                as_index=False,
            )
            .agg(
                mean_energy=(
                    "activation_energy",
                    "mean",
                ),
                mean_feature_norm=(
                    "feature_norm",
                    "mean",
                ),
                mean_abs_activation=(
                    "activation_abs_mean",
                    "mean",
                ),
                std_energy=(
                    "activation_energy",
                    "std",
                ),
            )
        )

        # -------------------------------------------------------------
        # Endpoint values.
        # -------------------------------------------------------------

        lfnn_rows = group[
            group["position_name"] == "LFNN"
        ]

        rfnn_rows = group[
            group["position_name"] == "RFNN"
        ]

        if len(lfnn_rows) == 0:
            continue

        if len(rfnn_rows) == 0:
            continue

        l_energy = float(
            lfnn_rows[
                "activation_energy"
            ].mean()
        )

        r_energy = float(
            rfnn_rows[
                "activation_energy"
            ].mean()
        )

        l_norm = float(
            lfnn_rows[
                "feature_norm"
            ].mean()
        )

        r_norm = float(
            rfnn_rows[
                "feature_norm"
            ].mean()
        )

        # -------------------------------------------------------------
        # Midpoint.
        # -------------------------------------------------------------

        midpoint = group[
            group["position_name"] == "MID"
        ]

        mid_energy = (
            float(
                midpoint[
                    "activation_energy"
                ].mean()
            )
            if len(midpoint)
            else np.nan
        )

        # -------------------------------------------------------------
        # Linear slope from LFNN to RFNN.
        # -------------------------------------------------------------

        slopes = []

        for pair_id, pair_group in group.groupby(
            "pair_id"
        ):

            ordered = (
                pair_group.sort_values(
                    "position_fraction"
                )
            )

            x = ordered[
                "position_fraction"
            ].to_numpy(
                dtype=float
            )

            y = ordered[
                "activation_energy"
            ].to_numpy(
                dtype=float
            )

            valid = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            if valid.sum() >= 2:
                slope = np.polyfit(
                    x[valid],
                    y[valid],
                    1,
                )[0]

                slopes.append(
                    float(slope)
                )

        # -------------------------------------------------------------
        # Monotonicity.
        # -------------------------------------------------------------

        monotonic_count = 0
        total_profile_count = 0

        for pair_id, pair_group in group.groupby(
            "pair_id"
        ):

            ordered = (
                pair_group.sort_values(
                    "position_fraction"
                )
            )

            values = ordered[
                "activation_energy"
            ].to_numpy(
                dtype=float
            )

            if len(values) >= 2:

                total_profile_count += 1

                # Strictly decreasing is unlikely with noisy features,
                # so we use non-increasing with a small tolerance.
                diffs = np.diff(values)

                if np.all(
                    diffs <= 1e-8
                ):
                    monotonic_count += 1

        summary_records.append(
            {
                "feature":
                    feature,

                "radius":
                    int(radius),

                "pairs":
                    int(
                        group["pair_id"]
                        .nunique()
                    ),

                "lfnn_mean_energy":
                    l_energy,

                "midpoint_mean_energy":
                    mid_energy,

                "rfnn_mean_energy":
                    r_energy,

                "rfnn_minus_lfnn_energy":
                    r_energy - l_energy,

                "rfnn_over_lfnn_energy":
                    (
                        r_energy / l_energy
                        if abs(l_energy) > 1e-12
                        else np.nan
                    ),

                "lfnn_mean_feature_norm":
                    l_norm,

                "rfnn_mean_feature_norm":
                    r_norm,

                "rfnn_minus_lfnn_feature_norm":
                    r_norm - l_norm,

                "mean_profile_slope":
                    (
                        float(
                            np.mean(slopes)
                        )
                        if slopes
                        else np.nan
                    ),

                "median_profile_slope":
                    (
                        float(
                            np.median(slopes)
                        )
                        if slopes
                        else np.nan
                    ),

                "std_profile_slope":
                    (
                        float(
                            np.std(slopes)
                        )
                        if slopes
                        else np.nan
                    ),

                "monotonic_nonincreasing_profiles":
                    monotonic_count,

                "total_profiles":
                    total_profile_count,

                "monotonic_nonincreasing_fraction":
                    (
                        monotonic_count
                        / total_profile_count
                        if total_profile_count
                        else np.nan
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_records
    )

    # =============================================================================
    # MEAN POSITION PROFILE
    # =============================================================================

    position_profile_df = (
        profile_df.groupby(
            [
                "feature",
                "radius",
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
            mean_activation_energy=(
                "activation_energy",
                "mean",
            ),
            median_activation_energy=(
                "activation_energy",
                "median",
            ),
            mean_feature_norm=(
                "feature_norm",
                "mean",
            ),
            mean_abs_activation=(
                "activation_abs_mean",
                "mean",
            ),
            mean_activation=(
                "activation_mean",
                "mean",
            ),
            std_activation_energy=(
                "activation_energy",
                "std",
            ),
        )
    )

    # =============================================================================
    # SAVE
    # =============================================================================

    profile_path = (
        OUTPUT_DIR
        / "part236_spatial_profile_records.csv"
    )

    prediction_path = (
        OUTPUT_DIR
        / "part236_prediction_records.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part236_spatial_profile_summary.csv"
    )

    mean_profile_path = (
        OUTPUT_DIR
        / "part236_mean_position_profiles.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part236_spatial_profile_audit_summary.json"
    )

    profile_df.to_csv(
        profile_path,
        index=False,
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    position_profile_df.to_csv(
        mean_profile_path,
        index=False,
    )

    # =============================================================================
    # JSON SUMMARY
    # =============================================================================

    json_summary = {
        "part": "2.36",
        "title": (
            "Paired LFNN/RFNN Decoder Spatial Profile Audit"
        ),
        "purpose": (
            "Analysis-only spatial profile audit along "
            "the LFNN-to-RFNN canonical-space direction."
        ),
        "checkpoint": str(CHECKPOINT),
        "device": str(DEVICE),
        "model_shape": list(MODEL_SHAPE),
        "radii": RADII,
        "profile_positions": [
            {
                "name": name,
                "fraction": fraction,
            }
            for name, fraction
            in PROFILE_POSITIONS
        ],
        "target_features": TARGET_FEATURES,
        "successful_cases": len(
            successful_cases
        ),
        "paired_observations": len(pairs),
        "profile_records": len(profile_df),
        "prediction_records": len(
            prediction_df
        ),
        "expected_profile_records": (
            len(pairs)
            * len(TARGET_FEATURES)
            * len(RADII)
            * len(PROFILE_POSITIONS)
        ),
        "files": {
            "profile_records":
                str(profile_path),
            "prediction_records":
                str(prediction_path),
            "summary":
                str(summary_path),
            "mean_position_profiles":
                str(mean_profile_path),
        },
        "training_performed": False,
        "checkpoint_modified": False,
        "dashboard_modified": False,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_summary,
            f,
            indent=2,
        )

    # =============================================================================
    # FINAL REPORT
    # =============================================================================

    section(
        "PART 2.36 COMPLETE"
    )

    print(
        "Successful cases:",
        len(successful_cases),
    )

    print(
        "Paired observations:",
        len(pairs),
    )

    print(
        "Profile records:",
        len(profile_df),
    )

    print(
        "Prediction records:",
        len(prediction_df),
    )

    print(
        "Expected profile records:",
        json_summary[
            "expected_profile_records"
        ],
    )

    print("\nSaved:")

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
        f"  {mean_profile_path}"
    )

    print(
        f"  {json_path}"
    )

    print("\nIMPORTANT:")
    print(
        "This experiment is analysis-only."
    )
    print(
        "No training/checkpoint/dashboard changes were made."
    )

    hook_manager.remove()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()