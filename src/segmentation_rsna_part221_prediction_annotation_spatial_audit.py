"""
PART 2.21 — PREDICTION–ANNOTATION SPATIAL AUDIT

Analysis-only audit of the Part 2.20B Epoch-5 checkpoint.

Important:
- No training.
- No checkpoint modification.
- No dashboard modification.
- No voxel-wise ground-truth masks are fabricated.
- RSNA coordinates remain point/localization annotations.

This file deliberately contains the exact geometry helpers used by Part 2.20B
instead of importing non-existent helper names from that training script.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label, map_coordinates


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DATASET_ROOT = (
    ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification"
)
TRAIN_IMAGES_DIR = DATASET_ROOT / "train_images"

MANIFEST_PATH = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part213_point_supervision_manifest"
    / "part213_point_manifest.csv"
)

CHECKPOINT_PATH = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part220b_geometry_corrected_training"
    / "checkpoints" / "part220b_epoch_05.pth"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part221_prediction_annotation_spatial_audit"
)
REPORT_DIR = OUTPUT_DIR / "reports"


# ============================================================================
# MODEL CONTRACT
# ============================================================================

SEED = 42
MODEL_SHAPE = (64, 96, 96)
NUM_CLASSES = 6
VALIDATION_CASES = 25

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

FORAMINAL_CLASSES = {2, 3}
THRESHOLDS = (0.30, 0.50, 0.70)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# DICOM LOADER
# ============================================================================

from segmentation_rsna_part11_controlled_pilot_training_corrected import (
    read_dicom_series_robust,
    resolve_series_dir,
)


# ============================================================================
# EXACT PART 2.20B GEOMETRY IMPLEMENTATION
# ============================================================================

def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero-length vector.")
    return vector / norm


def get_iop(ds):
    value = getattr(ds, "ImageOrientationPatient", None)
    if value is None or len(value) != 6:
        raise ValueError("Missing ImageOrientationPatient.")

    row_direction = normalize_vector(value[:3])
    column_direction = normalize_vector(value[3:])
    normal_direction = normalize_vector(
        np.cross(row_direction, column_direction)
    )
    return row_direction, column_direction, normal_direction


def get_pixel_spacing(ds):
    value = getattr(ds, "PixelSpacing", None)
    if value is None or len(value) != 2:
        raise ValueError("Missing PixelSpacing.")
    return float(value[0]), float(value[1])


def get_ipp(ds):
    value = getattr(ds, "ImagePositionPatient", None)
    if value is None or len(value) != 3:
        raise ValueError("Missing ImagePositionPatient.")
    return np.asarray(value, dtype=np.float64)


def build_geometry(records, image_shape):
    if len(records) == 0:
        raise ValueError("No DICOM records.")

    first = records[0]
    row_direction, column_direction, normal_direction = get_iop(first)
    row_spacing, column_spacing = get_pixel_spacing(first)

    slice_positions = np.asarray(
        [
            float(np.dot(get_ipp(ds), normal_direction))
            for ds in records
        ],
        dtype=np.float64,
    )

    sort_order = np.argsort(slice_positions)
    sorted_positions = slice_positions[sort_order]

    if len(sorted_positions) > 1:
        diffs = np.diff(sorted_positions)
        diffs = diffs[np.abs(diffs) > 1e-6]
        slice_spacing = (
            float(np.median(diffs)) if len(diffs) else 1.0
        )
    else:
        slice_spacing = 1.0

    depth, height, width = image_shape

    corner_points = []
    for ds in records:
        ipp = get_ipp(ds)
        for row in [0.0, float(height - 1)]:
            for col in [0.0, float(width - 1)]:
                point = (
                    ipp
                    + row * row_spacing * row_direction
                    + col * column_spacing * column_direction
                )
                corner_points.append(point)

    corner_points = np.asarray(corner_points, dtype=np.float64)

    patient_min = corner_points.min(axis=0)
    patient_max = corner_points.max(axis=0)

    return {
        "row_direction": row_direction,
        "column_direction": column_direction,
        "normal_direction": normal_direction,
        "row_spacing": row_spacing,
        "column_spacing": column_spacing,
        "slice_spacing": slice_spacing,
        "slice_positions": sorted_positions,
        "sort_order": sort_order,
        "origin": get_ipp(records[0]),
        "patient_min": patient_min,
        "patient_max": patient_max,
        "image_shape": image_shape,
    }


def patient_to_native(patient_points, geometry):
    patient_points = np.asarray(patient_points, dtype=np.float64)

    row_direction = geometry["row_direction"]
    column_direction = geometry["column_direction"]
    normal_direction = geometry["normal_direction"]
    row_spacing = geometry["row_spacing"]
    column_spacing = geometry["column_spacing"]
    slice_positions = geometry["slice_positions"]
    origin = geometry["origin"]

    delta = patient_points - origin

    x_mm = np.dot(delta, column_direction)
    y_mm = np.dot(delta, row_direction)
    z_physical = np.dot(delta, normal_direction)

    x = x_mm / max(column_spacing, 1e-8)
    y = y_mm / max(row_spacing, 1e-8)

    z = np.interp(
        z_physical + float(np.dot(origin, normal_direction)),
        slice_positions,
        np.arange(len(slice_positions), dtype=np.float64),
    )

    return z, y, x


def build_canonical_grid(geometry):
    depth, height, width = MODEL_SHAPE

    patient_min = geometry["patient_min"]
    patient_max = geometry["patient_max"]

    x_values = np.linspace(
        patient_min[0], patient_max[0], width, dtype=np.float64
    )
    y_values = np.linspace(
        patient_min[1], patient_max[1], height, dtype=np.float64
    )
    z_values = np.linspace(
        patient_min[2], patient_max[2], depth, dtype=np.float64
    )

    zz, yy, xx = np.meshgrid(
        z_values, y_values, x_values, indexing="ij"
    )
    return xx, yy, zz


def resample_to_canonical(image, records, geometry):
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got {image.shape}")

    xx, yy, zz = build_canonical_grid(geometry)

    patient_points = np.stack(
        [xx.ravel(), yy.ravel(), zz.ravel()], axis=1
    )

    z, y, x = patient_to_native(patient_points, geometry)

    depth, height, width = image.shape

    z = np.clip(z, 0, depth - 1)
    y = np.clip(y, 0, height - 1)
    x = np.clip(x, 0, width - 1)

    coordinates = np.vstack([z, y, x])

    canonical = map_coordinates(
        image,
        coordinates,
        order=1,
        mode="nearest",
    )

    return canonical.reshape(MODEL_SHAPE).astype(np.float32)


def normalize_volume(volume):
    volume = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(volume)

    if not finite.any():
        return np.zeros_like(volume, dtype=np.float32)

    valid = volume[finite]
    p1, p99 = np.percentile(valid, [1, 99])

    if p99 <= p1:
        mean = valid.mean()
        std = valid.std()
        if std < 1e-6:
            return np.zeros_like(volume, dtype=np.float32)
        volume = (volume - mean) / std
    else:
        volume = (volume - p1) / (p99 - p1)
        volume = np.clip(volume, 0.0, 1.0)

    return volume.astype(np.float32)


def native_point_to_patient(records, native_z, native_y, native_x):
    z_index = int(
        np.clip(round(native_z), 0, len(records) - 1)
    )
    ds = records[z_index]

    return (
        get_ipp(ds)
        + native_y * get_pixel_spacing(ds)[0] * get_iop(ds)[0]
        + native_x * get_pixel_spacing(ds)[1] * get_iop(ds)[1]
    )


def patient_point_to_canonical(patient_point, geometry):
    patient_min = geometry["patient_min"]
    patient_max = geometry["patient_max"]

    extent = patient_max - patient_min
    extent = np.where(np.abs(extent) < 1e-8, 1.0, extent)

    normalized = (patient_point - patient_min) / extent
    normalized = np.clip(normalized, 0.0, 1.0)

    d, h, w = MODEL_SHAPE

    canonical_x = normalized[0] * (w - 1)
    canonical_y = normalized[1] * (h - 1)
    canonical_z = normalized[2] * (d - 1)

    return float(canonical_z), float(canonical_y), float(canonical_x)


def load_case(study_id, series_id, point_df):
    series_dir = TRAIN_IMAGES_DIR / str(study_id) / str(series_id)

    if not series_dir.exists():
        row = pd.Series({
            "study_id": str(study_id),
            "series_id": str(series_id),
        })
        series_dir = resolve_series_dir(row)

    image, records, info = read_dicom_series_robust(series_dir)
    image = np.asarray(image, dtype=np.float32)

    geometry = build_geometry(records, image.shape)
    canonical = resample_to_canonical(
        image, records, geometry
    )
    canonical = normalize_volume(canonical)

    points = []

    for _, point in point_df.iterrows():
        patient_point = native_point_to_patient(
            records,
            float(point["native_z"]),
            float(point["native_y"]),
            float(point["native_x"]),
        )

        cz, cy, cx = patient_point_to_canonical(
            patient_point, geometry
        )

        points.append({
            "class_id": int(point["class_id"]),
            "class_name": str(point["class_name"]),
            "level": str(point["level"]),
            "patient_x": float(patient_point[0]),
            "patient_y": float(patient_point[1]),
            "patient_z": float(patient_point[2]),
            "z": cz,
            "y": cy,
            "x": cx,
        })

    return canonical, points, geometry


# ============================================================================
# VALIDATION SELECTION — SAME DESIGN AS PART 2.20B
# ============================================================================

def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    df = pd.read_csv(
        MANIFEST_PATH,
        dtype={"study_id": str, "series_id": str},
    )

    required = [
        "study_id", "series_id", "condition",
        "class_id", "class_name", "level",
        "native_z", "native_y", "native_x",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Manifest missing columns: {missing}"
        )

    return df


def select_validation_series(manifest):
    """
    Reproduce the Part 2.20B selection logic exactly:
    - shuffle unique studies with seed 42
    - use the first 395 studies as validation-study pool
    - take the first series per study
    - retain the first 25 selected series
    """
    rng = np.random.default_rng(SEED)

    studies = np.array(
        sorted(manifest["study_id"].astype(str).unique())
    )

    shuffled = studies.copy()
    rng.shuffle(shuffled)

    validation_studies = set(
        shuffled[:min(395, len(shuffled))]
    )

    validation_manifest = manifest[
        manifest["study_id"].astype(str).isin(
            validation_studies
        )
    ]

    series_table = (
        validation_manifest[
            ["study_id", "series_id"]
        ]
        .drop_duplicates()
        .sort_values(["study_id", "series_id"])
    )

    selected = []

    for study_id, group in series_table.groupby(
        "study_id", sort=True
    ):
        selected.append(group.iloc[0])

    selected = pd.DataFrame(selected)

    if len(selected) > VALIDATION_CASES:
        selected = selected.iloc[:VALIDATION_CASES].copy()

    return selected.reset_index(drop=True)


# ============================================================================
# MODEL
# ============================================================================

def build_model():
    from monai.networks.nets import SwinUNETR

    model = SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=12,
        spatial_dims=3,
        use_checkpoint=False,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
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

    clean_state = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state[clean_key] = value

    result = model.load_state_dict(
        clean_state, strict=False
    )

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Part 2.20B checkpoint mismatch: "
            f"missing={len(result.missing_keys)}, "
            f"unexpected={len(result.unexpected_keys)}"
        )

    model.eval()
    return model


# ============================================================================
# SPATIAL PREDICTION HELPERS
# ============================================================================

def connected_regions(probability, class_id, threshold):
    mask = probability[class_id] >= threshold

    if not mask.any():
        return []

    labels, count = cc_label(
        mask,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )

    result = []

    for region_id in range(1, count + 1):
        coords = np.argwhere(labels == region_id)
        if len(coords) == 0:
            continue

        centroid = coords.mean(axis=0)

        result.append({
            "voxels": int(len(coords)),
            "centroid_z": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_x": float(centroid[2]),
        })

    return result


def nearest_region(point, regions):
    if not regions:
        return math.nan, None

    best_distance = math.inf
    best_region = None

    for region in regions:
        centroid = np.array([
            region["centroid_z"],
            region["centroid_y"],
            region["centroid_x"],
        ], dtype=np.float32)

        distance = float(
            np.linalg.norm(point - centroid)
        )

        if distance < best_distance:
            best_distance = distance
            best_region = region

    return best_distance, best_region


@torch.no_grad()
def predict_case(model, image):
    tensor = torch.from_numpy(image).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(
        DEVICE, non_blocking=True
    )

    logits = model(tensor)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    probabilities = torch.softmax(logits, dim=1)

    return probabilities[0].detach().cpu().numpy()


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PART 2.21 — PREDICTION–ANNOTATION SPATIAL AUDIT")
    print("=" * 78)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    manifest = load_manifest()

    print(f"Manifest rows: {len(manifest)}")
    print(
        f"Annotated series: "
        f"{manifest['series_id'].nunique()}"
    )

    selected = select_validation_series(manifest)

    print(
        f"Validation series selected: {len(selected)}"
    )
    print(
        f"Validation studies selected: "
        f"{selected['study_id'].nunique()}"
    )

    model = build_model()
    print("Part 2.20B Epoch 5 checkpoint loaded successfully.")

    rows = []
    probability_rows = []
    failed = []

    for index, selected_row in selected.iterrows():
        study_id = str(selected_row["study_id"])
        series_id = str(selected_row["series_id"])

        print(
            f"\n[{index + 1:02d}/{len(selected):02d}] "
            f"{study_id}/{series_id}"
        )

        try:
            point_df = manifest[
                (manifest["study_id"].astype(str) == study_id)
                & (manifest["series_id"].astype(str) == series_id)
            ].copy()

            image, points, geometry = load_case(
                study_id,
                series_id,
                point_df,
            )

            probability = predict_case(
                model, image
            )

            # Build connected prediction regions for the two
            # important foraminal classes at each threshold.
            regions = {
                class_id: {
                    threshold: connected_regions(
                        probability,
                        class_id,
                        threshold,
                    )
                    for threshold in THRESHOLDS
                }
                for class_id in range(1, NUM_CLASSES)
            }

            foreground_ratio = float(
                (probability[1:].argmax(axis=0) >= 0)
                .mean()
            )

            # Correct foreground ratio calculation:
            predicted_labels = probability.argmax(axis=0)
            foreground_ratio = float(
                (predicted_labels != 0).mean()
            )

            for point in points:
                z = int(np.clip(
                    round(point["z"]), 0, MODEL_SHAPE[0] - 1
                ))
                y = int(np.clip(
                    round(point["y"]), 0, MODEL_SHAPE[1] - 1
                ))
                x = int(np.clip(
                    round(point["x"]), 0, MODEL_SHAPE[2] - 1
                ))

                point_xyz = np.array([
                    point["z"], point["y"], point["x"]
                ], dtype=np.float32)

                p = probability[:, z, y, x]

                predicted_class = int(np.argmax(p))
                true_class = int(point["class_id"])

                row = {
                    "study_id": study_id,
                    "series_id": series_id,
                    "level": point["level"],
                    "true_class_id": true_class,
                    "true_class_name": point["class_name"],
                    "predicted_class_id": predicted_class,
                    "predicted_class_name": CLASS_NAMES[
                        predicted_class
                    ],
                    "correct": int(
                        predicted_class == true_class
                    ),
                    "true_probability": float(
                        p[true_class]
                    ),
                    "predicted_probability": float(
                        p[predicted_class]
                    ),
                    "background_probability": float(p[0]),
                    "max_foreground_probability": float(
                        p[1:].max()
                    ),
                    "z": float(point["z"]),
                    "y": float(point["y"]),
                    "x": float(point["x"]),
                    "patient_x": point["patient_x"],
                    "patient_y": point["patient_y"],
                    "patient_z": point["patient_z"],
                }

                for threshold in THRESHOLDS:
                    distance, region = nearest_region(
                        point_xyz,
                        regions[true_class][threshold],
                    )

                    row[
                        f"true_class_distance_thr_{threshold:.2f}"
                    ] = distance

                    row[
                        f"true_class_region_voxels_thr_{threshold:.2f}"
                    ] = (
                        region["voxels"]
                        if region is not None
                        else 0
                    )

                for class_id in [2, 3]:
                    distance, region = nearest_region(
                        point_xyz,
                        regions[class_id][0.50],
                    )

                    row[
                        f"class_{class_id}_distance_thr_0.50"
                    ] = distance

                rows.append(row)

                probability_rows.append({
                    "study_id": study_id,
                    "series_id": series_id,
                    "level": point["level"],
                    "true_class_id": true_class,
                    "true_class_name": point["class_name"],
                    **{
                        f"class_{class_id}_probability":
                        float(p[class_id])
                        for class_id in range(NUM_CLASSES)
                    },
                })

            print(
                f"  Points: {len(points)}"
            )
            print(
                f"  Foreground ratio: "
                f"{foreground_ratio:.6f}"
            )

        except Exception as exc:
            failed.append({
                "study_id": study_id,
                "series_id": series_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(
                f"  FAILED: {type(exc).__name__}: {exc}"
            )

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError(
            "No validation points were successfully audited."
        )

    results["correct"] = (
        results["predicted_class_id"]
        == results["true_class_id"]
    )

    class_rows = []

    for class_id in range(1, NUM_CLASSES):
        subset = results[
            results["true_class_id"] == class_id
        ]

        if subset.empty:
            continue

        class_rows.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "points": len(subset),
            "point_accuracy": float(
                subset["correct"].mean()
            ),
            "mean_true_probability": float(
                subset["true_probability"].mean()
            ),
            "median_true_probability": float(
                subset["true_probability"].median()
            ),
            "hit_rate_0.50": float(
                (
                    subset["true_probability"] >= 0.50
                ).mean()
            ),
            "mean_distance_thr_0.30": float(
                subset[
                    "true_class_distance_thr_0.30"
                ].dropna().mean()
            )
            if subset[
                "true_class_distance_thr_0.30"
            ].notna().any()
            else math.nan,
            "mean_distance_thr_0.50": float(
                subset[
                    "true_class_distance_thr_0.50"
                ].dropna().mean()
            )
            if subset[
                "true_class_distance_thr_0.50"
            ].notna().any()
            else math.nan,
            "mean_distance_thr_0.70": float(
                subset[
                    "true_class_distance_thr_0.70"
                ].dropna().mean()
            )
            if subset[
                "true_class_distance_thr_0.70"
            ].notna().any()
            else math.nan,
        })

    class_summary = pd.DataFrame(class_rows)

    confusion_rows = []

    for (true_class, predicted_class), subset in (
        results.groupby(
            ["true_class_id", "predicted_class_id"]
        )
    ):
        confusion_rows.append({
            "true_class_id": int(true_class),
            "true_class_name": CLASS_NAMES[int(true_class)],
            "predicted_class_id": int(predicted_class),
            "predicted_class_name": CLASS_NAMES[
                int(predicted_class)
            ],
            "points": len(subset),
            "mean_true_probability": float(
                subset["true_probability"].mean()
            ),
            "mean_predicted_probability": float(
                subset["predicted_probability"].mean()
            ),
            "mean_lfnn_distance": float(
                subset[
                    "class_2_distance_thr_0.50"
                ].dropna().mean()
            )
            if subset[
                "class_2_distance_thr_0.50"
            ].notna().any()
            else math.nan,
            "mean_rfnn_distance": float(
                subset[
                    "class_3_distance_thr_0.50"
                ].dropna().mean()
            )
            if subset[
                "class_3_distance_thr_0.50"
            ].notna().any()
            else math.nan,
        })

    confusion_summary = pd.DataFrame(
        confusion_rows
    )

    # Within-series paired LFNN/RFNN analysis.
    left_right_rows = []

    for (study_id, series_id), group in results[
        results["true_class_id"].isin([2, 3])
    ].groupby(["study_id", "series_id"]):

        if set(group["true_class_id"].unique()) != {2, 3}:
            continue

        lfnn = group[
            group["true_class_id"] == 2
        ].iloc[0]

        rfnn = group[
            group["true_class_id"] == 3
        ].iloc[0]

        left_right_rows.append({
            "study_id": study_id,
            "series_id": series_id,
            "lfnn_x": float(lfnn["x"]),
            "rfnn_x": float(rfnn["x"]),
            "lfnn_patient_x": float(
                lfnn["patient_x"]
            ),
            "rfnn_patient_x": float(
                rfnn["patient_x"]
            ),
            "lfnn_predicted_class": lfnn[
                "predicted_class_name"
            ],
            "rfnn_predicted_class": rfnn[
                "predicted_class_name"
            ],
            "lfnn_true_probability": float(
                lfnn["true_probability"]
            ),
            "rfnn_true_probability": float(
                rfnn["true_probability"]
            ),
            "lfnn_to_rfnn_prediction_distance": float(
                lfnn["class_3_distance_thr_0.50"]
            )
            if pd.notna(
                lfnn["class_3_distance_thr_0.50"]
            )
            else math.nan,
            "rfnn_to_lfnn_prediction_distance": float(
                rfnn["class_2_distance_thr_0.50"]
            )
            if pd.notna(
                rfnn["class_2_distance_thr_0.50"]
            )
            else math.nan,
        })

    left_right_summary = pd.DataFrame(
        left_right_rows
    )

    probability_df = pd.DataFrame(
        probability_rows
    )

    probability_rows_summary = []

    for true_class in range(1, NUM_CLASSES):
        subset = probability_df[
            probability_df["true_class_id"] == true_class
        ]

        for probability_class in range(NUM_CLASSES):
            column = (
                f"class_{probability_class}_probability"
            )

            probability_rows_summary.append({
                "true_class_id": true_class,
                "true_class_name": CLASS_NAMES[
                    true_class
                ],
                "probability_class_id": probability_class,
                "probability_class_name": CLASS_NAMES[
                    probability_class
                ],
                "mean_probability": float(
                    subset[column].mean()
                ),
                "median_probability": float(
                    subset[column].median()
                ),
            })

    probability_summary = pd.DataFrame(
        probability_rows_summary
    )

    # ------------------------------------------------------------------------
    # Save CSV outputs
    # ------------------------------------------------------------------------

    results.to_csv(
        OUTPUT_DIR
        / "part221_point_prediction_matches.csv",
        index=False,
    )

    class_summary.to_csv(
        OUTPUT_DIR
        / "part221_class_spatial_summary.csv",
        index=False,
    )

    left_right_summary.to_csv(
        OUTPUT_DIR
        / "part221_series_left_right_analysis.csv",
        index=False,
    )

    probability_summary.to_csv(
        OUTPUT_DIR
        / "part221_probability_summary.csv",
        index=False,
    )

    confusion_summary.to_csv(
        OUTPUT_DIR
        / "part221_confusion_spatial_analysis.csv",
        index=False,
    )

    pd.DataFrame(failed).to_csv(
        OUTPUT_DIR / "part221_failed_series.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    rf = results[
        results["true_class_id"] == 3
    ]

    lf = results[
        results["true_class_id"] == 2
    ]

    summary = {
        "part": "2.21",
        "status": "COMPLETE",
        "checkpoint": str(CHECKPOINT_PATH),
        "validation_series_selected": int(
            len(selected)
        ),
        "processed_series": int(
            len(selected) - len(failed)
        ),
        "failed_series": int(len(failed)),
        "audited_points": int(len(results)),
        "overall_point_accuracy": float(
            results["correct"].mean()
        ),
        "macro_disease_accuracy": float(
            class_summary["point_accuracy"].mean()
        ),
        "rfnn_points": int(len(rf)),
        "rfnn_predicted_as_lfnn": int(
            (rf["predicted_class_id"] == 2).sum()
        ),
        "lfnn_points": int(len(lf)),
        "lfnn_predicted_as_rfnn": int(
            (lf["predicted_class_id"] == 3).sum()
        ),
        "mean_rfnn_true_probability": float(
            rf["true_probability"].mean()
        ),
        "mean_lfnn_true_probability": float(
            lf["true_probability"].mean()
        ),
        "training_performed": False,
        "checkpoint_modified": False,
        "dashboard_modified": False,
        "manual_voxel_ground_truth_fabricated": False,
        "scientific_limitation": (
            "RSNA annotations are point/localization annotations, "
            "not manual voxel-wise segmentation masks. Spatial "
            "distances therefore do not constitute voxel-wise "
            "Dice or segmentation ground-truth accuracy."
        ),
    }

    with open(
        OUTPUT_DIR / "part221_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    report = [
        "PART 2.21 — PREDICTION–ANNOTATION SPATIAL AUDIT",
        "=" * 78,
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Validation series selected: {len(selected)}",
        f"Processed series: {len(selected) - len(failed)}",
        f"Failed series: {len(failed)}",
        f"Audited points: {len(results)}",
        "",
        (
            "Overall point accuracy: "
            f"{summary['overall_point_accuracy']:.6f}"
        ),
        (
            "Macro disease accuracy: "
            f"{summary['macro_disease_accuracy']:.6f}"
        ),
        "",
        f"RFNN points: {len(rf)}",
        (
            "RFNN -> LFNN: "
            f"{summary['rfnn_predicted_as_lfnn']}"
        ),
        (
            "RFNN mean true probability: "
            f"{summary['mean_rfnn_true_probability']:.6f}"
        ),
        "",
        f"LFNN points: {len(lf)}",
        (
            "LFNN -> RFNN: "
            f"{summary['lfnn_predicted_as_rfnn']}"
        ),
        (
            "LFNN mean true probability: "
            f"{summary['mean_lfnn_true_probability']:.6f}"
        ),
        "",
        "Scientific limitation:",
        (
            "RSNA annotations are point/localization annotations, "
            "not manual voxel-wise segmentation masks."
        ),
        (
            "Spatial distances are therefore diagnostic measurements "
            "and not voxel-wise segmentation accuracy or Dice."
        ),
        "",
        "Training performed: NO",
        "Checkpoint modified: NO",
        "Dashboard modified: NO",
        "Manual voxel ground truth fabricated: NO",
    ]

    (
        REPORT_DIR
        / "part221_prediction_annotation_spatial_audit_report.txt"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("PART 2.21 COMPLETE")
    print("=" * 78)
    print(
        f"Processed series : "
        f"{len(selected) - len(failed)}/{len(selected)}"
    )
    print(
        f"Audited points   : {len(results)}"
    )
    print(
        f"Overall accuracy : "
        f"{summary['overall_point_accuracy']:.6f}"
    )
    print(
        f"Macro accuracy   : "
        f"{summary['macro_disease_accuracy']:.6f}"
    )
    print(
        f"RFNN -> LFNN     : "
        f"{summary['rfnn_predicted_as_lfnn']}"
    )
    print(
        f"LFNN -> RFNN     : "
        f"{summary['lfnn_predicted_as_rfnn']}"
    )
    print()
    print(f"Outputs: {OUTPUT_DIR}")
    print("Training performed: NO")
    print("Checkpoint modified: NO")
    print("Dashboard modified: NO")


if __name__ == "__main__":
    main()
