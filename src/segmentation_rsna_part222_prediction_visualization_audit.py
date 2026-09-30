"""
Part 2.22 — Prediction–Annotation Visualization Audit
======================================================

Purpose
-------
Analysis-only audit of the Part 2.20B Epoch-5 checkpoint.

This script:
- uses the RSNA point-supervision manifest from Part 2.13
- uses the Part 2.20B Epoch-5 checkpoint
- evaluates the same 25 study-disjoint validation series
- produces disease/level/case summaries
- creates diagnostic visualizations showing:
    * native MRI slice
    * annotation point
    * predicted class
    * true-class probability
    * nearest predicted region
    * spatial distance
- DOES NOT train
- DOES NOT modify any checkpoint
- DOES NOT modify the dashboard
- DOES NOT create voxel-wise ground truth
- DOES NOT claim Dice/clinical segmentation accuracy

Important:
RSNA coordinates are point/localization annotations, not manual voxel-wise
segmentation masks. The generated overlays are therefore diagnostic
prediction visualizations, not ground-truth segmentation overlays.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label as cc_label
from scipy.ndimage import center_of_mass
from scipy.ndimage import map_coordinates, zoom

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------
# Imports from the established project pipeline
# ---------------------------------------------------------------------
from segmentation_rsna_part11_controlled_pilot_training_corrected import (
    read_dicom_series_robust,
    resolve_series_dir,
)
# Part 2.13 does not expose CLASS_NAMES as a module-level symbol.
# Keep the established six-class contract local to this audit.

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part213_point_supervision_manifest"
    / "part213_point_manifest.csv"
)

CHECKPOINT_PATH = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part220b_geometry_corrected_training"
    / "checkpoints"
    / "part220b_epoch_05.pth"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part222_prediction_visualization_audit"
)

VIS_DIR = OUT_DIR / "visualizations"
REPORT_DIR = OUT_DIR / "reports"

MODEL_SHAPE = (64, 96, 96)
NUM_CLASSES = 6
FEATURE_SIZE = 12

N_VALIDATION_SERIES = 25
SEED = 42

# Keep visualization volume manageable while covering every disease.
MAX_VISUALIZATIONS_PER_DISEASE = 4
MAX_TOTAL_VISUALIZATIONS = 20

REGION_THRESHOLDS = (0.30, 0.50, 0.70)

CLASS_NAMES_LOCAL = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
try:
    from monai.networks.nets import SwinUNETR
except Exception as exc:
    raise RuntimeError(
        "MONAI is required for Part 2.22. Install it in the project venv."
    ) from exc


def build_model(device: torch.device) -> torch.nn.Module:
    model = SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        spatial_dims=3,
        use_checkpoint=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state = checkpoint["model"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    # Handle DataParallel-style prefixes safely.
    cleaned = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[7:]
        cleaned[new_key] = value

    result = model.load_state_dict(cleaned, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------
# DICOM geometry
# ---------------------------------------------------------------------
def build_geometry(datasets):
    if not datasets:
        raise RuntimeError("No DICOM datasets supplied.")

    first = datasets[0]

    iop = np.asarray(
        getattr(first, "ImageOrientationPatient"),
        dtype=np.float64,
    )
    if iop.size != 6:
        raise RuntimeError("Invalid ImageOrientationPatient.")

    row_dir = iop[:3]
    col_dir = iop[3:]
    row_dir = row_dir / (np.linalg.norm(row_dir) + 1e-12)
    col_dir = col_dir / (np.linalg.norm(col_dir) + 1e-12)

    normal = np.cross(row_dir, col_dir)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    pixel_spacing = np.asarray(
        getattr(first, "PixelSpacing", [1.0, 1.0]),
        dtype=np.float64,
    )
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])

    positions = np.asarray(
        [
            np.asarray(getattr(ds, "ImagePositionPatient"), dtype=np.float64)
            for ds in datasets
        ]
    )

    # Sort by physical position along the slice normal.
    order = np.argsort(positions @ normal)
    positions = positions[order]

    if len(positions) > 1:
        projected = positions @ normal
        diffs = np.diff(projected)
        valid = np.abs(diffs) > 1e-5
        if np.any(valid):
            slice_spacing = float(np.median(np.abs(diffs[valid])))
        else:
            slice_spacing = 1.0
    else:
        slice_spacing = float(
            getattr(first, "SpacingBetweenSlices", 1.0)
            or getattr(first, "SliceThickness", 1.0)
            or 1.0
        )

    return {
        "row_dir": row_dir,
        "col_dir": col_dir,
        "normal": normal,
        "row_spacing": row_spacing,
        "col_spacing": col_spacing,
        "slice_spacing": max(slice_spacing, 1e-6),
        "positions": positions,
    }


def native_point_to_patient(
    native_z: float,
    native_y: float,
    native_x: float,
    datasets,
):
    if not datasets:
        raise RuntimeError("No DICOM datasets.")

    # Match the Part 2.20B coordinate convention.
    z_idx = int(np.clip(round(float(native_z)), 0, len(datasets) - 1))
    ds = datasets[z_idx]

    iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
    row_dir = iop[:3]
    col_dir = iop[3:]

    row_dir = row_dir / (np.linalg.norm(row_dir) + 1e-12)
    col_dir = col_dir / (np.linalg.norm(col_dir) + 1e-12)

    ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
    spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)

    return (
        ipp
        + float(native_y) * float(spacing[0]) * row_dir
        + float(native_x) * float(spacing[1]) * col_dir
    )


def build_canonical_grid(datasets, native_shape):
    geom = build_geometry(datasets)

    patient_points = np.asarray(
        [np.asarray(ds.ImagePositionPatient, dtype=np.float64) for ds in datasets]
    )

    if patient_points.shape[0] == 1:
        patient_points = np.vstack(
            [
                patient_points,
                patient_points
                + geom["normal"] * geom["slice_spacing"],
            ]
        )

    # Physical extents of the acquired image corners.
    depth, height, width = native_shape
    corners = []

    for z in (0, depth - 1):
        for y in (0, height - 1):
            for x in (0, width - 1):
                ds = datasets[int(np.clip(round(z), 0, len(datasets) - 1))]
                iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
                row_dir = iop[:3]
                col_dir = iop[3:]
                row_dir /= np.linalg.norm(row_dir) + 1e-12
                col_dir /= np.linalg.norm(col_dir) + 1e-12

                ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
                spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)

                corners.append(
                    ipp
                    + y * spacing[0] * row_dir
                    + x * spacing[1] * col_dir
                )

    corners = np.asarray(corners)

    # Canonical axes are:
    # X = physical left/right direction derived from DICOM column direction
    # Y = physical row direction
    # Z = physical slice normal
    origin = corners.min(axis=0)
    max_corner = corners.max(axis=0)

    # Use a fixed positive orthonormal basis.
    ex = geom["col_dir"]
    ey = geom["row_dir"]
    ez = geom["normal"]

    # Project all corners onto the canonical basis.
    proj = np.column_stack(
        [
            corners @ ex,
            corners @ ey,
            corners @ ez,
        ]
    )

    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)

    # Avoid zero extents.
    extents = np.maximum(maxs - mins, 1e-6)

    shape = np.asarray(MODEL_SHAPE, dtype=np.float64)
    spacing = extents / np.maximum(shape - 1.0, 1.0)

    return {
        "origin_projection": mins,
        "basis": np.stack([ex, ey, ez], axis=0),
        "spacing": spacing,
        "mins": mins,
        "maxs": maxs,
        "shape": MODEL_SHAPE,
        "origin_patient": origin,
    }


def patient_to_native(patient_point, datasets):
    """Map patient coordinate to native DICOM continuous voxel coordinates."""
    if not datasets:
        raise RuntimeError("No DICOM datasets.")

    ds0 = datasets[0]
    iop = np.asarray(ds0.ImageOrientationPatient, dtype=np.float64)
    row_dir = iop[:3]
    col_dir = iop[3:]
    row_dir /= np.linalg.norm(row_dir) + 1e-12
    col_dir /= np.linalg.norm(col_dir) + 1e-12

    normal = np.cross(row_dir, col_dir)
    normal /= np.linalg.norm(normal) + 1e-12

    spacing = np.asarray(ds0.PixelSpacing, dtype=np.float64)
    positions = np.asarray(
        [np.asarray(ds.ImagePositionPatient, dtype=np.float64) for ds in datasets]
    )

    p = np.asarray(patient_point, dtype=np.float64)

    z_proj = positions @ normal
    z = float(np.argmin(np.abs(z_proj - np.dot(p, normal))))

    ipp = positions[int(z)]
    y = np.dot(p - ipp, row_dir) / float(spacing[0])
    x = np.dot(p - ipp, col_dir) / float(spacing[1])

    return np.asarray([z, y, x], dtype=np.float64)


def patient_point_to_canonical(patient_point, grid):
    p = np.asarray(patient_point, dtype=np.float64)

    proj = grid["basis"] @ p
    coord = (proj - grid["mins"]) / grid["spacing"]

    return coord


def resample_to_canonical(image_native, datasets, grid):
    """Resample acquired native volume into the bounded canonical grid."""
    image_native = np.asarray(image_native, dtype=np.float32)

    shape = np.asarray(grid["shape"], dtype=int)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    canonical_proj = (
        grid["mins"][None, None, None, :]
        + np.stack(
            [
                xx * grid["spacing"][0],
                yy * grid["spacing"][1],
                zz * grid["spacing"][2],
            ],
            axis=-1,
        )
    )

    # Convert canonical projection coordinates into patient coordinates.
    patient = np.einsum(
        "ij,zyxj->zyxi",
        grid["basis"],
        canonical_proj,
    )

    # The canonical projection is represented in patient coordinates.
    # Invert the basis transform explicitly.
    inv_basis = np.linalg.inv(grid["basis"])

    flat_patient = patient.reshape(-1, 3)
    flat_proj = flat_patient @ inv_basis.T

    # Native voxel coordinates.
    ds0 = datasets[0]
    iop = np.asarray(ds0.ImageOrientationPatient, dtype=np.float64)
    row_dir = iop[:3]
    col_dir = iop[3:]
    row_dir /= np.linalg.norm(row_dir) + 1e-12
    col_dir /= np.linalg.norm(col_dir) + 1e-12
    normal = np.cross(row_dir, col_dir)
    normal /= np.linalg.norm(normal) + 1e-12

    positions = np.asarray(
        [np.asarray(ds.ImagePositionPatient, dtype=np.float64) for ds in datasets]
    )
    spacing = np.asarray(ds0.PixelSpacing, dtype=np.float64)

    # Native coordinate mapping.
    zproj = positions @ normal
    native_z = np.interp(
        flat_patient @ normal,
        zproj,
        np.arange(len(zproj), dtype=np.float64),
    )

    ipp0 = positions[0]
    native_y = (flat_patient - ipp0) @ row_dir / float(spacing[0])
    native_x = (flat_patient - ipp0) @ col_dir / float(spacing[1])

    coords = np.vstack(
        [
            native_z,
            native_y,
            native_x,
        ]
    )

    valid = (
        (native_z >= 0)
        & (native_z <= image_native.shape[0] - 1)
        & (native_y >= 0)
        & (native_y <= image_native.shape[1] - 1)
        & (native_x >= 0)
        & (native_x <= image_native.shape[2] - 1)
    )

    sampled = map_coordinates(
        image_native,
        coords,
        order=1,
        mode="constant",
        cval=0.0,
    )

    sampled[~valid] = 0.0

    return sampled.reshape(tuple(shape.tolist())).astype(np.float32), valid.reshape(
        tuple(shape.tolist())
    )


def normalize_volume(volume):
    volume = np.asarray(volume, dtype=np.float32)
    valid = np.isfinite(volume)

    if not valid.any():
        return np.zeros_like(volume, dtype=np.float32)

    values = volume[valid]
    lo, hi = np.percentile(values, [1, 99])

    if hi <= lo:
        out = volume - float(values.mean())
        std = float(values.std())
        return out / max(std, 1e-6)

    out = np.clip(volume, lo, hi)
    out = (out - lo) / max(hi - lo, 1e-6)
    return out.astype(np.float32)


# ---------------------------------------------------------------------
# Validation selection
# ---------------------------------------------------------------------
def select_validation_series(manifest):
    """Reproduce the study-disjoint 25-series validation design."""
    unique = (
        manifest[
            [
                "study_id",
                "series_id",
                "series_description",
            ]
        ]
        .drop_duplicates()
        .sort_values(["study_id", "series_id"])
        .reset_index(drop=True)
    )

    studies = sorted(unique["study_id"].astype(str).unique())

    rng = np.random.default_rng(SEED)
    if len(studies) < N_VALIDATION_SERIES:
        raise RuntimeError(
            f"Only {len(studies)} studies available; "
            f"need {N_VALIDATION_SERIES}."
        )

    selected_studies = set(
        rng.choice(studies, size=N_VALIDATION_SERIES, replace=False).tolist()
    )

    selected = unique[
        unique["study_id"].astype(str).isin(selected_studies)
    ].copy()

    # Prefer one annotated series per selected study.
    selected = (
        selected.sort_values(["study_id", "series_id"])
        .groupby("study_id", as_index=False)
        .first()
    )

    return selected.reset_index(drop=True)


# ---------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------
@torch.no_grad()
def predict_volume(model, canonical_volume, device):
    x = torch.from_numpy(canonical_volume).float()[None, None].to(device)

    with torch.amp.autocast(
        "cuda",
        enabled=(device.type == "cuda"),
    ):
        logits = model(x)

    probability = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    return probability.astype(np.float32)


# ---------------------------------------------------------------------
# Connected region diagnostics
# ---------------------------------------------------------------------
def connected_components_for_class(probability_class, threshold):
    binary = probability_class >= float(threshold)

    if not binary.any():
        return []

    labeled, n = cc_label(binary)

    regions = []
    for component_id in range(1, n + 1):
        coords = np.argwhere(labeled == component_id)
        if coords.size == 0:
            continue

        center = coords.mean(axis=0)
        regions.append(
            {
                "component_id": int(component_id),
                "size": int(len(coords)),
                "centroid_z": float(center[0]),
                "centroid_y": float(center[1]),
                "centroid_x": float(center[2]),
            }
        )

    return regions


def nearest_region_distance(point, regions):
    if not regions:
        return math.nan, None

    p = np.asarray(point, dtype=np.float64)

    best = None
    best_distance = float("inf")

    for region in regions:
        c = np.asarray(
            [
                region["centroid_z"],
                region["centroid_y"],
                region["centroid_x"],
            ],
            dtype=np.float64,
        )

        distance = float(np.linalg.norm(p - c))

        if distance < best_distance:
            best_distance = distance
            best = region

    return best_distance, best


# ---------------------------------------------------------------------
# Native slice extraction
# ---------------------------------------------------------------------
def extract_native_slice(image_native, native_point):
    z, y, x = native_point

    zi = int(np.clip(round(float(z)), 0, image_native.shape[0] - 1))

    return (
        image_native[zi],
        zi,
        float(y),
        float(x),
    )


def resize_probability_slice(probability_slice, target_shape):
    """Resize a 2D probability map only for visualization."""
    h, w = target_shape

    if probability_slice.shape == target_shape:
        return probability_slice

    zoom_factors = (
        h / probability_slice.shape[0],
        w / probability_slice.shape[1],
    )

    return zoom(
        probability_slice,
        zoom_factors,
        order=1,
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def save_point_visualization(
    *,
    image_native,
    native_point,
    probability,
    true_class,
    predicted_class,
    true_probability,
    nearest_distance,
    nearest_region,
    study_id,
    series_id,
    level,
    series_description,
    output_path,
):
    image_slice, z_index, y_native, x_native = extract_native_slice(
        image_native,
        native_point,
    )

    image_slice = np.asarray(image_slice, dtype=np.float32)

    finite = np.isfinite(image_slice)
    if finite.any():
        lo, hi = np.percentile(image_slice[finite], [1, 99])
        image_display = np.clip(image_slice, lo, hi)
    else:
        image_display = image_slice

    if image_display.max() > image_display.min():
        image_display = (
            image_display - image_display.min()
        ) / (
            image_display.max() - image_display.min()
        )

    # Diagnostic probability slice in the model/canonical space.
    # The annotation is mapped to canonical coordinates.
    # For visualization we use the closest canonical Z plane.
    # This is deliberately labelled as model-space probability.
    #
    # The native MRI remains the acquisition-space image.
    canonical_point = None

    # The caller stores the canonical point in the object attached to
    # the nearest region dictionary when available.
    if nearest_region is not None and "_canonical_point" in nearest_region:
        canonical_point = nearest_region["_canonical_point"]

    if canonical_point is None:
        canonical_z = probability.shape[1] // 2
    else:
        canonical_z = int(
            np.clip(
                round(float(canonical_point[0])),
                0,
                probability.shape[1] - 1,
            )
        )

    # For a compact diagnostic panel, show the predicted-class
    # probability in its model-space axial slice.
    prob_class = probability[true_class]
    prob_slice = prob_class[canonical_z]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    ax = axes[0, 0]
    ax.imshow(image_display, cmap="gray")
    ax.scatter(
        [x_native],
        [y_native],
        marker="x",
        s=100,
        linewidths=2,
    )
    ax.set_title("Native MRI + RSNA point")
    ax.set_axis_off()

    ax = axes[0, 1]
    ax.imshow(image_display, cmap="gray")
    ax.scatter(
        [x_native],
        [y_native],
        marker="x",
        s=120,
        linewidths=2,
    )
    ax.set_title(f"Native slice z={z_index}")
    ax.set_axis_off()

    ax = axes[0, 2]
    ax.imshow(image_display, cmap="gray")
    ax.set_title("Native MRI")
    ax.set_axis_off()

    ax = axes[1, 0]
    im = ax.imshow(prob_slice, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title(
        f"Model-space P(true class)\n"
        f"{CLASS_NAMES_LOCAL.get(true_class, str(true_class))}"
    )
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.axis("off")

    status = "CORRECT" if predicted_class == true_class else "INCORRECT"

    lines = [
        f"Study: {study_id}",
        f"Series: {series_id}",
        f"Description: {series_description}",
        f"Level: {level}",
        "",
        f"True: {CLASS_NAMES_LOCAL.get(true_class, str(true_class))}",
        f"Predicted: {CLASS_NAMES_LOCAL.get(predicted_class, str(predicted_class))}",
        f"Status: {status}",
        "",
        f"True-class probability: {true_probability:.4f}",
        (
            f"Nearest true-class region distance: "
            f"{nearest_distance:.3f} voxels"
            if np.isfinite(nearest_distance)
            else "Nearest true-class region distance: N/A"
        ),
    ]

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    ax = axes[1, 2]
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        (
            "Interpretation\n\n"
            "This is a point-supervision diagnostic.\n"
            "The RSNA annotation is a localization point,\n"
            "not a voxel-wise segmentation mask.\n\n"
            "The probability map is model-space.\n"
            "The MRI image is native acquisition-space.\n\n"
            "Do not interpret the probability map as\n"
            "validated lesion ground truth."
        ),
        va="top",
        ha="left",
        fontsize=10,
    )

    fig.suptitle(
        f"Part 2.22 — Prediction–Annotation Audit | {status}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------
def main():
    print("=" * 78)
    print("PART 2.22 — PREDICTION–ANNOTATION VISUALIZATION AUDIT")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "study_id",
        "series_id",
        "series_description",
        "class_id",
        "class_name",
        "level",
        "native_z",
        "native_y",
        "native_x",
    }

    missing = sorted(required_columns - set(manifest.columns))
    if missing:
        raise RuntimeError(
            f"Manifest is missing required columns: {missing}"
        )

    print(f"Manifest rows: {len(manifest)}")
    print(
        "Annotated series:",
        manifest[["study_id", "series_id"]].drop_duplicates().shape[0],
    )

    selected = select_validation_series(manifest)

    print(f"Validation series selected: {len(selected)}")
    print(
        "Validation studies selected:",
        selected["study_id"].astype(str).nunique(),
    )

    model = build_model(device)
    print("Part 2.20B Epoch 5 checkpoint loaded successfully.")

    case_rows = []
    class_rows = []
    level_rows = []
    spatial_rows = []
    probability_rows = []
    confusion_rows = []
    failed_rows = []

    visualizations_written = 0
    disease_visualization_counts = {
        c: 0 for c in range(1, NUM_CLASSES)
    }

    for index, selected_row in selected.iterrows():
        study_id = str(selected_row["study_id"])
        series_id = str(selected_row["series_id"])

        print(
            f"\n[{index + 1:02d}/{len(selected):02d}] "
            f"{study_id}/{series_id}"
        )

        try:
            series_rows = manifest[
                (manifest["study_id"].astype(str) == study_id)
                & (manifest["series_id"].astype(str) == series_id)
            ].copy()

            if series_rows.empty:
                raise RuntimeError(
                    "No manifest annotations for selected series."
                )

            row0 = series_rows.iloc[0]

            series_dir = resolve_series_dir(row0)

            image_native, datasets, dicom_info = (
                read_dicom_series_robust(series_dir)
            )

            image_native = np.asarray(
                image_native,
                dtype=np.float32,
            )

            if image_native.ndim != 3:
                raise RuntimeError(
                    f"Expected 3D native image, got {image_native.shape}"
                )

            grid = build_canonical_grid(
                datasets,
                image_native.shape,
            )

            canonical, valid_mask = resample_to_canonical(
                image_native,
                datasets,
                grid,
            )

            canonical = normalize_volume(canonical)

            probability = predict_volume(
                model,
                canonical,
                device,
            )

            predicted_labels = probability.argmax(axis=0)
            foreground_ratio = float(
                (predicted_labels != 0).mean()
            )

            print(
                f"  Points: {len(series_rows)}"
            )
            print(
                f"  Foreground ratio: {foreground_ratio:.6f}"
            )

            case_id = f"{study_id}_{series_id}"

            # Store per-class regions once for each threshold.
            regions_by_threshold = {}

            for threshold in REGION_THRESHOLDS:
                regions_by_threshold[threshold] = {
                    class_id: connected_components_for_class(
                        probability[class_id],
                        threshold,
                    )
                    for class_id in range(1, NUM_CLASSES)
                }

            for _, ann in series_rows.iterrows():
                true_class = int(ann["class_id"])

                native_point = np.asarray(
                    [
                        float(ann["native_z"]),
                        float(ann["native_y"]),
                        float(ann["native_x"]),
                    ],
                    dtype=np.float64,
                )

                patient_point = native_point_to_patient(
                    native_point[0],
                    native_point[1],
                    native_point[2],
                    datasets,
                )

                canonical_point = patient_point_to_canonical(
                    patient_point,
                    grid,
                )

                canonical_point = np.asarray(
                    canonical_point,
                    dtype=np.float64,
                )

                canonical_point = np.clip(
                    canonical_point,
                    [0, 0, 0],
                    np.asarray(MODEL_SHAPE) - 1,
                )

                cz, cy, cx = (
                    int(round(float(v)))
                    for v in canonical_point
                )

                point_probability = probability[
                    :,
                    cz,
                    cy,
                    cx,
                ]

                predicted_class = int(
                    np.argmax(point_probability)
                )

                true_probability = float(
                    point_probability[true_class]
                )

                max_foreground_probability = float(
                    point_probability[1:].max()
                )

                # Nearest true-class region at threshold .50.
                threshold_regions = regions_by_threshold[0.50]
                true_regions = threshold_regions.get(
                    true_class,
                    [],
                )

                distance, nearest = nearest_region_distance(
                    canonical_point,
                    true_regions,
                )

                if nearest is not None:
                    nearest = dict(nearest)
                    nearest["_canonical_point"] = canonical_point

                correct = int(predicted_class == true_class)

                row = {
                    "study_id": study_id,
                    "series_id": series_id,
                    "series_description": str(
                        ann["series_description"]
                    ),
                    "level": str(ann["level"]),
                    "true_class": true_class,
                    "true_class_name": CLASS_NAMES_LOCAL.get(
                        true_class,
                        str(true_class),
                    ),
                    "predicted_class": predicted_class,
                    "predicted_class_name": CLASS_NAMES_LOCAL.get(
                        predicted_class,
                        str(predicted_class),
                    ),
                    "correct": correct,
                    "true_probability": true_probability,
                    "max_foreground_probability": max_foreground_probability,
                    "nearest_true_region_distance_voxels": distance,
                    "native_z": float(native_point[0]),
                    "native_y": float(native_point[1]),
                    "native_x": float(native_point[2]),
                    "patient_x": float(patient_point[0]),
                    "patient_y": float(patient_point[1]),
                    "patient_z": float(patient_point[2]),
                    "canonical_z": float(canonical_point[0]),
                    "canonical_y": float(canonical_point[1]),
                    "canonical_x": float(canonical_point[2]),
                    "native_shape": str(tuple(image_native.shape)),
                    "model_shape": str(MODEL_SHAPE),
                    "foreground_ratio": foreground_ratio,
                }

                case_rows.append(row)

                probability_rows.append(
                    {
                        "study_id": study_id,
                        "series_id": series_id,
                        "level": str(ann["level"]),
                        "true_class": true_class,
                        "true_probability": true_probability,
                        "max_foreground_probability": max_foreground_probability,
                    }
                )

                spatial_rows.append(
                    {
                        "study_id": study_id,
                        "series_id": series_id,
                        "true_class": true_class,
                        "level": str(ann["level"]),
                        "nearest_true_region_distance_voxels": distance,
                        "native_x": float(native_point[2]),
                        "patient_x": float(patient_point[0]),
                        "canonical_x": float(canonical_point[2]),
                    }
                )

                confusion_rows.append(
                    {
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "true_class_name": CLASS_NAMES_LOCAL.get(
                            true_class,
                            str(true_class),
                        ),
                        "predicted_class_name": CLASS_NAMES_LOCAL.get(
                            predicted_class,
                            str(predicted_class),
                        ),
                    }
                )

                # Disease-specific visualization quota.
                if (
                    visualizations_written < MAX_TOTAL_VISUALIZATIONS
                    and disease_visualization_counts.get(
                        true_class,
                        0,
                    )
                    < MAX_VISUALIZATIONS_PER_DISEASE
                ):
                    filename = (
                        f"{case_id}_"
                        f"{str(ann['level']).replace('/', '_')}_"
                        f"class{true_class}_"
                        f"pred{predicted_class}.png"
                    )

                    save_point_visualization(
                        image_native=image_native,
                        native_point=native_point,
                        probability=probability,
                        true_class=true_class,
                        predicted_class=predicted_class,
                        true_probability=true_probability,
                        nearest_distance=distance,
                        nearest_region=nearest,
                        study_id=study_id,
                        series_id=series_id,
                        level=str(ann["level"]),
                        series_description=str(
                            ann["series_description"]
                        ),
                        output_path=VIS_DIR / filename,
                    )

                    visualizations_written += 1
                    disease_visualization_counts[true_class] += 1

            # Per-case row.
            case_rows_for_case = [
                r
                for r in case_rows
                if r["study_id"] == study_id
                and r["series_id"] == series_id
            ]

            case_accuracy = float(
                np.mean(
                    [
                        r["correct"]
                        for r in case_rows_for_case
                    ]
                )
            )

            case_rows_summary = {
                "study_id": study_id,
                "series_id": series_id,
                "series_description": str(
                    series_rows.iloc[0]["series_description"]
                ),
                "points": len(case_rows_for_case),
                "accuracy": case_accuracy,
                "foreground_ratio": foreground_ratio,
            }

            # Save case summary separately below.
            class_rows.append(case_rows_summary)

        except Exception as exc:
            print(f"  FAILED: {exc}")

            failed_rows.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "error": str(exc),
                }
            )

    results = pd.DataFrame(case_rows)

    if results.empty:
        raise RuntimeError(
            "Part 2.22 produced no point-level results."
        )

    # -----------------------------------------------------------------
    # Summaries
    # -----------------------------------------------------------------
    disease_summary = (
        results.groupby(
            ["true_class", "true_class_name"],
            dropna=False,
        )
        .agg(
            points=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_true_probability=(
                "true_probability",
                "mean",
            ),
            median_true_probability=(
                "true_probability",
                "median",
            ),
            mean_nearest_region_distance=(
                "nearest_true_region_distance_voxels",
                "mean",
            ),
        )
        .reset_index()
    )

    level_summary = (
        results.groupby("level", dropna=False)
        .agg(
            points=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_true_probability=(
                "true_probability",
                "mean",
            ),
            mean_nearest_region_distance=(
                "nearest_true_region_distance_voxels",
                "mean",
            ),
        )
        .reset_index()
    )

    probability_summary = (
        results[
            [
                "true_class",
                "true_class_name",
                "true_probability",
                "max_foreground_probability",
            ]
        ]
        .groupby(
            ["true_class", "true_class_name"],
            dropna=False,
        )
        .agg(
            points=("true_probability", "size"),
            mean_true_probability=(
                "true_probability",
                "mean",
            ),
            median_true_probability=(
                "true_probability",
                "median",
            ),
            mean_max_foreground_probability=(
                "max_foreground_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    confusion_summary = (
        results.groupby(
            [
                "true_class",
                "true_class_name",
                "predicted_class",
                "predicted_class_name",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["true_class", "count"],
            ascending=[True, False],
        )
    )

    # Specific foraminal cross-confusions.
    rfnn_to_lfnn = int(
        (
            (results["true_class"] == 3)
            & (results["predicted_class"] == 2)
        ).sum()
    )

    lfnn_to_rfnn = int(
        (
            (results["true_class"] == 2)
            & (results["predicted_class"] == 3)
        ).sum()
    )

    overall_accuracy = float(results["correct"].mean())

    macro_accuracy = float(
        disease_summary["accuracy"].mean()
    )

    mean_probability = float(
        results["true_probability"].mean()
    )

    hit_rate_50 = float(
        (results["true_probability"] >= 0.50).mean()
    )

    mean_distance = float(
        results["nearest_true_region_distance_voxels"].mean()
    )

    # Case summaries are held in class_rows under the established name.
    case_summary = pd.DataFrame(class_rows)

    failed_summary = pd.DataFrame(failed_rows)

    # -----------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------
    results.to_csv(
        OUT_DIR / "part222_point_prediction_results.csv",
        index=False,
    )

    disease_summary.to_csv(
        OUT_DIR / "part222_disease_summary.csv",
        index=False,
    )

    level_summary.to_csv(
        OUT_DIR / "part222_level_summary.csv",
        index=False,
    )

    probability_summary.to_csv(
        OUT_DIR / "part222_probability_summary.csv",
        index=False,
    )

    confusion_summary.to_csv(
        OUT_DIR / "part222_confusion_summary.csv",
        index=False,
    )

    case_summary.to_csv(
        OUT_DIR / "part222_case_summary.csv",
        index=False,
    )

    results[
        [
            "study_id",
            "series_id",
            "true_class",
            "true_class_name",
            "predicted_class",
            "predicted_class_name",
            "correct",
            "native_x",
            "patient_x",
            "canonical_x",
            "nearest_true_region_distance_voxels",
        ]
    ].to_csv(
        OUT_DIR / "part222_spatial_summary.csv",
        index=False,
    )

    failed_summary.to_csv(
        OUT_DIR / "part222_failed_series.csv",
        index=False,
    )

    summary = {
        "part": "2.22",
        "status": "COMPLETE",
        "validation_series_processed": int(
            len(selected) - len(failed_rows)
        ),
        "validation_series_failed": int(len(failed_rows)),
        "audited_points": int(len(results)),
        "overall_point_accuracy": overall_accuracy,
        "macro_disease_accuracy": macro_accuracy,
        "mean_true_class_probability": mean_probability,
        "hit_rate_at_0_50": hit_rate_50,
        "mean_nearest_true_class_region_distance_voxels": mean_distance,
        "rfnn_to_lfnn": rfnn_to_lfnn,
        "lfnn_to_rfnn": lfnn_to_rfnn,
        "visualizations_written": int(visualizations_written),
        "checkpoint": str(CHECKPOINT_PATH),
        "training_performed": False,
        "checkpoint_modified": False,
        "dashboard_modified": False,
        "manual_voxel_ground_truth_created": False,
        "scientific_limitation": (
            "RSNA annotations are point/localization annotations, "
            "not manual voxel-wise segmentation masks. "
            "Part 2.22 visualizations are diagnostic prediction "
            "visualizations and must not be interpreted as validated "
            "voxel-wise disease segmentation."
        ),
    }

    with open(
        OUT_DIR / "part222_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Text report
    # -----------------------------------------------------------------
    report_lines = [
        "=" * 78,
        "PART 2.22 — PREDICTION–ANNOTATION VISUALIZATION AUDIT REPORT",
        "=" * 78,
        "",
        f"Validation series processed : {summary['validation_series_processed']}",
        f"Validation series failed    : {summary['validation_series_failed']}",
        f"Audited points              : {summary['audited_points']}",
        f"Overall point accuracy      : {overall_accuracy:.6f}",
        f"Macro disease accuracy      : {macro_accuracy:.6f}",
        f"Mean true-class probability : {mean_probability:.6f}",
        f"Hit rate @ 0.50             : {hit_rate_50:.6f}",
        f"Mean region distance        : {mean_distance:.6f}",
        f"RFNN -> LFNN                : {rfnn_to_lfnn}",
        f"LFNN -> RFNN                : {lfnn_to_rfnn}",
        f"Visualizations written      : {visualizations_written}",
        "",
        "Disease summary:",
        disease_summary.to_string(index=False),
        "",
        "Level summary:",
        level_summary.to_string(index=False),
        "",
        "Confusion summary:",
        confusion_summary.to_string(index=False),
        "",
        "Scientific limitation:",
        summary["scientific_limitation"],
        "",
        "Training performed: NO",
        "Checkpoint modified: NO",
        "Dashboard modified: NO",
    ]

    with open(
        REPORT_DIR / "part222_prediction_visualization_audit_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(report_lines))

    print("\n" + "=" * 78)
    print("PART 2.22 COMPLETE")
    print("=" * 78)
    print(
        f"Processed series : "
        f"{summary['validation_series_processed']}/"
        f"{len(selected)}"
    )
    print(
        f"Audited points   : {summary['audited_points']}"
    )
    print(
        f"Overall accuracy : {overall_accuracy:.6f}"
    )
    print(
        f"Macro accuracy   : {macro_accuracy:.6f}"
    )
    print(
        f"Mean probability : {mean_probability:.6f}"
    )
    print(
        f"Hit rate @0.50   : {hit_rate_50:.6f}"
    )
    print(
        f"Mean region dist : {mean_distance:.6f}"
    )
    print(
        f"RFNN -> LFNN     : {rfnn_to_lfnn}"
    )
    print(
        f"LFNN -> RFNN     : {lfnn_to_rfnn}"
    )
    print(
        f"Visualizations   : {visualizations_written}"
    )
    print("")
    print(f"Outputs: {OUT_DIR}")
    print("Training performed: NO")
    print("Checkpoint modified: NO")
    print("Dashboard modified: NO")


if __name__ == "__main__":
    main()
