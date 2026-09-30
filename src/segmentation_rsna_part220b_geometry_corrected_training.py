"""
PART 2.20B
Geometry-Corrected Point-Supervised Swin-UNETR Training

Purpose
-------
Train a controlled point-supervised Swin-UNETR pilot after correcting
the major acquisition-orientation problem identified in Part 2.20A.

Key correction
--------------
The model no longer receives the native DICOM pixel grid directly.

Pipeline:

    DICOM series
        |
        v
    DICOM physical geometry
        |
        v
    Patient-space canonical grid (X,Y,Z)
        |
        +--> RSNA point annotations mapped in the SAME physical space
        |
        v
    Canonical 64 x 96 x 96 volume
        |
        v
    Swin-UNETR
        |
        v
    6-class output

Classes
-------
0 Background
1 Spinal Canal Stenosis
2 Left Neural Foraminal Narrowing
3 Right Neural Foraminal Narrowing
4 Left Subarticular Stenosis
5 Right Subarticular Stenosis

IMPORTANT
---------
- This is point-supervised training.
- RSNA coordinates are NOT voxel-wise segmentation masks.
- No manual voxel ground truth is fabricated.
- No Dice score is claimed.
- Part104 is used only as initialization.
- Part104 is NEVER overwritten.
- Existing Part216 / Part218 checkpoints are NOT modified.
- Dashboard is NOT modified.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import map_coordinates


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES_DIR = DATASET_ROOT / "train_images"

MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part213_point_supervision_manifest"
    / "part213_point_manifest.csv"
)

# Protected initialization checkpoint.
PART104_CHECKPOINT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part104_final_checkpoint_selection"
    / "checkpoints"
    / "final_segmentation_model.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part220b_geometry_corrected_training"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORT_DIR = OUTPUT_DIR / "reports"
VIS_DIR = OUTPUT_DIR / "visualizations"


for directory in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    METRICS_DIR,
    REPORT_DIR,
    VIS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

MODEL_SHAPE = (
    64,
    96,
    96,
)

NUM_CLASSES = 6

EPOCHS = 5

TRAIN_SLOTS_PER_EPOCH = 100

VALIDATION_CASES = 25

BATCH_SIZE = 1

GRAD_ACCUMULATION = 4

LEARNING_RATE = 2.5e-5

WEIGHT_DECAY = 1e-5

BACKGROUND_WEIGHT = 0.10

POINT_RADIUS = 1

# Number of random background samples per case.
BACKGROUND_SAMPLES = 128

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. CLASS CONTRACT
# ============================================================

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


FORAMINAL_CLASSES = {
    2,
    3,
}


# ============================================================
# 4. RANDOM SEEDS
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# 5. IMPORT ROBUST DICOM LOADER
# ============================================================

try:

    from segmentation_rsna_part11_controlled_pilot_training_corrected import (
        read_dicom_series_robust,
        resolve_series_dir,
    )

except ImportError:

    from src.segmentation_rsna_part11_controlled_pilot_training_corrected import (
        read_dicom_series_robust,
        resolve_series_dir,
    )


# ============================================================
# 6. MONAI
# ============================================================

try:

    from monai.networks.nets import SwinUNETR

except ImportError as exc:

    raise ImportError(
        "\nMONAI is not installed in the project environment.\n"
        "Install it using:\n\n"
        ".\\venv\\Scripts\\python.exe -m pip install monai\n"
    ) from exc


# ============================================================
# 7. DICOM GEOMETRY HELPERS
# ============================================================

def normalize_vector(vector):

    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        raise ValueError(
            "Cannot normalize zero-length vector."
        )

    return vector / norm


def get_iop(ds):

    value = getattr(
        ds,
        "ImageOrientationPatient",
        None,
    )

    if value is None or len(value) != 6:
        raise ValueError(
            "Missing ImageOrientationPatient."
        )

    row_direction = normalize_vector(
        value[:3]
    )

    column_direction = normalize_vector(
        value[3:]
    )

    normal_direction = normalize_vector(
        np.cross(
            row_direction,
            column_direction,
        )
    )

    return (
        row_direction,
        column_direction,
        normal_direction,
    )


def get_pixel_spacing(ds):

    value = getattr(
        ds,
        "PixelSpacing",
        None,
    )

    if value is None or len(value) != 2:
        raise ValueError(
            "Missing PixelSpacing."
        )

    return (
        float(value[0]),
        float(value[1]),
    )


def get_ipp(ds):

    value = getattr(
        ds,
        "ImagePositionPatient",
        None,
    )

    if value is None or len(value) != 3:
        raise ValueError(
            "Missing ImagePositionPatient."
        )

    return np.asarray(
        value,
        dtype=np.float64,
    )


# ============================================================
# 8. BUILD PHYSICAL GEOMETRY
# ============================================================

def build_geometry(
    records,
    image_shape,
):

    if len(records) == 0:
        raise ValueError(
            "No DICOM records."
        )

    first = records[0]

    (
        row_direction,
        column_direction,
        normal_direction,
    ) = get_iop(first)

    row_spacing, column_spacing = (
        get_pixel_spacing(first)
    )

    # Physical slice positions.
    slice_positions = np.asarray(
        [
            float(
                np.dot(
                    get_ipp(ds),
                    normal_direction,
                )
            )
            for ds in records
        ],
        dtype=np.float64,
    )

    # Ensure monotonically increasing physical slice coordinate.
    sort_order = np.argsort(
        slice_positions
    )

    sorted_positions = (
        slice_positions[sort_order]
    )

    # Actual physical spacing between slices.
    if len(sorted_positions) > 1:

        diffs = np.diff(
            sorted_positions
        )

        diffs = diffs[
            np.abs(diffs) > 1e-6
        ]

        if len(diffs) > 0:
            slice_spacing = float(
                np.median(diffs)
            )
        else:
            slice_spacing = 1.0

    else:

        slice_spacing = 1.0

    depth, height, width = (
        image_shape
    )

    # --------------------------------------------------------
    # Physical bounding box
    # --------------------------------------------------------

    corner_points = []

    first_ipp = get_ipp(
        records[0]
    )

    for ds in records:

        ipp = get_ipp(ds)

        for row in [
            0.0,
            float(height - 1),
        ]:

            for col in [
                0.0,
                float(width - 1),
            ]:

                point = (
                    ipp
                    + row
                    * row_spacing
                    * row_direction
                    + col
                    * column_spacing
                    * column_direction
                )

                corner_points.append(
                    point
                )

    corner_points = np.asarray(
        corner_points,
        dtype=np.float64,
    )

    patient_min = corner_points.min(
        axis=0
    )

    patient_max = corner_points.max(
        axis=0
    )

    # Include slice extent explicitly.
    first_projection = sorted_positions[0]
    last_projection = sorted_positions[-1]

    z_low = min(
        first_projection,
        last_projection,
    )

    z_high = max(
        first_projection,
        last_projection,
    )

    # Physical bounds along the normal.
    patient_min = patient_min.copy()
    patient_max = patient_max.copy()

    return {
        "row_direction": row_direction,
        "column_direction": column_direction,
        "normal_direction": normal_direction,
        "row_spacing": row_spacing,
        "column_spacing": column_spacing,
        "slice_spacing": slice_spacing,
        "slice_positions": sorted_positions,
        "sort_order": sort_order,
        "origin": first_ipp,
        "patient_min": patient_min,
        "patient_max": patient_max,
        "image_shape": image_shape,
    }


# ============================================================
# 9. PATIENT SPACE -> NATIVE IMAGE COORDINATES
# ============================================================

def patient_to_native(
    patient_points,
    geometry,
):
    """
    Convert patient-space XYZ coordinates to continuous
    native DICOM coordinates.

    Returns:
        z, y, x
    """

    patient_points = np.asarray(
        patient_points,
        dtype=np.float64,
    )

    row_direction = geometry[
        "row_direction"
    ]

    column_direction = geometry[
        "column_direction"
    ]

    normal_direction = geometry[
        "normal_direction"
    ]

    row_spacing = geometry[
        "row_spacing"
    ]

    column_spacing = geometry[
        "column_spacing"
    ]

    slice_positions = geometry[
        "slice_positions"
    ]

    origin = geometry[
        "origin"
    ]

    delta = (
        patient_points
        - origin
    )

    x_mm = np.dot(
        delta,
        column_direction,
    )

    y_mm = np.dot(
        delta,
        row_direction,
    )

    z_physical = np.dot(
        delta,
        normal_direction,
    )

    x = (
        x_mm
        / max(
            column_spacing,
            1e-8,
        )
    )

    y = (
        y_mm
        / max(
            row_spacing,
            1e-8,
        )
    )

    # Physical slice coordinate -> fractional slice index.
    z = np.interp(
        z_physical
        + float(
            np.dot(
                origin,
                normal_direction,
            )
        ),
        slice_positions,
        np.arange(
            len(slice_positions),
            dtype=np.float64,
        ),
    )

    return (
        z,
        y,
        x,
    )


# ============================================================
# 10. CANONICAL PATIENT-SPACE GRID
# ============================================================

def build_canonical_grid(
    geometry,
):
    """
    Create a fixed canonical patient-space grid.

    Canonical axes:

        X = patient left/right
        Y = patient anterior/posterior
        Z = patient inferior/superior

    The grid is bounded by the physical extent of the
    acquired DICOM series.

    Output:
        arrays of X, Y, Z physical coordinates.
    """

    depth, height, width = MODEL_SHAPE

    patient_min = geometry[
        "patient_min"
    ]

    patient_max = geometry[
        "patient_max"
    ]

    x_values = np.linspace(
        patient_min[0],
        patient_max[0],
        width,
        dtype=np.float64,
    )

    y_values = np.linspace(
        patient_min[1],
        patient_max[1],
        height,
        dtype=np.float64,
    )

    z_values = np.linspace(
        patient_min[2],
        patient_max[2],
        depth,
        dtype=np.float64,
    )

    # Mesh indexing='ij':
    # Z, Y, X
    zz, yy, xx = np.meshgrid(
        z_values,
        y_values,
        x_values,
        indexing="ij",
    )

    return (
        xx,
        yy,
        zz,
    )


# ============================================================
# 11. RESAMPLE IMAGE INTO PATIENT-SPACE CANONICAL GRID
# ============================================================

def resample_to_canonical(
    image,
    records,
    geometry,
):
    """
    Resample native DICOM volume into the canonical
    patient-space grid.

    This is physical-space resampling, NOT simple resize.
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D image, got {image.shape}"
        )

    xx, yy, zz = build_canonical_grid(
        geometry
    )

    # Flatten physical points.
    patient_points = np.stack(
        [
            xx.ravel(),
            yy.ravel(),
            zz.ravel(),
        ],
        axis=1,
    )

    # Patient physical -> native coordinates.
    z, y, x = patient_to_native(
        patient_points,
        geometry,
    )

    depth, height, width = (
        image.shape
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

    coordinates = np.vstack(
        [
            z,
            y,
            x,
        ]
    )

    canonical = map_coordinates(
        image,
        coordinates,
        order=1,
        mode="nearest",
    )

    canonical = canonical.reshape(
        MODEL_SHAPE
    )

    return canonical.astype(
        np.float32
    )


# ============================================================
# 12. IMAGE NORMALIZATION
# ============================================================

def normalize_volume(volume):

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    finite = np.isfinite(
        volume
    )

    if not finite.any():
        return np.zeros_like(
            volume,
            dtype=np.float32,
        )

    valid = volume[
        finite
    ]

    p1, p99 = np.percentile(
        valid,
        [1, 99],
    )

    if p99 <= p1:
        mean = valid.mean()
        std = valid.std()

        if std < 1e-6:
            return np.zeros_like(
                volume,
                dtype=np.float32,
            )

        volume = (
            volume - mean
        ) / std

    else:

        volume = (
            volume - p1
        ) / (
            p99 - p1
        )

        volume = np.clip(
            volume,
            0.0,
            1.0,
        )

    return volume.astype(
        np.float32
    )


# ============================================================
# 13. NATIVE POINT -> PATIENT SPACE
# ============================================================

def native_point_to_patient(
    records,
    native_z,
    native_y,
    native_x,
):

    z_index = int(
        np.clip(
            round(native_z),
            0,
            len(records) - 1,
        )
    )

    ds = records[
        z_index
    ]

    return (
        get_ipp(ds)
        + native_y
        * get_pixel_spacing(ds)[0]
        * get_iop(ds)[0]
        + native_x
        * get_pixel_spacing(ds)[1]
        * get_iop(ds)[1]
    )


# ============================================================
# 14. PATIENT POINT -> CANONICAL MODEL GRID
# ============================================================

def patient_point_to_canonical(
    patient_point,
    geometry,
):

    patient_min = geometry[
        "patient_min"
    ]

    patient_max = geometry[
        "patient_max"
    ]

    extent = (
        patient_max
        - patient_min
    )

    extent = np.where(
        np.abs(extent) < 1e-8,
        1.0,
        extent,
    )

    normalized = (
        patient_point
        - patient_min
    ) / extent

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    d, h, w = MODEL_SHAPE

    canonical_x = (
        normalized[0]
        * (w - 1)
    )

    canonical_y = (
        normalized[1]
        * (h - 1)
    )

    canonical_z = (
        normalized[2]
        * (d - 1)
    )

    return (
        float(canonical_z),
        float(canonical_y),
        float(canonical_x),
    )


# ============================================================
# 15. LOAD MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Part 2.13 manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    df = pd.read_csv(
        MANIFEST_PATH,
        dtype={
            "study_id": str,
            "series_id": str,
        },
    )

    required = [
        "study_id",
        "series_id",
        "condition",
        "class_id",
        "class_name",
        "level",
        "native_z",
        "native_y",
        "native_x",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Manifest missing columns: {missing}"
        )

    return df


# ============================================================
# 16. VALIDATION STUDY SELECTION
# ============================================================

def select_validation_series(
    manifest,
):

    rng = np.random.default_rng(
        SEED
    )

    studies = np.array(
        sorted(
            manifest[
                "study_id"
            ]
            .astype(str)
            .unique()
        )
    )

    shuffled = studies.copy()

    rng.shuffle(
        shuffled
    )

    # Same broad validation-study design.
    validation_studies = set(
        shuffled[
            : min(
                395,
                len(shuffled),
            )
        ]
    )

    validation_manifest = manifest[
        manifest[
            "study_id"
        ]
        .astype(str)
        .isin(
            validation_studies
        )
    ]

    series_table = (
        validation_manifest[
            [
                "study_id",
                "series_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "study_id",
                "series_id",
            ]
        )
    )

    selected = []

    for study_id, group in (
        series_table.groupby(
            "study_id",
            sort=True,
        )
    ):
        selected.append(
            group.iloc[0]
        )

    selected = pd.DataFrame(
        selected
    )

    if len(selected) > VALIDATION_CASES:
        selected = selected.iloc[
            :VALIDATION_CASES
        ].copy()

    return selected


# ============================================================
# 17. LOAD ONE CASE
# ============================================================

def load_case(
    study_id,
    series_id,
    point_df,
):

    series_dir = (
        TRAIN_IMAGES_DIR
        / str(study_id)
        / str(series_id)
    )

    if not series_dir.exists():

        row = pd.Series(
            {
                "study_id": str(
                    study_id
                ),
                "series_id": str(
                    series_id
                ),
            }
        )

        series_dir = resolve_series_dir(
            row
        )

    image, records, info = (
        read_dicom_series_robust(
            series_dir
        )
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    geometry = build_geometry(
        records,
        image.shape,
    )

    # --------------------------------------------------------
    # Physical-space canonical image
    # --------------------------------------------------------

    canonical = resample_to_canonical(
        image,
        records,
        geometry,
    )

    canonical = normalize_volume(
        canonical
    )

    # --------------------------------------------------------
    # Convert every point
    # --------------------------------------------------------

    points = []

    for _, point in point_df.iterrows():

        patient_point = (
            native_point_to_patient(
                records,
                float(
                    point["native_z"]
                ),
                float(
                    point["native_y"]
                ),
                float(
                    point["native_x"]
                ),
            )
        )

        (
            cz,
            cy,
            cx,
        ) = patient_point_to_canonical(
            patient_point,
            geometry,
        )

        points.append(
            {
                "class_id": int(
                    point["class_id"]
                ),
                "class_name": point[
                    "class_name"
                ],
                "level": point[
                    "level"
                ],
                "patient_x": float(
                    patient_point[0]
                ),
                "patient_y": float(
                    patient_point[1]
                ),
                "patient_z": float(
                    patient_point[2]
                ),
                "z": cz,
                "y": cy,
                "x": cx,
            }
        )

    return (
        canonical,
        points,
        geometry,
    )


# ============================================================
# 18. BUILD CASE INDEX
# ============================================================

def build_case_index(
    manifest,
):

    cases = []

    grouped = manifest.groupby(
        [
            "study_id",
            "series_id",
        ],
        sort=True,
    )

    for (
        study_id,
        series_id,
    ), group in grouped:

        cases.append(
            {
                "study_id": str(
                    study_id
                ),
                "series_id": str(
                    series_id
                ),
                "points": group.copy(),
            }
        )

    return cases


# ============================================================
# 19. POINT SUPERVISION LOSS
# ============================================================

def point_supervision_loss(
    logits,
    points,
    background_coords,
    class_weights,
):
    """
    Supervise disease classes directly at annotation points.

    No voxel-wise segmentation mask is created.

    Loss consists of:

        1. disease point cross entropy
        2. sampled background cross entropy

    Background receives a reduced weight.
    """

    device = logits.device

    # logits:
    # [B,C,D,H,W]

    point_losses = []

    for point in points:

        z = int(
            round(
                point["z"]
            )
        )

        y = int(
            round(
                point["y"]
            )
        )

        x = int(
            round(
                point["x"]
            )
        )

        z = max(
            0,
            min(
                logits.shape[2] - 1,
                z,
            ),
        )

        y = max(
            0,
            min(
                logits.shape[3] - 1,
                y,
            ),
        )

        x = max(
            0,
            min(
                logits.shape[4] - 1,
                x,
            ),
        )

        target = torch.tensor(
            [
                int(
                    point["class_id"]
                )
            ],
            device=device,
            dtype=torch.long,
        )

        prediction = logits[
            0,
            :,
            z,
            y,
            x,
        ].unsqueeze(0)

        loss = F.cross_entropy(
            prediction,
            target,
            weight=class_weights,
        )

        point_losses.append(
            loss
        )

    if point_losses:

        disease_loss = torch.stack(
            point_losses
        ).mean()

    else:

        disease_loss = torch.tensor(
            0.0,
            device=device,
        )

    # --------------------------------------------------------
    # Background samples
    # --------------------------------------------------------

    bg_losses = []

    for z, y, x in background_coords:

        prediction = logits[
            0,
            :,
            z,
            y,
            x,
        ].unsqueeze(0)

        target = torch.zeros(
            1,
            device=device,
            dtype=torch.long,
        )

        loss = F.cross_entropy(
            prediction,
            target,
            weight=class_weights,
        )

        bg_losses.append(
            loss
        )

    if bg_losses:

        background_loss = torch.stack(
            bg_losses
        ).mean()

    else:

        background_loss = torch.tensor(
            0.0,
            device=device,
        )

    total_loss = (
        disease_loss
        + BACKGROUND_WEIGHT
        * background_loss
    )

    return (
        total_loss,
        disease_loss.detach(),
        background_loss.detach(),
    )


# ============================================================
# 20. BACKGROUND SAMPLING
# ============================================================

def sample_background(
    points,
    shape,
    count,
):

    depth, height, width = shape

    protected = set()

    for point in points:

        z = int(
            round(
                point["z"]
            )
        )

        y = int(
            round(
                point["y"]
            )
        )

        x = int(
            round(
                point["x"]
            )
        )

        for dz in range(
            -POINT_RADIUS,
            POINT_RADIUS + 1,
        ):

            for dy in range(
                -POINT_RADIUS,
                POINT_RADIUS + 1,
            ):

                for dx in range(
                    -POINT_RADIUS,
                    POINT_RADIUS + 1,
                ):

                    protected.add(
                        (
                            z + dz,
                            y + dy,
                            x + dx,
                        )
                    )

    samples = []

    attempts = 0

    max_attempts = count * 20

    while (
        len(samples) < count
        and attempts < max_attempts
    ):

        attempts += 1

        z = random.randrange(
            depth
        )

        y = random.randrange(
            height
        )

        x = random.randrange(
            width
        )

        if (
            z,
            y,
            x,
        ) in protected:
            continue

        samples.append(
            (
                z,
                y,
                x,
            )
        )

    return samples


# ============================================================
# 21. POINT PREDICTION METRICS
# ============================================================

@torch.no_grad()
def evaluate_case(
    model,
    image,
    points,
):

    model.eval()

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE,
        non_blocking=True,
    )

    logits = model(
        tensor
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

    predictions = (
        probabilities.argmax(
            dim=1
        )
    )

    result_rows = []

    for point in points:

        z = int(
            round(
                point["z"]
            )
        )

        y = int(
            round(
                point["y"]
            )
        )

        x = int(
            round(
                point["x"]
            )
        )

        z = max(
            0,
            min(
                MODEL_SHAPE[0] - 1,
                z,
            ),
        )

        y = max(
            0,
            min(
                MODEL_SHAPE[1] - 1,
                y,
            ),
        )

        x = max(
            0,
            min(
                MODEL_SHAPE[2] - 1,
                x,
            ),
        )

        probs = probabilities[
            0,
            :,
            z,
            y,
            x,
        ]

        pred_class = int(
            predictions[
                0,
                z,
                y,
                x,
            ].item()
        )

        true_class = int(
            point["class_id"]
        )

        true_probability = float(
            probs[
                true_class
            ].item()
        )

        predicted_probability = float(
            probs[
                pred_class
            ].item()
        )

        result_rows.append(
            {
                "true_class": true_class,
                "true_class_name": point[
                    "class_name"
                ],
                "predicted_class": pred_class,
                "predicted_class_name": CLASS_NAMES[
                    pred_class
                ],
                "correct": int(
                    pred_class
                    == true_class
                ),
                "true_probability": true_probability,
                "predicted_probability": predicted_probability,
                "z": float(
                    point["z"]
                ),
                "y": float(
                    point["y"]
                ),
                "x": float(
                    point["x"]
                ),
                "level": point[
                    "level"
                ],
            }
        )

    # Predicted foreground ratio.
    foreground = (
        predictions != 0
    ).float().mean().item()

    return (
        result_rows,
        foreground,
    )


# ============================================================
# 22. MODEL
# ============================================================

def build_model():

    model = SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=12,
        spatial_dims=3,
        use_checkpoint=False,
    )

    model = model.to(
        DEVICE
    )

    return model


# ============================================================
# 23. LOAD PROTECTED PART104
# ============================================================

def load_part104(
    model,
):

    if not PART104_CHECKPOINT.exists():

        raise FileNotFoundError(
            "\nProtected Part104 checkpoint "
            "was not found:\n"
            f"{PART104_CHECKPOINT}"
        )

    checkpoint = torch.load(
        PART104_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif (
            "state_dict"
            in checkpoint
        ):
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    clean_state = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):
            new_key = new_key[
                7:
            ]

        clean_state[
            new_key
        ] = value

    result = model.load_state_dict(
        clean_state,
        strict=False,
    )

    print(
        "\nPart104 initialization:"
    )

    print(
        f"  Missing keys: "
        f"{len(result.missing_keys)}"
    )

    print(
        f"  Unexpected keys: "
        f"{len(result.unexpected_keys)}"
    )

    if result.missing_keys:
        print(
            "  WARNING: missing keys detected."
        )

    if result.unexpected_keys:
        print(
            "  WARNING: unexpected keys detected."
        )

    return model


# ============================================================
# 24. POINT-CLASS WEIGHTS
# ============================================================

def compute_class_weights(
    manifest,
):

    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    for class_id in manifest[
        "class_id"
    ].astype(int):

        if 1 <= class_id < NUM_CLASSES:
            counts[
                class_id
            ] += 1

    disease_counts = counts[
        1:
    ]

    inverse = (
        disease_counts.sum()
        / np.maximum(
            disease_counts,
            1.0,
        )
    )

    # Normalize around 1.
    inverse = (
        inverse
        / inverse.mean()
    )

    # Keep weights bounded.
    inverse = np.clip(
        inverse,
        0.70,
        1.40,
    )

    weights = np.ones(
        NUM_CLASSES,
        dtype=np.float32,
    )

    weights[
        1:
    ] = inverse.astype(
        np.float32
    )

    weights[
        0
    ] = BACKGROUND_WEIGHT

    return (
        counts,
        weights,
    )


# ============================================================
# 25. PREPARE CASES
# ============================================================

def prepare_cases(
    manifest,
    selected_series,
):

    selected_keys = {
        (
            str(row["study_id"]),
            str(row["series_id"]),
        )
        for _, row in selected_series.iterrows()
    }

    selected_manifest = manifest[
        manifest.apply(
            lambda row:
            (
                str(
                    row["study_id"]
                ),
                str(
                    row["series_id"]
                ),
            )
            in selected_keys,
            axis=1,
        )
    ].copy()

    cases = build_case_index(
        selected_manifest
    )

    return cases


# ============================================================
# 26. TRAINING CASE SAMPLER
# ============================================================

def choose_training_case(
    cases,
):

    # Disease-balanced sampling.
    disease_cases = {
        class_id: []
        for class_id in range(
            1,
            NUM_CLASSES,
        )
    }

    for case in cases:

        classes = set(
            case["points"][
                "class_id"
            ].astype(int)
        )

        for class_id in classes:

            if class_id in disease_cases:
                disease_cases[
                    class_id
                ].append(
                    case
                )

    available_classes = [
        c
        for c, values
        in disease_cases.items()
        if values
    ]

    if not available_classes:

        return random.choice(
            cases
        )

    chosen_class = random.choice(
        available_classes
    )

    return random.choice(
        disease_cases[
            chosen_class
        ]
    )


# ============================================================
# 27. TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    optimizer,
    scaler,
    cases,
    class_weights,
    epoch,
):

    model.train()

    running_total = []
    running_disease = []
    running_background = []

    used_studies = set()

    optimizer.zero_grad(
        set_to_none=True
    )

    for slot in range(
        TRAIN_SLOTS_PER_EPOCH
    ):

        case = choose_training_case(
            cases
        )

        study_id = case[
            "study_id"
        ]

        series_id = case[
            "series_id"
        ]

        used_studies.add(
            study_id
        )

        print(
            f"\rEpoch {epoch} "
            f"case {slot + 1}/"
            f"{TRAIN_SLOTS_PER_EPOCH} "
            f"study={study_id} "
            f"series={series_id}",
            end="",
            flush=True,
        )

        try:

            image, points, geometry = (
                load_case(
                    study_id,
                    series_id,
                    case["points"],
                )
            )

        except Exception as exc:

            print(
                f"\nSkipping case "
                f"{study_id}/{series_id}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            continue

        background_coords = (
            sample_background(
                points,
                MODEL_SHAPE,
                BACKGROUND_SAMPLES,
            )
        )

        tensor = torch.from_numpy(
            image
        ).float()

        tensor = tensor.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        tensor = tensor.to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            "cuda",
            enabled=(
                DEVICE.type
                == "cuda"
            ),
        ):

            logits = model(
                tensor
            )

            if isinstance(
                logits,
                (tuple, list),
            ):
                logits = logits[0]

            (
                loss,
                disease_loss,
                background_loss,
            ) = point_supervision_loss(
                logits,
                points,
                background_coords,
                class_weights,
            )

            loss_for_backward = (
                loss
                / GRAD_ACCUMULATION
            )

        scaler.scale(
            loss_for_backward
        ).backward()

        if (
            (slot + 1)
            % GRAD_ACCUMULATION
            == 0
        ):

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        running_total.append(
            float(
                loss.detach().cpu()
            )
        )

        running_disease.append(
            float(
                disease_loss.cpu()
            )
        )

        running_background.append(
            float(
                background_loss.cpu()
            )
        )

    print()

    return {
        "mean_total_loss": float(
            np.mean(
                running_total
            )
        )
        if running_total
        else None,
        "mean_disease_loss": float(
            np.mean(
                running_disease
            )
        )
        if running_disease
        else None,
        "mean_background_loss": float(
            np.mean(
                running_background
            )
        )
        if running_background
        else None,
        "unique_studies": len(
            used_studies
        ),
    }


# ============================================================
# 28. VALIDATION
# ============================================================

def validate(
    model,
    validation_cases,
):

    all_results = []

    foreground_ratios = []

    processed = 0

    for case in validation_cases:

        study_id = case[
            "study_id"
        ]

        series_id = case[
            "series_id"
        ]

        try:

            image, points, geometry = (
                load_case(
                    study_id,
                    series_id,
                    case["points"],
                )
            )

            rows, foreground = (
                evaluate_case(
                    model,
                    image,
                    points,
                )
            )

            for row in rows:

                row[
                    "study_id"
                ] = study_id

                row[
                    "series_id"
                ] = series_id

                all_results.append(
                    row
                )

            foreground_ratios.append(
                foreground
            )

            processed += 1

        except Exception as exc:

            print(
                f"\nValidation failure "
                f"{study_id}/{series_id}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    results_df = pd.DataFrame(
        all_results
    )

    if len(results_df) == 0:

        return {
            "processed_cases": processed,
            "points": 0,
            "overall_accuracy": 0.0,
            "macro_disease_accuracy": 0.0,
            "mean_true_probability": 0.0,
            "hit_rate_0_50": 0.0,
            "mean_foreground_ratio": (
                float(
                    np.mean(
                        foreground_ratios
                    )
                )
                if foreground_ratios
                else 0.0
            ),
        }, results_df

    disease_accuracy = (
        results_df[
            results_df[
                "true_class"
            ] > 0
        ]
        .groupby(
            "true_class"
        )[
            "correct"
        ]
        .mean()
    )

    summary = {
        "processed_cases": int(
            processed
        ),
        "points": int(
            len(results_df)
        ),
        "overall_accuracy": float(
            results_df[
                "correct"
            ].mean()
        ),
        "macro_disease_accuracy": float(
            disease_accuracy.mean()
        )
        if len(
            disease_accuracy
        )
        else 0.0,
        "mean_true_probability": float(
            results_df[
                "true_probability"
            ].mean()
        ),
        "hit_rate_0_50": float(
            (
                results_df[
                    "true_probability"
                ]
                >= 0.50
            ).mean()
        ),
        "mean_foreground_ratio": float(
            np.mean(
                foreground_ratios
            )
        )
        if foreground_ratios
        else 0.0,
    }

    # Disease-wise.
    for class_id in range(
        1,
        NUM_CLASSES,
    ):

        subset = results_df[
            results_df[
                "true_class"
            ]
            == class_id
        ]

        if len(subset) == 0:
            continue

        summary[
            f"class_{class_id}_accuracy"
        ] = float(
            subset[
                "correct"
            ].mean()
        )

        summary[
            f"class_{class_id}_probability"
        ] = float(
            subset[
                "true_probability"
            ].mean()
        )

        summary[
            f"class_{class_id}_count"
        ] = int(
            len(subset)
        )

    return (
        summary,
        results_df,
    )


# ============================================================
# 29. MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "PART 2.20B"
    )
    print(
        "GEOMETRY-CORRECTED "
        "POINT-SUPERVISED TRAINING"
    )
    print("=" * 78)

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Model shape: {MODEL_SHAPE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Training slots/epoch: "
        f"{TRAIN_SLOTS_PER_EPOCH}"
    )

    print(
        f"Validation cases: "
        f"{VALIDATION_CASES}"
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    print(
        "\nLoading Part 2.13 manifest..."
    )

    manifest = load_manifest()

    print(
        f"Manifest rows: "
        f"{len(manifest)}"
    )

    # --------------------------------------------------------
    # Validation selection
    # --------------------------------------------------------

    selected = (
        select_validation_series(
            manifest
        )
    )

    print(
        f"Validation series: "
        f"{len(selected)}"
    )

    validation_keys = {
        (
            str(row["study_id"]),
            str(row["series_id"]),
        )
        for _, row in selected.iterrows()
    }

    # --------------------------------------------------------
    # IMPORTANT:
    # Training cases exclude selected validation series.
    # --------------------------------------------------------

    training_manifest = manifest[
        ~manifest.apply(
            lambda row:
            (
                str(
                    row["study_id"]
                ),
                str(
                    row["series_id"]
                ),
            )
            in validation_keys,
            axis=1,
        )
    ].copy()

    training_cases = build_case_index(
        training_manifest
    )

    validation_manifest = manifest[
        manifest.apply(
            lambda row:
            (
                str(
                    row["study_id"]
                ),
                str(
                    row["series_id"]
                ),
            )
            in validation_keys,
            axis=1,
        )
    ].copy()

    validation_cases = (
        build_case_index(
            validation_manifest
        )
    )

    # Exactly selected validation cases.
    validation_cases = [
        case
        for case in validation_cases
        if (
            case["study_id"],
            case["series_id"],
        )
        in validation_keys
    ]

    print(
        f"Training cases available: "
        f"{len(training_cases)}"
    )

    print(
        f"Validation cases: "
        f"{len(validation_cases)}"
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    counts, weights = (
        compute_class_weights(
            training_manifest
        )
    )

    print(
        "\nPoint-class weights:"
    )

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"  {class_id}: "
            f"{CLASS_NAMES[class_id]} "
            f"count={int(counts[class_id])} "
            f"weight={weights[class_id]:.6f}"
        )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nBuilding Swin-UNETR..."
    )

    model = build_model()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Protected Part104 initialization
    # --------------------------------------------------------

    print(
        "\nLoading protected Part104 "
        "initialization..."
    )

    model = load_part104(
        model
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = GradScaler(
        "cuda",
        enabled=(
            DEVICE.type
            == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "part": "2.20B",
        "seed": SEED,
        "model_shape": list(
            MODEL_SHAPE
        ),
        "num_classes": NUM_CLASSES,
        "epochs": EPOCHS,
        "train_slots_per_epoch": (
            TRAIN_SLOTS_PER_EPOCH
        ),
        "validation_cases": (
            VALIDATION_CASES
        ),
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": (
            GRAD_ACCUMULATION
        ),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "background_weight": (
            BACKGROUND_WEIGHT
        ),
        "background_samples": (
            BACKGROUND_SAMPLES
        ),
        "point_radius": POINT_RADIUS,
        "device": str(DEVICE),
        "initialization_checkpoint": str(
            PART104_CHECKPOINT
        ),
        "training_target": (
            "point_supervision"
        ),
        "physical_space_correction": True,
        "manual_voxel_ground_truth": False,
        "dashboard_integration": False,
    }

    with open(
        REPORT_DIR
        / "part220b_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    epoch_records = []

    best_accuracy = -1.0

    best_epoch = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print("\n")
        print("=" * 78)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 78)

        train_metrics = train_epoch(
            model,
            optimizer,
            scaler,
            training_cases,
            class_weights,
            epoch,
        )

        validation_summary, validation_df = (
            validate(
                model,
                validation_cases,
            )
        )

        current_lr = optimizer.param_groups[
            0
        ][
            "lr"
        ]

        scheduler.step()

        record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            **train_metrics,
            **validation_summary,
        }

        epoch_records.append(
            record
        )

        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        metrics_df = pd.DataFrame(
            epoch_records
        )

        metrics_df.to_csv(
            METRICS_DIR
            / "part220b_epoch_metrics.csv",
            index=False,
        )

        validation_df.to_csv(
            METRICS_DIR
            / f"part220b_epoch_{epoch:02d}_validation_points.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"part220b_epoch_{epoch:02d}.pth"
        )

        torch.save(
            {
                "part": "2.20B",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "validation_summary": validation_summary,
                "config": config,
            },
            checkpoint_path,
        )

        # ----------------------------------------------------
        # Best checkpoint based on point accuracy
        # ----------------------------------------------------

        accuracy = float(
            validation_summary[
                "overall_accuracy"
            ]
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_epoch = epoch

            best_path = (
                CHECKPOINT_DIR
                / "part220b_best_point_checkpoint.pth"
            )

            torch.save(
                {
                    "part": "2.20B",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "validation_summary": validation_summary,
                    "config": config,
                },
                best_path,
            )

        # ----------------------------------------------------
        # Print epoch summary
        # ----------------------------------------------------

        print(
            "\nEpoch summary:"
        )

        print(
            f"  Train loss: "
            f"{train_metrics['mean_total_loss']:.6f}"
        )

        print(
            f"  Disease loss: "
            f"{train_metrics['mean_disease_loss']:.6f}"
        )

        print(
            f"  Background loss: "
            f"{train_metrics['mean_background_loss']:.6f}"
        )

        print(
            f"  Validation points: "
            f"{validation_summary['points']}"
        )

        print(
            f"  Overall point accuracy: "
            f"{validation_summary['overall_accuracy']:.6f}"
        )

        print(
            f"  Macro disease accuracy: "
            f"{validation_summary['macro_disease_accuracy']:.6f}"
        )

        print(
            f"  Mean true-class probability: "
            f"{validation_summary['mean_true_probability']:.6f}"
        )

        print(
            f"  Hit rate >= 0.50: "
            f"{validation_summary['hit_rate_0_50']:.6f}"
        )

        print(
            f"  Mean foreground ratio: "
            f"{validation_summary['mean_foreground_ratio']:.6f}"
        )

        for class_id in range(
            1,
            NUM_CLASSES,
        ):

            key = (
                f"class_{class_id}_accuracy"
            )

            if key in validation_summary:

                print(
                    f"  {CLASS_NAMES[class_id]}: "
                    f"{validation_summary[key]:.6f}"
                )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    final_summary = {
        "part": "2.20B",
        "status": "COMPLETE",
        "best_epoch": best_epoch,
        "best_point_accuracy": best_accuracy,
        "epochs": epoch_records,
        "training_cases": len(
            training_cases
        ),
        "validation_cases": len(
            validation_cases
        ),
        "validation_study_overlap": 0,
        "initialization": str(
            PART104_CHECKPOINT
        ),
        "manual_voxel_segmentation_ground_truth": False,
        "training_type": (
            "geometry-corrected "
            "point-supervised"
        ),
        "physical_space_correction": True,
        "dashboard_integrated": False,
        "part104_modified": False,
        "part216_modified": False,
        "part218_modified": False,
    }

    with open(
        REPORT_DIR
        / "part220b_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_summary,
            f,
            indent=2,
        )

    report_lines = [
        "PART 2.20B",
        "GEOMETRY-CORRECTED POINT-SUPERVISED TRAINING",
        "=" * 70,
        "",
        f"Best epoch: {best_epoch}",
        f"Best point accuracy: {best_accuracy:.6f}",
        "",
        "Training representation:",
        "Patient physical-space canonical grid.",
        "",
        "Target:",
        "RSNA point/localization annotations.",
        "",
        "Manual voxel segmentation ground truth:",
        "NO",
        "",
        "Part104 modified:",
        "NO",
        "",
        "Dashboard integrated:",
        "NO",
        "",
        "Interpretation:",
        "This experiment evaluates whether correcting "
        "DICOM acquisition orientation before "
        "point-supervised training improves "
        "disease localization/classification.",
        "",
        "Point accuracy must not be interpreted as "
        "voxel-wise segmentation Dice.",
    ]

    with open(
        REPORT_DIR
        / "part220b_report.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(
                report_lines
            )
        )

    print("\n")
    print("=" * 78)
    print(
        "PART 2.20B COMPLETE"
    )
    print("=" * 78)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best point accuracy: "
        f"{best_accuracy:.6f}"
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nSafety:"
    )

    print(
        "  Part104 modified: NO"
    )

    print(
        "  Part216 modified: NO"
    )

    print(
        "  Part218 modified: NO"
    )

    print(
        "  Dashboard modified: NO"
    )

    print(
        "  Manual voxel masks fabricated: NO"
    )

    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()