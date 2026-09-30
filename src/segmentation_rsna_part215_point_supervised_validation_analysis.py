"""
PART 2.15 — RSNA POINT-SUPERVISED VALIDATION & PER-DISEASE ANALYSIS
===================================================================

Purpose
-------
Evaluate the Part 2.14 point-supervised Swin-UNETR pilot checkpoint without
performing any additional training.

This script evaluates the same deterministic study-level validation split
used by Part 2.14 and reports:

1. Overall point-class accuracy
2. Per-disease point-class accuracy
3. Per-level point-class accuracy
4. Confusion matrix at RSNA annotation points
5. Correct-class probability at annotation points
6. Probability hit rate >= 0.50
7. Point localization diagnostic distance
8. Predicted foreground voxel volume by disease
9. False-positive diagnostic counts at annotated points
10. Representative qualitative visualizations
11. JSON/CSV reports

SCIENTIFIC CONTRACT
-------------------
The RSNA labels are point/localization annotations, NOT manual voxel-wise
segmentation masks.

Therefore this script DOES NOT report Dice as clinical segmentation accuracy.
It reports point-supervision metrics and diagnostic predicted-region statistics.

No model weights are modified.
No Part104 checkpoint is modified.
No dashboard is changed.
"""

from __future__ import annotations

import gc
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import SwinUNETR


# =============================================================================
# PATHS
# =============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

PART11_PATH = (
    SRC_DIR
    / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part213_point_supervision_manifest"
    / "part213_point_manifest.csv"
)

PART214_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part214_point_supervised_training"
    / "checkpoints"
    / "part214_epoch_03.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part215_point_supervised_validation_analysis"
)

METRICS_DIR = OUTPUT_DIR / "metrics"
VIS_DIR = OUTPUT_DIR / "visualizations"
REPORT_DIR = OUTPUT_DIR / "reports"


# =============================================================================
# LOCKED MODEL / DATA CONTRACT
# =============================================================================

PATCH_SIZE = (64, 96, 96)
IN_CHANNELS = 1
NUM_CLASSES = 6
FEATURE_SIZE = 12

SEED = 42
VAL_SERIES_LIMIT = 25
TRAIN_FRACTION = 0.80

# Visualization and diagnostic thresholds.
PROBABILITY_THRESHOLD = 0.50
LOCAL_SEARCH_RADIUS = 2

LABELS = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

SHORT_LABELS = {
    0: "Background",
    1: "Spinal Canal",
    2: "Left Foraminal",
    3: "Right Foraminal",
    4: "Left Subarticular",
    5: "Right Subarticular",
}

LEVELS = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def section(title: str) -> None:
    print()
    print("=" * 82)
    print(title)
    print("=" * 82)


def clean_id(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        if float(value).is_integer():
            return str(int(value))

    return str(value).strip()


def safe_float(value: Any) -> float | None:
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return str(lower[candidate.lower()])

    if required:
        raise KeyError(
            f"Could not find any of these columns: {list(candidates)}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def resolve_manifest_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {
        "study_id": find_column(df, ["study_id", "study"]),
        "series_id": find_column(df, ["series_id", "series"]),
        "class_id": find_column(
            df,
            ["class_id", "disease_id", "label"],
        ),
        "class_name": find_column(
            df,
            ["class_name", "condition"],
            required=False,
        ),
        "level": find_column(
            df,
            ["level", "level_name"],
            required=False,
        ),
        "model_z": find_column(
            df,
            ["model_z", "model_slice", "grid_z"],
        ),
        "model_y": find_column(
            df,
            ["model_y", "grid_y"],
        ),
        "model_x": find_column(
            df,
            ["model_x", "grid_x"],
        ),
        "series_description": find_column(
            df,
            ["series_description", "series_desc"],
            required=False,
        ),
    }


# =============================================================================
# PART 11 IMPORT
# =============================================================================

def load_part11():
    if not PART11_PATH.exists():
        raise FileNotFoundError(
            f"Part 11 file not found:\n{PART11_PATH}"
        )

    spec = spec_from_file_location(
        "segmentation_rsna_part11_for_part215",
        PART11_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not create Part 11 import specification."
        )

    module = module_from_spec(spec)
    sys.modules["segmentation_rsna_part11_for_part215"] = module
    spec.loader.exec_module(module)

    required = [
        "resolve_series_dir",
        "read_dicom_series_robust",
        "resize_3d",
    ]

    missing = [
        name for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise AttributeError(
            "Part 11 is missing required API: "
            + ", ".join(missing)
        )

    return module


# =============================================================================
# MANIFEST / SPLIT
# =============================================================================

def build_point_records(
    manifest: pd.DataFrame,
    columns: Dict[str, str],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:

    groups: Dict[
        Tuple[str, str],
        List[Dict[str, Any]]
    ] = defaultdict(list)

    for _, row in manifest.iterrows():
        study_id = clean_id(row[columns["study_id"]])
        series_id = clean_id(row[columns["series_id"]])

        if not study_id or not series_id:
            continue

        class_id = int(round(float(row[columns["class_id"]])))

        if class_id < 1 or class_id >= NUM_CLASSES:
            raise ValueError(
                f"Invalid class_id={class_id}; expected 1..5."
            )

        z = int(
            np.clip(
                round(float(row[columns["model_z"]])),
                0,
                PATCH_SIZE[0] - 1,
            )
        )
        y = int(
            np.clip(
                round(float(row[columns["model_y"]])),
                0,
                PATCH_SIZE[1] - 1,
            )
        )
        x = int(
            np.clip(
                round(float(row[columns["model_x"]])),
                0,
                PATCH_SIZE[2] - 1,
            )
        )

        level = ""
        if columns["level"]:
            level = str(row[columns["level"]]).strip()

        series_description = ""
        if columns["series_description"]:
            series_description = str(
                row[columns["series_description"]]
            ).strip()

        groups[(study_id, series_id)].append(
            {
                "class_id": class_id,
                "level": level,
                "z": z,
                "y": y,
                "x": x,
                "series_description": series_description,
            }
        )

    return dict(groups)


def split_series_by_study(
    groups: Dict[
        Tuple[str, str],
        List[Dict[str, Any]]
    ],
    seed: int,
    train_fraction: float = TRAIN_FRACTION,
) -> Tuple[
    List[Tuple[str, str]],
    List[Tuple[str, str]],
]:

    studies = sorted(
        {study_id for study_id, _ in groups.keys()}
    )

    rng = random.Random(seed)
    rng.shuffle(studies)

    n_train = int(
        round(len(studies) * train_fraction)
    )

    train_studies = set(studies[:n_train])

    train_series = [
        key
        for key in sorted(groups)
        if key[0] in train_studies
    ]

    val_series = [
        key
        for key in sorted(groups)
        if key[0] not in train_studies
    ]

    return train_series, val_series


# =============================================================================
# IMAGE LOADING / PREPROCESSING
# =============================================================================

def preprocess_image_part11(
    image_native: np.ndarray,
    part11: Any,
) -> torch.Tensor:

    image = np.asarray(
        image_native,
        dtype=np.float32,
    )

    image = part11.resize_3d(
        image,
        PATCH_SIZE,
        is_mask=False,
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    lo = float(np.percentile(image, 1.0))
    hi = float(np.percentile(image, 99.0))

    if hi > lo:
        image = np.clip(image, lo, hi)
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(
            image,
            dtype=np.float32,
        )

    return torch.from_numpy(
        image.astype(np.float32)
    ).unsqueeze(0).contiguous()


def load_image_case(
    study_id: str,
    series_id: str,
    part11: Any,
) -> Tuple[torch.Tensor, np.ndarray]:

    row = pd.Series(
        {
            "study_id": study_id,
            "series_id": series_id,
        }
    )

    series_dir = part11.resolve_series_dir(row)

    image_native, _, _ = (
        part11.read_dicom_series_robust(series_dir)
    )

    image_native = np.asarray(
        image_native,
        dtype=np.float32,
    )

    image_tensor = preprocess_image_part11(
        image_native,
        part11,
    )

    return image_tensor, image_native


# =============================================================================
# MODEL
# =============================================================================

def create_model(device: torch.device) -> torch.nn.Module:

    model = SwinUNETR(
        spatial_dims=3,
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        use_checkpoint=False,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: {parameter_count:,}"
    )

    return model


def load_part214_checkpoint(
    model: torch.nn.Module,
) -> None:

    checkpoint = torch.load(
        PART214_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get(
                "state_dict",
                checkpoint,
            ),
        )
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "Part214 checkpoint does not contain a state dictionary."
        )

    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        cleaned[new_key] = value

    incompatible = model.load_state_dict(
        cleaned,
        strict=False,
    )

    print(
        f"Missing keys    : {len(incompatible.missing_keys)}"
    )
    print(
        f"Unexpected keys : {len(incompatible.unexpected_keys)}"
    )

    if incompatible.missing_keys:
        raise RuntimeError(
            "Part214 checkpoint is not fully compatible with "
            "the canonical Swin-UNETR architecture."
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Part214 checkpoint contains unexpected model keys."
        )

    del checkpoint
    del state_dict
    del cleaned


# =============================================================================
# POINT-LEVEL METRICS
# =============================================================================

def nearest_localization_distance(
    probability_volume: np.ndarray,
    z: int,
    y: int,
    x: int,
    threshold: float = PROBABILITY_THRESHOLD,
) -> float | None:

    z0 = max(0, z - LOCAL_SEARCH_RADIUS)
    z1 = min(PATCH_SIZE[0], z + LOCAL_SEARCH_RADIUS + 1)

    y0 = max(0, y - LOCAL_SEARCH_RADIUS)
    y1 = min(PATCH_SIZE[1], y + LOCAL_SEARCH_RADIUS + 1)

    x0 = max(0, x - LOCAL_SEARCH_RADIUS)
    x1 = min(PATCH_SIZE[2], x + LOCAL_SEARCH_RADIUS + 1)

    local = probability_volume[
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    positions = np.argwhere(
        local >= threshold
    )

    if len(positions) == 0:
        return None

    target = np.array(
        [
            z - z0,
            y - y0,
            x - x0,
        ],
        dtype=np.float32,
    )

    positions = positions.astype(
        np.float32
    )

    distances = np.sqrt(
        np.sum(
            (positions - target) ** 2,
            axis=1,
        )
    )

    return float(distances.min())


def evaluate_case(
    model: torch.nn.Module,
    image: torch.Tensor,
    points: List[Dict[str, Any]],
    device: torch.device,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
    np.ndarray,
]:

    x = image.unsqueeze(0).to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0].detach().cpu().numpy()

    predicted_labels = np.argmax(
        probabilities,
        axis=0,
    ).astype(np.uint8)

    predicted_foreground = (
        predicted_labels > 0
    )

    predicted_foreground_voxels = int(
        predicted_foreground.sum()
    )

    predicted_foreground_ratio = float(
        predicted_foreground.mean()
    )

    point_rows = []

    for point in points:
        class_id = int(point["class_id"])
        z = int(point["z"])
        y = int(point["y"])
        xx = int(point["x"])

        point_probabilities = probabilities[
            :,
            z,
            y,
            xx,
        ]

        predicted_class = int(
            np.argmax(point_probabilities)
        )

        correct_probability = float(
            point_probabilities[class_id]
        )

        predicted_probability = float(
            point_probabilities[predicted_class]
        )

        distance = nearest_localization_distance(
            probabilities[class_id],
            z,
            y,
            xx,
        )

        point_rows.append(
            {
                "class_id": class_id,
                "class_name": LABELS[class_id],
                "level": point["level"],
                "z": z,
                "y": y,
                "x": xx,
                "predicted_class_id": predicted_class,
                "predicted_class_name": LABELS.get(
                    predicted_class,
                    str(predicted_class),
                ),
                "correct_probability": correct_probability,
                "predicted_probability": predicted_probability,
                "correct_class_hit_050": int(
                    correct_probability >= 0.50
                ),
                "point_class_correct": int(
                    predicted_class == class_id
                ),
                "localization_distance_voxels": distance,
            }
        )

    class_foreground_voxels = {}

    for class_id in range(1, NUM_CLASSES):
        count = int(
            (predicted_labels == class_id).sum()
        )

        class_foreground_voxels[
            str(class_id)
        ] = count

    case_summary = {
        "predicted_foreground_voxels": predicted_foreground_voxels,
        "predicted_foreground_ratio": predicted_foreground_ratio,
        "predicted_class_voxels": class_foreground_voxels,
    }

    return (
        point_rows,
        case_summary,
        predicted_labels,
    )


# =============================================================================
# TABLE AGGREGATION
# =============================================================================

def aggregate_disease_metrics(
    point_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for class_id in range(1, NUM_CLASSES):
        subset = point_df[
            point_df["class_id"] == class_id
        ]

        if subset.empty:
            continue

        distances = subset[
            "localization_distance_voxels"
        ].dropna()

        rows.append(
            {
                "class_id": class_id,
                "disease": LABELS[class_id],
                "points": int(len(subset)),
                "correct_points": int(
                    subset["point_class_correct"].sum()
                ),
                "accuracy": float(
                    subset["point_class_correct"].mean()
                ),
                "mean_correct_probability": float(
                    subset["correct_probability"].mean()
                ),
                "median_correct_probability": float(
                    subset["correct_probability"].median()
                ),
                "hit_rate_probability_0.50": float(
                    subset["correct_class_hit_050"].mean()
                ),
                "localization_cases": int(
                    len(distances)
                ),
                "mean_localization_distance_voxels": (
                    float(distances.mean())
                    if len(distances)
                    else np.nan
                ),
                "median_localization_distance_voxels": (
                    float(distances.median())
                    if len(distances)
                    else np.nan
                ),
                "predicted_as_background": int(
                    (
                        subset["predicted_class_id"] == 0
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def aggregate_level_metrics(
    point_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for level in LEVELS:
        subset = point_df[
            point_df["level"].astype(str) == level
        ]

        if subset.empty:
            continue

        distances = subset[
            "localization_distance_voxels"
        ].dropna()

        rows.append(
            {
                "level": level,
                "points": int(len(subset)),
                "correct_points": int(
                    subset["point_class_correct"].sum()
                ),
                "accuracy": float(
                    subset["point_class_correct"].mean()
                ),
                "mean_correct_probability": float(
                    subset["correct_probability"].mean()
                ),
                "hit_rate_probability_0.50": float(
                    subset["correct_class_hit_050"].mean()
                ),
                "localization_cases": int(
                    len(distances)
                ),
                "mean_localization_distance_voxels": (
                    float(distances.mean())
                    if len(distances)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_confusion_matrix(
    point_df: pd.DataFrame,
) -> pd.DataFrame:

    matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    for _, row in point_df.iterrows():
        actual = int(row["class_id"])
        predicted = int(row["predicted_class_id"])
        matrix[actual, predicted] += 1

    names = [
        LABELS[i]
        for i in range(NUM_CLASSES)
    ]

    return pd.DataFrame(
        matrix,
        index=[
            f"Actual {name}"
            for name in names
        ],
        columns=[
            f"Predicted {name}"
            for name in names
        ],
    )


def build_false_positive_analysis(
    point_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for predicted_class in range(NUM_CLASSES):
        subset = point_df[
            point_df["predicted_class_id"]
            == predicted_class
        ]

        if subset.empty:
            continue

        actual_distribution = (
            subset["class_id"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        row = {
            "predicted_class_id": predicted_class,
            "predicted_class": LABELS[predicted_class],
            "predicted_point_count": int(
                len(subset)
            ),
            "correct_predictions": int(
                (
                    subset["class_id"]
                    == predicted_class
                ).sum()
            ),
            "incorrect_predictions": int(
                (
                    subset["class_id"]
                    != predicted_class
                ).sum()
            ),
        }

        for actual_class in range(1, NUM_CLASSES):
            row[
                f"actual_class_{actual_class}_count"
            ] = int(
                actual_distribution.get(
                    actual_class,
                    0,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION
# =============================================================================

def save_case_visualization(
    image_tensor: torch.Tensor,
    predicted_labels: np.ndarray,
    points: List[Dict[str, Any]],
    study_id: str,
    series_id: str,
) -> Path:

    volume = (
        image_tensor.squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    if not points:
        z = volume.shape[0] // 2
        y = volume.shape[1] // 2
        x = volume.shape[2] // 2
    else:
        z = int(
            round(
                np.mean(
                    [p["z"] for p in points]
                )
            )
        )
        y = int(
            round(
                np.mean(
                    [p["y"] for p in points]
                )
            )
        )
        x = int(
            round(
                np.mean(
                    [p["x"] for p in points]
                )
            )
        )

    # Three native model-grid views around the annotated region.
    axial = volume[z]
    coronal = volume[:, y, :]
    sagittal = volume[:, :, x]

    pred_axial = predicted_labels[z]
    pred_coronal = predicted_labels[:, y, :]
    pred_sagittal = predicted_labels[:, :, x]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 8),
    )

    views = [
        (axial, pred_axial, "Axial"),
        (coronal, pred_coronal, "Coronal"),
        (sagittal, pred_sagittal, "Sagittal"),
    ]

    for col, (img, pred, title) in enumerate(views):
        axes[0, col].imshow(
            img,
            cmap="gray",
        )
        axes[0, col].set_title(
            f"{title} MRI"
        )
        axes[0, col].axis("off")

        axes[1, col].imshow(
            img,
            cmap="gray",
        )
        axes[1, col].imshow(
            np.ma.masked_where(
                pred == 0,
                pred,
            ),
            alpha=0.45,
            interpolation="nearest",
            cmap="tab10",
            vmin=0,
            vmax=NUM_CLASSES - 1,
        )
        axes[1, col].set_title(
            f"{title} predicted labels"
        )
        axes[1, col].axis("off")

    fig.suptitle(
        f"Part 2.15 — Study {study_id} / Series {series_id}\n"
        f"Point-supervised diagnostic visualization "
        f"(model grid {PATCH_SIZE})"
    )

    fig.tight_layout()

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        VIS_DIR
        / f"{study_id}_{series_id}_validation.png"
    )

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    seed_everything(SEED)

    for directory in [
        OUTPUT_DIR,
        METRICS_DIR,
        VIS_DIR,
        REPORT_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    section(
        "PART 2.15 — POINT-SUPERVISED VALIDATION & "
        "PER-DISEASE ANALYSIS"
    )

    print(
        f"Project root       : {PROJECT_ROOT}"
    )
    print(
        f"Manifest           : {MANIFEST_PATH}"
    )
    print(
        f"Checkpoint         : {PART214_CHECKPOINT}"
    )
    print(
        f"Output             : {OUTPUT_DIR}"
    )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Part 2.13 manifest not found:\n{MANIFEST_PATH}"
        )

    if not PART214_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Part 2.14 Epoch 3 checkpoint not found:\n"
            f"{PART214_CHECKPOINT}"
        )

    section("LOADING PART 2.13 MANIFEST")

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Manifest rows    : {len(manifest)}"
    )
    print(
        f"Manifest columns : {len(manifest.columns)}"
    )

    columns = resolve_manifest_columns(
        manifest
    )

    print("Resolved columns:")
    for key, value in columns.items():
        print(
            f"  {key:20s} -> {value}"
        )

    groups = build_point_records(
        manifest,
        columns,
    )

    total_points = sum(
        len(v)
        for v in groups.values()
    )

    print(
        f"Annotated series : {len(groups)}"
    )
    print(
        f"Mapped points    : {total_points}"
    )

    section("REPRODUCING PART 2.14 STUDY-LEVEL SPLIT")

    train_series, val_series = (
        split_series_by_study(
            groups,
            SEED,
            TRAIN_FRACTION,
        )
    )

    val_cases = val_series[
        :VAL_SERIES_LIMIT
    ]

    train_studies = {
        study_id
        for study_id, _ in train_series
    }

    val_studies = {
        study_id
        for study_id, _ in val_cases
    }

    overlap = train_studies.intersection(
        val_studies
    )

    if overlap:
        raise RuntimeError(
            "DATA LEAKAGE: training and validation studies overlap."
        )

    print(
        f"Available validation series : {len(val_series)}"
    )
    print(
        f"Evaluation validation cases : {len(val_cases)}"
    )
    print(
        f"Study overlap               : {len(overlap)}"
    )

    if len(val_cases) != VAL_SERIES_LIMIT:
        raise RuntimeError(
            "The expected 25-case validation set could not be reproduced."
        )

    section("LOADING PART 11")

    part11 = load_part11()

    print(
        "✓ Robust mixed-DICOM loader available."
    )
    print(
        "✓ Part 11 preprocessing convention loaded."
    )
    print(
        "✓ No pseudo-mask targets are used."
    )

    section("DEVICE")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device : {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU    : {torch.cuda.get_device_name(0)}"
        )

    section("CREATING MODEL")

    model = create_model(
        device
    )

    section("LOADING PART 2.14 EPOCH 3")

    load_part214_checkpoint(
        model
    )

    model.eval()

    print(
        "✓ Part 2.14 Epoch 3 loaded."
    )
    print(
        "✓ Evaluation only; weights will not be modified."
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    section(
        f"EVALUATING {len(val_cases)} VALIDATION SERIES"
    )

    point_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []

    case_errors: List[Dict[str, Any]] = []
    visualization_paths: List[str] = []

    start_time = time.time()

    for index, (study_id, series_id) in enumerate(
        val_cases,
        start=1,
    ):

        print(
            f"\n[{index}/{len(val_cases)}] "
            f"{study_id}/{series_id}"
        )

        try:
            image, _ = load_image_case(
                study_id,
                series_id,
                part11,
            )

            points = groups[
                (study_id, series_id)
            ]

            case_point_rows, case_summary, predicted_labels = (
                evaluate_case(
                    model=model,
                    image=image,
                    points=points,
                    device=device,
                )
            )

            for row in case_point_rows:
                row["study_id"] = study_id
                row["series_id"] = series_id
                row["series_description"] = (
                    points[0].get(
                        "series_description",
                        "",
                    )
                    if points
                    else ""
                )

            point_rows.extend(
                case_point_rows
            )

            case_row = {
                "study_id": study_id,
                "series_id": series_id,
                "series_description": (
                    points[0].get(
                        "series_description",
                        "",
                    )
                    if points
                    else ""
                ),
                "annotation_points": len(points),
                **case_summary,
            }

            case_rows.append(
                case_row
            )

            # Save every successful validation case visualization.
            visualization_path = (
                save_case_visualization(
                    image_tensor=image,
                    predicted_labels=predicted_labels,
                    points=points,
                    study_id=study_id,
                    series_id=series_id,
                )
            )

            visualization_paths.append(
                str(visualization_path)
            )

            print(
                f"  points                     : {len(points)}"
            )
            print(
                f"  predicted foreground voxels: "
                f"{case_summary['predicted_foreground_voxels']}"
            )
            print(
                f"  predicted foreground ratio : "
                f"{case_summary['predicted_foreground_ratio']:.6f}"
            )

            del image
            del predicted_labels

            if device.type == "cuda":
                torch.cuda.empty_cache()

            gc.collect()

        except Exception as exc:

            error_record = {
                "study_id": study_id,
                "series_id": series_id,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

            case_errors.append(
                error_record
            )

            print(
                f"  [WARNING] case failed: "
                f"{error_record['error']}"
            )

    elapsed = (
        time.time() - start_time
    )

    if not point_rows:
        raise RuntimeError(
            "No validation points were successfully evaluated."
        )

    # -------------------------------------------------------------------------
    # DataFrames
    # -------------------------------------------------------------------------

    point_df = pd.DataFrame(
        point_rows
    )

    case_df = pd.DataFrame(
        case_rows
    )

    disease_df = aggregate_disease_metrics(
        point_df
    )

    level_df = aggregate_level_metrics(
        point_df
    )

    confusion_df = build_confusion_matrix(
        point_df
    )

    false_positive_df = (
        build_false_positive_analysis(
            point_df
        )
    )

    # -------------------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------------------

    total_evaluated_points = len(
        point_df
    )

    correct_points = int(
        point_df["point_class_correct"].sum()
    )

    point_accuracy = float(
        correct_points
        / total_evaluated_points
    )

    mean_probability = float(
        point_df["correct_probability"].mean()
    )

    median_probability = float(
        point_df["correct_probability"].median()
    )

    probability_hit_rate = float(
        point_df["correct_class_hit_050"].mean()
    )

    distance_series = point_df[
        "localization_distance_voxels"
    ].dropna()

    mean_distance = (
        float(distance_series.mean())
        if len(distance_series)
        else None
    )

    median_distance = (
        float(distance_series.median())
        if len(distance_series)
        else None
    )

    # Macro average over the five disease classes.
    disease_accuracy_macro = (
        float(
            disease_df["accuracy"].mean()
        )
        if not disease_df.empty
        else None
    )

    disease_probability_macro = (
        float(
            disease_df[
                "mean_correct_probability"
            ].mean()
        )
        if not disease_df.empty
        else None
    )

    if not case_df.empty:
        mean_foreground_voxels = float(
            case_df[
                "predicted_foreground_voxels"
            ].mean()
        )

        median_foreground_voxels = float(
            case_df[
                "predicted_foreground_voxels"
            ].median()
        )

        mean_foreground_ratio = float(
            case_df[
                "predicted_foreground_ratio"
            ].mean()
        )
    else:
        mean_foreground_voxels = None
        median_foreground_voxels = None
        mean_foreground_ratio = None

    overall_df = pd.DataFrame(
        [
            {
                "validation_cases_requested": len(
                    val_cases
                ),
                "validation_cases_successful": len(
                    case_df
                ),
                "validation_cases_failed": len(
                    case_errors
                ),
                "points_evaluated": (
                    total_evaluated_points
                ),
                "correct_points": correct_points,
                "point_class_accuracy": point_accuracy,
                "macro_disease_accuracy": (
                    disease_accuracy_macro
                ),
                "mean_correct_class_probability": (
                    mean_probability
                ),
                "median_correct_class_probability": (
                    median_probability
                ),
                "probability_hit_rate_0.50": (
                    probability_hit_rate
                ),
                "mean_localization_distance_voxels": (
                    mean_distance
                ),
                "median_localization_distance_voxels": (
                    median_distance
                ),
                "localization_points_within_search": (
                    len(distance_series)
                ),
                "mean_predicted_foreground_voxels": (
                    mean_foreground_voxels
                ),
                "median_predicted_foreground_voxels": (
                    median_foreground_voxels
                ),
                "mean_predicted_foreground_ratio": (
                    mean_foreground_ratio
                ),
                "macro_disease_correct_probability": (
                    disease_probability_macro
                ),
                "elapsed_seconds": elapsed,
            }
        ]
    )

    # -------------------------------------------------------------------------
    # Save CSV outputs
    # -------------------------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_df.to_csv(
        METRICS_DIR / "overall_metrics.csv",
        index=False,
    )

    disease_df.to_csv(
        METRICS_DIR / "disease_metrics.csv",
        index=False,
    )

    level_df.to_csv(
        METRICS_DIR / "level_metrics.csv",
        index=False,
    )

    point_df.to_csv(
        METRICS_DIR / "point_predictions.csv",
        index=False,
    )

    case_df.to_csv(
        METRICS_DIR / "case_foreground_metrics.csv",
        index=False,
    )

    confusion_df.to_csv(
        METRICS_DIR / "confusion_matrix.csv",
    )

    false_positive_df.to_csv(
        METRICS_DIR / "false_positive_analysis.csv",
        index=False,
    )

    if case_errors:
        pd.DataFrame(
            case_errors
        ).to_csv(
            METRICS_DIR / "case_errors.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------

    summary = {
        "part": "2.15",
        "status": (
            "completed"
            if not case_errors
            else "completed_with_case_errors"
        ),
        "experiment": (
            "RSNA point-supervised Swin-UNETR "
            "validation and per-disease analysis"
        ),
        "scientific_note": (
            "RSNA annotations are point/localization annotations, "
            "not manual voxel-wise segmentation masks. "
            "The reported metrics therefore evaluate point-level "
            "classification/localization behavior and predicted "
            "foreground statistics, not clinical voxel-wise segmentation accuracy."
        ),
        "manifest": str(
            MANIFEST_PATH
        ),
        "checkpoint": str(
            PART214_CHECKPOINT
        ),
        "model": {
            "architecture": "SwinUNETR",
            "spatial_dims": 3,
            "in_channels": IN_CHANNELS,
            "out_channels": NUM_CLASSES,
            "feature_size": FEATURE_SIZE,
            "patch_size": list(PATCH_SIZE),
        },
        "split": {
            "seed": SEED,
            "train_fraction": TRAIN_FRACTION,
            "validation_series_requested": len(
                val_cases
            ),
            "study_overlap": len(overlap),
        },
        "overall": overall_df.iloc[0].to_dict(),
        "disease_metrics": (
            disease_df.to_dict(
                orient="records"
            )
        ),
        "level_metrics": (
            level_df.to_dict(
                orient="records"
            )
        ),
        "case_errors": case_errors,
        "visualizations": visualization_paths,
    }

    # Convert numpy scalars to JSON-safe values.
    def json_safe(value: Any) -> Any:
        if isinstance(
            value,
            (np.integer,),
        ):
            return int(value)

        if isinstance(
            value,
            (np.floating,),
        ):
            return (
                float(value)
                if np.isfinite(value)
                else None
            )

        if pd.isna(value):
            return None

        return value

    def recursively_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                str(k): recursively_safe(v)
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [
                recursively_safe(v)
                for v in obj
            ]

        return json_safe(obj)

    summary = recursively_safe(
        summary
    )

    summary_path = (
        REPORT_DIR
        / "part215_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Human-readable report
    # -------------------------------------------------------------------------

    report_lines = [
        "PART 2.15 — POINT-SUPERVISED VALIDATION "
        "& PER-DISEASE ANALYSIS",
        "",
        "Status: "
        + (
            "COMPLETED"
            if not case_errors
            else "COMPLETED WITH CASE ERRORS"
        ),
        "",
        "Scientific contract:",
        "RSNA annotations are point/localization annotations.",
        "They are not manual voxel-wise segmentation masks.",
        "Therefore Dice is not reported as clinical segmentation accuracy.",
        "",
        f"Validation cases requested: {len(val_cases)}",
        f"Validation cases successful: {len(case_df)}",
        f"Validation cases failed: {len(case_errors)}",
        f"Points evaluated: {total_evaluated_points}",
        f"Point-class accuracy: {point_accuracy:.6f}",
        f"Macro disease accuracy: "
        f"{disease_accuracy_macro:.6f}",
        f"Mean correct-class probability: "
        f"{mean_probability:.6f}",
        f"Median correct-class probability: "
        f"{median_probability:.6f}",
        f"Probability hit rate >= 0.50: "
        f"{probability_hit_rate:.6f}",
        f"Mean localization distance: "
        f"{mean_distance}",
        f"Median localization distance: "
        f"{median_distance}",
        f"Mean predicted foreground voxels: "
        f"{mean_foreground_voxels}",
        f"Mean predicted foreground ratio: "
        f"{mean_foreground_ratio}",
        "",
        "PER-DISEASE:",
    ]

    for _, row in disease_df.iterrows():
        report_lines.append(
            f"  {row['disease']}: "
            f"points={int(row['points'])}, "
            f"accuracy={row['accuracy']:.6f}, "
            f"mean_probability="
            f"{row['mean_correct_probability']:.6f}, "
            f"mean_distance="
            f"{row['mean_localization_distance_voxels']}"
        )

    report_lines.extend(
        [
            "",
            "PER-LEVEL:",
        ]
    )

    for _, row in level_df.iterrows():
        report_lines.append(
            f"  {row['level']}: "
            f"points={int(row['points'])}, "
            f"accuracy={row['accuracy']:.6f}, "
            f"mean_probability="
            f"{row['mean_correct_probability']:.6f}, "
            f"mean_distance="
            f"{row['mean_localization_distance_voxels']}"
        )

    report_lines.extend(
        [
            "",
            "OUTPUT FILES:",
            f"  {METRICS_DIR / 'overall_metrics.csv'}",
            f"  {METRICS_DIR / 'disease_metrics.csv'}",
            f"  {METRICS_DIR / 'level_metrics.csv'}",
            f"  {METRICS_DIR / 'point_predictions.csv'}",
            f"  {METRICS_DIR / 'case_foreground_metrics.csv'}",
            f"  {METRICS_DIR / 'confusion_matrix.csv'}",
            f"  {METRICS_DIR / 'false_positive_analysis.csv'}",
            f"  {summary_path}",
        ]
    )

    if case_errors:
        report_lines.extend(
            [
                "",
                "FAILED CASES:",
            ]
        )

        for error in case_errors:
            report_lines.append(
                f"  {error['study_id']}/"
                f"{error['series_id']}: "
                f"{error['error']}"
            )

    report_path = (
        REPORT_DIR
        / "part215_validation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n".join(report_lines)
        )

    # -------------------------------------------------------------------------
    # Terminal summary
    # -------------------------------------------------------------------------

    section("PART 2.15 RESULTS")

    print(
        f"Validation cases     : {len(case_df)}/{len(val_cases)}"
    )
    print(
        f"Points evaluated     : {total_evaluated_points}"
    )
    print(
        f"Point accuracy       : {point_accuracy:.6f}"
    )
    print(
        f"Macro disease accuracy: "
        f"{disease_accuracy_macro:.6f}"
    )
    print(
        f"Mean point probability: "
        f"{mean_probability:.6f}"
    )
    print(
        f"Hit rate @ 0.50      : "
        f"{probability_hit_rate:.6f}"
    )
    print(
        f"Mean localization distance: "
        f"{mean_distance}"
    )
    print(
        f"Median localization distance: "
        f"{median_distance}"
    )

    print()
    print("Per-disease results:")

    for _, row in disease_df.iterrows():
        print(
            f"  {row['disease']:<35s} "
            f"accuracy={row['accuracy']:.4f} "
            f"prob={row['mean_correct_probability']:.4f}"
        )

    print()
    print("Per-level results:")

    for _, row in level_df.iterrows():
        print(
            f"  {row['level']:<8s} "
            f"accuracy={row['accuracy']:.4f} "
            f"points={int(row['points'])}"
        )

    print()
    print(
        f"Report : {report_path}"
    )
    print(
        f"JSON   : {summary_path}"
    )
    print(
        f"Metrics: {METRICS_DIR}"
    )
    print(
        f"Visuals: {VIS_DIR}"
    )

    print()
    print(
        "IMPORTANT: These results are point-supervision diagnostics, "
        "not clinical voxel-wise segmentation accuracy."
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    gc.collect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPart 2.15 interrupted by user.")
        raise
    except Exception as exc:
        print()
        print("=" * 82)
        print("PART 2.15 FAILED")
        print("=" * 82)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
