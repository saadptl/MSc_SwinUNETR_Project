"""
PART 2.16 — RSNA DISEASE-BALANCED POINT-SUPERVISED SWIN-UNETR TRAINING
======================================================

Purpose
-------
Train a separate Swin-UNETR experiment using the validated RSNA point
annotations produced by Part 2.13.

IMPORTANT SCIENTIFIC CONTRACT
-----------------------------
The RSNA annotations are POINT annotations, not voxel-wise segmentation
masks. This script therefore does NOT fabricate a full segmentation mask.

Supervision consists of:
    1. Positive classification loss at the annotated model-grid point.
    2. A small, explicitly marked set of background-negative voxels sampled
       away from annotated points.

The background term is deliberately down-weighted. Unannotated voxels are
NOT treated as ground-truth disease-free everywhere.

This is a controlled disease-balanced PILOT:
    training slots/epoch : 100
    validation series    : 25
    epochs               : 5

Part 2.16 adds disease-aware sampling and bounded point-class weighting.
The pilot must succeed before scaling the experiment.

Protected checkpoint
--------------------
Part 104 is loaded only as initialization. It is NEVER overwritten.

Output
------
outputs/segmentation/rsna_part216_point_supervised_balanced_training/

No dashboard integration is performed by this script.
No disease overlay is enabled by this script.
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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
TRAIN_IMAGES_DIR = RSNA_ROOT / "train_images"

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

PROTECTED_PART104 = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part104_final_checkpoint_selection"
    / "checkpoints"
    / "final_segmentation_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part216_point_supervised_balanced_training"
)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
VIS_DIR = OUTPUT_DIR / "visualizations"
REPORT_DIR = OUTPUT_DIR / "reports"


# =============================================================================
# LOCKED MODEL / DATA CONTRACT
# =============================================================================

PATCH_SIZE = (64, 96, 96)
IN_CHANNELS = 1
NUM_CLASSES = 6
FEATURE_SIZE = 12

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5

EPOCHS = 5
TRAIN_SERIES_LIMIT = 100
VAL_SERIES_LIMIT = 25

BACKGROUND_SAMPLES_PER_CASE = 128
BACKGROUND_LOSS_WEIGHT = 0.10

SEED = 42
USE_AMP = True

LABELS = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

# Part 2.16 disease-balancing controls.
BALANCE_POWER = 0.50
MIN_CLASS_WEIGHT = 0.50
MAX_CLASS_WEIGHT = 2.50


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# PRINTING
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# PART 11 IMPORT
# =============================================================================

def load_part11():
    if not PART11_PATH.exists():
        raise FileNotFoundError(
            f"Part 11 file not found:\n{PART11_PATH}"
        )

    spec = spec_from_file_location(
        "segmentation_rsna_part11_for_part214",
        PART11_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not create Part 11 import specification.")

    module = module_from_spec(spec)
    sys.modules["segmentation_rsna_part11_for_part214"] = module
    spec.loader.exec_module(module)

    required = [
        "resolve_series_dir",
        "read_dicom_series_robust",
        "resize_3d",
    ]

    missing = [name for name in required if not hasattr(module, name)]

    if missing:
        raise AttributeError(
            "Part 11 is missing required API: "
            + ", ".join(missing)
        )

    return module


# =============================================================================
# MANIFEST COLUMN RESOLUTION
# =============================================================================

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
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def resolve_manifest_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {
        "study_id": find_column(
            df,
            ["study_id", "study"],
        ),
        "series_id": find_column(
            df,
            ["series_id", "series"],
        ),
        "class_id": find_column(
            df,
            ["class_id", "disease_id", "label"],
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
    }


# =============================================================================
# ID / COORDINATE HELPERS
# =============================================================================

def clean_id(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        if float(value).is_integer():
            return str(int(value))

    return str(value).strip()


def clean_class_id(value: Any) -> int:
    value = int(round(float(value)))

    if value < 1 or value >= NUM_CLASSES:
        raise ValueError(
            f"Invalid point class_id={value}. Expected 1..5."
        )

    return value


def clamp_index(value: Any, upper: int) -> int:
    return int(
        np.clip(
            int(round(float(value))),
            0,
            upper - 1,
        )
    )


# =============================================================================
# GROUP MANIFEST INTO SERIES
# =============================================================================

def build_series_groups(
    manifest: pd.DataFrame,
    columns: Dict[str, str],
) -> Dict[Tuple[str, str], List[Tuple[int, int, int, int]]]:
    groups: Dict[
        Tuple[str, str],
        List[Tuple[int, int, int, int]]
    ] = defaultdict(list)

    for _, row in manifest.iterrows():
        study_id = clean_id(row[columns["study_id"]])
        series_id = clean_id(row[columns["series_id"]])

        if not study_id or not series_id:
            continue

        class_id = clean_class_id(
            row[columns["class_id"]]
        )

        z = clamp_index(
            row[columns["model_z"]],
            PATCH_SIZE[0],
        )
        y = clamp_index(
            row[columns["model_y"]],
            PATCH_SIZE[1],
        )
        x = clamp_index(
            row[columns["model_x"]],
            PATCH_SIZE[2],
        )

        groups[(study_id, series_id)].append(
            (class_id, z, y, x)
        )

    return dict(groups)


# =============================================================================
# STUDY-LEVEL SPLIT
# =============================================================================

def split_series_by_study(
    groups: Dict[Tuple[str, str], List[Tuple[int, int, int, int]]],
    seed: int,
    train_fraction: float = 0.80,
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
    val_studies = set(studies[n_train:])

    train_series = [
        key for key in sorted(groups)
        if key[0] in train_studies
    ]

    val_series = [
        key for key in sorted(groups)
        if key[0] in val_studies
    ]

    return train_series, val_series



# =============================================================================
# PART 2.16 — DISEASE-BALANCED TRAINING SCHEDULE
# =============================================================================

def build_balanced_training_schedule(
    candidate_cases,
    groups,
    limit,
    seed,
):
    """
    Disease-aware case sampling.

    Rare disease classes receive greater exposure. Sampling is with
    replacement because this is a controlled pilot; repeated cases are
    repeated observations, not additional patients.
    """
    rng = np.random.default_rng(seed)

    class_counts = {
        class_id: 0 for class_id in range(1, NUM_CLASSES)
    }
    case_classes = {}

    for case in candidate_cases:
        points = groups.get(case, [])
        classes = {
            int(point[0])
            for point in points
            if 1 <= int(point[0]) < NUM_CLASSES
        }
        case_classes[case] = classes

        for point in points:
            class_id = int(point[0])
            if 1 <= class_id < NUM_CLASSES:
                class_counts[class_id] += 1

    raw = {
        class_id: 1.0 / (max(class_counts[class_id], 1) ** BALANCE_POWER)
        for class_id in range(1, NUM_CLASSES)
    }

    mean_raw = float(np.mean(list(raw.values())))
    class_weights = {
        class_id: float(
            np.clip(
                raw[class_id] / mean_raw,
                MIN_CLASS_WEIGHT,
                MAX_CLASS_WEIGHT,
            )
        )
        for class_id in range(1, NUM_CLASSES)
    }

    case_weights = []
    for case in candidate_cases:
        classes = case_classes[case]
        if not classes:
            weight = 0.25
        else:
            weight = float(
                np.mean([class_weights[c] for c in classes])
            )
            if len(classes) > 1:
                weight *= 1.10
        case_weights.append(max(weight, 1e-6))

    probabilities = np.asarray(case_weights, dtype=np.float64)
    probabilities /= probabilities.sum()

    selected = []

    # Guarantee at least one training case for every disease when possible.
    for class_id in range(1, NUM_CLASSES):
        eligible = [
            case for case in candidate_cases
            if class_id in case_classes.get(case, set())
        ]
        if eligible and len(selected) < limit:
            selected.append(
                eligible[int(rng.integers(0, len(eligible)))]
            )

    remaining = max(0, limit - len(selected))
    if remaining:
        indices = rng.choice(
            len(candidate_cases),
            size=remaining,
            replace=True,
            p=probabilities,
        )
        selected.extend(
            candidate_cases[int(i)]
            for i in np.asarray(indices).reshape(-1)
        )

    rng.shuffle(selected)
    return selected, class_counts, class_weights


def build_point_class_weights(
    train_cases,
    groups,
    device,
):
    """Bounded inverse-frequency CE weights for sparse disease points."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    for case in train_cases:
        for class_id, _, _, _ in groups.get(case, []):
            counts[int(class_id)] += 1

    raw = 1.0 / np.sqrt(np.maximum(counts[1:], 1.0))
    raw /= float(np.mean(raw))
    raw = np.clip(
        raw,
        MIN_CLASS_WEIGHT,
        MAX_CLASS_WEIGHT,
    )

    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    weights[1:] = raw.astype(np.float32)

    print("Point-level class weights:")
    for class_id in range(1, NUM_CLASSES):
        print(
            f"  {LABELS[class_id]:35s} "
            f"count={int(counts[class_id]):5d} "
            f"weight={weights[class_id]:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image_part11(
    image_native: np.ndarray,
    part11: Any,
) -> torch.Tensor:
    """
    Reproduce the image portion of Part 11 preprocessing exactly:

        native image
            -> resize to (64,96,96), trilinear, align_corners=False
            -> nan/inf cleanup
            -> 1st/99th percentile clipping
            -> [0,1] normalization

    No pseudo-mask is loaded.
    """
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

    tensor = torch.from_numpy(
        image.astype(np.float32)
    ).unsqueeze(0)

    if tuple(tensor.shape[-3:]) != PATCH_SIZE:
        raise RuntimeError(
            f"Unexpected preprocessed image shape: {tuple(tensor.shape)}"
        )

    return tensor.contiguous()


# =============================================================================
# DICOM CASE LOADING
# =============================================================================

def load_image_case(
    study_id: str,
    series_id: str,
    part11: Any,
) -> torch.Tensor:
    row = pd.Series(
        {
            "study_id": study_id,
            "series_id": series_id,
        }
    )

    series_dir = part11.resolve_series_dir(row)

    image_native, _, dicom_info = (
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

    return image_tensor


# =============================================================================
# POINT LOSS
# =============================================================================

def make_background_samples(
    points: Sequence[Tuple[int, int, int, int]],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select a small number of negative/background voxels.

    Only voxels at least 3 grid units away from all annotated points
    are eligible. This avoids forcing immediate tissue around a point
    to background.

    These are weak negatives, not voxel-wise ground truth.
    """
    point_xyz = np.array(
        [(z, y, x) for _, z, y, x in points],
        dtype=np.int32,
    )

    selected: List[Tuple[int, int, int]] = []

    max_attempts = max(1000, count * 30)

    for _ in range(max_attempts):
        if len(selected) >= count:
            break

        z = int(rng.integers(0, PATCH_SIZE[0]))
        y = int(rng.integers(0, PATCH_SIZE[1]))
        x = int(rng.integers(0, PATCH_SIZE[2]))

        if point_xyz.size:
            distances = np.sqrt(
                np.sum(
                    (point_xyz - np.array([z, y, x])) ** 2,
                    axis=1,
                )
            )

            if float(distances.min()) < 3.0:
                continue

        selected.append((z, y, x))

    if not selected:
        return np.empty(
            (0, 3),
            dtype=np.int64,
        )

    return np.asarray(
        selected,
        dtype=np.int64,
    )


def point_supervision_loss(
    logits: torch.Tensor,
    points: Sequence[Tuple[int, int, int, int]],
    rng: np.random.Generator,
    point_class_weights: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Sparse point-supervision objective.

    Positive term:
        CE at every annotated RSNA point.

    Background term:
        CE on a small set of sampled voxels away from all points,
        multiplied by BACKGROUND_LOSS_WEIGHT.

    There is deliberately no full voxel-wise target mask.
    """
    if logits.ndim != 5:
        raise ValueError(
            f"Expected logits [B,C,D,H,W], got {tuple(logits.shape)}"
        )

    if logits.shape[0] != 1:
        raise ValueError("Pilot uses batch size 1.")

    if not points:
        raise ValueError("No supervision points supplied.")

    z_idx = torch.tensor(
        [p[1] for p in points],
        device=logits.device,
        dtype=torch.long,
    )
    y_idx = torch.tensor(
        [p[2] for p in points],
        device=logits.device,
        dtype=torch.long,
    )
    x_idx = torch.tensor(
        [p[3] for p in points],
        device=logits.device,
        dtype=torch.long,
    )
    labels = torch.tensor(
        [p[0] for p in points],
        device=logits.device,
        dtype=torch.long,
    )

    point_logits = logits[
        0,
        :,
        z_idx,
        y_idx,
        x_idx,
    ].transpose(0, 1).contiguous()

    positive_loss = F.cross_entropy(
        point_logits,
        labels,
        weight=point_class_weights,
    )

    background_coords = make_background_samples(
        points,
        BACKGROUND_SAMPLES_PER_CASE,
        rng,
    )

    if len(background_coords) > 0:
        bz = torch.from_numpy(
            background_coords[:, 0]
        ).to(logits.device)

        by = torch.from_numpy(
            background_coords[:, 1]
        ).to(logits.device)

        bx = torch.from_numpy(
            background_coords[:, 2]
        ).to(logits.device)

        bg_logits = logits[
            0,
            :,
            bz,
            by,
            bx,
        ].transpose(0, 1).contiguous()

        bg_labels = torch.zeros(
            len(background_coords),
            device=logits.device,
            dtype=torch.long,
        )

        background_loss = F.cross_entropy(
            bg_logits,
            bg_labels,
        )
    else:
        background_loss = torch.zeros(
            (),
            device=logits.device,
            dtype=logits.dtype,
        )

    total = (
        positive_loss
        + BACKGROUND_LOSS_WEIGHT * background_loss
    )

    return total, {
        "total_loss": float(total.detach().item()),
        "point_loss": float(positive_loss.detach().item()),
        "background_loss": float(
            background_loss.detach().item()
        ),
        "num_points": float(len(points)),
        "num_background": float(len(background_coords)),
    }


# =============================================================================
# POINT VALIDATION METRICS
# =============================================================================

@torch.no_grad()
def evaluate_points(
    model: torch.nn.Module,
    cases: Sequence[Tuple[str, str]],
    groups: Dict[Tuple[str, str], List[Tuple[int, int, int, int]]],
    part11: Any,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    correct = 0
    total = 0
    probability_sum = 0.0
    hit_050 = 0
    distance_sum = 0.0
    distance_count = 0
    disease_counts = defaultdict(int)
    disease_correct = defaultdict(int)
    disease_probability = defaultdict(float)

    case_count = 0
    case_errors: List[Dict[str, str]] = []

    for case_index, (study_id, series_id) in enumerate(cases, start=1):
        try:
            image = load_image_case(
                study_id,
                series_id,
                part11,
            )

            x = image.unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            logits = model(x)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            points = groups[
                (study_id, series_id)
            ]

            for class_id, z, y, xx in points:
                p = probabilities[
                    0,
                    :,
                    z,
                    y,
                    xx,
                ]

                predicted = int(
                    torch.argmax(p).item()
                )

                correct_probability = float(
                    p[class_id].item()
                )

                disease_counts[class_id] += 1
                disease_probability[class_id] += (
                    correct_probability
                )

                total += 1
                probability_sum += correct_probability

                if predicted == class_id:
                    correct += 1
                    disease_correct[class_id] += 1

                if predicted == class_id and correct_probability >= 0.50:
                    hit_050 += 1

            # Diagnostic distance from point to the nearest voxel belonging
            # to the correct predicted class in a small 5x5x5 neighborhood.
            # This is only a localization diagnostic; it is NOT a Dice metric.
            for class_id, z, y, xx in points:
                z0 = max(0, z - 2)
                z1 = min(PATCH_SIZE[0], z + 3)
                y0 = max(0, y - 2)
                y1 = min(PATCH_SIZE[1], y + 3)
                x0 = max(0, xx - 2)
                x1 = min(PATCH_SIZE[2], xx + 3)

                local = probabilities[
                    0,
                    class_id,
                    z0:z1,
                    y0:y1,
                    x0:x1,
                ]

                local_max = float(local.max().item())

                if local_max > 0.50:
                    local_positions = torch.nonzero(
                        local >= 0.50,
                        as_tuple=False,
                    )

                    if len(local_positions):
                        target = np.array(
                            [
                                z - z0,
                                y - y0,
                                xx - x0,
                            ],
                            dtype=np.float32,
                        )

                        positions = (
                            local_positions.detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )

                        distances = np.sqrt(
                            np.sum(
                                (positions - target) ** 2,
                                axis=1,
                            )
                        )

                        distance_sum += float(
                            distances.min()
                        )
                        distance_count += 1

            case_count += 1

            if case_index % 5 == 0 or case_index == len(cases):
                print(
                    f"  Validation {case_index}/{len(cases)} "
                    f"| points={total}"
                )

            del x, logits, probabilities, image
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            case_errors.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

    class_metrics = {}

    for class_id in range(1, NUM_CLASSES):
        n = disease_counts[class_id]

        class_metrics[str(class_id)] = {
            "class_name": LABELS[class_id],
            "points": int(n),
            "accuracy": (
                float(disease_correct[class_id] / n)
                if n
                else None
            ),
            "mean_correct_probability": (
                float(
                    disease_probability[class_id] / n
                )
                if n
                else None
            ),
        }

    result = {
        "cases_evaluated": case_count,
        "case_errors": case_errors,
        "points": total,
        "point_class_accuracy": (
            float(correct / total)
            if total
            else None
        ),
        "point_correct_probability": (
            float(probability_sum / total)
            if total
            else None
        ),
        "point_hit_probability_0.50": (
            float(hit_050 / total)
            if total
            else None
        ),
        "mean_localization_distance_voxels": (
            float(distance_sum / distance_count)
            if distance_count
            else None
        ),
        "localization_distance_cases": int(
            distance_count
        ),
        "per_class": class_metrics,
    }

    return result


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cases: Sequence[Tuple[str, str]],
    groups: Dict[Tuple[str, str], List[Tuple[int, int, int, int]]],
    part11: Any,
    device: torch.device,
    epoch: int,
    point_class_weights: torch.Tensor | None = None,
) -> Dict[str, float]:
    model.train()

    rng = np.random.default_rng(
        SEED + epoch
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    totals = defaultdict(float)
    successful = 0
    failures = []

    for case_index, (study_id, series_id) in enumerate(
        cases,
        start=1,
    ):
        try:
            image = load_image_case(
                study_id,
                series_id,
                part11,
            )

            x = image.unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            points = groups[
                (study_id, series_id)
            ]

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(
                    USE_AMP
                    and device.type == "cuda"
                ),
            ):
                logits = model(x)

                loss, loss_info = point_supervision_loss(
                    logits,
                    points,
                    rng,
                    point_class_weights=point_class_weights,
                )

                scaled_loss = (
                    loss / GRADIENT_ACCUMULATION
                )

            scaler.scale(
                scaled_loss
            ).backward()

            should_step = (
                (case_index % GRADIENT_ACCUMULATION == 0)
                or case_index == len(cases)
            )

            if should_step:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            for key, value in loss_info.items():
                totals[key] += value

            successful += 1

            if case_index % 10 == 0 or case_index == len(cases):
                avg_loss = (
                    totals["total_loss"] / successful
                )

                print(
                    f"  Train {case_index}/{len(cases)} "
                    f"| avg_loss={avg_loss:.5f}"
                )

            del x, logits, image, loss

            if device.type == "cuda":
                torch.cuda.empty_cache()

            gc.collect()

        except Exception as exc:
            failures.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

            print(
                f"  [WARNING] case failed: "
                f"{study_id}/{series_id} -> {exc}"
            )

    if successful == 0:
        raise RuntimeError(
            "No training cases completed successfully."
        )

    return {
        "cases_successful": successful,
        "cases_failed": len(failures),
        "mean_total_loss": (
            totals["total_loss"] / successful
        ),
        "mean_point_loss": (
            totals["point_loss"] / successful
        ),
        "mean_background_loss": (
            totals["background_loss"] / successful
        ),
        "mean_points_per_case": (
            totals["num_points"] / successful
        ),
        "failed_cases": failures,
    }


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
) -> Path:
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        CHECKPOINT_DIR
        / f"part216_epoch_{epoch:02d}.pth"
    )

    torch.save(
        {
            "part": "2.16",
            "experiment": "RSNA disease-balanced point-supervised Swin-UNETR",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": {
                "patch_size": list(PATCH_SIZE),
                "in_channels": IN_CHANNELS,
                "num_classes": NUM_CLASSES,
                "feature_size": FEATURE_SIZE,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation": GRADIENT_ACCUMULATION,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "epochs": EPOCHS,
                "train_series_limit": TRAIN_SERIES_LIMIT,
                "val_series_limit": VAL_SERIES_LIMIT,
                "background_samples_per_case": (
                    BACKGROUND_SAMPLES_PER_CASE
                ),
                "background_loss_weight": (
                    BACKGROUND_LOSS_WEIGHT
                ),
                "balance_power": BALANCE_POWER,
                "min_class_weight": MIN_CLASS_WEIGHT,
                "max_class_weight": MAX_CLASS_WEIGHT,
                "sampling": "disease-aware with replacement",
                "seed": SEED,
            },
        },
        path,
    )

    return path


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    seed_everything(SEED)

    for directory in [
        OUTPUT_DIR,
        CHECKPOINT_DIR,
        METRICS_DIR,
        PREDICTION_DIR,
        VIS_DIR,
        REPORT_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    section(
        "PART 2.14 — RSNA POINT-SUPERVISED SWIN-UNETR"
    )

    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Manifest          : {MANIFEST_PATH}")
    print(f"Part 11           : {PART11_PATH}")
    print(f"Protected Part104 : {PROTECTED_PART104}")
    print(f"Output             : {OUTPUT_DIR}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Part 2.13 manifest not found:\n{MANIFEST_PATH}"
        )

    if not PROTECTED_PART104.exists():
        raise FileNotFoundError(
            f"Protected Part104 checkpoint not found:\n"
            f"{PROTECTED_PART104}"
        )

    section("LOADING PART 2.13 MANIFEST")

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Manifest rows      : {len(manifest)}"
    )
    print(
        f"Manifest columns   : {len(manifest.columns)}"
    )

    columns = resolve_manifest_columns(
        manifest
    )

    print("Resolved columns:")
    for key, value in columns.items():
        print(
            f"  {key:10s} -> {value}"
        )

    groups = build_series_groups(
        manifest,
        columns,
    )

    print(
        f"Annotated series   : {len(groups)}"
    )
    print(
        f"Mapped points      : "
        f"{sum(len(v) for v in groups.values())}"
    )

    if len(groups) < (
        TRAIN_SERIES_LIMIT + VAL_SERIES_LIMIT
    ):
        raise RuntimeError(
            "Not enough annotated series for the pilot."
        )

    section("LOADING ESTABLISHED PART 11")

    part11 = load_part11()

    print("✓ Part 11 imported.")
    print("✓ Robust DICOM loader available.")
    print("✓ Part 11 resize_3d available.")
    print("✓ Pseudo-mask loading is NOT used.")

    section("STUDY-LEVEL TRAIN / VALIDATION SPLIT")

    train_series, val_series = split_series_by_study(
        groups,
        SEED,
    )

    # Keep every train candidate available. Part 2.16 creates a
    # disease-balanced schedule from this study-disjoint pool.
    train_candidate_cases = list(train_series)

    val_cases = val_series[
        :VAL_SERIES_LIMIT
    ]

    print(
        f"Available train series : {len(train_series)}"
    )
    print(
        f"Available val series   : {len(val_series)}"
    )
    print(
        f"Available train candidates : {len(train_candidate_cases)}"
    )
    print(
        f"Pilot val series       : {len(val_cases)}"
    )

    if not train_candidate_cases or not val_cases:
        raise RuntimeError(
            "Study-level split produced an empty pilot set."
        )

    train_studies = {
        study_id
        for study_id, _ in train_candidate_cases
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
            "DATA LEAKAGE DETECTED: train/validation studies overlap."
        )

    print(
        f"✓ Study overlap : {len(overlap)}"
    )

    section("DEVICE")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device            : {device}")

    if device.type == "cuda":
        print(
            f"GPU               : "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            f"VRAM allocated     : "
            f"{torch.cuda.memory_allocated() / 1024**3:.3f} GB"
        )

    section("CREATING SWIN-UNETR")

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
        f"Parameters        : {parameter_count:,}"
    )

    section("LOADING PART104 INITIALIZATION")

    checkpoint = torch.load(
        PROTECTED_PART104,
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
            "Part104 checkpoint does not contain a state dictionary."
        )

    cleaned_state = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        cleaned_state[new_key] = value

    incompatible = model.load_state_dict(
        cleaned_state,
        strict=False,
    )

    print(
        f"Missing keys       : "
        f"{len(incompatible.missing_keys)}"
    )
    print(
        f"Unexpected keys    : "
        f"{len(incompatible.unexpected_keys)}"
    )

    if incompatible.missing_keys:
        raise RuntimeError(
            "Part104 is not fully compatible with the canonical "
            "Swin-UNETR architecture."
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Part104 contains unexpected model keys."
        )

    print(
        "✓ Part104 loaded as initialization only."
    )
    print(
        "✓ Part104 file will NOT be modified."
    )

    del checkpoint
    del state_dict
    del cleaned_state

    initial_schedule, initial_counts, initial_weights = (
        build_balanced_training_schedule(
            train_candidate_cases,
            groups,
            TRAIN_SERIES_LIMIT,
            SEED,
        )
    )

    point_class_weights = build_point_class_weights(
        initial_schedule,
        groups,
        device,
    )

    print()
    print("Disease-balanced training:")
    print(f"  Candidate train series : {len(train_candidate_cases)}")
    print(f"  Training slots/epoch   : {TRAIN_SERIES_LIMIT}")
    print("  Sampling               : disease-aware with replacement")
    print("  Validation             : fixed and study-disjoint")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            USE_AMP
            and device.type == "cuda"
        ),
    )

    history: List[Dict[str, Any]] = []

    section("CONTROLLED PILOT")

    print(
        f"Training slots/epoch    : {TRAIN_SERIES_LIMIT}"
    )
    print(
        f"Validation cases        : {len(val_cases)}"
    )
    print(
        f"Epochs                  : {EPOCHS}"
    )
    print(
        f"Patch                   : {PATCH_SIZE}"
    )
    print(
        f"Gradient accumulation   : {GRADIENT_ACCUMULATION}"
    )
    print(
        f"Background samples/case : {BACKGROUND_SAMPLES_PER_CASE}"
    )
    print(
        f"Background loss weight  : {BACKGROUND_LOSS_WEIGHT}"
    )

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        section(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        epoch_start = time.time()

        epoch_train_cases, epoch_counts, epoch_weights = (
            build_balanced_training_schedule(
                train_candidate_cases,
                groups,
                TRAIN_SERIES_LIMIT,
                SEED + epoch,
            )
        )

        print(
            f"Balanced schedule unique cases: "
            f"{len(set(epoch_train_cases))}/{len(epoch_train_cases)}"
        )

        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            cases=epoch_train_cases,
            groups=groups,
            part11=part11,
            device=device,
            epoch=epoch,
            point_class_weights=point_class_weights,
        )

        train_metrics["unique_cases"] = len(set(epoch_train_cases))
        train_metrics["schedule_slots"] = len(epoch_train_cases)

        print(
            f"Training mean loss : "
            f"{train_metrics['mean_total_loss']:.6f}"
        )

        section(
            f"VALIDATION — EPOCH {epoch}"
        )

        val_metrics = evaluate_points(
            model=model,
            cases=val_cases,
            groups=groups,
            part11=part11,
            device=device,
        )

        print(
            f"Point accuracy     : "
            f"{val_metrics['point_class_accuracy']}"
        )
        print(
            f"Correct probability: "
            f"{val_metrics['point_correct_probability']}"
        )
        print(
            f"Hit @ 0.50         : "
            f"{val_metrics['point_hit_probability_0.50']}"
        )

        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        epoch_record = {
            "epoch": epoch,
            "elapsed_seconds": (
                time.time() - epoch_start
            ),
            "train": train_metrics,
            "validation": val_metrics,
            "checkpoint": str(checkpoint_path),
        }

        history.append(
            epoch_record
        )

        with open(
            METRICS_DIR / "part214_training_history.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                history,
                handle,
                indent=2,
            )

        print(
            f"Checkpoint saved   : {checkpoint_path}"
        )

    total_elapsed = (
        time.time() - start_time
    )

    final_summary = {
        "part": "2.16",
        "status": "completed",
        "experiment": "RSNA disease-balanced point-supervised Swin-UNETR",
        "scientific_note": (
            "RSNA annotations are point supervision only. "
            "No voxel-wise ground-truth segmentation masks were created. "
            "Disease balancing uses weighted case sampling with replacement; "
            "repeated cases are not additional patients."
        ),
        "manifest": str(MANIFEST_PATH),
        "manifest_rows": int(len(manifest)),
        "annotated_series": int(len(groups)),
        "mapped_points": int(
            sum(len(v) for v in groups.values())
        ),
        "training_candidate_series": int(
            len(train_candidate_cases)
        ),
        "training_slots_per_epoch": int(
            TRAIN_SERIES_LIMIT
        ),
        "pilot_val_series": [
            [study, series]
            for study, series in val_cases
        ],
        "study_overlap": int(len(overlap)),
        "model": {
            "architecture": "SwinUNETR",
            "spatial_dims": 3,
            "in_channels": IN_CHANNELS,
            "out_channels": NUM_CLASSES,
            "feature_size": FEATURE_SIZE,
            "parameters": parameter_count,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "amp": bool(
                USE_AMP and device.type == "cuda"
            ),
        },
        "total_elapsed_seconds": total_elapsed,
        "history": history,
        "protected_checkpoint": str(
            PROTECTED_PART104
        ),
    }

    with open(
        REPORT_DIR / "part214_final_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            final_summary,
            handle,
            indent=2,
        )

    report_lines = [
        "PART 2.14 — POINT-SUPERVISED SWIN-UNETR",
        "",
        "Status: COMPLETED",
        "",
        f"Annotated series: {len(groups)}",
        f"Mapped points: {sum(len(v) for v in groups.values())}",
        f"Training candidate series: {len(train_candidate_cases)}",
        f"Training slots per epoch: {TRAIN_SERIES_LIMIT}",
        f"Pilot validation series: {len(val_cases)}",
        f"Study overlap: {len(overlap)}",
        "",
        "No voxel-wise ground-truth segmentation masks were created.",
        "RSNA labels were used as sparse point supervision.",
        "Disease balancing uses weighted case sampling with replacement; "
        "repeated cases are not additional patients.",
        "",
        f"Protected Part104: {PROTECTED_PART104}",
        "",
        "Epoch history:",
    ]

    for item in history:
        val = item["validation"]
        report_lines.extend(
            [
                f"  Epoch {item['epoch']}:",
                (
                    f"    train loss = "
                    f"{item['train']['mean_total_loss']:.6f}"
                ),
                (
                    f"    point accuracy = "
                    f"{val['point_class_accuracy']}"
                ),
                (
                    f"    correct probability = "
                    f"{val['point_correct_probability']}"
                ),
                (
                    f"    hit @ 0.50 = "
                    f"{val['point_hit_probability_0.50']}"
                ),
            ]
        )

    with open(
        REPORT_DIR / "part214_training_report.txt",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n".join(report_lines)
        )

    section("PART 2.14 COMPLETE")

    print(
        "✓ Point-supervised training pilot completed."
    )
    print(
        "✓ No voxel-wise segmentation mask was fabricated."
    )
    print(
        "✓ Protected Part104 checkpoint was not overwritten."
    )
    print(
        f"✓ Outputs: {OUTPUT_DIR}"
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    gc.collect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        raise
    except Exception as exc:
        print()
        print("=" * 78)
        print("PART 2.14 FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
