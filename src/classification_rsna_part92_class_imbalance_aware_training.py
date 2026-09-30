from pathlib import Path
import json
import random
import time
import math

import numpy as np
import pandas as pd
import pydicom

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# PART 92
# RSNA CLASSIFICATION — CLASS-IMBALANCE-AWARE TRAINING
# ============================================================
#
# Purpose:
#   Correct the Part 90 majority-class collapse using a controlled
#   class-weighted loss.
#
# Locked controls inherited from Part 90:
#   - RSNA 2024 Lumbar Spine Degenerative Classification
#   - Part 87 study-level split
#   - 1580 train studies
#   - 395 validation studies
#   - batch size 1
#   - gradient accumulation 4
#   - 5 epochs
#   - learning rate 1e-4
#   - weight decay 1e-5
#   - input volume 32 x 224 x 224
#   - 25 targets
#   - 3 severity classes
#   - seed 42
#   - AMP enabled
#
# Only intentional training change:
#   CLASS-WEIGHTED CROSS ENTROPY
#
# Part 91 showed:
#   overall accuracy       = 0.775389
#   majority baseline      = 0.775389
#   Normal/Mild recall     = 1.0
#   Moderate recall        = 0.0
#   Severe recall          = 0.0
#
# Therefore accuracy alone is NOT sufficient.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RSNA_ROOT = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES = RSNA_ROOT / "train_images"

TRAIN_MANIFEST = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part87_study_level_split"
    / "part87_train_manifest.csv"
)

VAL_MANIFEST = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part87_study_level_split"
    / "part87_validation_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part92_class_imbalance_aware_training"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

REPORT_DIR = ROOT / "reports"

HISTORY_CSV = (
    OUTPUT_DIR
    / "part92_training_history.csv"
)

VAL_RESULTS_CSV = (
    OUTPUT_DIR
    / "part92_validation_results.csv"
)

PER_TARGET_CSV = (
    OUTPUT_DIR
    / "part92_per_target_metrics.csv"
)

CONFUSION_CSV = (
    OUTPUT_DIR
    / "part92_confusion_matrices.csv"
)

CLASS_DISTRIBUTION_CSV = (
    OUTPUT_DIR
    / "part92_prediction_class_distribution.csv"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "part92_best_model.pth"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "part92_final_model.pth"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "part92_class_imbalance_aware_summary.json"
)

REPORT_TXT = (
    REPORT_DIR
    / "part92_class_imbalance_aware_report.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 1

GRADIENT_ACCUMULATION = 4

NUM_WORKERS = 0

EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

TARGET_DEPTH = 32

TARGET_HEIGHT = 224

TARGET_WIDTH = 224

NUM_CLASSES = 3

NUM_TARGETS = 25

USE_AMP = True

# ------------------------------------------------------------
# Controlled class weights
# ------------------------------------------------------------
#
# Normal/Mild is the majority class.
# Moderate and Severe receive increased loss weight.
#
# These are deliberately moderate rather than raw inverse-
# frequency weights, to avoid unstable optimization.
# ------------------------------------------------------------

CLASS_WEIGHTS = torch.tensor(
    [1.0, 2.5, 4.0],
    dtype=torch.float32,
)


CONDITIONS = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]

LEVELS = [
    "l1_l2",
    "l2_l3",
    "l3_l4",
    "l4_l5",
    "l5_s1",
]

TARGET_COLUMNS = [
    f"{condition}_{level}"
    for condition in CONDITIONS
    for level in LEVELS
]


SEVERITY_TO_INDEX = {
    "Normal/Mild": 0,
    "Moderate": 1,
    "Severe": 2,
}

INDEX_TO_SEVERITY = {
    0: "Normal/Mild",
    1: "Moderate",
    2: "Severe",
}


# ============================================================
# UTILITIES
# ============================================================

def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clean_value(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


# ============================================================
# DICOM LOADING
# ============================================================

def discover_dicom_files(series_path):

    path = Path(str(series_path))

    if not path.exists():
        return []

    try:
        files = [
            p
            for p in path.iterdir()
            if p.is_file()
        ]
    except Exception:
        return []

    files.sort(key=lambda p: p.name)

    return files


def load_dicom_series(series_path):

    files = discover_dicom_files(series_path)

    if not files:
        raise FileNotFoundError(
            f"No DICOM files found: {series_path}"
        )

    slices = []

    for path in files:

        try:
            ds = pydicom.dcmread(
                str(path),
                force=True
            )

            if not hasattr(ds, "PixelData"):
                continue

            image = ds.pixel_array

            if image.ndim != 2:
                continue

            instance_number = getattr(
                ds,
                "InstanceNumber",
                None
            )

            position = getattr(
                ds,
                "ImagePositionPatient",
                None
            )

            z_position = None

            if (
                position is not None
                and len(position) >= 3
            ):
                try:
                    z_position = float(position[2])
                except Exception:
                    z_position = None

            image = np.asarray(
                image,
                dtype=np.float32
            )

            slices.append(
                {
                    "array": image,
                    "instance_number": instance_number,
                    "z_position": z_position,
                    "path": path,
                }
            )

        except Exception:
            continue

    if not slices:
        raise RuntimeError(
            f"No valid DICOM slices found: {series_path}"
        )

    # Prefer physical z-position when available.
    if all(
        item["z_position"] is not None
        for item in slices
    ):
        slices.sort(
            key=lambda item: item["z_position"]
        )

    elif all(
        item["instance_number"] is not None
        for item in slices
    ):
        slices.sort(
            key=lambda item: int(
                item["instance_number"]
            )
        )

    else:
        slices.sort(
            key=lambda item: item["path"].name
        )

    # --------------------------------------------------------
    # Harmonize in-plane dimensions.
    # --------------------------------------------------------

    max_height = max(
        item["array"].shape[0]
        for item in slices
    )

    max_width = max(
        item["array"].shape[1]
        for item in slices
    )

    volume_slices = []

    for item in slices:

        image = item["array"]

        h, w = image.shape

        output = np.zeros(
            (max_height, max_width),
            dtype=np.float32
        )

        y0 = max(
            0,
            (max_height - h) // 2
        )

        x0 = max(
            0,
            (max_width - w) // 2
        )

        output[
            y0:y0 + h,
            x0:x0 + w
        ] = image[
            :min(h, max_height),
            :min(w, max_width)
        ]

        volume_slices.append(output)

    volume = np.stack(
        volume_slices,
        axis=0
    )

    return volume


# ============================================================
# VOLUME PREPROCESSING
# ============================================================

def normalize_volume(volume):

    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    finite_mask = np.isfinite(volume)

    if not finite_mask.any():
        return np.zeros_like(
            volume,
            dtype=np.float32
        )

    valid = volume[finite_mask]

    low = np.percentile(
        valid,
        1
    )

    high = np.percentile(
        valid,
        99
    )

    if high <= low:
        low = float(valid.min())
        high = float(valid.max())

    volume = np.clip(
        volume,
        low,
        high
    )

    if high > low:
        volume = (
            volume - low
        ) / (
            high - low
        )
    else:
        volume.fill(0.0)

    volume = np.nan_to_num(
        volume,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return volume.astype(
        np.float32
    )


def resize_volume(volume):

    tensor = torch.from_numpy(
        volume
    ).float()

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=(
            TARGET_DEPTH,
            TARGET_HEIGHT,
            TARGET_WIDTH,
        ),
        mode="trilinear",
        align_corners=False,
    )

    tensor = tensor.squeeze(
        0
    ).squeeze(
        0
    )

    return tensor


# ============================================================
# LABEL ENCODING
# ============================================================

def encode_targets(row):

    labels = []
    mask = []

    for column in TARGET_COLUMNS:

        value = clean_value(
            row[column]
        )

        if value is None:

            labels.append(0)
            mask.append(0)

        else:

            if value not in SEVERITY_TO_INDEX:
                raise ValueError(
                    f"Unknown severity '{value}' "
                    f"in column '{column}'"
                )

            labels.append(
                SEVERITY_TO_INDEX[value]
            )

            mask.append(1)

    return (
        torch.tensor(
            labels,
            dtype=torch.long
        ),
        torch.tensor(
            mask,
            dtype=torch.bool
        ),
    )


# ============================================================
# DATASET
# ============================================================

class RSNAClassificationDataset(Dataset):

    def __init__(
        self,
        manifest,
    ):

        self.manifest = manifest.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.manifest)

    def __getitem__(self, index):

        row = self.manifest.iloc[index]

        series_path = row["series_path"]

        volume = load_dicom_series(
            series_path
        )

        volume = normalize_volume(
            volume
        )

        image = resize_volume(
            volume
        )

        image = image.unsqueeze(0)

        labels, mask = encode_targets(
            row
        )

        return (
            image,
            labels,
            mask,
            str(row["study_id"]),
        )


# ============================================================
# MODEL
# ============================================================
#
# Lightweight 3D CNN matching the Part 90 baseline design:
#
# Input:
#   [B, 1, 32, 224, 224]
#
# Output:
#   [B, 25, 3]
#
# ============================================================

class Part90CNN(nn.Module):

    def __init__(
        self,
        num_targets=25,
        num_classes=3,
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                1,
                8,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(8),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool3d(
                kernel_size=2
            ),

            nn.Conv3d(
                8,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(16),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool3d(
                kernel_size=2
            ),

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(32),

            nn.ReLU(
                inplace=True
            ),

            nn.AdaptiveAvgPool3d(
                (1, 1, 1)
            ),
        )

        self.classifier = nn.Linear(
            32,
            num_targets * num_classes
        )

        self.num_targets = num_targets
        self.num_classes = num_classes

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(x)

        x = x.view(
            x.shape[0],
            self.num_targets,
            self.num_classes,
        )

        return x


# ============================================================
# LOSS
# ============================================================

def weighted_masked_cross_entropy(
    logits,
    targets,
    valid_mask,
    class_weights,
):

    batch_size = logits.shape[0]

    logits = logits.reshape(
        batch_size * NUM_TARGETS,
        NUM_CLASSES
    )

    targets = targets.reshape(
        batch_size * NUM_TARGETS
    )

    valid_mask = valid_mask.reshape(
        batch_size * NUM_TARGETS
    )

    if valid_mask.sum().item() == 0:
        return logits.sum() * 0.0

    logits = logits[
        valid_mask
    ]

    targets = targets[
        valid_mask
    ]

    return F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
    )


# ============================================================
# METRIC HELPERS
# ============================================================

def confusion_matrix_from_arrays(
    true_values,
    pred_values,
):

    matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    for true_value, pred_value in zip(
        true_values,
        pred_values
    ):

        matrix[
            int(true_value),
            int(pred_value)
        ] += 1

    return matrix


def safe_divide(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def metrics_from_confusion(
    matrix
):

    total = matrix.sum()

    correct = np.trace(
        matrix
    )

    accuracy = safe_divide(
        correct,
        total
    )

    recalls = []
    precisions = []
    f1_values = []

    for class_id in range(
        NUM_CLASSES
    ):

        tp = matrix[
            class_id,
            class_id
        ]

        fn = (
            matrix[
                class_id,
                :
            ].sum()
            - tp
        )

        fp = (
            matrix[
                :,
                class_id
            ].sum()
            - tp
        )

        recall = safe_divide(
            tp,
            tp + fn
        )

        precision = safe_divide(
            tp,
            tp + fp
        )

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2.0
                * precision
                * recall
                / (precision + recall)
            )

        recalls.append(
            recall
        )

        precisions.append(
            precision
        )

        f1_values.append(
            f1
        )

    balanced_accuracy = float(
        np.mean(recalls)
    )

    macro_precision = float(
        np.mean(precisions)
    )

    macro_recall = float(
        np.mean(recalls)
    )

    macro_f1 = float(
        np.mean(f1_values)
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "class_precision": precisions,
        "class_recall": recalls,
        "class_f1": f1_values,
    }


# ============================================================
# ONE EPOCH — TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    class_weights,
    device,
):

    model.train()

    running_loss = 0.0
    valid_batches = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    start_time = time.time()

    for batch_index, batch in enumerate(
        loader
    ):

        images, targets, masks, _ = batch

        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(
                USE_AMP
                and device.type == "cuda"
            ),
        ):

            logits = model(
                images
            )

            loss = weighted_masked_cross_entropy(
                logits,
                targets,
                masks,
                class_weights,
            )

            backward_loss = (
                loss
                / GRADIENT_ACCUMULATION
            )

        if scaler is not None:

            scaler.scale(
                backward_loss
            ).backward()

        else:

            backward_loss.backward()

        should_step = (
            (
                batch_index + 1
            )
            % GRADIENT_ACCUMULATION
            == 0
            or (
                batch_index + 1
                == len(loader)
            )
        )

        if should_step:

            if scaler is not None:

                scaler.unscale_(
                    optimizer
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            if scaler is not None:

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        running_loss += float(
            loss.detach().item()
        )

        valid_batches += 1

        del (
            images,
            targets,
            masks,
            logits,
            loss,
            backward_loss,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = (
        time.time()
        - start_time
    )

    return (
        running_loss
        / max(valid_batches, 1),
        elapsed,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    class_weights,
    device,
):

    model.eval()

    running_loss = 0.0
    batches = 0

    all_true = []
    all_pred = []
    all_study_ids = []

    per_target_true = [
        []
        for _ in range(NUM_TARGETS)
    ]

    per_target_pred = [
        []
        for _ in range(NUM_TARGETS)
    ]

    per_target_studies = [
        []
        for _ in range(NUM_TARGETS)
    ]

    for batch_index, batch in enumerate(
        loader
    ):

        (
            images,
            targets,
            masks,
            study_ids,
        ) = batch

        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(
                USE_AMP
                and device.type == "cuda"
            ),
        ):

            logits = model(
                images
            )

            loss = weighted_masked_cross_entropy(
                logits,
                targets,
                masks,
                class_weights,
            )

        running_loss += float(
            loss.item()
        )

        batches += 1

        predictions = torch.argmax(
            logits,
            dim=2
        )

        for target_index in range(
            NUM_TARGETS
        ):

            valid = masks[
                :,
                target_index
            ]

            if not valid.any():
                continue

            true_values = targets[
                valid,
                target_index
            ].detach().cpu().numpy()

            pred_values = predictions[
                valid,
                target_index
            ].detach().cpu().numpy()

            for true_value, pred_value in zip(
                true_values,
                pred_values
            ):

                per_target_true[
                    target_index
                ].append(
                    int(true_value)
                )

                per_target_pred[
                    target_index
                ].append(
                    int(pred_value)
                )

                per_target_studies[
                    target_index
                ].append(
                    str(
                        study_ids[
                            int(
                                np.where(
                                    valid.cpu().numpy()
                                )[0][
                                    len(
                                        per_target_studies[
                                            target_index
                                        ]
                                    )
                                    % valid.sum().item()
                                ]
                            )
                        ]
                    )
                )

            all_true.extend(
                true_values.tolist()
            )

            all_pred.extend(
                pred_values.tolist()
            )

            all_study_ids.extend(
                [
                    str(x)
                    for x in study_ids
                ]
            )

        del (
            images,
            targets,
            masks,
            logits,
            predictions,
            loss,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    global_matrix = confusion_matrix_from_arrays(
        all_true,
        all_pred,
    )

    global_metrics = metrics_from_confusion(
        global_matrix
    )

    per_target_rows = []

    confusion_rows = []

    for target_index, target_name in enumerate(
        TARGET_COLUMNS
    ):

        true_values = per_target_true[
            target_index
        ]

        pred_values = per_target_pred[
            target_index
        ]

        matrix = confusion_matrix_from_arrays(
            true_values,
            pred_values,
        )

        metrics = metrics_from_confusion(
            matrix
        )

        row = {
            "target_index": target_index,
            "target": target_name,
            "valid_labels": len(
                true_values
            ),
            "accuracy": metrics[
                "accuracy"
            ],
            "balanced_accuracy": metrics[
                "balanced_accuracy"
            ],
            "macro_precision": metrics[
                "macro_precision"
            ],
            "macro_recall": metrics[
                "macro_recall"
            ],
            "macro_f1": metrics[
                "macro_f1"
            ],
            "normal_mild_recall": metrics[
                "class_recall"
            ][0],
            "moderate_recall": metrics[
                "class_recall"
            ][1],
            "severe_recall": metrics[
                "class_recall"
            ][2],
            "normal_mild_precision": metrics[
                "class_precision"
            ][0],
            "moderate_precision": metrics[
                "class_precision"
            ][1],
            "severe_precision": metrics[
                "class_precision"
            ][2],
            "normal_mild_f1": metrics[
                "class_f1"
            ][0],
            "moderate_f1": metrics[
                "class_f1"
            ][1],
            "severe_f1": metrics[
                "class_f1"
            ][2],
        }

        per_target_rows.append(
            row
        )

        for true_class in range(
            NUM_CLASSES
        ):

            for pred_class in range(
                NUM_CLASSES
            ):

                confusion_rows.append(
                    {
                        "target_index":
                            target_index,
                        "target":
                            target_name,
                        "true_class":
                            INDEX_TO_SEVERITY[
                                true_class
                            ],
                        "predicted_class":
                            INDEX_TO_SEVERITY[
                                pred_class
                            ],
                        "count":
                            int(
                                matrix[
                                    true_class,
                                    pred_class
                                ]
                            ),
                    }
                )

    mean_target_balanced_accuracy = float(
        np.mean(
            [
                row[
                    "balanced_accuracy"
                ]
                for row in per_target_rows
            ]
        )
    )

    mean_target_macro_f1 = float(
        np.mean(
            [
                row[
                    "macro_f1"
                ]
                for row in per_target_rows
            ]
        )
    )

    class_counts_true = np.bincount(
        np.asarray(
            all_true,
            dtype=np.int64
        ),
        minlength=NUM_CLASSES,
    )

    class_counts_pred = np.bincount(
        np.asarray(
            all_pred,
            dtype=np.int64
        ),
        minlength=NUM_CLASSES,
    )

    majority_accuracy = safe_divide(
        class_counts_true.max(),
        class_counts_true.sum(),
    )

    result = {
        "loss": safe_divide(
            running_loss,
            batches
        ),
        "accuracy": global_metrics[
            "accuracy"
        ],
        "balanced_accuracy": global_metrics[
            "balanced_accuracy"
        ],
        "macro_precision": global_metrics[
            "macro_precision"
        ],
        "macro_recall": global_metrics[
            "macro_recall"
        ],
        "macro_f1": global_metrics[
            "macro_f1"
        ],
        "mean_target_balanced_accuracy":
            mean_target_balanced_accuracy,
        "mean_target_macro_f1":
            mean_target_macro_f1,
        "majority_accuracy":
            majority_accuracy,
        "class_counts_true":
            class_counts_true.tolist(),
        "class_counts_pred":
            class_counts_pred.tolist(),
        "class_recall":
            global_metrics[
                "class_recall"
            ],
        "class_precision":
            global_metrics[
                "class_precision"
            ],
        "class_f1":
            global_metrics[
                "class_f1"
            ],
        "valid_labels":
            len(all_true),
        "global_confusion_matrix":
            global_matrix.tolist(),
        "per_target_rows":
            per_target_rows,
        "confusion_rows":
            confusion_rows,
    }

    return result


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    metrics,
):

    checkpoint = {
        "part": 92,
        "epoch": epoch,
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "scaler_state_dict":
            (
                scaler.state_dict()
                if scaler is not None
                else None
            ),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation":
            GRADIENT_ACCUMULATION,
        "epochs": EPOCHS,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "target_depth":
            TARGET_DEPTH,
        "target_height":
            TARGET_HEIGHT,
        "target_width":
            TARGET_WIDTH,
        "num_targets":
            NUM_TARGETS,
        "num_classes":
            NUM_CLASSES,
        "class_weights":
            CLASS_WEIGHTS.tolist(),
        "validation_metrics":
            metrics,
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    banner(
        "PART 92 — CLASS-IMBALANCE-AWARE "
        "CLASSIFICATION TRAINING"
    )

    print(
        "Purpose:"
    )

    print(
        "Correct the Part 90 majority-class "
        "collapse using class-weighted loss."
    )

    print()

    print(
        "No SPIDER data."
    )

    print(
        "No test set."
    )

    print(
        "Same Part 87 study-level split."
    )

    print(
        "Same Part 90 model configuration."
    )

    print()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM   : "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    print(
        f"PyTorch: {torch.__version__}"
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    banner(
        "PATH VALIDATION"
    )

    required_paths = {
        "RSNA dataset":
            RSNA_ROOT,
        "Train manifest":
            TRAIN_MANIFEST,
        "Validation manifest":
            VAL_MANIFEST,
    }

    for name, path in required_paths.items():

        print(
            f"{path} : "
            f"{'FOUND' if path.exists() else 'MISSING'}"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Required path missing: {path}"
            )

    # --------------------------------------------------------
    # Load manifests
    # --------------------------------------------------------

    banner(
        "LOADING PART 87 STUDY-LEVEL SPLIT"
    )

    train_manifest = pd.read_csv(
        TRAIN_MANIFEST
    )

    val_manifest = pd.read_csv(
        VAL_MANIFEST
    )

    train_ids = set(
        train_manifest[
            "study_id"
        ].astype(str)
    )

    val_ids = set(
        val_manifest[
            "study_id"
        ].astype(str)
    )

    overlap = (
        train_ids
        & val_ids
    )

    print(
        f"Training studies   : "
        f"{len(train_manifest)}"
    )

    print(
        f"Validation studies : "
        f"{len(val_manifest)}"
    )

    print(
        f"Study overlap      : "
        f"{len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Study-level leakage detected."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = RSNAClassificationDataset(
        train_manifest
    )

    val_dataset = RSNAClassificationDataset(
        val_manifest
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        f"Train samples : "
        f"{len(train_dataset)}"
    )

    print(
        f"Val samples   : "
        f"{len(val_dataset)}"
    )

    print(
        f"Train batches : "
        f"{len(train_loader)}"
    )

    print(
        f"Val batches   : "
        f"{len(val_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    banner(
        "CREATING PART 90-COMPATIBLE MODEL"
    )

    model = Part90CNN(
        num_targets=NUM_TARGETS,
        num_classes=NUM_CLASSES,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters : "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = (
        CLASS_WEIGHTS
        .to(device)
    )

    banner(
        "CLASS-WEIGHTED LOSS"
    )

    print(
        "Normal/Mild : "
        f"{CLASS_WEIGHTS[0].item():.4f}"
    )

    print(
        "Moderate    : "
        f"{CLASS_WEIGHTS[1].item():.4f}"
    )

    print(
        "Severe      : "
        f"{CLASS_WEIGHTS[2].item():.4f}"
    )

    print()
    print(
        "Missing labels remain masked."
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = None

    if (
        USE_AMP
        and device.type == "cuda"
    ):

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    banner(
        "STARTING PART 92 TRAINING"
    )

    history = []

    best_metric = -float("inf")

    best_epoch = 0

    total_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        train_loss, train_time = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                class_weights,
                device,
            )
        )

        validation = evaluate(
            model,
            val_loader,
            class_weights,
            device,
        )

        val_balanced = validation[
            "balanced_accuracy"
        ]

        val_macro_f1 = validation[
            "macro_f1"
        ]

        history_row = {
            "epoch":
                epoch,
            "train_loss":
                train_loss,
            "val_loss":
                validation[
                    "loss"
                ],
            "val_accuracy":
                validation[
                    "accuracy"
                ],
            "val_balanced_accuracy":
                val_balanced,
            "val_macro_precision":
                validation[
                    "macro_precision"
                ],
            "val_macro_recall":
                validation[
                    "macro_recall"
                ],
            "val_macro_f1":
                val_macro_f1,
            "mean_target_balanced_accuracy":
                validation[
                    "mean_target_balanced_accuracy"
                ],
            "mean_target_macro_f1":
                validation[
                    "mean_target_macro_f1"
                ],
            "majority_accuracy":
                validation[
                    "majority_accuracy"
                ],
            "normal_mild_recall":
                validation[
                    "class_recall"
                ][0],
            "moderate_recall":
                validation[
                    "class_recall"
                ][1],
            "severe_recall":
                validation[
                    "class_recall"
                ][2],
            "normal_mild_precision":
                validation[
                    "class_precision"
                ][0],
            "moderate_precision":
                validation[
                    "class_precision"
                ][1],
            "severe_precision":
                validation[
                    "class_precision"
                ][2],
            "normal_mild_f1":
                validation[
                    "class_f1"
                ][0],
            "moderate_f1":
                validation[
                    "class_f1"
                ][1],
            "severe_f1":
                validation[
                    "class_f1"
                ][2],
            "valid_labels":
                validation[
                    "valid_labels"
                ],
            "pred_normal_mild":
                validation[
                    "class_counts_pred"
                ][0],
            "pred_moderate":
                validation[
                    "class_counts_pred"
                ][1],
            "pred_severe":
                validation[
                    "class_counts_pred"
                ][2],
            "epoch_time_sec":
                time.time()
                - epoch_start,
        }

        history.append(
            history_row
        )

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train loss       : "
            f"{train_loss:.6f}"
        )

        print(
            f"Val loss         : "
            f"{validation['loss']:.6f}"
        )

        print(
            f"Accuracy         : "
            f"{validation['accuracy']:.6f}"
        )

        print(
            f"Balanced accuracy: "
            f"{val_balanced:.6f}"
        )

        print(
            f"Macro F1         : "
            f"{val_macro_f1:.6f}"
        )

        print(
            f"Normal/Mild R    : "
            f"{validation['class_recall'][0]:.6f}"
        )

        print(
            f"Moderate R       : "
            f"{validation['class_recall'][1]:.6f}"
        )

        print(
            f"Severe R         : "
            f"{validation['class_recall'][2]:.6f}"
        )

        print(
            f"Predicted classes: "
            f"NM={validation['class_counts_pred'][0]}, "
            f"M={validation['class_counts_pred'][1]}, "
            f"S={validation['class_counts_pred'][2]}"
        )

        # ----------------------------------------------------
        # Select best checkpoint by macro F1.
        # ----------------------------------------------------

        if val_macro_f1 > best_metric:

            best_metric = val_macro_f1

            best_epoch = epoch

            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                scaler,
                epoch,
                validation,
            )

            print(
                "✓ New best checkpoint saved."
            )

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    save_checkpoint(
        FINAL_CHECKPOINT,
        model,
        optimizer,
        scaler,
        EPOCHS,
        validation,
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Final evaluation artifacts
    # --------------------------------------------------------

    final_validation = evaluate(
        model,
        val_loader,
        class_weights,
        device,
    )

    # --------------------------------------------------------
    # Per-target metrics
    # --------------------------------------------------------

    per_target_df = pd.DataFrame(
        final_validation[
            "per_target_rows"
        ]
    )

    per_target_df.to_csv(
        PER_TARGET_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    confusion_df = pd.DataFrame(
        final_validation[
            "confusion_rows"
        ]
    )

    confusion_df.to_csv(
        CONFUSION_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    true_counts = (
        final_validation[
            "class_counts_true"
        ]
    )

    pred_counts = (
        final_validation[
            "class_counts_pred"
        ]
    )

    distribution_df = pd.DataFrame(
        {
            "class_id": [
                0,
                1,
                2,
            ],
            "class_name": [
                "Normal/Mild",
                "Moderate",
                "Severe",
            ],
            "true_count": true_counts,
            "predicted_count": pred_counts,
        }
    )

    distribution_df.to_csv(
        CLASS_DISTRIBUTION_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Validation results
    # --------------------------------------------------------

    validation_summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "accuracy",
                "value":
                    final_validation[
                        "accuracy"
                    ],
            },
            {
                "metric":
                    "majority_accuracy",
                "value":
                    final_validation[
                        "majority_accuracy"
                    ],
            },
            {
                "metric":
                    "balanced_accuracy",
                "value":
                    final_validation[
                        "balanced_accuracy"
                    ],
            },
            {
                "metric":
                    "macro_precision",
                "value":
                    final_validation[
                        "macro_precision"
                    ],
            },
            {
                "metric":
                    "macro_recall",
                "value":
                    final_validation[
                        "macro_recall"
                    ],
            },
            {
                "metric":
                    "macro_f1",
                "value":
                    final_validation[
                        "macro_f1"
                    ],
            },
            {
                "metric":
                    "mean_target_balanced_accuracy",
                "value":
                    final_validation[
                        "mean_target_balanced_accuracy"
                    ],
            },
            {
                "metric":
                    "mean_target_macro_f1",
                "value":
                    final_validation[
                        "mean_target_macro_f1"
                    ],
            },
            {
                "metric":
                    "normal_mild_recall",
                "value":
                    final_validation[
                        "class_recall"
                    ][0],
            },
            {
                "metric":
                    "moderate_recall",
                "value":
                    final_validation[
                        "class_recall"
                    ][1],
            },
            {
                "metric":
                    "severe_recall",
                "value":
                    final_validation[
                        "class_recall"
                    ][2],
            },
        ]
    )

    validation_summary_df.to_csv(
        VAL_RESULTS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Diagnosis
    # --------------------------------------------------------

    moderate_recall = final_validation[
        "class_recall"
    ][1]

    severe_recall = final_validation[
        "class_recall"
    ][2]

    predicted_minority_fraction = safe_divide(
        (
            pred_counts[1]
            + pred_counts[2]
        ),
        sum(pred_counts),
    )

    accuracy_gain = (
        final_validation[
            "accuracy"
        ]
        - final_validation[
            "majority_accuracy"
        ]
    )

    if (
        predicted_minority_fraction
        < 0.01
        and moderate_recall
        < 0.05
        and severe_recall
        < 0.05
    ):

        diagnosis = (
            "CLASS_WEIGHTING_DID_NOT_BREAK_MAJORITY_COLLAPSE"
        )

    elif (
        moderate_recall
        >= 0.10
        or severe_recall
        >= 0.10
    ):

        diagnosis = (
            "CLASS_WEIGHTING_REVEALS_NONTRIVIAL_SEVERITY_SIGNAL"
        )

    else:

        diagnosis = (
            "CLASS_WEIGHTING_PARTIALLY_REDUCES_MAJORITY_COLLAPSE"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_time = (
        time.time()
        - total_start
    )

    summary = {
        "part": 92,
        "status":
            "PASS — CLASS-IMBALANCE-AWARE "
            "TRAINING COMPLETE",
        "dataset":
            "RSNA 2024 Lumbar Spine Degenerative Classification",
        "train_studies":
            len(train_manifest),
        "validation_studies":
            len(val_manifest),
        "study_overlap":
            len(overlap),
        "model":
            "Part 90-compatible lightweight 3D CNN",
        "num_targets":
            NUM_TARGETS,
        "num_classes":
            NUM_CLASSES,
        "batch_size":
            BATCH_SIZE,
        "gradient_accumulation":
            GRADIENT_ACCUMULATION,
        "epochs":
            EPOCHS,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "class_weights":
            CLASS_WEIGHTS.tolist(),
        "seed":
            SEED,
        "best_epoch":
            best_epoch,
        "best_macro_f1":
            best_metric,
        "final_accuracy":
            final_validation[
                "accuracy"
            ],
        "majority_accuracy":
            final_validation[
                "majority_accuracy"
            ],
        "accuracy_gain_over_majority":
            accuracy_gain,
        "balanced_accuracy":
            final_validation[
                "balanced_accuracy"
            ],
        "macro_precision":
            final_validation[
                "macro_precision"
            ],
        "macro_recall":
            final_validation[
                "macro_recall"
            ],
        "macro_f1":
            final_validation[
                "macro_f1"
            ],
        "normal_mild_recall":
            final_validation[
                "class_recall"
            ][0],
        "moderate_recall":
            moderate_recall,
        "severe_recall":
            severe_recall,
        "true_class_counts":
            true_counts,
        "predicted_class_counts":
            pred_counts,
        "predicted_minority_fraction":
            predicted_minority_fraction,
        "diagnosis":
            diagnosis,
        "total_training_time_sec":
            total_time,
        "best_checkpoint":
            str(BEST_CHECKPOINT),
        "final_checkpoint":
            str(FINAL_CHECKPOINT),
        "training_history":
            str(HISTORY_CSV),
        "validation_results":
            str(VAL_RESULTS_CSV),
        "per_target_metrics":
            str(PER_TARGET_CSV),
        "confusion_matrices":
            str(CONFUSION_CSV),
        "class_distribution":
            str(CLASS_DISTRIBUTION_CSV),
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------

    with open(
        REPORT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "PART 92 — CLASS-IMBALANCE-AWARE "
            "CLASSIFICATION TRAINING\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            "Dataset: RSNA 2024 Lumbar Spine "
            "Degenerative Classification\n"
        )

        f.write(
            "Train studies: "
            f"{len(train_manifest)}\n"
        )

        f.write(
            "Validation studies: "
            f"{len(val_manifest)}\n"
        )

        f.write(
            "Study overlap: "
            f"{len(overlap)}\n\n"
        )

        f.write(
            "Class weights:\n"
        )

        f.write(
            f"  Normal/Mild = "
            f"{CLASS_WEIGHTS[0].item():.4f}\n"
        )

        f.write(
            f"  Moderate    = "
            f"{CLASS_WEIGHTS[1].item():.4f}\n"
        )

        f.write(
            f"  Severe      = "
            f"{CLASS_WEIGHTS[2].item():.4f}\n\n"
        )

        f.write(
            "FINAL VALIDATION METRICS\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            f"Accuracy              : "
            f"{final_validation['accuracy']:.6f}\n"
        )

        f.write(
            f"Majority baseline     : "
            f"{final_validation['majority_accuracy']:.6f}\n"
        )

        f.write(
            f"Accuracy gain         : "
            f"{accuracy_gain:.6f}\n"
        )

        f.write(
            f"Balanced accuracy     : "
            f"{final_validation['balanced_accuracy']:.6f}\n"
        )

        f.write(
            f"Macro precision       : "
            f"{final_validation['macro_precision']:.6f}\n"
        )

        f.write(
            f"Macro recall          : "
            f"{final_validation['macro_recall']:.6f}\n"
        )

        f.write(
            f"Macro F1              : "
            f"{final_validation['macro_f1']:.6f}\n"
        )

        f.write(
            f"Normal/Mild recall    : "
            f"{final_validation['class_recall'][0]:.6f}\n"
        )

        f.write(
            f"Moderate recall       : "
            f"{final_validation['class_recall'][1]:.6f}\n"
        )

        f.write(
            f"Severe recall         : "
            f"{final_validation['class_recall'][2]:.6f}\n\n"
        )

        f.write(
            "TRUE CLASS COUNTS\n"
        )

        f.write(
            f"Normal/Mild : {true_counts[0]}\n"
        )

        f.write(
            f"Moderate    : {true_counts[1]}\n"
        )

        f.write(
            f"Severe      : {true_counts[2]}\n\n"
        )

        f.write(
            "PREDICTED CLASS COUNTS\n"
        )

        f.write(
            f"Normal/Mild : {pred_counts[0]}\n"
        )

        f.write(
            f"Moderate    : {pred_counts[1]}\n"
        )

        f.write(
            f"Severe      : {pred_counts[2]}\n\n"
        )

        f.write(
            "DIAGNOSIS\n"
        )

        f.write(
            diagnosis
            + "\n\n"
        )

        f.write(
            "Interpretation:\n"
        )

        f.write(
            "The Part 91 baseline predicted "
            "Normal/Mild for every valid label. "
            "Part 92 introduces class-weighted "
            "cross-entropy while keeping the "
            "dataset split and model configuration "
            "controlled.\n"
        )

        f.write(
            "Accuracy must be interpreted together "
            "with balanced accuracy, macro F1, "
            "and Moderate/Severe recall.\n"
        )

    # --------------------------------------------------------
    # Final console
    # --------------------------------------------------------

    banner(
        "PART 92 COMPLETE"
    )

    print(
        f"Best epoch           : "
        f"{best_epoch}"
    )

    print(
        f"Best macro F1        : "
        f"{best_metric:.6f}"
    )

    print(
        f"Accuracy             : "
        f"{final_validation['accuracy']:.6f}"
    )

    print(
        f"Majority baseline    : "
        f"{final_validation['majority_accuracy']:.6f}"
    )

    print(
        f"Accuracy gain        : "
        f"{accuracy_gain:.6f}"
    )

    print(
        f"Balanced accuracy    : "
        f"{final_validation['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro F1             : "
        f"{final_validation['macro_f1']:.6f}"
    )

    print(
        f"Normal/Mild recall   : "
        f"{final_validation['class_recall'][0]:.6f}"
    )

    print(
        f"Moderate recall      : "
        f"{moderate_recall:.6f}"
    )

    print(
        f"Severe recall        : "
        f"{severe_recall:.6f}"
    )

    print()

    print(
        f"Diagnosis: {diagnosis}"
    )

    print()

    print(
        f"Best checkpoint:\n"
        f"{BEST_CHECKPOINT}"
    )

    print()

    print(
        f"Report:\n"
        f"{REPORT_TXT}"
    )

    print()

    print(
        "STATUS: PASS — "
        "CLASS-IMBALANCE-AWARE "
        "TRAINING COMPLETE"
    )


if __name__ == "__main__":
    main()