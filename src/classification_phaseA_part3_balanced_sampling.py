"""
PHASE A - PART 3
TARGETED BALANCED-SAMPLING CLASSIFICATION RETRAINING

Purpose
-------
Target the confirmed majority-class collapse in the classification branch.

Locked components:
- Existing study-level train/validation split
- Seed = 42
- Same SwinClassifier architecture
- Image size = 224
- 3 classes
- Weighted CrossEntropyLoss

New targeted intervention:
- WeightedRandomSampler on TRAINING DATA ONLY

Validation:
- Original study-level validation set
- No sampler
- No class weighting in validation metrics
- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-class precision/recall/F1
- Confusion matrix

Model selection:
- PRIMARY: validation macro-F1
- SECONDARY: balanced accuracy
- Accuracy is NOT the primary selection metric.

IMPORTANT
---------
This script does NOT overwrite:
    models/best_model.pth

This script does NOT overwrite:
    models/corrected_study_split_weighted_best_model.pth

A new checkpoint is created for Part A3.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from tqdm import tqdm


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
PHASE4M_DIR = OUTPUTS_DIR / "phase4m"

TRAIN_CSV = (
    PHASE4M_DIR
    / "proposed_train_samples.csv"
)

VAL_CSV = (
    PHASE4M_DIR
    / "proposed_validation_samples.csv"
)

TRAIN_STUDIES_CSV = (
    PHASE4M_DIR
    / "proposed_train_study_ids.csv"
)

VAL_STUDIES_CSV = (
    PHASE4M_DIR
    / "proposed_validation_study_ids.csv"
)

# Existing checkpoints are read only.
ORIGINAL_CHECKPOINT = (
    MODELS_DIR
    / "best_model.pth"
)

CORRECTED_CHECKPOINT = (
    MODELS_DIR
    / "corrected_study_split_weighted_best_model.pth"
)

# New A3 checkpoint.
A3_CHECKPOINT = (
    MODELS_DIR
    / "phaseA_part3_balanced_sampling_best_model.pth"
)

OUTPUT_DIR = (
    OUTPUTS_DIR
    / "phaseA_classification"
    / "part3_balanced_sampling"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HISTORY_CSV = (
    OUTPUT_DIR
    / "training_history.csv"
)

FINAL_METRICS_JSON = (
    OUTPUT_DIR
    / "final_metrics.json"
)

BEST_METRICS_JSON = (
    OUTPUT_DIR
    / "best_metrics.json"
)

CONFUSION_MATRIX_CSV = (
    OUTPUT_DIR
    / "best_confusion_matrix.csv"
)

CLASSIFICATION_REPORT_CSV = (
    OUTPUT_DIR
    / "best_classification_report.csv"
)

SAMPLER_DISTRIBUTION_CSV = (
    OUTPUT_DIR
    / "sampler_expected_distribution.csv"
)

RUN_CONFIG_JSON = (
    OUTPUT_DIR
    / "run_config.json"
)


# ============================================================================
# IMPORT EXISTING PROJECT MODEL
# ============================================================================

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.model import create_model


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

IMG_SIZE = 224

BATCH_SIZE = 4

EPOCHS = 15

NUM_WORKERS = 0

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 1e-5

NUM_CLASSES = 3

CLASS_NAMES = [
    "Normal/Mild",
    "Moderate",
    "Severe",
]

CLASS_WEIGHTS = np.array(
    [
        0.167594,
        0.788787,
        2.043618,
    ],
    dtype=np.float32,
)

# Sampler weights use inverse class frequency.
# This deliberately differs from the CE weights:
# CE handles loss importance; sampler changes exposure
# of minority examples during training.
SAMPLER_POWER = 1.0

# Gradient accumulation is kept at 1 for the targeted run.
GRAD_ACCUMULATION_STEPS = 1

USE_AMP = True


# ============================================================================
# EXPECTED DATASET CONTRACT
# ============================================================================

EXPECTED_TRAIN_SAMPLES = 38925
EXPECTED_VAL_SAMPLES = 9732

EXPECTED_TRAIN_STUDIES = 1580
EXPECTED_VAL_STUDIES = 394


# ============================================================================
# RANDOM SEED
# ============================================================================

def seed_everything(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# HELPERS
# ============================================================================

def banner(title: str) -> None:

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def sha256_file(path: Path) -> str | None:

    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> str:

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in normalized:

            return normalized[key]

    raise KeyError(
        "Could not find required column. "
        f"Tried: {candidates}. "
        f"Available: {list(df.columns)}"
    )


def normalize_label(
    value,
) -> int:

    if isinstance(
        value,
        (int, np.integer),
    ):

        return int(value)

    if isinstance(
        value,
        float,
    ):

        return int(value)

    text = str(value).strip()

    mapping = {

        "0": 0,
        "1": 1,
        "2": 2,

        "normal": 0,
        "normal/mild": 0,
        "normal_mild": 0,
        "normal mild": 0,

        "moderate": 1,

        "severe": 2,
    }

    key = text.lower()

    if key not in mapping:

        raise ValueError(
            f"Unknown class label: {value}"
        )

    return mapping[key]


# ============================================================================
# DATASET
# ============================================================================

class RSNAClassificationDataset(Dataset):
    """
    Lightweight dataset adapter.

    The existing project CSVs are expected to contain
    an image/path-like column and a class label column.

    The loader intentionally supports common column names
    used in the existing project.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_column: str,
        label_column: str,
        img_size: int = 224,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.image_column = image_column

        self.label_column = label_column

        self.img_size = img_size

    def __len__(self):

        return len(self.df)

    def _load_image(
        self,
        path_value,
    ):

        path = Path(
            str(path_value)
        )

        if not path.is_absolute():

            path = PROJECT_ROOT / path

        if not path.exists():

            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        # ------------------------------------------------------------
        # DICOM
        # ------------------------------------------------------------

        suffix = path.suffix.lower()

        if suffix in [
            ".dcm",
            ".dicom",
        ]:

            import pydicom

            ds = pydicom.dcmread(
                str(path),
                force=True,
            )

            image = ds.pixel_array.astype(
                np.float32
            )

        else:

            # --------------------------------------------------------
            # Standard image formats
            # --------------------------------------------------------

            from PIL import Image

            image = np.asarray(
                Image.open(path).convert(
                    "L"
                ),
                dtype=np.float32,
            )

        if image.ndim != 2:

            image = np.squeeze(
                image
            )

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Robust per-image normalization.
        min_value = float(
            image.min()
        )

        max_value = float(
            image.max()
        )

        if max_value > min_value:

            image = (
                image - min_value
            ) / (
                max_value - min_value
            )

        else:

            image = np.zeros_like(
                image,
                dtype=np.float32,
            )

        tensor = torch.from_numpy(
            image
        ).float()

        tensor = tensor.unsqueeze(
            0
        )

        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(
                self.img_size,
                self.img_size,
            ),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # ------------------------------------------------------------
        # SwinClassifier expects 3 channels.
        # Replicate grayscale MRI.
        # ------------------------------------------------------------

        tensor = tensor.repeat(
            3,
            1,
            1,
        )

        return tensor

    def __getitem__(
        self,
        index: int,
    ):

        row = self.df.iloc[index]

        image = self._load_image(
            row[
                self.image_column
            ]
        )

        label = normalize_label(
            row[
                self.label_column
            ]
        )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )


# ============================================================================
# DATAFRAME PREPARATION
# ============================================================================

def prepare_dataframe(
    csv_path: Path,
) -> Tuple[
    pd.DataFrame,
    str,
    str,
]:

    df = pd.read_csv(
        csv_path
    )

    image_column = find_column(
        df,
        [
            "image_path",
            "image",
            "path",
            "filepath",
            "file_path",
            "dicom_path",
            "dcm_path",
        ],
    )

    label_column = find_column(
        df,
        [
            "label",
            "class",
            "target",
            "severity",
            "grade",
            "class_id",
        ],
    )

    df = df.copy()

    df["_normalized_label"] = (
        df[label_column]
        .apply(normalize_label)
    )

    return (
        df,
        image_column,
        label_column,
    )


# ============================================================================
# STUDY SPLIT VALIDATION
# ============================================================================

def validate_study_split() -> Dict:

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    train_study_df = pd.read_csv(
        TRAIN_STUDIES_CSV
    )

    val_study_df = pd.read_csv(
        VAL_STUDIES_CSV
    )

    train_study_column = find_column(
        train_study_df,
        [
            "study_id",
            "study",
        ],
    )

    val_study_column = find_column(
        val_study_df,
        [
            "study_id",
            "study",
        ],
    )

    train_studies = set(
        train_study_df[
            train_study_column
        ]
        .astype(int)
    )

    val_studies = set(
        val_study_df[
            val_study_column
        ]
        .astype(int)
    )

    overlap = (
        train_studies
        & val_studies
    )

    checks = {

        "train_samples":
            len(train_df)
            == EXPECTED_TRAIN_SAMPLES,

        "validation_samples":
            len(val_df)
            == EXPECTED_VAL_SAMPLES,

        "train_studies":
            len(train_studies)
            == EXPECTED_TRAIN_STUDIES,

        "validation_studies":
            len(val_studies)
            == EXPECTED_VAL_STUDIES,

        "study_overlap_zero":
            len(overlap) == 0,

    }

    if not all(
        checks.values()
    ):

        raise RuntimeError(
            "Study-level split validation failed:\n"
            + json.dumps(
                checks,
                indent=2,
            )
        )

    return {

        "train_samples":
            len(train_df),

        "validation_samples":
            len(val_df),

        "train_studies":
            len(train_studies),

        "validation_studies":
            len(val_studies),

        "study_overlap":
            len(overlap),

        "checks":
            checks,

    }


# ============================================================================
# CLASS DISTRIBUTION
# ============================================================================

def class_distribution(
    df: pd.DataFrame,
) -> Dict[int, int]:

    counts = (
        df["_normalized_label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        int(k): int(v)
        for k, v in counts.items()
    }


# ============================================================================
# WEIGHTED RANDOM SAMPLER
# ============================================================================

def build_sampler(
    train_df: pd.DataFrame,
) -> Tuple[
    WeightedRandomSampler,
    np.ndarray,
    Dict,
]:

    labels = (
        train_df[
            "_normalized_label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES,
    )

    if np.any(
        counts == 0
    ):

        raise RuntimeError(
            "A training class has zero samples."
        )

    # Inverse-frequency sampling.
    inverse_frequency = (
        1.0 / counts.astype(
            np.float64
        )
    )

    class_sampling_weights = (
        inverse_frequency
        ** SAMPLER_POWER
    )

    sample_weights = (
        class_sampling_weights[
            labels
        ]
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(
            train_df
        ),
        replacement=True,
        generator=torch.Generator().manual_seed(
            SEED
        ),
    )

    expected_probability = (
        class_sampling_weights
        * counts
    )

    expected_probability = (
        expected_probability
        / expected_probability.sum()
    )

    sampler_info = {

        "raw_class_counts":
            counts.tolist(),

        "class_sampling_weights":
            class_sampling_weights.tolist(),

        "expected_sample_probability":
            expected_probability.tolist(),

        "sampler_power":
            SAMPLER_POWER,

    }

    return (
        sampler,
        sample_weights,
        sampler_info,
    )


# ============================================================================
# MODEL
# ============================================================================

def build_model():

    model = create_model(
        num_classes=NUM_CLASSES
    )

    return model.to(
        DEVICE
    )


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_initialization(
    model: nn.Module,
) -> str:

    """
    Initialize from the corrected checkpoint.

    If the corrected checkpoint is incompatible for any reason,
    fail loudly rather than silently switching models.
    """

    if not CORRECTED_CHECKPOINT.exists():

        raise FileNotFoundError(
            "Corrected checkpoint missing:\n"
            f"{CORRECTED_CHECKPOINT}"
        )

    checkpoint = torch.load(
        CORRECTED_CHECKPOINT,
        map_location="cpu",
    )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Corrected checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    return sha256_file(
        CORRECTED_CHECKPOINT
    )


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict:

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_precision = (
        precision_score(
            y_true,
            y_pred,
            labels=list(
                range(NUM_CLASSES)
            ),
            average="macro",
            zero_division=0,
        )
    )

    macro_recall = (
        recall_score(
            y_true,
            y_pred,
            labels=list(
                range(NUM_CLASSES)
            ),
            average="macro",
            zero_division=0,
        )
    )

    macro_f1 = (
        f1_score(
            y_true,
            y_pred,
            labels=list(
                range(NUM_CLASSES)
            ),
            average="macro",
            zero_division=0,
        )
    )

    per_class_precision = (
        precision_score(
            y_true,
            y_pred,
            labels=list(
                range(NUM_CLASSES)
            ),
            average=None,
            zero_division=0,
        )
    )

    per_class_recall = (
        recall_score(
            y_true,
            y_pred,
            labels=list(
                range(NUM_CLASSES)
            ),
            average=None,
            zero_division=0,
        )
    )

    per_class_f1 = (
        f1_score(
            y_true,
            y_pred,
            labels=list(
                range(NUM_CLASSES)
            ),
            average=None,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    return {

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "per_class": {

            CLASS_NAMES[i]: {

                "precision":
                    float(
                        per_class_precision[i]
                    ),

                "recall":
                    float(
                        per_class_recall[i]
                    ),

                "f1":
                    float(
                        per_class_f1[i]
                    ),

            }

            for i in range(
                NUM_CLASSES
            )
        },

        "confusion_matrix":
            cm.tolist(),

        "support":
            np.bincount(
                np.asarray(
                    y_true,
                    dtype=np.int64,
                ),
                minlength=NUM_CLASSES,
            ).tolist(),

    }


# ============================================================================
# TRAINING EPOCH
# ============================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
) -> float:

    model.train()

    running_loss = 0.0

    optimizer.zero_grad(
        set_to_none=True
    )

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for step, (
        images,
        labels,
    ) in enumerate(progress):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            enabled=(
                USE_AMP
                and DEVICE.type
                == "cuda"
            )
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels,
            )

            loss_for_backward = (
                loss
                / GRAD_ACCUMULATION_STEPS
            )

        scaler.scale(
            loss_for_backward
        ).backward()

        if (
            (step + 1)
            % GRAD_ACCUMULATION_STEPS
            == 0
        ):

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    # Handle a final incomplete accumulation.
    if (
        len(loader)
        % GRAD_ACCUMULATION_STEPS
        != 0
    ):

        scaler.step(
            optimizer
        )

        scaler.update()

        optimizer.zero_grad(
            set_to_none=True
        )

    return (
        running_loss
        / len(loader.dataset)
    )


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
) -> Tuple[
    float,
    Dict,
]:

    model.eval()

    running_loss = 0.0

    y_true = []

    y_pred = []

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            enabled=(
                USE_AMP
                and DEVICE.type
                == "cuda"
            )
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels,
            )

        predictions = (
            torch.argmax(
                outputs,
                dim=1,
            )
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        y_true.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        y_pred.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    validation_loss = (
        running_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    return (
        validation_loss,
        metrics,
    )


# ============================================================================
# BEST MODEL COMPARISON
# ============================================================================

def is_better(
    metrics: Dict,
    best_metrics: Dict | None,
) -> bool:

    if best_metrics is None:

        return True

    current_key = (
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
        metrics["macro_recall"],
    )

    best_key = (
        best_metrics["macro_f1"],
        best_metrics["balanced_accuracy"],
        best_metrics["macro_recall"],
    )

    return current_key > best_key


# ============================================================================
# MAIN
# ============================================================================

def main():

    banner(
        "PHASE A - PART 3"
    )

    print(
        "TARGETED BALANCED-SAMPLING CLASSIFICATION RETRAINING"
    )

    print()

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )


    # ------------------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------------------

    banner(
        "1. VERIFY EXISTING EXPERIMENTS"
    )

    for path in [
        TRAIN_CSV,
        VAL_CSV,
        TRAIN_STUDIES_CSV,
        VAL_STUDIES_CSV,
        CORRECTED_CHECKPOINT,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

        print(
            f"EXISTS: {path}"
        )

    print()

    print(
        "Original checkpoint SHA256:"
    )

    print(
        sha256_file(
            ORIGINAL_CHECKPOINT
        )
    )

    print()

    print(
        "Corrected checkpoint SHA256:"
    )

    print(
        sha256_file(
            CORRECTED_CHECKPOINT
        )
    )


    # ------------------------------------------------------------------------
    # STUDY SPLIT
    # ------------------------------------------------------------------------

    banner(
        "2. LOCK STUDY-LEVEL SPLIT"
    )

    split_info = (
        validate_study_split()
    )

    print(
        json.dumps(
            split_info,
            indent=2,
        )
    )


    # ------------------------------------------------------------------------
    # DATAFRAMES
    # ------------------------------------------------------------------------

    banner(
        "3. LOAD TRAIN / VALIDATION CSVs"
    )

    train_df, train_image_col, train_label_col = (
        prepare_dataframe(
            TRAIN_CSV
        )
    )

    val_df, val_image_col, val_label_col = (
        prepare_dataframe(
            VAL_CSV
        )
    )

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    print(
        f"Training image column: {train_image_col}"
    )

    print(
        f"Training label column: {train_label_col}"
    )

    print(
        f"Validation image column: {val_image_col}"
    )

    print(
        f"Validation label column: {val_label_col}"
    )


    # ------------------------------------------------------------------------
    # CLASS DISTRIBUTION
    # ------------------------------------------------------------------------

    banner(
        "4. CLASS DISTRIBUTION"
    )

    train_distribution = (
        class_distribution(
            train_df
        )
    )

    val_distribution = (
        class_distribution(
            val_df
        )
    )

    print(
        "Training:"
    )

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"  {class_id} "
            f"{CLASS_NAMES[class_id]:<15} "
            f"{train_distribution.get(class_id, 0)}"
        )

    print()

    print(
        "Validation:"
    )

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"  {class_id} "
            f"{CLASS_NAMES[class_id]:<15} "
            f"{val_distribution.get(class_id, 0)}"
        )


    # ------------------------------------------------------------------------
    # DATASETS
    # ------------------------------------------------------------------------

    banner(
        "5. CREATE DATASETS"
    )

    train_dataset = (
        RSNAClassificationDataset(
            train_df,
            train_image_col,
            train_label_col,
            IMG_SIZE,
        )
    )

    val_dataset = (
        RSNAClassificationDataset(
            val_df,
            val_image_col,
            val_label_col,
            IMG_SIZE,
        )
    )

    print(
        f"Train dataset: {len(train_dataset)}"
    )

    print(
        f"Validation dataset: {len(val_dataset)}"
    )


    # ------------------------------------------------------------------------
    # SAMPLER
    # ------------------------------------------------------------------------

    banner(
        "6. BUILD BALANCED TRAINING SAMPLER"
    )

    (
        sampler,
        sample_weights,
        sampler_info,
    ) = build_sampler(
        train_df
    )

    print(
        "Class sampling weights:"
    )

    for i in range(
        NUM_CLASSES
    ):

        print(
            f"  {CLASS_NAMES[i]:<15}: "
            f"{sampler_info['class_sampling_weights'][i]:.8f}"
        )

    print()

    print(
        "Expected sampled class distribution:"
    )

    for i in range(
        NUM_CLASSES
    ):

        probability = (
            sampler_info[
                "expected_sample_probability"
            ][i]
        )

        print(
            f"  {CLASS_NAMES[i]:<15}: "
            f"{probability:.4f} "
            f"({probability * 100:.2f}%)"
        )

    sampler_distribution_df = pd.DataFrame(
        {
            "class_id":
                list(range(NUM_CLASSES)),

            "class_name":
                CLASS_NAMES,

            "raw_count":
                sampler_info[
                    "raw_class_counts"
                ],

            "sampling_weight":
                sampler_info[
                    "class_sampling_weights"
                ],

            "expected_probability":
                sampler_info[
                    "expected_sample_probability"
                ],
        }
    )

    sampler_distribution_df.to_csv(
        SAMPLER_DISTRIBUTION_CSV,
        index=False,
    )


    # ------------------------------------------------------------------------
    # DATALOADERS
    # ------------------------------------------------------------------------

    banner(
        "7. CREATE DATALOADERS"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        drop_last=False,
    )

    print(
        f"Training batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
    )

    print(
        "Validation sampler: NONE"
    )


    # ------------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------------

    banner(
        "8. BUILD MODEL"
    )

    model = build_model()

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: {total_parameters:,}"
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )


    # ------------------------------------------------------------------------
    # INITIALIZE FROM CORRECTED MODEL
    # ------------------------------------------------------------------------

    banner(
        "9. INITIALIZE FROM CORRECTED CHECKPOINT"
    )

    corrected_sha = (
        load_initialization(
            model
        )
    )

    print(
        "Initialization source:"
    )

    print(
        CORRECTED_CHECKPOINT
    )

    print()

    print(
        "Initialization checkpoint SHA256:"
    )

    print(
        corrected_sha
    )

    print()

    print(
        "Strict model-state loading: PASS"
    )


    # ------------------------------------------------------------------------
    # LOSS
    # ------------------------------------------------------------------------

    banner(
        "10. WEIGHTED CROSS-ENTROPY"
    )

    class_weight_tensor = (
        torch.tensor(
            CLASS_WEIGHTS,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    criterion = (
        nn.CrossEntropyLoss(
            weight=class_weight_tensor
        )
    )

    print(
        "Class weights:"
    )

    for i in range(
        NUM_CLASSES
    ):

        print(
            f"  {CLASS_NAMES[i]:<15}: "
            f"{CLASS_WEIGHTS[i]:.6f}"
        )


    # ------------------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------------------

    banner(
        "11. OPTIMIZER / SCHEDULER"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.10,
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        "Scheduler: CosineAnnealingLR"
    )


    # ------------------------------------------------------------------------
    # AMP
    # ------------------------------------------------------------------------

    scaler = GradScaler(
        enabled=(
            USE_AMP
            and DEVICE.type == "cuda"
        )
    )


    # ------------------------------------------------------------------------
    # SAVE CONFIG
    # ------------------------------------------------------------------------

    run_config = {

        "phase":
            "A",

        "part":
            "3",

        "experiment":
            "study_level_weighted_ce_balanced_sampling",

        "seed":
            SEED,

        "image_size":
            IMG_SIZE,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "num_workers":
            NUM_WORKERS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "class_names":
            CLASS_NAMES,

        "class_weights":
            CLASS_WEIGHTS.tolist(),

        "sampler":
            "WeightedRandomSampler",

        "sampler_power":
            SAMPLER_POWER,

        "selection_metric":
            "macro_f1",

        "secondary_selection_metric":
            "balanced_accuracy",

        "train_samples":
            len(train_df),

        "validation_samples":
            len(val_df),

        "train_studies":
            EXPECTED_TRAIN_STUDIES,

        "validation_studies":
            EXPECTED_VAL_STUDIES,

        "study_overlap":
            0,

        "initialization_checkpoint":
            str(
                CORRECTED_CHECKPOINT
            ),

        "initialization_checkpoint_sha256":
            corrected_sha,

        "original_checkpoint_sha256":
            sha256_file(
                ORIGINAL_CHECKPOINT
            ),

    }

    with RUN_CONFIG_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            run_config,
            f,
            indent=2,
        )


    # ------------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------------

    banner(
        "12. START TARGETED TRAINING"
    )

    print(
        "IMPORTANT: Existing checkpoints will NOT be overwritten."
    )

    print(
        f"New best checkpoint: {A3_CHECKPOINT}"
    )

    print()

    history = []

    best_metrics = None

    best_epoch = None

    best_checkpoint_sha = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()

        print(
            f"---------------- EPOCH "
            f"{epoch}/{EPOCHS} ----------------"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )

        val_loss, metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step()

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        row = {

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_loss":
                val_loss,

            "accuracy":
                metrics["accuracy"],

            "balanced_accuracy":
                metrics[
                    "balanced_accuracy"
                ],

            "macro_precision":
                metrics[
                    "macro_precision"
                ],

            "macro_recall":
                metrics[
                    "macro_recall"
                ],

            "macro_f1":
                metrics[
                    "macro_f1"
                ],

            "learning_rate":
                current_lr,

        }

        history.append(
            row
        )

        print()

        print(
            f"Train loss           : "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss      : "
            f"{val_loss:.6f}"
        )

        print(
            f"Accuracy             : "
            f"{metrics['accuracy']:.6f}"
        )

        print(
            f"Balanced accuracy    : "
            f"{metrics['balanced_accuracy']:.6f}"
        )

        print(
            f"Macro precision      : "
            f"{metrics['macro_precision']:.6f}"
        )

        print(
            f"Macro recall         : "
            f"{metrics['macro_recall']:.6f}"
        )

        print(
            f"Macro F1             : "
            f"{metrics['macro_f1']:.6f}"
        )

        print()

        print(
            "Per-class:"
        )

        for class_name in CLASS_NAMES:

            class_metrics = (
                metrics[
                    "per_class"
                ][class_name]
            )

            print(
                f"  {class_name:<15} "
                f"Precision="
                f"{class_metrics['precision']:.4f} "
                f"Recall="
                f"{class_metrics['recall']:.4f} "
                f"F1="
                f"{class_metrics['f1']:.4f}"
            )


        # ------------------------------------------------------------
        # Best checkpoint
        # ------------------------------------------------------------

        if is_better(
            metrics,
            best_metrics,
        ):

            best_metrics = metrics
            best_epoch = epoch

            checkpoint_payload = {

                "experiment":
                    "study_level_weighted_ce_balanced_sampling",

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "class_weights":
                    CLASS_WEIGHTS.tolist(),

                "split_seed":
                    SEED,

                "study_overlap":
                    0,

                "train_studies":
                    EXPECTED_TRAIN_STUDIES,

                "validation_studies":
                    EXPECTED_VAL_STUDIES,

                "validation_accuracy":
                    metrics["accuracy"],

                "validation_balanced_accuracy":
                    metrics[
                        "balanced_accuracy"
                    ],

                "validation_macro_precision":
                    metrics[
                        "macro_precision"
                    ],

                "validation_macro_recall":
                    metrics[
                        "macro_recall"
                    ],

                "validation_macro_f1":
                    metrics[
                        "macro_f1"
                    ],

                "validation_per_class":
                    metrics[
                        "per_class"
                    ],

                "validation_confusion_matrix":
                    metrics[
                        "confusion_matrix"
                    ],

                "validation_support":
                    metrics[
                        "support"
                    ],

                "sampler":
                    "WeightedRandomSampler",

                "sampler_power":
                    SAMPLER_POWER,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "image_size":
                    IMG_SIZE,

                "batch_size":
                    BATCH_SIZE,

                "initialization_checkpoint":
                    str(
                        CORRECTED_CHECKPOINT
                    ),

                "initialization_checkpoint_sha256":
                    corrected_sha,

                "seed":
                    SEED,

            }

            torch.save(
                checkpoint_payload,
                A3_CHECKPOINT,
            )

            best_checkpoint_sha = (
                sha256_file(
                    A3_CHECKPOINT
                )
            )

            print()

            print(
                "NEW BEST CHECKPOINT SAVED"
            )

            print(
                f"Epoch: {epoch}"
            )

            print(
                f"Macro F1: "
                f"{metrics['macro_f1']:.6f}"
            )

            print(
                f"Balanced accuracy: "
                f"{metrics['balanced_accuracy']:.6f}"
            )

            print(
                f"SHA256: "
                f"{best_checkpoint_sha}"
            )


        # ------------------------------------------------------------
        # Save history continuously
        # ------------------------------------------------------------

        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_CSV,
            index=False,
        )


    # ------------------------------------------------------------------------
    # FINAL RESULTS
    # ------------------------------------------------------------------------

    banner(
        "13. TRAINING COMPLETE"
    )

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_CSV,
        index=False,
    )

    final_row = (
        history[-1]
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print()

    print(
        "Best validation metrics:"
    )

    print(
        f"  Accuracy          : "
        f"{best_metrics['accuracy']:.6f}"
    )

    print(
        f"  Balanced Accuracy : "
        f"{best_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"  Macro Precision   : "
        f"{best_metrics['macro_precision']:.6f}"
    )

    print(
        f"  Macro Recall      : "
        f"{best_metrics['macro_recall']:.6f}"
    )

    print(
        f"  Macro F1          : "
        f"{best_metrics['macro_f1']:.6f}"
    )


    # ------------------------------------------------------------------------
    # SAVE BEST METRICS
    # ------------------------------------------------------------------------

    best_report = {

        "best_epoch":
            best_epoch,

        "best_metrics":
            best_metrics,

        "checkpoint":
            str(
                A3_CHECKPOINT
            ),

        "checkpoint_sha256":
            best_checkpoint_sha,

        "initialization_checkpoint":
            str(
                CORRECTED_CHECKPOINT
            ),

        "initialization_checkpoint_sha256":
            corrected_sha,

        "experiment":
            "study_level_weighted_ce_balanced_sampling",

        "sampler":
            sampler_info,

        "study_split":
            split_info,

    }

    with BEST_METRICS_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            best_report,
            f,
            indent=2,
        )


    # ------------------------------------------------------------------------
    # FINAL METRICS
    # ------------------------------------------------------------------------

    final_report = {

        "final_epoch":
            EPOCHS,

        "final_metrics":
            calculate_metrics(
                # Re-evaluation is not required here because
                # final_row already contains aggregate metrics.
                # Store the final history values.
                [],
                [],
            )
            if False
            else {
                "accuracy":
                    final_row[
                        "accuracy"
                    ],

                "balanced_accuracy":
                    final_row[
                        "balanced_accuracy"
                    ],

                "macro_precision":
                    final_row[
                        "macro_precision"
                    ],

                "macro_recall":
                    final_row[
                        "macro_recall"
                    ],

                "macro_f1":
                    final_row[
                        "macro_f1"
                    ],
            },

        "best_epoch":
            best_epoch,

        "best_metrics":
            best_metrics,

        "checkpoint":
            str(
                A3_CHECKPOINT
            ),

        "checkpoint_sha256":
            best_checkpoint_sha,

    }

    with FINAL_METRICS_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_report,
            f,
            indent=2,
        )


    # ------------------------------------------------------------------------
    # BEST CONFUSION MATRIX
    # ------------------------------------------------------------------------

    best_cm = np.asarray(
        best_metrics[
            "confusion_matrix"
        ]
    )

    cm_df = pd.DataFrame(
        best_cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    cm_df.to_csv(
        CONFUSION_MATRIX_CSV
    )


    # ------------------------------------------------------------------------
    # BEST CLASSIFICATION REPORT
    # ------------------------------------------------------------------------

    report_rows = []

    for class_name in CLASS_NAMES:

        values = (
            best_metrics[
                "per_class"
            ][class_name]
        )

        report_rows.append(
            {
                "class":
                    class_name,

                "precision":
                    values[
                        "precision"
                    ],

                "recall":
                    values[
                        "recall"
                    ],

                "f1":
                    values[
                        "f1"
                    ],

                "support":
                    best_metrics[
                        "support"
                    ][
                        CLASS_NAMES.index(
                            class_name
                        )
                    ],
            }
        )

    report_df = pd.DataFrame(
        report_rows
    )

    report_df.to_csv(
        CLASSIFICATION_REPORT_CSV,
        index=False,
    )


    # ------------------------------------------------------------------------
    # PROJECT VERDICT
    # ------------------------------------------------------------------------

    banner(
        "14. PART A3 MODEL VERDICT"
    )

    best_macro_f1 = (
        best_metrics[
            "macro_f1"
        ]
    )

    best_balanced_accuracy = (
        best_metrics[
            "balanced_accuracy"
        ]
    )

    if (
        best_macro_f1 >= 0.65
        and best_balanced_accuracy >= 0.65
    ):

        verdict = (
            "STRONG_ENOUGH_FOR_FREEZING"
        )

    elif (
        best_macro_f1 >= 0.45
        and best_balanced_accuracy >= 0.45
    ):

        verdict = (
            "SUBSTANTIAL_IMPROVEMENT_BUT_REVIEW"
        )

    else:

        verdict = (
            "STILL_NEEDS_IMPROVEMENT"
        )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best Macro F1: "
        f"{best_macro_f1:.6f}"
    )

    print(
        f"Best Balanced Accuracy: "
        f"{best_balanced_accuracy:.6f}"
    )

    print()

    print(
        f"PART A3 VERDICT: {verdict}"
    )


    # ------------------------------------------------------------------------
    # FINAL SAFETY CHECK
    # ------------------------------------------------------------------------

    banner(
        "15. CHECKPOINT SAFETY"
    )

    print(
        "Original checkpoint remains:"
    )

    print(
        ORIGINAL_CHECKPOINT
    )

    print(
        sha256_file(
            ORIGINAL_CHECKPOINT
        )
    )

    print()

    print(
        "Corrected checkpoint remains:"
    )

    print(
        CORRECTED_CHECKPOINT
    )

    print(
        sha256_file(
            CORRECTED_CHECKPOINT
        )
    )

    print()

    print(
        "A3 checkpoint:"
    )

    print(
        A3_CHECKPOINT
    )

    print(
        sha256_file(
            A3_CHECKPOINT
        )
    )


    # ------------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------------

    banner(
        "PHASE A - PART 3 COMPLETE"
    )

    print(
        "Targeted balanced-sampling training completed."
    )

    print()

    print(
        "Reports:"
    )

    print(
        HISTORY_CSV
    )

    print(
        BEST_METRICS_JSON
    )

    print(
        FINAL_METRICS_JSON
    )

    print(
        CONFUSION_MATRIX_CSV
    )

    print(
        CLASSIFICATION_REPORT_CSV
    )

    print(
        SAMPLER_DISTRIBUTION_CSV
    )

    print(
        RUN_CONFIG_JSON
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT judge this experiment from accuracy alone."
    )

    print(
        "Macro-F1 and balanced accuracy are the primary "
        "classification-quality indicators."
    )


if __name__ == "__main__":

    main()