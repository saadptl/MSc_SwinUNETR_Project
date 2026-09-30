from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import pydicom

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# PART 91
# RSNA CLASSIFICATION BASELINE FORENSIC EVALUATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RSNA_ROOT = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

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

CHECKPOINT = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part90_controlled_baseline"
    / "checkpoints"
    / "part90_best_model.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part91_baseline_forensic_evaluation"
)

REPORT_DIR = ROOT / "reports"

PER_TARGET_CSV = (
    OUTPUT_DIR
    / "part91_per_target_metrics.csv"
)

CONFUSION_CSV = (
    OUTPUT_DIR
    / "part91_confusion_matrices.csv"
)

PREDICTIONS_CSV = (
    OUTPUT_DIR
    / "part91_validation_predictions.csv"
)

CLASS_DISTRIBUTION_CSV = (
    OUTPUT_DIR
    / "part91_prediction_class_distribution.csv"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "part91_baseline_forensic_summary.json"
)

REPORT_TXT = (
    REPORT_DIR
    / "part91_baseline_forensic_report.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 1

NUM_WORKERS = 0

TARGET_DEPTH = 32
TARGET_HEIGHT = 224
TARGET_WIDTH = 224

NUM_CLASSES = 3
NUM_TARGETS = 25

USE_AMP = True


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


# ============================================================
# DICOM LOADER
# ============================================================

def discover_dicom_files(
    series_path
):

    path = Path(
        str(series_path)
    )

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

    files.sort(
        key=lambda p: p.name
    )

    return files


def load_dicom_series(
    series_path
):

    files = discover_dicom_files(
        series_path
    )

    if not files:

        raise FileNotFoundError(
            f"No DICOM files found: "
            f"{series_path}"
        )

    slices = []

    for path in files:

        try:

            ds = pydicom.dcmread(
                str(path),
                force=True
            )

            if not hasattr(
                ds,
                "PixelData"
            ):

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

                    z_position = float(
                        position[2]
                    )

                except Exception:

                    z_position = None

            slices.append(
                {
                    "array":
                        np.asarray(
                            image,
                            dtype=np.float32
                        ),

                    "instance_number":
                        instance_number,

                    "z_position":
                        z_position,

                    "path":
                        path,
                }
            )

        except Exception:

            continue

    if not slices:

        raise RuntimeError(
            f"No readable DICOM slices: "
            f"{series_path}"
        )

    # --------------------------------------------------------
    # Slice ordering
    # --------------------------------------------------------

    if all(
        item["z_position"] is not None
        for item in slices
    ):

        slices.sort(
            key=lambda x:
                x["z_position"]
        )

    elif all(
        item["instance_number"] is not None
        for item in slices
    ):

        slices.sort(
            key=lambda x:
                int(x["instance_number"])
        )

    else:

        slices.sort(
            key=lambda x:
                x["path"].name
        )

    # --------------------------------------------------------
    # Harmonize dimensions
    # --------------------------------------------------------

    max_h = max(
        item["array"].shape[0]
        for item in slices
    )

    max_w = max(
        item["array"].shape[1]
        for item in slices
    )

    volume = np.zeros(
        (
            len(slices),
            max_h,
            max_w,
        ),
        dtype=np.float32
    )

    for index, item in enumerate(
        slices
    ):

        image = item["array"]

        h, w = image.shape

        volume[
            index,
            :h,
            :w
        ] = image

    return volume


# ============================================================
# PREPROCESSING
# ============================================================

def normalize_volume(
    volume
):

    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    finite = np.isfinite(
        volume
    )

    if not finite.any():

        raise ValueError(
            "Volume contains no finite values."
        )

    valid = volume[
        finite
    ]

    low = np.percentile(
        valid,
        1.0
    )

    high = np.percentile(
        valid,
        99.0
    )

    if high <= low:

        low = float(
            valid.min()
        )

        high = float(
            valid.max()
        )

    clipped = np.clip(
        volume,
        low,
        high
    )

    if high > low:

        normalized = (
            clipped - low
        ) / (
            high - low
        )

    else:

        normalized = np.zeros_like(
            clipped
        )

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return normalized.astype(
        np.float32
    )


def center_crop_or_pad_3d(
    volume
):

    source = np.asarray(
        volume,
        dtype=np.float32
    )

    src_d, src_h, src_w = (
        source.shape
    )

    output = np.zeros(
        (
            TARGET_DEPTH,
            TARGET_HEIGHT,
            TARGET_WIDTH,
        ),
        dtype=np.float32
    )

    d_start = max(
        0,
        (src_d - TARGET_DEPTH) // 2
    )

    h_start = max(
        0,
        (src_h - TARGET_HEIGHT) // 2
    )

    w_start = max(
        0,
        (src_w - TARGET_WIDTH) // 2
    )

    d_end = min(
        src_d,
        d_start + TARGET_DEPTH
    )

    h_end = min(
        src_h,
        h_start + TARGET_HEIGHT
    )

    w_end = min(
        src_w,
        w_start + TARGET_WIDTH
    )

    cropped = source[
        d_start:d_end,
        h_start:h_end,
        w_start:w_end
    ]

    out_d = max(
        0,
        (TARGET_DEPTH - cropped.shape[0]) // 2
    )

    out_h = max(
        0,
        (TARGET_HEIGHT - cropped.shape[1]) // 2
    )

    out_w = max(
        0,
        (TARGET_WIDTH - cropped.shape[2]) // 2
    )

    output[
        out_d:
        out_d + cropped.shape[0],

        out_h:
        out_h + cropped.shape[1],

        out_w:
        out_w + cropped.shape[2]
    ] = cropped

    return output


def preprocess_mri(
    volume
):

    volume = normalize_volume(
        volume
    )

    volume = center_crop_or_pad_3d(
        volume
    )

    tensor = torch.from_numpy(
        volume
    ).unsqueeze(0)

    return tensor.float()


# ============================================================
# LABEL ENCODING
# ============================================================

def encode_labels(
    row
):

    labels = np.zeros(
        NUM_TARGETS,
        dtype=np.int64
    )

    mask = np.zeros(
        NUM_TARGETS,
        dtype=np.float32
    )

    for index, column in enumerate(
        TARGET_COLUMNS
    ):

        value = row[column]

        if pd.isna(value):

            continue

        value = str(
            value
        ).strip()

        if value not in SEVERITY_TO_INDEX:

            raise ValueError(
                f"Unknown label: "
                f"{value}"
            )

        labels[index] = (
            SEVERITY_TO_INDEX[
                value
            ]
        )

        mask[index] = 1.0

    return (
        torch.from_numpy(labels),
        torch.from_numpy(mask)
    )


# ============================================================
# DATASET
# ============================================================

class RSNAClassificationDataset(
    Dataset
):

    def __init__(
        self,
        manifest
    ):

        self.manifest = (
            manifest
            .reset_index(drop=True)
            .copy()
        )

    def __len__(self):

        return len(
            self.manifest
        )

    def __getitem__(
        self,
        index
    ):

        row = self.manifest.iloc[
            index
        ]

        volume = load_dicom_series(
            row["series_path"]
        )

        image = preprocess_mri(
            volume
        )

        labels, mask = encode_labels(
            row
        )

        return {
            "image":
                image,

            "labels":
                labels,

            "mask":
                mask,

            "study_id":
                str(row["study_id"]),

            "series_id":
                str(row["series_id"]),
        }


def collate_fn(
    batch
):

    return {
        "images":
            torch.stack(
                [
                    item["image"]
                    for item in batch
                ]
            ),

        "labels":
            torch.stack(
                [
                    item["labels"]
                    for item in batch
                ]
            ),

        "mask":
            torch.stack(
                [
                    item["mask"]
                    for item in batch
                ]
            ),

        "study_ids":
            [
                item["study_id"]
                for item in batch
            ],

        "series_ids":
            [
                item["series_id"]
                for item in batch
            ],
    }


# ============================================================
# MODEL
# ============================================================

class ConvBlock3D(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool3d(
                2
            ),
        )

    def forward(
        self,
        x
    ):

        return self.block(x)


class Lightweight3DCNN(
    nn.Module
):

    def __init__(
        self,
        num_targets=25,
        num_classes=3
    ):

        super().__init__()

        self.features = nn.Sequential(

            ConvBlock3D(
                1,
                8
            ),

            ConvBlock3D(
                8,
                16
            ),

            ConvBlock3D(
                16,
                32
            ),

            ConvBlock3D(
                32,
                64
            ),
        )

        self.pool = (
            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Linear(
            64,
            num_targets * num_classes
        )

        self.num_targets = (
            num_targets
        )

        self.num_classes = (
            num_classes
        )

    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = self.pool(
            x
        )

        x = torch.flatten(
            x,
            1
        )

        x = self.classifier(
            x
        )

        x = x.view(
            x.shape[0],
            self.num_targets,
            self.num_classes
        )

        return x


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def safe_divide(
    numerator,
    denominator
):

    if denominator == 0:

        return 0.0

    return (
        numerator
        / denominator
    )


def calculate_class_metrics(
    confusion
):

    metrics = []

    total = (
        confusion.sum()
    )

    for class_index in range(
        NUM_CLASSES
    ):

        tp = confusion[
            class_index,
            class_index
        ]

        fp = (
            confusion[
                :,
                class_index
            ].sum()
            - tp
        )

        fn = (
            confusion[
                class_index,
                :
            ].sum()
            - tp
        )

        tn = (
            total
            - tp
            - fp
            - fn
        )

        precision = safe_divide(
            tp,
            tp + fp
        )

        recall = safe_divide(
            tp,
            tp + fn
        )

        specificity = safe_divide(
            tn,
            tn + fp
        )

        f1 = safe_divide(
            2 * precision * recall,
            precision + recall
        )

        balanced_component = (
            recall
            + specificity
        ) / 2.0

        metrics.append(
            {
                "class_index":
                    class_index,

                "class_name":
                    INDEX_TO_SEVERITY[
                        class_index
                    ],

                "tp":
                    int(tp),

                "fp":
                    int(fp),

                "fn":
                    int(fn),

                "tn":
                    int(tn),

                "precision":
                    precision,

                "recall":
                    recall,

                "specificity":
                    specificity,

                "f1":
                    f1,

                "balanced_component":
                    balanced_component,
            }
        )

    return metrics


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device
):

    model.eval()

    confusion_matrices = [
        np.zeros(
            (
                NUM_CLASSES,
                NUM_CLASSES
            ),
            dtype=np.int64
        )
        for _ in range(
            NUM_TARGETS
        )
    ]

    prediction_rows = []

    total_correct = 0
    total_valid = 0

    total_loss = 0.0
    loss_batches = 0

    for batch_number, batch in enumerate(
        loader,
        start=1
    ):

        images = batch[
            "images"
        ].to(
            device,
            non_blocking=True
        )

        labels = batch[
            "labels"
        ].to(
            device,
            non_blocking=True
        )

        masks = batch[
            "mask"
        ].to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            enabled=(
                USE_AMP
                and device.type == "cuda"
            ),
            dtype=torch.float16,
        ):

            logits = model(
                images
            )

        predictions = torch.argmax(
            logits,
            dim=2
        )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        for sample_index in range(
            images.shape[0]
        ):

            row = {
                "study_id":
                    batch[
                        "study_ids"
                    ][sample_index],

                "series_id":
                    batch[
                        "series_ids"
                    ][sample_index],
            }

            for target_index, target in enumerate(
                TARGET_COLUMNS
            ):

                if (
                    masks[
                        sample_index,
                        target_index
                    ].item()
                    > 0
                ):

                    true_value = int(
                        labels[
                            sample_index,
                            target_index
                        ].item()
                    )

                    pred_value = int(
                        predictions[
                            sample_index,
                            target_index
                        ].item()
                    )

                    row[
                        f"{target}_true"
                    ] = true_value

                    row[
                        f"{target}_pred"
                    ] = pred_value

                    confusion_matrices[
                        target_index
                    ][
                        true_value,
                        pred_value
                    ] += 1

                    total_valid += 1

                    if (
                        true_value
                        == pred_value
                    ):

                        total_correct += 1

                else:

                    row[
                        f"{target}_true"
                    ] = np.nan

                    row[
                        f"{target}_pred"
                    ] = int(
                        predictions[
                            sample_index,
                            target_index
                        ].item()
                    )

            prediction_rows.append(
                row
            )

        if batch_number % 50 == 0:

            print(
                f"    Evaluated "
                f"{batch_number}/"
                f"{len(loader)} batches"
            )

    accuracy = safe_divide(
        total_correct,
        total_valid
    )

    return {
        "confusion_matrices":
            confusion_matrices,

        "prediction_rows":
            prediction_rows,

        "accuracy":
            accuracy,

        "total_valid":
            total_valid,

        "total_correct":
            total_correct,
    }


# ============================================================
# MAJORITY BASELINE
# ============================================================

def calculate_majority_baseline(
    val_manifest
):

    total_correct = 0
    total_valid = 0

    target_rows = []

    for target_index, target in enumerate(
        TARGET_COLUMNS
    ):

        values = []

        for value in (
            val_manifest[target]
        ):

            if pd.isna(value):

                continue

            values.append(
                SEVERITY_TO_INDEX[
                    str(value).strip()
                ]
            )

        if not values:

            continue

        counts = np.bincount(
            values,
            minlength=NUM_CLASSES
        )

        majority_class = int(
            np.argmax(counts)
        )

        correct = int(
            counts[
                majority_class
            ]
        )

        total = int(
            len(values)
        )

        total_correct += correct
        total_valid += total

        target_rows.append(
            {
                "target":
                    target,

                "majority_class":
                    majority_class,

                "majority_class_name":
                    INDEX_TO_SEVERITY[
                        majority_class
                    ],

                "correct":
                    correct,

                "valid":
                    total,

                "accuracy":
                    safe_divide(
                        correct,
                        total
                    ),
            }
        )

    overall = safe_divide(
        total_correct,
        total_valid
    )

    return (
        overall,
        target_rows
    )


# ============================================================
# WRITE CONFUSION MATRIX CSV
# ============================================================

def save_confusion_matrices(
    confusion_matrices
):

    rows = []

    for target_index, target in enumerate(
        TARGET_COLUMNS
    ):

        matrix = (
            confusion_matrices[
                target_index
            ]
        )

        for true_index in range(
            NUM_CLASSES
        ):

            for pred_index in range(
                NUM_CLASSES
            ):

                rows.append(
                    {
                        "target":
                            target,

                        "condition":
                            CONDITIONS[
                                target_index // 5
                            ],

                        "level":
                            LEVELS[
                                target_index % 5
                            ],

                        "true_class":
                            true_index,

                        "true_class_name":
                            INDEX_TO_SEVERITY[
                                true_index
                            ],

                        "predicted_class":
                            pred_index,

                        "predicted_class_name":
                            INDEX_TO_SEVERITY[
                                pred_index
                            ],

                        "count":
                            int(
                                matrix[
                                    true_index,
                                    pred_index
                                ]
                            ),
                    }
                )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        CONFUSION_CSV,
        index=False
    )


# ============================================================
# PER-TARGET METRICS
# ============================================================

def calculate_per_target_metrics(
    confusion_matrices
):

    rows = []

    for target_index, target in enumerate(
        TARGET_COLUMNS
    ):

        matrix = (
            confusion_matrices[
                target_index
            ]
        )

        class_metrics = (
            calculate_class_metrics(
                matrix
            )
        )

        valid = int(
            matrix.sum()
        )

        correct = int(
            np.trace(matrix)
        )

        accuracy = safe_divide(
            correct,
            valid
        )

        macro_precision = float(
            np.mean(
                [
                    item["precision"]
                    for item in class_metrics
                ]
            )
        )

        macro_recall = float(
            np.mean(
                [
                    item["recall"]
                    for item in class_metrics
                ]
            )
        )

        macro_f1 = float(
            np.mean(
                [
                    item["f1"]
                    for item in class_metrics
                ]
            )
        )

        balanced_accuracy = float(
            np.mean(
                [
                    item["balanced_component"]
                    for item in class_metrics
                ]
            )
        )

        condition = CONDITIONS[
            target_index // 5
        ]

        level = LEVELS[
            target_index % 5
        ]

        rows.append(
            {
                "target":
                    target,

                "condition":
                    condition,

                "level":
                    level,

                "valid_samples":
                    valid,

                "accuracy":
                    accuracy,

                "macro_precision":
                    macro_precision,

                "macro_recall":
                    macro_recall,

                "macro_f1":
                    macro_f1,

                "balanced_accuracy":
                    balanced_accuracy,

                "normal_mild_precision":
                    class_metrics[0][
                        "precision"
                    ],

                "normal_mild_recall":
                    class_metrics[0][
                        "recall"
                    ],

                "normal_mild_f1":
                    class_metrics[0][
                        "f1"
                    ],

                "moderate_precision":
                    class_metrics[1][
                        "precision"
                    ],

                "moderate_recall":
                    class_metrics[1][
                        "recall"
                    ],

                "moderate_f1":
                    class_metrics[1][
                        "f1"
                    ],

                "severe_precision":
                    class_metrics[2][
                        "precision"
                    ],

                "severe_recall":
                    class_metrics[2][
                        "recall"
                    ],

                "severe_f1":
                    class_metrics[2][
                        "f1"
                    ],
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        PER_TARGET_CSV,
        index=False
    )

    return dataframe


# ============================================================
# PREDICTION CLASS DISTRIBUTION
# ============================================================

def calculate_prediction_distribution(
    confusion_matrices
):

    rows = []

    for target_index, target in enumerate(
        TARGET_COLUMNS
    ):

        matrix = (
            confusion_matrices[
                target_index
            ]
        )

        true_counts = matrix.sum(
            axis=1
        )

        pred_counts = matrix.sum(
            axis=0
        )

        total_true = int(
            true_counts.sum()
        )

        total_pred = int(
            pred_counts.sum()
        )

        for class_index in range(
            NUM_CLASSES
        ):

            rows.append(
                {
                    "target":
                        target,

                    "condition":
                        CONDITIONS[
                            target_index // 5
                        ],

                    "level":
                        LEVELS[
                            target_index % 5
                        ],

                    "class_index":
                        class_index,

                    "class_name":
                        INDEX_TO_SEVERITY[
                            class_index
                        ],

                    "true_count":
                        int(
                            true_counts[
                                class_index
                            ]
                        ),

                    "true_fraction":
                        safe_divide(
                            true_counts[
                                class_index
                            ],
                            total_true
                        ),

                    "predicted_count":
                        int(
                            pred_counts[
                                class_index
                            ]
                        ),

                    "predicted_fraction":
                        safe_divide(
                            pred_counts[
                                class_index
                            ],
                            total_pred
                        ),
                }
            )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        CLASS_DISTRIBUTION_CSV,
        index=False
    )

    return dataframe


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    banner(
        "PART 91 — BASELINE CLASSIFICATION FORENSIC EVALUATION"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda:0"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Device : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    banner(
        "PATH VALIDATION"
    )

    required_paths = [
        RSNA_ROOT,
        TRAIN_MANIFEST,
        VAL_MANIFEST,
        CHECKPOINT,
    ]

    for path in required_paths:

        print(
            f"{path} : "
            f"{'FOUND' if path.exists() else 'MISSING'}"
        )

        if not path.exists():

            raise FileNotFoundError(
                str(path)
            )

    # --------------------------------------------------------
    # Load manifests
    # --------------------------------------------------------

    banner(
        "LOADING VALIDATION MANIFEST"
    )

    train_manifest = pd.read_csv(
        TRAIN_MANIFEST
    )

    val_manifest = pd.read_csv(
        VAL_MANIFEST
    )

    print(
        f"Training studies   : "
        f"{len(train_manifest)}"
    )

    print(
        f"Validation studies : "
        f"{len(val_manifest)}"
    )

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

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
        train_ids.intersection(
            val_ids
        )
    )

    print(
        f"Study overlap      : "
        f"{len(overlap)}"
    )

    if overlap:

        raise RuntimeError(
            "Study leakage detected."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = (
        RSNAClassificationDataset(
            val_manifest
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=collate_fn,
    )

    print(
        f"Validation batches : "
        f"{len(loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    banner(
        "LOADING PART 90 CHECKPOINT"
    )

    model = Lightweight3DCNN(
        num_targets=NUM_TARGETS,
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        f"Checkpoint epoch : "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Stored val loss  : "
        f"{checkpoint.get('validation_loss')}"
    )

    print(
        f"Stored accuracy  : "
        f"{checkpoint.get('validation_accuracy')}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    banner(
        "EVALUATING PART 90 BEST CHECKPOINT"
    )

    evaluation = evaluate_model(
        model,
        loader,
        device
    )

    print(
        f"\nOverall accuracy : "
        f"{evaluation['accuracy']:.6f}"
    )

    print(
        f"Correct labels   : "
        f"{evaluation['total_correct']}"
    )

    print(
        f"Valid labels     : "
        f"{evaluation['total_valid']}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        evaluation[
            "prediction_rows"
        ]
    )

    prediction_df.to_csv(
        PREDICTIONS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    per_target_df = (
        calculate_per_target_metrics(
            evaluation[
                "confusion_matrices"
            ]
        )
    )

    save_confusion_matrices(
        evaluation[
            "confusion_matrices"
        ]
    )

    distribution_df = (
        calculate_prediction_distribution(
            evaluation[
                "confusion_matrices"
            ]
        )
    )

    # --------------------------------------------------------
    # Majority baseline
    # --------------------------------------------------------

    banner(
        "MAJORITY-CLASS BASELINE"
    )

    (
        majority_accuracy,
        majority_rows
    ) = calculate_majority_baseline(
        val_manifest
    )

    print(
        f"Majority baseline accuracy : "
        f"{majority_accuracy:.6f}"
    )

    print(
        f"Part 90 model accuracy      : "
        f"{evaluation['accuracy']:.6f}"
    )

    print(
        f"Difference                  : "
        f"{evaluation['accuracy'] - majority_accuracy:.6f}"
    )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    macro_accuracy = float(
        per_target_df[
            "accuracy"
        ].mean()
    )

    macro_balanced_accuracy = float(
        per_target_df[
            "balanced_accuracy"
        ].mean()
    )

    macro_f1 = float(
        per_target_df[
            "macro_f1"
        ].mean()
    )

    macro_precision = float(
        per_target_df[
            "macro_precision"
        ].mean()
    )

    macro_recall = float(
        per_target_df[
            "macro_recall"
        ].mean()
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    total_predicted = (
        distribution_df
        .groupby(
            "class_name"
        )[
            "predicted_count"
        ]
        .sum()
    )

    total_true = (
        distribution_df
        .groupby(
            "class_name"
        )[
            "true_count"
        ]
        .sum()
    )

    print(
        "\nTRUE LABEL DISTRIBUTION"
    )

    for class_name in [
        "Normal/Mild",
        "Moderate",
        "Severe",
    ]:

        count = int(
            total_true.get(
                class_name,
                0
            )
        )

        print(
            f"{class_name:<15}: "
            f"{count}"
        )

    print(
        "\nPREDICTED LABEL DISTRIBUTION"
    )

    for class_name in [
        "Normal/Mild",
        "Moderate",
        "Severe",
    ]:

        count = int(
            total_predicted.get(
                class_name,
                0
            )
        )

        print(
            f"{class_name:<15}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # Severe recall
    # --------------------------------------------------------

    severe_recall = float(
        per_target_df[
            "severe_recall"
        ].mean()
    )

    moderate_recall = float(
        per_target_df[
            "moderate_recall"
        ].mean()
    )

    normal_recall = float(
        per_target_df[
            "normal_mild_recall"
        ].mean()
    )

    print(
        "\nMEAN CLASS RECALL"
    )

    print(
        f"Normal/Mild : "
        f"{normal_recall:.6f}"
    )

    print(
        f"Moderate    : "
        f"{moderate_recall:.6f}"
    )

    print(
        f"Severe      : "
        f"{severe_recall:.6f}"
    )

    # --------------------------------------------------------
    # Diagnose imbalance behavior
    # --------------------------------------------------------

    predicted_severe = int(
        total_predicted.get(
            "Severe",
            0
        )
    )

    predicted_moderate = int(
        total_predicted.get(
            "Moderate",
            0
        )
    )

    predicted_normal = int(
        total_predicted.get(
            "Normal/Mild",
            0
        )
    )

    predicted_total = (
        predicted_normal
        + predicted_moderate
        + predicted_severe
    )

    predicted_normal_fraction = (
        safe_divide(
            predicted_normal,
            predicted_total
        )
    )

    if (
        predicted_normal_fraction
        >= 0.95
        and (
            moderate_recall < 0.10
            or severe_recall < 0.10
        )
    ):

        diagnosis = (
            "MAJORITY_CLASS_COLLAPSE"
        )

    elif (
        evaluation["accuracy"]
        <= majority_accuracy + 0.02
        and (
            moderate_recall < 0.20
            or severe_recall < 0.20
        )
    ):

        diagnosis = (
            "BASELINE_ACCURACY_LARGELY_DRIVEN_BY_CLASS_IMBALANCE"
        )

    else:

        diagnosis = (
            "BASELINE_SHOWS_NONTRIVIAL_SEVERITY_SIGNAL"
        )

    print(
        f"\nDIAGNOSIS: {diagnosis}"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_lines = []

    report_lines.append(
        "=" * 78
    )

    report_lines.append(
        "PART 91 — CLASSIFICATION BASELINE FORENSIC EVALUATION"
    )

    report_lines.append(
        "=" * 78
    )

    report_lines.append("")

    report_lines.append(
        "DATASET"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        f"Training studies: "
        f"{len(train_manifest)}"
    )

    report_lines.append(
        f"Validation studies: "
        f"{len(val_manifest)}"
    )

    report_lines.append(
        f"Study overlap: "
        f"{len(overlap)}"
    )

    report_lines.append("")

    report_lines.append(
        "MODEL"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        "Part 90 Lightweight 3D CNN"
    )

    report_lines.append(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch')}"
    )

    report_lines.append("")

    report_lines.append(
        "OVERALL PERFORMANCE"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        f"Model accuracy: "
        f"{evaluation['accuracy']:.6f}"
    )

    report_lines.append(
        f"Majority baseline: "
        f"{majority_accuracy:.6f}"
    )

    report_lines.append(
        f"Model - majority: "
        f"{evaluation['accuracy'] - majority_accuracy:.6f}"
    )

    report_lines.append(
        f"Macro target accuracy: "
        f"{macro_accuracy:.6f}"
    )

    report_lines.append(
        f"Macro balanced accuracy: "
        f"{macro_balanced_accuracy:.6f}"
    )

    report_lines.append(
        f"Macro precision: "
        f"{macro_precision:.6f}"
    )

    report_lines.append(
        f"Macro recall: "
        f"{macro_recall:.6f}"
    )

    report_lines.append(
        f"Macro F1: "
        f"{macro_f1:.6f}"
    )

    report_lines.append("")

    report_lines.append(
        "CLASS RECALL"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        f"Normal/Mild recall: "
        f"{normal_recall:.6f}"
    )

    report_lines.append(
        f"Moderate recall: "
        f"{moderate_recall:.6f}"
    )

    report_lines.append(
        f"Severe recall: "
        f"{severe_recall:.6f}"
    )

    report_lines.append("")

    report_lines.append(
        "PREDICTION DISTRIBUTION"
    )

    report_lines.append(
        "-" * 78
    )

    for class_name in [
        "Normal/Mild",
        "Moderate",
        "Severe",
    ]:

        report_lines.append(
            f"{class_name}: "
            f"{int(total_predicted.get(class_name, 0))}"
        )

    report_lines.append("")

    report_lines.append(
        "DIAGNOSIS"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        diagnosis
    )

    report_lines.append("")

    report_lines.append(
        "INTERPRETATION"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        "Overall accuracy must not be used alone because "
        "the RSNA classification targets are strongly imbalanced."
    )

    report_lines.append(
        "Macro F1, balanced accuracy, and Moderate/Severe "
        "recall are therefore emphasized."
    )

    report_lines.append("")

    report_lines.append(
        "FINAL STATUS"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        "PASS — Part 90 checkpoint forensic evaluation completed."
    )

    report_lines.append(
        "No training was performed."
    )

    report_lines.append(
        "No segmentation checkpoint was modified."
    )

    report = "\n".join(
        report_lines
    )

    with open(
        REPORT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    summary = {

        "part": 91,

        "title":
            "Classification Baseline Forensic Evaluation",

        "seed":
            SEED,

        "checkpoint":
            str(CHECKPOINT),

        "checkpoint_epoch":
            int(
                checkpoint.get(
                    "epoch",
                    -1
                )
            ),

        "dataset": {

            "training_studies":
                int(len(train_manifest)),

            "validation_studies":
                int(len(val_manifest)),

            "study_overlap":
                int(len(overlap)),
        },

        "overall": {

            "accuracy":
                float(
                    evaluation["accuracy"]
                ),

            "majority_baseline_accuracy":
                float(
                    majority_accuracy
                ),

            "difference_from_majority":
                float(
                    evaluation["accuracy"]
                    - majority_accuracy
                ),

            "macro_target_accuracy":
                macro_accuracy,

            "macro_balanced_accuracy":
                macro_balanced_accuracy,

            "macro_precision":
                macro_precision,

            "macro_recall":
                macro_recall,

            "macro_f1":
                macro_f1,

            "valid_labels":
                int(
                    evaluation[
                        "total_valid"
                    ]
                ),
        },

        "class_recall": {

            "normal_mild":
                normal_recall,

            "moderate":
                moderate_recall,

            "severe":
                severe_recall,
        },

        "prediction_distribution": {

            "normal_mild":
                predicted_normal,

            "moderate":
                predicted_moderate,

            "severe":
                predicted_severe,
        },

        "diagnosis":
            diagnosis,

        "outputs": {

            "per_target_metrics":
                str(PER_TARGET_CSV),

            "confusion_matrices":
                str(CONFUSION_CSV),

            "predictions":
                str(PREDICTIONS_CSV),

            "class_distribution":
                str(CLASS_DISTRIBUTION_CSV),

            "report":
                str(REPORT_TXT),

            "summary":
                str(SUMMARY_JSON),
        },

        "training_performed":
            False,

        "segmentation_modified":
            False,

        "status":
            "PASS",
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
    # Final console
    # --------------------------------------------------------

    banner(
        "PART 91 COMPLETE"
    )

    print(
        f"Overall accuracy          : "
        f"{evaluation['accuracy']:.6f}"
    )

    print(
        f"Majority baseline         : "
        f"{majority_accuracy:.6f}"
    )

    print(
        f"Macro balanced accuracy   : "
        f"{macro_balanced_accuracy:.6f}"
    )

    print(
        f"Macro F1                  : "
        f"{macro_f1:.6f}"
    )

    print(
        f"Normal/Mild recall        : "
        f"{normal_recall:.6f}"
    )

    print(
        f"Moderate recall           : "
        f"{moderate_recall:.6f}"
    )

    print(
        f"Severe recall             : "
        f"{severe_recall:.6f}"
    )

    print(
        f"\nDiagnosis: "
        f"{diagnosis}"
    )

    print(
        f"\nPer-target metrics:\n"
        f"{PER_TARGET_CSV}"
    )

    print(
        f"\nConfusion matrices:\n"
        f"{CONFUSION_CSV}"
    )

    print(
        f"\nPredictions:\n"
        f"{PREDICTIONS_CSV}"
    )

    print(
        f"\nClass distribution:\n"
        f"{CLASS_DISTRIBUTION_CSV}"
    )

    print(
        f"\nReport:\n"
        f"{REPORT_TXT}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_JSON}"
    )

    print(
        "\nSTATUS: PASS — "
        "BASELINE FORENSIC EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()