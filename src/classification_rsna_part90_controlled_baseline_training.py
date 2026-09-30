from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import pydicom

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# PART 90
# RSNA CLASSIFICATION CONTROLLED 3D CNN BASELINE
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
    / "rsna_part90_controlled_baseline"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints"
)

REPORT_DIR = ROOT / "reports"

HISTORY_CSV = (
    OUTPUT_DIR
    / "part90_training_history.csv"
)

VAL_RESULTS_CSV = (
    OUTPUT_DIR
    / "part90_validation_results.csv"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "part90_best_model.pth"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "part90_final_model.pth"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "part90_controlled_baseline_summary.json"
)

REPORT_TXT = (
    REPORT_DIR
    / "part90_controlled_baseline_report.txt"
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
# DICOM LOADING
# ============================================================

def discover_dicom_files(series_path):

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


def load_dicom_series(series_path):

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
    # Sort slices
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
    # Harmonize in-plane dimensions
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

    for i, item in enumerate(
        slices
    ):

        image = item["array"]

        h, w = image.shape

        volume[
            i,
            :h,
            :w
        ] = image

    return volume


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_volume(volume):

    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    finite = np.isfinite(
        volume
    )

    if not finite.any():

        raise ValueError(
            "Volume has no finite values."
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


# ============================================================
# CROP / PAD
# ============================================================

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

    out_d_start = max(
        0,
        (TARGET_DEPTH - cropped.shape[0]) // 2
    )

    out_h_start = max(
        0,
        (TARGET_HEIGHT - cropped.shape[1]) // 2
    )

    out_w_start = max(
        0,
        (TARGET_WIDTH - cropped.shape[2]) // 2
    )

    output[
        out_d_start:
        out_d_start + cropped.shape[0],

        out_h_start:
        out_h_start + cropped.shape[1],

        out_w_start:
        out_w_start + cropped.shape[2]
    ] = cropped

    return output


def preprocess_mri(volume):

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

def encode_labels(row):

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
                f"Unknown label '{value}' "
                f"in {column}"
            )

        labels[index] = (
            SEVERITY_TO_INDEX[value]
        )

        mask[index] = 1.0

    return (
        torch.from_numpy(labels),
        torch.from_numpy(mask),
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

        study_id = str(
            row["study_id"]
        )

        series_id = str(
            row["series_id"]
        )

        series_path = Path(
            str(row["series_path"])
        )

        volume = load_dicom_series(
            series_path
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

            "label_mask":
                mask,

            "study_id":
                study_id,

            "series_id":
                series_id,
        }


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    images = torch.stack(
        [
            item["image"]
            for item in batch
        ],
        dim=0
    )

    labels = torch.stack(
        [
            item["labels"]
            for item in batch
        ],
        dim=0
    )

    masks = torch.stack(
        [
            item["label_mask"]
            for item in batch
        ],
        dim=0
    )

    return {
        "images":
            images,

        "labels":
            labels,

        "label_mask":
            masks,

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
# LIGHTWEIGHT 3D CNN
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
                kernel_size=2,
                stride=2
            ),
        )

    def forward(self, x):

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

        self.pool = nn.AdaptiveAvgPool3d(
            1
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

    def forward(self, x):

        x = self.features(
            x
        )

        x = self.pool(
            x
        )

        x = torch.flatten(
            x,
            start_dim=1
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
# MASKED CROSS ENTROPY
# ============================================================

def masked_cross_entropy(
    logits,
    labels,
    mask
):

    batch_size = logits.shape[0]

    total_loss = torch.tensor(
        0.0,
        device=logits.device
    )

    total_valid = torch.tensor(
        0.0,
        device=logits.device
    )

    for target_index in range(
        NUM_TARGETS
    ):

        target_logits = logits[
            :,
            target_index,
            :
        ]

        target_labels = labels[
            :,
            target_index
        ]

        target_mask = mask[
            :,
            target_index
        ]

        valid = target_mask > 0

        if valid.any():

            loss = F.cross_entropy(
                target_logits[valid],
                target_labels[valid],
                reduction="sum"
            )

            total_loss = (
                total_loss
                + loss
            )

            total_valid = (
                total_valid
                + valid.float().sum()
            )

    if total_valid.item() == 0:

        return torch.tensor(
            0.0,
            device=logits.device,
            requires_grad=True
        )

    return (
        total_loss
        / total_valid
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    labels,
    mask
):

    predictions = torch.argmax(
        logits,
        dim=2
    )

    total_correct = 0
    total_valid = 0

    per_target = []

    for target_index in range(
        NUM_TARGETS
    ):

        valid = (
            mask[:, target_index]
            > 0
        )

        if valid.any():

            target_pred = predictions[
                valid,
                target_index
            ]

            target_true = labels[
                valid,
                target_index
            ]

            correct = int(
                (
                    target_pred
                    == target_true
                )
                .sum()
                .item()
            )

            count = int(
                valid.sum().item()
            )

            total_correct += correct
            total_valid += count

            accuracy = (
                correct / count
            )

        else:

            count = 0
            accuracy = float("nan")

        per_target.append(
            accuracy
        )

    overall_accuracy = (
        total_correct / total_valid
        if total_valid > 0
        else 0.0
    )

    valid_target_accuracies = [
        x
        for x in per_target
        if np.isfinite(x)
    ]

    macro_accuracy = (
        float(
            np.mean(
                valid_target_accuracies
            )
        )
        if valid_target_accuracies
        else 0.0
    )

    return {
        "accuracy":
            overall_accuracy,

        "macro_target_accuracy":
            macro_accuracy,

        "valid_labels":
            total_valid,
    }


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch
):

    model.train()

    running_loss = 0.0

    valid_batches = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    start_time = time.time()

    for batch_index, batch in enumerate(
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
            "label_mask"
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

            loss = masked_cross_entropy(
                logits,
                labels,
                masks
            )

            scaled_loss = (
                loss
                / GRADIENT_ACCUMULATION
            )

        scaler.scale(
            scaled_loss
        ).backward()

        if (
            batch_index
            % GRADIENT_ACCUMULATION
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
            float(loss.item())
        )

        valid_batches += 1

        if batch_index % 100 == 0:

            print(
                f"    Epoch {epoch} "
                f"batch {batch_index}/"
                f"{len(loader)} "
                f"loss={loss.item():.5f}"
            )

    # --------------------------------------------------------
    # Handle remaining gradients
    # --------------------------------------------------------

    if (
        valid_batches
        % GRADIENT_ACCUMULATION
        != 0
    ):

        scaler.step(
            optimizer
        )

        scaler.update()

        optimizer.zero_grad(
            set_to_none=True
        )

    elapsed = (
        time.time()
        - start_time
    )

    mean_loss = (
        running_loss
        / max(
            valid_batches,
            1
        )
    )

    return (
        mean_loss,
        elapsed
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0.0

    batch_count = 0

    total_correct = 0

    total_valid = 0

    target_correct = np.zeros(
        NUM_TARGETS,
        dtype=np.int64
    )

    target_total = np.zeros(
        NUM_TARGETS,
        dtype=np.int64
    )

    study_predictions = []

    for batch in loader:

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
            "label_mask"
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

            loss = masked_cross_entropy(
                logits,
                labels,
                masks
            )

        total_loss += (
            float(loss.item())
        )

        batch_count += 1

        predictions = torch.argmax(
            logits,
            dim=2
        )

        for target_index in range(
            NUM_TARGETS
        ):

            valid = (
                masks[
                    :,
                    target_index
                ]
                > 0
            )

            if valid.any():

                pred = predictions[
                    valid,
                    target_index
                ]

                true = labels[
                    valid,
                    target_index
                ]

                correct = int(
                    (
                        pred == true
                    )
                    .sum()
                    .item()
                )

                count = int(
                    valid.sum().item()
                )

                target_correct[
                    target_index
                ] += correct

                target_total[
                    target_index
                ] += count

                total_correct += correct

                total_valid += count

        # Save compact validation prediction
        # records for later inspection.
        for sample_index in range(
            images.shape[0]
        ):

            study_predictions.append(
                {
                    "study_id":
                        batch["study_ids"][
                            sample_index
                        ],

                    "series_id":
                        batch["series_ids"][
                            sample_index
                        ],
                }
            )

    val_loss = (
        total_loss
        / max(
            batch_count,
            1
        )
    )

    accuracy = (
        total_correct
        / total_valid
        if total_valid > 0
        else 0.0
    )

    target_accuracies = []

    for target_index in range(
        NUM_TARGETS
    ):

        if target_total[
            target_index
        ] > 0:

            target_accuracies.append(
                target_correct[
                    target_index
                ]
                / target_total[
                    target_index
                ]
            )

        else:

            target_accuracies.append(
                np.nan
            )

    valid_target_accuracies = [
        x
        for x in target_accuracies
        if np.isfinite(x)
    ]

    macro_accuracy = (
        float(
            np.mean(
                valid_target_accuracies
            )
        )
        if valid_target_accuracies
        else 0.0
    )

    return {
        "loss":
            val_loss,

        "accuracy":
            accuracy,

        "macro_target_accuracy":
            macro_accuracy,

        "valid_labels":
            total_valid,

        "target_accuracies":
            target_accuracies,

        "target_correct":
            target_correct,

        "target_total":
            target_total,

        "study_predictions":
            study_predictions,
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    train_loss,
    val_loss,
    val_accuracy,
    val_macro_accuracy
):

    checkpoint = {

        "part": 90,

        "epoch":
            epoch,

        "seed":
            SEED,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "train_loss":
            train_loss,

        "validation_loss":
            val_loss,

        "validation_accuracy":
            val_accuracy,

        "validation_macro_target_accuracy":
            val_macro_accuracy,

        "config": {

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

            "model":
                "Lightweight3DCNN",
        },
    }

    torch.save(
        checkpoint,
        path
    )


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

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    banner(
        "PART 90 — CONTROLLED 3D CNN CLASSIFICATION BASELINE"
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
        f"Device          : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM            : "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    print(
        f"PyTorch         : "
        f"{torch.__version__}"
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    banner(
        "INPUT VALIDATION"
    )

    required_paths = [
        RSNA_ROOT,
        TRAIN_IMAGES,
        TRAIN_MANIFEST,
        VAL_MANIFEST,
    ]

    for path in required_paths:

        print(
            f"{str(path):<75} "
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
        "LOADING CLASSIFICATION MANIFESTS"
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
    # Leakage check
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

    overlap = train_ids.intersection(
        val_ids
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
    # Datasets
    # --------------------------------------------------------

    banner(
        "CREATING DATASETS"
    )

    train_dataset = (
        RSNAClassificationDataset(
            train_manifest
        )
    )

    val_dataset = (
        RSNAClassificationDataset(
            val_manifest
        )
    )

    print(
        f"Training dataset   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation dataset : "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=collate_fn,
    )

    print(
        f"Training batches   : "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches : "
        f"{len(val_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    banner(
        "CREATING LIGHTWEIGHT 3D CNN"
    )

    model = Lightweight3DCNN(
        num_targets=NUM_TARGETS,
        num_classes=NUM_CLASSES
    )

    model = model.to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters    : "
        f"{parameter_count:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            USE_AMP
            and device.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_val_loss = float(
        "inf"
    )

    best_epoch = None

    banner(
        "STARTING CONTROLLED BASELINE TRAINING"
    )

    print(
        f"Epochs               : {EPOCHS}"
    )

    print(
        f"Batch size            : {BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation : "
        f"{GRADIENT_ACCUMULATION}"
    )

    print(
        f"Learning rate         : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Mixed precision       : "
        f"{USE_AMP}"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        banner(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        train_loss, epoch_time = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                epoch
            )
        )

        validation = validate(
            model,
            val_loader,
            device
        )

        val_loss = (
            validation["loss"]
        )

        val_accuracy = (
            validation["accuracy"]
        )

        val_macro_accuracy = (
            validation[
                "macro_target_accuracy"
            ]
        )

        print(
            "\nEPOCH RESULT"
        )

        print(
            f"Train loss          : "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss     : "
            f"{val_loss:.6f}"
        )

        print(
            f"Validation accuracy : "
            f"{val_accuracy:.6f}"
        )

        print(
            f"Macro target acc.   : "
            f"{val_macro_accuracy:.6f}"
        )

        print(
            f"Valid labels        : "
            f"{validation['valid_labels']}"
        )

        print(
            f"Epoch time          : "
            f"{epoch_time:.2f} sec"
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "validation_loss":
                    val_loss,

                "validation_accuracy":
                    val_accuracy,

                "validation_macro_target_accuracy":
                    val_macro_accuracy,

                "valid_labels":
                    validation[
                        "valid_labels"
                    ],

                "epoch_seconds":
                    epoch_time,
            }
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                scaler,
                epoch,
                train_loss,
                val_loss,
                val_accuracy,
                val_macro_accuracy
            )

            print(
                "\nBEST CHECKPOINT UPDATED"
            )

            print(
                BEST_CHECKPOINT
            )

        # ----------------------------------------------------
        # Clear CUDA cache
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    save_checkpoint(
        FINAL_CHECKPOINT,
        model,
        optimizer,
        scaler,
        EPOCHS,
        history[-1]["train_loss"],
        history[-1]["validation_loss"],
        history[-1]["validation_accuracy"],
        history[-1][
            "validation_macro_target_accuracy"
        ]
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
    # Best model validation
    # --------------------------------------------------------

    banner(
        "BEST CHECKPOINT VERIFICATION"
    )

    best_checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    best_validation = validate(
        model,
        val_loader,
        device
    )

    print(
        f"Best epoch          : "
        f"{best_checkpoint['epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation['loss']:.6f}"
    )

    print(
        f"Best accuracy       : "
        f"{best_validation['accuracy']:.6f}"
    )

    print(
        f"Best macro accuracy : "
        f"{best_validation['macro_target_accuracy']:.6f}"
    )

    # --------------------------------------------------------
    # Target-level validation results
    # --------------------------------------------------------

    target_rows = []

    for index, target in enumerate(
        TARGET_COLUMNS
    ):

        target_total = int(
            best_validation[
                "target_total"
            ][index]
        )

        target_correct = int(
            best_validation[
                "target_correct"
            ][index]
        )

        target_accuracy = (
            target_correct
            / target_total
            if target_total > 0
            else np.nan
        )

        condition = (
            CONDITIONS[
                index // 5
            ]
        )

        level = (
            LEVELS[
                index % 5
            ]
        )

        target_rows.append(
            {
                "target":
                    target,

                "condition":
                    condition,

                "level":
                    level,

                "correct":
                    target_correct,

                "valid_samples":
                    target_total,

                "accuracy":
                    target_accuracy,
            }
        )

    target_df = pd.DataFrame(
        target_rows
    )

    target_df.to_csv(
        VAL_RESULTS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    final_status = (
        "PASS"
        if (
            np.isfinite(
                best_validation["loss"]
            )
            and best_validation[
                "valid_labels"
            ] > 0
        )
        else "REVIEW_REQUIRED"
    )

    report_lines = []

    report_lines.append(
        "=" * 78
    )

    report_lines.append(
        "PART 90 — CONTROLLED 3D CNN CLASSIFICATION BASELINE"
    )

    report_lines.append(
        "=" * 78
    )

    report_lines.append("")

    report_lines.append(
        f"Training studies: {len(train_manifest)}"
    )

    report_lines.append(
        f"Validation studies: {len(val_manifest)}"
    )

    report_lines.append(
        f"Study overlap: {len(overlap)}"
    )

    report_lines.append("")

    report_lines.append(
        "MODEL"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        "Lightweight 3D CNN"
    )

    report_lines.append(
        f"Parameters: {parameter_count:,}"
    )

    report_lines.append(
        f"Input: "
        f"(1, {TARGET_DEPTH}, "
        f"{TARGET_HEIGHT}, {TARGET_WIDTH})"
    )

    report_lines.append(
        f"Targets: {NUM_TARGETS}"
    )

    report_lines.append(
        f"Classes: {NUM_CLASSES}"
    )

    report_lines.append("")

    report_lines.append(
        "TRAINING"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        f"Epochs: {EPOCHS}"
    )

    report_lines.append(
        f"Batch size: {BATCH_SIZE}"
    )

    report_lines.append(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION}"
    )

    report_lines.append(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    report_lines.append(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    report_lines.append(
        f"AMP: {USE_AMP}"
    )

    report_lines.append("")

    report_lines.append(
        "BEST RESULT"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        f"Best epoch: {best_checkpoint['epoch']}"
    )

    report_lines.append(
        f"Validation loss: "
        f"{best_validation['loss']:.6f}"
    )

    report_lines.append(
        f"Validation accuracy: "
        f"{best_validation['accuracy']:.6f}"
    )

    report_lines.append(
        f"Macro target accuracy: "
        f"{best_validation['macro_target_accuracy']:.6f}"
    )

    report_lines.append(
        f"Valid labels: "
        f"{best_validation['valid_labels']}"
    )

    report_lines.append("")

    report_lines.append(
        "CHECKPOINT"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        str(BEST_CHECKPOINT)
    )

    report_lines.append("")

    report_lines.append(
        "FINAL STATUS"
    )

    report_lines.append(
        "-" * 78
    )

    report_lines.append(
        f"{final_status} — "
        "Controlled classification baseline completed."
    )

    report_lines.append("")

    report_lines.append(
        "This experiment establishes a baseline."
    )

    report_lines.append(
        "It does not replace the planned Swin-based "
        "classification architecture."
    )

    report_lines.append(
        "Segmentation checkpoints were not modified."
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

        "part": 90,

        "title":
            "Controlled 3D CNN Classification Baseline",

        "seed":
            SEED,

        "device":
            str(device),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else "CPU"
            ),

        "dataset": {

            "training_studies":
                int(len(train_manifest)),

            "validation_studies":
                int(len(val_manifest)),

            "study_overlap":
                int(len(overlap)),
        },

        "model": {

            "name":
                "Lightweight3DCNN",

            "parameters":
                int(parameter_count),

            "targets":
                NUM_TARGETS,

            "classes":
                NUM_CLASSES,
        },

        "input": {

            "channels":
                1,

            "depth":
                TARGET_DEPTH,

            "height":
                TARGET_HEIGHT,

            "width":
                TARGET_WIDTH,
        },

        "training": {

            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "gradient_accumulation":
                GRADIENT_ACCUMULATION,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "mixed_precision":
                USE_AMP,
        },

        "best_result": {

            "epoch":
                int(
                    best_checkpoint["epoch"]
                ),

            "validation_loss":
                float(
                    best_validation["loss"]
                ),

            "validation_accuracy":
                float(
                    best_validation["accuracy"]
                ),

            "macro_target_accuracy":
                float(
                    best_validation[
                        "macro_target_accuracy"
                    ]
                ),

            "valid_labels":
                int(
                    best_validation[
                        "valid_labels"
                    ]
                ),
        },

        "outputs": {

            "history":
                str(HISTORY_CSV),

            "validation_results":
                str(VAL_RESULTS_CSV),

            "best_checkpoint":
                str(BEST_CHECKPOINT),

            "final_checkpoint":
                str(FINAL_CHECKPOINT),

            "report":
                str(REPORT_TXT),

            "summary":
                str(SUMMARY_JSON),
        },

        "segmentation_modified":
            False,

        "overall_status":
            final_status,
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
        "PART 90 COMPLETE"
    )

    print(
        f"Best epoch          : "
        f"{best_checkpoint['epoch']}"
    )

    print(
        f"Best val loss       : "
        f"{best_validation['loss']:.6f}"
    )

    print(
        f"Best val accuracy   : "
        f"{best_validation['accuracy']:.6f}"
    )

    print(
        f"Best macro accuracy : "
        f"{best_validation['macro_target_accuracy']:.6f}"
    )

    print(
        f"\nHistory:\n{HISTORY_CSV}"
    )

    print(
        f"\nValidation results:\n{VAL_RESULTS_CSV}"
    )

    print(
        f"\nBest checkpoint:\n{BEST_CHECKPOINT}"
    )

    print(
        f"\nFinal checkpoint:\n{FINAL_CHECKPOINT}"
    )

    print(
        f"\nReport:\n{REPORT_TXT}"
    )

    print(
        f"\nSummary:\n{SUMMARY_JSON}"
    )

    print(
        "\nSTATUS: "
        f"{final_status} — "
        "CONTROLLED CLASSIFICATION BASELINE COMPLETE"
    )


if __name__ == "__main__":
    main()