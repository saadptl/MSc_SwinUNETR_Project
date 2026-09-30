from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import SimpleITK as sitk

from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric


# ============================================================
# PHASE 3 - PART 10
# ACTUAL SWIN-UNETR SEGMENTATION TRAINING
# ============================================================

print("=" * 75)
print("PHASE 3 - PART 10")
print("ACTUAL SWIN-UNETR SEGMENTATION TRAINING")
print("=" * 75)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

SPLIT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "segmentation_split"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "training"
)

CHECKPOINT_DIR = (
    OUTPUT_ROOT
    / "checkpoints"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nTRAINING DATA")
print(SPLIT_ROOT)

print("\nOUTPUT DIRECTORY")
print(OUTPUT_ROOT)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


def seed_everything(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior.
    # This can reduce performance slightly but improves
    # reproducibility for the academic experiment.

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = (
    96,
    96,
    96
)

BATCH_SIZE = 1

NUM_WORKERS = 0

NUM_CLASSES = 4

IN_CHANNELS = 1

FEATURE_SIZE = 24

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

PATIENCE = 10

USE_AMP = True

GRADIENT_ACCUMULATION = 1

MAX_GRAD_NORM = 1.0


CLASS_NAMES = {
    0: "Background",
    1: "Vertebrae",
    2: "Spinal Canal",
    3: "Intervertebral Disc",
}


print("\n" + "=" * 75)
print("TRAINING CONFIGURATION")
print("=" * 75)

print(
    "Patch size:",
    PATCH_SIZE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Workers:",
    NUM_WORKERS
)

print(
    "Classes:",
    NUM_CLASSES
)

print(
    "Feature size:",
    FEATURE_SIZE
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Weight decay:",
    WEIGHT_DECAY
)

print(
    "AMP:",
    USE_AMP
)

print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION
)

print(
    "Early stopping patience:",
    PATIENCE
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


print("\n" + "=" * 75)
print("DEVICE")
print("=" * 75)

print(
    "Device:",
    DEVICE
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    total_memory = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / 1024**3
    )

    print(
        f"GPU memory: {total_memory:.2f} GB"
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

else:

    print(
        "CUDA unavailable."
    )

    print(
        "WARNING: Training on CPU will be extremely slow."
    )


# ============================================================
# DATASET
# ============================================================

class SegmentationDataset(
    Dataset
):

    def __init__(
        self,
        split_root: Path,
        split_name: str,
        patch_size=(96, 96, 96),
    ):

        self.split_root = (
            split_root
            / split_name
        )

        self.image_dir = (
            self.split_root
            / "images"
        )

        self.mask_dir = (
            self.split_root
            / "masks"
        )

        self.patch_size = (
            patch_size
        )

        if not self.image_dir.exists():

            raise FileNotFoundError(
                f"Image directory not found:\n"
                f"{self.image_dir}"
            )

        if not self.mask_dir.exists():

            raise FileNotFoundError(
                f"Mask directory not found:\n"
                f"{self.mask_dir}"
            )

        self.image_files = sorted(
            self.image_dir.glob(
                "*.mha"
            )
        )

        mask_lookup = {
            path.name: path
            for path in self.mask_dir.glob(
                "*.mha"
            )
        }

        self.samples = []

        for image_path in self.image_files:

            mask_path = mask_lookup.get(
                image_path.name
            )

            if mask_path is not None:

                self.samples.append(
                    (
                        image_path,
                        mask_path
                    )
                )

        if not self.samples:

            raise RuntimeError(
                f"No valid image/mask pairs "
                f"found for {split_name}."
            )


    def __len__(self):

        return len(
            self.samples
        )


    @staticmethod
    def _load_mha(path):

        image = sitk.ReadImage(
            str(path)
        )

        array = sitk.GetArrayFromImage(
            image
        )

        return array


    @staticmethod
    def _center_crop_or_pad(
        array,
        target_shape,
        pad_value=0,
    ):

        result = np.full(
            target_shape,
            pad_value,
            dtype=array.dtype,
        )

        source_shape = array.shape

        source_slices = []
        target_slices = []

        for source_size, target_size in zip(
            source_shape,
            target_shape
        ):

            if source_size >= target_size:

                source_start = (
                    source_size
                    -
                    target_size
                ) // 2

                source_end = (
                    source_start
                    +
                    target_size
                )

                source_slices.append(
                    slice(
                        source_start,
                        source_end
                    )
                )

                target_slices.append(
                    slice(
                        0,
                        target_size
                    )
                )

            else:

                target_start = (
                    target_size
                    -
                    source_size
                ) // 2

                target_end = (
                    target_start
                    +
                    source_size
                )

                source_slices.append(
                    slice(
                        0,
                        source_size
                    )
                )

                target_slices.append(
                    slice(
                        target_start,
                        target_end
                    )
                )

        result[
            tuple(target_slices)
        ] = array[
            tuple(source_slices)
        ]

        return result


    @staticmethod
    def _normalize_image(array):

        array = array.astype(
            np.float32
        )

        finite_values = array[
            np.isfinite(array)
        ]

        if finite_values.size == 0:

            return np.zeros_like(
                array,
                dtype=np.float32
            )

        low = np.percentile(
            finite_values,
            1.0
        )

        high = np.percentile(
            finite_values,
            99.0
        )

        if high <= low:

            minimum = finite_values.min()

            maximum = finite_values.max()

            if maximum > minimum:

                array = (
                    array
                    -
                    minimum
                ) / (
                    maximum
                    -
                    minimum
                )

            else:

                array = np.zeros_like(
                    array,
                    dtype=np.float32
                )

        else:

            array = np.clip(
                array,
                low,
                high
            )

            array = (
                array
                -
                low
            ) / (
                high
                -
                low
            )

        array = np.nan_to_num(
            array,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        return array.astype(
            np.float32
        )


    def __getitem__(
        self,
        index
    ):

        image_path, mask_path = (
            self.samples[index]
        )

        image = self._load_mha(
            image_path
        )

        mask = self._load_mha(
            mask_path
        )

        # ----------------------------------------------------
        # Verify dimensions
        # ----------------------------------------------------

        if image.shape != mask.shape:

            raise RuntimeError(
                "Image/mask shape mismatch:\n"
                f"{image_path.name}\n"
                f"Image: {image.shape}\n"
                f"Mask: {mask.shape}"
            )

        # ----------------------------------------------------
        # Normalize MRI
        # ----------------------------------------------------

        image = self._normalize_image(
            image
        )

        # ----------------------------------------------------
        # Center crop/pad
        # ----------------------------------------------------

        image = (
            self._center_crop_or_pad(
                image,
                self.patch_size,
                pad_value=0,
            )
        )

        mask = (
            self._center_crop_or_pad(
                mask,
                self.patch_size,
                pad_value=0,
            )
        )

        # ----------------------------------------------------
        # Validate labels
        # ----------------------------------------------------

        mask = mask.astype(
            np.int64
        )

        unique_labels = np.unique(
            mask
        )

        invalid_labels = [
            int(label)
            for label in unique_labels
            if label < 0
            or label >= NUM_CLASSES
        ]

        if invalid_labels:

            raise RuntimeError(
                f"Invalid labels in "
                f"{mask_path.name}: "
                f"{invalid_labels}"
            )

        # ----------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image.copy()
        ).unsqueeze(0)

        mask_tensor = torch.from_numpy(
            mask.copy()
        ).long()

        return (
            image_tensor,
            mask_tensor
        )


# ============================================================
# CREATE DATASETS
# ============================================================

print("\n" + "=" * 75)
print("CREATING DATASETS")
print("=" * 75)


train_dataset = SegmentationDataset(
    SPLIT_ROOT,
    "train",
    PATCH_SIZE,
)

val_dataset = SegmentationDataset(
    SPLIT_ROOT,
    "validation",
    PATCH_SIZE,
)

test_dataset = SegmentationDataset(
    SPLIT_ROOT,
    "test",
    PATCH_SIZE,
)


print(
    "Train samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


print("\n" + "=" * 75)
print("DATALOADERS")
print("=" * 75)

print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)

print(
    "Test batches:",
    len(test_loader)
)


# ============================================================
# VERIFY FIRST SAMPLE
# ============================================================

print("\n" + "=" * 75)
print("FIRST TRAINING SAMPLE")
print("=" * 75)


sample_image, sample_mask = (
    train_dataset[0]
)


print(
    "Image shape:",
    tuple(sample_image.shape)
)

print(
    "Mask shape:",
    tuple(sample_mask.shape)
)

print(
    "Image dtype:",
    sample_image.dtype
)

print(
    "Mask dtype:",
    sample_mask.dtype
)

print(
    "Image min:",
    float(sample_image.min())
)

print(
    "Image max:",
    float(sample_image.max())
)

print(
    "Mask labels:",
    torch.unique(
        sample_mask
    ).tolist()
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\n" + "=" * 75)
print("CREATING SWIN-UNETR")
print("=" * 75)


model = SwinUNETR(
    in_channels=IN_CHANNELS,
    out_channels=NUM_CLASSES,
    feature_size=FEATURE_SIZE,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=False,
    spatial_dims=3,
)

model = model.to(
    DEVICE
)


print(
    "✓ Swin-UNETR created."
)

print(
    "Model device:",
    DEVICE
)


# ============================================================
# MODEL PARAMETERS
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print("\n" + "=" * 75)
print("MODEL PARAMETERS")
print("=" * 75)

print(
    "Total parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)


# ============================================================
# LOSS
# ============================================================

loss_function = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
)


print("\n" + "=" * 75)
print("LOSS")
print("=" * 75)

print(
    "DiceCELoss"
)

print(
    "Dice + Cross Entropy"
)

print(
    "Softmax: enabled"
)

print(
    "One-hot target: enabled"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# AMP
# ============================================================

amp_enabled = (
    USE_AMP
    and
    DEVICE.type == "cuda"
)


if amp_enabled:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


print("\n" + "=" * 75)
print("MIXED PRECISION")
print("=" * 75)

print(
    "AMP enabled:",
    amp_enabled
)


# ============================================================
# METRIC
# ============================================================

dice_metric = DiceMetric(
    include_background=False,
    reduction="mean",
    get_not_nans=False,
)


# ============================================================
# CHECKPOINT PATHS
# ============================================================

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best_model.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "last_model.pth"
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = []

best_val_dice = -1.0

best_epoch = 0

epochs_without_improvement = 0


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    scaler,
):

    model.train()

    running_loss = 0.0

    number_of_batches = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    for batch_index, (
        images,
        masks
    ) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=amp_enabled,
        ):

            outputs = model(
                images
            )

            loss = loss_function(
                outputs,
                masks.unsqueeze(1)
            )

            loss = (
                loss
                /
                GRADIENT_ACCUMULATION
            )

        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

        else:

            loss.backward()

        should_update = (
            (
                batch_index + 1
            )
            %
            GRADIENT_ACCUMULATION
            == 0
        )

        if should_update:

            if scaler is not None:

                scaler.unscale_(
                    optimizer
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM
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

        running_loss += (
            float(loss.detach().item())
            *
            GRADIENT_ACCUMULATION
        )

        number_of_batches += 1

        del (
            images,
            masks,
            outputs,
            loss,
        )

    return (
        running_loss
        /
        max(number_of_batches, 1)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    loss_function,
):

    model.eval()

    running_loss = 0.0

    number_of_batches = 0

    class_dice_values = []

    for images, masks in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=amp_enabled,
        ):

            outputs = model(
                images
            )

            loss = loss_function(
                outputs,
                masks.unsqueeze(1)
            )

        running_loss += (
            float(loss.item())
        )

        number_of_batches += 1

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        batch_class_dice = []

        for class_id in range(
            1,
            NUM_CLASSES
        ):

            prediction_class = (
                predictions == class_id
            )

            target_class = (
                masks == class_id
            )

            intersection = (
                prediction_class
                &
                target_class
            ).sum().float()

            prediction_sum = (
                prediction_class
            ).sum().float()

            target_sum = (
                target_class
            ).sum().float()

            denominator = (
                prediction_sum
                +
                target_sum
            )

            if denominator.item() == 0:

                dice = 1.0

            else:

                dice = (
                    (
                        2.0
                        *
                        intersection
                    )
                    /
                    denominator
                ).item()

            batch_class_dice.append(
                float(dice)
            )

        class_dice_values.append(
            batch_class_dice
        )

        del (
            images,
            masks,
            outputs,
            probabilities,
            predictions,
        )

    mean_loss = (
        running_loss
        /
        max(number_of_batches, 1)
    )

    dice_array = np.asarray(
        class_dice_values,
        dtype=np.float64
    )

    per_class_dice = (
        dice_array.mean(
            axis=0
        )
    )

    mean_dice = float(
        per_class_dice.mean()
    )

    return (
        mean_loss,
        mean_dice,
        per_class_dice,
    )


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 75)
print("STARTING ACTUAL TRAINING")
print("=" * 75)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Best model will be selected using "
    "mean foreground Dice."
)


training_start = time.time()


for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.time()

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    print("\n" + "-" * 75)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print("-" * 75)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        loss_function,
        scaler,
    )


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    val_loss, val_dice, per_class_dice = (
        validate(
            model,
            val_loader,
            loss_function,
        )
    )


    epoch_time = (
        time.time()
        -
        epoch_start
    )


    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    if torch.cuda.is_available():

        peak_memory = (
            torch.cuda.max_memory_allocated()
            /
            1024**3
        )

    else:

        peak_memory = 0.0


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Train Loss      : {train_loss:.6f}"
    )

    print(
        f"Validation Loss : {val_loss:.6f}"
    )

    print(
        f"Mean Dice       : {val_dice:.6f}"
    )


    for class_id in range(
        1,
        NUM_CLASSES
    ):

        print(
            f"{CLASS_NAMES[class_id]:18s}: "
            f"{per_class_dice[class_id - 1]:.6f}"
        )


    print(
        f"Epoch time      : "
        f"{epoch_time / 60:.2f} min"
    )

    print(
        f"Peak GPU memory : "
        f"{peak_memory:.2f} GB"
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history_row = {

        "epoch": epoch,

        "train_loss": train_loss,

        "validation_loss": val_loss,

        "mean_dice": val_dice,

        "vertebrae_dice": (
            per_class_dice[0]
        ),

        "spinal_canal_dice": (
            per_class_dice[1]
        ),

        "intervertebral_disc_dice": (
            per_class_dice[2]
        ),

        "epoch_time_minutes": (
            epoch_time / 60
        ),

        "peak_gpu_memory_gb": (
            peak_memory
        ),
    }


    history.append(
        history_row
    )


    # --------------------------------------------------------
    # SAVE HISTORY AFTER EVERY EPOCH
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        OUTPUT_ROOT
        / "training_history.csv",
        index=False
    )


    # --------------------------------------------------------
    # SAVE LAST CHECKPOINT
    # --------------------------------------------------------

    last_checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "best_val_dice":
            best_val_dice,

        "history":
            history,

        "config": {

            "patch_size":
                PATCH_SIZE,

            "batch_size":
                BATCH_SIZE,

            "num_classes":
                NUM_CLASSES,

            "feature_size":
                FEATURE_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "seed":
                SEED,
        },
    }


    torch.save(
        last_checkpoint,
        LAST_CHECKPOINT
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        best_epoch = epoch

        epochs_without_improvement = 0

        best_checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_val_dice":
                best_val_dice,

            "per_class_dice": {

                "vertebrae":
                    float(per_class_dice[0]),

                "spinal_canal":
                    float(per_class_dice[1]),

                "intervertebral_disc":
                    float(per_class_dice[2]),
            },

            "history":
                history,

            "config": {

                "patch_size":
                    PATCH_SIZE,

                "batch_size":
                    BATCH_SIZE,

                "num_classes":
                    NUM_CLASSES,

                "feature_size":
                    FEATURE_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "seed":
                    SEED,
            },
        }


        torch.save(
            best_checkpoint,
            BEST_CHECKPOINT
        )


        print(
            "\n✓ New best model saved."
        )

        print(
            f"Best validation Dice: "
            f"{best_val_dice:.6f}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement for "
            f"{epochs_without_improvement} "
            f"epoch(s)."
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


    # --------------------------------------------------------
    # CUDA CACHE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# TRAINING COMPLETE
# ============================================================

total_training_time = (
    time.time()
    -
    training_start
)


print("\n" + "=" * 75)
print("TRAINING COMPLETE")
print("=" * 75)

print(
    "Best epoch:",
    best_epoch
)

print(
    f"Best validation Dice: "
    f"{best_val_dice:.6f}"
)

print(
    f"Total training time: "
    f"{total_training_time / 3600:.2f} hours"
)


# ============================================================
# SAVE FINAL HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_path = (
    OUTPUT_ROOT
    / "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ============================================================
# CREATE TRAINING CURVES
# ============================================================

if len(history_df) > 0:

    # --------------------------------------------------------
    # LOSS CURVE
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        history_df["epoch"],
        history_df["validation_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Swin-UNETR Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    loss_curve_path = (
        OUTPUT_ROOT
        / "loss_curve.png"
    )

    plt.savefig(
        loss_curve_path,
        dpi=200
    )

    plt.close()


    # --------------------------------------------------------
    # DICE CURVE
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history_df["epoch"],
        history_df["mean_dice"],
        label="Mean Dice"
    )

    plt.plot(
        history_df["epoch"],
        history_df["vertebrae_dice"],
        label="Vertebrae"
    )

    plt.plot(
        history_df["epoch"],
        history_df["spinal_canal_dice"],
        label="Spinal Canal"
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "intervertebral_disc_dice"
        ],
        label="Intervertebral Disc"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Dice Score"
    )

    plt.title(
        "Swin-UNETR Validation Dice"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    dice_curve_path = (
        OUTPUT_ROOT
        / "dice_curve.png"
    )

    plt.savefig(
        dice_curve_path,
        dpi=200
    )

    plt.close()


# ============================================================
# SAVE TRAINING CONFIGURATION
# ============================================================

training_config = {

    "phase": "Phase 3 - Part 10",

    "model":
        "Swin-UNETR",

    "input_channels":
        IN_CHANNELS,

    "output_classes":
        NUM_CLASSES,

    "class_names":
        CLASS_NAMES,

    "patch_size":
        PATCH_SIZE,

    "feature_size":
        FEATURE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs_requested":
        EPOCHS,

    "epochs_completed":
        len(history),

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "amp":
        amp_enabled,

    "gradient_accumulation":
        GRADIENT_ACCUMULATION,

    "seed":
        SEED,

    "best_epoch":
        best_epoch,

    "best_validation_dice":
        best_val_dice,

    "device":
        str(DEVICE),

    "pytorch_version":
        torch.__version__,

    "cuda_version":
        torch.version.cuda,

    "training_samples":
        len(train_dataset),

    "validation_samples":
        len(val_dataset),

    "test_samples":
        len(test_dataset),
}


config_path = (
    OUTPUT_ROOT
    / "training_config.json"
)


with open(
    config_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        training_config,
        file,
        indent=4,
        default=str
    )


# ============================================================
# TRAINING REPORT
# ============================================================

report_path = (
    OUTPUT_ROOT
    / "training_summary.txt"
)


report = f"""
PHASE 3 - PART 10
SWIN-UNETR SEGMENTATION TRAINING REPORT
========================================

Dataset
-------
Training volumes:
{len(train_dataset)}

Validation volumes:
{len(val_dataset)}

Test volumes:
{len(test_dataset)}

Model
-----
Swin-UNETR

Input channels:
{IN_CHANNELS}

Output classes:
{NUM_CLASSES}

Patch size:
{PATCH_SIZE}

Feature size:
{FEATURE_SIZE}

Training
--------
Requested epochs:
{EPOCHS}

Completed epochs:
{len(history)}

Learning rate:
{LEARNING_RATE}

Weight decay:
{WEIGHT_DECAY}

Batch size:
{BATCH_SIZE}

AMP:
{amp_enabled}

Seed:
{SEED}

Results
-------
Best epoch:
{best_epoch}

Best validation mean foreground Dice:
{best_val_dice:.6f}

Training time:
{total_training_time / 3600:.2f} hours

Checkpoints
-----------
Best model:
{BEST_CHECKPOINT}

Last model:
{LAST_CHECKPOINT}

History:
{history_path}

Configuration:
{config_path}

Important
---------
This training uses the patient-level split generated
during Phase 3 - Part 9.

No patient appears simultaneously in the training,
validation, and test partitions.
"""


report_path.write_text(
    report,
    encoding="utf-8"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print(
    "Best checkpoint:",
    BEST_CHECKPOINT
)

print(
    "Last checkpoint:",
    LAST_CHECKPOINT
)

print(
    "Training history:",
    history_path
)

print(
    "Training configuration:",
    config_path
)

print(
    "Training report:",
    report_path
)

if len(history_df) > 0:

    print(
        "Loss curve:",
        OUTPUT_ROOT
        / "loss_curve.png"
    )

    print(
        "Dice curve:",
        OUTPUT_ROOT
        / "dice_curve.png"
    )


print("\n" + "=" * 75)
print("PHASE 3 - PART 10 COMPLETE")
print("=" * 75)