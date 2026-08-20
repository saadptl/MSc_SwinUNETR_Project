from pathlib import Path
import json
import gc

import torch
import torch.nn as nn

from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss


# ============================================================
# PHASE 3 - PART 8
# MEMORY-SAFE SWIN-UNETR TRAINING CONFIGURATION
# ============================================================

print("=" * 75)
print("PHASE 3 - PART 8")
print("MEMORY-SAFE SWIN-UNETR TRAINING CONFIGURATION")
print("=" * 75)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "training_config"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONFIG_FILE = (
    OUTPUT_DIR
    / "segmentation_training_config.json"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "phase3_part8_training_readiness.txt"
)


print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nOUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================
# DEVICE
# ============================================================

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:

    DEVICE = torch.device(
        "cuda:0"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


print("\n" + "=" * 75)
print("DEVICE")
print("=" * 75)

print(
    "Device:",
    DEVICE
)

if CUDA_AVAILABLE:

    GPU_NAME = (
        torch.cuda.get_device_name(0)
    )

    GPU_MEMORY_GB = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / 1024**3
    )

    print(
        "GPU:",
        GPU_NAME
    )

    print(
        "GPU memory:",
        f"{GPU_MEMORY_GB:.2f} GB"
    )

else:

    GPU_NAME = "CPU"

    GPU_MEMORY_GB = 0.0

    print(
        "GPU: Not available"
    )


# ============================================================
# MODEL CONFIGURATION
# ============================================================

PATCH_SIZE = (
    96,
    96,
    96
)

IN_CHANNELS = 1

OUT_CHANNELS = 4

FEATURE_SIZE = 24

DROPOUT_RATE = 0.0

BATCH_SIZE = 1

NUM_WORKERS = 0


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

GRADIENT_ACCUMULATION_STEPS = 1

USE_AMP = True

USE_GRADIENT_CHECKPOINTING = False

RANDOM_SEED = 42


print("\n" + "=" * 75)
print("MODEL CONFIGURATION")
print("=" * 75)

print(
    "Input channels:",
    IN_CHANNELS
)

print(
    "Output classes:",
    OUT_CHANNELS
)

print(
    "Patch size:",
    PATCH_SIZE
)

print(
    "Feature size:",
    FEATURE_SIZE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Workers:",
    NUM_WORKERS
)


print("\n" + "=" * 75)
print("TRAINING CONFIGURATION")
print("=" * 75)

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
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION_STEPS
)

print(
    "Mixed precision:",
    USE_AMP
)

print(
    "Gradient checkpointing:",
    USE_GRADIENT_CHECKPOINTING
)


# ============================================================
# SEED
# ============================================================

torch.manual_seed(
    RANDOM_SEED
)

if CUDA_AVAILABLE:

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# IMPORT VERSIONS
# ============================================================

import monai

print("\n" + "=" * 75)
print("SOFTWARE")
print("=" * 75)

print(
    "PyTorch:",
    torch.__version__
)

print(
    "MONAI:",
    monai.__version__
)

print(
    "CUDA:",
    torch.version.cuda
)


# ============================================================
# AMP VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("MIXED PRECISION VALIDATION")
print("=" * 75)


if CUDA_AVAILABLE:

    amp_enabled = True

    print(
        "✓ CUDA available."
    )

    print(
        "✓ AMP will be enabled for training."
    )

else:

    amp_enabled = False

    print(
        "⚠ CUDA unavailable."
    )

    print(
        "AMP disabled for CPU."
    )


# ============================================================
# MODEL CREATION
# ============================================================

print("\n" + "=" * 75)
print("CREATING TRAINING MODEL")
print("=" * 75)


model = SwinUNETR(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    feature_size=FEATURE_SIZE,
    dropout_path_rate=DROPOUT_RATE,
    use_checkpoint=USE_GRADIENT_CHECKPOINTING,
)


model = model.to(
    DEVICE
)


print(
    "✓ Model created."
)

print(
    "✓ Model device:",
    DEVICE
)


# ============================================================
# PARAMETER COUNT
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
print("MODEL SIZE")
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
# LOSS FUNCTION
# ============================================================

print("\n" + "=" * 75)
print("SEGMENTATION LOSS")
print("=" * 75)


loss_function = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    include_background=True,
)


print(
    "Loss:",
    "DiceCELoss"
)

print(
    "Components:",
    "Dice + Cross Entropy"
)

print(
    "Softmax:",
    "Enabled"
)

print(
    "One-hot target:",
    "Enabled"
)


# ============================================================
# OPTIMIZER
# ============================================================

print("\n" + "=" * 75)
print("OPTIMIZER")
print("=" * 75)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


print(
    "Optimizer:",
    "AdamW"
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Weight decay:",
    WEIGHT_DECAY
)


# ============================================================
# AMP SCALER
# ============================================================

if CUDA_AVAILABLE:

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    print(
        "✓ CUDA GradScaler created."
    )

else:

    scaler = None

    print(
        "GradScaler:",
        "Disabled"
    )


# ============================================================
# TEMPORARY TRAINING-SHAPE TEST
# ============================================================

print("\n" + "=" * 75)
print("TRAINING-SHAPE TEST")
print("=" * 75)


test_input = torch.rand(
    (
        BATCH_SIZE,
        IN_CHANNELS,
        PATCH_SIZE[0],
        PATCH_SIZE[1],
        PATCH_SIZE[2],
    ),
    dtype=torch.float32,
    device=DEVICE,
)

test_target = torch.randint(
    low=0,
    high=OUT_CHANNELS,
    size=(
        BATCH_SIZE,
        PATCH_SIZE[0],
        PATCH_SIZE[1],
        PATCH_SIZE[2],
    ),
    dtype=torch.long,
    device=DEVICE,
)


print(
    "Input:",
    tuple(test_input.shape)
)

print(
    "Target:",
    tuple(test_target.shape)
)

print(
    "Target labels:",
    torch.unique(
        test_target
    ).detach().cpu().tolist()
)


# ============================================================
# TRAINING STEP DRY RUN
# ============================================================

print("\n" + "=" * 75)
print("TRAINING STEP DRY RUN")
print("=" * 75)

print(
    "This is NOT training the project model."
)

print(
    "No checkpoint will be saved."
)

print(
    "The optimizer update is reversed after validation."
)


if CUDA_AVAILABLE:

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()


optimizer.zero_grad(
    set_to_none=True
)


try:

    if CUDA_AVAILABLE:

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            test_output = model(
                test_input
            )

            test_loss = loss_function(
                test_output,
                test_target.unsqueeze(1),
            )

    else:

        test_output = model(
            test_input
        )

        test_loss = loss_function(
            test_output,
            test_target.unsqueeze(1),
        )


    print(
        "Output:",
        tuple(
            test_output.shape
        )
    )

    print(
        "Loss:",
        float(
            test_loss.detach().cpu()
        )
    )


    if CUDA_AVAILABLE:

        scaler.scale(
            test_loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

    else:

        test_loss.backward()

        optimizer.step()


    print(
        "✓ Forward pass successful."
    )

    print(
        "✓ Loss calculation successful."
    )

    print(
        "✓ Backward pass successful."
    )

    print(
        "✓ Temporary optimizer step successful."
    )


except RuntimeError as exc:

    print(
        "\nTRAINING STEP FAILED"
    )

    print(exc)

    if CUDA_AVAILABLE:

        current_memory = (
            torch.cuda.memory_allocated()
            / 1024**2
        )

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        print(
            "Current GPU memory:",
            f"{current_memory:.2f} MB"
        )

        print(
            "Peak GPU memory:",
            f"{peak_memory:.2f} MB"
        )

    raise


# ============================================================
# GPU MEMORY
# ============================================================

if CUDA_AVAILABLE:

    torch.cuda.synchronize()

    peak_memory_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

else:

    peak_memory_mb = 0.0


print("\n" + "=" * 75)
print("TRAINING MEMORY")
print("=" * 75)

if CUDA_AVAILABLE:

    print(
        "Peak allocated GPU memory:",
        f"{peak_memory_mb:.2f} MB"
    )

    print(
        "Total GPU memory:",
        f"{GPU_MEMORY_GB * 1024:.2f} MB"
    )

    memory_percentage = (
        peak_memory_mb
        /
        (GPU_MEMORY_GB * 1024)
        *
        100
    )

    print(
        "Peak memory usage:",
        f"{memory_percentage:.2f}%"
    )

else:

    memory_percentage = 0.0

    print(
        "GPU memory test unavailable."
    )


# ============================================================
# MEMORY SAFETY ASSESSMENT
# ============================================================

print("\n" + "=" * 75)
print("MEMORY SAFETY ASSESSMENT")
print("=" * 75)


if CUDA_AVAILABLE:

    if memory_percentage < 90:

        memory_status = "SAFE"

        print(
            "✓ Configuration is within "
            "the conservative memory target."
        )

    elif memory_percentage < 97:

        memory_status = "BORDERLINE"

        print(
            "⚠ Configuration is close "
            "to the GPU memory limit."
        )

    else:

        memory_status = "UNSAFE"

        print(
            "✗ Configuration is too close "
            "to the GPU memory limit."
        )

else:

    memory_status = "CPU_ONLY"

    print(
        "⚠ GPU unavailable."
    )


# ============================================================
# OUTPUT SHAPE VALIDATION
# ============================================================

expected_output_shape = (
    BATCH_SIZE,
    OUT_CHANNELS,
    PATCH_SIZE[0],
    PATCH_SIZE[1],
    PATCH_SIZE[2],
)


if tuple(
    test_output.shape
) != expected_output_shape:

    raise RuntimeError(
        "Training output shape mismatch.\n"
        f"Expected: {expected_output_shape}\n"
        f"Received: {tuple(test_output.shape)}"
    )


print(
    "✓ Training output shape verified."
)


# ============================================================
# CLEAN TEMPORARY OBJECTS
# ============================================================

del test_output
del test_loss
del test_input
del test_target
del model
del optimizer
del loss_function

if scaler is not None:

    del scaler


if CUDA_AVAILABLE:

    torch.cuda.empty_cache()

gc.collect()


# ============================================================
# CONFIGURATION DICTIONARY
# ============================================================

configuration = {

    "phase": "Phase 3 - Part 8",

    "architecture": "Swin-UNETR",

    "input_channels": IN_CHANNELS,

    "output_classes": OUT_CHANNELS,

    "classes": {
        "0": "Background",
        "1": "Vertebrae",
        "2": "Spinal Canal",
        "3": "Intervertebral Disc",
    },

    "patch_size": list(
        PATCH_SIZE
    ),

    "batch_size": BATCH_SIZE,

    "feature_size": FEATURE_SIZE,

    "dropout_rate": DROPOUT_RATE,

    "epochs": EPOCHS,

    "learning_rate": LEARNING_RATE,

    "weight_decay": WEIGHT_DECAY,

    "optimizer": "AdamW",

    "loss": "DiceCELoss",

    "loss_components": [
        "Dice",
        "Cross Entropy",
    ],

    "mixed_precision": amp_enabled,

    "gradient_accumulation_steps":
        GRADIENT_ACCUMULATION_STEPS,

    "gradient_checkpointing":
        USE_GRADIENT_CHECKPOINTING,

    "num_workers": NUM_WORKERS,

    "random_seed": RANDOM_SEED,

    "device": str(DEVICE),

    "gpu": GPU_NAME,

    "gpu_memory_gb": GPU_MEMORY_GB,

    "pytorch_version":
        torch.__version__,

    "monai_version":
        monai.__version__,

    "cuda_version":
        torch.version.cuda,

    "total_parameters":
        total_parameters,

    "trainable_parameters":
        trainable_parameters,

    "dry_run_output_shape":
        list(
            expected_output_shape
        ),

    "peak_memory_mb":
        peak_memory_mb,

    "peak_memory_percentage":
        memory_percentage,

    "memory_status":
        memory_status,

    "training_started":
        False,

    "checkpoint_modified":
        False,
}


# ============================================================
# SAVE JSON CONFIGURATION
# ============================================================

CONFIG_FILE.write_text(
    json.dumps(
        configuration,
        indent=4,
    ),
    encoding="utf-8",
)


# ============================================================
# SAVE HUMAN-READABLE REPORT
# ============================================================

report = f"""
PHASE 3 - PART 8
MEMORY-SAFE SWIN-UNETR TRAINING CONFIGURATION
==============================================

PROJECT
-------
Project root:
{PROJECT_ROOT}

MODEL
-----
Architecture: Swin-UNETR
Input channels: {IN_CHANNELS}
Output classes: {OUT_CHANNELS}
Feature size: {FEATURE_SIZE}
Dropout: {DROPOUT_RATE}

PATCH
-----
Patch size: {PATCH_SIZE}
Batch size: {BATCH_SIZE}

CLASSES
-------
0 = Background
1 = Vertebrae
2 = Spinal Canal
3 = Intervertebral Disc

TRAINING
--------
Epochs: {EPOCHS}
Optimizer: AdamW
Learning rate: {LEARNING_RATE}
Weight decay: {WEIGHT_DECAY}
Loss: DiceCELoss
Loss components: Dice + Cross Entropy

MEMORY OPTIMIZATION
-------------------
Mixed precision: {amp_enabled}
Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}
Gradient checkpointing: {USE_GRADIENT_CHECKPOINTING}
Workers: {NUM_WORKERS}

HARDWARE
--------
Device: {DEVICE}
GPU: {GPU_NAME}
GPU memory: {GPU_MEMORY_GB:.2f} GB

VALIDATION
----------
Expected model output:
{expected_output_shape}

Actual model output:
{tuple(test_output.shape) if 'test_output' in locals() else 'validated'}

Forward pass: PASSED
Loss calculation: PASSED
Backward pass: PASSED
Optimizer dry-run step: PASSED
Output shape: PASSED

GPU MEMORY
----------
Peak allocated memory:
{peak_memory_mb:.2f} MB

Peak memory percentage:
{memory_percentage:.2f}%

Memory status:
{memory_status}

IMPORTANT
---------
This Part 8 script does NOT train the segmentation model.

No segmentation checkpoint was created.
No existing classification checkpoint was modified.

The optimizer step was executed only as a temporary
dry-run validation and all temporary model objects
were deleted afterward.
"""


REPORT_FILE.write_text(
    report,
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print(
    "Configuration:",
    CONFIG_FILE
)

print(
    "Readiness report:",
    REPORT_FILE
)


print("\n" + "=" * 75)
print("PHASE 3 - PART 8 COMPLETE")
print("=" * 75)