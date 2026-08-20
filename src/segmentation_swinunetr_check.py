from pathlib import Path
import gc

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.networks.nets import SwinUNETR


# ============================================================
# PHASE 3 - PART 7
# SWIN-UNETR ARCHITECTURE & GPU DRY RUN
# ============================================================

print("=" * 75)
print("PHASE 3 - PART 7")
print("SWIN-UNETR ARCHITECTURE & GPU DRY RUN")
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
    / "architecture"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nOUTPUT DIRECTORY")
print(OUTPUT_DIR)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

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

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA version:",
        torch.version.cuda
    )

    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = (
    96,
    96,
    96
)

IN_CHANNELS = 1

OUT_CHANNELS = 4

BATCH_SIZE = 1


# Conservative architecture for
# RTX 2050 4 GB validation.
#
# We first validate the model with
# feature_size=24.
#
# This is a dry-run configuration,
# NOT the final training configuration.

FEATURE_SIZE = 24

DROPOUT_RATE = 0.0


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
    "Dropout:",
    DROPOUT_RATE
)

print(
    "Batch size:",
    BATCH_SIZE
)


# ============================================================
# MONAI VERSION
# ============================================================

import monai

print(
    "MONAI version:",
    monai.__version__
)

print(
    "PyTorch version:",
    torch.__version__
)


# ============================================================
# MEMORY HELPER
# ============================================================

def gpu_memory_mb():

    if not torch.cuda.is_available():

        return {
            "allocated": 0.0,
            "reserved": 0.0,
        }

    return {
        "allocated":
            torch.cuda.memory_allocated()
            / 1024**2,

        "reserved":
            torch.cuda.memory_reserved()
            / 1024**2,
    }


# ============================================================
# CLEAN GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    torch.cuda.empty_cache()

    gc.collect()


# ============================================================
# CREATE MODEL
# ============================================================

print("\n" + "=" * 75)
print("CREATING SWIN-UNETR")
print("=" * 75)


model = SwinUNETR(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    feature_size=FEATURE_SIZE,
    dropout_path_rate=DROPOUT_RATE,
    use_checkpoint=False,
)


model = model.to(
    DEVICE
)


print(
    "✓ Swin-UNETR model created."
)

print(
    "✓ Model moved to:",
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

print(
    "Approximate parameter memory:",
    f"{total_parameters * 4 / 1024**2:.2f} MB"
)


# ============================================================
# CREATE TEST INPUT
# ============================================================

print("\n" + "=" * 75)
print("TEST INPUT")
print("=" * 75)


input_tensor = torch.rand(
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


print(
    "Input shape:",
    tuple(
        input_tensor.shape
    )
)

print(
    "Input dtype:",
    input_tensor.dtype
)

print(
    "Input device:",
    input_tensor.device
)

print(
    "Input minimum:",
    float(
        input_tensor.min()
    )
)

print(
    "Input maximum:",
    float(
        input_tensor.max()
    )
)


# ============================================================
# MEMORY BEFORE FORWARD
# ============================================================

if torch.cuda.is_available():

    torch.cuda.synchronize()

memory_before = gpu_memory_mb()


print("\n" + "=" * 75)
print("MEMORY BEFORE FORWARD")
print("=" * 75)

print(
    "Allocated:",
    f"{memory_before['allocated']:.2f} MB"
)

print(
    "Reserved:",
    f"{memory_before['reserved']:.2f} MB"
)


# ============================================================
# FORWARD PASS
# ============================================================

print("\n" + "=" * 75)
print("FORWARD PASS")
print("=" * 75)


try:

    with torch.no_grad():

        output = model(
            input_tensor
        )

except RuntimeError as exc:

    print("\nFORWARD PASS FAILED")

    print(exc)

    if torch.cuda.is_available():

        print(
            "\nGPU memory at failure:"
        )

        failure_memory = (
            gpu_memory_mb()
        )

        print(
            "Allocated:",
            f"{failure_memory['allocated']:.2f} MB"
        )

        print(
            "Reserved:",
            f"{failure_memory['reserved']:.2f} MB"
        )

    del model
    del input_tensor

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    gc.collect()

    raise


# ============================================================
# FORWARD RESULT
# ============================================================

if isinstance(
    output,
    (list, tuple)
):

    output_tensor = output[0]

else:

    output_tensor = output


print(
    "Output shape:",
    tuple(
        output_tensor.shape
    )
)

print(
    "Output dtype:",
    output_tensor.dtype
)

print(
    "Output device:",
    output_tensor.device
)


# ============================================================
# EXPECTED OUTPUT
# ============================================================

expected_output_shape = (
    BATCH_SIZE,
    OUT_CHANNELS,
    PATCH_SIZE[0],
    PATCH_SIZE[1],
    PATCH_SIZE[2],
)


print(
    "Expected shape:",
    expected_output_shape
)


if tuple(
    output_tensor.shape
) != expected_output_shape:

    raise RuntimeError(
        "Swin-UNETR output shape mismatch.\n"
        f"Expected: {expected_output_shape}\n"
        f"Received: {tuple(output_tensor.shape)}"
    )


print(
    "✓ Swin-UNETR output shape verified."
)


# ============================================================
# OUTPUT FINITENESS
# ============================================================

if not torch.isfinite(
    output_tensor
).all():

    raise RuntimeError(
        "Model output contains NaN or Inf."
    )


print(
    "✓ Model output contains only finite values."
)


# ============================================================
# MEMORY AFTER FORWARD
# ============================================================

if torch.cuda.is_available():

    torch.cuda.synchronize()

memory_after_forward = (
    gpu_memory_mb()
)


print("\n" + "=" * 75)
print("MEMORY AFTER FORWARD")
print("=" * 75)

print(
    "Allocated:",
    f"{memory_after_forward['allocated']:.2f} MB"
)

print(
    "Reserved:",
    f"{memory_after_forward['reserved']:.2f} MB"
)


# ============================================================
# FORWARD MEMORY INCREASE
# ============================================================

forward_memory_increase = (
    memory_after_forward["allocated"]
    -
    memory_before["allocated"]
)


print(
    "Forward allocated increase:",
    f"{forward_memory_increase:.2f} MB"
)


# ============================================================
# BACKWARD PASS TEST
# ============================================================

print("\n" + "=" * 75)
print("BACKWARD PASS DRY RUN")
print("=" * 75)

print(
    "This is NOT model training."
)

print(
    "A temporary loss is used only to "
    "verify gradient computation."
)


# Delete inference output
del output_tensor
del output


if torch.cuda.is_available():

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()


# ------------------------------------------------------------
# Smaller temporary tensor for gradient test
# ------------------------------------------------------------

gradient_input = torch.rand(
    (
        BATCH_SIZE,
        IN_CHANNELS,
        PATCH_SIZE[0],
        PATCH_SIZE[1],
        PATCH_SIZE[2],
    ),
    dtype=torch.float32,
    device=DEVICE,
    requires_grad=True,
)


try:

    gradient_output = model(
        gradient_input
    )

    if isinstance(
        gradient_output,
        (list, tuple)
    ):

        gradient_output = (
            gradient_output[0]
        )

    temporary_loss = (
        gradient_output.mean()
    )

    temporary_loss.backward()

except RuntimeError as exc:

    print(
        "\nBACKWARD PASS FAILED"
    )

    print(exc)

    del gradient_input

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    gc.collect()

    raise


# ============================================================
# GRADIENT VALIDATION
# ============================================================

gradient_found = False

gradient_nan = False


for parameter in model.parameters():

    if parameter.grad is not None:

        gradient_found = True

        if not torch.isfinite(
            parameter.grad
        ).all():

            gradient_nan = True

            break


if not gradient_found:

    raise RuntimeError(
        "No model gradients were produced."
    )


if gradient_nan:

    raise RuntimeError(
        "NaN or Inf detected in model gradients."
    )


print(
    "✓ Gradients successfully generated."
)

print(
    "✓ No NaN/Inf gradients detected."
)


# ============================================================
# PEAK MEMORY
# ============================================================

if torch.cuda.is_available():

    torch.cuda.synchronize()

    peak_memory = (
        torch.cuda.max_memory_allocated()
        /
        1024**2
    )

else:

    peak_memory = 0.0


print(
    "Peak GPU allocated memory:",
    f"{peak_memory:.2f} MB"
)


# ============================================================
# CLEAN TEMPORARY MODEL
# ============================================================

del temporary_loss
del gradient_output
del gradient_input
del input_tensor
del model


if torch.cuda.is_available():

    torch.cuda.empty_cache()

gc.collect()


# ============================================================
# SAVE ARCHITECTURE REPORT
# ============================================================

architecture_report = (
    OUTPUT_DIR
    /
    "swinunetr_architecture_report.txt"
)


report_text = f"""
PHASE 3 - PART 7
SWIN-UNETR ARCHITECTURE & GPU DRY RUN
======================================

Model
-----
Architecture: Swin-UNETR
MONAI version: {monai.__version__}
PyTorch version: {torch.__version__}

Configuration
-------------
Input channels: {IN_CHANNELS}
Output classes: {OUT_CHANNELS}
Patch size: {PATCH_SIZE}
Feature size: {FEATURE_SIZE}
Dropout: {DROPOUT_RATE}
Batch size: {BATCH_SIZE}

Expected Input
--------------
{(
    BATCH_SIZE,
    IN_CHANNELS,
    PATCH_SIZE[0],
    PATCH_SIZE[1],
    PATCH_SIZE[2],
)}

Expected Output
---------------
{expected_output_shape}

Parameters
----------
Total parameters: {total_parameters:,}
Trainable parameters: {trainable_parameters:,}

GPU
---
Device: {DEVICE}
"""

if torch.cuda.is_available():

    report_text += (
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}\n"
        f"CUDA: "
        f"{torch.version.cuda}\n"
        f"Peak allocated memory: "
        f"{peak_memory:.2f} MB\n"
    )

else:

    report_text += (
        "CUDA unavailable.\n"
    )


report_text += """
Validation
----------

Forward pass: PASSED
Output shape validation: PASSED
Finite output validation: PASSED
Backward gradient validation: PASSED

IMPORTANT
---------
This script performs architecture and
gradient validation only.

No model training was performed.
No checkpoint was modified.
"""


architecture_report.write_text(
    report_text,
    encoding="utf-8"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT")
print("=" * 75)

print(
    "Architecture report:",
    architecture_report
)


print("\n" + "=" * 75)
print("PHASE 3 - PART 7 COMPLETE")
print("=" * 75)