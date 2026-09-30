"""
PHASE 4 - PART 10
RSNA SWIN-UNETR TRAINING PREFLIGHT / GPU FEASIBILITY

Purpose
-------
Perform a safe, deterministic preflight before full RSNA segmentation
training.

This part:
1. Loads the Part 8 RSNA-only manifests.
2. Loads representative DICOM + pseudo-mask samples using the Part 9 loader.
3. Tests several 3D preprocessing configurations.
4. Builds a 6-class Swin-UNETR.
5. Runs forward pass + DiceCE loss + backward pass.
6. Tests AMP when CUDA is available.
7. Measures peak GPU memory.
8. Reports the safest configuration for the RTX 2050 4 GB GPU.

NO full training is performed.
NO checkpoint is saved.
NO model weights are permanently modified.
SPIDER IS NOT USED.

Expected classes:
0 Background
1 Spinal Canal Stenosis
2 Left Neural Foraminal Narrowing
3 Right Neural Foraminal Narrowing
4 Left Subarticular Stenosis
5 Right Subarticular Stenosis
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR


# ============================================================================
# PATHS
# ============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES_DIR = RSNA_ROOT / "train_images"

PART8_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
)

MANIFEST_DIR = PART8_DIR / "manifests"

PART9_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part9_3d_dataset_loader"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part10_training_preflight"
)

REPORT_DIR = OUTPUT_DIR / "reports"


TRAIN_MANIFEST = MANIFEST_DIR / "rsna_part8_train_manifest.csv"
VAL_MANIFEST = MANIFEST_DIR / "rsna_part8_validation_manifest.csv"
TEST_MANIFEST = MANIFEST_DIR / "rsna_part8_test_manifest.csv"


# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_CLASSES = 6
IN_CHANNELS = 1

# RTX 2050 target.
GPU_MEMORY_GB_TARGET = 4.0

# Preflight does not train for multiple iterations.
# It tests one real forward/backward step per candidate.
CANDIDATE_PATCHES = [
    (48, 64, 64),
    (64, 64, 64),
    (64, 80, 80),
    (64, 96, 96),
]

# SwinUNETR requires spatial dimensions compatible with its hierarchy.
# feature_size=12 is deliberately conservative for a 4 GB GPU.
FEATURE_SIZE = 12

BATCH_SIZE = 1

USE_AMP = True

# Number of representative cases used for preprocessing checks.
NUM_SAMPLE_CASES = 3

# Limit CPU threads so the preflight does not monopolize the machine.
CPU_THREADS = min(4, max(1, torch.get_num_threads()))

# Only a very small number of candidate configurations are tested.
# The goal is feasibility, not benchmark optimization.
FORWARD_TIMEOUT_SECONDS = 600


# ============================================================================
# PRINTING
# ============================================================================

def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================================
# IMPORT PART 9 LOADER
# ============================================================================

def load_part9_module():
    """
    Import the Part 9 loader from src.

    This avoids duplicating DICOM/NPZ loading logic and keeps Part 10
    consistent with the validated Part 9 pipeline.
    """

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    try:
        import segmentation_rsna_part9_3d_dataset_loader as part9
    except Exception as exc:
        raise RuntimeError(
            "Could not import Part 9 loader. "
            "Make sure segmentation_rsna_part9_3d_dataset_loader.py "
            "exists in src."
        ) from exc

    return part9


# ============================================================================
# PREPROCESSING
# ============================================================================

def resize_3d(
    array: np.ndarray,
    target_shape: Tuple[int, int, int],
    is_mask: bool,
) -> np.ndarray:
    """
    Resize a [D,H,W] array to target_shape.

    Image: trilinear interpolation.
    Mask : nearest-neighbor interpolation.
    """

    if tuple(array.shape) == tuple(target_shape):
        return array.copy()

    tensor = torch.from_numpy(
        array.astype(np.float32, copy=False)
    ).unsqueeze(0).unsqueeze(0)

    if is_mask:
        resized = F.interpolate(
            tensor,
            size=target_shape,
            mode="nearest",
        )
    else:
        resized = F.interpolate(
            tensor,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )

    result = resized.squeeze(0).squeeze(0).cpu().numpy()

    if is_mask:
        return np.rint(result).astype(np.int64)

    return result.astype(np.float32)


def center_crop_or_pad(
    array: np.ndarray,
    target_shape: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Center crop or zero-pad a [D,H,W] array.

    This function is included for diagnostic comparison. The main
    preflight uses deterministic resizing so that all candidates have
    a fixed input size.
    """

    result = np.full(
        target_shape,
        pad_value,
        dtype=array.dtype,
    )

    source_slices = []
    target_slices = []

    for src, target in zip(
        array.shape,
        target_shape,
    ):

        if src >= target:
            start = (src - target) // 2
            source_slices.append(
                slice(
                    start,
                    start + target,
                )
            )
            target_slices.append(
                slice(0, target)
            )

        else:
            target_start = (target - src) // 2
            source_slices.append(
                slice(0, src)
            )
            target_slices.append(
                slice(
                    target_start,
                    target_start + src,
                )
            )

    result[
        tuple(target_slices)
    ] = array[
        tuple(source_slices)
    ]

    return result


def preprocess_sample(
    image: np.ndarray,
    mask: np.ndarray,
    target_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic Part 10 preprocessing.

    Resize image with trilinear interpolation and mask with nearest
    neighbor interpolation.
    """

    image_out = resize_3d(
        image,
        target_shape,
        is_mask=False,
    )

    mask_out = resize_3d(
        mask,
        target_shape,
        is_mask=True,
    )

    mask_out = np.clip(
        mask_out,
        0,
        NUM_CLASSES - 1,
    ).astype(np.int64)

    image_out = np.nan_to_num(
        image_out,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    image_out = np.clip(
        image_out,
        0.0,
        1.0,
    ).astype(np.float32)

    return image_out, mask_out


# ============================================================================
# ONE-HOT TARGET
# ============================================================================

def mask_to_one_hot(
    mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Convert [B,D,H,W] integer target to [B,C,D,H,W].
    """

    return F.one_hot(
        mask.long(),
        num_classes=num_classes,
    ).permute(
        0,
        4,
        1,
        2,
        3,
    ).float()


# ============================================================================
# MODEL
# ============================================================================

def create_model(
    device: torch.device,
) -> torch.nn.Module:
    """
    Create the MONAI Swin-UNETR model.

    The installed MONAI version does not accept img_size in the
    SwinUNETR constructor, so spatial dimensions are controlled by
    the input tensor rather than by a constructor argument.
    """

    model = SwinUNETR(
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        use_checkpoint=False,
        spatial_dims=3,
    )

    model = model.to(device)

    return model


# ============================================================================
# MEMORY
# ============================================================================

def reset_cuda_memory() -> None:

    if torch.cuda.is_available():

        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        torch.cuda.empty_cache()

        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    gc.collect()


def gpu_memory_stats() -> Dict[str, float]:

    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
        }

    return {
        "allocated_gb": (
            torch.cuda.memory_allocated()
            / (1024 ** 3)
        ),
        "reserved_gb": (
            torch.cuda.memory_reserved()
            / (1024 ** 3)
        ),
        "peak_allocated_gb": (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        ),
        "peak_reserved_gb": (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        ),
    }


# ============================================================================
# MODEL PARAMETER COUNT
# ============================================================================

def parameter_summary(
    model: torch.nn.Module,
) -> Dict[str, int]:

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total": int(total),
        "trainable": int(trainable),
    }


# ============================================================================
# LOAD REPRESENTATIVE CASES
# ============================================================================

def load_representative_cases(
    part9: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load representative cases through the actual Part 9 public API.

    Part 9 does not expose an RSNASegmentationDataset class.
    It exposes load_case(row), which is the validated loader.
    """

    manifest_paths = {
        "train": TRAIN_MANIFEST,
        "validation": VAL_MANIFEST,
        "test": TEST_MANIFEST,
    }

    cases = {}

    for split, manifest_path in manifest_paths.items():

        manifest = pd.read_csv(manifest_path)

        count = min(
            NUM_SAMPLE_CASES,
            len(manifest),
        )

        if count == 0:
            cases[split] = []
            continue

        indices = np.linspace(
            0,
            len(manifest) - 1,
            count,
            dtype=int,
        )

        split_cases = []

        for index in indices:

            index = int(index)
            row = manifest.iloc[index]

            image, mask, info = part9.load_case(row)

            sample = {
                "image": image,
                "mask": mask,
                "study_id": str(info["study_id"]),
                "series_id": str(info["series_id"]),
                "series_description": str(
                    info.get("series_description", "")
                ),
                "raw_mask_shape": str(
                    info.get("raw_mask_shape", "")
                ),
                "alignment_mode": str(
                    info.get("alignment_mode", "")
                ),
                "aligned_slice_index": info.get(
                    "aligned_slice_index",
                    None,
                ),
                "volume_shape": str(
                    info.get(
                        "volume_shape",
                        str(image.shape),
                    )
                ),
            }

            split_cases.append(
                {
                    "index": index,
                    "sample": sample,
                }
            )

        cases[split] = split_cases

    return cases


# ============================================================================
# PREPROCESSING SANITY
# ============================================================================

def preprocessing_sanity(
    cases: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:

    rows = []

    for split, split_cases in cases.items():

        for case in split_cases:

            sample = case["sample"]

            row = {
                "split": split,
                "index": case["index"],
                "study_id": sample["study_id"],
                "series_id": sample["series_id"],
                "series_description": sample[
                    "series_description"
                ],
                "native_shape": str(
                    tuple(sample["image"].shape)
                ),
            }

            for target in CANDIDATE_PATCHES:

                start = time.perf_counter()

                image, mask = preprocess_sample(
                    sample["image"],
                    sample["mask"],
                    target,
                )

                elapsed = (
                    time.perf_counter()
                    - start
                )

                labels = sorted(
                    np.unique(mask).astype(
                        int
                    ).tolist()
                )

                row[
                    f"{target}_image_shape"
                ] = str(
                    tuple(image.shape)
                )

                row[
                    f"{target}_mask_shape"
                ] = str(
                    tuple(mask.shape)
                )

                row[
                    f"{target}_labels"
                ] = str(labels)

                row[
                    f"{target}_foreground"
                ] = int(
                    np.count_nonzero(mask > 0)
                )

                row[
                    f"{target}_preprocess_seconds"
                ] = round(
                    elapsed,
                    4,
                )

            rows.append(row)

    return rows


# ============================================================================
# SINGLE MODEL PREFLIGHT
# ============================================================================

def run_candidate(
    target_shape: Tuple[int, int, int],
    case: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:

    sample = case["sample"]

    result: Dict[str, Any] = {
        "target_shape": str(
            target_shape
        ),
        "study_id": sample[
            "study_id"
        ],
        "series_id": sample[
            "series_id"
        ],
        "series_description": sample[
            "series_description"
        ],
        "status": "FAIL",
        "forward_seconds": None,
        "backward_seconds": None,
        "total_seconds": None,
        "loss": None,
        "peak_allocated_gb": None,
        "peak_reserved_gb": None,
        "error": "",
    }

    reset_cuda_memory()

    model = None
    optimizer = None
    scaler = None

    try:

        print()
        print(
            f"Candidate patch: "
            f"{target_shape}"
        )

        # ---------------------------------------------------------------
        # PREPROCESS
        # ---------------------------------------------------------------

        image, mask = preprocess_sample(
            sample["image"],
            sample["mask"],
            target_shape,
        )

        image_tensor = torch.from_numpy(
            image
        ).unsqueeze(0).unsqueeze(0)

        mask_tensor = torch.from_numpy(
            mask
        ).unsqueeze(0).long()

        image_tensor = image_tensor.to(
            device,
            non_blocking=True,
        )

        mask_tensor = mask_tensor.to(
            device,
            non_blocking=True,
        )

        # ---------------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------------

        print(
            "  Creating Swin-UNETR..."
        )

        model = create_model(
            device
        )

        params = parameter_summary(
            model
        )

        print(
            f"  Parameters: "
            f"{params['trainable']:,}"
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
        )

        loss_function = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=True,
        )

        # ---------------------------------------------------------------
        # FORWARD
        # ---------------------------------------------------------------

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        with torch.autocast(
            device_type="cuda"
            if device.type == "cuda"
            else "cpu",
            dtype=torch.float16
            if device.type == "cuda"
            else torch.bfloat16,
            enabled=(
                USE_AMP
                and device.type == "cuda"
            ),
        ):

            output = model(
                image_tensor
            )

            loss = loss_function(
                output,
                mask_tensor.unsqueeze(1),
            )

        forward_seconds = (
            time.perf_counter()
            - start
        )

        if forward_seconds > FORWARD_TIMEOUT_SECONDS:
            raise TimeoutError(
                "Forward pass exceeded "
                f"{FORWARD_TIMEOUT_SECONDS} seconds."
            )

        print(
            f"  Forward: "
            f"{forward_seconds:.2f} s"
        )

        print(
            f"  Output shape: "
            f"{tuple(output.shape)}"
        )

        print(
            f"  Loss: "
            f"{float(loss.detach().cpu()):.6f}"
        )

        # ---------------------------------------------------------------
        # BACKWARD
        # ---------------------------------------------------------------

        start = time.perf_counter()

        loss.backward()

        backward_seconds = (
            time.perf_counter()
            - start
        )

        print(
            f"  Backward: "
            f"{backward_seconds:.2f} s"
        )

        # ---------------------------------------------------------------
        # OPTIMIZER STEP
        # ---------------------------------------------------------------

        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_seconds = (
            forward_seconds
            + backward_seconds
        )

        memory = gpu_memory_stats()

        print(
            f"  Peak allocated: "
            f"{memory['peak_allocated_gb']:.3f} GB"
        )

        print(
            f"  Peak reserved:  "
            f"{memory['peak_reserved_gb']:.3f} GB"
        )

        # Conservative pass criterion:
        # keep at least 10% of 4 GB nominal capacity free.
        memory_safe = (
            not torch.cuda.is_available()
            or memory["peak_reserved_gb"]
            <= GPU_MEMORY_GB_TARGET * 0.90
        )

        result.update(
            {
                "status": (
                    "PASS"
                    if memory_safe
                    else "CAUTION"
                ),
                "forward_seconds": round(
                    forward_seconds,
                    3,
                ),
                "backward_seconds": round(
                    backward_seconds,
                    3,
                ),
                "total_seconds": round(
                    total_seconds,
                    3,
                ),
                "loss": round(
                    float(
                        loss.detach().cpu()
                    ),
                    6,
                ),
                "peak_allocated_gb": round(
                    memory[
                        "peak_allocated_gb"
                    ],
                    4,
                ),
                "peak_reserved_gb": round(
                    memory[
                        "peak_reserved_gb"
                    ],
                    4,
                ),
                "memory_safe": memory_safe,
                "error": "",
            }
        )

        print(
            f"  Result: "
            f"{result['status']}"
        )

    except torch.cuda.OutOfMemoryError as exc:

        reset_cuda_memory()

        result.update(
            {
                "status": "OOM",
                "error": (
                    "CUDA out of memory: "
                    f"{exc}"
                ),
                "memory_safe": False,
            }
        )

        print(
            "  Result: OOM"
        )

    except Exception as exc:

        result.update(
            {
                "status": "FAIL",
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                "memory_safe": False,
            }
        )

        print(
            f"  Result: FAIL - "
            f"{result['error']}"
        )

    finally:

        del scaler

        if optimizer is not None:
            del optimizer

        if model is not None:
            del model

        reset_cuda_memory()

    return result


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    header(
        "PHASE 4 - PART 10\n"
        "RSNA SWIN-UNETR TRAINING PREFLIGHT / GPU FEASIBILITY"
    )

    print()
    print("PROJECT ROOT")
    print(PROJECT_ROOT)

    print()
    print("RSNA DATASET")
    print(RSNA_ROOT)

    print()
    print("PART 8 MANIFESTS")
    print(MANIFEST_DIR)

    print()
    print("PART 9 LOADER")
    print(
        SRC_DIR
        / "segmentation_rsna_part9_3d_dataset_loader.py"
    )

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------------
    # PATH VALIDATION
    # ---------------------------------------------------------------------

    header(
        "PATH VALIDATION"
    )

    required_paths = {
        "RSNA root": RSNA_ROOT,
        "train_images": TRAIN_IMAGES_DIR,
        "Part 8 manifest directory": MANIFEST_DIR,
        "train manifest": TRAIN_MANIFEST,
        "validation manifest": VAL_MANIFEST,
        "test manifest": TEST_MANIFEST,
        "Part 9 loader": (
            SRC_DIR
            / "segmentation_rsna_part9_3d_dataset_loader.py"
        ),
    }

    for name, path in required_paths.items():

        print(
            f"{name:<30}: "
            f"{'FOUND' if path.exists() else 'MISSING'}"
        )

    missing = [
        name
        for name, path in required_paths.items()
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required paths: "
            + ", ".join(missing)
        )

    # ---------------------------------------------------------------------
    # ENVIRONMENT
    # ---------------------------------------------------------------------

    header(
        "PYTORCH / GPU ENVIRONMENT"
    )

    torch.set_num_threads(
        CPU_THREADS
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"PyTorch version : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available  : "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"Device          : "
        f"{device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

        props = torch.cuda.get_device_properties(
            0
        )

        print(
            f"GPU memory      : "
            f"{props.total_memory / (1024 ** 3):.2f} GB"
        )

        print(
            f"CUDA runtime    : "
            f"{torch.version.cuda}"
        )

    print(
        f"CPU threads     : "
        f"{CPU_THREADS}"
    )

    print(
        f"AMP requested   : "
        f"{USE_AMP}"
    )

    print(
        f"Feature size    : "
        f"{FEATURE_SIZE}"
    )

    print(
        f"Classes         : "
        f"{NUM_CLASSES}"
    )

    print(
        f"Batch size      : "
        f"{BATCH_SIZE}"
    )

    # ---------------------------------------------------------------------
    # EXPECTED LABELS
    # ---------------------------------------------------------------------

    header(
        "SEGMENTATION TARGET"
    )

    labels = {
        0: "Background",
        1: "Spinal Canal Stenosis",
        2: "Left Neural Foraminal Narrowing",
        3: "Right Neural Foraminal Narrowing",
        4: "Left Subarticular Stenosis",
        5: "Right Subarticular Stenosis",
    }

    for index, name in labels.items():
        print(
            f"{index}: {name}"
        )

    # ---------------------------------------------------------------------
    # LOAD PART 9
    # ---------------------------------------------------------------------

    header(
        "LOADING VALIDATED PART 9 DATA LOADER"
    )

    part9 = load_part9_module()

    print(
        "✓ Part 9 loader imported."
    )

    cases = load_representative_cases(
        part9
    )

    for split, split_cases in cases.items():
        print(
            f"{split:<12}: "
            f"{len(split_cases)} representative cases"
        )

    # ---------------------------------------------------------------------
    # PREPROCESSING SANITY
    # ---------------------------------------------------------------------

    header(
        "PREPROCESSING CANDIDATE SANITY CHECK"
    )

    preprocessing_rows = preprocessing_sanity(
        cases
    )

    preprocessing_df = pd.DataFrame(
        preprocessing_rows
    )

    preprocessing_path = (
        OUTPUT_DIR
        / "rsna_part10_preprocessing_sanity.csv"
    )

    preprocessing_df.to_csv(
        preprocessing_path,
        index=False,
    )

    print(
        f"Saved: {preprocessing_path}"
    )

    print()

    for target in CANDIDATE_PATCHES:

        foregrounds = []

        for row in preprocessing_rows:

            key = (
                f"{target}_foreground"
            )

            if key in row:
                foregrounds.append(
                    row[key]
                )

        if foregrounds:

            print(
                f"{str(target):<18} "
                f"mean foreground: "
                f"{np.mean(foregrounds):.1f}"
            )

    # ---------------------------------------------------------------------
    # MODEL CREATION TEST
    # ---------------------------------------------------------------------

    header(
        "MODEL CREATION TEST"
    )

    reset_cuda_memory()

    model = create_model(
        device
    )

    params = parameter_summary(
        model
    )

    print(
        "✓ Swin-UNETR created."
    )

    print(
        f"Total parameters     : "
        f"{params['total']:,}"
    )

    print(
        f"Trainable parameters : "
        f"{params['trainable']:,}"
    )

    print(
        f"Output classes       : "
        f"{NUM_CLASSES}"
    )

    del model

    reset_cuda_memory()

    # ---------------------------------------------------------------------
    # SELECT ONE REPRESENTATIVE TRAIN CASE
    # ---------------------------------------------------------------------

    train_cases = cases.get(
        "train",
        []
    )

    if not train_cases:
        raise RuntimeError(
            "No representative train case available."
        )

    primary_case = train_cases[0]

    print()
    print(
        "Primary preflight case:"
    )

    print(
        f"Study  : "
        f"{primary_case['sample']['study_id']}"
    )

    print(
        f"Series : "
        f"{primary_case['sample']['series_id']}"
    )

    print(
        f"Type   : "
        f"{primary_case['sample']['series_description']}"
    )

    # ---------------------------------------------------------------------
    # RUN CANDIDATES
    # ---------------------------------------------------------------------

    header(
        "RUNNING FORWARD + BACKWARD GPU PREFLIGHT"
    )

    print(
        "Each candidate performs ONE real forward pass, "
        "DiceCE loss calculation, backward pass and optimizer step."
    )

    candidate_results = []

    for target_shape in CANDIDATE_PATCHES:

        result = run_candidate(
            target_shape,
            primary_case,
            device,
        )

        candidate_results.append(
            result
        )

        # Once OOM happens at a larger configuration, continue only
        # with the remaining smaller/equal candidates already defined.
        # The candidates are ordered from smaller to larger.
        if result["status"] == "OOM":
            print(
                "  OOM detected; continuing to next "
                "candidate is intentionally skipped."
            )
            break

    candidate_df = pd.DataFrame(
        candidate_results
    )

    candidate_path = (
        OUTPUT_DIR
        / "rsna_part10_gpu_preflight_results.csv"
    )

    candidate_df.to_csv(
        candidate_path,
        index=False,
    )

    print(
        f"Saved: {candidate_path}"
    )

    # ---------------------------------------------------------------------
    # CHOOSE CONFIGURATION
    # ---------------------------------------------------------------------

    header(
        "CONFIGURATION DECISION"
    )

    safe_results = [
        r
        for r in candidate_results
        if r.get("status") == "PASS"
    ]

    caution_results = [
        r
        for r in candidate_results
        if r.get("status") == "CAUTION"
    ]

    if safe_results:

        # Prefer the largest safe candidate.
        selected = safe_results[-1]

        decision = (
            "PASS - selected configuration completed "
            "forward and backward successfully within "
            "the conservative memory target."
        )

    elif caution_results:

        selected = caution_results[0]

        decision = (
            "CAUTION - a configuration completed "
            "forward/backward but exceeded the conservative "
            "memory target. Use the smallest successful "
            "configuration for full training."
        )

    else:

        selected = None

        decision = (
            "FAIL - no candidate completed the "
            "forward/backward preflight."
        )

    if selected is not None:

        selected_shape = selected[
            "target_shape"
        ]

        print(
            f"Selected patch size : "
            f"{selected_shape}"
        )

        print(
            f"Peak allocated      : "
            f"{selected['peak_allocated_gb']:.3f} GB"
        )

        print(
            f"Peak reserved       : "
            f"{selected['peak_reserved_gb']:.3f} GB"
        )

        print(
            f"Forward time        : "
            f"{selected['forward_seconds']:.3f} s"
        )

        print(
            f"Backward time       : "
            f"{selected['backward_seconds']:.3f} s"
        )

    else:

        selected_shape = None

        print(
            "No training configuration selected."
        )

    print()
    print(
        decision
    )

    # ---------------------------------------------------------------------
    # SAVE SUMMARY
    # ---------------------------------------------------------------------

    summary = {
        "phase": "Phase 4 - Part 10",
        "title": (
            "RSNA Swin-UNETR training preflight "
            "and GPU feasibility"
        ),
        "rsna_only": True,
        "spider_used": False,
        "training_performed": False,
        "model_weights_permanently_modified": False,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "feature_size": FEATURE_SIZE,
        "amp_requested": USE_AMP,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "gpu_memory_target_gb": GPU_MEMORY_GB_TARGET,
        "candidate_patch_sizes": [
            list(x)
            for x in CANDIDATE_PATCHES
        ],
        "candidate_results": candidate_results,
        "selected_patch_size": selected_shape,
        "decision": decision,
    }

    summary_path = (
        OUTPUT_DIR
        / "phase4_part10_training_preflight_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"Saved: {summary_path}"
    )

    report_path = (
        REPORT_DIR
        / "phase4_part10_training_preflight_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 4 - PART 10\n"
            "RSNA SWIN-UNETR TRAINING PREFLIGHT / "
            "GPU FEASIBILITY\n\n"
        )

        f.write(
            f"Device: {device}\n"
        )

        if torch.cuda.is_available():
            f.write(
                "GPU: "
                + torch.cuda.get_device_name(0)
                + "\n"
            )

        f.write(
            f"Classes: {NUM_CLASSES}\n"
        )

        f.write(
            f"Feature size: {FEATURE_SIZE}\n"
        )

        f.write(
            f"AMP requested: {USE_AMP}\n"
        )

        f.write(
            f"Batch size: {BATCH_SIZE}\n\n"
        )

        f.write(
            "Candidate results\n"
        )
        f.write(
            "-" * 60
            + "\n"
        )

        for result in candidate_results:

            f.write(
                f"Patch: "
                f"{result['target_shape']}\n"
            )

            f.write(
                f"Status: "
                f"{result['status']}\n"
            )

            f.write(
                f"Peak allocated GB: "
                f"{result['peak_allocated_gb']}\n"
            )

            f.write(
                f"Peak reserved GB: "
                f"{result['peak_reserved_gb']}\n"
            )

            f.write(
                f"Forward seconds: "
                f"{result['forward_seconds']}\n"
            )

            f.write(
                f"Backward seconds: "
                f"{result['backward_seconds']}\n"
            )

            if result["error"]:
                f.write(
                    f"Error: "
                    f"{result['error']}\n"
                )

            f.write("\n")

        f.write(
            "FINAL DECISION\n"
        )

        f.write(
            decision
            + "\n\n"
        )

        f.write(
            "SPIDER used: NO\n"
            "Training performed: NO\n"
            "Model weights permanently modified: NO\n"
        )

    print(
        f"Saved: {report_path}"
    )

    # ---------------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------------

    header(
        "PART 10 FINAL SUMMARY"
    )

    print(
        f"Candidate configurations tested : "
        f"{len(candidate_results)}"
    )

    print(
        f"Successful configurations       : "
        f"{len(safe_results)}"
    )

    print(
        f"Caution configurations          : "
        f"{len(caution_results)}"
    )

    print()

    for result in candidate_results:

        print(
            f"{result['target_shape']:<18} "
            f"{result['status']:<8} "
            f"peak reserved="
            f"{result['peak_reserved_gb']}"
        )

    print()
    print(
        "SPIDER used            : NO"
    )

    print(
        "Full training performed: NO"
    )

    print(
        "Permanent weights saved: NO"
    )

    print()
    print(
        "FINAL DECISION"
    )

    print(
        decision
    )

    print()
    print(
        "OUTPUT DIRECTORY"
    )

    print(
        OUTPUT_DIR
    )

    # Part 10 should fail explicitly if no configuration works.
    if selected is None:
        raise RuntimeError(
            "Part 10 preflight failed. "
            "No candidate patch completed the real "
            "Swin-UNETR forward/backward test."
        )

    header(
        "PHASE 4 - PART 10 COMPLETE"
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print()
        print("=" * 78)
        print("PART 10 ERROR")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        sys.exit(1)
