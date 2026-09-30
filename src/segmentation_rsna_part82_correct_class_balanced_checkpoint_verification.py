"""
PART 82 — CORRECT CLASS-BALANCED PART 71 CHECKPOINT VERIFICATION
================================================================

Purpose
-------
Part 81 accidentally selected the baseline epoch-3 checkpoint while
investigating the historical Part 71 class-balanced result.

The historical Part 71 report contains an E3 foreground Dice of about
0.044026 for the CLASS-BALANCED run.

Part 82 performs the final missing verification:

    part71_r2_full_class_balanced_epoch3.pth

It performs EVALUATION ONLY.

NO:
- training
- backward()
- optimizer.step()
- gradient computation
- checkpoint modification
- new checkpoint creation
- ablation experiment

It evaluates the exact saved class-balanced epoch-3 checkpoint on the
same first 50 validation cases used by the established Part 71 setup.

It reports:
- global foreground Dice
- mean case foreground Dice
- total target foreground
- total predicted foreground
- classwise Dice
- comparison against historical 0.044026

Run from the project root:

python ".\\src\\segmentation_rsna_part82_correct_class_balanced_checkpoint_verification.py"

IMPORTANT
---------
This script deliberately evaluates ONLY:

    part71_r2_full_class_balanced_epoch3.pth

It does not automatically choose a baseline checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(
    r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"
)

SRC_DIR = PROJECT_ROOT / "src"

PART9_PATH = (
    SRC_DIR
    / "segmentation_rsna_part9_3d_dataset_loader.py"
)

PART11_PATH = (
    SRC_DIR
    / "segmentation_rsna_part11_controlled_pilot_training.py"
)

PART71_PATH = (
    SRC_DIR
    / "segmentation_rsna_part71_class_balanced_dicece_training.py"
)

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART71_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part71_class_balanced_dicece_training"
)

CHECKPOINT_DIR = (
    PART71_DIR
    / "checkpoints"
)

VAL_COHORT = (
    PART15_DIR
    / "part15_validation_cohort.csv"
)

TARGET_CHECKPOINT = (
    CHECKPOINT_DIR
    / "part71_r2_full_class_balanced_epoch3.pth"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part82_correct_class_balanced_checkpoint_verification"
)

REPORT_DIR = OUT_DIR / "reports"


# =====================================================================
# LOCKED VALUES
# =====================================================================

NUM_CLASSES = 6
VAL_N = 50

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)

HISTORICAL_REFERENCE = 0.04402626277260595

EXPECTED_PART15_SHA256 = (
    "0900c0e6490fdaddf455763b465d4a607acfb310349c9f9ec82a61117016aec6"
)

PART15_INIT = (
    PART15_DIR
    / "checkpoints"
    / "part15_initialization_from_part11.pth"
)


# =====================================================================
# UTILITIES
# =====================================================================

def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_module(
    name: str,
    path: Path,
):

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Unable to load module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    spec.loader.exec_module(
        module
    )

    return module


def as_dhw(
    tensor: torch.Tensor,
) -> torch.Tensor:

    x = tensor.detach().cpu()

    while x.ndim > 3:

        singleton_axes = [
            i
            for i, size in enumerate(
                x.shape
            )
            if size == 1
        ]

        if not singleton_axes:

            raise ValueError(
                f"Cannot reduce tensor "
                f"{tuple(x.shape)} to DHW"
            )

        x = x.squeeze(
            singleton_axes[0]
        )

    if x.ndim != 3:

        raise ValueError(
            f"Expected DHW tensor, "
            f"got {tuple(x.shape)}"
        )

    return x


def centered_crop(
    image: torch.Tensor,
    target: torch.Tensor,
):

    d, h, w = image.shape

    cd, ch, cw = CROP_SHAPE

    if (
        d < cd
        or h < ch
        or w < cw
    ):

        raise ValueError(
            f"Input shape {tuple(image.shape)} "
            f"is smaller than crop {CROP_SHAPE}"
        )

    z0 = (
        d - cd
    ) // 2

    y0 = (
        h - ch
    ) // 2

    x0 = (
        w - cw
    ) // 2

    image_crop = image[
        z0:z0 + cd,
        y0:y0 + ch,
        x0:x0 + cw,
    ]

    target_crop = target[
        z0:z0 + cd,
        y0:y0 + ch,
        x0:x0 + cw,
    ]

    return (
        image_crop,
        target_crop,
    )


def extract_state_dict(
    checkpoint: Any,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, Any],
]:

    metadata = {}

    if isinstance(
        checkpoint,
        dict,
    ):

        metadata = {
            "epoch": checkpoint.get(
                "epoch"
            ),
            "best_val_dice": checkpoint.get(
                "best_val_dice"
            ),
            "seed": checkpoint.get(
                "seed"
            ),
            "patch_size": checkpoint.get(
                "patch_size"
            ),
            "feature_size": checkpoint.get(
                "feature_size"
            ),
            "num_classes": checkpoint.get(
                "num_classes"
            ),
            "train_cases": checkpoint.get(
                "train_cases"
            ),
            "validation_cases": checkpoint.get(
                "validation_cases"
            ),
            "source_checkpoint": checkpoint.get(
                "source_checkpoint"
            ),
            "history": checkpoint.get(
                "history"
            ),
        }

        if (
            "model_state_dict"
            in checkpoint
        ):

            return (
                checkpoint[
                    "model_state_dict"
                ],
                metadata,
            )

        # Direct state dict fallback.
        tensor_values = [
            value
            for value in checkpoint.values()
            if torch.is_tensor(value)
        ]

        if tensor_values:

            return (
                checkpoint,
                metadata,
            )

    raise ValueError(
        "Unable to extract model_state_dict "
        "from class-balanced checkpoint."
    )


def foreground_dice_from_masks(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:

    pred_fg = (
        prediction > 0
    )

    target_fg = (
        target > 0
    )

    intersection = int(
        (pred_fg & target_fg)
        .sum()
        .item()
    )

    pred_count = int(
        pred_fg.sum().item()
    )

    target_count = int(
        target_fg.sum().item()
    )

    denominator = (
        pred_count
        + target_count
    )

    if denominator == 0:
        return 1.0

    return (
        2.0
        * intersection
        / denominator
    )


# =====================================================================
# MAIN VERIFICATION
# =====================================================================

def main():

    print(
        "=" * 78
    )

    print(
        "PART 82 — CORRECT CLASS-BALANCED PART 71 CHECKPOINT VERIFICATION"
    )

    print(
        "=" * 78
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # FILE CHECK
    # ---------------------------------------------------------------

    print(
        "\n[FILE CHECK]"
    )

    required_files = [
        PART9_PATH,
        PART11_PATH,
        PART71_PATH,
        VAL_COHORT,
        PART15_INIT,
        TARGET_CHECKPOINT,
    ]

    for path in required_files:

        print(
            f"  {'FOUND' if path.exists() else 'MISSING'} "
            f"{path}"
        )

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Required file(s) missing:\n"
            + "\n".join(
                str(x)
                for x in missing
            )
        )

    # ---------------------------------------------------------------
    # PART 15 IDENTITY
    # ---------------------------------------------------------------

    print(
        "\n[PART 15 INITIALIZATION IDENTITY]"
    )

    part15_hash = sha256_file(
        PART15_INIT
    )

    part15_hash_match = (
        part15_hash
        == EXPECTED_PART15_SHA256
    )

    print(
        f"  SHA256: {part15_hash}"
    )

    print(
        f"  Expected: "
        f"{EXPECTED_PART15_SHA256}"
    )

    print(
        f"  Match: "
        f"{part15_hash_match}"
    )

    # ---------------------------------------------------------------
    # TARGET CHECKPOINT IDENTITY
    # ---------------------------------------------------------------

    print(
        "\n[TARGET CHECKPOINT IDENTITY]"
    )

    checkpoint_hash = sha256_file(
        TARGET_CHECKPOINT
    )

    print(
        f"  Checkpoint:"
    )

    print(
        f"    {TARGET_CHECKPOINT}"
    )

    print(
        f"  SHA256:"
    )

    print(
        f"    {checkpoint_hash}"
    )

    checkpoint_size = (
        TARGET_CHECKPOINT.stat().st_size
    )

    print(
        f"  Size: "
        f"{checkpoint_size:,} bytes"
    )

    # ---------------------------------------------------------------
    # LOAD PROJECT MODULES
    # ---------------------------------------------------------------

    part9 = load_module(
        "part9_part82",
        PART9_PATH,
    )

    part11 = load_module(
        "part11_part82",
        PART11_PATH,
    )

    # ---------------------------------------------------------------
    # LOAD CHECKPOINT
    # ---------------------------------------------------------------

    print(
        "\n[CHECKPOINT METADATA]"
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"  Evaluation device: "
        f"{device}"
    )

    checkpoint = torch.load(
        TARGET_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    state_dict, metadata = (
        extract_state_dict(
            checkpoint
        )
    )

    print(
        f"  Epoch: "
        f"{metadata.get('epoch')}"
    )

    print(
        f"  Seed: "
        f"{metadata.get('seed')}"
    )

    print(
        f"  Feature size: "
        f"{metadata.get('feature_size')}"
    )

    print(
        f"  Number of classes: "
        f"{metadata.get('num_classes')}"
    )

    print(
        f"  Train cases recorded: "
        f"{metadata.get('train_cases')}"
    )

    print(
        f"  Validation cases recorded: "
        f"{metadata.get('validation_cases')}"
    )

    print(
        f"  Source checkpoint: "
        f"{metadata.get('source_checkpoint')}"
    )

    # ---------------------------------------------------------------
    # CREATE MODEL
    # ---------------------------------------------------------------

    print(
        "\n[MODEL STATE VERIFICATION]"
    )

    model = part11.create_model(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"  Model: "
        f"{type(model).__name__}"
    )

    print(
        f"  Parameters: "
        f"{parameter_count:,}"
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print(
        f"  Missing keys: "
        f"{len(missing_keys)}"
    )

    print(
        f"  Unexpected keys: "
        f"{len(unexpected_keys)}"
    )

    state_load_exact = (
        len(missing_keys) == 0
        and len(unexpected_keys) == 0
    )

    print(
        f"  Exact state load: "
        f"{state_load_exact}"
    )

    model.eval()

    # ---------------------------------------------------------------
    # VALIDATION COHORT
    # ---------------------------------------------------------------

    val_df = pd.read_csv(
        VAL_COHORT
    ).iloc[:VAL_N].reset_index(
        drop=True
    )

    print(
        "\n[VALIDATION COHORT]"
    )

    print(
        f"  Cases evaluated: "
        f"{len(val_df)}"
    )

    # ---------------------------------------------------------------
    # ACCUMULATORS
    # ---------------------------------------------------------------

    total_target_fg = 0
    total_pred_fg = 0
    total_fg_intersection = 0

    class_intersection = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    class_pred = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    class_target = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    case_fg_dice = []

    per_case = []

    # ---------------------------------------------------------------
    # EVALUATION-ONLY FORWARD PASSES
    # ---------------------------------------------------------------

    print(
        "\n[DIRECT EVALUATION]"
    )

    print(
        "  IMPORTANT: no gradients, "
        "no optimizer, no training."
    )

    for i, (_, row) in enumerate(
        val_df.iterrows()
    ):

        loaded = part11.load_tensor_case(
            row,
            part9,
        )

        image = as_dhw(
            loaded[0]
        ).float()

        target = as_dhw(
            loaded[1]
        ).long()

        image, target = centered_crop(
            image,
            target,
        )

        input_tensor = (
            image
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        with torch.inference_mode():

            logits = model(
                input_tensor
            )

            prediction = torch.argmax(
                logits,
                dim=1,
            ).squeeze(0).cpu()

        # Safety-only spatial alignment.
        # This should not normally be required.
        if prediction.shape != target.shape:

            prediction = F.interpolate(
                prediction
                .float()
                .unsqueeze(0)
                .unsqueeze(0),
                size=target.shape,
                mode="nearest",
            ).squeeze().long()

        target_fg = int(
            (target > 0)
            .sum()
            .item()
        )

        pred_fg = int(
            (prediction > 0)
            .sum()
            .item()
        )

        intersection_fg = int(
            (
                (prediction > 0)
                & (target > 0)
            )
            .sum()
            .item()
        )

        fg_dice = (
            foreground_dice_from_masks(
                prediction,
                target,
            )
        )

        total_target_fg += (
            target_fg
        )

        total_pred_fg += (
            pred_fg
        )

        total_fg_intersection += (
            intersection_fg
        )

        case_fg_dice.append(
            fg_dice
        )

        # Classwise accumulation.
        for c in range(
            NUM_CLASSES
        ):

            pred_c = (
                prediction == c
            )

            target_c = (
                target == c
            )

            class_intersection[c] += int(
                (
                    pred_c
                    & target_c
                )
                .sum()
                .item()
            )

            class_pred[c] += int(
                pred_c.sum().item()
            )

            class_target[c] += int(
                target_c.sum().item()
            )

        case_id = str(
            row.get(
                "study_id",
                row.get(
                    "series_id",
                    i,
                ),
            )
        )

        per_case.append(
            {
                "index": i,
                "case_id": case_id,
                "target_fg": target_fg,
                "pred_fg": pred_fg,
                "intersection_fg":
                    intersection_fg,
                "fg_dice": fg_dice,
            }
        )

        if (
            (i + 1) % 10 == 0
            or i == len(val_df) - 1
        ):

            print(
                f"  Evaluated "
                f"{i + 1}/{len(val_df)}"
            )

    # ---------------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------------

    global_fg_dice = (
        2.0
        * total_fg_intersection
        / (
            total_target_fg
            + total_pred_fg
        )
        if (
            total_target_fg
            + total_pred_fg
        )
        else 1.0
    )

    mean_case_fg_dice = float(
        np.mean(
            case_fg_dice
        )
    )

    median_case_fg_dice = float(
        np.median(
            case_fg_dice
        )
    )

    classwise = {}

    for c in range(
        1,
        NUM_CLASSES,
    ):

        denominator = (
            class_pred[c]
            + class_target[c]
        )

        dice = (
            2.0
            * class_intersection[c]
            / denominator
            if denominator
            else 1.0
        )

        precision = (
            class_intersection[c]
            / class_pred[c]
            if class_pred[c]
            else 0.0
        )

        recall = (
            class_intersection[c]
            / class_target[c]
            if class_target[c]
            else 0.0
        )

        classwise[str(c)] = {
            "dice": float(dice),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "intersection": float(
                class_intersection[c]
            ),
            "pred_voxels": float(
                class_pred[c]
            ),
            "target_voxels": float(
                class_target[c]
            ),
        }

    global_difference = abs(
        global_fg_dice
        - HISTORICAL_REFERENCE
    )

    mean_difference = abs(
        mean_case_fg_dice
        - HISTORICAL_REFERENCE
    )

    # ---------------------------------------------------------------
    # DECISION
    # ---------------------------------------------------------------

    # 0.005 is deliberately a practical forensic tolerance, not a
    # claim of exact equality.
    global_matches = (
        global_difference
        <= 0.005
    )

    mean_matches = (
        mean_difference
        <= 0.005
    )

    if global_matches:

        status = (
            "CLASS_BALANCED_CHECKPOINT_MATCHES_HISTORICAL_PART71_REFERENCE"
        )

        conclusion = (
            "The saved class-balanced Part 71 epoch-3 checkpoint "
            "produces a global foreground Dice consistent with the "
            "historical 0.044026 result. Part 79's compatibility "
            "implementation should not be used as a replacement for "
            "the original Part 71 pipeline."
        )

    elif mean_matches:

        status = (
            "CLASS_BALANCED_CHECKPOINT_MEAN_CASE_METRIC_MATCHES_HISTORICAL_REFERENCE"
        )

        conclusion = (
            "The saved class-balanced Part 71 epoch-3 checkpoint "
            "matches the historical reference under mean-case "
            "foreground Dice, indicating that the historical result "
            "was likely based on a case-aggregated metric rather than "
            "the global voxel-pooled Dice used here."
        )

    else:

        status = (
            "CLASS_BALANCED_CHECKPOINT_DOES_NOT_REPRODUCE_HISTORICAL_REFERENCE"
        )

        conclusion = (
            "The actual saved class-balanced Part 71 epoch-3 "
            "checkpoint does not reproduce the historical 0.044026 "
            "under either of the two direct foreground-Dice "
            "aggregations evaluated here. Do not launch another "
            "training ablation; inspect the original Part 71 "
            "metric/report-generation code next."
        )

    # ---------------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "PART 82 VERIFICATION RESULTS"
    )

    print(
        "=" * 78
    )

    print(
        f"Target checkpoint:"
    )

    print(
        f"  {TARGET_CHECKPOINT.name}"
    )

    print(
        f"\nHistorical Part 71 E3 reference:"
    )

    print(
        f"  {HISTORICAL_REFERENCE:.12f}"
    )

    print(
        f"\nDirect global foreground Dice:"
    )

    print(
        f"  {global_fg_dice:.12f}"
    )

    print(
        f"\nDirect mean-case foreground Dice:"
    )

    print(
        f"  {mean_case_fg_dice:.12f}"
    )

    print(
        f"\nDirect median-case foreground Dice:"
    )

    print(
        f"  {median_case_fg_dice:.12f}"
    )

    print(
        f"\nGlobal absolute difference:"
    )

    print(
        f"  {global_difference:.12f}"
    )

    print(
        f"\nMean-case absolute difference:"
    )

    print(
        f"  {mean_difference:.12f}"
    )

    print(
        f"\nTotal target foreground:"
    )

    print(
        f"  {total_target_fg}"
    )

    print(
        f"\nTotal predicted foreground:"
    )

    print(
        f"  {total_pred_fg}"
    )

    print(
        "\nClasswise foreground metrics:"
    )

    for c, values in classwise.items():

        print(
            f"  C{c}: "
            f"Dice={values['dice']:.9f}, "
            f"Precision={values['precision']:.9f}, "
            f"Recall={values['recall']:.9f}, "
            f"Pred={values['pred_voxels']:.0f}, "
            f"Target={values['target_voxels']:.0f}"
        )

    print(
        "\nSTATUS:"
    )

    print(
        f"  {status}"
    )

    print(
        "\nCONCLUSION:"
    )

    print(
        f"  {conclusion}"
    )

    # ---------------------------------------------------------------
    # SAVE REPORTS
    # ---------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "part": 82,
        "title":
            "Correct Class-Balanced Part 71 Checkpoint Verification",
        "evaluation_only": True,
        "optimizer_steps": 0,
        "backward_passes": 0,
        "checkpoint_modified": False,
        "project_root":
            str(PROJECT_ROOT),
        "target_checkpoint":
            str(TARGET_CHECKPOINT),
        "target_checkpoint_sha256":
            checkpoint_hash,
        "target_checkpoint_bytes":
            checkpoint_size,
        "part15_init_sha256":
            part15_hash,
        "part15_init_sha256_matches":
            part15_hash_match,
        "device":
            str(device),
        "validation_cases":
            len(val_df),
        "full_shape":
            FULL_SHAPE,
        "crop_shape":
            CROP_SHAPE,
        "historical_reference":
            HISTORICAL_REFERENCE,
        "checkpoint_metadata":
            metadata,
        "model":
            {
                "class":
                    type(model).__name__,
                "parameter_count":
                    parameter_count,
                "state_load_exact":
                    state_load_exact,
                "missing_keys":
                    list(
                        missing_keys
                    ),
                "unexpected_keys":
                    list(
                        unexpected_keys
                    ),
            },
        "metrics":
            {
                "global_foreground_dice":
                    float(
                        global_fg_dice
                    ),
                "mean_case_foreground_dice":
                    float(
                        mean_case_fg_dice
                    ),
                "median_case_foreground_dice":
                    float(
                        median_case_fg_dice
                    ),
                "global_absolute_difference":
                    float(
                        global_difference
                    ),
                "mean_case_absolute_difference":
                    float(
                        mean_difference
                    ),
                "total_target_foreground":
                    int(
                        total_target_fg
                    ),
                "total_predicted_foreground":
                    int(
                        total_pred_fg
                    ),
                "total_foreground_intersection":
                    int(
                        total_fg_intersection
                    ),
            },
        "classwise":
            classwise,
        "per_case":
            per_case,
        "decision":
            {
                "status":
                    status,
                "global_matches_reference":
                    global_matches,
                "mean_case_matches_reference":
                    mean_matches,
                "conclusion":
                    conclusion,
            },
    }

    json_path = (
        REPORT_DIR
        / "part82_class_balanced_checkpoint_verification.json"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    csv_path = (
        REPORT_DIR
        / "part82_class_balanced_checkpoint_verification.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "status",
                "historical_reference",
                "global_foreground_dice",
                "mean_case_foreground_dice",
                "median_case_foreground_dice",
                "global_absolute_difference",
                "mean_case_absolute_difference",
                "target_foreground",
                "predicted_foreground",
                "global_matches_reference",
                "mean_case_matches_reference",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "status":
                    status,
                "historical_reference":
                    HISTORICAL_REFERENCE,
                "global_foreground_dice":
                    global_fg_dice,
                "mean_case_foreground_dice":
                    mean_case_fg_dice,
                "median_case_foreground_dice":
                    median_case_fg_dice,
                "global_absolute_difference":
                    global_difference,
                "mean_case_absolute_difference":
                    mean_difference,
                "target_foreground":
                    total_target_fg,
                "predicted_foreground":
                    total_pred_fg,
                "global_matches_reference":
                    global_matches,
                "mean_case_matches_reference":
                    mean_matches,
            }
        )

    txt_path = (
        REPORT_DIR
        / "part82_report.txt"
    )

    text_lines = [
        "PART 82 — CORRECT CLASS-BALANCED PART 71 CHECKPOINT VERIFICATION",
        "=" * 72,
        "",
        "EVALUATION ONLY — NO TRAINING PERFORMED.",
        "",
        f"Target checkpoint: {TARGET_CHECKPOINT}",
        f"Checkpoint SHA256: {checkpoint_hash}",
        "",
        f"Historical E3 reference: {HISTORICAL_REFERENCE:.12f}",
        f"Global foreground Dice: {global_fg_dice:.12f}",
        f"Mean-case foreground Dice: {mean_case_fg_dice:.12f}",
        f"Median-case foreground Dice: {median_case_fg_dice:.12f}",
        f"Global absolute difference: {global_difference:.12f}",
        f"Mean-case absolute difference: {mean_difference:.12f}",
        "",
        f"Target foreground: {total_target_fg}",
        f"Predicted foreground: {total_pred_fg}",
        "",
        "CLASSWISE:",
    ]

    for c, values in classwise.items():

        text_lines.append(
            f"C{c}: "
            f"Dice={values['dice']:.9f}, "
            f"Precision={values['precision']:.9f}, "
            f"Recall={values['recall']:.9f}, "
            f"Pred={values['pred_voxels']:.0f}, "
            f"Target={values['target_voxels']:.0f}"
        )

    text_lines.extend(
        [
            "",
            f"STATUS: {status}",
            "",
            "CONCLUSION:",
            conclusion,
        ]
    )

    txt_path.write_text(
        "\n".join(
            text_lines
        ),
        encoding="utf-8",
    )

    print(
        "\nReports written:"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {txt_path}"
    )

    print(
        "\nNO TRAINING WAS PERFORMED."
    )


if __name__ == "__main__":
    main()
