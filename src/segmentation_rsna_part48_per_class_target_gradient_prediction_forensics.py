
"""
PART 48 — PER-CLASS TARGET / GRADIENT / PREDICTION FORENSICS

Purpose
-------
Determine whether the five foreground disease classes have materially
different target prevalence, prediction behavior, probability, or gradient
signals.

This is a READ-ONLY forensic diagnostic. No training is performed.

Classes:
    0 Background
    1 Spinal Canal Stenosis
    2 Left Neural Foraminal Narrowing
    3 Right Neural Foraminal Narrowing
    4 Left Subarticular Stenosis
    5 Right Subarticular Stenosis

The diagnostic analyzes existing checkpoints from Part 43 and Part 44.

For every checkpoint and every class it measures:
    - target voxel count
    - target percentage
    - predicted voxel count
    - predicted percentage
    - class probability
    - class probability on target-class voxels
    - one-vs-background Dice
    - signed output-logit gradient on target-class voxels
    - absolute output-logit gradient
    - class/background gradient magnitude ratio
    - class-vs-background logit gap on target-class voxels

Loss components:
    - Dice only
    - CE only
    - Combined Dice + CE

No optimizer steps.
No model modification.
No SPIDER.
No test set.
"""

from pathlib import Path
import sys
import csv
import json
import hashlib
import importlib.util

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"
)

SRC = ROOT / "src"

PART11_PATH = (
    SRC / "segmentation_rsna_part11_controlled_pilot_training.py"
)

PART9_PATH = (
    SRC / "segmentation_rsna_part9_3d_dataset_loader.py"
)

P15 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

INIT = (
    P15
    / "checkpoints"
    / "part15_initialization_from_part11.pth"
)

VACSV = P15 / "part15_validation_cohort.csv"

P43 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part43_improved_spatial_sampling_training"
)

P43_RANDOM = P43 / "random_spatial_crop"
P43_CENTERED = P43 / "foreground_centered_spatial_crop"

P44 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part44_loss_component_class_imbalance_diagnostic"
)

P44_ORIGINAL = P44 / "original_dicece"

OUT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part48_per_class_target_gradient_prediction_forensics"
)

REPORT = OUT / "reports"
REPORT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

VAL_N = 20

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)
SEED_OFFSET = 5000

NUM_CLASSES = 6

CLASS_NAMES = {
    0: "Background",
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}

FOREGROUND_CLASSES = [1, 2, 3, 4, 5]

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HELPERS
# ============================================================

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def extract_state(obj):
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            return obj["model_state_dict"]

        if "state_dict" in obj:
            return obj["state_dict"]

    return obj


# ============================================================
# DATA
# ============================================================

def centered_crop(image, mask, shape, seed):
    D, H, W = image.shape
    cd, ch, cw = shape

    rng = np.random.default_rng(seed)

    foreground = torch.nonzero(
        mask > 0,
        as_tuple=False,
    )

    if len(foreground) > 0:
        point = foreground[
            int(rng.integers(0, len(foreground)))
        ]

        center = [
            int(point[0])
            + int(rng.integers(-cd // 8, cd // 8 + 1)),
            int(point[1])
            + int(rng.integers(-ch // 8, ch // 8 + 1)),
            int(point[2])
            + int(rng.integers(-cw // 8, cw // 8 + 1)),
        ]
    else:
        center = [
            D // 2,
            H // 2,
            W // 2,
        ]

    starts = [
        max(
            0,
            min(
                D - cd,
                center[0] - cd // 2,
            ),
        ),
        max(
            0,
            min(
                H - ch,
                center[1] - ch // 2,
            ),
        ),
        max(
            0,
            min(
                W - cw,
                center[2] - cw // 2,
            ),
        ),
    ]

    d, h, w = starts

    return (
        image[d:d + cd, h:h + ch, w:w + cw],
        mask[d:d + cd, h:h + ch, w:w + cw],
    )


def load_cases(part11, part9):
    df = pd.read_csv(VACSV)

    cases = []

    n = min(
        VAL_N,
        len(df),
    )

    for i in range(n):
        row = df.iloc[i]

        loaded = part11.load_tensor_case(
            row,
            part9,
        )

        image = torch.as_tensor(loaded[0])
        mask = torch.as_tensor(loaded[1])

        if image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)

        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        image = image.float()
        mask = mask.long()

        if tuple(image.shape) != FULL_SHAPE:
            raise RuntimeError(
                f"Unexpected image shape {tuple(image.shape)}"
            )

        if tuple(mask.shape) != FULL_SHAPE:
            raise RuntimeError(
                f"Unexpected mask shape {tuple(mask.shape)}"
            )

        image, mask = centered_crop(
            image,
            mask,
            CROP_SHAPE,
            SEED_OFFSET + i,
        )

        cases.append(
            (image, mask)
        )

        if (
            i == 0
            or (i + 1) % 10 == 0
            or i + 1 == n
        ):
            print(
                f"Validation case {i + 1:03d}/{n} "
                f"FG={(mask > 0).sum().item()}"
            )

    return cases


# ============================================================
# MODEL
# ============================================================

def make_model(part11):
    return part11.create_model(DEVICE)


def load_checkpoint(model, path):
    state = torch.load(
        path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        extract_state(state),
        strict=True,
    )


# ============================================================
# LOSSES
# ============================================================

dice_fn = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    lambda_dice=1.0,
    lambda_ce=0.0,
)

ce_fn = torch.nn.CrossEntropyLoss()


# ============================================================
# PER-CLASS PROBE
# ============================================================

def probe_case(
    model,
    image,
    target,
    checkpoint_name,
):
    x = image[None, None].to(DEVICE)
    y = target[None].to(DEVICE)

    # One prediction forward.
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        with torch.amp.autocast(
            "cuda",
            enabled=DEVICE.type == "cuda",
        ):
            prediction_logits = model(x)

    probability = torch.softmax(
        prediction_logits,
        dim=1,
    )

    prediction = torch.argmax(
        probability,
        dim=1,
    )

    case_rows = []

    # --------------------------------------------------------
    # Three independent loss-gradient probes.
    # --------------------------------------------------------

    gradient_results = {}

    for loss_type in (
        "dice",
        "ce",
        "combined",
    ):
        model.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda",
            enabled=DEVICE.type == "cuda",
        ):
            logits = model(x)

            dice_loss = dice_fn(
                logits,
                y.unsqueeze(1),
            )

            ce_loss = ce_fn(
                logits,
                y,
            )

            if loss_type == "dice":
                loss = dice_loss

            elif loss_type == "ce":
                loss = ce_loss

            else:
                loss = dice_loss + ce_loss

        logits.retain_grad()
        loss.backward()

        gradient_results[loss_type] = (
            logits.detach(),
            logits.grad.detach(),
            float(loss.item()),
            float(dice_loss.item()),
            float(ce_loss.item()),
        )

        model.zero_grad(set_to_none=True)

    # --------------------------------------------------------
    # Per class.
    # --------------------------------------------------------

    for class_id in range(NUM_CLASSES):
        class_name = CLASS_NAMES[class_id]

        target_class = y == class_id
        pred_class = prediction == class_id

        target_count = int(
            target_class.sum().item()
        )

        pred_count = int(
            pred_class.sum().item()
        )

        total_voxels = target_class.numel()

        target_pct = (
            100.0 * target_count / total_voxels
        )

        pred_pct = (
            100.0 * pred_count / total_voxels
        )

        class_probability = (
            probability[:, class_id]
            .mean()
            .item()
        )

        if target_count > 0:
            probability_on_target = (
                probability[:, class_id][
                    target_class
                ]
                .mean()
                .item()
            )
        else:
            probability_on_target = float("nan")

        tp = (
            target_class & pred_class
        ).sum().item()

        dice = (
            2.0 * tp
            / (target_count + pred_count)
            if target_count + pred_count > 0
            else 1.0
        )

        # Background logit for comparison.
        bg_logit = prediction_logits[:, 0]

        class_logit = prediction_logits[
            :, class_id
        ]

        if target_count > 0:
            class_bg_gap = (
                (
                    class_logit[target_class]
                    - bg_logit[target_class]
                )
                .mean()
                .item()
            )
        else:
            class_bg_gap = float("nan")

        for loss_type in (
            "dice",
            "ce",
            "combined",
        ):
            logits_used, grad, loss_value, dice_value, ce_value = (
                gradient_results[loss_type]
            )

            class_grad = grad[:, class_id]

            abs_class_grad = (
                class_grad.abs()
            )

            abs_bg_grad = (
                grad[:, 0].abs()
            )

            class_abs_mean = (
                abs_class_grad.mean().item()
            )

            bg_abs_mean = (
                abs_bg_grad.mean().item()
            )

            ratio = (
                class_abs_mean
                / (bg_abs_mean + 1e-12)
            )

            if target_count > 0:
                signed_class_on_target = (
                    class_grad[target_class]
                    .mean()
                    .item()
                )

                signed_bg_on_target = (
                    grad[:, 0][target_class]
                    .mean()
                    .item()
                )

                abs_class_on_target = (
                    abs_class_grad[target_class]
                    .mean()
                    .item()
                )

                abs_bg_on_target = (
                    abs_bg_grad[target_class]
                    .mean()
                    .item()
                )

            else:
                signed_class_on_target = float("nan")
                signed_bg_on_target = float("nan")
                abs_class_on_target = float("nan")
                abs_bg_on_target = float("nan")

            class_direction = (
                "UP"
                if signed_class_on_target < 0
                else "DOWN"
                if signed_class_on_target > 0
                else "NEUTRAL"
            )

            row = {
                "checkpoint": checkpoint_name,
                "class_id": class_id,
                "class_name": class_name,
                "loss_type": loss_type,
                "target_voxels": target_count,
                "target_percent": target_pct,
                "predicted_voxels": pred_count,
                "predicted_percent": pred_pct,
                "class_probability": float(
                    class_probability
                ),
                "class_probability_on_target": float(
                    probability_on_target
                ),
                "class_dice": float(dice),
                "class_bg_logit_gap_on_target": float(
                    class_bg_gap
                ),
                "loss_value": loss_value,
                "dice_loss_value": dice_value,
                "ce_loss_value": ce_value,
                "class_abs_grad_mean": float(
                    class_abs_mean
                ),
                "background_abs_grad_mean": float(
                    bg_abs_mean
                ),
                "class_background_grad_ratio": float(
                    ratio
                ),
                "signed_class_grad_on_target": float(
                    signed_class_on_target
                ),
                "signed_background_grad_on_target": float(
                    signed_bg_on_target
                ),
                "abs_class_grad_on_target": float(
                    abs_class_on_target
                ),
                "abs_background_grad_on_target": float(
                    abs_bg_on_target
                ),
                "class_gradient_direction_on_target": (
                    class_direction
                ),
            }

            case_rows.append(row)

    return case_rows


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

def checkpoint_candidates(directory):
    found = {}

    if not directory.exists():
        return found

    for path in sorted(
        directory.glob("*.pth")
    ):
        name = path.name.lower()

        if (
            "epoch_01" in name
            or "epoch1" in name
        ):
            found["epoch_01"] = path

        elif (
            "epoch_02" in name
            or "epoch2" in name
        ):
            found["epoch_02"] = path

        elif (
            "epoch_03" in name
            or "epoch3" in name
        ):
            found["epoch_03"] = path

        elif (
            "epoch_04" in name
            or "epoch4" in name
        ):
            found["epoch_04"] = path

        elif (
            "epoch_05" in name
            or "epoch5" in name
        ):
            found["epoch_05"] = path

        elif "best_model" in name:
            found["best_model"] = path

    return found


# ============================================================
# CONDITION ANALYSIS
# ============================================================

def analyze_condition(
    label,
    directory,
    part11,
    cases,
):
    print("\n" + "=" * 82)
    print(label)
    print("=" * 82)

    candidates = checkpoint_candidates(
        directory
    )

    if not candidates:
        print(
            f"No checkpoints found in {directory}"
        )
        return []

    order = [
        "epoch_01",
        "epoch_02",
        "epoch_03",
        "epoch_04",
        "epoch_05",
        "best_model",
    ]

    model = make_model(part11)

    rows = []

    for checkpoint_name in order:
        path = candidates.get(
            checkpoint_name
        )

        if path is None:
            continue

        load_checkpoint(
            model,
            path,
        )

        print(
            f"\n{checkpoint_name}"
        )

        case_records = []

        for image, target in cases:
            case_records.extend(
                probe_case(
                    model,
                    image,
                    target,
                    checkpoint_name,
                )
            )

        # Add condition/checkpoint metadata.
        for row in case_records:
            row["condition"] = label
            row["checkpoint_path"] = str(path)
            row["checkpoint_sha256"] = (
                sha256_file(path)
            )

        rows.extend(case_records)

        # Compact per-class combined summary.
        combined = [
            r
            for r in case_records
            if r["loss_type"] == "combined"
        ]

        print(
            "  class | target% | pred vox | "
            "prob | Dice | grad ratio | signed grad"
        )

        for class_id in range(
            NUM_CLASSES
        ):
            subset = [
                r
                for r in combined
                if r["class_id"] == class_id
            ]

            if not subset:
                continue

            def avg(key):
                return float(
                    np.nanmean(
                        [
                            r[key]
                            for r in subset
                        ]
                    )
                )

            print(
                f"  {class_id} "
                f"{avg('target_percent'):8.4f} "
                f"{avg('predicted_voxels'):9.1f} "
                f"{avg('class_probability'):7.4f} "
                f"{avg('class_dice'):7.5f} "
                f"{avg('class_background_grad_ratio'):9.6f} "
                f"{avg('signed_class_grad_on_target'):+.3e}"
            )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 82)
    print("PART 48 PATH VALIDATION")
    print("=" * 82)

    paths = [
        ("Project root", ROOT),
        ("Part 11", PART11_PATH),
        ("Part 9", PART9_PATH),
        ("Part 15 initialization", INIT),
        ("Part 15 validation cohort", VACSV),
        ("Part 43 random directory", P43_RANDOM),
        ("Part 43 centered directory", P43_CENTERED),
        ("Part 44 original directory", P44_ORIGINAL),
    ]

    for label, path in paths:
        print(
            f"{label:<42}: "
            f"{'FOUND' if path.exists() else 'MISSING'}"
        )

    required = [
        ROOT,
        PART11_PATH,
        PART9_PATH,
        INIT,
        VACSV,
    ]

    if not all(
        p.exists()
        for p in required
    ):
        raise FileNotFoundError(
            "Required Part 48 input is missing."
        )

    print("\n" + "=" * 82)
    print(
        "PART 48 — PER-CLASS TARGET / GRADIENT / "
        "PREDICTION FORENSICS"
    )
    print("=" * 82)

    print(f"PyTorch : {torch.__version__}")
    print(f"Device : {DEVICE}")
    print(f"Validation subset : {VAL_N}")
    print(f"Full volume : {FULL_SHAPE}")
    print(f"Evaluation crop : {CROP_SHAPE}")
    print("Training performed : NO")
    print("Optimizer steps : NO")
    print("Part 15 overwritten : NO")
    print("Part 43/44 modified : NO")
    print("SPIDER used : NO")
    print("Test set used : NO")
    print(
        f"Initialization SHA256 : "
        f"{sha256_file(INIT)}"
    )

    part11 = load_module(
        PART11_PATH,
        "part11_part48",
    )

    part9 = load_module(
        PART9_PATH,
        "part9_part48",
    )

    print("\nPART 48 VALIDATION DATA PRELOAD")
    print("-" * 82)

    cases = load_cases(
        part11,
        part9,
    )

    print("\nPART 48 SHAPE SMOKE TEST")

    for i in range(
        min(3, len(cases))
    ):
        image, mask = cases[i]

        print(
            f"Case {i + 1}: "
            f"image={tuple(image.shape)} "
            f"mask={tuple(mask.shape)} "
            f"FG={(mask > 0).sum().item()}"
        )

    print(
        "✓ Shape smoke test PASSED."
    )

    all_rows = []

    all_rows += analyze_condition(
        "part43_random",
        P43_RANDOM,
        part11,
        cases,
    )

    all_rows += analyze_condition(
        "part43_centered",
        P43_CENTERED,
        part11,
        cases,
    )

    all_rows += analyze_condition(
        "part44_original",
        P44_ORIGINAL,
        part11,
        cases,
    )

    if not all_rows:
        raise RuntimeError(
            "No checkpoints were found."
        )

    csv_path = (
        REPORT
        / "part48_per_class_forensics.csv"
    )

    fields = sorted(
        {
            key
            for row in all_rows
            for key in row.keys()
        }
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # --------------------------------------------------------
    # Aggregate final report.
    # --------------------------------------------------------

    combined_rows = [
        r
        for r in all_rows
        if r["loss_type"] == "combined"
    ]

    class_summary = []

    for condition in sorted(
        set(
            r["condition"]
            for r in combined_rows
        )
    ):
        for checkpoint in sorted(
            set(
                r["checkpoint"]
                for r in combined_rows
                if r["condition"] == condition
            )
        ):
            for class_id in range(
                NUM_CLASSES
            ):
                subset = [
                    r
                    for r in combined_rows
                    if (
                        r["condition"] == condition
                        and r["checkpoint"] == checkpoint
                        and r["class_id"] == class_id
                    )
                ]

                if not subset:
                    continue

                def mean(key):
                    return float(
                        np.nanmean(
                            [
                                r[key]
                                for r in subset
                            ]
                        )
                    )

                class_summary.append({
                    "condition": condition,
                    "checkpoint": checkpoint,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "target_voxels": mean(
                        "target_voxels"
                    ),
                    "target_percent": mean(
                        "target_percent"
                    ),
                    "predicted_voxels": mean(
                        "predicted_voxels"
                    ),
                    "predicted_percent": mean(
                        "predicted_percent"
                    ),
                    "class_probability": mean(
                        "class_probability"
                    ),
                    "class_probability_on_target": mean(
                        "class_probability_on_target"
                    ),
                    "class_dice": mean(
                        "class_dice"
                    ),
                    "class_bg_logit_gap_on_target": mean(
                        "class_bg_logit_gap_on_target"
                    ),
                    "class_background_grad_ratio": mean(
                        "class_background_grad_ratio"
                    ),
                    "signed_class_grad_on_target": mean(
                        "signed_class_grad_on_target"
                    ),
                    "signed_background_grad_on_target": mean(
                        "signed_background_grad_on_target"
                    ),
                })

    summary_csv = (
        REPORT
        / "part48_class_summary_combined.csv"
    )

    summary_fields = sorted(
        {
            key
            for row in class_summary
            for key in row.keys()
        }
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields,
        )
        writer.writeheader()
        writer.writerows(class_summary)

    # --------------------------------------------------------
    # Determine sparse classes from validation target.
    # --------------------------------------------------------

    target_by_class = {}

    for class_id in range(
        NUM_CLASSES
    ):
        values = [
            r["target_voxels"]
            for r in class_summary
            if (
                r["condition"] == "part43_random"
                and r["checkpoint"] == "epoch_01"
                and r["class_id"] == class_id
            )
        ]

        target_by_class[class_id] = (
            float(np.nanmean(values))
            if values
            else float("nan")
        )

    fg_targets = [
        target_by_class[c]
        for c in FOREGROUND_CLASSES
        if np.isfinite(
            target_by_class[c]
        )
    ]

    rarest_class = None

    if fg_targets:
        rarest_class = min(
            FOREGROUND_CLASSES,
            key=lambda c: target_by_class[c]
        )

    summary = {
        "part": 48,
        "purpose": (
            "per-class target, prediction and gradient forensics"
        ),
        "device": str(DEVICE),
        "validation_subset": VAL_N,
        "crop_shape": CROP_SHAPE,
        "training_performed": False,
        "optimizer_steps": False,
        "part15_overwritten": False,
        "part43_44_modified": False,
        "initialization_sha256": sha256_file(INIT),
        "classes": CLASS_NAMES,
        "mean_target_voxels_by_class_epoch1_part43_random": {
            str(k): v
            for k, v in target_by_class.items()
        },
        "rarest_foreground_class": (
            {
                "class_id": rarest_class,
                "class_name": (
                    CLASS_NAMES[rarest_class]
                    if rarest_class is not None
                    else None
                ),
            }
            if rarest_class is not None
            else None
        ),
        "interpretation": (
            "PER_CLASS_FORENSICS_COMPLETE_REVIEW_CLASS_SPECIFIC_IMBALANCE"
        ),
        "raw_csv": str(csv_path),
        "combined_summary_csv": str(summary_csv),
    }

    summary_path = (
        REPORT
        / "part48_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 82)
    print("PART 48 COMPLETE")
    print("=" * 82)
    print(
        "No training or optimizer updates were performed."
    )

    if rarest_class is not None:
        print(
            f"Rarest foreground class by target voxels "
            f"(epoch_01 reference) : "
            f"{rarest_class} — "
            f"{CLASS_NAMES[rarest_class]}"
        )

    print(
        f"Raw per-class CSV : {csv_path}"
    )

    print(
        f"Combined class summary : {summary_csv}"
    )

    print(
        f"Summary JSON : {summary_path}"
    )


if __name__ == "__main__":
    main()
