
"""
PART 47 — LOSS-GRADIENT DIRECTION DECOMPOSITION

Purpose
-------
Determine which component of the original DiceCE objective drives the
progressive foreground/background logit separation seen in Part 46.

This is a TRAINING-DYNAMICS diagnostic, not a production training run.

For selected checkpoints, it measures the gradient of:
    1. Dice loss only
    2. Cross-entropy loss only
    3. Dice + CE combined

For each loss component it reports:
    - gradient magnitude on foreground output channels
    - gradient magnitude on background output channel
    - foreground/background gradient ratio
    - gradient on foreground channels at target-foreground voxels
    - background-channel gradient at target-foreground voxels
    - signed mean gradient on foreground channels at target-FG voxels
    - signed mean gradient on background channel at target-FG voxels
    - signed mean gradient on foreground channels at target-BG voxels
    - signed mean gradient on background channel at target-BG voxels

Interpretation is based on gradient SIGN:
    Gradient descent changes parameter/logit approximately as:
        logit_update ∝ -gradient

Therefore:
    positive foreground-channel gradient at target-FG
        -> pushes foreground logits DOWN
    negative foreground-channel gradient at target-FG
        -> pushes foreground logits UP

The same logic is applied to the background channel.

Design
------
- Existing Part 44 original DiceCE checkpoints.
- Existing Part 43 random/centered checkpoints for comparison.
- First 20 Part 15 validation cases.
- Same foreground-centered evaluation crop (32,64,64).
- No training.
- No optimizer steps.
- No checkpoint modification.
- Part 15/43/44 untouched.
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

PART11_PATH = SRC / "segmentation_rsna_part11_controlled_pilot_training.py"
PART9_PATH = SRC / "segmentation_rsna_part9_3d_dataset_loader.py"

P15 = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

INIT = P15 / "checkpoints" / "part15_initialization_from_part11.pth"
VACSV = P15 / "part15_validation_cohort.csv"

P43 = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part43_improved_spatial_sampling_training"
)

P43_RANDOM = P43 / "random_spatial_crop"
P43_CENTERED = P43 / "foreground_centered_spatial_crop"

P44 = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part44_loss_component_class_imbalance_diagnostic"
)

P44_ORIGINAL = P44 / "original_dicece"

OUT = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part47_loss_gradient_direction_decomposition"
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

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODULE / FILE HELPERS
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


def load_validation_cases(part11, part9):
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

        image = torch.as_tensor(
            loaded[0]
        )

        mask = torch.as_tensor(
            loaded[1]
        )

        if image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)

        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        image = image.float()
        mask = mask.long()

        if tuple(image.shape) != FULL_SHAPE:
            raise RuntimeError(
                f"Unexpected image shape: {tuple(image.shape)}"
            )

        if tuple(mask.shape) != FULL_SHAPE:
            raise RuntimeError(
                f"Unexpected mask shape: {tuple(mask.shape)}"
            )

        image, mask = centered_crop(
            image,
            mask,
            CROP_SHAPE,
            SEED_OFFSET + i,
        )

        cases.append(
            (
                image,
                mask,
            )
        )

        if (
            i == 0
            or (i + 1) % 10 == 0
            or i + 1 == n
        ):
            print(
                f"Validation case {i + 1:03d}/{n} "
                f"cropFG={(mask > 0).sum().item()}"
            )

    mean_fg = np.mean(
        [
            (mask > 0).sum().item()
            for _, mask in cases
        ]
    )

    print(
        f"Mean validation crop FG : {mean_fg:.2f}"
    )

    return cases


# ============================================================
# MODEL
# ============================================================

def make_model(part11):
    return part11.create_model(DEVICE)


def extract_state(obj):
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            return obj["model_state_dict"]

        if "state_dict" in obj:
            return obj["state_dict"]

    return obj


def load_checkpoint(model, path):
    state = torch.load(
        path,
        map_location=DEVICE,
    )

    state = extract_state(state)

    model.load_state_dict(
        state,
        strict=True,
    )


# ============================================================
# LOSS COMPONENTS
# ============================================================

dice_loss_fn = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    lambda_dice=1.0,
    lambda_ce=0.0,
)

ce_loss_fn = torch.nn.CrossEntropyLoss()


# ============================================================
# GRADIENT ANALYSIS
# ============================================================

def analyze_loss_gradient(
    logits,
    target,
    loss_type,
):
    """
    Analyze output-logit gradient direction.

    loss_type:
        "dice"
        "ce"
        "combined"
    """

    target_5d = target.unsqueeze(1)

    dice_loss = dice_loss_fn(
        logits,
        target_5d,
    )

    ce_loss = ce_loss_fn(
        logits,
        target,
    )

    if loss_type == "dice":
        loss = dice_loss

    elif loss_type == "ce":
        loss = ce_loss

    elif loss_type == "combined":
        loss = dice_loss + ce_loss

    else:
        raise ValueError(loss_type)

    logits.retain_grad()

    loss.backward()

    grad = logits.grad.detach()

    target_fg = target > 0
    target_bg = ~target_fg

    # Signed gradients.
    signed_fg_channels = grad[:, 1:].mean(dim=1)
    signed_bg_channel = grad[:, 0]

    # Absolute gradients.
    abs_grad = grad.abs()

    fg_abs = abs_grad[:, 1:].mean().item()
    bg_abs = abs_grad[:, 0].mean().item()

    fg_abs_ratio = (
        fg_abs / (bg_abs + 1e-12)
    )

    if target_fg.any():
        signed_fg_on_fg = (
            signed_fg_channels[target_fg]
            .mean()
            .item()
        )

        signed_bg_on_fg = (
            signed_bg_channel[target_fg]
            .mean()
            .item()
        )

        abs_fg_on_fg = (
            abs_grad[:, 1:]
            .mean(dim=1)[target_fg]
            .mean()
            .item()
        )

        abs_bg_on_fg = (
            abs_grad[:, 0][target_fg]
            .mean()
            .item()
        )

    else:
        signed_fg_on_fg = float("nan")
        signed_bg_on_fg = float("nan")
        abs_fg_on_fg = float("nan")
        abs_bg_on_fg = float("nan")

    if target_bg.any():
        signed_fg_on_bg = (
            signed_fg_channels[target_bg]
            .mean()
            .item()
        )

        signed_bg_on_bg = (
            signed_bg_channel[target_bg]
            .mean()
            .item()
        )

        abs_fg_on_bg = (
            abs_grad[:, 1:]
            .mean(dim=1)[target_bg]
            .mean()
            .item()
        )

        abs_bg_on_bg = (
            abs_grad[:, 0][target_bg]
            .mean()
            .item()
        )

    else:
        signed_fg_on_bg = float("nan")
        signed_bg_on_bg = float("nan")
        abs_fg_on_bg = float("nan")
        abs_bg_on_bg = float("nan")

    # A positive foreground gradient means gradient descent pushes
    # foreground logits DOWN. A negative gradient pushes them UP.
    foreground_direction_on_fg = (
        "DOWN"
        if signed_fg_on_fg > 0
        else "UP"
        if signed_fg_on_fg < 0
        else "NEUTRAL"
    )

    background_direction_on_fg = (
        "DOWN"
        if signed_bg_on_fg > 0
        else "UP"
        if signed_bg_on_fg < 0
        else "NEUTRAL"
    )

    return {
        "loss_type": loss_type,
        "loss_value": float(loss.item()),
        "dice_loss_value": float(
            dice_loss.item()
        ),
        "ce_loss_value": float(
            ce_loss.item()
        ),
        "fg_channel_abs_grad": float(
            fg_abs
        ),
        "bg_channel_abs_grad": float(
            bg_abs
        ),
        "fg_bg_abs_grad_ratio": float(
            fg_abs_ratio
        ),
        "signed_fg_grad_on_target_fg": float(
            signed_fg_on_fg
        ),
        "signed_bg_grad_on_target_fg": float(
            signed_bg_on_fg
        ),
        "signed_fg_grad_on_target_bg": float(
            signed_fg_on_bg
        ),
        "signed_bg_grad_on_target_bg": float(
            signed_bg_on_bg
        ),
        "abs_fg_grad_on_target_fg": float(
            abs_fg_on_fg
        ),
        "abs_bg_grad_on_target_fg": float(
            abs_bg_on_fg
        ),
        "abs_fg_grad_on_target_bg": float(
            abs_fg_on_bg
        ),
        "abs_bg_grad_on_target_bg": float(
            abs_bg_on_bg
        ),
        "foreground_direction_on_target_fg": (
            foreground_direction_on_fg
        ),
        "background_direction_on_target_fg": (
            background_direction_on_fg
        ),
    }


def prediction_metrics(logits, target):
    with torch.no_grad():
        probability = torch.softmax(
            logits,
            dim=1,
        )

        prediction = torch.argmax(
            probability,
            dim=1,
        )

        pred_fg = prediction > 0
        target_fg = target > 0

        tp = (
            pred_fg & target_fg
        ).sum().item()

        pred_count = pred_fg.sum().item()
        target_count = target_fg.sum().item()

        fg_dice = (
            2.0 * tp
            / (pred_count + target_count)
            if pred_count + target_count > 0
            else 1.0
        )

        fg_probability = (
            probability[:, 1:]
            .sum(dim=1)
            .mean()
            .item()
        )

        bg_probability = (
            probability[:, 0]
            .mean()
            .item()
        )

        return {
            "fg_dice": float(fg_dice),
            "pred_fg": float(pred_count),
            "target_fg": float(target_count),
            "fg_probability": float(
                fg_probability
            ),
            "background_probability": float(
                bg_probability
            ),
            "empty": int(
                pred_count == 0
            ),
        }


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
# CHECKPOINT PROBE
# ============================================================

def probe_checkpoint(
    model,
    cases,
):
    model.eval()

    component_records = []
    prediction_records = []

    for image, target in cases:
        x = image[
            None,
            None,
        ].to(DEVICE)

        y = target[
            None
        ].to(DEVICE)

        # One forward pass for predictions.
        with torch.no_grad():
            with torch.amp.autocast(
                "cuda",
                enabled=DEVICE.type == "cuda",
            ):
                prediction_logits = model(x)

        prediction_records.append(
            prediction_metrics(
                prediction_logits,
                y,
            )
        )

        # Three independent gradient probes.
        for loss_type in (
            "dice",
            "ce",
            "combined",
        ):
            model.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=DEVICE.type == "cuda",
            ):
                logits = model(x)

            result = analyze_loss_gradient(
                logits,
                y,
                loss_type,
            )

            component_records.append(
                result
            )

            model.zero_grad(
                set_to_none=True
            )

    # Average each loss component over cases.
    component_summary = {}

    for loss_type in (
        "dice",
        "ce",
        "combined",
    ):
        subset = [
            r
            for r in component_records
            if r["loss_type"] == loss_type
        ]

        keys = [
            k
            for k in subset[0].keys()
            if k not in (
                "loss_type",
                "foreground_direction_on_target_fg",
                "background_direction_on_target_fg",
            )
        ]

        summary = {
            k: float(
                np.nanmean(
                    [
                        r[k]
                        for r in subset
                    ]
                )
            )
            for k in keys
        }

        # Direction is reported from the averaged signed gradients.
        fg_signed = summary[
            "signed_fg_grad_on_target_fg"
        ]

        bg_signed = summary[
            "signed_bg_grad_on_target_fg"
        ]

        summary[
            "foreground_direction_on_target_fg"
        ] = (
            "DOWN"
            if fg_signed > 0
            else "UP"
            if fg_signed < 0
            else "NEUTRAL"
        )

        summary[
            "background_direction_on_target_fg"
        ] = (
            "DOWN"
            if bg_signed > 0
            else "UP"
            if bg_signed < 0
            else "NEUTRAL"
        )

        component_summary[
            loss_type
        ] = summary

    prediction_summary = {
        k: float(
            np.nanmean(
                [
                    r[k]
                    for r in prediction_records
                ]
            )
        )
        for k in prediction_records[0]
        if k != "empty"
    }

    prediction_summary["empty"] = int(
        sum(
            r["empty"]
            for r in prediction_records
        )
    )

    return (
        component_summary,
        prediction_summary,
    )


# ============================================================
# MAIN ANALYSIS
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
            f"No checkpoints found: {directory}"
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

        components, predictions = (
            probe_checkpoint(
                model,
                cases,
            )
        )

        for loss_type in (
            "dice",
            "ce",
            "combined",
        ):
            row = {
                "condition": label,
                "checkpoint": checkpoint_name,
                "checkpoint_path": str(path),
                "checkpoint_sha256": sha256_file(path),
                "loss_type": loss_type,
                **predictions,
                **components[loss_type],
            }

            rows.append(row)

        c = components["combined"]

        print(
            f"{checkpoint_name:<14} "
            f"PredFG={predictions['pred_fg']:.1f} "
            f"FGProb={predictions['fg_probability']:.6f} "
            f"FGDice={predictions['fg_dice']:.6f}"
        )

        for loss_type in (
            "dice",
            "ce",
            "combined",
        ):
            c = components[loss_type]

            print(
                f"  {loss_type:<9} "
                f"loss={c['loss_value']:.6f} "
                f"FGgrad={c['fg_channel_abs_grad']:.3e} "
                f"BGgrad={c['bg_channel_abs_grad']:.3e} "
                f"ratio={c['fg_bg_abs_grad_ratio']:.6f} "
                f"sFG@FG={c['signed_fg_grad_on_target_fg']:+.3e} "
                f"sBG@FG={c['signed_bg_grad_on_target_fg']:+.3e} "
                f"dirFG={c['foreground_direction_on_target_fg']}"
            )

    return rows


def main():
    print("=" * 82)
    print("PART 47 PATH VALIDATION")
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
            "Required Part 47 input missing."
        )

    print("\n" + "=" * 82)
    print(
        "PART 47 — LOSS-GRADIENT DIRECTION DECOMPOSITION"
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
        f"Part 15 initialization SHA256 : "
        f"{sha256_file(INIT)}"
    )

    part11 = load_module(
        PART11_PATH,
        "part11_part47",
    )

    part9 = load_module(
        PART9_PATH,
        "part9_part47",
    )

    print("\nPART 47 VALIDATION DATA PRELOAD")
    print("-" * 82)

    cases = load_validation_cases(
        part11,
        part9,
    )

    print("\nPART 47 SHAPE SMOKE TEST")

    for i in range(
        min(3, len(cases))
    ):
        image, mask = cases[i]

        print(
            f"Case {i + 1}: "
            f"image={tuple(image.shape)} "
            f"mask={tuple(mask.shape)} "
            f"cropFG={(mask > 0).sum().item()}"
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
        / "part47_gradient_decomposition.csv"
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
    # High-level automatic interpretation.
    # --------------------------------------------------------

    combined = [
        r
        for r in all_rows
        if r["loss_type"] == "combined"
    ]

    dice_rows = [
        r
        for r in all_rows
        if r["loss_type"] == "dice"
    ]

    ce_rows = [
        r
        for r in all_rows
        if r["loss_type"] == "ce"
    ]

    positive_ce_fg = sum(
        1
        for r in ce_rows
        if r["signed_fg_grad_on_target_fg"] > 0
    )

    positive_dice_fg = sum(
        1
        for r in dice_rows
        if r["signed_fg_grad_on_target_fg"] > 0
    )

    positive_combined_fg = sum(
        1
        for r in combined
        if r["signed_fg_grad_on_target_fg"] > 0
    )

    interpretation = (
        "REVIEW_SIGNED_GRADIENT_COMPONENTS"
    )

    if ce_rows:
        if positive_ce_fg == len(
            ce_rows
        ):
            interpretation = (
                "CE_PUSHES_FOREGROUND_LOGITS_DOWN_ON_TARGET_FOREGROUND"
            )

        elif positive_ce_fg == 0:
            interpretation = (
                "CE_DOES_NOT_PUSH_FOREGROUND_LOGITS_DOWN_CONSISTENTLY"
            )

        else:
            interpretation = (
                "CE_FOREGROUND_GRADIENT_DIRECTION_CHANGES_BY_CHECKPOINT"
            )

    summary = {
        "part": 47,
        "purpose": (
            "loss-gradient direction decomposition"
        ),
        "device": str(DEVICE),
        "validation_subset": VAL_N,
        "crop_shape": CROP_SHAPE,
        "training_performed": False,
        "optimizer_steps": False,
        "part15_overwritten": False,
        "initialization_sha256": sha256_file(INIT),
        "checkpoint_records": len(all_rows),
        "dice_positive_fg_gradient_records": (
            positive_dice_fg
        ),
        "ce_positive_fg_gradient_records": (
            positive_ce_fg
        ),
        "combined_positive_fg_gradient_records": (
            positive_combined_fg
        ),
        "interpretation": interpretation,
        "csv": str(csv_path),
    }

    summary_path = (
        REPORT
        / "part47_summary.json"
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
    print("PART 47 COMPLETE")
    print("=" * 82)
    print(
        "No training or optimizer updates were performed."
    )
    print(
        f"Gradient records : {len(all_rows)}"
    )
    print(
        f"CE positive FG-gradient records : "
        f"{positive_ce_fg}"
    )
    print(
        f"Dice positive FG-gradient records : "
        f"{positive_dice_fg}"
    )
    print(
        f"Combined positive FG-gradient records : "
        f"{positive_combined_fg}"
    )
    print(
        f"Interpretation : {interpretation}"
    )
    print(
        f"CSV : {csv_path}"
    )
    print(
        f"Summary JSON : {summary_path}"
    )


if __name__ == "__main__":
    main()
