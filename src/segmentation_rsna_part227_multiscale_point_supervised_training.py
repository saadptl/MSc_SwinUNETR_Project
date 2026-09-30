"""
PART 2.27 — MULTI-SCALE POINT-SUPERVISED SWIN-UNETR TRAINING

Purpose
-------
Train a controlled weak/point-supervised Swin-UNETR refinement model using
the RSNA point annotations at multiple prediction scales.

This experiment:
- initializes from protected Part 2.20B Epoch-5 checkpoint
- uses the verified Part 2.13 point manifest
- uses the exact Part 2.20B physical-space/canonical preprocessing
- uses the same study-disjoint validation design
- does NOT create voxel-wise ground-truth masks
- does NOT overwrite Part 2.20B
- does NOT modify the dashboard

Supervision
-----------
1. Exact annotation-point CE.
2. Radius-2 probability aggregation loss.
3. Radius-4 probability aggregation loss.
4. Sampled background CE.

The radius losses aggregate model probabilities around the annotated point.
They do NOT label every voxel in those neighborhoods as diseased.

Important limitation
--------------------
RSNA coordinates are point/localization annotations, not manual voxel-wise
segmentation masks. This experiment evaluates point-level disease recognition
and contextual supervision; it does not establish voxel-wise segmentation
accuracy.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import segmentation_rsna_part220b_geometry_corrected_training as part220b


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42

EPOCHS = 5
TRAIN_SLOTS_PER_EPOCH = 100
VALIDATION_CASES = 25

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4

LR = 1.5e-5
WEIGHT_DECAY = 1e-5

# Multi-scale point-supervision weights.
POINT_LOSS_WEIGHT = 1.00
RADIUS2_LOSS_WEIGHT = 0.50
RADIUS4_LOSS_WEIGHT = 0.25

# Background supervision is deliberately weak because the RSNA labels
# do not provide voxel-wise background masks.
BACKGROUND_SAMPLES = 128
BACKGROUND_LOSS_WEIGHT = 0.10

# Disease balancing.
# RFNN and RSS are explicitly oversampled because Part 2.25 showed weak
# point-level performance for these classes.
RFNN_EXTRA_WEIGHT = 2.0
RSS_EXTRA_WEIGHT = 1.5

RADIUS2 = 2
RADIUS4 = 4

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part213_point_supervision_manifest"
    / "part213_point_manifest.csv"
)

INIT_CHECKPOINT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part220b_geometry_corrected_training"
    / "checkpoints"
    / "part220b_epoch_05.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = OUTPUT_DIR / "reports"

for directory in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    METRICS_DIR,
    PREDICTION_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Validation cohort
# ---------------------------------------------------------------------

def build_validation_cases(manifest):
    """
    Exact Part 2.20B validation study selection, followed by direct
    series grouping.

    This avoids relying on Part 2.20B's internal case-index return type.
    """
    validation_series = part220b.select_validation_series(manifest).copy()

    validation_series["study_id"] = (
        validation_series["study_id"].astype(str)
    )
    validation_series["series_id"] = (
        validation_series["series_id"].astype(str)
    )

    valid_keys = set(
        zip(
            validation_series["study_id"],
            validation_series["series_id"],
        )
    )

    work = manifest.copy()
    work["study_id"] = work["study_id"].astype(str)
    work["series_id"] = work["series_id"].astype(str)

    validation_manifest = work[
        work.apply(
            lambda r: (
                r["study_id"],
                r["series_id"],
            ) in valid_keys,
            axis=1,
        )
    ].copy()

    bad = validation_manifest[
        ~validation_manifest["study_id"].str.fullmatch(r"\d+")
        | ~validation_manifest["series_id"].str.fullmatch(r"\d+")
    ]

    if not bad.empty:
        raise RuntimeError(
            "Validation manifest contains invalid study/series IDs."
        )

    case_groups = list(
        validation_manifest.groupby(
            ["study_id", "series_id"],
            sort=True,
        )
    )

    if len(case_groups) != VALIDATION_CASES:
        raise RuntimeError(
            f"Expected {VALIDATION_CASES} validation cases, "
            f"got {len(case_groups)}."
        )

    return validation_series, validation_manifest, case_groups


# ---------------------------------------------------------------------
# Training case index
# ---------------------------------------------------------------------

def build_training_index(manifest, validation_manifest):
    """
    Build study-disjoint training series.

    Any study present in the validation cohort is excluded from training.
    """
    validation_studies = set(
        validation_manifest["study_id"].astype(str).unique()
    )

    work = manifest.copy()
    work["study_id"] = work["study_id"].astype(str)
    work["series_id"] = work["series_id"].astype(str)

    training = work[
        ~work["study_id"].isin(validation_studies)
    ].copy()

    groups = []

    for (study_id, series_id), point_df in training.groupby(
        ["study_id", "series_id"],
        sort=True,
    ):
        groups.append(
            (
                str(study_id),
                str(series_id),
                point_df.copy(),
            )
        )

    if not groups:
        raise RuntimeError("No training series remain after study split.")

    return groups


# ---------------------------------------------------------------------
# Disease-aware sampler
# ---------------------------------------------------------------------

def build_sampling_weights(training_cases):
    """
    Assign case weights based on disease content.

    RFNN receives additional sampling weight because Part 2.25 reported
    0% point accuracy for RFNN.

    RSS also receives additional sampling weight because it was weak.
    """
    weights = []

    for study_id, series_id, point_df in training_cases:
        classes = set(
            point_df["class_id"].astype(int).tolist()
        )

        weight = 1.0

        if 3 in classes:
            weight *= RFNN_EXTRA_WEIGHT

        if 5 in classes:
            weight *= RSS_EXTRA_WEIGHT

        weights.append(weight)

    weights = np.asarray(weights, dtype=np.float64)

    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(
            len(training_cases),
            dtype=np.float64,
        )

    weights /= weights.sum()

    return weights


def choose_training_case(
    training_cases,
    sampling_weights,
):
    index = np.random.choice(
        len(training_cases),
        p=sampling_weights,
    )

    return training_cases[int(index)]


# ---------------------------------------------------------------------
# Point extraction
# ---------------------------------------------------------------------

def normalize_loaded_points(points):
    """
    Part 2.20B load_case() returns point coordinates as z/y/x.
    """
    if isinstance(points, pd.DataFrame):
        df = points.copy()
    else:
        df = pd.DataFrame(points)

    required = [
        "class_id",
        "z",
        "y",
        "x",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Loaded point data missing {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


# ---------------------------------------------------------------------
# Multi-scale point loss
# ---------------------------------------------------------------------

def point_cross_entropy(
    probabilities,
    points,
):
    """
    Exact point-level CE.

    probabilities:
        [C,D,H,W]

    points:
        DataFrame containing class_id and z/y/x.
    """
    losses = []

    c, d, h, w = probabilities.shape

    for _, p in points.iterrows():
        cls = int(p["class_id"])

        if cls < 1 or cls >= c:
            continue

        z = int(
            np.clip(
                round(float(p["z"])),
                0,
                d - 1,
            )
        )
        y = int(
            np.clip(
                round(float(p["y"])),
                0,
                h - 1,
            )
        )
        x = int(
            np.clip(
                round(float(p["x"])),
                0,
                w - 1,
            )
        )

        # -log probability of annotated class.
        losses.append(
            -torch.log(
                probabilities[cls, z, y, x]
                .clamp_min(1e-6)
            )
        )

    if not losses:
        return probabilities.sum() * 0.0

    return torch.stack(losses).mean()


def multi_scale_probability_loss(
    probabilities,
    points,
    radius,
):
    """
    Aggregate probabilities in a local 3D neighborhood and encourage
    the annotated disease class to have the largest aggregate probability.

    This is NOT a voxel-wise target mask.

    The target applies to the aggregated neighborhood probability only.
    """
    losses = []

    c, d, h, w = probabilities.shape

    for _, p in points.iterrows():
        cls = int(p["class_id"])

        if cls < 1 or cls >= c:
            continue

        z = int(
            np.clip(
                round(float(p["z"])),
                0,
                d - 1,
            )
        )
        y = int(
            np.clip(
                round(float(p["y"])),
                0,
                h - 1,
            )
        )
        x = int(
            np.clip(
                round(float(p["x"])),
                0,
                w - 1,
            )
        )

        z0 = max(0, z - radius)
        z1 = min(d, z + radius + 1)

        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)

        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)

        neighborhood = probabilities[
            :,
            z0:z1,
            y0:y1,
            x0:x1,
        ]

        # Mean class probability in the neighborhood.
        pooled = neighborhood.mean(
            dim=(1, 2, 3)
        )

        # Cross-entropy on the pooled class distribution.
        pooled = pooled.clamp_min(1e-6)
        pooled = pooled / pooled.sum().clamp_min(1e-6)

        losses.append(
            -torch.log(pooled[cls])
        )

    if not losses:
        return probabilities.sum() * 0.0

    return torch.stack(losses).mean()


def background_sample_loss(
    probabilities,
    points,
    samples=BACKGROUND_SAMPLES,
):
    """
    Weak background regularization.

    Random voxels are sampled only outside small neighborhoods around
    annotated points. They are not treated as manual segmentation
    background ground truth; the loss simply discourages uncontrolled
    foreground activation away from annotations.

    This term is deliberately weak.
    """
    c, d, h, w = probabilities.shape

    blocked = torch.zeros(
        (d, h, w),
        dtype=torch.bool,
        device=probabilities.device,
    )

    for _, p in points.iterrows():
        z = int(
            np.clip(round(float(p["z"])), 0, d - 1)
        )
        y = int(
            np.clip(round(float(p["y"])), 0, h - 1)
        )
        x = int(
            np.clip(round(float(p["x"])), 0, w - 1)
        )

        r = RADIUS4

        z0 = max(0, z-r)
        z1 = min(d, z+r+1)
        y0 = max(0, y-r)
        y1 = min(h, y+r+1)
        x0 = max(0, x-r)
        x1 = min(w, x+r+1)

        blocked[
            z0:z1,
            y0:y1,
            x0:x1,
        ] = True

    available = torch.nonzero(
        ~blocked,
        as_tuple=False,
    )

    if available.numel() == 0:
        return probabilities.sum() * 0.0

    n = min(
        samples,
        available.shape[0],
    )

    selected = available[
        torch.randperm(
            available.shape[0],
            device=available.device,
        )[:n]
    ]

    z = selected[:, 0]
    y = selected[:, 1]
    x = selected[:, 2]

    bg_prob = probabilities[
        0,
        z,
        y,
        x,
    ].clamp_min(1e-6)

    return -torch.log(bg_prob).mean()


# ---------------------------------------------------------------------
# One training case
# ---------------------------------------------------------------------

def train_case(
    model,
    optimizer,
    scaler,
    image,
    points,
):
    image_tensor = (
        torch.from_numpy(
            np.asarray(
                image,
                dtype=np.float32,
            )
        )
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda()
    )

    with autocast(
        "cuda",
        enabled=torch.cuda.is_available(),
    ):
        logits = model(image_tensor)
        probabilities = F.softmax(
            logits,
            dim=1,
        )[0]

        point_loss = point_cross_entropy(
            probabilities,
            points,
        )

        r2_loss = multi_scale_probability_loss(
            probabilities,
            points,
            RADIUS2,
        )

        r4_loss = multi_scale_probability_loss(
            probabilities,
            points,
            RADIUS4,
        )

        bg_loss = background_sample_loss(
            probabilities,
            points,
        )

        total_loss = (
            POINT_LOSS_WEIGHT * point_loss
            + RADIUS2_LOSS_WEIGHT * r2_loss
            + RADIUS4_LOSS_WEIGHT * r4_loss
            + BACKGROUND_LOSS_WEIGHT * bg_loss
        )

    return (
        total_loss,
        point_loss.detach(),
        r2_loss.detach(),
        r4_loss.detach(),
        bg_loss.detach(),
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_case(
    model,
    image,
    points,
):
    model.eval()

    tensor = (
        torch.from_numpy(
            np.asarray(
                image,
                dtype=np.float32,
            )
        )
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(next(model.parameters()).device)
    )

    with autocast(
        "cuda",
        enabled=torch.cuda.is_available(),
    ):
        logits = model(tensor)
        probabilities = F.softmax(
            logits,
            dim=1,
        )[0]

    probs = probabilities.detach().cpu()
    labels = probs.argmax(dim=0)

    rows = []

    c, d, h, w = probs.shape

    for _, p in points.iterrows():
        true_class = int(p["class_id"])

        z = int(
            np.clip(round(float(p["z"])), 0, d - 1)
        )
        y = int(
            np.clip(round(float(p["y"])), 0, h - 1)
        )
        x = int(
            np.clip(round(float(p["x"])), 0, w - 1)
        )

        pred_class = int(labels[z, y, x])
        true_probability = float(
            probs[true_class, z, y, x]
        )

        foreground_ratio = float(
            (labels != 0).float().mean()
        )

        rows.append(
            {
                "true_class": true_class,
                "true_name": CLASS_NAMES[true_class],
                "predicted_class": pred_class,
                "predicted_name": CLASS_NAMES.get(
                    pred_class,
                    str(pred_class),
                ),
                "correct": int(
                    pred_class == true_class
                ),
                "true_probability": true_probability,
                "hit_at_0_50": int(
                    true_probability >= 0.50
                ),
                "z": z,
                "y": y,
                "x": x,
                "foreground_ratio": foreground_ratio,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    seed_everything()

    print("=" * 72)
    print("PART 2.27 — MULTI-SCALE POINT-SUPERVISED SWIN-UNETR TRAINING")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this controlled RTX 2050 experiment."
        )

    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initialization checkpoint: {INIT_CHECKPOINT}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    if not INIT_CHECKPOINT.exists():
        raise FileNotFoundError(INIT_CHECKPOINT)

    manifest = part220b.load_manifest()

    print(f"Manifest rows: {len(manifest)}")

    validation_series, validation_manifest, validation_cases = (
        build_validation_cases(manifest)
    )

    training_cases = build_training_index(
        manifest,
        validation_manifest,
    )

    print(
        f"Training series: {len(training_cases)}"
    )
    print(
        f"Validation series: {len(validation_cases)}"
    )
    print(
        f"Validation studies: "
        f"{validation_manifest['study_id'].nunique()}"
    )

    sampling_weights = build_sampling_weights(
        training_cases
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = part220b.build_model()

    checkpoint = torch.load(
        INIT_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"Initialization missing keys: {len(missing)}"
    )
    print(
        f"Initialization unexpected keys: {len(unexpected)}"
    )

    if missing or unexpected:
        raise RuntimeError(
            "Part 2.20B checkpoint/model mismatch."
        )

    model = model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )

    history = []
    best_accuracy = -float("inf")
    best_epoch = None

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(
            set_to_none=True
        )

        train_total = []
        train_point = []
        train_r2 = []
        train_r4 = []
        train_bg = []

        unique_cases = set()

        for slot in range(
            TRAIN_SLOTS_PER_EPOCH
        ):
            study_id, series_id, point_df = (
                choose_training_case(
                    training_cases,
                    sampling_weights,
                )
            )

            unique_cases.add(
                (study_id, series_id)
            )

            image, loaded_points, geometry = (
                part220b.load_case(
                    study_id,
                    series_id,
                    point_df,
                )
            )

            loaded_points = normalize_loaded_points(
                loaded_points
            )

            (
                total_loss,
                point_loss,
                r2_loss,
                r4_loss,
                bg_loss,
            ) = train_case(
                model,
                optimizer,
                scaler,
                image,
                loaded_points,
            )

            scaled_loss = (
                total_loss / GRAD_ACCUMULATION
            )

            scaler.scale(
                scaled_loss
            ).backward()

            if (
                (slot + 1) % GRAD_ACCUMULATION == 0
                or slot == TRAIN_SLOTS_PER_EPOCH - 1
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(
                    set_to_none=True
                )

            train_total.append(
                float(total_loss.item())
            )
            train_point.append(
                float(point_loss.item())
            )
            train_r2.append(
                float(r2_loss.item())
            )
            train_r4.append(
                float(r4_loss.item())
            )
            train_bg.append(
                float(bg_loss.item())
            )

        scheduler.step()

        # -----------------------------------------------------------
        # Validation
        # -----------------------------------------------------------

        validation_rows = []
        validation_fg = []

        model.eval()

        for study_id, series_id, point_df in (
            [
                (
                    str(study),
                    str(series),
                    group.copy(),
                )
                for (study, series), group
                in validation_cases
            ]
        ):
            image, loaded_points, geometry = (
                part220b.load_case(
                    study_id,
                    series_id,
                    point_df,
                )
            )

            loaded_points = normalize_loaded_points(
                loaded_points
            )

            case_results = evaluate_case(
                model,
                image,
                loaded_points,
            )

            case_results["study_id"] = study_id
            case_results["series_id"] = series_id

            validation_rows.append(
                case_results
            )

            validation_fg.append(
                float(
                    case_results[
                        "foreground_ratio"
                    ].mean()
                )
            )

        val_df = pd.concat(
            validation_rows,
            ignore_index=True,
        )

        overall_accuracy = float(
            val_df["correct"].mean()
        )

        disease_accuracy = (
            val_df.groupby(
                "true_class"
            )["correct"]
            .mean()
        )

        macro_accuracy = float(
            disease_accuracy.mean()
        )

        mean_probability = float(
            val_df["true_probability"].mean()
        )

        hit_rate = float(
            val_df["hit_at_0_50"].mean()
        )

        mean_fg = float(
            np.mean(validation_fg)
        )

        disease_metrics = {}

        for cid in range(1, 6):
            sub = val_df[
                val_df["true_class"] == cid
            ]

            if len(sub):
                disease_metrics[
                    CLASS_NAMES[cid]
                ] = {
                    "points": int(len(sub)),
                    "accuracy": float(
                        sub["correct"].mean()
                    ),
                    "mean_true_probability": float(
                        sub["true_probability"].mean()
                    ),
                    "hit_rate_at_0_50": float(
                        sub["hit_at_0_50"].mean()
                    ),
                }

        record = {
            "epoch": epoch,
            "train_loss": float(
                np.mean(train_total)
            ),
            "train_point_loss": float(
                np.mean(train_point)
            ),
            "train_radius2_loss": float(
                np.mean(train_r2)
            ),
            "train_radius4_loss": float(
                np.mean(train_r4)
            ),
            "train_background_loss": float(
                np.mean(train_bg)
            ),
            "unique_training_cases": len(
                unique_cases
            ),
            "validation_points": int(
                len(val_df)
            ),
            "overall_accuracy": overall_accuracy,
            "macro_disease_accuracy": macro_accuracy,
            "mean_true_probability": mean_probability,
            "hit_rate_at_0_50": hit_rate,
            "mean_foreground_ratio": mean_fg,
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "disease_metrics": disease_metrics,
        }

        history.append(record)

        print("")
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )
        print(
            f"  unique training cases: "
            f"{len(unique_cases)}/{TRAIN_SLOTS_PER_EPOCH}"
        )
        print(
            f"  train loss: "
            f"{record['train_loss']:.6f}"
        )
        print(
            f"  point loss: "
            f"{record['train_point_loss']:.6f}"
        )
        print(
            f"  radius2 loss: "
            f"{record['train_radius2_loss']:.6f}"
        )
        print(
            f"  radius4 loss: "
            f"{record['train_radius4_loss']:.6f}"
        )
        print(
            f"  background loss: "
            f"{record['train_background_loss']:.6f}"
        )
        print(
            f"  validation points: "
            f"{len(val_df)}"
        )
        print(
            f"  overall accuracy: "
            f"{overall_accuracy:.6f}"
        )
        print(
            f"  macro disease accuracy: "
            f"{macro_accuracy:.6f}"
        )
        print(
            f"  mean true probability: "
            f"{mean_probability:.6f}"
        )
        print(
            f"  hit @0.50: "
            f"{hit_rate:.6f}"
        )
        print(
            f"  mean foreground ratio: "
            f"{mean_fg:.6f}"
        )

        for cid in range(1, 6):
            name = CLASS_NAMES[cid]
            metrics = disease_metrics.get(name)

            if metrics:
                print(
                    f"  {name}: "
                    f"{metrics['accuracy']:.6f}"
                )

        # Save per-epoch metrics.
        (
            METRICS_DIR
            / f"part227_epoch_{epoch:02d}_validation.csv"
        ).write_text(
            val_df.to_csv(index=False),
            encoding="utf-8",
        )

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"part227_epoch_{epoch:02d}.pth"
        )

        torch.save(
            {
                "part": "2.27",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "validation_metrics": record,
                "initialization_checkpoint": str(
                    INIT_CHECKPOINT
                ),
            },
            checkpoint_path,
        )

        if macro_accuracy > best_accuracy:
            best_accuracy = macro_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "part": "2.27",
                    "best_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": record,
                    "initialization_checkpoint": str(
                        INIT_CHECKPOINT
                    ),
                },
                CHECKPOINT_DIR
                / "part227_best_macro_disease.pth",
            )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    summary = {
        "part": "2.27",
        "status": "COMPLETE",
        "epochs": EPOCHS,
        "train_slots_per_epoch": TRAIN_SLOTS_PER_EPOCH,
        "validation_cases": len(validation_cases),
        "validation_points": int(
            history[-1]["validation_points"]
        ),
        "initialization_checkpoint": str(
            INIT_CHECKPOINT
        ),
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "point_loss_weight": POINT_LOSS_WEIGHT,
        "radius2_loss_weight": RADIUS2_LOSS_WEIGHT,
        "radius4_loss_weight": RADIUS4_LOSS_WEIGHT,
        "background_loss_weight": BACKGROUND_LOSS_WEIGHT,
        "radii": [RADIUS2, RADIUS4],
        "best_epoch_by_macro_disease_accuracy": best_epoch,
        "best_macro_disease_accuracy": best_accuracy,
        "history": history,
        "training_performed": True,
        "part220b_modified": False,
        "part104_modified": False,
        "dashboard_modified": False,
        "voxel_ground_truth_fabricated": False,
        "scientific_limitation": (
            "RSNA annotations are point/localization annotations, not "
            "manual voxel-wise segmentation masks. Multi-scale losses "
            "supervise aggregated point neighborhoods and do not establish "
            "voxel-wise segmentation ground truth."
        ),
    }

    (REPORT_DIR / "part227_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "PART 2.27 — MULTI-SCALE POINT-SUPERVISED SWIN-UNETR TRAINING",
        "",
        f"Validation cases: {len(validation_cases)}",
        f"Validation points: {history[-1]['validation_points']}",
        f"Best epoch: {best_epoch}",
        f"Best macro disease accuracy: {best_accuracy:.12f}",
        "",
        "EPOCH SUMMARY",
    ]

    for h in history:
        report_lines.append(
            f"Epoch {h['epoch']}: "
            f"overall={h['overall_accuracy']:.6f}, "
            f"macro={h['macro_disease_accuracy']:.6f}, "
            f"prob={h['mean_true_probability']:.6f}, "
            f"hit={h['hit_rate_at_0_50']:.6f}, "
            f"fg={h['mean_foreground_ratio']:.6f}"
        )

    report_lines += [
        "",
        "DISEASE METRICS — FINAL EPOCH",
    ]

    for name, metrics in history[-1][
        "disease_metrics"
    ].items():
        report_lines.append(
            f"{name}: "
            f"points={metrics['points']}, "
            f"accuracy={metrics['accuracy']:.6f}, "
            f"prob={metrics['mean_true_probability']:.6f}, "
            f"hit={metrics['hit_rate_at_0_50']:.6f}"
        )

    report_lines += [
        "",
        "Training performed: YES",
        "Part 2.20B modified: NO",
        "Part104 modified: NO",
        "Dashboard modified: NO",
        "Voxel ground truth fabricated: NO",
        "",
        "RSNA coordinates remain point/localization annotations.",
        "The multi-scale losses do not constitute manual voxel-wise masks.",
    ]

    (REPORT_DIR / "part227_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "epoch": h["epoch"],
                "train_loss": h["train_loss"],
                "point_loss": h["train_point_loss"],
                "radius2_loss": h["train_radius2_loss"],
                "radius4_loss": h["train_radius4_loss"],
                "background_loss": h["train_background_loss"],
                "overall_accuracy": h["overall_accuracy"],
                "macro_disease_accuracy": h[
                    "macro_disease_accuracy"
                ],
                "mean_true_probability": h[
                    "mean_true_probability"
                ],
                "hit_rate_at_0_50": h[
                    "hit_rate_at_0_50"
                ],
                "mean_foreground_ratio": h[
                    "mean_foreground_ratio"
                ],
                "learning_rate": h["learning_rate"],
            }
            for h in history
        ]
    ).to_csv(
        METRICS_DIR / "part227_training_history.csv",
        index=False,
    )

    print("")
    print("=" * 72)
    print("PART 2.27 COMPLETE")
    print("=" * 72)
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Best macro disease accuracy: "
        f"{best_accuracy:.6f}"
    )
    print(
        f"Final overall accuracy: "
        f"{history[-1]['overall_accuracy']:.6f}"
    )
    print(
        f"Final macro disease accuracy: "
        f"{history[-1]['macro_disease_accuracy']:.6f}"
    )
    print(
        f"Final mean true probability: "
        f"{history[-1]['mean_true_probability']:.6f}"
    )
    print(
        f"Final hit @0.50: "
        f"{history[-1]['hit_rate_at_0_50']:.6f}"
    )
    print(
        f"Final foreground ratio: "
        f"{history[-1]['mean_foreground_ratio']:.6f}"
    )
    print("")
    print(f"Outputs: {OUTPUT_DIR}")
    print("")
    print("Part 2.20B modified: NO")
    print("Part104 modified: NO")
    print("Dashboard modified: NO")
    print("Voxel ground truth fabricated: NO")


if __name__ == "__main__":
    main()
