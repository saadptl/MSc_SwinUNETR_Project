"""
PART 2.14
Point-Supervised Swin-UNETR Controlled Training Pilot

Purpose
-------
Train a separate Swin-UNETR experiment using the RSNA point annotations
validated by Part 2.13.

IMPORTANT
---------
RSNA annotations are point/localization annotations, NOT voxel-wise
segmentation masks.

Therefore this experiment does NOT fabricate segmentation masks.

The loss uses:
    1. positive point supervision
    2. sampled background/negative supervision
    3. a small spatial consistency regularizer

No existing protected checkpoint is overwritten.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp import GradScaler, autocast

from monai.networks.nets import SwinUNETR


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MODEL_SHAPE = (64, 96, 96)

NUM_CLASSES = 6
IN_CHANNELS = 1
FEATURE_SIZE = 12

TRAIN_SERIES = 100
VAL_SERIES = 25

EPOCHS = 3

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5

# Number of negative/background points per series.
NEGATIVE_POINTS_PER_SERIES = 64

# Point supervision is the primary objective.
POINT_LOSS_WEIGHT = 1.0

# Background supervision is intentionally weaker.
BACKGROUND_LOSS_WEIGHT = 0.10

# Small regularization encouraging local smoothness.
SMOOTHNESS_WEIGHT = 0.01


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part213_point_supervision_manifest"
    / "part213_point_manifest.csv"
)

PROTECTED_CHECKPOINT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part104_final_checkpoint_selection"
    / "checkpoints"
    / "final_segmentation_model.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part214_point_supervised_training"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
VIS_DIR = OUTPUT_DIR / "visualizations"
REPORT_DIR = OUTPUT_DIR / "reports"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

def prepare_output_dirs() -> None:
    for directory in (
        OUTPUT_DIR,
        CHECKPOINT_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        VIS_DIR,
        REPORT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# MANIFEST
# ============================================================

def load_manifest() -> pd.DataFrame:

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Part 2.13 manifest not found:\n{MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "study_id",
        "series_id",
        "class_id",
        "class_name",
        "level",
        "model_z",
        "model_y",
        "model_x",
        "model_z_float",
        "model_y_float",
        "model_x_float",
        "model_shape",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Part 2.13 manifest is missing required columns:\n"
            + "\n".join(missing)
        )

    df["study_id"] = df["study_id"].astype(str)
    df["series_id"] = df["series_id"].astype(str)

    df["class_id"] = df["class_id"].astype(int)

    for column in (
        "model_z",
        "model_y",
        "model_x",
    ):
        df[column] = df[column].astype(int)

    for column in (
        "model_z_float",
        "model_y_float",
        "model_x_float",
    ):
        df[column] = df[column].astype(float)

    return df


# ============================================================
# SERIES-LEVEL SPLIT
# ============================================================

def make_series_split(df: pd.DataFrame):

    series_keys = (
        df[["study_id", "series_id"]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )

    required = TRAIN_SERIES + VAL_SERIES

    if len(series_keys) < required:
        raise RuntimeError(
            f"Need at least {required} series, "
            f"but manifest contains {len(series_keys)}."
        )

    train_keys = series_keys.iloc[:TRAIN_SERIES].copy()
    val_keys = series_keys.iloc[
        TRAIN_SERIES:TRAIN_SERIES + VAL_SERIES
    ].copy()

    train_key_set = set(
        zip(
            train_keys.study_id,
            train_keys.series_id,
        )
    )

    val_key_set = set(
        zip(
            val_keys.study_id,
            val_keys.series_id,
        )
    )

    train_df = df[
        df.apply(
            lambda row: (
                row.study_id,
                row.series_id,
            ) in train_key_set,
            axis=1,
        )
    ].copy()

    val_df = df[
        df.apply(
            lambda row: (
                row.study_id,
                row.series_id,
            ) in val_key_set,
            axis=1,
        )
    ].copy()

    return train_df, val_df


# ============================================================
# LOAD PART 11
# ============================================================

def load_part11():

    path = (
        ROOT
        / "src"
        / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Part 11 file not found:\n{path}"
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "part11_module",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not load Part 11.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# DICOM → MODEL INPUT
# ============================================================

def load_series_tensor(part11, study_id: str, series_id: str):
    """
    Load the DICOM series using Part 11 DICOM utilities.

    No pseudo-mask is loaded.

    The same image preprocessing used by Part 11 is reproduced
    without requiring its pseudo-mask.
    """

    row = pd.Series(
        {
            "study_id": study_id,
            "series_id": series_id,
        }
    )

    series_dir = part11.resolve_series_dir(row)

    image_native, dicom_records, dicom_info = (
        part11.read_dicom_series_robust(series_dir)
    )

    image_native = np.asarray(
        image_native,
        dtype=np.float32,
    )

    if image_native.ndim != 3:
        raise ValueError(
            f"Expected 3D image, got {image_native.shape}"
        )

    # --------------------------------------------------------
    # Resize to established model shape
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        image_native
    ).float()

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=MODEL_SHAPE,
        mode="trilinear",
        align_corners=False,
    )

    tensor = tensor.squeeze(0)

    # --------------------------------------------------------
    # Percentile normalization
    # --------------------------------------------------------

    values = tensor.numpy()

    p1 = np.percentile(values, 1)
    p99 = np.percentile(values, 99)

    if p99 > p1:
        values = np.clip(
            values,
            p1,
            p99,
        )

        values = (
            values - p1
        ) / (p99 - p1)

    else:
        values = np.zeros_like(values)

    tensor = torch.from_numpy(
        values.astype(np.float32)
    )

    return tensor, image_native.shape


# ============================================================
# POINT GROUPING
# ============================================================

def group_points(df: pd.DataFrame):

    grouped = {}

    for (study_id, series_id), group in df.groupby(
        ["study_id", "series_id"]
    ):

        points = []

        for _, row in group.iterrows():

            z = int(row["model_z"])
            y = int(row["model_y"])
            x = int(row["model_x"])

            class_id = int(row["class_id"])

            if not (
                0 <= z < MODEL_SHAPE[0]
                and 0 <= y < MODEL_SHAPE[1]
                and 0 <= x < MODEL_SHAPE[2]
            ):
                continue

            points.append(
                {
                    "z": z,
                    "y": y,
                    "x": x,
                    "class_id": class_id,
                    "class_name": row["class_name"],
                    "level": row["level"],
                }
            )

        grouped[
            (study_id, series_id)
        ] = points

    return grouped


# ============================================================
# SPARSE POINT LOSS
# ============================================================

def point_supervision_loss(
    logits: torch.Tensor,
    points,
):
    """
    Sparse positive supervision.

    logits:
        [B, C, Z, Y, X]

    Each RSNA point explicitly supervises its annotated
    disease class.

    No surrounding voxel is declared positive.
    """

    if not points:
        return logits.sum() * 0.0

    positive_losses = []

    for point in points:

        z = point["z"]
        y = point["y"]
        x = point["x"]
        class_id = point["class_id"]

        point_logits = logits[
            0,
            :,
            z,
            y,
            x,
        ].unsqueeze(0)

        target = torch.tensor(
            [class_id],
            dtype=torch.long,
            device=logits.device,
        )

        loss = F.cross_entropy(
            point_logits,
            target,
        )

        positive_losses.append(loss)

    return torch.stack(
        positive_losses
    ).mean()


# ============================================================
# BACKGROUND SAMPLE LOSS
# ============================================================

def background_loss(
    logits: torch.Tensor,
    points,
):
    """
    Weak background supervision.

    Random locations sufficiently far from annotation points
    are sampled as background.

    This is deliberately weighted much lower than the
    positive point term.
    """

    occupied = {
        (
            point["z"],
            point["y"],
            point["x"],
        )
        for point in points
    }

    candidates = []

    attempts = 0
    maximum_attempts = NEGATIVE_POINTS_PER_SERIES * 30

    while (
        len(candidates) < NEGATIVE_POINTS_PER_SERIES
        and attempts < maximum_attempts
    ):

        attempts += 1

        z = random.randrange(MODEL_SHAPE[0])
        y = random.randrange(MODEL_SHAPE[1])
        x = random.randrange(MODEL_SHAPE[2])

        if (z, y, x) in occupied:
            continue

        # Minimum local separation from positive points.
        too_close = False

        for point in points:

            dz = z - point["z"]
            dy = y - point["y"]
            dx = x - point["x"]

            distance_sq = (
                dz * dz
                + dy * dy
                + dx * dx
            )

            if distance_sq < 25:
                too_close = True
                break

        if not too_close:
            candidates.append(
                (z, y, x)
            )

    if not candidates:
        return logits.sum() * 0.0

    losses = []

    for z, y, x in candidates:

        point_logits = logits[
            0,
            :,
            z,
            y,
            x,
        ].unsqueeze(0)

        target = torch.zeros(
            1,
            dtype=torch.long,
            device=logits.device,
        )

        losses.append(
            F.cross_entropy(
                point_logits,
                target,
            )
        )

    return torch.stack(losses).mean()


# ============================================================
# SPATIAL SMOOTHNESS
# ============================================================

def smoothness_loss(logits: torch.Tensor):

    probability = torch.softmax(
        logits,
        dim=1,
    )

    dz = torch.abs(
        probability[:, :, 1:, :, :]
        - probability[:, :, :-1, :, :]
    ).mean()

    dy = torch.abs(
        probability[:, :, :, 1:, :]
        - probability[:, :, :, :-1, :]
    ).mean()

    dx = torch.abs(
        probability[:, :, :, :, 1:]
        - probability[:, :, :, :, :-1]
    ).mean()

    return (
        dz + dy + dx
    ) / 3.0


# ============================================================
# POINT VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_points(
    model,
    part11,
    grouped_points,
    device,
):

    model.eval()

    total_points = 0
    correct_class = 0

    probabilities = []

    distances = []

    series_items = list(
        grouped_points.items()
    )

    for index, (
        key,
        points,
    ) in enumerate(series_items):

        study_id, series_id = key

        try:

            image, _ = load_series_tensor(
                part11,
                study_id,
                series_id,
            )

            image = image.unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            with autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                logits = model(image)

            prob = torch.softmax(
                logits,
                dim=1,
            )

            for point in points:

                z = point["z"]
                y = point["y"]
                x = point["x"]

                class_id = point["class_id"]

                p = float(
                    prob[
                        0,
                        class_id,
                        z,
                        y,
                        x,
                    ].item()
                )

                predicted_class = int(
                    torch.argmax(
                        prob[
                            0,
                            :,
                            z,
                            y,
                            x,
                        ]
                    ).item()
                )

                total_points += 1

                if predicted_class == class_id:
                    correct_class += 1

                probabilities.append(p)

                # ------------------------------------------------
                # Local predicted-region distance
                # ------------------------------------------------

                class_map = (
                    prob[
                        0,
                        class_id,
                    ]
                    >= 0.5
                )

                locations = torch.nonzero(
                    class_map,
                    as_tuple=False,
                )

                if locations.numel() > 0:

                    target = torch.tensor(
                        [z, y, x],
                        device=locations.device,
                        dtype=torch.float32,
                    )

                    coords = locations.float()

                    d = torch.sqrt(
                        torch.sum(
                            (coords - target) ** 2,
                            dim=1,
                        )
                    )

                    distances.append(
                        float(d.min().item())
                    )

        except Exception as exc:

            print(
                f"  Validation warning "
                f"{study_id}/{series_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    result = {
        "points_evaluated": total_points,
        "point_class_accuracy": (
            correct_class / total_points
            if total_points
            else 0.0
        ),
        "mean_probability_at_point": (
            float(np.mean(probabilities))
            if probabilities
            else 0.0
        ),
        "median_probability_at_point": (
            float(np.median(probabilities))
            if probabilities
            else 0.0
        ),
        "mean_predicted_region_distance_voxels": (
            float(np.mean(distances))
            if distances
            else None
        ),
        "median_predicted_region_distance_voxels": (
            float(np.median(distances))
            if distances
            else None
        ),
    }

    return result


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
):

    path = (
        CHECKPOINT_DIR
        / f"part214_epoch_{epoch:02d}.pth"
    )

    torch.save(
        {
            "part": "2.14",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_shape": MODEL_SHAPE,
            "num_classes": NUM_CLASSES,
            "feature_size": FEATURE_SIZE,
            "seed": SEED,
        },
        path,
    )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("PART 2.14 — POINT-SUPERVISED SWIN-UNETR")
    print("=" * 78)

    set_seed()
    prepare_output_dirs()

    print("\nConfiguration:")
    print(f"  Model shape          : {MODEL_SHAPE}")
    print(f"  Training series      : {TRAIN_SERIES}")
    print(f"  Validation series    : {VAL_SERIES}")
    print(f"  Epochs               : {EPOCHS}")
    print(f"  Batch size           : {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRAD_ACCUMULATION}")
    print(f"  Learning rate        : {LEARNING_RATE}")

    # --------------------------------------------------------
    # Protected checkpoint
    # --------------------------------------------------------

    if not PROTECTED_CHECKPOINT.exists():

        raise FileNotFoundError(
            "Protected Part 104 checkpoint was not found:\n"
            f"{PROTECTED_CHECKPOINT}"
        )

    print(
        "\nProtected checkpoint:"
    )
    print(
        f"  {PROTECTED_CHECKPOINT}"
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    df = load_manifest()

    print("\nManifest:")
    print(
        f"  Annotation points : {len(df)}"
    )
    print(
        "  Annotated series  : "
        f"{df[['study_id','series_id']].drop_duplicates().shape[0]}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_df, val_df = make_series_split(
        df
    )

    print("\nPilot split:")
    print(
        "  Training series   : "
        f"{train_df[['study_id','series_id']].drop_duplicates().shape[0]}"
    )

    print(
        "  Validation series : "
        f"{val_df[['study_id','series_id']].drop_duplicates().shape[0]}"
    )

    train_points = group_points(
        train_df
    )

    val_points = group_points(
        val_df
    )

    # --------------------------------------------------------
    # Part 11
    # --------------------------------------------------------

    print(
        "\nLoading established Part 11 DICOM utilities..."
    )

    part11 = load_part11()

    print(
        "  Part 11: PASS"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nCreating Swin-UNETR..."
    )

    model = SwinUNETR(
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        spatial_dims=3,
        use_checkpoint=False,
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Load Part 104 initialization
    # --------------------------------------------------------

    checkpoint = torch.load(
        PROTECTED_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get(
            "state_dict",
            checkpoint,
        ),
    )

    cleaned_state = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):
            new_key = new_key[
                len("module.") :
            ]

        cleaned_state[
            new_key
        ] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state,
            strict=False,
        )
    )

    print(
        "  Initialization checkpoint loaded."
    )

    print(
        f"  Missing keys    : {len(missing)}"
    )

    print(
        f"  Unexpected keys : {len(unexpected)}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
    )
    history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting controlled pilot...")
    print("-" * 78)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_total = 0.0
        running_point = 0.0
        running_background = 0.0
        running_smooth = 0.0

        processed = 0

        train_items = list(
            train_points.items()
        )

        random.Random(
            SEED + epoch
        ).shuffle(
            train_items
        )

        for item_index, (
            key,
            points,
        ) in enumerate(
            train_items
        ):

            study_id, series_id = key

            try:

                image, native_shape = (
                    load_series_tensor(
                        part11,
                        study_id,
                        series_id,
                    )
                )

                image = image.unsqueeze(
                    0
                ).to(
                    device,
                    non_blocking=True,
                )

                with autocast(
                    "cuda",
                    enabled=torch.cuda.is_available(),
                ):

                    logits = model(
                        image
                    )

                    loss_point = (
                        point_supervision_loss(
                            logits,
                            points,
                        )
                    )

                    loss_background = (
                        background_loss(
                            logits,
                            points,
                        )
                    )

                    loss_smooth = (
                        smoothness_loss(
                            logits
                        )
                    )

                    loss = (
                        POINT_LOSS_WEIGHT
                        * loss_point
                        + BACKGROUND_LOSS_WEIGHT
                        * loss_background
                        + SMOOTHNESS_WEIGHT
                        * loss_smooth
                    )

                    loss_scaled = (
                        loss
                        / GRAD_ACCUMULATION
                    )

                scaler.scale(
                    loss_scaled
                ).backward()

                if (
                    (item_index + 1)
                    % GRAD_ACCUMULATION
                    == 0
                    or
                    item_index
                    == len(train_items) - 1
                ):

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                running_total += float(
                    loss.item()
                )

                running_point += float(
                    loss_point.item()
                )

                running_background += float(
                    loss_background.item()
                )

                running_smooth += float(
                    loss_smooth.item()
                )

                processed += 1

                if (
                    processed == 1
                    or processed % 10 == 0
                    or processed == len(train_items)
                ):

                    print(
                        f"  Epoch {epoch}/{EPOCHS} "
                        f"| {processed}/{len(train_items)} "
                        f"| loss={loss.item():.5f}"
                    )

            except RuntimeError as exc:

                if (
                    "out of memory"
                    in str(exc).lower()
                    and torch.cuda.is_available()
                ):

                    torch.cuda.empty_cache()

                    print(
                        "\nCUDA OUT OF MEMORY:"
                    )

                    print(
                        "  Reduce model/input size before "
                        "continuing."
                    )

                    raise

                print(
                    f"  Training warning "
                    f"{study_id}/{series_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

            except Exception as exc:

                print(
                    f"  Training warning "
                    f"{study_id}/{series_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print(
            f"\nValidation epoch {epoch}..."
        )

        validation = evaluate_points(
            model,
            part11,
            val_points,
            device,
        )

        epoch_metrics = {
            "epoch": epoch,
            "processed_training_series": processed,
            "train_total_loss": (
                running_total / max(processed, 1)
            ),
            "train_point_loss": (
                running_point / max(processed, 1)
            ),
            "train_background_loss": (
                running_background / max(processed, 1)
            ),
            "train_smoothness_loss": (
                running_smooth / max(processed, 1)
            ),
            "validation": validation,
        }

        history.append(
            epoch_metrics
        )

        print(
            "\nValidation metrics:"
        )

        print(
            "  Points evaluated     : "
            f"{validation['points_evaluated']}"
        )

        print(
            "  Point class accuracy : "
            f"{validation['point_class_accuracy']:.6f}"
        )

        print(
            "  Mean point probability: "
            f"{validation['mean_probability_at_point']:.6f}"
        )

        print(
            "  Median point probability: "
            f"{validation['median_probability_at_point']:.6f}"
        )

        print(
            "  Mean region distance : "
            f"{validation['mean_predicted_region_distance_voxels']}"
        )

        print(
            "  Median region distance: "
            f"{validation['median_predicted_region_distance_voxels']}"
        )

        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            epoch,
            epoch_metrics,
        )

        print(
            "\nCheckpoint saved:"
        )

        print(
            f"  {checkpoint_path}"
        )

        with open(
            METRICS_DIR
            / "part214_training_history.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    summary = {
        "part": "2.14",
        "status": "COMPLETE",
        "experiment": (
            "controlled_point_supervised_training_pilot"
        ),
        "manifest": str(
            MANIFEST_PATH
        ),
        "model_shape": MODEL_SHAPE,
        "num_classes": NUM_CLASSES,
        "feature_size": FEATURE_SIZE,
        "training_series": TRAIN_SERIES,
        "validation_series": VAL_SERIES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "history": history,
        "scientific_limitation": (
            "RSNA coordinates are point/localization "
            "annotations, not manual voxel-wise "
            "segmentation masks."
        ),
        "dashboard_enabled": False,
        "protected_checkpoint_modified": False,
    }

    with open(
        REPORT_DIR
        / "part214_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n")
    print("=" * 78)
    print("PART 2.14 COMPLETE")
    print("=" * 78)

    print(
        "\nTraining checkpoint(s):"
    )

    print(
        f"  {CHECKPOINT_DIR}"
    )

    print(
        "\nMetrics:"
    )

    print(
        f"  {METRICS_DIR}"
    )

    print(
        "\nReport:"
    )

    print(
        f"  {REPORT_DIR / 'part214_summary.json'}"
    )

    print(
        "\nProtected Part 104 checkpoint was NOT overwritten."
    )

    print(
        "Dashboard integration remains disabled."
    )


if __name__ == "__main__":
    main()