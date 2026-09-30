"""
PART 3.1
STRONG POINT-SUPERVISED SWIN-UNETR TRAINING

Goal
----
Build the next actual model-improvement checkpoint using the verified
Part 2.20B physical-space preprocessing and the existing Swin-UNETR.

Scientific contract
-------------------
- RSNA labels are point/localization annotations, NOT voxel masks.
- No manual voxel ground truth is fabricated.
- No Dice score is claimed.
- A new untouched 25-study test cohort is reserved.
- The existing 25-study development cohort is used for epoch/model selection.
- Test results are produced only from the selected best development checkpoint.
- Existing checkpoints and dashboard are never overwritten.

Training objective
------------------
1. Point cross entropy.
2. Target-vs-background margin loss.
3. Distance-weighted local point supervision.
4. Weak background regularization outside protected point neighborhoods.
5. Disease-balanced case sampling.

Initialization
--------------
Prefer the previously strongest measured Part 2.16 Epoch-5 checkpoint if
present. Otherwise fall back to Part 2.27 best checkpoint.

Output
------
outputs/segmentation/rsna_part31_strong_point_supervised_training/
"""

from __future__ import annotations

import json
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


# ============================================================================
# CONFIG
# ============================================================================

SEED = 42

MODEL_SHAPE = (64, 96, 96)
NUM_CLASSES = 6

EPOCHS = 8
TRAIN_SLOTS_PER_EPOCH = 100

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4

LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 1.0e-5

# Loss weights.
POINT_CE_WEIGHT = 1.00
MARGIN_WEIGHT = 0.35
LOCAL_WEIGHT = 0.30
BACKGROUND_WEIGHT = 0.08

# Target-vs-background margin at annotated points.
TARGET_BG_MARGIN = 0.50

# Local weak supervision. This is a weighted probability aggregate,
# not a voxel-wise segmentation mask.
LOCAL_RADIUS = 2
LOCAL_SIGMA = 1.25

BACKGROUND_SAMPLES = 128
BACKGROUND_EXCLUSION_RADIUS = 4

# Disease-aware case sampling.
RFNN_CASE_WEIGHT = 2.0
RSS_CASE_WEIGHT = 1.5

VALIDATION_CASES = 25
TEST_CASES = 25

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

PART216_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part216_point_supervised_balanced_training"
    / "checkpoints"
)

PART227_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part31_strong_point_supervised_training"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORT_DIR = OUTPUT_DIR / "reports"

for directory in (
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    METRICS_DIR,
    REPORT_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# ============================================================================
# CHECKPOINT SELECTION
# ============================================================================

def choose_initialization_checkpoint():

    candidates = []

    if PART216_DIR.exists():

        preferred = [
            PART216_DIR / "part216_epoch_05.pth",
            PART216_DIR / "part216_best_macro_disease.pth",
        ]

        for path in preferred:
            if path.exists():
                candidates.append(path)

        candidates.extend(
            sorted(
                PART216_DIR.glob("*.pth"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        )

    if PART227_DIR.exists():

        preferred = [
            PART227_DIR / "part227_best_macro_disease.pth",
            PART227_DIR / "part227_epoch_05.pth",
        ]

        for path in preferred:
            if path.exists():
                candidates.append(path)

        candidates.extend(
            sorted(
                PART227_DIR.glob("*.pth"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        )

    seen = set()

    for path in candidates:

        path = path.resolve()

        if path in seen:
            continue

        seen.add(path)

        if path.exists():
            return path

    raise FileNotFoundError(
        "No Part 2.16 or Part 2.27 checkpoint was found.\n"
        f"Checked:\n{PART216_DIR}\n{PART227_DIR}"
    )


# ============================================================================
# DATA SPLITS
# ============================================================================

def build_dev_split(manifest):

    dev = part220b.select_validation_series(
        manifest
    ).copy()

    dev["study_id"] = dev["study_id"].astype(str)
    dev["series_id"] = dev["series_id"].astype(str)

    if len(dev) != VALIDATION_CASES:
        raise RuntimeError(
            f"Expected {VALIDATION_CASES} development cases, "
            f"found {len(dev)}."
        )

    return dev


def build_test_split(manifest, dev_series):

    """
    Reproduce the deterministic Part 2.20B study shuffle, then reserve
    the next 25 studies after the original development-study block.

    These studies were not used by Part 2.20B/2.27 validation.
    """

    work = manifest.copy()
    work["study_id"] = work["study_id"].astype(str)
    work["series_id"] = work["series_id"].astype(str)

    dev_studies = set(
        dev_series["study_id"].astype(str)
    )

    all_studies = np.array(
        sorted(
            work["study_id"].unique()
        )
    )

    rng = np.random.default_rng(SEED)
    shuffled = all_studies.copy()
    rng.shuffle(shuffled)

    broad_validation_studies = set(
        shuffled[
            : min(395, len(shuffled))
        ]
    )

    forbidden = (
        dev_studies
        | broad_validation_studies
    )

    test_candidates = [
        study
        for study in shuffled
        if study not in forbidden
    ]

    if len(test_candidates) < TEST_CASES:

        raise RuntimeError(
            "Not enough untouched studies for test cohort."
        )

    test_studies = test_candidates[
        :TEST_CASES
    ]

    rows = []

    for study_id in test_studies:

        series = (
            work[
                work["study_id"] == str(study_id)
            ][
                ["study_id", "series_id"]
            ]
            .drop_duplicates()
            .sort_values(
                ["study_id", "series_id"]
            )
        )

        if len(series) == 0:
            continue

        rows.append(
            series.iloc[0]
        )

    test = pd.DataFrame(rows)

    if len(test) != TEST_CASES:

        raise RuntimeError(
            f"Expected {TEST_CASES} test cases, "
            f"found {len(test)}."
        )

    return test.reset_index(drop=True)


def manifest_for_series(
    manifest,
    series_df,
):

    keys = set(
        zip(
            series_df["study_id"].astype(str),
            series_df["series_id"].astype(str),
        )
    )

    work = manifest.copy()
    work["study_id"] = work["study_id"].astype(str)
    work["series_id"] = work["series_id"].astype(str)

    return work[
        work.apply(
            lambda r: (
                r["study_id"],
                r["series_id"],
            ) in keys,
            axis=1,
        )
    ].copy()


def build_case_groups(
    manifest,
    series_df,
):

    selected_manifest = manifest_for_series(
        manifest,
        series_df,
    )

    groups = []

    for (
        study_id,
        series_id,
    ), group in selected_manifest.groupby(
        ["study_id", "series_id"],
        sort=True,
    ):

        groups.append(
            (
                str(study_id),
                str(series_id),
                group.copy(),
            )
        )

    return groups


def build_training_cases(
    manifest,
    dev_series,
    test_series,
):

    excluded_studies = set(
        dev_series["study_id"].astype(str)
    ) | set(
        test_series["study_id"].astype(str)
    )

    work = manifest.copy()
    work["study_id"] = work["study_id"].astype(str)
    work["series_id"] = work["series_id"].astype(str)

    training = work[
        ~work["study_id"].isin(
            excluded_studies
        )
    ].copy()

    groups = []

    for (
        study_id,
        series_id,
    ), group in training.groupby(
        ["study_id", "series_id"],
        sort=True,
    ):

        groups.append(
            (
                str(study_id),
                str(series_id),
                group.copy(),
            )
        )

    if not groups:
        raise RuntimeError(
            "No training cases remain."
        )

    return groups


# ============================================================================
# SAMPLING
# ============================================================================

def build_sampling_weights(
    training_cases,
):

    weights = []

    for (
        study_id,
        series_id,
        points,
    ) in training_cases:

        classes = set(
            points["class_id"]
            .astype(int)
            .tolist()
        )

        weight = 1.0

        if 3 in classes:
            weight *= RFNN_CASE_WEIGHT

        if 5 in classes:
            weight *= RSS_CASE_WEIGHT

        weights.append(weight)

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    weights /= weights.sum()

    return weights


def choose_training_case(
    training_cases,
    weights,
):

    index = np.random.choice(
        len(training_cases),
        p=weights,
    )

    return training_cases[
        int(index)
    ]


# ============================================================================
# CLASS WEIGHTS
# ============================================================================

def compute_class_weights(
    training_manifest,
):

    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    for cid in training_manifest[
        "class_id"
    ].astype(int):

        if 1 <= cid < NUM_CLASSES:
            counts[cid] += 1

    disease_counts = counts[1:]

    inverse = (
        disease_counts.sum()
        /
        np.maximum(
            disease_counts,
            1.0,
        )
    )

    inverse /= inverse.mean()

    inverse = np.clip(
        inverse,
        0.75,
        1.35,
    )

    weights = np.ones(
        NUM_CLASSES,
        dtype=np.float32,
    )

    weights[1:] = inverse.astype(
        np.float32
    )

    # Background is not a manual voxel class.
    # Keep its point/background regularization weak.
    weights[0] = 0.25

    return (
        counts,
        torch.tensor(
            weights,
            dtype=torch.float32,
            device=DEVICE,
        ),
    )


# ============================================================================
# POINT HANDLING
# ============================================================================

def normalize_points(points):

    if isinstance(
        points,
        pd.DataFrame,
    ):
        df = points.copy()
    else:
        df = pd.DataFrame(points)

    required = [
        "class_id",
        "class_name",
        "level",
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
            f"Loaded point data missing: {missing}"
        )

    return df


def load_case_compatible(
    study_id,
    series_id,
    point_df,
):
    """Load a Part 2.20B case while accepting its supported return shapes.

    Depending on the local Part 2.20B revision, load_case() may return:
        (image, points, geometry)
    or:
        (image, points, geometry, extra_metadata)

    The extra value is not required by Part 3.1.
    """

    loaded = part220b.load_case(
        study_id,
        series_id,
        point_df,
    )

    if not isinstance(loaded, (tuple, list)):
        raise TypeError(
            "part220b.load_case() returned "
            f"{type(loaded).__name__}; expected tuple/list."
        )

    if len(loaded) == 3:
        image, points, geometry = loaded

    elif len(loaded) == 4:
        image, points, geometry, _extra = loaded

    else:
        raise ValueError(
            "Unexpected part220b.load_case() return length: "
            f"{len(loaded)}"
        )

    return image, normalize_points(points), geometry


def point_index(
    point,
    shape,
):

    if len(shape) == 4:
        _, d, h, w = shape
    elif len(shape) == 3:
        d, h, w = shape
    else:
        raise ValueError(
            f"Expected 3D or channel-first 4D shape, got {shape}"
        )

    z = int(
        np.clip(
            round(float(point["z"])),
            0,
            d - 1,
        )
    )

    y = int(
        np.clip(
            round(float(point["y"])),
            0,
            h - 1,
        )
    )

    x = int(
        np.clip(
            round(float(point["x"])),
            0,
            w - 1,
        )
    )

    return z, y, x


# ============================================================================
# LOSS COMPONENTS
# ============================================================================

def point_cross_entropy(
    logits,
    points,
    class_weights,
):

    losses = []

    for _, p in points.iterrows():

        cid = int(p["class_id"])

        if cid < 1 or cid >= NUM_CLASSES:
            continue

        z, y, x = point_index(
            p,
            logits.shape[-3:],
        )

        point_logits = (
            logits[
                0,
                :,
                z,
                y,
                x,
            ]
            .unsqueeze(0)
        )

        target = torch.tensor(
            [cid],
            device=logits.device,
            dtype=torch.long,
        )

        losses.append(
            F.cross_entropy(
                point_logits,
                target,
                weight=class_weights,
            )
        )

    if not losses:
        return logits.sum() * 0.0

    return torch.stack(losses).mean()


def target_background_margin_loss(
    logits,
    points,
):

    losses = []

    for _, p in points.iterrows():

        cid = int(p["class_id"])

        if cid < 1 or cid >= NUM_CLASSES:
            continue

        z, y, x = point_index(
            p,
            logits.shape[-3:],
        )

        target_logit = logits[
            0,
            cid,
            z,
            y,
            x,
        ]

        background_logit = logits[
            0,
            0,
            z,
            y,
            x,
        ]

        margin = (
            target_logit
            - background_logit
        )

        losses.append(
            F.softplus(
                TARGET_BG_MARGIN
                - margin
            )
        )

    if not losses:
        return logits.sum() * 0.0

    return torch.stack(losses).mean()


def gaussian_local_probability_loss(
    logits,
    points,
):

    """
    Weak local supervision.

    The point remains the only explicit annotation.
    A Gaussian-weighted average of the predicted probability around
    that point is encouraged to support the annotated class.

    This is NOT a voxel-wise ground-truth mask.
    """

    probabilities = F.softmax(
        logits,
        dim=1,
    )[0]

    losses = []

    radius = LOCAL_RADIUS
    sigma2 = LOCAL_SIGMA ** 2

    for _, p in points.iterrows():

        cid = int(p["class_id"])

        if cid < 1 or cid >= NUM_CLASSES:
            continue

        z, y, x = point_index(
            p,
            probabilities.shape[-3:],
        )

        z0 = max(0, z - radius)
        z1 = min(
            probabilities.shape[-3],
            z + radius + 1,
        )

        y0 = max(0, y - radius)
        y1 = min(
            probabilities.shape[-2],
            y + radius + 1,
        )

        x0 = max(0, x - radius)
        x1 = min(
            probabilities.shape[-1],
            x + radius + 1,
        )

        patch = probabilities[
            :,
            z0:z1,
            y0:y1,
            x0:x1,
        ]

        zz = torch.arange(
            z0,
            z1,
            device=logits.device,
        ).view(-1, 1, 1)

        yy = torch.arange(
            y0,
            y1,
            device=logits.device,
        ).view(1, -1, 1)

        xx = torch.arange(
            x0,
            x1,
            device=logits.device,
        ).view(1, 1, -1)

        distances2 = (
            (zz - z) ** 2
            + (yy - y) ** 2
            + (xx - x) ** 2
        ).float()

        weights = torch.exp(
            -distances2
            /
            (2.0 * sigma2)
        )

        weights = weights / (
            weights.sum()
            .clamp_min(1e-8)
        )

        target_probability = (
            patch[cid]
            * weights
        ).sum()

        losses.append(
            -torch.log(
                target_probability
                .clamp_min(1e-6)
            )
        )

    if not losses:
        return logits.sum() * 0.0

    return torch.stack(losses).mean()


def sample_background(
    points,
    shape,
    count,
):

    d, h, w = shape

    blocked = set()

    for _, p in points.iterrows():

        z, y, x = point_index(
            p,
            shape,
        )

        r = BACKGROUND_EXCLUSION_RADIUS

        for dz in range(
            -r,
            r + 1,
        ):
            for dy in range(
                -r,
                r + 1,
            ):
                for dx in range(
                    -r,
                    r + 1,
                ):

                    zz = z + dz
                    yy = y + dy
                    xx = x + dx

                    if (
                        0 <= zz < d
                        and
                        0 <= yy < h
                        and
                        0 <= xx < w
                    ):
                        blocked.add(
                            (zz, yy, xx)
                        )

    samples = []

    max_attempts = count * 30

    for _ in range(max_attempts):

        if len(samples) >= count:
            break

        z = random.randrange(d)
        y = random.randrange(h)
        x = random.randrange(w)

        if (z, y, x) in blocked:
            continue

        samples.append(
            (z, y, x)
        )

    return samples


def background_loss(
    logits,
    points,
):

    samples = sample_background(
        points,
        logits.shape[-3:],
        BACKGROUND_SAMPLES,
    )

    if not samples:
        return logits.sum() * 0.0

    losses = []

    for z, y, x in samples:

        pixel_logits = (
            logits[
                0,
                :,
                z,
                y,
                x,
            ]
            .unsqueeze(0)
        )

        target = torch.zeros(
            1,
            device=logits.device,
            dtype=torch.long,
        )

        losses.append(
            F.cross_entropy(
                pixel_logits,
                target,
            )
        )

    return torch.stack(
        losses
    ).mean()


# ============================================================================
# MODEL
# ============================================================================

def build_model():

    model = part220b.build_model()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: {parameter_count:,}"
    )

    if parameter_count != 4_078_116:
        raise RuntimeError(
            f"Unexpected parameter count: "
            f"{parameter_count}"
        )

    return model


def load_checkpoint(
    model,
    checkpoint_path,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get(
                "state_dict",
                checkpoint,
            ),
        )

    else:

        state_dict = checkpoint

    clean_state = {}

    for key, value in state_dict.items():

        key = (
            key[7:]
            if key.startswith("module.")
            else key
        )

        clean_state[key] = value

    result = model.load_state_dict(
        clean_state,
        strict=False,
    )

    print(
        f"Initialization missing keys: "
        f"{len(result.missing_keys)}"
    )

    print(
        f"Initialization unexpected keys: "
        f"{len(result.unexpected_keys)}"
    )

    if result.missing_keys or result.unexpected_keys:

        raise RuntimeError(
            "Initialization checkpoint does not match "
            "the current Swin-UNETR architecture."
        )

    return model


# ============================================================================
# FORWARD
# ============================================================================

def forward_model(
    model,
    image,
):

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
        .to(
            DEVICE,
            non_blocking=True,
        )
    )

    with autocast(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
        ),
    ):

        logits = model(
            tensor
        )

        if isinstance(
            logits,
            (tuple, list),
        ):
            logits = logits[0]

    return logits


# ============================================================================
# TRAIN ONE CASE
# ============================================================================

def train_one_case(
    model,
    optimizer,
    scaler,
    image,
    points,
    class_weights,
):

    with autocast(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
        ),
    ):

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
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        logits = model(
            tensor
        )

        if isinstance(
            logits,
            (tuple, list),
        ):
            logits = logits[0]

        point_loss = (
            point_cross_entropy(
                logits,
                points,
                class_weights,
            )
        )

        margin_loss = (
            target_background_margin_loss(
                logits,
                points,
            )
        )

        local_loss = (
            gaussian_local_probability_loss(
                logits,
                points,
            )
        )

        bg_loss = (
            background_loss(
                logits,
                points,
            )
        )

        total = (
            POINT_CE_WEIGHT * point_loss
            +
            MARGIN_WEIGHT * margin_loss
            +
            LOCAL_WEIGHT * local_loss
            +
            BACKGROUND_WEIGHT * bg_loss
        )

    return (
        total,
        point_loss.detach(),
        margin_loss.detach(),
        local_loss.detach(),
        bg_loss.detach(),
    )


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def evaluate_cases(
    model,
    cases,
):

    model.eval()

    rows = []
    foreground_ratios = []
    processed = 0

    for (
        study_id,
        series_id,
        point_df,
    ) in cases:

        try:

            image, points, geometry = load_case_compatible(
                study_id,
                series_id,
                point_df,
            )

            logits = forward_model(
                model,
                image,
            )

            probabilities = F.softmax(
                logits,
                dim=1,
            )

            predictions = (
                probabilities.argmax(
                    dim=1
                )
            )

            foreground_ratios.append(
                float(
                    (predictions != 0)
                    .float()
                    .mean()
                    .item()
                )
            )

            probs = probabilities[
                0
            ].float().cpu()

            labels = predictions[
                0
            ].cpu()

            for _, p in points.iterrows():

                true_class = int(
                    p["class_id"]
                )

                z, y, x = point_index(
                    p,
                    probs.shape,
                )

                predicted_class = int(
                    labels[
                        z,
                        y,
                        x,
                    ].item()
                )

                true_probability = float(
                    probs[
                        true_class,
                        z,
                        y,
                        x,
                    ].item()
                )

                rows.append(
                    {
                        "study_id": study_id,
                        "series_id": series_id,
                        "level": p["level"],
                        "true_class": true_class,
                        "true_class_name":
                            CLASS_NAMES[
                                true_class
                            ],
                        "predicted_class":
                            predicted_class,
                        "predicted_class_name":
                            CLASS_NAMES[
                                predicted_class
                            ],
                        "correct": int(
                            predicted_class
                            == true_class
                        ),
                        "true_probability":
                            true_probability,
                        "hit_at_0_50": int(
                            true_probability
                            >= 0.50
                        ),
                        "z": z,
                        "y": y,
                        "x": x,
                    }
                )

            processed += 1

        except Exception as exc:

            print(
                f"\nValidation/test failure "
                f"{study_id}/{series_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    df = pd.DataFrame(rows)

    if len(df) == 0:

        raise RuntimeError(
            "No evaluation points were produced."
        )

    disease_accuracy = (
        df.groupby(
            "true_class"
        )["correct"]
        .mean()
    )

    summary = {
        "processed_cases": processed,
        "points": int(len(df)),
        "overall_accuracy": float(
            df["correct"].mean()
        ),
        "macro_disease_accuracy": float(
            disease_accuracy.mean()
        ),
        "mean_true_probability": float(
            df["true_probability"].mean()
        ),
        "hit_rate_0_50": float(
            (
                df["true_probability"]
                >= 0.50
            ).mean()
        ),
        "mean_foreground_ratio": float(
            np.mean(
                foreground_ratios
            )
        ),
    }

    disease_metrics = {}

    for cid in range(
        1,
        NUM_CLASSES,
    ):

        subset = df[
            df["true_class"] == cid
        ]

        if len(subset) == 0:
            continue

        disease_metrics[
            CLASS_NAMES[cid]
        ] = {
            "points": int(
                len(subset)
            ),
            "accuracy": float(
                subset["correct"].mean()
            ),
            "mean_true_probability":
                float(
                    subset[
                        "true_probability"
                    ].mean()
                ),
            "hit_rate_at_0_50":
                float(
                    subset[
                        "hit_at_0_50"
                    ].mean()
                ),
        }

    summary[
        "disease_metrics"
    ] = disease_metrics

    return summary, df


# ============================================================================
# MAIN
# ============================================================================

def main():

    seed_everything()

    print("=" * 80)
    print("PART 3.1")
    print("STRONG POINT-SUPERVISED SWIN-UNETR TRAINING")
    print("=" * 80)

    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA is required for this training experiment."
        )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"Model shape: {MODEL_SHAPE}"
    )

    manifest = part220b.load_manifest()

    print(
        f"Manifest rows: {len(manifest)}"
    )

    # ------------------------------------------------------------------------
    # Development and untouched test cohorts
    # ------------------------------------------------------------------------

    dev_series = build_dev_split(
        manifest
    )

    test_series = build_test_split(
        manifest,
        dev_series,
    )

    dev_manifest = manifest_for_series(
        manifest,
        dev_series,
    )

    test_manifest = manifest_for_series(
        manifest,
        test_series,
    )

    training_cases = build_training_cases(
        manifest,
        dev_series,
        test_series,
    )

    dev_cases = build_case_groups(
        manifest,
        dev_series,
    )

    test_cases = build_case_groups(
        manifest,
        test_series,
    )

    print(
        f"Training series: {len(training_cases)}"
    )

    print(
        f"Development cases: {len(dev_cases)}"
    )

    print(
        f"Untouched test cases: {len(test_cases)}"
    )

    training_studies = set(
        x[0] for x in training_cases
    )

    dev_studies = set(
        dev_series["study_id"]
        .astype(str)
    )

    test_studies = set(
        test_series["study_id"]
        .astype(str)
    )

    if (
        training_studies & dev_studies
        or
        training_studies & test_studies
        or
        dev_studies & test_studies
    ):
        raise RuntimeError(
            "Study leakage detected."
        )

    print(
        "Study-disjoint split: PASSED"
    )

    print(
        f"Training studies: {len(training_studies)}"
    )

    print(
        f"Development studies: {len(dev_studies)}"
    )

    print(
        f"Test studies: {len(test_studies)}"
    )

    # ------------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------------

    train_manifest = manifest_for_series(
        manifest,
        pd.DataFrame(
            [
                {
                    "study_id": s,
                    "series_id": se,
                }
                for s, se, _ in training_cases
            ]
        ),
    )

    counts, class_weights = (
        compute_class_weights(
            train_manifest
        )
    )

    print()
    print("Class point counts / weights:")

    for cid in range(
        NUM_CLASSES
    ):

        print(
            f"  {cid} "
            f"{CLASS_NAMES[cid]}: "
            f"count={int(counts[cid])} "
            f"weight={class_weights[cid].item():.6f}"
        )

    sampling_weights = (
        build_sampling_weights(
            training_cases
        )
    )

    # ------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------

    init_checkpoint = (
        choose_initialization_checkpoint()
    )

    print()
    print(
        "Initialization checkpoint:"
    )
    print(init_checkpoint)

    model = build_model()

    model = load_checkpoint(
        model,
        init_checkpoint,
    )

    model = model.to(
        DEVICE
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = GradScaler(
        "cuda",
        enabled=True,
    )

    # ------------------------------------------------------------------------
    # Save configuration
    # ------------------------------------------------------------------------

    config = {
        "part": "3.1",
        "seed": SEED,
        "model_shape": list(
            MODEL_SHAPE
        ),
        "num_classes": NUM_CLASSES,
        "epochs": EPOCHS,
        "train_slots_per_epoch":
            TRAIN_SLOTS_PER_EPOCH,
        "gradient_accumulation":
            GRAD_ACCUMULATION,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "point_ce_weight":
            POINT_CE_WEIGHT,
        "margin_weight":
            MARGIN_WEIGHT,
        "local_weight":
            LOCAL_WEIGHT,
        "background_weight":
            BACKGROUND_WEIGHT,
        "target_background_margin":
            TARGET_BG_MARGIN,
        "local_radius":
            LOCAL_RADIUS,
        "local_sigma":
            LOCAL_SIGMA,
        "background_samples":
            BACKGROUND_SAMPLES,
        "background_exclusion_radius":
            BACKGROUND_EXCLUSION_RADIUS,
        "rfnn_case_weight":
            RFNN_CASE_WEIGHT,
        "rss_case_weight":
            RSS_CASE_WEIGHT,
        "development_cases":
            len(dev_cases),
        "test_cases":
            len(test_cases),
        "training_series":
            len(training_cases),
        "initialization_checkpoint":
            str(init_checkpoint),
        "manual_voxel_ground_truth":
            False,
        "dashboard_modified":
            False,
    }

    (
        REPORT_DIR
        / "part31_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    history = []

    best_macro = -float(
        "inf"
    )

    best_epoch = None

    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        losses = []
        point_losses = []
        margin_losses = []
        local_losses = []
        bg_losses = []

        used_cases = set()

        for slot in range(
            TRAIN_SLOTS_PER_EPOCH
        ):

            (
                study_id,
                series_id,
                point_df,
            ) = choose_training_case(
                training_cases,
                sampling_weights,
            )

            used_cases.add(
                (
                    study_id,
                    series_id,
                )
            )

            print(
                f"\rEpoch {epoch}/{EPOCHS} "
                f"case {slot + 1}/"
                f"{TRAIN_SLOTS_PER_EPOCH} "
                f"study={study_id} "
                f"series={series_id}",
                end="",
                flush=True,
            )

            try:

                image, points, geometry = load_case_compatible(
                    study_id,
                    series_id,
                    point_df,
                )

                (
                    total,
                    point_loss,
                    margin_loss,
                    local_loss,
                    bg_loss,
                ) = train_one_case(
                    model,
                    optimizer,
                    scaler,
                    image,
                    points,
                    class_weights,
                )

                scaled = (
                    total
                    /
                    GRAD_ACCUMULATION
                )

                scaler.scale(
                    scaled
                ).backward()

                if (
                    (slot + 1)
                    % GRAD_ACCUMULATION
                    == 0
                    or
                    slot
                    ==
                    TRAIN_SLOTS_PER_EPOCH - 1
                ):

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                losses.append(
                    float(
                        total.detach()
                        .cpu()
                    )
                )

                point_losses.append(
                    float(
                        point_loss.cpu()
                    )
                )

                margin_losses.append(
                    float(
                        margin_loss.cpu()
                    )
                )

                local_losses.append(
                    float(
                        local_loss.cpu()
                    )
                )

                bg_losses.append(
                    float(
                        bg_loss.cpu()
                    )
                )

            except Exception as exc:

                print(
                    f"\nSkipping "
                    f"{study_id}/{series_id}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        print()

        scheduler.step()

        # --------------------------------------------------------------------
        # Development validation
        # --------------------------------------------------------------------

        dev_summary, dev_df = (
            evaluate_cases(
                model,
                dev_cases,
            )
        )

        record = {
            "epoch": epoch,
            "train_loss":
                float(np.mean(losses)),
            "point_loss":
                float(np.mean(point_losses)),
            "margin_loss":
                float(np.mean(margin_losses)),
            "local_loss":
                float(np.mean(local_losses)),
            "background_loss":
                float(np.mean(bg_losses)),
            "unique_training_cases":
                len(used_cases),
            "learning_rate":
                float(
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
            "development_summary":
                dev_summary,
        }

        history.append(
            record
        )

        dev_df.to_csv(
            METRICS_DIR
            / f"part31_epoch_{epoch:02d}_development_points.csv",
            index=False,
        )

        torch.save(
            {
                "part": "3.1",
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "scheduler_state_dict":
                    scheduler.state_dict(),
                "development_metrics":
                    dev_summary,
                "initialization_checkpoint":
                    str(init_checkpoint),
                "config": config,
            },
            CHECKPOINT_DIR
            / f"part31_epoch_{epoch:02d}.pth",
        )

        macro = float(
            dev_summary[
                "macro_disease_accuracy"
            ]
        )

        if macro > best_macro:

            best_macro = macro
            best_epoch = epoch

            torch.save(
                {
                    "part": "3.1",
                    "best_epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "development_metrics":
                        dev_summary,
                    "initialization_checkpoint":
                        str(init_checkpoint),
                    "config": config,
                },
                CHECKPOINT_DIR
                / "part31_best_development_macro.pth",
            )

        print()
        print("=" * 80)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 80)

        print(
            f"Train loss: "
            f"{record['train_loss']:.6f}"
        )

        print(
            f"Point CE: "
            f"{record['point_loss']:.6f}"
        )

        print(
            f"Margin: "
            f"{record['margin_loss']:.6f}"
        )

        print(
            f"Local: "
            f"{record['local_loss']:.6f}"
        )

        print(
            f"Background: "
            f"{record['background_loss']:.6f}"
        )

        print(
            f"Development overall: "
            f"{dev_summary['overall_accuracy']:.6f}"
        )

        print(
            f"Development macro disease: "
            f"{dev_summary['macro_disease_accuracy']:.6f}"
        )

        print(
            f"Development mean probability: "
            f"{dev_summary['mean_true_probability']:.6f}"
        )

        print(
            f"Development hit @0.50: "
            f"{dev_summary['hit_rate_0_50']:.6f}"
        )

        print(
            f"Development foreground ratio: "
            f"{dev_summary['mean_foreground_ratio']:.6f}"
        )

        for name, metrics in (
            dev_summary[
                "disease_metrics"
            ].items()
        ):

            print(
                f"  {name}: "
                f"{metrics['accuracy']:.6f}"
            )

    # ------------------------------------------------------------------------
    # Final untouched test evaluation
    # ------------------------------------------------------------------------

    best_path = (
        CHECKPOINT_DIR
        / "part31_best_development_macro.pth"
    )

    checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(
        DEVICE
    )

    test_summary, test_df = (
        evaluate_cases(
            model,
            test_cases,
        )
    )

    test_df.to_csv(
        METRICS_DIR
        / "part31_final_untouched_test_points.csv",
        index=False,
    )

    final_summary = {
        "part": "3.1",
        "status": "COMPLETE",
        "best_development_epoch":
            best_epoch,
        "best_development_macro":
            best_macro,
        "development_cases":
            len(dev_cases),
        "untouched_test_cases":
            len(test_cases),
        "training_series":
            len(training_cases),
        "test_summary":
            test_summary,
        "history":
            history,
        "initialization_checkpoint":
            str(init_checkpoint),
        "training_performed":
            True,
        "manual_voxel_ground_truth":
            False,
        "dashboard_modified":
            False,
        "existing_checkpoints_modified":
            False,
        "scientific_note":
            (
                "The RSNA annotations are point/localization labels. "
                "The model is trained with weak point-derived spatial "
                "objectives; this does not establish manual voxel-wise "
                "segmentation ground truth."
            ),
    }

    (
        REPORT_DIR
        / "part31_summary.json"
    ).write_text(
        json.dumps(
            final_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "epoch": h["epoch"],
                "train_loss": h["train_loss"],
                "point_loss": h["point_loss"],
                "margin_loss": h["margin_loss"],
                "local_loss": h["local_loss"],
                "background_loss":
                    h["background_loss"],
                "development_overall":
                    h[
                        "development_summary"
                    ][
                        "overall_accuracy"
                    ],
                "development_macro":
                    h[
                        "development_summary"
                    ][
                        "macro_disease_accuracy"
                    ],
                "development_probability":
                    h[
                        "development_summary"
                    ][
                        "mean_true_probability"
                    ],
                "development_hit":
                    h[
                        "development_summary"
                    ][
                        "hit_rate_0_50"
                    ],
                "development_foreground":
                    h[
                        "development_summary"
                    ][
                        "mean_foreground_ratio"
                    ],
            }
            for h in history
        ]
    ).to_csv(
        METRICS_DIR
        / "part31_training_history.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("PART 3.1 COMPLETE")
    print("=" * 80)

    print(
        f"Best development epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best development macro: "
        f"{best_macro:.6f}"
    )

    print()
    print("UNTOUCHED TEST RESULTS")

    print(
        f"Test cases: "
        f"{test_summary['processed_cases']}"
    )

    print(
        f"Test points: "
        f"{test_summary['points']}"
    )

    print(
        f"Test overall: "
        f"{test_summary['overall_accuracy']:.6f}"
    )

    print(
        f"Test macro disease: "
        f"{test_summary['macro_disease_accuracy']:.6f}"
    )

    print(
        f"Test mean true probability: "
        f"{test_summary['mean_true_probability']:.6f}"
    )

    print(
        f"Test hit @0.50: "
        f"{test_summary['hit_rate_0_50']:.6f}"
    )

    print(
        f"Test foreground ratio: "
        f"{test_summary['mean_foreground_ratio']:.6f}"
    )

    for name, metrics in (
        test_summary[
            "disease_metrics"
        ].items()
    ):

        print(
            f"  {name}: "
            f"{metrics['accuracy']:.6f}"
        )

    print()
    print(
        "Checkpoint:"
    )

    print(
        best_path
    )

    print()
    print(
        "Dashboard modified: NO"
    )

    print(
        "Existing checkpoints modified: NO"
    )

    print(
        "Manual voxel masks fabricated: NO"
    )

    print(
        "Untouched test cohort used only after "
        "development checkpoint selection: YES"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
