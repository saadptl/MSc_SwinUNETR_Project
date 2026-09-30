"""
PART 2.18
Left/Right Neural Foraminal Refinement Training

Purpose
-------
Refine the Part 2.16 point-supervised Swin-UNETR model with
specific emphasis on distinguishing:

    Class 2 = Left Neural Foraminal Narrowing
    Class 3 = Right Neural Foraminal Narrowing

The experiment:

    1. Starts from Part 2.16 Epoch-5 checkpoint.
    2. Keeps the original 6-class output contract.
    3. Uses point/localization supervision only.
    4. Uses disease-aware sampling.
    5. Adds an explicit left-vs-right margin loss.
    6. Keeps other diseases represented.
    7. Uses a fixed study-disjoint validation set.
    8. Does NOT fabricate voxel-wise masks.
    9. Does NOT modify Part104.
   10. Does NOT modify the dashboard.

Scientific limitation
---------------------
RSNA coordinates are point/localization annotations, not
manual voxel-wise segmentation masks.

Therefore this experiment does NOT produce a clinical
voxel-wise Dice score.
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
from torch.amp import autocast, GradScaler

from monai.networks.nets import SwinUNETR


# ================================================================
# PROJECT PATHS
# ================================================================

ROOT = Path(__file__).resolve().parents[1]

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
    / "rsna_part216_point_supervised_balanced_training"
    / "checkpoints"
    / "part216_epoch_05.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part218_foraminal_refinement_training"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORT_DIR = OUTPUT_DIR / "reports"

for directory in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    METRICS_DIR,
    REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

MODEL_SHAPE = (
    64,
    96,
    96,
)

NUM_CLASSES = 6

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

# Important classes
LEFT_FORAMINAL = 2
RIGHT_FORAMINAL = 3

# Training
EPOCHS = 5

TRAINING_SLOTS_PER_EPOCH = 100

BATCH_SIZE = 1

GRAD_ACCUMULATION = 4

LEARNING_RATE = 2.5e-5

WEIGHT_DECAY = 1e-5

# Background points are deliberately limited.
BACKGROUND_SAMPLES_PER_CASE = 128

BACKGROUND_LOSS_WEIGHT = 0.10

# Additional emphasis on the two foraminal classes.
FORAMINAL_CLASS_WEIGHT = 1.75

# Margin loss:
# force true foraminal class probability/logit above
# the opposite side.
FORAMINAL_MARGIN = 0.50

FORAMINAL_MARGIN_WEIGHT = 0.75

# Fixed validation set size.
NUM_VALIDATION_SERIES = 25

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ================================================================
# IMPORT PART11 ROBUST DICOM LOADER
# ================================================================

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from segmentation_rsna_part11_controlled_pilot_training_corrected import (
    read_dicom_series_robust,
    resolve_series_dir,
)


# ================================================================
# MODEL
# ================================================================

def build_model():

    model = SwinUNETR(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=12,
        use_checkpoint=False,
    )

    return model


def clean_state_dict(
    state_dict,
):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        for prefix in [
            "module.",
            "model.",
            "network.",
        ]:

            if new_key.startswith(prefix):

                new_key = new_key[
                    len(prefix):
                ]

        cleaned[new_key] = value

    return cleaned


def load_initial_checkpoint(
    model,
):

    if not INIT_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Part 2.16 checkpoint not found:\n"
            f"{INIT_CHECKPOINT}"
        )

    checkpoint = torch.load(
        INIT_CHECKPOINT,
        map_location="cpu",
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "model" in checkpoint:

            state_dict = checkpoint[
                "model"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    state_dict = clean_state_dict(
        state_dict
    )

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print(
        f"Initialization checkpoint:"
    )

    print(
        INIT_CHECKPOINT
    )

    print(
        f"Missing keys: {len(missing)}"
    )

    print(
        f"Unexpected keys: {len(unexpected)}"
    )

    if missing:

        print(
            "WARNING: missing model keys detected."
        )

    if unexpected:

        print(
            "WARNING: unexpected model keys detected."
        )

    return model


# ================================================================
# DATASET / MANIFEST
# ================================================================

def load_manifest():

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    df = pd.read_csv(
        MANIFEST_PATH
    )

    df["study_id"] = (
        df["study_id"]
        .astype(str)
    )

    df["series_id"] = (
        df["series_id"]
        .astype(str)
    )

    df["class_id"] = (
        df["class_id"]
        .astype(int)
    )

    df["model_z_float"] = (
        df["model_z_float"]
        .astype(float)
    )

    df["model_y_float"] = (
        df["model_y_float"]
        .astype(float)
    )

    df["model_x_float"] = (
        df["model_x_float"]
        .astype(float)
    )

    return df


# ================================================================
# STUDY-DISJOINT SPLIT
# ================================================================

def create_split(
    manifest_df,
):

    studies = sorted(
        manifest_df[
            "study_id"
        ].unique()
    )

    rng = random.Random(
        SEED
    )

    studies = list(
        studies
    )

    rng.shuffle(
        studies
    )

    split_index = int(
        len(studies) * 0.80
    )

    train_studies = set(
        studies[:split_index]
    )

    validation_studies = set(
        studies[split_index:]
    )

    train_df = manifest_df[
        manifest_df[
            "study_id"
        ].isin(
            train_studies
        )
    ].copy()

    validation_df = manifest_df[
        manifest_df[
            "study_id"
        ].isin(
            validation_studies
        )
    ].copy()

    validation_series = (
        validation_df[
            [
                "study_id",
                "series_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "study_id",
                "series_id",
            ]
        )
        .head(
            NUM_VALIDATION_SERIES
        )
    )

    overlap = (
        set(
            validation_series[
                "study_id"
            ]
        )
        &
        train_studies
    )

    print(
        f"Training studies: "
        f"{len(train_studies)}"
    )

    print(
        f"Validation studies: "
        f"{len(validation_studies)}"
    )

    print(
        f"Validation series: "
        f"{len(validation_series)}"
    )

    print(
        f"Study overlap: "
        f"{len(overlap)}"
    )

    if overlap:

        raise RuntimeError(
            "Study leakage detected."
        )

    return (
        train_df,
        validation_df,
        train_studies,
        validation_studies,
        validation_series,
    )


# ================================================================
# SERIES TABLE
# ================================================================

def make_series_table(
    df,
):

    rows = []

    grouped = df.groupby(
        [
            "study_id",
            "series_id",
        ]
    )

    for (
        study_id,
        series_id,
    ), group in grouped:

        class_ids = set(
            group[
                "class_id"
            ].astype(int)
        )

        rows.append(
            {
                "study_id":
                    str(study_id),

                "series_id":
                    str(series_id),

                "classes":
                    sorted(
                        class_ids
                    ),

                "has_left_foraminal":
                    LEFT_FORAMINAL
                    in class_ids,

                "has_right_foraminal":
                    RIGHT_FORAMINAL
                    in class_ids,

                "has_foraminal":
                    (
                        LEFT_FORAMINAL
                        in class_ids
                    )
                    or
                    (
                        RIGHT_FORAMINAL
                        in class_ids
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ================================================================
# DICOM LOADING
# ================================================================

def normalize_volume(
    volume,
):

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    finite = np.isfinite(
        volume
    )

    if not finite.any():

        return np.zeros_like(
            volume,
            dtype=np.float32,
        )

    values = volume[
        finite
    ]

    p1 = np.percentile(
        values,
        1,
    )

    p99 = np.percentile(
        values,
        99,
    )

    if p99 <= p1:

        return np.zeros_like(
            volume,
            dtype=np.float32,
        )

    volume = np.clip(
        volume,
        p1,
        p99,
    )

    volume = (
        volume - p1
    ) / (
        p99 - p1
    )

    return volume.astype(
        np.float32
    )


def resize_volume(
    volume,
    target_shape,
):

    from scipy.ndimage import zoom

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    factors = [
        target_shape[i]
        /
        volume.shape[i]
        for i in range(3)
    ]

    resized = zoom(
        volume,
        factors,
        order=1,
    )

    # Defensive shape correction.
    output = np.zeros(
        target_shape,
        dtype=np.float32,
    )

    z = min(
        target_shape[0],
        resized.shape[0],
    )

    y = min(
        target_shape[1],
        resized.shape[1],
    )

    x = min(
        target_shape[2],
        resized.shape[2],
    )

    output[
        :z,
        :y,
        :x,
    ] = resized[
        :z,
        :y,
        :x,
    ]

    return output


def load_series(
    study_id,
    series_id,
):

    row = pd.Series(
        {
            "study_id":
                study_id,

            "series_id":
                series_id,
        }
    )

    series_dir = (
        resolve_series_dir(
            row
        )
    )

    image_native, records, info = (
        read_dicom_series_robust(
            series_dir
        )
    )

    image_native = normalize_volume(
        image_native
    )

    image_model = resize_volume(
        image_native,
        MODEL_SHAPE,
    )

    tensor = torch.from_numpy(
        image_model
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    return (
        tensor,
        image_native,
        records,
        info,
    )


# ================================================================
# POINT EXTRACTION
# ================================================================

def extract_points(
    case_df,
):

    points = []

    for _, row in case_df.iterrows():

        z = float(
            np.clip(
                row[
                    "model_z_float"
                ],
                0,
                MODEL_SHAPE[0] - 1,
            )
        )

        y = float(
            np.clip(
                row[
                    "model_y_float"
                ],
                0,
                MODEL_SHAPE[1] - 1,
            )
        )

        x = float(
            np.clip(
                row[
                    "model_x_float"
                ],
                0,
                MODEL_SHAPE[2] - 1,
            )
        )

        points.append(
            {
                "class_id":
                    int(
                        row[
                            "class_id"
                        ]
                    ),

                "z": z,
                "y": y,
                "x": x,

                "level":
                    str(
                        row[
                            "level"
                        ]
                    ),
            }
        )

    return points


# ================================================================
# DISEASE-AWARE TRAINING CASE SAMPLER
# ================================================================

class ForaminalCaseSampler:

    def __init__(
        self,
        series_df,
        seed=42,
    ):

        self.series_df = (
            series_df
            .reset_index(
                drop=True
            )
        )

        self.rng = random.Random(
            seed
        )

        self.foraminal = (
            self.series_df[
                self.series_df[
                    "has_foraminal"
                ]
            ]
        )

        self.left = (
            self.series_df[
                self.series_df[
                    "has_left_foraminal"
                ]
            ]
        )

        self.right = (
            self.series_df[
                self.series_df[
                    "has_right_foraminal"
                ]
            ]
        )

        self.other = (
            self.series_df[
                ~self.series_df[
                    "has_foraminal"
                ]
            ]
        )

    def sample(
        self,
        n,
    ):

        selected = []

        for _ in range(n):

            r = self.rng.random()

            # 60% specifically target foraminal cases.
            if r < 0.60:

                pool = (
                    self.foraminal
                )

            # 20% non-foraminal cases.
            elif r < 0.80:

                pool = (
                    self.other
                )

            # 20% unrestricted training cases.
            else:

                pool = (
                    self.series_df
                )

            if len(pool) == 0:

                pool = (
                    self.series_df
                )

            index = self.rng.randrange(
                len(pool)
            )

            selected.append(
                pool.iloc[index]
            )

        return selected


# ================================================================
# POINT LOGIT EXTRACTION
# ================================================================

def sample_logits_at_points(
    logits,
    points,
):

    """
    logits:
        [1, C, Z, Y, X]

    Returns:
        [N, C]
    """

    values = []

    for point in points:

        z = int(
            np.clip(
                round(
                    point["z"]
                ),
                0,
                MODEL_SHAPE[0] - 1,
            )
        )

        y = int(
            np.clip(
                round(
                    point["y"]
                ),
                0,
                MODEL_SHAPE[1] - 1,
            )
        )

        x = int(
            np.clip(
                round(
                    point["x"]
                ),
                0,
                MODEL_SHAPE[2] - 1,
            )
        )

        values.append(
            logits[
                0,
                :,
                z,
                y,
                x,
            ]
        )

    if not values:

        return torch.empty(
            (
                0,
                NUM_CLASSES,
            ),
            device=logits.device,
        )

    return torch.stack(
        values
    )


# ================================================================
# BACKGROUND POINTS
# ================================================================

def generate_background_points(
    probability=None,
    count=128,
):

    """
    Generate deterministic random background coordinates.

    These are not ground-truth segmentation labels.

    Background samples simply prevent the point-supervised model
    from assigning every location to a disease.
    """

    points = []

    for _ in range(count):

        points.append(
            {
                "class_id": 0,

                "z": random.uniform(
                    0,
                    MODEL_SHAPE[0] - 1,
                ),

                "y": random.uniform(
                    0,
                    MODEL_SHAPE[1] - 1,
                ),

                "x": random.uniform(
                    0,
                    MODEL_SHAPE[2] - 1,
                ),

                "level": "BACKGROUND",
            }
        )

    return points


# ================================================================
# LOSS
# ================================================================

def calculate_point_loss(
    logits,
    points,
):

    if len(points) == 0:

        return (
            logits.sum() * 0.0,
            {
                "ce": 0.0,
                "margin": 0.0,
                "points": 0,
                "foraminal_points": 0,
            },
        )

    point_logits = (
        sample_logits_at_points(
            logits,
            points,
        )
    )

    targets = torch.tensor(
        [
            p["class_id"]
            for p in points
        ],
        dtype=torch.long,
        device=logits.device,
    )

    # ------------------------------------------------------------
    # CLASS WEIGHTS
    # ------------------------------------------------------------

    class_weights = torch.ones(
        NUM_CLASSES,
        device=logits.device,
    )

    # Background deliberately low.
    class_weights[
        0
    ] = BACKGROUND_LOSS_WEIGHT

    # Foraminal classes receive additional emphasis.
    class_weights[
        LEFT_FORAMINAL
    ] = FORAMINAL_CLASS_WEIGHT

    class_weights[
        RIGHT_FORAMINAL
    ] = FORAMINAL_CLASS_WEIGHT

    ce_loss = F.cross_entropy(
        point_logits,
        targets,
        weight=class_weights,
    )

    # ------------------------------------------------------------
    # LEFT/RIGHT FORAMINAL MARGIN LOSS
    # ------------------------------------------------------------

    margin_terms = []

    foraminal_count = 0

    for i, point in enumerate(
        points
    ):

        class_id = (
            point["class_id"]
        )

        if class_id == LEFT_FORAMINAL:

            true_logit = point_logits[
                i,
                LEFT_FORAMINAL,
            ]

            opposite_logit = point_logits[
                i,
                RIGHT_FORAMINAL,
            ]

            margin_terms.append(
                F.relu(
                    FORAMINAL_MARGIN
                    - true_logit
                    + opposite_logit
                )
            )

            foraminal_count += 1

        elif class_id == RIGHT_FORAMINAL:

            true_logit = point_logits[
                i,
                RIGHT_FORAMINAL,
            ]

            opposite_logit = point_logits[
                i,
                LEFT_FORAMINAL,
            ]

            margin_terms.append(
                F.relu(
                    FORAMINAL_MARGIN
                    - true_logit
                    + opposite_logit
                )
            )

            foraminal_count += 1

    if margin_terms:

        margin_loss = torch.stack(
            margin_terms
        ).mean()

    else:

        margin_loss = (
            logits.sum() * 0.0
        )

    total_loss = (
        ce_loss
        +
        FORAMINAL_MARGIN_WEIGHT
        *
        margin_loss
    )

    return (
        total_loss,
        {
            "ce":
                float(
                    ce_loss.detach()
                    .cpu()
                    .item()
                ),

            "margin":
                float(
                    margin_loss.detach()
                    .cpu()
                    .item()
                ),

            "points":
                len(points),

            "foraminal_points":
                foraminal_count,
        },
    )


# ================================================================
# TRAINING CASE
# ================================================================

def train_one_case(
    model,
    optimizer,
    scaler,
    case_df,
):

    study_id = str(
        case_df.iloc[0][
            "study_id"
        ]
    )

    series_id = str(
        case_df.iloc[0][
            "series_id"
        ]
    )

    image_tensor, _, _, _ = (
        load_series(
            study_id,
            series_id,
        )
    )

    points = extract_points(
        case_df
    )

    background_points = (
        generate_background_points(
            count=
                BACKGROUND_SAMPLES_PER_CASE
        )
    )

    points = (
        points
        +
        background_points
    )

    image_tensor = (
        image_tensor.to(
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
            image_tensor
        )

        loss, stats = (
            calculate_point_loss(
                logits,
                points,
            )
        )

    return (
        loss,
        stats,
    )


# ================================================================
# VALIDATION
# ================================================================

@torch.no_grad()
def validate(
    model,
    validation_series,
    manifest_df,
):

    model.eval()

    all_results = []

    for _, series_row in (
        validation_series.iterrows()
    ):

        study_id = str(
            series_row[
                "study_id"
            ]
        )

        series_id = str(
            series_row[
                "series_id"
            ]
        )

        case_df = manifest_df[
            (
                manifest_df[
                    "study_id"
                ]
                ==
                study_id
            )
            &
            (
                manifest_df[
                    "series_id"
                ]
                ==
                series_id
            )
        ]

        if case_df.empty:

            continue

        image_tensor, _, _, _ = (
            load_series(
                study_id,
                series_id,
            )
        )

        points = extract_points(
            case_df
        )

        image_tensor = (
            image_tensor.to(
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
                image_tensor
            )

        point_logits = (
            sample_logits_at_points(
                logits,
                points,
            )
        )

        probabilities = (
            torch.softmax(
                point_logits,
                dim=1,
            )
        )

        predictions = (
            torch.argmax(
                probabilities,
                dim=1,
            )
        )

        for i, point in enumerate(
            points
        ):

            true_class = (
                point["class_id"]
            )

            predicted_class = int(
                predictions[
                    i
                ].item()
            )

            true_probability = float(
                probabilities[
                    i,
                    true_class,
                ].item()
            )

            # Probability of the opposite foraminal side.
            opposite_probability = None

            if (
                true_class
                ==
                LEFT_FORAMINAL
            ):

                opposite_probability = float(
                    probabilities[
                        i,
                        RIGHT_FORAMINAL,
                    ].item()
                )

            elif (
                true_class
                ==
                RIGHT_FORAMINAL
            ):

                opposite_probability = float(
                    probabilities[
                        i,
                        LEFT_FORAMINAL,
                    ].item()
                )

            all_results.append(
                {
                    "study_id":
                        study_id,

                    "series_id":
                        series_id,

                    "level":
                        point["level"],

                    "true_class_id":
                        true_class,

                    "true_class_name":
                        CLASS_NAMES[
                            true_class
                        ],

                    "predicted_class_id":
                        predicted_class,

                    "predicted_class_name":
                        CLASS_NAMES[
                            predicted_class
                        ],

                    "correct":
                        int(
                            predicted_class
                            ==
                            true_class
                        ),

                    "true_probability":
                        true_probability,

                    "opposite_foraminal_probability":
                        opposite_probability,

                    "hit_at_0_50":
                        int(
                            true_probability
                            >=
                            0.50
                        ),
                }
            )

    return pd.DataFrame(
        all_results
    )


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    results_df,
):

    if results_df.empty:

        return {
            "point_accuracy": 0.0,
            "macro_disease_accuracy": 0.0,
            "mean_true_probability": 0.0,
            "hit_rate": 0.0,
            "foraminal_accuracy": 0.0,
            "left_foraminal_accuracy": 0.0,
            "right_foraminal_accuracy": 0.0,
        }

    disease_df = (
        results_df[
            results_df[
                "true_class_id"
            ].isin(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                ]
            )
        ]
        .groupby(
            [
                "true_class_id",
                "true_class_name",
            ]
        )
        .agg(
            points=(
                "correct",
                "size",
            ),
            accuracy=(
                "correct",
                "mean",
            ),
            mean_probability=(
                "true_probability",
                "mean",
            ),
            hit_rate=(
                "hit_at_0_50",
                "mean",
            ),
        )
        .reset_index()
    )

    disease_accuracy = (
        float(
            disease_df[
                "accuracy"
            ].mean()
        )
        if not disease_df.empty
        else 0.0
    )

    left_df = results_df[
        results_df[
            "true_class_id"
        ]
        ==
        LEFT_FORAMINAL
    ]

    right_df = results_df[
        results_df[
            "true_class_id"
        ]
        ==
        RIGHT_FORAMINAL
    ]

    foraminal_df = results_df[
        results_df[
            "true_class_id"
        ].isin(
            [
                LEFT_FORAMINAL,
                RIGHT_FORAMINAL,
            ]
        )
    ]

    metrics = {
        "point_accuracy":
            float(
                results_df[
                    "correct"
                ].mean()
            ),

        "macro_disease_accuracy":
            disease_accuracy,

        "mean_true_probability":
            float(
                results_df[
                    "true_probability"
                ].mean()
            ),

        "hit_rate":
            float(
                results_df[
                    "hit_at_0_50"
                ].mean()
            ),

        "foraminal_accuracy":
            float(
                foraminal_df[
                    "correct"
                ].mean()
            )
            if not foraminal_df.empty
            else 0.0,

        "left_foraminal_accuracy":
            float(
                left_df[
                    "correct"
                ].mean()
            )
            if not left_df.empty
            else 0.0,

        "right_foraminal_accuracy":
            float(
                right_df[
                    "correct"
                ].mean()
            )
            if not right_df.empty
            else 0.0,

        "left_to_right_errors":
            int(
                (
                    (
                        results_df[
                            "true_class_id"
                        ]
                        ==
                        LEFT_FORAMINAL
                    )
                    &
                    (
                        results_df[
                            "predicted_class_id"
                        ]
                        ==
                        RIGHT_FORAMINAL
                    )
                ).sum()
            ),

        "right_to_left_errors":
            int(
                (
                    (
                        results_df[
                            "true_class_id"
                        ]
                        ==
                        RIGHT_FORAMINAL
                    )
                    &
                    (
                        results_df[
                            "predicted_class_id"
                        ]
                        ==
                        LEFT_FORAMINAL
                    )
                ).sum()
            ),
    }

    return metrics


# ================================================================
# CHECKPOINT
# ================================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
):

    path = (
        CHECKPOINT_DIR
        /
        f"part218_epoch_{epoch:02d}.pth"
    )

    torch.save(
        {
            "part":
                "2.18",

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "metrics":
                metrics,

            "config":
                {
                    "seed":
                        SEED,

                    "model_shape":
                        MODEL_SHAPE,

                    "num_classes":
                        NUM_CLASSES,

                    "learning_rate":
                        LEARNING_RATE,

                    "foraminal_class_weight":
                        FORAMINAL_CLASS_WEIGHT,

                    "foraminal_margin":
                        FORAMINAL_MARGIN,

                    "foraminal_margin_weight":
                        FORAMINAL_MARGIN_WEIGHT,
                },
        },
        path,
    )

    return path


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 80)
    print(
        "PART 2.18"
    )
    print(
        "LEFT/RIGHT NEURAL FORAMINAL REFINEMENT TRAINING"
    )
    print("=" * 80)

    print()

    print(
        f"Project root:\n{ROOT}"
    )

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()

    # ------------------------------------------------------------
    # MANIFEST
    # ------------------------------------------------------------

    manifest_df = (
        load_manifest()
    )

    print(
        f"Manifest rows: "
        f"{len(manifest_df)}"
    )

    print(
        f"Annotated series: "
        f"{manifest_df[['study_id','series_id']].drop_duplicates().shape[0]}"
    )

    # ------------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------------

    (
        train_df,
        validation_df,
        train_studies,
        validation_studies,
        validation_series,
    ) = create_split(
        manifest_df
    )

    # ------------------------------------------------------------
    # TRAINING SERIES
    # ------------------------------------------------------------

    train_series_df = (
        make_series_table(
            train_df
        )
    )

    print()

    print(
        f"Training candidate series: "
        f"{len(train_series_df)}"
    )

    print(
        f"Training series with foraminal annotations: "
        f"{train_series_df['has_foraminal'].sum()}"
    )

    print(
        f"Training series with left foraminal annotations: "
        f"{train_series_df['has_left_foraminal'].sum()}"
    )

    print(
        f"Training series with right foraminal annotations: "
        f"{train_series_df['has_right_foraminal'].sum()}"
    )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = build_model()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    model = (
        load_initial_checkpoint(
            model
        )
    )

    model.to(
        DEVICE
    )

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = GradScaler(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
        ),
    )

    # ------------------------------------------------------------
    # SAMPLER
    # ------------------------------------------------------------

    sampler = (
        ForaminalCaseSampler(
            train_series_df,
            seed=SEED,
        )
    )

    print()

    print(
        "Training strategy:"
    )

    print(
        "  60% foraminal-focused cases"
    )

    print(
        "  20% non-foraminal cases"
    )

    print(
        "  20% unrestricted training cases"
    )

    print(
        f"  {FORAMINAL_CLASS_WEIGHT}x "
        "foraminal CE weight"
    )

    print(
        f"  {FORAMINAL_MARGIN_WEIGHT}x "
        "left/right margin loss"
    )

    print(
        f"  Margin = "
        f"{FORAMINAL_MARGIN}"
    )

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    history = []

    best_metric = -math.inf

    best_checkpoint = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print(
            "=" * 80
        )

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print(
            "=" * 80
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        selected_cases = sampler.sample(
            TRAINING_SLOTS_PER_EPOCH
        )

        losses = []

        ce_losses = []

        margin_losses = []

        unique_case_ids = set()

        foraminal_points_seen = 0

        for slot, case in enumerate(
            selected_cases,
            start=1,
        ):

            study_id = str(
                case[
                    "study_id"
                ]
            )

            series_id = str(
                case[
                    "series_id"
                ]
            )

            unique_case_ids.add(
                (
                    study_id,
                    series_id,
                )
            )

            case_df = train_df[
                (
                    train_df[
                        "study_id"
                    ]
                    ==
                    study_id
                )
                &
                (
                    train_df[
                        "series_id"
                    ]
                    ==
                    series_id
                )
            ]

            try:

                loss, stats = (
                    train_one_case(
                        model,
                        optimizer,
                        scaler,
                        case_df,
                    )
                )

                scaled_loss = (
                    loss
                    /
                    GRAD_ACCUMULATION
                )

                scaler.scale(
                    scaled_loss
                ).backward()

                if (
                    slot
                    %
                    GRAD_ACCUMULATION
                    == 0
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
                        loss.detach()
                        .cpu()
                        .item()
                    )
                )

                ce_losses.append(
                    stats["ce"]
                )

                margin_losses.append(
                    stats["margin"]
                )

                foraminal_points_seen += (
                    stats[
                        "foraminal_points"
                    ]
                )

            except Exception as exc:

                print()

                print(
                    f"Training case failed: "
                    f"Study={study_id} "
                    f"Series={series_id}"
                )

                print(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        # --------------------------------------------------------
        # FINAL ACCUMULATION STEP
        # --------------------------------------------------------

        if (
            len(selected_cases)
            %
            GRAD_ACCUMULATION
            != 0
        ):

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        scheduler.step()

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        validation_results = validate(
            model,
            validation_series,
            manifest_df,
        )

        metrics = calculate_metrics(
            validation_results
        )

        mean_loss = (
            float(
                np.mean(
                    losses
                )
            )
            if losses
            else float("nan")
        )

        mean_ce = (
            float(
                np.mean(
                    ce_losses
                )
            )
            if ce_losses
            else float("nan")
        )

        mean_margin = (
            float(
                np.mean(
                    margin_losses
                )
            )
            if margin_losses
            else float("nan")
        )

        current_lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        epoch_record = {
            "epoch":
                epoch,

            "mean_train_loss":
                mean_loss,

            "mean_ce_loss":
                mean_ce,

            "mean_margin_loss":
                mean_margin,

            "learning_rate":
                current_lr,

            "unique_training_cases":
                len(
                    unique_case_ids
                ),

            "training_slots":
                TRAINING_SLOTS_PER_EPOCH,

            "foraminal_points_seen":
                foraminal_points_seen,

            "validation_points":
                len(
                    validation_results
                ),

            **metrics,
        }

        history.append(
            epoch_record
        )

        # --------------------------------------------------------
        # SAVE METRICS
        # --------------------------------------------------------

        validation_results.to_csv(
            METRICS_DIR
            /
            f"part218_epoch_{epoch:02d}_validation_points.csv",
            index=False,
        )

        with open(
            METRICS_DIR
            /
            f"part218_epoch_{epoch:02d}_metrics.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                epoch_record,
                f,
                indent=2,
            )

        # --------------------------------------------------------
        # CHECKPOINT
        # --------------------------------------------------------

        checkpoint_path = (
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                metrics,
            )
        )

        # --------------------------------------------------------
        # TERMINAL
        # --------------------------------------------------------

        print()

        print(
            f"Unique cases: "
            f"{len(unique_case_ids)}/"
            f"{TRAINING_SLOTS_PER_EPOCH}"
        )

        print(
            f"Mean training loss: "
            f"{mean_loss:.6f}"
        )

        print(
            f"Mean CE loss: "
            f"{mean_ce:.6f}"
        )

        print(
            f"Mean foraminal margin loss: "
            f"{mean_margin:.6f}"
        )

        print(
            f"Validation points: "
            f"{len(validation_results)}"
        )

        print(
            f"Overall point accuracy: "
            f"{metrics['point_accuracy']:.6f}"
        )

        print(
            f"Macro disease accuracy: "
            f"{metrics['macro_disease_accuracy']:.6f}"
        )

        print(
            f"Foraminal accuracy: "
            f"{metrics['foraminal_accuracy']:.6f}"
        )

        print(
            f"Left foraminal accuracy: "
            f"{metrics['left_foraminal_accuracy']:.6f}"
        )

        print(
            f"Right foraminal accuracy: "
            f"{metrics['right_foraminal_accuracy']:.6f}"
        )

        print(
            f"Left -> Right errors: "
            f"{metrics['left_to_right_errors']}"
        )

        print(
            f"Right -> Left errors: "
            f"{metrics['right_to_left_errors']}"
        )

        print(
            f"Mean true probability: "
            f"{metrics['mean_true_probability']:.6f}"
        )

        print(
            f"Hit rate >= 0.50: "
            f"{metrics['hit_rate']:.6f}"
        )

        print(
            f"Checkpoint: "
            f"{checkpoint_path}"
        )

        # --------------------------------------------------------
        # BEST CHECKPOINT
        # --------------------------------------------------------

        # Primary criterion:
        # mean of left/right foraminal accuracy.
        foraminal_score = (
            metrics[
                "foraminal_accuracy"
            ]
        )

        if (
            foraminal_score
            >
            best_metric
        ):

            best_metric = (
                foraminal_score
            )

            best_checkpoint = (
                checkpoint_path
            )

            best_path = (
                CHECKPOINT_DIR
                /
                "part218_best_foraminal.pth"
            )

            torch.save(
                {
                    "part":
                        "2.18",

                    "best_epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "metrics":
                        metrics,

                    "source_initialization":
                        str(
                            INIT_CHECKPOINT
                        ),
                },
                best_path,
            )

            print(
                f"New best foraminal checkpoint: "
                f"{best_path}"
            )

    # ============================================================
    # FINAL HISTORY
    # ============================================================

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        OUTPUT_DIR
        /
        "part218_training_history.csv",
        index=False,
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    final_summary = {
        "part":
            "2.18",

        "status":
            "COMPLETE",

        "initialization_checkpoint":
            str(
                INIT_CHECKPOINT
            ),

        "best_checkpoint":
            str(
                best_checkpoint
            )
            if best_checkpoint
            else None,

        "best_foraminal_accuracy":
            best_metric,

        "epochs":
            EPOCHS,

        "training_slots_per_epoch":
            TRAINING_SLOTS_PER_EPOCH,

        "model_shape":
            list(
                MODEL_SHAPE
            ),

        "num_classes":
            NUM_CLASSES,

        "foraminal_class_weight":
            FORAMINAL_CLASS_WEIGHT,

        "foraminal_margin":
            FORAMINAL_MARGIN,

        "foraminal_margin_weight":
            FORAMINAL_MARGIN_WEIGHT,

        "dashboard_integrated":
            False,

        "manual_voxel_ground_truth":
            False,

        "voxelwise_dice_reported":
            False,

        "scientific_limitation":
            (
                "RSNA coordinates are point/localization "
                "annotations rather than manual voxel-wise "
                "segmentation masks."
            ),

        "history":
            history,
    }

    with open(
        OUTPUT_DIR
        /
        "part218_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_summary,
            f,
            indent=2,
        )

    # ============================================================
    # FINAL REPORT
    # ============================================================

    report_path = (
        REPORT_DIR
        /
        "part218_training_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PART 2.18\n"
        )

        f.write(
            "LEFT/RIGHT NEURAL FORAMINAL "
            "REFINEMENT TRAINING\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Initialization:\n"
            f"{INIT_CHECKPOINT}\n\n"
        )

        f.write(
            f"Best checkpoint:\n"
            f"{best_checkpoint}\n\n"
        )

        f.write(
            "TRAINING HISTORY\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            history_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "SCIENTIFIC LIMITATION\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            "RSNA provides point/localization "
            "annotations rather than manual "
            "voxel-wise segmentation masks. "
            "Therefore Part 2.18 does not "
            "report clinical voxel-wise Dice "
            "or claim accurate voxel-level "
            "segmentation.\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "Part104 was not modified.\n"
        )

        f.write(
            "Part216 was not overwritten.\n"
        )

        f.write(
            "The dashboard was not modified.\n"
        )

    # ============================================================
    # FINAL TERMINAL OUTPUT
    # ============================================================

    print()
    print(
        "=" * 80
    )

    print(
        "PART 2.18 COMPLETE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        history_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Best foraminal accuracy: "
        f"{best_metric:.6f}"
    )

    print(
        f"Best checkpoint: "
        f"{best_checkpoint}"
    )

    print()

    print(
        "Part104 was NOT modified."
    )

    print(
        "Part216 was NOT overwritten."
    )

    print(
        "Dashboard was NOT modified."
    )

    print()

    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()