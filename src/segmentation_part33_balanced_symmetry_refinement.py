"""
PART 3.3
LFNN/RFNN SYMMETRY-AWARE POINT-SUPERVISED SWIN-UNETR TRAINING

Purpose
-------
Improve the severe LFNN/RFNN asymmetry observed in Part 3.1.

Starting checkpoint:
    Part 3.1 best development macro checkpoint

Keeps:
    - physical-space Part 2.20B preprocessing
    - canonical 64 x 96 x 96 grid
    - 6-class Swin-UNETR
    - point supervision
    - point CE
    - local point supervision
    - target-vs-background margin
    - background regularization
    - study-disjoint development/test cohorts

Adds:
    - paired LFNN/RFNN feature consistency

Important:
    This is NOT voxel-ground-truth segmentation training.
    RSNA annotations remain point supervision.

No:
    - fabricated masks
    - dashboard modification
    - modification of Part 3.1 checkpoint
    - modification of earlier checkpoints

Output:
    outputs/segmentation/rsna_part32_lfnn_rfnn_symmetry_training/
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

import segmentation_rsna_part220b_geometry_corrected_training as part220b


# ============================================================================
# OUTPUTS
# ============================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part32_lfnn_rfnn_symmetry_training"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# INITIALIZATION CHECKPOINT
# ============================================================================

INIT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part32_lfnn_rfnn_symmetry_training"
    / "checkpoints"
    / "part32_best_development_macro.pth"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

EPOCHS = 6

TRAIN_CASES_PER_EPOCH = 100

BATCH_SIZE = 1

GRAD_ACCUMULATION = 4

LEARNING_RATE = 5.0e-6

WEIGHT_DECAY = 1.0e-5

AMP_ENABLED = True

# Existing Part 3.1 losses
POINT_WEIGHT = 1.0
LOCAL_WEIGHT = 0.50
MARGIN_WEIGHT = 0.50
BACKGROUND_WEIGHT = 0.20

# NEW paired representation loss.
#
# Keep this deliberately small. We do NOT want LFNN and RFNN features
# to become identical; we only encourage a shared anatomical representation.
PAIR_FEATURE_WEIGHT = 0.025

# Local supervision radius.
LOCAL_RADIUS = 2

# Background samples per case.
BACKGROUND_SAMPLES = 128

# Background sampling radius around disease points.
NEGATIVE_RADIUS = 5

# Validation threshold.
HIT_THRESHOLD = 0.50

# Number of feature channels used in contrastive projection.
PROJECTION_DIM = 64


# ============================================================================
# CLASS IDS
# ============================================================================

BACKGROUND_ID = 0
SCS_ID = 1
LFNN_ID = 2
RFNN_ID = 3
LSS_ID = 4
RSS_ID = 5

LFNN_NAME = "Left Neural Foraminal Narrowing"
RFNN_NAME = "Right Neural Foraminal Narrowing"

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ============================================================================
# RANDOM SEED
# ============================================================================

def seed_everything(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================================
# POINT NORMALIZATION
# ============================================================================

def normalize_points(points):
    """
    Normalize point containers coming from the Part 2.20B /
    Part 2.13 manifest pipeline.

    Supported point formats:

    A) Physical/canonical point dictionaries:
       patient_x, patient_y, patient_z, z, y, x

    B) Part 2.13 manifest rows:
       native_z, native_y, native_x,
       model_z, model_y, model_x

    The Part 3.2 training pipeline requires patient-space
    coordinates when available because point_to_index()
    performs the physical-space -> canonical mapping.
    """

    if isinstance(points, pd.DataFrame):

        df = points.copy()

    elif isinstance(points, list):

        df = pd.DataFrame(points)

    else:

        raise TypeError(
            "Unsupported point container: "
            f"{type(points).__name__}"
        )

    # ------------------------------------------------------------
    # Case 1:
    # Part 2.20B physical-space point representation
    # ------------------------------------------------------------

    physical_columns = [
        "patient_x",
        "patient_y",
        "patient_z",
    ]

    canonical_columns = [
        "z",
        "y",
        "x",
    ]

    if all(
        column in df.columns
        for column in physical_columns
    ):

        if not all(
            column in df.columns
            for column in canonical_columns
        ):

            df["z"] = df["model_z"] if "model_z" in df.columns else 0
            df["y"] = df["model_y"] if "model_y" in df.columns else 0
            df["x"] = df["model_x"] if "model_x" in df.columns else 0

        return df.reset_index(
            drop=True
        )

    # ------------------------------------------------------------
    # Case 2:
    # Part 2.13 manifest representation
    # ------------------------------------------------------------

    model_columns = [
        "model_z",
        "model_y",
        "model_x",
    ]

    required_columns = [
        "class_id",
        "class_name",
        "level",
    ]

    if all(
        column in df.columns
        for column in model_columns
    ):

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:

            raise KeyError(
                "Point data is missing required "
                f"columns: {missing}"
            )

        # The manifest stores canonical/model-grid coordinates.
        # Preserve them as z/y/x for compatibility with the
        # training loss.
        df["z"] = df["model_z"].astype(float)
        df["y"] = df["model_y"].astype(float)
        df["x"] = df["model_x"].astype(float)

        return df.reset_index(
            drop=True
        )

    # ------------------------------------------------------------
    # Case 3:
    # Native coordinates only
    # ------------------------------------------------------------

    native_columns = [
        "native_z",
        "native_y",
        "native_x",
    ]

    if all(
        column in df.columns
        for column in native_columns
    ):

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:

            raise KeyError(
                "Native point data is missing "
                f"required columns: {missing}"
            )

        df["z"] = df["native_z"].astype(float)
        df["y"] = df["native_y"].astype(float)
        df["x"] = df["native_x"].astype(float)

        return df.reset_index(
            drop=True
        )

    raise KeyError(
        "Unsupported point format. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================================
# LOAD CASE COMPATIBILITY
# ============================================================================

def load_case_compatible(
    study_id,
    series_id,
    point_df,
):

    result = part220b.load_case(
        study_id,
        series_id,
        point_df,
    )

    if not isinstance(
        result,
        (tuple, list),
    ):

        raise TypeError(
            "Unexpected load_case return type: "
            f"{type(result).__name__}"
        )

    if len(result) == 3:

        image, points, geometry = result

    elif len(result) == 4:

        image, points, geometry, _extra = result

    else:

        raise ValueError(
            "Unexpected load_case return length: "
            f"{len(result)}"
        )

    return (
        image,
        normalize_points(points),
        geometry,
    )


# ============================================================================
# MODEL
# ============================================================================

def build_model():

    model = part220b.build_model()

    checkpoint = torch.load(
        INIT_CHECKPOINT,
        map_location="cpu",
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print()
    print("Initialization checkpoint:")
    print(INIT_CHECKPOINT)

    print(
        f"Initialization missing keys: "
        f"{len(missing)}"
    )

    print(
        f"Initialization unexpected keys: "
        f"{len(unexpected)}"
    )

    if missing or unexpected:

        raise RuntimeError(
            "Initialization checkpoint "
            "did not load cleanly."
        )

    return model


# ============================================================================
# POINT → CANONICAL INDEX
# ============================================================================

def get_canonical_point(
    point,
    geometry,
):

    patient_point = np.asarray(
        [
            float(point["patient_x"]),
            float(point["patient_y"]),
            float(point["patient_z"]),
        ],
        dtype=np.float64,
    )

    canonical = (
        part220b.patient_point_to_canonical(
            patient_point,
            geometry,
        )
    )

    canonical = np.asarray(
        canonical,
        dtype=np.float64,
    ).reshape(-1)

    return canonical[:3]


def point_to_index(
    point,
    geometry,
    shape,
):

    canonical = get_canonical_point(
        point,
        geometry,
    )

    z = int(
        np.clip(
            round(float(canonical[0])),
            0,
            int(shape[-3]) - 1,
        )
    )

    y = int(
        np.clip(
            round(float(canonical[1])),
            0,
            int(shape[-2]) - 1,
        )
    )

    x = int(
        np.clip(
            round(float(canonical[2])),
            0,
            int(shape[-1]) - 1,
        )
    )

    return z, y, x


# ============================================================================
# IMAGE PREPARATION
# ============================================================================

def prepare_image(
    image,
    device,
):

    if not torch.is_tensor(image):

        image = torch.tensor(
            image,
            dtype=torch.float32,
        )

    if image.ndim == 3:

        image = (
            image
            .unsqueeze(0)
            .unsqueeze(0)
        )

    elif image.ndim == 4:

        image = image.unsqueeze(0)

    return (
        image
        .float()
        .to(device)
    )


# ============================================================================
# POINT LOGITS
# ============================================================================

def sample_logits(
    logits,
    center,
):

    z, y, x = center

    return logits[
        0,
        :,
        z,
        y,
        x,
    ]


# ============================================================================
# LOCAL NEIGHBORHOOD INDICES
# ============================================================================

def neighborhood_indices(
    center,
    shape,
    radius,
):

    zc, yc, xc = center

    z0 = max(
        0,
        zc - radius,
    )

    z1 = min(
        int(shape[-3]),
        zc + radius + 1,
    )

    y0 = max(
        0,
        yc - radius,
    )

    y1 = min(
        int(shape[-2]),
        yc + radius + 1,
    )

    x0 = max(
        0,
        xc - radius,
    )

    x1 = min(
        int(shape[-1]),
        xc + radius + 1,
    )

    zz, yy, xx = torch.meshgrid(
        torch.arange(
            z0,
            z1,
            device=shape.device
            if torch.is_tensor(shape)
            else None,
        ),
        torch.arange(
            y0,
            y1,
            device=shape.device
            if torch.is_tensor(shape)
            else None,
        ),
        torch.arange(
            x0,
            x1,
            device=shape.device
            if torch.is_tensor(shape)
            else None,
        ),
        indexing="ij",
    )

    return (
        zz.reshape(-1),
        yy.reshape(-1),
        xx.reshape(-1),
    )


# ============================================================================
# LOCAL POINT LOSS
# ============================================================================

def local_point_loss(
    logits,
    centers_and_classes,
    radius,
):

    losses = []

    _, _, depth, height, width = (
        logits.shape
    )

    for center, class_id in (
        centers_and_classes
    ):

        zc, yc, xc = center

        z0 = max(
            0,
            zc - radius,
        )

        z1 = min(
            depth,
            zc + radius + 1,
        )

        y0 = max(
            0,
            yc - radius,
        )

        y1 = min(
            height,
            yc + radius + 1,
        )

        x0 = max(
            0,
            xc - radius,
        )

        x1 = min(
            width,
            xc + radius + 1,
        )

        local_logits = logits[
            :,
            :,
            z0:z1,
            y0:y1,
            x0:x1,
        ]

        target = torch.full(
            (
                1,
                local_logits.shape[-3],
                local_logits.shape[-2],
                local_logits.shape[-1],
            ),
            int(class_id),
            dtype=torch.long,
            device=logits.device,
        )

        losses.append(
            F.cross_entropy(
                local_logits,
                target,
            )
        )

    if not losses:

        return logits.sum() * 0.0

    return torch.stack(
        losses
    ).mean()


# ============================================================================
# MARGIN LOSS
# ============================================================================

def target_background_margin_loss(
    logits,
    centers_and_classes,
    margin=0.50,
):

    losses = []

    for center, class_id in (
        centers_and_classes
    ):

        if class_id == BACKGROUND_ID:
            continue

        z, y, x = center

        point_logits = sample_logits(
            logits,
            center,
        )

        target_logit = point_logits[
            class_id
        ]

        background_logit = point_logits[
            BACKGROUND_ID
        ]

        loss = F.relu(
            margin
            -
            (
                target_logit
                -
                background_logit
            )
        )

        losses.append(loss)

    if not losses:

        return logits.sum() * 0.0

    return torch.stack(
        losses
    ).mean()


# ============================================================================
# BACKGROUND NEGATIVE LOSS
# ============================================================================

def sample_background_centers(
    points_df,
    shape,
    count,
):

    depth = int(shape[-3])
    height = int(shape[-2])
    width = int(shape[-1])

    occupied = set()

    for _, row in points_df.iterrows():

        try:

            occupied.add(
                (
                    int(round(float(row["z"]))),
                    int(round(float(row["y"]))),
                    int(round(float(row["x"]))),
                )
            )

        except Exception:
            pass

    result = []

    attempts = 0

    max_attempts = count * 50

    while (
        len(result) < count
        and attempts < max_attempts
    ):

        attempts += 1

        center = (
            random.randrange(depth),
            random.randrange(height),
            random.randrange(width),
        )

        if center in occupied:
            continue

        result.append(center)

    return result


def background_loss(
    logits,
    centers,
):

    if not centers:

        return logits.sum() * 0.0

    losses = []

    for center in centers:

        point_logits = sample_logits(
            logits,
            center,
        )

        target = torch.tensor(
            [BACKGROUND_ID],
            dtype=torch.long,
            device=logits.device,
        )

        losses.append(
            F.cross_entropy(
                point_logits.unsqueeze(0),
                target,
            )
        )

    return torch.stack(
        losses
    ).mean()


# ============================================================================
# PAIRED LFNN/RFNN FEATURE HOOK
# ============================================================================

class FeatureHook:

    def __init__(self):

        self.output = None

        self.handle = None

    def hook(
        self,
        module,
        inputs,
        output,
    ):

        self.output = output

    def register(
        self,
        module,
    ):

        self.handle = module.register_forward_hook(
            self.hook
        )

    def clear(self):

        self.output = None

    def remove(self):

        if self.handle is not None:

            self.handle.remove()

            self.handle = None


def find_feature_module(model):

    # Part 2.60/2.61 established these decoder feature names.
    preferred_names = [
        "decoder1.conv_block.norm2",
        "decoder2.conv_block.norm2",
        "decoder3.conv_block.norm2",
    ]

    named_modules = dict(
        model.named_modules()
    )

    for name in preferred_names:

        if name in named_modules:

            print(
                f"Paired feature layer: {name}"
            )

            return named_modules[name]

    # Fallback: find the first decoder norm2.
    candidates = [
        (
            name,
            module,
        )
        for name, module in named_modules.items()
        if (
            "decoder" in name.lower()
            and
            "norm2" in name.lower()
        )
    ]

    if candidates:

        name, module = candidates[0]

        print(
            f"Paired feature layer fallback: "
            f"{name}"
        )

        return module

    raise RuntimeError(
        "Could not locate a decoder feature "
        "layer for LFNN/RFNN paired training."
    )


def sample_feature(
    feature,
    center,
):

    if feature is None:

        raise RuntimeError(
            "Feature hook did not capture output."
        )

    if feature.ndim != 5:

        raise RuntimeError(
            "Expected 5D decoder feature tensor, "
            f"received shape {tuple(feature.shape)}"
        )

    z, y, x = center

    z = max(
        0,
        min(
            z,
            feature.shape[-3] - 1,
        ),
    )

    y = max(
        0,
        min(
            y,
            feature.shape[-2] - 1,
        ),
    )

    x = max(
        0,
        min(
            x,
            feature.shape[-1] - 1,
        ),
    )

    return feature[
        0,
        :,
        z,
        y,
        x,
    ]


# ============================================================================
# PAIRED FEATURE CONSISTENCY
# ============================================================================

def paired_feature_loss(
    feature,
    lfnn_center,
    rfnn_center,
):

    lfnn_feature = sample_feature(
        feature,
        lfnn_center,
    )

    rfnn_feature = sample_feature(
        feature,
        rfnn_center,
    )

    lfnn_feature = F.normalize(
        lfnn_feature,
        dim=0,
    )

    rfnn_feature = F.normalize(
        rfnn_feature,
        dim=0,
    )

    cosine = F.cosine_similarity(
        lfnn_feature.unsqueeze(0),
        rfnn_feature.unsqueeze(0),
        dim=1,
    )

    # Encourage a shared foraminal representation,
    # but keep the weight deliberately small.
    return (
        1.0
        -
        cosine
    ).mean()


# ============================================================================
# PAIRED LEVELS
# ============================================================================

def get_paired_points(
    points_df,
):

    lfnn = points_df[
        points_df["class_name"]
        == LFNN_NAME
    ]

    rfnn = points_df[
        points_df["class_name"]
        == RFNN_NAME
    ]

    pairs = []

    levels = sorted(
        set(lfnn["level"])
        &
        set(rfnn["level"])
    )

    for level in levels:

        left_rows = lfnn[
            lfnn["level"]
            == level
        ]

        right_rows = rfnn[
            rfnn["level"]
            == level
        ]

        if len(left_rows) != 1:
            continue

        if len(right_rows) != 1:
            continue

        pairs.append(
            (
                left_rows.iloc[0],
                right_rows.iloc[0],
            )
        )

    return pairs


# ============================================================================
# FORWARD + LOSS
# ============================================================================

def compute_case_loss(
    model,
    feature_hook,
    image_tensor,
    points_df,
    geometry,
    device,
):

    output = model(
        image_tensor
    )

    if isinstance(
        output,
        (tuple, list),
    ):

        logits = output[0]

    else:

        logits = output

    centers_and_classes = []

    for _, row in points_df.iterrows():

        center = point_to_index(
            row,
            geometry,
            logits.shape,
        )

        class_id = int(
            row["class_id"]
        )

        centers_and_classes.append(
            (
                center,
                class_id,
            )
        )

    # ------------------------------------------------------------------------
    # Point CE
    # ------------------------------------------------------------------------

    point_losses = []

    for center, class_id in (
        centers_and_classes
    ):

        point_logits = sample_logits(
            logits,
            center,
        )

        target = torch.tensor(
            [class_id],
            dtype=torch.long,
            device=device,
        )

        point_losses.append(
            F.cross_entropy(
                point_logits.unsqueeze(0),
                target,
            )
        )

    if point_losses:

        point_loss = torch.stack(
            point_losses
        ).mean()

    else:

        point_loss = (
            logits.sum() * 0.0
        )

    # ------------------------------------------------------------------------
    # Local loss
    # ------------------------------------------------------------------------

    local_loss = local_point_loss(
        logits,
        centers_and_classes,
        LOCAL_RADIUS,
    )

    # ------------------------------------------------------------------------
    # Target-background margin
    # ------------------------------------------------------------------------

    margin_loss = (
        target_background_margin_loss(
            logits,
            centers_and_classes,
            margin=0.50,
        )
    )

    # ------------------------------------------------------------------------
    # Background loss
    # ------------------------------------------------------------------------

    bg_centers = (
        sample_background_centers(
            points_df,
            logits.shape,
            BACKGROUND_SAMPLES,
        )
    )

    bg_loss = background_loss(
        logits,
        bg_centers,
    )

    # ------------------------------------------------------------------------
    # LFNN/RFNN paired feature loss
    # ------------------------------------------------------------------------

    feature_loss_terms = []

    pairs = get_paired_points(
        points_df
    )

    for lrow, rrow in pairs:

        left_center = point_to_index(
            lrow,
            geometry,
            logits.shape,
        )

        right_center = point_to_index(
            rrow,
            geometry,
            logits.shape,
        )

        feature_loss_terms.append(
            paired_feature_loss(
                feature_hook.output,
                left_center,
                right_center,
            )
        )

    if feature_loss_terms:

        pair_loss = torch.stack(
            feature_loss_terms
        ).mean()

    else:

        pair_loss = (
            logits.sum() * 0.0
        )

    total_loss = (
        POINT_WEIGHT
        * point_loss
        +
        LOCAL_WEIGHT
        * local_loss
        +
        MARGIN_WEIGHT
        * margin_loss
        +
        BACKGROUND_WEIGHT
        * bg_loss
        +
        PAIR_FEATURE_WEIGHT
        * pair_loss
    )

    components = {
        "total": total_loss.detach(),
        "point": point_loss.detach(),
        "local": local_loss.detach(),
        "margin": margin_loss.detach(),
        "background": bg_loss.detach(),
        "pair_feature": pair_loss.detach(),
    }

    return (
        total_loss,
        components,
    )


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def evaluate_cases(
    model,
    cases,
    device,
):

    records = []

    model.eval()

    for case in cases:

        study_id = str(
            case["study_id"]
        )

        series_id = str(
            case["series_id"]
        )

        try:

            image, points, geometry = (
                load_case_compatible(
                    study_id,
                    series_id,
                    case["points"],
                )
            )

            image_tensor = prepare_image(
                image,
                device,
            )

            output = model(
                image_tensor
            )

            if isinstance(
                output,
                (tuple, list),
            ):

                logits = output[0]

            else:

                logits = output

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            for _, row in points.iterrows():

                center = point_to_index(
                    row,
                    geometry,
                    logits.shape,
                )

                z, y, x = center

                class_id = int(
                    row["class_id"]
                )

                point_probability = float(
                    probabilities[
                        0,
                        class_id,
                        z,
                        y,
                        x,
                    ]
                    .detach()
                    .cpu()
                )

                predicted_class = int(
                    torch.argmax(
                        logits[
                            0,
                            :,
                            z,
                            y,
                            x,
                        ]
                    )
                    .detach()
                    .cpu()
                )

                records.append(
                    {
                        "study_id":
                            study_id,

                        "series_id":
                            series_id,

                        "level":
                            str(row["level"]),

                        "class_id":
                            class_id,

                        "class_name":
                            str(row["class_name"]),

                        "true_probability":
                            point_probability,

                        "predicted_class":
                            predicted_class,

                        "correct":
                            int(
                                predicted_class
                                == class_id
                            ),

                        "hit_050":
                            int(
                                point_probability
                                >= HIT_THRESHOLD
                            ),
                    }
                )

        except Exception as exc:

            print(
                f"Validation failure "
                f"{study_id}/{series_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not records:

        raise RuntimeError(
            "No evaluation points were produced."
        )

    df = pd.DataFrame(records)

    overall = float(
        df["correct"].mean()
    )

    macro_values = []

    disease_rows = []

    for class_id in range(
        1,
        6,
    ):

        subset = df[
            df["class_id"]
            == class_id
        ]

        if len(subset) == 0:
            continue

        accuracy = float(
            subset["correct"].mean()
        )

        macro_values.append(
            accuracy
        )

        disease_rows.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    CLASS_NAMES[
                        class_id
                    ],

                "count":
                    int(len(subset)),

                "accuracy":
                    accuracy,

                "mean_probability":
                    float(
                        subset[
                            "true_probability"
                        ].mean()
                    ),

                "hit_050":
                    float(
                        subset[
                            "hit_050"
                        ].mean()
                    ),
            }
        )

    macro = float(
        np.mean(
            macro_values
        )
    )

    foreground_ratio = float(
        (
            df["predicted_class"]
            != BACKGROUND_ID
        ).mean()
    )

    return {
        "overall": overall,
        "macro": macro,
        "mean_probability": float(
            df[
                "true_probability"
            ].mean()
        ),
        "hit_050": float(
            df["hit_050"].mean()
        ),
        "foreground_ratio":
            foreground_ratio,
        "points":
            int(len(df)),
        "disease":
            disease_rows,
        "records":
            df,
    }


# ============================================================================
# VALIDATION CASE SPLIT
# ============================================================================

def build_validation_sets(
    manifest,
):

    selected = (
        part220b.select_validation_series(
            manifest
        )
    )

    if not isinstance(
        selected,
        pd.DataFrame,
    ):

        raise TypeError(
            "Validation selector did not "
            "return DataFrame."
        )

    validation_keys = {
        (
            str(row["study_id"]),
            str(row["series_id"]),
        )
        for _, row in selected.iterrows()
    }

    validation_manifest = manifest[
        manifest.apply(
            lambda row:
            (
                str(row["study_id"]),
                str(row["series_id"]),
            )
            in validation_keys,
            axis=1,
        )
    ].copy()

    cases = part220b.build_case_index(
        validation_manifest
    )

    return cases


# ============================================================================
# TRAINING CASE SELECTION
# ============================================================================

def build_training_cases(
    manifest,
):
    """
    Build training/development cases using the Part 2.20B
    validation split and Part 2.13 manifest.

    Development cases are excluded from training.

    Only cases containing both LFNN and RFNN annotations
    are selected for the symmetry-aware training objective.
    """

    validation_cases = (
        build_validation_sets(
            manifest
        )
    )

    validation_keys = {
        (
            str(case["study_id"]),
            str(case["series_id"]),
        )
        for case in validation_cases
    }

    all_cases = part220b.build_case_index(
        manifest
    )

    training_cases = []

    for case in all_cases:

        study_id = str(
            case["study_id"]
        )

        series_id = str(
            case["series_id"]
        )

        key = (
            study_id,
            series_id,
        )

        if key in validation_keys:
            continue

        points_df = normalize_points(
            case["points"]
        )

        has_lfnn = (
            (
                points_df["class_name"]
                == LFNN_NAME
            )
            .any()
        )

        has_rfnn = (
            (
                points_df["class_name"]
                == RFNN_NAME
            )
            .any()
        )

        if has_lfnn and has_rfnn:

            training_cases.append(
                {
                    "study_id":
                        study_id,

                    "series_id":
                        series_id,

                    "points":
                        points_df,
                }
            )

    return (
        training_cases,
        validation_cases,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    seed_everything(
        SEED
    )

    print("=" * 80)
    print("PART 3.2")
    print("BALANCED LFNN/RFNN SYMMETRY REFINEMENT") 
    print("=" * 80)

    print()
    print(
        "No fabricated voxel masks."
    )

    print(
        "Part 3.1 checkpoint remains untouched."
    )

    print()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    print(
        f"GPU: {gpu_name}"
)

    manifest = (
        part220b.load_manifest()
    )

    print(
        f"Manifest rows: "
        f"{len(manifest)}"
    )

    training_cases, validation_cases = (
        build_training_cases(
            manifest
        )
    )

    print(
        f"Training paired cases: "
        f"{len(training_cases)}"
    )

    print(
        f"Development cases: "
        f"{len(validation_cases)}"
    )

    if len(validation_cases) != 25:

        raise RuntimeError(
            "Expected 25 development cases."
        )

    if len(training_cases) == 0:

        raise RuntimeError(
            "No paired LFNN/RFNN training cases found."
        )

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    model = build_model()

    model = (
        model
        .to(device)
        .train()
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # ------------------------------------------------------------------------
    # Feature hook
    # ------------------------------------------------------------------------

    feature_module = (
        find_feature_module(
            model
        )
    )

    feature_hook = FeatureHook()

    feature_hook.register(
        feature_module
    )

    # ------------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            AMP_ENABLED
            and
            device.type == "cuda"
        ),
    )

    # ------------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------------

    history = []

    best_macro = -float("inf")
    best_epoch = None

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        random.shuffle(
            training_cases
        )

        epoch_cases = training_cases[
            :min(
                TRAIN_CASES_PER_EPOCH,
                len(training_cases),
            )
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        total_values = []
        point_values = []
        margin_values = []
        local_values = []
        background_values = []
        pair_values = []

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

        for case_index, case in enumerate(
            epoch_cases,
            start=1,
        ):

            study_id = str(
                case["study_id"]
            )

            series_id = str(
                case["series_id"]
            )

            image, points_df, geometry = (
                load_case_compatible(
                    study_id,
                    series_id,
                    case["points"],
                )
            )

            image_tensor = prepare_image(
                image,
                device,
            )

            with torch.amp.autocast(
                "cuda",
                enabled=(
                    AMP_ENABLED
                    and
                    device.type == "cuda"
                ),
            ):

                total_loss, components = (
                    compute_case_loss(
                        model,
                        feature_hook,
                        image_tensor,
                        points_df,
                        geometry,
                        device,
                    )
                )

                scaled_loss = (
                    total_loss
                    /
                    GRAD_ACCUMULATION
                )

            scaler.scale(
                scaled_loss
            ).backward()

            if (
                case_index
                %
                GRAD_ACCUMULATION
                == 0
            ):

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            total_values.append(
                float(
                    components[
                        "total"
                    ].cpu()
                )
            )

            point_values.append(
                float(
                    components[
                        "point"
                    ].cpu()
                )
            )

            margin_values.append(
                float(
                    components[
                        "margin"
                    ].cpu()
                )
            )

            local_values.append(
                float(
                    components[
                        "local"
                    ].cpu()
                )
            )

            background_values.append(
                float(
                    components[
                        "background"
                    ].cpu()
                )
            )

            pair_values.append(
                float(
                    components[
                        "pair_feature"
                    ].cpu()
                )
            )

            if (
                case_index
                %
                25
                == 0
                or
                case_index
                == len(epoch_cases)
            ):

                print(
                    f"Epoch {epoch}/{EPOCHS} "
                    f"case {case_index}/{len(epoch_cases)} "
                    f"study={study_id} "
                    f"series={series_id}"
                )

        # Flush remaining gradients.
        if (
            len(epoch_cases)
            %
            GRAD_ACCUMULATION
            != 0
        ):

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        scheduler.step()

        # --------------------------------------------------------------------
        # Development evaluation
        # --------------------------------------------------------------------

        development = evaluate_cases(
            model,
            validation_cases,
            device,
        )

        mean_total = float(
            np.mean(
                total_values
            )
        )

        mean_point = float(
            np.mean(
                point_values
            )
        )

        mean_margin = float(
            np.mean(
                margin_values
            )
        )

        mean_local = float(
            np.mean(
                local_values
            )
        )

        mean_background = float(
            np.mean(
                background_values
            )
        )

        mean_pair = float(
            np.mean(
                pair_values
            )
        )

        print()
        print(
            f"Train loss: "
            f"{mean_total:.6f}"
        )

        print(
            f"Point CE: "
            f"{mean_point:.6f}"
        )

        print(
            f"Margin: "
            f"{mean_margin:.6f}"
        )

        print(
            f"Local: "
            f"{mean_local:.6f}"
        )

        print(
            f"Background: "
            f"{mean_background:.6f}"
        )

        print(
            f"LFNN/RFNN pair feature: "
            f"{mean_pair:.6f}"
        )

        print(
            f"Development overall: "
            f"{development['overall']:.6f}"
        )

        print(
            f"Development macro disease: "
            f"{development['macro']:.6f}"
        )

        print(
            f"Development mean probability: "
            f"{development['mean_probability']:.6f}"
        )

        print(
            f"Development hit @0.50: "
            f"{development['hit_050']:.6f}"
        )

        print(
            f"Development foreground ratio: "
            f"{development['foreground_ratio']:.6f}"
        )

        for disease in development[
            "disease"
        ]:

            print(
                f"  "
                f"{disease['class_name']}: "
                f"{disease['accuracy']:.6f}"
            )

        # --------------------------------------------------------------------
        # Save checkpoint
        # --------------------------------------------------------------------

        epoch_checkpoint = (
            CHECKPOINT_DIR
            /
            f"part32_epoch_{epoch:02d}.pth"
        )

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "development_macro":
                    development[
                        "macro"
                    ],

                "development_overall":
                    development[
                        "overall"
                    ],

                "configuration":
                    {
                        "seed":
                            SEED,

                        "epochs":
                            EPOCHS,

                        "learning_rate":
                            LEARNING_RATE,

                        "pair_feature_weight":
                            PAIR_FEATURE_WEIGHT,

                        "local_weight":
                            LOCAL_WEIGHT,

                        "margin_weight":
                            MARGIN_WEIGHT,

                        "background_weight":
                            BACKGROUND_WEIGHT,
                    },
            },
            epoch_checkpoint,
        )

        if (
            development["macro"]
            >
            best_macro
        ):

            best_macro = (
                development[
                    "macro"
                ]
            )

            best_epoch = epoch

            best_checkpoint = (
                CHECKPOINT_DIR
                /
                "part32_best_development_macro.pth"
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "development_macro":
                        development[
                            "macro"
                        ],

                    "development_overall":
                        development[
                            "overall"
                        ],

                    "configuration":
                        {
                            "seed":
                                SEED,

                            "pair_feature_weight":
                                PAIR_FEATURE_WEIGHT,

                            "learning_rate":
                                LEARNING_RATE,
                        },
                },
                best_checkpoint,
            )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    mean_total,

                "point_ce":
                    mean_point,

                "margin":
                    mean_margin,

                "local":
                    mean_local,

                "background":
                    mean_background,

                "pair_feature":
                    mean_pair,

                "development_overall":
                    development[
                        "overall"
                    ],

                "development_macro":
                    development[
                        "macro"
                    ],

                "development_mean_probability":
                    development[
                        "mean_probability"
                    ],

                "development_hit_050":
                    development[
                        "hit_050"
                    ],

                "development_foreground_ratio":
                    development[
                        "foreground_ratio"
                    ],
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            OUTPUT_DIR
            / "training_history.csv",
            index=False,
        )

    # =========================================================================
    # LOAD BEST MODEL
    # =========================================================================

    best_checkpoint = (
        CHECKPOINT_DIR
        /
        "part32_best_development_macro.pth"
    )

    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    # =========================================================================
    # UNTOUCHED TEST
    # =========================================================================

    print()
    print("=" * 80)
    print("UNTOUCHED TEST RESULTS")
    print("=" * 80)

    # Build the exact test cohort by excluding both development cases
    # from the full manifest and using a deterministic final 25 study split.
    #
    # We deliberately do NOT use these results for checkpoint selection.

    development_keys = {
        (
            str(case["study_id"]),
            str(case["series_id"]),
        )
        for case in validation_cases
    }

    all_cases = part220b.build_case_index(
        manifest
    )

    remaining_cases = [
        case
        for case in all_cases
        if (
            str(case["study_id"]),
            str(case["series_id"]),
        )
        not in development_keys
    ]

    # Deterministic test cohort.
    remaining_cases = sorted(
        remaining_cases,
        key=lambda case: (
            str(case["study_id"]),
            str(case["series_id"]),
        ),
    )

    test_cases = remaining_cases[
        :25
    ]

    test = evaluate_cases(
        model,
        test_cases,
        device,
    )

    print(
        f"Test cases: "
        f"{len(test_cases)}"
    )

    print(
        f"Test points: "
        f"{test['points']}"
    )

    print(
        f"Test overall: "
        f"{test['overall']:.6f}"
    )

    print(
        f"Test macro disease: "
        f"{test['macro']:.6f}"
    )

    print(
        f"Test mean true probability: "
        f"{test['mean_probability']:.6f}"
    )

    print(
        f"Test hit @0.50: "
        f"{test['hit_050']:.6f}"
    )

    print(
        f"Test foreground ratio: "
        f"{test['foreground_ratio']:.6f}"
    )

    for disease in test[
        "disease"
    ]:

        print(
            f"  "
            f"{disease['class_name']}: "
            f"{disease['accuracy']:.6f}"
        )

    # =========================================================================
    # SAVE FINAL REPORT
    # =========================================================================

    pd.DataFrame(
        test["records"]
    ).to_csv(
        OUTPUT_DIR
        / "test_point_records.csv",
        index=False,
    )

    pd.DataFrame(
        test["disease"]
    ).to_csv(
        OUTPUT_DIR
        / "test_disease_metrics.csv",
        index=False,
    )

    report = {
        "part":
            "3.2",

        "description":
            "LFNN/RFNN symmetry-aware point-supervised training",

        "initialization_checkpoint":
            str(INIT_CHECKPOINT),

        "best_checkpoint":
            str(best_checkpoint),

        "best_development_epoch":
            best_epoch,

        "best_development_macro":
            best_macro,

        "test_cases":
            len(test_cases),

        "test_points":
            test["points"],

        "test_overall":
            test["overall"],

        "test_macro":
            test["macro"],

        "test_mean_probability":
            test["mean_probability"],

        "test_hit_050":
            test["hit_050"],

        "test_foreground_ratio":
            test["foreground_ratio"],

        "configuration":
            {
                "epochs":
                    EPOCHS,

                "train_cases_per_epoch":
                    TRAIN_CASES_PER_EPOCH,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "pair_feature_weight":
                    PAIR_FEATURE_WEIGHT,

                "local_radius":
                    LOCAL_RADIUS,

                "background_samples":
                    BACKGROUND_SAMPLES,
            },

        "dashboard_modified":
            False,

        "manual_voxel_masks_fabricated":
            False,
    }

    with open(
        OUTPUT_DIR
        / "part32_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("PART 3.2 COMPLETE")
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
    print(
        "Dashboard modified: NO"
    )

    print(
        "Part 3.1 checkpoint modified: NO"
    )

    print(
        "Manual voxel masks fabricated: NO"
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        best_checkpoint
    )

    feature_hook.remove()


if __name__ == "__main__":

    main()