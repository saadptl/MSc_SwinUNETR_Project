from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================================
# PROJECT PATHS
# ============================================================================

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# PART 2.20B PHYSICAL-SPACE PIPELINE
# ============================================================================

import segmentation_rsna_part220b_geometry_corrected_training as part220b


# ============================================================================
# FINAL MODEL
# ============================================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part33_balanced_symmetry_refinement"
    / "checkpoints"
    / "part33_best_development_macro.pth"
)


CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


NUM_CLASSES = 6

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# CHECKPOINT HASH
# ============================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# ============================================================================
# MODEL CHECKPOINT EXTRACTION
# ============================================================================

def _extract_state_dict(
    checkpoint,
):

    if not isinstance(
        checkpoint,
        dict,
    ):

        return checkpoint

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):

        value = checkpoint.get(
            key
        )

        if isinstance(
            value,
            dict,
        ):

            return value

    tensor_values = [
        value
        for value in checkpoint.values()
        if torch.is_tensor(value)
    ]

    if tensor_values:

        return checkpoint

    raise RuntimeError(
        "Could not find a model state dictionary "
        "inside the Part 3.3 checkpoint."
    )


# ============================================================================
# BUILD FINAL MODEL
# ============================================================================

def build_final_model():

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            "Part 3.3 final checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    model = part220b.build_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    state_dict = _extract_state_dict(
        checkpoint
    )

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    if missing or unexpected:

        raise RuntimeError(
            "Part 3.3 checkpoint did not load cleanly.\n"
            f"Missing keys: {len(missing)}\n"
            f"Unexpected keys: {len(unexpected)}"
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if parameter_count != 4_078_116:

        raise RuntimeError(
            "Unexpected model parameter count:\n"
            f"{parameter_count:,}"
        )

    model = (
        model
        .to(DEVICE)
        .eval()
    )

    return model


# ============================================================================
# LOAD MANIFEST
# ============================================================================

def load_manifest():

    manifest = part220b.load_manifest()

    if manifest.empty:

        raise RuntimeError(
            "RSNA manifest is empty."
        )

    required = {
        "study_id",
        "series_id",
    }

    missing = (
        required
        - set(manifest.columns)
    )

    if missing:

        raise RuntimeError(
            "Manifest is missing columns: "
            f"{sorted(missing)}"
        )

    return manifest


# ============================================================================
# FIND SERIES
# ============================================================================

def find_series(
    manifest,
    study_id,
    series_id,
):

    study_id = str(
        study_id
    ).strip()

    series_id = str(
        series_id
    ).strip()

    rows = manifest[
        (
            manifest["study_id"]
            .astype(str)
            == study_id
        )
        &
        (
            manifest["series_id"]
            .astype(str)
            == series_id
        )
    ].copy()

    if rows.empty:

        raise ValueError(
            "The requested RSNA study/series was not found "
            "in the manifest.\n\n"
            f"Study ID: {study_id}\n"
            f"Series ID: {series_id}"
        )

    return rows


# ============================================================================
# CONVERT IMAGE TO TENSOR
# ============================================================================

def _image_tensor(
    image,
):

    if torch.is_tensor(image):

        tensor = image.float()

    else:

        tensor = torch.as_tensor(
            image,
            dtype=torch.float32,
        )

    if tensor.ndim != 3:

        raise RuntimeError(
            "Expected canonical MRI volume "
            "with shape (D,H,W).\n"
            f"Received: {tuple(tensor.shape)}"
        )

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return tensor.to(
        DEVICE,
        non_blocking=True,
    )


# ============================================================================
# INFERENCE
# ============================================================================

@torch.no_grad()
def _predict(
    model,
    image_tensor,
):

    if DEVICE.type == "cuda":

        with torch.amp.autocast(
            "cuda",
            enabled=True,
        ):

            output = model(
                image_tensor
            )

    else:

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

    if logits.ndim != 5:

        raise RuntimeError(
            "Unexpected Swin-UNETR output shape:\n"
            f"{tuple(logits.shape)}"
        )

    if logits.shape[1] != NUM_CLASSES:

        raise RuntimeError(
            "Expected six output classes.\n"
            f"Received: {logits.shape[1]}"
        )

    probabilities = torch.softmax(
        logits.float(),
        dim=1,
    )

    prediction = torch.argmax(
        probabilities,
        dim=1,
    )

    return (
        logits,
        probabilities,
        prediction,
    )


# ============================================================================
# GEOMETRY VALUE
# ============================================================================

def _geometry_value(
    geometry,
    name,
    default=None,
):

    if isinstance(
        geometry,
        dict,
    ):

        return geometry.get(
            name,
            default,
        )

    return getattr(
        geometry,
        name,
        default,
    )


# ============================================================================
# PHYSICAL EXTENTS
# ============================================================================

def physical_extents(
    geometry,
    image_shape,
):

    patient_min = _geometry_value(
        geometry,
        "patient_min",
    )

    patient_max = _geometry_value(
        geometry,
        "patient_max",
    )

    if (
        patient_min is not None
        and
        patient_max is not None
    ):

        patient_min = np.asarray(
            patient_min,
            dtype=float,
        )

        patient_max = np.asarray(
            patient_max,
            dtype=float,
        )

        if (
            patient_min.size >= 3
            and
            patient_max.size >= 3
        ):

            return {
                "x": (
                    float(
                        min(
                            patient_min[0],
                            patient_max[0],
                        )
                    ),
                    float(
                        max(
                            patient_min[0],
                            patient_max[0],
                        )
                    ),
                ),
                "y": (
                    float(
                        min(
                            patient_min[1],
                            patient_max[1],
                        )
                    ),
                    float(
                        max(
                            patient_min[1],
                            patient_max[1],
                        )
                    ),
                ),
                "z": (
                    float(
                        min(
                            patient_min[2],
                            patient_max[2],
                        )
                    ),
                    float(
                        max(
                            patient_min[2],
                            patient_max[2],
                        )
                    ),
                ),
            }

    depth, height, width = (
        image_shape
    )

    return {
        "x": (
            0.0,
            float(width - 1),
        ),
        "y": (
            0.0,
            float(height - 1),
        ),
        "z": (
            0.0,
            float(depth - 1),
        ),
    }


# ============================================================================
# LOAD + INFER ONE RSNA SERIES
# ============================================================================

def run_final_inference(
    study_id,
    series_id,
    manifest=None,
    model=None,
):

    if manifest is None:

        manifest = load_manifest()

    point_df = find_series(
        manifest,
        study_id,
        series_id,
    )

    if model is None:

        model = build_final_model()

    study_id = str(
        study_id
    )

    series_id = str(
        series_id
    )

    result = part220b.load_case(
        study_id,
        series_id,
        point_df,
    )

    image_native = None
    try:
        import pandas as pd
        series_dir = part220b.TRAIN_IMAGES_DIR / str(study_id) / str(series_id)
        if not series_dir.exists():
            row = pd.Series({"study_id": str(study_id), "series_id": str(series_id)})
            series_dir = part220b.resolve_series_dir(row)
        if series_dir.exists():
            image_native, records, info = part220b.read_dicom_series_robust(series_dir)
    except Exception:
        image_native = None

    if len(result) == 3:

        image, points, geometry = result

    elif len(result) == 4:

        image, points, geometry, _extra = result

    else:

        raise RuntimeError(
            "Unexpected Part 2.20B load_case result."
        )

    image_tensor = _image_tensor(
        image
    )

    (
        logits,
        probabilities,
        prediction,
    ) = _predict(
        model,
        image_tensor,
    )

    image_np = (
        image_tensor
        [0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    logits_np = (
        logits
        [0]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    probabilities_np = (
        probabilities
        [0]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    prediction_np = (
        prediction
        [0]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.uint8
        )
    )

    metadata = {
        "study_id": study_id,
        "series_id": series_id,
    }

    if "series_description" in point_df.columns:

        values = (
            point_df[
                "series_description"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        metadata[
            "series_description"
        ] = (
            values[0]
            if values
            else "Unknown"
        )

    metadata[
        "annotation_rows"
    ] = int(
        len(point_df)
    )

    metadata[
        "canonical_shape"
    ] = list(
        image_np.shape
    )

    extents = physical_extents(
        geometry,
        image_np.shape,
    )

    metadata[
        "physical_extents_mm"
    ] = extents

    metadata[
        "model"
    ] = "MONAI Swin-UNETR"

    metadata[
        "model_variant"
    ] = "Part 3.3 Balanced LFNN/RFNN Symmetry Refinement"

    metadata[
        "checkpoint"
    ] = str(
        CHECKPOINT
    )

    metadata[
        "checkpoint_sha256"
    ] = sha256_file(
        CHECKPOINT
    )

    metadata[
        "device"
    ] = str(
        DEVICE
    )

    metadata[
        "model_parameters"
    ] = 4_078_116

    metadata[
        "voxel_segmentation_validated"
    ] = False

    metadata[
        "manual_voxel_ground_truth_available"
    ] = False

    return {
        "study_id": study_id,
        "series_id": series_id,
        "image": image_np,
        "image_native": image_native,
        "logits": logits_np,
        "probabilities": probabilities_np,
        "prediction": prediction_np,
        "points": points,
        "geometry": geometry,
        "physical_extents": extents,
        "metadata": metadata,
        "checkpoint": str(
            CHECKPOINT
        ),
        "checkpoint_sha256": sha256_file(
            CHECKPOINT
        ),
        "device": str(
            DEVICE
        ),
        "model_parameters": 4_078_116,
    }


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def probability_summary(
    probabilities,
):

    rows = []

    for class_id in range(
        1,
        NUM_CLASSES,
    ):

        values = probabilities[
            class_id
        ]

        rows.append(
            {
                "Disease":
                    CLASS_NAMES[
                        class_id
                    ],

                "Mean probability":
                    float(
                        values.mean()
                    ),

                "Maximum probability":
                    float(
                        values.max()
                    ),

                "P99 probability":
                    float(
                        np.percentile(
                            values,
                            99,
                        )
                    ),

                "Voxels ≥ 0.50":
                    int(
                        np.count_nonzero(
                            values >= 0.50
                        )
                    ),

                "Voxels ≥ 0.75":
                    int(
                        np.count_nonzero(
                            values >= 0.75
                        )
                    ),

                "Voxels ≥ 0.90":
                    int(
                        np.count_nonzero(
                            values >= 0.90
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )