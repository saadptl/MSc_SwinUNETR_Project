from pathlib import Path
import sys
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import zoom

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "integrated"
    / "part94_swinunetr_classification_pipeline"
)

VIS_DIR = OUTPUT_DIR / "visualizations"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT CHECKPOINTS
# ============================================================

SEGMENTATION_CHECKPOINT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part84_final_reproducible_training"
    / "checkpoints"
    / "part84_best_model.pth"
)

CLASSIFICATION_CHECKPOINT = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part92_class_imbalance_aware_training"
    / "checkpoints"
    / "part92_best_model.pth"
)

VALIDATION_MANIFEST = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part87_study_level_split"
    / "part87_validation_manifest.csv"
)

SEGMENTATION_VALIDATION_COHORT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_validation_cohort.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# Classification input
CLASS_DEPTH = 32
CLASS_HEIGHT = 224
CLASS_WIDTH = 224

NUM_TARGETS = 25
NUM_CLASSES = 3

CLASS_NAMES = [
    "Normal/Mild",
    "Moderate",
    "Severe",
]

CONDITIONS = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]

LEVELS = [
    "l1_l2",
    "l2_l3",
    "l3_l4",
    "l4_l5",
    "l5_s1",
]

TARGET_COLUMNS = [
    f"{condition}_{level}"
    for condition in CONDITIONS
    for level in LEVELS
]

# Segmentation input/output geometry
SEG_FULL_SHAPE = (
    64,
    96,
    96,
)

SEG_CROP_SHAPE = (
    32,
    64,
    64,
)

SEG_NUM_CLASSES = 6

SEG_CLASS_NAMES = [
    "Background",
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

def import_project_modules():

    sys.path.insert(
        0,
        str(SRC_DIR),
    )

    import segmentation_rsna_part11_controlled_pilot_training as part11

    import classification_rsna_part89_dataset_pipeline as part89

    import classification_rsna_part92_class_imbalance_aware_training as part92

    return (
        part11,
        part89,
        part92,
    )


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_required_files():

    required = [
        SEGMENTATION_CHECKPOINT,
        CLASSIFICATION_CHECKPOINT,
        VALIDATION_MANIFEST,
        SEGMENTATION_VALIDATION_COHORT,
    ]

    print("\nCHECKING REQUIRED FILES")

    for path in required:

        print(
            f"{path.name:<55}",
            "PASS" if path.exists() else "MISSING",
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )


# ============================================================
# CHECKPOINT INSPECTION
# ============================================================

def inspect_checkpoint(
    path,
    name,
):

    print("\n" + "=" * 72)
    print(f"{name} CHECKPOINT")
    print("=" * 72)

    print("Path :", path)
    print(
        "Size :",
        f"{path.stat().st_size:,}",
        "bytes",
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        print(
            "Keys :",
            list(checkpoint.keys()),
        )

        if "epoch" in checkpoint:

            print(
                "Epoch:",
                checkpoint["epoch"],
            )

        if "part" in checkpoint:

            print(
                "Part :",
                checkpoint["part"],
            )

    return checkpoint


# ============================================================
# STATE DICT EXTRACTION
# ============================================================

def extract_state_dict(
    checkpoint,
):

    if isinstance(
        checkpoint,
        nn.Module,
    ):

        return checkpoint.state_dict()

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Unsupported checkpoint type."
        )

    for key in [
        "model_state_dict",
        "state_dict",
    ]:

        if key in checkpoint:

            value = checkpoint[key]

            if isinstance(
                value,
                dict,
            ):

                return value

    if any(
        isinstance(
            value,
            torch.Tensor,
        )
        for value in checkpoint.values()
    ):

        return checkpoint

    raise RuntimeError(
        "Could not find model state_dict."
    )


def clean_state_dict(
    state_dict,
):

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return cleaned


# ============================================================
# LOAD SWIN-UNETR
# ============================================================

def load_segmentation_model(
    part11,
):

    print(
        "\nLOADING FROZEN PART84 SWIN-UNETR"
    )

    checkpoint = torch.load(
        SEGMENTATION_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = part11.create_model(
        DEVICE
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "Strict load      : PASS"
    )

    print(
        "Missing keys     :",
        len(result.missing_keys),
    )

    print(
        "Unexpected keys  :",
        len(result.unexpected_keys),
    )

    model.eval()

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters       :",
        f"{parameters:,}",
    )

    return (
        model,
        checkpoint,
        parameters,
    )


# ============================================================
# LOAD CLASSIFICATION MODEL
# ============================================================

def load_classification_model(
    part92,
):

    print(
        "\nLOADING FROZEN PART92 CLASSIFIER"
    )

    checkpoint = torch.load(
        CLASSIFICATION_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = part92.Part90CNN(
        num_targets=NUM_TARGETS,
        num_classes=NUM_CLASSES,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "Strict load      : PASS"
    )

    print(
        "Missing keys     :",
        len(result.missing_keys),
    )

    print(
        "Unexpected keys  :",
        len(result.unexpected_keys),
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters       :",
        f"{parameters:,}",
    )

    return (
        model,
        checkpoint,
        parameters,
    )


# ============================================================
# SEGMENTATION CROP
# ============================================================

def centered_crop(
    image,
    crop_shape,
):

    if isinstance(
        image,
        torch.Tensor,
    ):

        image = image.detach().cpu().numpy()

    image = np.asarray(
        image
    )

    if image.ndim == 4:

        image = np.squeeze(
            image
        )

    if image.ndim != 3:

        raise RuntimeError(
            "Expected 3D image for crop, got "
            f"{image.shape}"
        )

    target_d, target_h, target_w = (
        crop_shape
    )

    d, h, w = image.shape

    start_d = max(
        0,
        (d - target_d) // 2,
    )

    start_h = max(
        0,
        (h - target_h) // 2,
    )

    start_w = max(
        0,
        (w - target_w) // 2,
    )

    end_d = min(
        d,
        start_d + target_d,
    )

    end_h = min(
        h,
        start_h + target_h,
    )

    end_w = min(
        w,
        start_w + target_w,
    )

    crop = image[
        start_d:end_d,
        start_h:end_h,
        start_w:end_w,
    ]

    # Pad if necessary
    padded = np.zeros(
        crop_shape,
        dtype=np.float32,
    )

    actual_d = crop.shape[0]
    actual_h = crop.shape[1]
    actual_w = crop.shape[2]

    padded[
        :actual_d,
        :actual_h,
        :actual_w,
    ] = crop

    return padded


# ============================================================
# 3D RESIZE
# ============================================================

def resize_3d(
    image,
    target_shape,
):

    image = np.asarray(
        image
    ).astype(
        np.float32
    )

    factors = [
        target_shape[i] / image.shape[i]
        for i in range(3)
    ]

    resized = zoom(
        image,
        factors,
        order=1,
    )

    # Exact final shape correction
    output = np.zeros(
        target_shape,
        dtype=np.float32,
    )

    d = min(
        target_shape[0],
        resized.shape[0],
    )

    h = min(
        target_shape[1],
        resized.shape[1],
    )

    w = min(
        target_shape[2],
        resized.shape[2],
    )

    output[
        :d,
        :h,
        :w,
    ] = resized[
        :d,
        :h,
        :w,
    ]

    return output


# ============================================================
# CLASSIFICATION IMAGE CONVERSION
# ============================================================

def prepare_classification_input(
    image,
):

    image = np.asarray(
        image
    ).astype(
        np.float32
    )

    if image.ndim == 4:

        image = np.squeeze(
            image
        )

    if image.ndim != 3:

        raise RuntimeError(
            "Classification input must be 3D."
        )

    # Normalize
    minimum = np.min(
        image
    )

    maximum = np.max(
        image
    )

    if maximum > minimum:

        image = (
            image - minimum
        ) / (
            maximum - minimum
        )

    else:

        image = np.zeros_like(
            image,
            dtype=np.float32,
        )

    image = resize_3d(
        image,
        (
            CLASS_DEPTH,
            CLASS_HEIGHT,
            CLASS_WIDTH,
        ),
    )

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.unsqueeze(
        0
    )

    return tensor


# ============================================================
# SEGMENTATION INPUT PREPARATION
# ============================================================

def prepare_segmentation_input(
    image,
):

    image = np.asarray(
        image
    ).astype(
        np.float32
    )

    if image.ndim == 4:

        image = np.squeeze(
            image
        )

    if image.ndim != 3:

        raise RuntimeError(
            "Segmentation input must be 3D."
        )

    # Part84 operates on 64x96x96 full geometry.
    image = resize_3d(
        image,
        SEG_FULL_SHAPE,
    )

    minimum = np.min(
        image
    )

    maximum = np.max(
        image
    )

    if maximum > minimum:

        image = (
            image - minimum
        ) / (
            maximum - minimum
        )

    else:

        image = np.zeros_like(
            image,
            dtype=np.float32,
        )

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.unsqueeze(
        0
    )

    return tensor


# ============================================================
# SEGMENTATION PREDICTION
# ============================================================

@torch.no_grad()
def run_segmentation(
    model,
    image,
):

    tensor = prepare_segmentation_input(
        image
    )

    tensor = tensor.to(
        DEVICE
    )

    logits = model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    prediction = torch.argmax(
        probabilities,
        dim=1,
    )

    prediction = (
        prediction[0]
        .detach()
        .cpu()
        .numpy()
    )

    probabilities = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    return (
        prediction,
        probabilities,
    )


# ============================================================
# CLASSIFICATION PREDICTION
# ============================================================

@torch.no_grad()
def run_classification(
    model,
    image,
):

    tensor = prepare_classification_input(
        image
    )

    tensor = tensor.to(
        DEVICE
    )

    logits = model(
        tensor
    )

    if logits.ndim == 2:

        logits = logits.view(
            logits.shape[0],
            NUM_TARGETS,
            NUM_CLASSES,
        )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    predictions = torch.argmax(
        probabilities,
        dim=-1,
    )

    probabilities = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    predictions = (
        predictions[0]
        .detach()
        .cpu()
        .numpy()
    )

    return (
        predictions,
        probabilities,
    )


# ============================================================
# CLASSIFICATION RESULT TABLE
# ============================================================

def create_classification_results(
    predictions,
    probabilities,
):

    rows = []

    for index, target_name in enumerate(
        TARGET_COLUMNS
    ):

        condition = CONDITIONS[
            index // 5
        ]

        level = LEVELS[
            index % 5
        ]

        predicted_class = int(
            predictions[index]
        )

        rows.append(
            {
                "target_index": index,
                "condition": condition,
                "level": level,
                "target": target_name,
                "predicted_class": predicted_class,
                "predicted_severity": CLASS_NAMES[
                    predicted_class
                ],
                "prob_normal_mild": float(
                    probabilities[index, 0]
                ),
                "prob_moderate": float(
                    probabilities[index, 1]
                ),
                "prob_severe": float(
                    probabilities[index, 2]
                ),
                "confidence": float(
                    np.max(
                        probabilities[index]
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SEGMENTATION RESULT SUMMARY
# ============================================================

def create_segmentation_summary(
    prediction,
    probabilities,
):

    total_voxels = prediction.size

    foreground = prediction != 0

    foreground_voxels = int(
        foreground.sum()
    )

    foreground_fraction = (
        foreground_voxels
        / total_voxels
    )

    class_counts = {}

    class_fractions = {}

    for class_index, class_name in enumerate(
        SEG_CLASS_NAMES
    ):

        count = int(
            np.sum(
                prediction
                == class_index
            )
        )

        fraction = (
            count
            / total_voxels
        )

        class_counts[
            class_name
        ] = count

        class_fractions[
            class_name
        ] = float(fraction)

    foreground_confidence_values = []

    for class_index in range(
        1,
        SEG_NUM_CLASSES,
    ):

        mask = (
            prediction
            == class_index
        )

        if np.any(mask):

            foreground_confidence_values.extend(
                probabilities[
                    class_index
                ][mask].tolist()
            )

    if len(
        foreground_confidence_values
    ) > 0:

        mean_foreground_confidence = float(
            np.mean(
                foreground_confidence_values
            )
        )

    else:

        mean_foreground_confidence = 0.0

    return {
        "total_voxels": int(
            total_voxels
        ),
        "foreground_voxels": foreground_voxels,
        "foreground_fraction": float(
            foreground_fraction
        ),
        "mean_foreground_confidence": (
            mean_foreground_confidence
        ),
        "class_counts": class_counts,
        "class_fractions": class_fractions,
    }


# ============================================================
# STUDY ID EXTRACTION
# ============================================================

def find_study_id(
    row,
):

    possible_columns = [
        "study_id",
        "StudyInstanceUID",
        "study",
    ]

    for column in possible_columns:

        if column in row.index:

            return str(
                row[column]
            )

    return "UNKNOWN"


# ============================================================
# SERIES / IMAGE PATH EXTRACTION
# ============================================================

def find_image_path(
    row,
):

    possible_columns = [
        "series_path",
        "image_path",
        "series_dir",
        "dicom_dir",
        "path",
    ]

    for column in possible_columns:

        if column in row.index:

            value = row[column]

            if pd.notna(value):

                path = Path(
                    str(value)
                )

                if path.exists():

                    return path

    return None


# ============================================================
# LOAD IMAGE THROUGH PART89
# ============================================================

def load_image_from_dataset(
    dataset,
    index,
):

    item = dataset[index]

    if isinstance(
        item,
        dict,
    ):

        for key in [
            "image",
            "images",
            "volume",
            "x",
        ]:

            if key in item:

                image = item[key]

                if isinstance(
                    image,
                    torch.Tensor,
                ):

                    image = (
                        image
                        .detach()
                        .cpu()
                        .numpy()
                    )

                image = np.asarray(
                    image
                )

                if image.ndim == 4:

                    image = np.squeeze(
                        image
                    )

                return image

    elif isinstance(
        item,
        (tuple, list),
    ):

        image = item[0]

        if isinstance(
            image,
            torch.Tensor,
        ):

            image = (
                image
                .detach()
                .cpu()
                .numpy()
            )

        image = np.asarray(
            image
        )

        if image.ndim == 4:

            image = np.squeeze(
                image
            )

        return image

    raise RuntimeError(
        "Could not extract image from "
        f"dataset item at index {index}."
    )


# ============================================================
# SAVE SEGMENTATION NUMPY OUTPUT
# ============================================================

def save_segmentation_output(
    study_id,
    prediction,
    probabilities,
):

    safe_id = str(
        study_id
    ).replace(
        "/",
        "_",
    ).replace(
        "\\",
        "_",
    )

    prediction_path = (
        OUTPUT_DIR
        / f"{safe_id}_segmentation.npy"
    )

    probability_path = (
        OUTPUT_DIR
        / f"{safe_id}_segmentation_probabilities.npy"
    )

    np.save(
        prediction_path,
        prediction.astype(
            np.uint8
        ),
    )

    np.save(
        probability_path,
        probabilities.astype(
            np.float16
        ),
    )

    return (
        prediction_path,
        probability_path,
    )


# ============================================================
# MAIN INTEGRATION
# ============================================================

def main():

    set_seed(
        SEED
    )

    print(
        "=" * 80
    )

    print(
        "PART 94 — INTEGRATED SWIN-UNETR + CLASSIFICATION PIPELINE"
    )

    print(
        "=" * 80
    )

    print(
        "Project root :",
        ROOT,
    )

    print(
        "Device       :",
        DEVICE,
    )

    if torch.cuda.is_available():

        print(
            "GPU          :",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    validate_required_files()

    # --------------------------------------------------------
    # IMPORT
    # --------------------------------------------------------

    print(
        "\nIMPORTING PROJECT MODULES"
    )

    (
        part11,
        part89,
        part92,
    ) = import_project_modules()

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    (
        segmentation_model,
        segmentation_checkpoint,
        segmentation_parameters,
    ) = load_segmentation_model(
        part11
    )

    (
        classification_model,
        classification_checkpoint,
        classification_parameters,
    ) = load_classification_model(
        part92
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print(
        "\nLOADING CLASSIFICATION VALIDATION MANIFEST"
    )

    validation_df = pd.read_csv(
        VALIDATION_MANIFEST
    )

    print(
        "Validation studies:",
        len(validation_df),
    )

    # --------------------------------------------------------
    # CREATE CLASSIFICATION DATASET
    # --------------------------------------------------------

    dataset_class = (
        part89.RSNALumbarClassificationDataset
    )

    try:

        dataset = dataset_class(
            manifest=validation_df
        )

    except Exception:

        try:

            dataset = dataset_class(
                VALIDATION_MANIFEST
            )

        except Exception as exc:

            raise RuntimeError(
                "Could not create Part89 dataset:\n"
                f"{exc}"
            )

    print(
        "Dataset samples:",
        len(dataset),
    )

    # --------------------------------------------------------
    # NUMBER OF STUDIES
    # --------------------------------------------------------

    num_studies = len(
        dataset
    )

    # For integration validation we process all
    # validation studies, but limit stored visual
    # artifacts.
    processed_rows = []

    classification_rows = []

    segmentation_rows = []

    # --------------------------------------------------------
    # PROCESS VALIDATION STUDIES
    # --------------------------------------------------------

    print(
        "\nSTARTING INTEGRATED INFERENCE"
    )

    print(
        "This stage performs inference only."
    )

    print(
        "No model parameters are updated."
    )

    for index in range(
        num_studies
    ):

        try:

            row = validation_df.iloc[
                index
            ]

            study_id = find_study_id(
                row
            )

            # ----------------------------------------------
            # Load image
            # ----------------------------------------------

            image = load_image_from_dataset(
                dataset,
                index,
            )

            # ----------------------------------------------
            # Segmentation
            # ----------------------------------------------

            (
                segmentation_prediction,
                segmentation_probabilities,
            ) = run_segmentation(
                segmentation_model,
                image,
            )

            segmentation_summary = (
                create_segmentation_summary(
                    segmentation_prediction,
                    segmentation_probabilities,
                )
            )

            # ----------------------------------------------
            # Classification
            # ----------------------------------------------

            (
                classification_predictions,
                classification_probabilities,
            ) = run_classification(
                classification_model,
                image,
            )

            classification_df = (
                create_classification_results(
                    classification_predictions,
                    classification_probabilities,
                )
            )

            # ----------------------------------------------
            # Save segmentation arrays
            # ----------------------------------------------

            (
                prediction_path,
                probability_path,
            ) = save_segmentation_output(
                study_id,
                segmentation_prediction,
                segmentation_probabilities,
            )

            # ----------------------------------------------
            # Classification rows
            # ----------------------------------------------

            for _, result in (
                classification_df.iterrows()
            ):

                result_dict = (
                    result.to_dict()
                )

                result_dict[
                    "study_id"
                ] = study_id

                result_dict[
                    "dataset_index"
                ] = index

                classification_rows.append(
                    result_dict
                )

            # ----------------------------------------------
            # Segmentation row
            # ----------------------------------------------

            segmentation_row = {
                "study_id": study_id,
                "dataset_index": index,
                "total_voxels": (
                    segmentation_summary[
                        "total_voxels"
                    ]
                ),
                "foreground_voxels": (
                    segmentation_summary[
                        "foreground_voxels"
                    ]
                ),
                "foreground_fraction": (
                    segmentation_summary[
                        "foreground_fraction"
                    ]
                ),
                "mean_foreground_confidence": (
                    segmentation_summary[
                        "mean_foreground_confidence"
                    ]
                ),
                "segmentation_prediction_file": (
                    str(
                        prediction_path
                    )
                ),
                "segmentation_probability_file": (
                    str(
                        probability_path
                    )
                ),
            }

            for class_name, count in (
                segmentation_summary[
                    "class_counts"
                ].items()
            ):

                safe_name = (
                    class_name
                    .lower()
                    .replace(
                        " ",
                        "_",
                    )
                    .replace(
                        "/",
                        "_",
                    )
                )

                segmentation_row[
                    f"{safe_name}_voxels"
                ] = count

            segmentation_rows.append(
                segmentation_row
            )

            # ----------------------------------------------
            # Combined study record
            # ----------------------------------------------

            processed_rows.append(
                {
                    "study_id": study_id,
                    "dataset_index": index,
                    "status": "PASS",
                    "segmentation_foreground_fraction": (
                        segmentation_summary[
                            "foreground_fraction"
                        ]
                    ),
                    "segmentation_foreground_voxels": (
                        segmentation_summary[
                            "foreground_voxels"
                        ]
                    ),
                    "classification_mean_confidence": (
                        float(
                            np.mean(
                                np.max(
                                    classification_probabilities,
                                    axis=1,
                                )
                            )
                        )
                    ),
                    "classification_mean_severe_probability": (
                        float(
                            np.mean(
                                classification_probabilities[
                                    :,
                                    2,
                                ]
                            )
                        )
                    ),
                }
            )

            # ----------------------------------------------
            # Console progress
            # ----------------------------------------------

            if (
                index + 1
            ) % 25 == 0:

                print(
                    f"Processed "
                    f"{index + 1}/"
                    f"{num_studies}"
                )

        except Exception as exc:

            study_id = (
                find_study_id(
                    validation_df.iloc[
                        index
                    ]
                )
            )

            print(
                "\nFAIL:",
                study_id,
                str(exc),
            )

            processed_rows.append(
                {
                    "study_id": study_id,
                    "dataset_index": index,
                    "status": "FAIL",
                    "error": str(exc),
                }
            )

    # --------------------------------------------------------
    # DATAFRAMES
    # --------------------------------------------------------

    processed_df = pd.DataFrame(
        processed_rows
    )

    classification_df = pd.DataFrame(
        classification_rows
    )

    segmentation_df = pd.DataFrame(
        segmentation_rows
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    integrated_csv = (
        OUTPUT_DIR
        / "part94_integrated_study_results.csv"
    )

    classification_csv = (
        OUTPUT_DIR
        / "part94_classification_predictions.csv"
    )

    segmentation_csv = (
        OUTPUT_DIR
        / "part94_segmentation_summary.csv"
    )

    processed_df.to_csv(
        integrated_csv,
        index=False,
    )

    classification_df.to_csv(
        classification_csv,
        index=False,
    )

    segmentation_df.to_csv(
        segmentation_csv,
        index=False,
    )

    # --------------------------------------------------------
    # SUCCESS STATISTICS
    # --------------------------------------------------------

    total = len(
        processed_df
    )

    successful = int(
        (
            processed_df[
                "status"
            ]
            == "PASS"
        ).sum()
    )

    failed = total - successful

    success_rate = (
        successful / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # SEGMENTATION SUMMARY
    # --------------------------------------------------------

    if len(
        segmentation_df
    ) > 0:

        mean_segmentation_foreground = float(
            segmentation_df[
                "foreground_fraction"
            ].mean()
        )

        median_segmentation_foreground = float(
            segmentation_df[
                "foreground_fraction"
            ].median()
        )

        mean_segmentation_voxels = float(
            segmentation_df[
                "foreground_voxels"
            ].mean()
        )

    else:

        mean_segmentation_foreground = 0.0
        median_segmentation_foreground = 0.0
        mean_segmentation_voxels = 0.0

    # --------------------------------------------------------
    # CLASSIFICATION SUMMARY
    # --------------------------------------------------------

    if len(
        classification_df
    ) > 0:

        predicted_distribution = (
            classification_df[
                "predicted_severity"
            ]
            .value_counts()
            .to_dict()
        )

        mean_confidence = float(
            classification_df[
                "confidence"
            ].mean()
        )

        mean_severe_probability = float(
            classification_df[
                "prob_severe"
            ].mean()
        )

    else:

        predicted_distribution = {}
        mean_confidence = 0.0
        mean_severe_probability = 0.0

    # --------------------------------------------------------
    # INTEGRATION VALIDATION
    # --------------------------------------------------------

    integration_checks = {
        "segmentation_checkpoint_exists": (
            SEGMENTATION_CHECKPOINT.exists()
        ),
        "classification_checkpoint_exists": (
            CLASSIFICATION_CHECKPOINT.exists()
        ),
        "validation_manifest_exists": (
            VALIDATION_MANIFEST.exists()
        ),
        "segmentation_model_loaded_strictly": True,
        "classification_model_loaded_strictly": True,
        "validation_studies_processed": int(
            total
        ),
        "successful_studies": int(
            successful
        ),
        "failed_studies": int(
            failed
        ),
        "success_rate": float(
            success_rate
        ),
        "segmentation_output_generated": (
            successful > 0
        ),
        "classification_output_generated": (
            successful > 0
        ),
        "end_to_end_inference_available": (
            successful > 0
        ),
    }

    if (
        successful == total
        and total > 0
    ):

        status = (
            "PASS — INTEGRATED INFERENCE PIPELINE VALIDATED"
        )

    elif successful > 0:

        status = (
            "PARTIAL — SOME STUDIES FAILED"
        )

    else:

        status = (
            "FAIL — NO STUDIES SUCCESSFULLY PROCESSED"
        )

    # --------------------------------------------------------
    # SUMMARY JSON
    # --------------------------------------------------------

    summary = {
        "part": "Part94",
        "status": status,
        "device": str(DEVICE),
        "segmentation_model": {
            "checkpoint": str(
                SEGMENTATION_CHECKPOINT
            ),
            "parameters": int(
                segmentation_parameters
            ),
            "architecture": "SwinUNETR",
            "source": "Part84",
        },
        "classification_model": {
            "checkpoint": str(
                CLASSIFICATION_CHECKPOINT
            ),
            "parameters": int(
                classification_parameters
            ),
            "architecture": "Part90CNN",
            "source": "Part92",
        },
        "validation": {
            "manifest": str(
                VALIDATION_MANIFEST
            ),
            "total_studies": int(
                total
            ),
            "successful_studies": int(
                successful
            ),
            "failed_studies": int(
                failed
            ),
            "success_rate": float(
                success_rate
            ),
        },
        "segmentation_summary": {
            "mean_foreground_fraction": (
                mean_segmentation_foreground
            ),
            "median_foreground_fraction": (
                median_segmentation_foreground
            ),
            "mean_foreground_voxels": (
                mean_segmentation_voxels
            ),
        },
        "classification_summary": {
            "predicted_distribution": (
                predicted_distribution
            ),
            "mean_confidence": (
                mean_confidence
            ),
            "mean_severe_probability": (
                mean_severe_probability
            ),
        },
        "integration_checks": integration_checks,
        "methodological_note": (
            "Part94 performs sequential inference using "
            "the frozen Part84 Swin-UNETR segmentation model "
            "and the frozen Part92 classification model. "
            "The classification model was not jointly trained "
            "with Swin-UNETR features; therefore this pipeline "
            "should be described as an integrated sequential "
            "inference framework rather than end-to-end "
            "joint training."
        ),
    }

    summary_path = (
        REPORT_DIR
        / "part94_integrated_pipeline_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "part94_integrated_pipeline_report.txt"
    )

    report_lines = []

    report_lines.append(
        "=" * 80
    )

    report_lines.append(
        "PART 94 — INTEGRATED SWIN-UNETR + CLASSIFICATION PIPELINE"
    )

    report_lines.append(
        "=" * 80
    )

    report_lines.append("")

    report_lines.append(
        "OBJECTIVE"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        "Validate sequential inference using the frozen "
        "Part84 Swin-UNETR segmentation checkpoint and "
        "the frozen Part92 classification checkpoint."
    )

    report_lines.append("")

    report_lines.append(
        "SEGMENTATION COMPONENT"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        "Architecture : SwinUNETR"
    )

    report_lines.append(
        "Source       : Part84"
    )

    report_lines.append(
        f"Parameters   : {segmentation_parameters:,}"
    )

    report_lines.append(
        f"Checkpoint   : {SEGMENTATION_CHECKPOINT}"
    )

    report_lines.append("")

    report_lines.append(
        "CLASSIFICATION COMPONENT"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        "Architecture : Part90CNN"
    )

    report_lines.append(
        "Source       : Part92"
    )

    report_lines.append(
        f"Parameters   : {classification_parameters:,}"
    )

    report_lines.append(
        f"Checkpoint   : {CLASSIFICATION_CHECKPOINT}"
    )

    report_lines.append("")

    report_lines.append(
        "VALIDATION"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        f"Total studies      : {total}"
    )

    report_lines.append(
        f"Successful studies : {successful}"
    )

    report_lines.append(
        f"Failed studies     : {failed}"
    )

    report_lines.append(
        f"Success rate       : {success_rate:.6f}"
    )

    report_lines.append("")

    report_lines.append(
        "SEGMENTATION INFERENCE SUMMARY"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        "Mean foreground fraction : "
        f"{mean_segmentation_foreground:.6f}"
    )

    report_lines.append(
        "Median foreground fraction : "
        f"{median_segmentation_foreground:.6f}"
    )

    report_lines.append(
        "Mean foreground voxels : "
        f"{mean_segmentation_voxels:.2f}"
    )

    report_lines.append("")

    report_lines.append(
        "CLASSIFICATION INFERENCE SUMMARY"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        f"Mean prediction confidence : "
        f"{mean_confidence:.6f}"
    )

    report_lines.append(
        f"Mean Severe probability : "
        f"{mean_severe_probability:.6f}"
    )

    report_lines.append(
        "Predicted distribution:"
    )

    for class_name, count in (
        predicted_distribution.items()
    ):

        report_lines.append(
            f"  {class_name}: {count}"
        )

    report_lines.append("")

    report_lines.append(
        "METHODOLOGICAL LIMITATION"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        "The segmentation and classification components "
        "were trained separately."
    )

    report_lines.append(
        "Part94 therefore validates a sequential integrated "
        "inference pipeline rather than joint end-to-end "
        "optimization of Swin-UNETR and the classifier."
    )

    report_lines.append(
        "The Part92 classifier does not consume learned "
        "Swin-UNETR feature maps."
    )

    report_lines.append(
        "This distinction should be stated clearly in the "
        "final dissertation."
    )

    report_lines.append("")

    report_lines.append(
        "FINAL STATUS"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        status
    )

    report_lines.append("")

    report_lines.append(
        "=" * 80
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "PART 94 FINAL RESULT"
    )

    print(
        "=" * 80
    )

    print(
        "Total studies      :",
        total,
    )

    print(
        "Successful studies :",
        successful,
    )

    print(
        "Failed studies     :",
        failed,
    )

    print(
        f"Success rate       : {success_rate:.4f}"
    )

    print(
        "\nSegmentation:"
    )

    print(
        f"Mean foreground fraction : "
        f"{mean_segmentation_foreground:.6f}"
    )

    print(
        f"Mean foreground voxels   : "
        f"{mean_segmentation_voxels:.2f}"
    )

    print(
        "\nClassification:"
    )

    print(
        f"Mean confidence          : "
        f"{mean_confidence:.6f}"
    )

    print(
        f"Mean Severe probability  : "
        f"{mean_severe_probability:.6f}"
    )

    print(
        "\nFINAL STATUS:"
    )

    print(
        status
    )

    print(
        "\nOUTPUTS:"
    )

    print(
        integrated_csv
    )

    print(
        classification_csv
    )

    print(
        segmentation_csv
    )

    print(
        summary_path
    )

    print(
        report_path
    )

    print(
        "\n" + "=" * 80
    )


if __name__ == "__main__":

    main()