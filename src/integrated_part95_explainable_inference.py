from pathlib import Path
import sys
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

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
    / "part95_explainable_inference"
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
# CHECKPOINTS
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


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

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

SEG_NUM_CLASSES = 6

SEG_CLASS_NAMES = [
    "Background",
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]

SEG_FULL_SHAPE = (
    64,
    96,
    96,
)

CLASS_SHAPE = (
    32,
    224,
    224,
)


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

def import_modules():

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
# CHECKPOINT HELPERS
# ============================================================

def extract_state_dict(checkpoint):

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
            "Unsupported checkpoint format."
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


def clean_state_dict(state_dict):

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
# LOAD SEGMENTATION MODEL
# ============================================================

def load_segmentation_model(
    part11,
):

    print(
        "\nLOADING PART84 SWIN-UNETR"
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
        "Strict load:",
        "PASS"
        if (
            len(result.missing_keys) == 0
            and len(result.unexpected_keys) == 0
        )
        else "FAIL",
    )

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    return model


# ============================================================
# LOAD CLASSIFICATION MODEL
# ============================================================

def load_classification_model(
    part92,
):

    print(
        "\nLOADING PART92 CLASSIFIER"
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
        "Strict load:",
        "PASS"
        if (
            len(result.missing_keys) == 0
            and len(result.unexpected_keys) == 0
        )
        else "FAIL",
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    return model


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset(
    part89,
):

    dataset_class = (
        part89.RSNALumbarClassificationDataset
    )

    manifest = pd.read_csv(
        VALIDATION_MANIFEST
    )

    attempts = [
        {
            "manifest": manifest
        },
        {
            "manifest_path": str(
                VALIDATION_MANIFEST
            )
        },
        {
            "csv_path": str(
                VALIDATION_MANIFEST
            )
        },
    ]

    last_error = None

    for kwargs in attempts:

        try:

            dataset = dataset_class(
                **kwargs
            )

            print(
                "Dataset creation: PASS"
            )

            return dataset

        except Exception as exc:

            last_error = exc

    try:

        dataset = dataset_class(
            manifest
        )

        print(
            "Dataset creation: PASS"
        )

        return dataset

    except Exception as exc:

        last_error = exc

    raise RuntimeError(
        "Could not create validation dataset:\n"
        f"{last_error}"
    )


# ============================================================
# STUDY ID
# ============================================================

def get_study_id(
    manifest,
    index,
):

    row = manifest.iloc[
        index
    ]

    if "study_id" in row.index:

        return str(
            row["study_id"]
        )

    return f"study_{index:04d}"


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def extract_image(
    item,
):

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

                while image.ndim > 3:

                    image = np.squeeze(
                        image,
                        axis=0,
                    )

                return image

    if isinstance(
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

        while image.ndim > 3:

            image = np.squeeze(
                image,
                axis=0,
            )

        return image

    raise RuntimeError(
        "Could not extract image."
    )


# ============================================================
# NORMALIZE 3D IMAGE
# ============================================================

def normalize_image(
    image,
):

    image = np.asarray(
        image
    ).astype(
        np.float32
    )

    minimum = float(
        image.min()
    )

    maximum = float(
        image.max()
    )

    if maximum > minimum:

        image = (
            image - minimum
        ) / (
            maximum - minimum
        )

    else:

        image = np.zeros_like(
            image
        )

    return image


# ============================================================
# SIMPLE 3D RESIZE
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

    # Use torch interpolation to avoid introducing
    # another dependency.

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    resized = torch.nn.functional.interpolate(
        tensor,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )

    return (
        resized[
            0,
            0,
        ]
        .numpy()
    )


# ============================================================
# SEGMENTATION INPUT
# ============================================================

def prepare_segmentation_input(
    image,
):

    image = normalize_image(
        image
    )

    image = resize_3d(
        image,
        SEG_FULL_SHAPE,
    )

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    return tensor.to(
        DEVICE
    )


# ============================================================
# CLASSIFICATION INPUT
# ============================================================

def prepare_classification_input(
    image,
):

    image = normalize_image(
        image
    )

    image = resize_3d(
        image,
        CLASS_SHAPE,
    )

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    return tensor.to(
        DEVICE
    )


# ============================================================
# SEGMENTATION INFERENCE
# ============================================================

@torch.no_grad()
def predict_segmentation(
    model,
    image,
):

    tensor = prepare_segmentation_input(
        image
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
# CLASSIFICATION INFERENCE
# ============================================================

@torch.no_grad()
def predict_classification(
    model,
    image,
):

    tensor = prepare_classification_input(
        image
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

    predictions = (
        predictions[0]
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
        predictions,
        probabilities,
    )


# ============================================================
# SEGMENTATION SUMMARY
# ============================================================

def segmentation_summary(
    prediction,
    probabilities,
):

    total = prediction.size

    foreground = prediction != 0

    foreground_voxels = int(
        foreground.sum()
    )

    foreground_fraction = (
        foreground_voxels
        / total
    )

    class_counts = {}

    for index, name in enumerate(
        SEG_CLASS_NAMES
    ):

        class_counts[name] = int(
            np.sum(
                prediction == index
            )
        )

    foreground_confidence = []

    for index in range(
        1,
        SEG_NUM_CLASSES,
    ):

        mask = (
            prediction == index
        )

        if np.any(mask):

            foreground_confidence.extend(
                probabilities[
                    index
                ][mask].tolist()
            )

    if foreground_confidence:

        mean_confidence = float(
            np.mean(
                foreground_confidence
            )
        )

    else:

        mean_confidence = 0.0

    return {
        "total_voxels": int(total),
        "foreground_voxels": foreground_voxels,
        "foreground_fraction": float(
            foreground_fraction
        ),
        "mean_foreground_confidence": (
            mean_confidence
        ),
        "class_counts": class_counts,
    }


# ============================================================
# CLASSIFICATION TABLE
# ============================================================

def classification_table(
    predictions,
    probabilities,
):

    rows = []

    for index, target in enumerate(
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
                "target": target,
                "predicted_class": (
                    predicted_class
                ),
                "predicted_severity": (
                    CLASS_NAMES[
                        predicted_class
                    ]
                ),
                "normal_mild_probability": (
                    float(
                        probabilities[
                            index,
                            0,
                        ]
                    )
                ),
                "moderate_probability": (
                    float(
                        probabilities[
                            index,
                            1,
                        ]
                    )
                ),
                "severe_probability": (
                    float(
                        probabilities[
                            index,
                            2,
                        ]
                    )
                ),
                "confidence": float(
                    np.max(
                        probabilities[
                            index
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CHOOSE REPRESENTATIVE SLICES
# ============================================================

def choose_slices(
    prediction,
    number=3,
):

    foreground_counts = np.sum(
        prediction != 0,
        axis=(1, 2),
    )

    depth = prediction.shape[0]

    # Slice with maximum foreground
    max_slice = int(
        np.argmax(
            foreground_counts
        )
    )

    # Central slice
    center_slice = depth // 2

    # Highest foreground slice
    ranked = np.argsort(
        foreground_counts
    )[::-1]

    alternative = None

    for candidate in ranked:

        if abs(
            int(candidate)
            - max_slice
        ) >= max(
            2,
            depth // 10,
        ):

            alternative = int(
                candidate
            )

            break

    if alternative is None:

        alternative = center_slice

    slices = [
        center_slice,
        max_slice,
        alternative,
    ]

    # Remove duplicates while preserving order
    unique = []

    for value in slices:

        if value not in unique:

            unique.append(
                value
            )

    return unique[:number]


# ============================================================
# GENERATE EXPLAINABILITY FIGURE
# ============================================================

def save_explainability_figure(
    study_id,
    image,
    prediction,
    probabilities,
    classification_df,
    segmentation_info,
):

    slices = choose_slices(
        prediction
    )

    safe_id = (
        str(study_id)
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
    )

    figure_path = (
        VIS_DIR
        / f"{safe_id}_explainability.png"
    )

    fig = plt.figure(
        figsize=(
            15,
            11,
        )
    )

    grid = fig.add_gridspec(
        2,
        3,
    )

    # --------------------------------------------------------
    # Image normalization
    # --------------------------------------------------------

    image_norm = normalize_image(
        image
    )

    # Resize image to segmentation geometry
    image_seg = resize_3d(
        image_norm,
        prediction.shape,
    )

    # --------------------------------------------------------
    # Three MRI/segmentation panels
    # --------------------------------------------------------

    for panel_index, slice_index in enumerate(
        slices
    ):

        ax = fig.add_subplot(
            grid[0, panel_index]
        )

        mri_slice = image_seg[
            slice_index
        ]

        mask_slice = prediction[
            slice_index
        ]

        ax.imshow(
            mri_slice,
            cmap="gray",
        )

        # Overlay segmentation using alpha.
        # The actual segmentation class IDs remain
        # available in the saved NPY output.

        overlay = np.ma.masked_where(
            mask_slice == 0,
            mask_slice,
        )

        ax.imshow(
            overlay,
            alpha=0.45,
            interpolation="nearest",
        )

        ax.set_title(
            f"MRI + Segmentation\n"
            f"Slice {slice_index}"
        )

        ax.axis(
            "off"
        )

    # --------------------------------------------------------
    # Classification summary panel
    # --------------------------------------------------------

    ax = fig.add_subplot(
        grid[1, 0]
    )

    class_counts = (
        classification_df[
            "predicted_severity"
        ]
        .value_counts()
    )

    values = [
        class_counts.get(
            name,
            0,
        )
        for name in CLASS_NAMES
    ]

    ax.bar(
        CLASS_NAMES,
        values,
    )

    ax.set_title(
        "Predicted Severity Distribution"
    )

    ax.set_ylabel(
        "Number of Targets"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    # --------------------------------------------------------
    # Probability summary panel
    # --------------------------------------------------------

    ax = fig.add_subplot(
        grid[1, 1]
    )

    mean_probabilities = [
        float(
            classification_df[
                "normal_mild_probability"
            ].mean()
        ),
        float(
            classification_df[
                "moderate_probability"
            ].mean()
        ),
        float(
            classification_df[
                "severe_probability"
            ].mean()
        ),
    ]

    ax.bar(
        CLASS_NAMES,
        mean_probabilities,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_title(
        "Mean Classification Probability"
    )

    ax.set_ylabel(
        "Probability"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    # --------------------------------------------------------
    # Study summary panel
    # --------------------------------------------------------

    ax = fig.add_subplot(
        grid[1, 2]
    )

    ax.axis(
        "off"
    )

    mean_confidence = float(
        classification_df[
            "confidence"
        ].mean()
    )

    severe_targets = int(
        (
            classification_df[
                "predicted_severity"
            ]
            == "Severe"
        ).sum()
    )

    moderate_targets = int(
        (
            classification_df[
                "predicted_severity"
            ]
            == "Moderate"
        ).sum()
    )

    normal_targets = int(
        (
            classification_df[
                "predicted_severity"
            ]
            == "Normal/Mild"
        ).sum()
    )

    text = (
        f"STUDY: {study_id}\n\n"
        f"Segmentation\n"
        f"Foreground voxels: "
        f"{segmentation_info['foreground_voxels']:,}\n"
        f"Foreground fraction: "
        f"{segmentation_info['foreground_fraction']:.4f}\n"
        f"Mean FG confidence: "
        f"{segmentation_info['mean_foreground_confidence']:.4f}\n\n"
        f"Classification\n"
        f"Normal/Mild: {normal_targets}\n"
        f"Moderate: {moderate_targets}\n"
        f"Severe: {severe_targets}\n"
        f"Mean confidence: {mean_confidence:.4f}\n\n"
        f"Interpretation:\n"
        f"Spatial segmentation output is shown above;\n"
        f"classification probabilities summarize the\n"
        f"separately trained severity classifier."
    )

    ax.text(
        0.02,
        0.98,
        text,
        verticalalignment="top",
        fontsize=10,
        transform=ax.transAxes,
    )

    fig.suptitle(
        "Explainable Lumbar Spine MRI Inference",
        fontsize=16,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    fig.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return figure_path


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    print(
        "=" * 80
    )

    print(
        "PART 95 — EXPLAINABLE INTEGRATED INFERENCE"
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
    # Required files
    # --------------------------------------------------------

    required_files = [
        SEGMENTATION_CHECKPOINT,
        CLASSIFICATION_CHECKPOINT,
        VALIDATION_MANIFEST,
    ]

    print(
        "\nCHECKING REQUIRED FILES"
    )

    for path in required_files:

        exists = path.exists()

        print(
            f"{path.name:<50}",
            "PASS"
            if exists
            else "MISSING",
        )

        if not exists:

            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    # --------------------------------------------------------
    # Import
    # --------------------------------------------------------

    print(
        "\nIMPORTING PROJECT MODULES"
    )

    (
        part11,
        part89,
        part92,
    ) = import_modules()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    segmentation_model = (
        load_segmentation_model(
            part11
        )
    )

    classification_model = (
        load_classification_model(
            part92
        )
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    print(
        "\nCREATING VALIDATION DATASET"
    )

    dataset = create_dataset(
        part89
    )

    manifest = pd.read_csv(
        VALIDATION_MANIFEST
    )

    print(
        "Validation studies:",
        len(dataset),
    )

    # --------------------------------------------------------
    # Process representative studies
    # --------------------------------------------------------

    # We create visual explanations for a representative
    # subset rather than generating 395 large PNGs.
    #
    # Every study is still passed through inference.
    # The first 10 successful studies receive visualizations.

    MAX_VISUALIZATIONS = 10

    processed_rows = []
    all_classification_rows = []
    all_segmentation_rows = []

    visualization_count = 0

    print(
        "\nSTARTING EXPLAINABLE INFERENCE"
    )

    print(
        "All validation studies are evaluated."
    )

    print(
        f"Representative visualizations: "
        f"{MAX_VISUALIZATIONS}"
    )

    for index in range(
        len(dataset)
    ):

        study_id = get_study_id(
            manifest,
            index,
        )

        try:

            item = dataset[
                index
            ]

            image = extract_image(
                item
            )

            # ----------------------------------------------
            # Segmentation
            # ----------------------------------------------

            (
                segmentation_prediction,
                segmentation_probabilities,
            ) = predict_segmentation(
                segmentation_model,
                image,
            )

            segmentation_info = (
                segmentation_summary(
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
            ) = predict_classification(
                classification_model,
                image,
            )

            classification_df = (
                classification_table(
                    classification_predictions,
                    classification_probabilities,
                )
            )

            classification_df[
                "study_id"
            ] = study_id

            classification_df[
                "dataset_index"
            ] = index

            all_classification_rows.append(
                classification_df
            )

            # ----------------------------------------------
            # Segmentation result
            # ----------------------------------------------

            segmentation_row = {
                "study_id": study_id,
                "dataset_index": index,
                "foreground_voxels": (
                    segmentation_info[
                        "foreground_voxels"
                    ]
                ),
                "foreground_fraction": (
                    segmentation_info[
                        "foreground_fraction"
                    ]
                ),
                "mean_foreground_confidence": (
                    segmentation_info[
                        "mean_foreground_confidence"
                    ]
                ),
            }

            for (
                class_name,
                count,
            ) in segmentation_info[
                "class_counts"
            ].items():

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

            all_segmentation_rows.append(
                segmentation_row
            )

            # ----------------------------------------------
            # Combined study result
            # ----------------------------------------------

            mean_confidence = float(
                classification_df[
                    "confidence"
                ].mean()
            )

            mean_severe_probability = float(
                classification_df[
                    "severe_probability"
                ].mean()
            )

            processed_rows.append(
                {
                    "study_id": study_id,
                    "dataset_index": index,
                    "status": "PASS",
                    "segmentation_foreground_voxels": (
                        segmentation_info[
                            "foreground_voxels"
                        ]
                    ),
                    "segmentation_foreground_fraction": (
                        segmentation_info[
                            "foreground_fraction"
                        ]
                    ),
                    "segmentation_mean_foreground_confidence": (
                        segmentation_info[
                            "mean_foreground_confidence"
                        ]
                    ),
                    "classification_mean_confidence": (
                        mean_confidence
                    ),
                    "classification_mean_severe_probability": (
                        mean_severe_probability
                    ),
                }
            )

            # ----------------------------------------------
            # Representative visualization
            # ----------------------------------------------

            if (
                visualization_count
                < MAX_VISUALIZATIONS
            ):

                figure_path = (
                    save_explainability_figure(
                        study_id,
                        image,
                        segmentation_prediction,
                        segmentation_probabilities,
                        classification_df,
                        segmentation_info,
                    )
                )

                processed_rows[-1][
                    "explainability_figure"
                ] = str(
                    figure_path
                )

                visualization_count += 1

            # ----------------------------------------------
            # Progress
            # ----------------------------------------------

            if (
                index + 1
            ) % 25 == 0:

                print(
                    f"Processed "
                    f"{index + 1}/"
                    f"{len(dataset)}"
                )

        except Exception as exc:

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
    # DataFrames
    # --------------------------------------------------------

    integrated_df = pd.DataFrame(
        processed_rows
    )

    classification_results_df = (
        pd.concat(
            all_classification_rows,
            ignore_index=True,
        )
        if all_classification_rows
        else pd.DataFrame()
    )

    segmentation_results_df = (
        pd.DataFrame(
            all_segmentation_rows
        )
        if all_segmentation_rows
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    integrated_path = (
        OUTPUT_DIR
        / "part95_integrated_explainable_results.csv"
    )

    classification_path = (
        OUTPUT_DIR
        / "part95_classification_probabilities.csv"
    )

    segmentation_path = (
        OUTPUT_DIR
        / "part95_segmentation_evidence.csv"
    )

    integrated_df.to_csv(
        integrated_path,
        index=False,
    )

    classification_results_df.to_csv(
        classification_path,
        index=False,
    )

    segmentation_results_df.to_csv(
        segmentation_path,
        index=False,
    )

    # --------------------------------------------------------
    # Success statistics
    # --------------------------------------------------------

    total = len(
        processed_rows
    )

    successful = int(
        (
            integrated_df[
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
    # Aggregate statistics
    # --------------------------------------------------------

    if len(
        segmentation_results_df
    ) > 0:

        mean_foreground_fraction = float(
            segmentation_results_df[
                "foreground_fraction"
            ].mean()
        )

        mean_foreground_voxels = float(
            segmentation_results_df[
                "foreground_voxels"
            ].mean()
        )

    else:

        mean_foreground_fraction = 0.0
        mean_foreground_voxels = 0.0

    if len(
        classification_results_df
    ) > 0:

        mean_confidence = float(
            classification_results_df[
                "confidence"
            ].mean()
        )

        mean_severe_probability = float(
            classification_results_df[
                "severe_probability"
            ].mean()
        )

        predicted_distribution = (
            classification_results_df[
                "predicted_severity"
            ]
            .value_counts()
            .to_dict()
        )

    else:

        mean_confidence = 0.0
        mean_severe_probability = 0.0
        predicted_distribution = {}

    # --------------------------------------------------------
    # Methodological status
    # --------------------------------------------------------

    status = (
        "PASS — EXPLAINABLE INFERENCE VALIDATED"
        if (
            successful == total
            and total > 0
        )
        else
        "PARTIAL — SOME STUDIES FAILED"
        if successful > 0
        else
        "FAIL — NO SUCCESSFUL INFERENCE"
    )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {
        "part": "Part95",
        "status": status,
        "device": str(DEVICE),
        "purpose": (
            "Generate spatial and probabilistic "
            "explanations for the integrated inference "
            "pipeline."
        ),
        "validation": {
            "total_studies": int(total),
            "successful_studies": int(successful),
            "failed_studies": int(failed),
            "success_rate": float(success_rate),
        },
        "segmentation": {
            "source_checkpoint": str(
                SEGMENTATION_CHECKPOINT
            ),
            "architecture": "SwinUNETR",
            "mean_foreground_voxels": (
                mean_foreground_voxels
            ),
            "mean_foreground_fraction": (
                mean_foreground_fraction
            ),
        },
        "classification": {
            "source_checkpoint": str(
                CLASSIFICATION_CHECKPOINT
            ),
            "architecture": "Part90CNN",
            "mean_confidence": (
                mean_confidence
            ),
            "mean_severe_probability": (
                mean_severe_probability
            ),
            "predicted_distribution": (
                predicted_distribution
            ),
        },
        "visualizations": {
            "generated": int(
                visualization_count
            ),
            "directory": str(
                VIS_DIR
            ),
        },
        "methodological_note": (
            "Part95 provides spatial evidence through "
            "the Swin-UNETR predicted segmentation mask "
            "and probabilistic evidence through the "
            "separately trained classification model. "
            "It should not be described as Grad-CAM, SHAP, "
            "or another post-hoc feature-attribution method."
        ),
    }

    summary_path = (
        REPORT_DIR
        / "part95_explainable_inference_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "part95_explainable_inference_report.txt"
    )

    report_lines = []

    report_lines.append(
        "=" * 80
    )

    report_lines.append(
        "PART 95 — EXPLAINABLE INTEGRATED INFERENCE"
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
        "Generate interpretable spatial and probabilistic "
        "outputs from the integrated lumbar-spine MRI "
        "inference pipeline."
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
        f"Success rate       : {success_rate:.4f}"
    )

    report_lines.append("")

    report_lines.append(
        "SEGMENTATION EVIDENCE"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        f"Mean foreground voxels : "
        f"{mean_foreground_voxels:.2f}"
    )

    report_lines.append(
        f"Mean foreground fraction : "
        f"{mean_foreground_fraction:.6f}"
    )

    report_lines.append("")

    report_lines.append(
        "CLASSIFICATION EVIDENCE"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        f"Mean classification confidence : "
        f"{mean_confidence:.6f}"
    )

    report_lines.append(
        f"Mean Severe probability : "
        f"{mean_severe_probability:.6f}"
    )

    report_lines.append(
        "Predicted severity distribution:"
    )

    for name, count in (
        predicted_distribution.items()
    ):

        report_lines.append(
            f"  {name}: {count}"
        )

    report_lines.append("")

    report_lines.append(
        "EXPLAINABILITY APPROACH"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        "1. MRI slices provide the original image context."
    )

    report_lines.append(
        "2. Swin-UNETR segmentation provides spatial "
        "disease-region evidence."
    )

    report_lines.append(
        "3. Classification probabilities provide "
        "severity prediction evidence."
    )

    report_lines.append(
        "4. Confidence values quantify the classifier's "
        "probability concentration."
    )

    report_lines.append("")

    report_lines.append(
        "IMPORTANT LIMITATION"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        "This is spatial/output-level explainability."
    )

    report_lines.append(
        "The analysis does not claim Grad-CAM, SHAP, "
        "integrated gradients, or feature-level attribution."
    )

    report_lines.append(
        "The segmentation and classification components "
        "were trained separately."
    )

    report_lines.append(
        "Therefore the segmentation mask should be "
        "interpreted as spatial evidence rather than "
        "proof that the classifier directly used those "
        "specific voxels."
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
    # Console summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "PART 95 FINAL RESULT"
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
        f"Success rate       : "
        f"{success_rate:.4f}"
    )

    print(
        "\nExplainability:"
    )

    print(
        "Visualizations     :",
        visualization_count,
    )

    print(
        f"Mean FG fraction   : "
        f"{mean_foreground_fraction:.6f}"
    )

    print(
        f"Mean classifier confidence : "
        f"{mean_confidence:.6f}"
    )

    print(
        f"Mean Severe probability : "
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
        integrated_path
    )

    print(
        classification_path
    )

    print(
        segmentation_path
    )

    print(
        summary_path
    )

    print(
        report_path
    )

    print(
        "\nVISUALIZATIONS:"
    )

    print(
        VIS_DIR
    )

    print(
        "\n" + "=" * 80
    )


if __name__ == "__main__":

    main()