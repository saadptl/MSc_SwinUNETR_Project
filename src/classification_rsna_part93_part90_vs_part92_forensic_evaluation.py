from pathlib import Path
import sys
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part93_part90_vs_part92_forensic_evaluation"
)

REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

PART87_VAL = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part87_study_level_split"
    / "part87_validation_manifest.csv"
)

PART90_CHECKPOINT = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part90_controlled_baseline"
    / "checkpoints"
    / "part90_best_model.pth"
)

PART92_CHECKPOINT = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part92_class_imbalance_aware_training"
    / "checkpoints"
    / "part92_best_model.pth"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 1
NUM_WORKERS = 0

TARGET_DEPTH = 32
TARGET_HEIGHT = 224
TARGET_WIDTH = 224

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

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
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
# IMPORT ACTUAL PART89 DATASET
# ============================================================

def import_part89():

    sys.path.insert(0, str(SRC_DIR))

    import classification_rsna_part89_dataset_pipeline as part89

    return part89


# ============================================================
# IMPORT ACTUAL PART90 / PART92 MODEL CLASSES
# ============================================================

def import_training_modules():

    sys.path.insert(0, str(SRC_DIR))

    import classification_rsna_part90_controlled_baseline_training as part90
    import classification_rsna_part92_class_imbalance_aware_training as part92

    return part90, part92


# ============================================================
# CHECKPOINT INFORMATION
# ============================================================

def inspect_checkpoint(path):

    print("\n" + "=" * 72)
    print("CHECKPOINT")
    print("=" * 72)

    print("Path :", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

    print("Exists :", True)
    print(
        "Size   :",
        f"{path.stat().st_size:,}",
        "bytes",
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        print(
            "Keys   :",
            list(checkpoint.keys()),
        )

        if "epoch" in checkpoint:
            print(
                "Epoch  :",
                checkpoint["epoch"],
            )

        if "part" in checkpoint:
            print(
                "Part   :",
                checkpoint["part"],
            )

        if "validation_metrics" in checkpoint:
            print(
                "Validation metrics available:",
                True,
            )

    return checkpoint


# ============================================================
# STATE DICT EXTRACTION
# ============================================================

def extract_state_dict(checkpoint):

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unsupported checkpoint type: "
            f"{type(checkpoint)}"
        )

    for key in [
        "model_state_dict",
        "state_dict",
    ]:

        if key in checkpoint:

            value = checkpoint[key]

            if isinstance(value, dict):
                return value

    # Some checkpoints may directly be state_dicts
    tensor_values = [
        value
        for value in checkpoint.values()
        if isinstance(value, torch.Tensor)
    ]

    if len(tensor_values) > 0:
        return checkpoint

    raise RuntimeError(
        "Could not locate model_state_dict."
    )


# ============================================================
# STATE DICT CLEANING
# ============================================================

def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned[new_key] = value

    return cleaned


# ============================================================
# ACTUAL MODEL LOADING
# ============================================================

def load_actual_model(
    checkpoint_path,
    model_class,
    model_name,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # Full serialized model
    if isinstance(checkpoint, nn.Module):

        model = checkpoint.to(DEVICE)
        model.eval()

        return model

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    print("\n" + "-" * 72)
    print("LOADING", model_name)
    print("-" * 72)

    print(
        "Using exact training-script class:",
        model_class.__name__,
    )

    # --------------------------------------------------------
    # Try normal constructor
    # --------------------------------------------------------

    constructor_attempts = [
        {
            "num_targets": NUM_TARGETS,
            "num_classes": NUM_CLASSES,
        },
        {},
    ]

    model = None
    last_error = None

    for kwargs in constructor_attempts:

        try:

            model = model_class(
                **kwargs
            )

            break

        except Exception as exc:

            last_error = exc

    if model is None:

        raise RuntimeError(
            f"Could not construct {model_class.__name__}.\n"
            f"Last error: {last_error}"
        )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Strict load
    # --------------------------------------------------------

    try:

        result = model.load_state_dict(
            state_dict,
            strict=True,
        )

    except Exception as exc:

        print(
            "\nSTRICT LOAD FAILED."
        )

        print(
            "This means the constructor arguments or "
            "architecture differ from the saved checkpoint."
        )

        raise RuntimeError(
            f"Could not strictly load {model_name} checkpoint:\n"
            f"{exc}"
        )

    print(
        "Strict load       : PASS"
    )

    print(
        "Missing keys      :",
        len(result.missing_keys),
    )

    print(
        "Unexpected keys   :",
        len(result.unexpected_keys),
    )

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters  :",
        f"{parameter_count:,}",
    )

    return model, checkpoint, parameter_count


# ============================================================
# DATASET CLASS
# ============================================================

def create_validation_dataset(part89):

    if not hasattr(
        part89,
        "RSNALumbarClassificationDataset",
    ):

        raise RuntimeError(
            "Part89 does not contain "
            "RSNALumbarClassificationDataset."
        )

    dataset_class = (
        part89.RSNALumbarClassificationDataset
    )

    print(
        "\nDataset class:",
        dataset_class.__name__,
    )

    manifest = pd.read_csv(
        PART87_VAL
    )

    print(
        "Validation manifest rows:",
        len(manifest),
    )

    # --------------------------------------------------------
    # Try the constructor forms used by Part89.
    # --------------------------------------------------------

    attempts = [
        (
            "manifest_path + dimensions",
            {
                "manifest_path": str(PART87_VAL),
                "target_depth": TARGET_DEPTH,
                "target_height": TARGET_HEIGHT,
                "target_width": TARGET_WIDTH,
            },
        ),
        (
            "csv_path + dimensions",
            {
                "csv_path": str(PART87_VAL),
                "target_depth": TARGET_DEPTH,
                "target_height": TARGET_HEIGHT,
                "target_width": TARGET_WIDTH,
            },
        ),
        (
            "manifest dataframe + dimensions",
            {
                "manifest": manifest,
                "target_depth": TARGET_DEPTH,
                "target_height": TARGET_HEIGHT,
                "target_width": TARGET_WIDTH,
            },
        ),
        (
            "dataframe + dimensions",
            {
                "df": manifest,
                "target_depth": TARGET_DEPTH,
                "target_height": TARGET_HEIGHT,
                "target_width": TARGET_WIDTH,
            },
        ),
        (
            "manifest path only",
            {
                "manifest_path": str(PART87_VAL),
            },
        ),
        (
            "csv path only",
            {
                "csv_path": str(PART87_VAL),
            },
        ),
        (
            "manifest only",
            {
                "manifest": manifest,
            },
        ),
    ]

    last_error = None

    for description, kwargs in attempts:

        try:

            dataset = dataset_class(
                **kwargs
            )

            print(
                "Dataset constructor PASS:",
                description,
            )

            return dataset

        except Exception as exc:

            last_error = exc

    # Final positional fallback
    try:

        dataset = dataset_class(
            manifest
        )

        print(
            "Dataset constructor PASS: "
            "positional manifest"
        )

        return dataset

    except Exception as exc:

        last_error = exc

    raise RuntimeError(
        "Could not construct "
        "RSNALumbarClassificationDataset.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# BATCH UNPACKING
# ============================================================

def unpack_batch(batch):

    if isinstance(batch, dict):

        image = None
        labels = None
        mask = None

        for key in [
            "image",
            "images",
            "volume",
            "x",
        ]:

            if key in batch:
                image = batch[key]
                break

        for key in [
            "label",
            "labels",
            "target",
            "targets",
            "y",
        ]:

            if key in batch:
                labels = batch[key]
                break

        for key in [
            "mask",
            "label_mask",
            "target_mask",
            "valid_mask",
        ]:

            if key in batch:
                mask = batch[key]
                break

        if image is None:
            raise RuntimeError(
                "Image not found in batch keys: "
                f"{list(batch.keys())}"
            )

        if labels is None:
            raise RuntimeError(
                "Labels not found in batch keys: "
                f"{list(batch.keys())}"
            )

        return image, labels, mask

    if isinstance(batch, (tuple, list)):

        if len(batch) >= 3:
            return (
                batch[0],
                batch[1],
                batch[2],
            )

        if len(batch) == 2:
            return (
                batch[0],
                batch[1],
                None,
            )

    raise RuntimeError(
        "Unsupported batch structure: "
        f"{type(batch)}"
    )


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_labels(labels):

    if isinstance(labels, torch.Tensor):

        labels = labels.detach().cpu().numpy()

    else:

        labels = np.asarray(labels)

    # One-hot -> integer
    if (
        labels.ndim == 3
        and labels.shape[-1] == NUM_CLASSES
    ):

        labels = np.argmax(
            labels,
            axis=-1,
        )

    if labels.ndim == 1:

        labels = labels.reshape(
            1,
            -1,
        )

    labels = labels.astype(
        np.int64
    )

    return labels


# ============================================================
# MASK NORMALIZATION
# ============================================================

def normalize_mask(mask, labels):

    if mask is None:

        return labels >= 0

    if isinstance(mask, torch.Tensor):

        mask = mask.detach().cpu().numpy()

    else:

        mask = np.asarray(mask)

    if mask.ndim == 1:

        mask = mask.reshape(
            1,
            -1,
        )

    return mask.astype(bool)


# ============================================================
# LOGIT NORMALIZATION
# ============================================================

def normalize_logits(logits):

    if logits.ndim == 2:

        if (
            logits.shape[1]
            != NUM_TARGETS * NUM_CLASSES
        ):

            raise RuntimeError(
                "Unexpected flattened logits shape: "
                f"{tuple(logits.shape)}"
            )

        logits = logits.view(
            logits.shape[0],
            NUM_TARGETS,
            NUM_CLASSES,
        )

    if logits.ndim != 3:

        raise RuntimeError(
            "Unexpected logits shape: "
            f"{tuple(logits.shape)}"
        )

    return logits


# ============================================================
# METRICS
# ============================================================

def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return np.nan

    return numerator / denominator


def calculate_metrics(
    y_true,
    y_pred,
    valid_mask,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid_mask = np.asarray(
        valid_mask
    ).astype(bool)

    true_flat = y_true[
        valid_mask
    ]

    pred_flat = y_pred[
        valid_mask
    ]

    if len(true_flat) == 0:

        raise RuntimeError(
            "No valid labels available."
        )

    confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    for true_value, pred_value in zip(
        true_flat,
        pred_flat,
    ):

        if (
            0 <= true_value < NUM_CLASSES
            and 0 <= pred_value < NUM_CLASSES
        ):

            confusion[
                int(true_value),
                int(pred_value),
            ] += 1

    total = confusion.sum()

    accuracy = (
        np.trace(confusion) / total
        if total > 0
        else np.nan
    )

    true_counts = confusion.sum(
        axis=1
    )

    majority_class = int(
        np.argmax(true_counts)
    )

    majority_baseline = (
        true_counts[majority_class]
        / total
        if total > 0
        else np.nan
    )

    precision = []
    recall = []
    f1 = []

    for class_index in range(
        NUM_CLASSES
    ):

        tp = confusion[
            class_index,
            class_index,
        ]

        fp = (
            confusion[
                :,
                class_index,
            ].sum()
            - tp
        )

        fn = (
            confusion[
                class_index,
                :,
            ].sum()
            - tp
        )

        p = safe_divide(
            tp,
            tp + fp,
        )

        r = safe_divide(
            tp,
            tp + fn,
        )

        if (
            np.isnan(p)
            or np.isnan(r)
            or p + r == 0
        ):

            f = np.nan

        else:

            f = (
                2 * p * r
                / (p + r)
            )

        precision.append(p)
        recall.append(r)
        f1.append(f)

    balanced_accuracy = np.nanmean(
        recall
    )

    macro_precision = np.nanmean(
        precision
    )

    macro_recall = np.nanmean(
        recall
    )

    macro_f1 = np.nanmean(
        f1
    )

    predicted_counts = np.bincount(
        pred_flat,
        minlength=NUM_CLASSES,
    )

    predicted_fraction = (
        predicted_counts
        / len(pred_flat)
    )

    return {
        "valid_labels": int(total),
        "accuracy": float(accuracy),
        "majority_class": majority_class,
        "majority_baseline": float(
            majority_baseline
        ),
        "accuracy_gain_over_majority": float(
            accuracy - majority_baseline
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "class_precision": [
            float(x)
            if not np.isnan(x)
            else None
            for x in precision
        ],
        "class_recall": [
            float(x)
            if not np.isnan(x)
            else None
            for x in recall
        ],
        "class_f1": [
            float(x)
            if not np.isnan(x)
            else None
            for x in f1
        ],
        "confusion_matrix": (
            confusion.tolist()
        ),
        "predicted_counts": (
            predicted_counts.tolist()
        ),
        "predicted_fraction": (
            predicted_fraction.tolist()
        ),
        "true_counts": (
            true_counts.tolist()
        ),
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    model_name,
):

    model.eval()

    all_true = []
    all_pred = []
    all_mask = []

    print("\n" + "=" * 72)
    print("EVALUATING", model_name)
    print("=" * 72)

    for batch_index, batch in enumerate(
        loader
    ):

        images, labels, mask = (
            unpack_batch(batch)
        )

        if not isinstance(
            images,
            torch.Tensor,
        ):

            images = torch.as_tensor(
                images
            )

        if not isinstance(
            labels,
            torch.Tensor,
        ):

            labels = torch.as_tensor(
                labels
            )

        images = images.float().to(
            DEVICE
        )

        logits = model(images)

        logits = normalize_logits(
            logits
        )

        predictions = torch.argmax(
            logits,
            dim=-1,
        )

        labels_np = normalize_labels(
            labels
        )

        mask_np = normalize_mask(
            mask,
            labels_np,
        )

        pred_np = (
            predictions
            .detach()
            .cpu()
            .numpy()
        )

        if labels_np.shape != pred_np.shape:

            raise RuntimeError(
                "Label/prediction shape mismatch: "
                f"{labels_np.shape} vs "
                f"{pred_np.shape}"
            )

        all_true.append(
            labels_np
        )

        all_pred.append(
            pred_np
        )

        all_mask.append(
            mask_np
        )

        if (
            batch_index + 1
        ) % 50 == 0:

            print(
                f"Processed "
                f"{batch_index + 1}/"
                f"{len(loader)}"
            )

    y_true = np.concatenate(
        all_true,
        axis=0,
    )

    y_pred = np.concatenate(
        all_pred,
        axis=0,
    )

    valid_mask = np.concatenate(
        all_mask,
        axis=0,
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        valid_mask,
    )

    metrics["model_name"] = model_name

    return (
        metrics,
        y_true,
        y_pred,
        valid_mask,
    )


# ============================================================
# PER-TARGET METRICS
# ============================================================

def per_target_metrics(
    y_true,
    y_pred,
    valid_mask,
    model_name,
):

    rows = []

    for index, target_name in enumerate(
        TARGET_COLUMNS
    ):

        metrics = calculate_metrics(
            y_true[:, index],
            y_pred[:, index],
            valid_mask[:, index],
        )

        condition = CONDITIONS[
            index // 5
        ]

        level = LEVELS[
            index % 5
        ]

        rows.append(
            {
                "model": model_name,
                "target_index": index,
                "condition": condition,
                "level": level,
                "target": target_name,
                "valid_labels": metrics[
                    "valid_labels"
                ],
                "accuracy": metrics[
                    "accuracy"
                ],
                "majority_baseline": metrics[
                    "majority_baseline"
                ],
                "accuracy_gain_over_majority": (
                    metrics[
                        "accuracy_gain_over_majority"
                    ]
                ),
                "balanced_accuracy": metrics[
                    "balanced_accuracy"
                ],
                "macro_precision": metrics[
                    "macro_precision"
                ],
                "macro_recall": metrics[
                    "macro_recall"
                ],
                "macro_f1": metrics[
                    "macro_f1"
                ],
                "normal_mild_recall": metrics[
                    "class_recall"
                ][0],
                "moderate_recall": metrics[
                    "class_recall"
                ][1],
                "severe_recall": metrics[
                    "class_recall"
                ][2],
                "normal_mild_f1": metrics[
                    "class_f1"
                ][0],
                "moderate_f1": metrics[
                    "class_f1"
                ][1],
                "severe_f1": metrics[
                    "class_f1"
                ][2],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CONFUSION MATRIX TABLE
# ============================================================

def confusion_matrix_rows(
    y_true,
    y_pred,
    valid_mask,
    model_name,
):

    rows = []

    for target_index, target_name in enumerate(
        TARGET_COLUMNS
    ):

        yt = y_true[
            :,
            target_index,
        ][
            valid_mask[
                :,
                target_index,
            ]
        ]

        yp = y_pred[
            :,
            target_index,
        ][
            valid_mask[
                :,
                target_index,
            ]
        ]

        cm = np.zeros(
            (
                NUM_CLASSES,
                NUM_CLASSES,
            ),
            dtype=np.int64,
        )

        for t, p in zip(
            yt,
            yp,
        ):

            cm[
                int(t),
                int(p),
            ] += 1

        for true_class in range(
            NUM_CLASSES
        ):

            for predicted_class in range(
                NUM_CLASSES
            ):

                rows.append(
                    {
                        "model": model_name,
                        "target_index": target_index,
                        "target": target_name,
                        "true_class": CLASS_NAMES[
                            true_class
                        ],
                        "predicted_class": CLASS_NAMES[
                            predicted_class
                        ],
                        "count": int(
                            cm[
                                true_class,
                                predicted_class,
                            ]
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

def create_comparison(
    metrics90,
    metrics92,
):

    rows = []

    metric_pairs = [
        (
            "Accuracy",
            metrics90["accuracy"],
            metrics92["accuracy"],
        ),
        (
            "Majority baseline",
            metrics90["majority_baseline"],
            metrics92["majority_baseline"],
        ),
        (
            "Accuracy gain over majority",
            metrics90[
                "accuracy_gain_over_majority"
            ],
            metrics92[
                "accuracy_gain_over_majority"
            ],
        ),
        (
            "Balanced accuracy",
            metrics90["balanced_accuracy"],
            metrics92["balanced_accuracy"],
        ),
        (
            "Macro precision",
            metrics90["macro_precision"],
            metrics92["macro_precision"],
        ),
        (
            "Macro recall",
            metrics90["macro_recall"],
            metrics92["macro_recall"],
        ),
        (
            "Macro F1",
            metrics90["macro_f1"],
            metrics92["macro_f1"],
        ),
        (
            "Normal/Mild recall",
            metrics90["class_recall"][0],
            metrics92["class_recall"][0],
        ),
        (
            "Moderate recall",
            metrics90["class_recall"][1],
            metrics92["class_recall"][1],
        ),
        (
            "Severe recall",
            metrics90["class_recall"][2],
            metrics92["class_recall"][2],
        ),
    ]

    for name, value90, value92 in (
        metric_pairs
    ):

        rows.append(
            {
                "metric": name,
                "part90": value90,
                "part92": value92,
                "delta_part92_minus_part90": (
                    value92 - value90
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DIAGNOSIS
# ============================================================

def determine_diagnosis(
    metrics90,
    metrics92,
):

    f1_delta = (
        metrics92["macro_f1"]
        - metrics90["macro_f1"]
    )

    balanced_delta = (
        metrics92["balanced_accuracy"]
        - metrics90["balanced_accuracy"]
    )

    moderate_delta = (
        metrics92["class_recall"][1]
        - metrics90["class_recall"][1]
    )

    severe_delta = (
        metrics92["class_recall"][2]
        - metrics90["class_recall"][2]
    )

    if (
        f1_delta > 0.03
        and balanced_delta > 0.02
        and moderate_delta > 0.02
    ):

        diagnosis = (
            "PART92_REDUCES_MAJORITY_CLASS_DOMINANCE"
        )

    elif (
        f1_delta > 0.01
        or balanced_delta > 0.01
    ):

        diagnosis = (
            "PART92_SHOWS_NONTRIVIAL_MINOR_CLASS_SIGNAL"
        )

    else:

        diagnosis = (
            "PART92_DOES_NOT_SHOW_CLEAR_OVERALL_GAIN"
        )

    if severe_delta > 0:

        severity_note = (
            "Severe recall improved relative to Part90."
        )

    else:

        severity_note = (
            "Severe recall did not improve; "
            "severe-class detection remains weak."
        )

    pred92_normal_fraction = (
        metrics92["predicted_fraction"][0]
    )

    if pred92_normal_fraction >= 0.90:

        distribution_note = (
            "Part92 remains strongly concentrated "
            "on Normal/Mild predictions."
        )

    else:

        distribution_note = (
            "Part92 prediction distribution is "
            "not overwhelmingly concentrated "
            "on Normal/Mild."
        )

    return (
        diagnosis,
        severity_note,
        distribution_note,
    )


# ============================================================
# TEXT REPORT
# ============================================================

def write_report(
    metrics90,
    metrics92,
    comparison_df,
    diagnosis,
    severity_note,
    distribution_note,
    part90_parameters,
    part92_parameters,
):

    report_path = (
        REPORT_DIR
        / "part93_part90_vs_part92_forensic_report.txt"
    )

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "PART 93 — PART90 VS PART92 CLASSIFICATION FORENSIC EVALUATION"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "OBJECTIVE"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "Evaluate the saved Part90 and Part92 checkpoints "
        "on the identical Part87 study-level validation set."
    )

    lines.append("")

    lines.append(
        "ARCHITECTURE VALIDITY NOTE"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Part90 model parameters: "
        f"{part90_parameters:,}"
    )

    lines.append(
        f"Part92 model parameters: "
        f"{part92_parameters:,}"
    )

    lines.append(
        "Part90 uses the actual Lightweight3DCNN class "
        "from the Part90 training script."
    )

    lines.append(
        "Part92 uses the actual Part90CNN class "
        "from the Part92 training script."
    )

    lines.append(
        "Because the architectures differ, Part92 must "
        "NOT be described as a pure class-weight-only "
        "ablation of Part90."
    )

    lines.append("")

    lines.append(
        "PART90 RESULTS"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Accuracy              : "
        f"{metrics90['accuracy']:.6f}"
    )

    lines.append(
        f"Majority baseline     : "
        f"{metrics90['majority_baseline']:.6f}"
    )

    lines.append(
        f"Accuracy gain         : "
        f"{metrics90['accuracy_gain_over_majority']:.6f}"
    )

    lines.append(
        f"Balanced accuracy     : "
        f"{metrics90['balanced_accuracy']:.6f}"
    )

    lines.append(
        f"Macro precision       : "
        f"{metrics90['macro_precision']:.6f}"
    )

    lines.append(
        f"Macro recall          : "
        f"{metrics90['macro_recall']:.6f}"
    )

    lines.append(
        f"Macro F1              : "
        f"{metrics90['macro_f1']:.6f}"
    )

    lines.append(
        f"Normal/Mild recall    : "
        f"{metrics90['class_recall'][0]:.6f}"
    )

    lines.append(
        f"Moderate recall       : "
        f"{metrics90['class_recall'][1]:.6f}"
    )

    lines.append(
        f"Severe recall         : "
        f"{metrics90['class_recall'][2]:.6f}"
    )

    lines.append("")

    lines.append(
        "PART92 RESULTS"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Accuracy              : "
        f"{metrics92['accuracy']:.6f}"
    )

    lines.append(
        f"Majority baseline     : "
        f"{metrics92['majority_baseline']:.6f}"
    )

    lines.append(
        f"Accuracy gain         : "
        f"{metrics92['accuracy_gain_over_majority']:.6f}"
    )

    lines.append(
        f"Balanced accuracy     : "
        f"{metrics92['balanced_accuracy']:.6f}"
    )

    lines.append(
        f"Macro precision       : "
        f"{metrics92['macro_precision']:.6f}"
    )

    lines.append(
        f"Macro recall          : "
        f"{metrics92['macro_recall']:.6f}"
    )

    lines.append(
        f"Macro F1              : "
        f"{metrics92['macro_f1']:.6f}"
    )

    lines.append(
        f"Normal/Mild recall    : "
        f"{metrics92['class_recall'][0]:.6f}"
    )

    lines.append(
        f"Moderate recall       : "
        f"{metrics92['class_recall'][1]:.6f}"
    )

    lines.append(
        f"Severe recall         : "
        f"{metrics92['class_recall'][2]:.6f}"
    )

    lines.append("")

    lines.append(
        "COMPARISON"
    )

    lines.append(
        "-" * 80
    )

    for _, row in comparison_df.iterrows():

        lines.append(
            f"{row['metric']:<32}"
            f" Part90={row['part90']:.6f}"
            f" Part92={row['part92']:.6f}"
            f" Delta={row['delta_part92_minus_part90']:+.6f}"
        )

    lines.append("")

    lines.append(
        "PREDICTED CLASS DISTRIBUTION"
    )

    lines.append(
        "-" * 80
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        lines.append(
            f"{class_name:<15}"
            f" Part90={metrics90['predicted_fraction'][index]:.6f}"
            f" Part92={metrics92['predicted_fraction'][index]:.6f}"
        )

    lines.append("")

    lines.append(
        "FINAL DIAGNOSIS"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        diagnosis
    )

    lines.append(
        severity_note
    )

    lines.append(
        distribution_note
    )

    lines.append("")

    lines.append(
        "INTERPRETATION"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        "The RSNA classification targets are strongly "
        "class-imbalanced, therefore raw accuracy alone "
        "is insufficient."
    )

    lines.append(
        "Macro F1, balanced accuracy, and minority-class "
        "recall are emphasized."
    )

    lines.append(
        "A reduction in raw accuracy can be acceptable "
        "if it corresponds to meaningful gains in "
        "Moderate and Severe detection."
    )

    lines.append(
        "Because Part90 and Part92 use different model "
        "architectures, their difference cannot be "
        "attributed exclusively to class weighting."
    )

    lines.append("")

    lines.append(
        "=" * 80
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print(
        "=" * 80
    )

    print(
        "PART 93 — PART90 VS PART92 FORENSIC EVALUATION"
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

    required_files = [
        PART87_VAL,
        PART90_CHECKPOINT,
        PART92_CHECKPOINT,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # INSPECT CHECKPOINTS
    # --------------------------------------------------------

    checkpoint90 = inspect_checkpoint(
        PART90_CHECKPOINT
    )

    checkpoint92 = inspect_checkpoint(
        PART92_CHECKPOINT
    )

    # --------------------------------------------------------
    # IMPORT MODULES
    # --------------------------------------------------------

    print(
        "\nIMPORTING ACTUAL TRAINING MODULES"
    )

    part89 = import_part89()

    part90_module, part92_module = (
        import_training_modules()
    )

    print(
        "Part89 dataset:",
        part89.RSNALumbarClassificationDataset.__name__,
    )

    print(
        "Part90 model:",
        part90_module.Lightweight3DCNN.__name__,
    )

    print(
        "Part92 model:",
        part92_module.Part90CNN.__name__,
    )

    # --------------------------------------------------------
    # CREATE DATASET
    # --------------------------------------------------------

    print(
        "\nBUILDING VALIDATION DATASET"
    )

    val_dataset = create_validation_dataset(
        part89
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    print(
        "Validation batches:",
        len(val_loader),
    )

    # --------------------------------------------------------
    # LOAD PART90 EXACT MODEL
    # --------------------------------------------------------

    (
        part90_model,
        _,
        part90_parameters,
    ) = load_actual_model(
        PART90_CHECKPOINT,
        part90_module.Lightweight3DCNN,
        "Part90",
    )

    # --------------------------------------------------------
    # LOAD PART92 EXACT MODEL
    # --------------------------------------------------------

    (
        part92_model,
        _,
        part92_parameters,
    ) = load_actual_model(
        PART92_CHECKPOINT,
        part92_module.Part90CNN,
        "Part92",
    )

    # --------------------------------------------------------
    # EVALUATE PART90
    # --------------------------------------------------------

    (
        metrics90,
        y_true90,
        y_pred90,
        mask90,
    ) = evaluate_model(
        part90_model,
        val_loader,
        "Part90",
    )

    # --------------------------------------------------------
    # EVALUATE PART92
    # --------------------------------------------------------

    (
        metrics92,
        y_true92,
        y_pred92,
        mask92,
    ) = evaluate_model(
        part92_model,
        val_loader,
        "Part92",
    )

    # --------------------------------------------------------
    # VERIFY IDENTICAL VALIDATION LABELS
    # --------------------------------------------------------

    if not np.array_equal(
        y_true90,
        y_true92,
    ):

        raise RuntimeError(
            "Part90 and Part92 did not receive identical labels."
        )

    if not np.array_equal(
        mask90,
        mask92,
    ):

        raise RuntimeError(
            "Part90 and Part92 did not receive identical "
            "valid-label masks."
        )

    print(
        "\nVALIDATION LABEL CONSISTENCY: PASS"
    )

    # --------------------------------------------------------
    # PER TARGET
    # --------------------------------------------------------

    per90 = per_target_metrics(
        y_true90,
        y_pred90,
        mask90,
        "Part90",
    )

    per92 = per_target_metrics(
        y_true92,
        y_pred92,
        mask92,
        "Part92",
    )

    per_target_df = pd.concat(
        [
            per90,
            per92,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    cm90 = confusion_matrix_rows(
        y_true90,
        y_pred90,
        mask90,
        "Part90",
    )

    cm92 = confusion_matrix_rows(
        y_true92,
        y_pred92,
        mask92,
        "Part92",
    )

    confusion_df = pd.concat(
        [
            cm90,
            cm92,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    comparison_df = create_comparison(
        metrics90,
        metrics92,
    )

    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    (
        diagnosis,
        severity_note,
        distribution_note,
    ) = determine_diagnosis(
        metrics90,
        metrics92,
    )

    # --------------------------------------------------------
    # SAVE CSV FILES
    # --------------------------------------------------------

    per_target_path = (
        OUTPUT_DIR
        / "part93_per_target_metrics.csv"
    )

    confusion_path = (
        OUTPUT_DIR
        / "part93_confusion_matrices.csv"
    )

    comparison_path = (
        OUTPUT_DIR
        / "part93_part90_vs_part92_comparison.csv"
    )

    per_target_df.to_csv(
        per_target_path,
        index=False,
    )

    confusion_df.to_csv(
        confusion_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE SUMMARY JSON
    # --------------------------------------------------------

    summary = {
        "part": "Part93",
        "status": "PASS",
        "purpose": (
            "Forensic comparison of Part90 baseline "
            "and Part92 class-imbalance-aware checkpoint."
        ),
        "device": str(DEVICE),
        "validation_samples": len(
            val_dataset
        ),
        "validation_batches": len(
            val_loader
        ),
        "part90_model_class": (
            "Lightweight3DCNN"
        ),
        "part92_model_class": (
            "Part90CNN"
        ),
        "part90_parameters": int(
            part90_parameters
        ),
        "part92_parameters": int(
            part92_parameters
        ),
        "architecture_difference": True,
        "part90_metrics": metrics90,
        "part92_metrics": metrics92,
        "diagnosis": diagnosis,
        "severity_note": severity_note,
        "distribution_note": distribution_note,
    }

    summary_path = (
        REPORT_DIR
        / "part93_part90_vs_part92_forensic_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # WRITE REPORT
    # --------------------------------------------------------

    report_path = write_report(
        metrics90,
        metrics92,
        comparison_df,
        diagnosis,
        severity_note,
        distribution_note,
        part90_parameters,
        part92_parameters,
    )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "PART 93 FINAL RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        "\nPART90"
    )

    print(
        f"Accuracy          : "
        f"{metrics90['accuracy']:.6f}"
    )

    print(
        f"Majority baseline : "
        f"{metrics90['majority_baseline']:.6f}"
    )

    print(
        f"Balanced accuracy : "
        f"{metrics90['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro F1          : "
        f"{metrics90['macro_f1']:.6f}"
    )

    print(
        f"Moderate recall   : "
        f"{metrics90['class_recall'][1]:.6f}"
    )

    print(
        f"Severe recall     : "
        f"{metrics90['class_recall'][2]:.6f}"
    )

    print(
        "\nPART92"
    )

    print(
        f"Accuracy          : "
        f"{metrics92['accuracy']:.6f}"
    )

    print(
        f"Majority baseline : "
        f"{metrics92['majority_baseline']:.6f}"
    )

    print(
        f"Balanced accuracy : "
        f"{metrics92['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro F1          : "
        f"{metrics92['macro_f1']:.6f}"
    )

    print(
        f"Moderate recall   : "
        f"{metrics92['class_recall'][1]:.6f}"
    )

    print(
        f"Severe recall     : "
        f"{metrics92['class_recall'][2]:.6f}"
    )

    print(
        "\nMODEL PARAMETERS"
    )

    print(
        f"Part90: {part90_parameters:,}"
    )

    print(
        f"Part92: {part92_parameters:,}"
    )

    print(
        "\nDIAGNOSIS"
    )

    print(
        diagnosis
    )

    print(
        severity_note
    )

    print(
        distribution_note
    )

    print(
        "\nOUTPUTS"
    )

    print(
        per_target_path
    )

    print(
        confusion_path
    )

    print(
        comparison_path
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

    print(
        "STATUS: PASS — PART93 FORENSIC EVALUATION COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()