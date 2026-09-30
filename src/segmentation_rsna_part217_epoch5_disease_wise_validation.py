"""
Part 2.17
Epoch-5 Disease-Wise Point-Supervised Validation

Purpose:
    Evaluate Part 2.16 Epoch-5 checkpoint on the SAME 25
    study-disjoint validation series.

Important:
    RSNA annotations are point/localization annotations.
    This script does NOT create voxel-wise ground truth masks
    and does NOT report voxel-wise Dice.

Outputs:
    outputs/segmentation/
        rsna_part217_epoch5_disease_wise_validation/
            metrics/
            visualizations/
            reports/
            part217_summary.json
            part217_point_results.csv
            part217_disease_metrics.csv
            part217_level_metrics.csv
            part217_confusion_matrix.csv
            part217_case_metrics.csv
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
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part213_point_supervision_manifest"
    / "part213_point_manifest.csv"
)

CHECKPOINT_PATH = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part216_point_supervised_balanced_training"
    / "checkpoints"
    / "part216_epoch_05.pth"
)

PART216_METRICS_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part216_point_supervised_balanced_training"
    / "metrics"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part217_epoch5_disease_wise_validation"
)

METRICS_DIR = OUTPUT_DIR / "metrics"
VIS_DIR = OUTPUT_DIR / "visualizations"
REPORT_DIR = OUTPUT_DIR / "reports"

for directory in [
    OUTPUT_DIR,
    METRICS_DIR,
    VIS_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

SEED = 42

NUM_VALIDATION_CASES = 25

MODEL_SHAPE = (64, 96, 96)

NUM_CLASSES = 6

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

DISEASE_CLASS_IDS = [1, 2, 3, 4, 5]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------
# IMPORT PROJECT MODULES
# ---------------------------------------------------------------------

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

APP_DIR = ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from segmentation_rsna_part11_controlled_pilot_training_corrected import (  # noqa
    read_dicom_series_robust,
    resolve_series_dir,
)


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

from monai.networks.nets import SwinUNETR


def build_model():
    """
    Build the exact Swin-UNETR architecture used by the project.
    """

    model = SwinUNETR(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=12,
        use_checkpoint=False,
    )

    return model


def clean_state_dict(state_dict):
    """
    Remove common wrapper prefixes.
    """

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        for prefix in [
            "module.",
            "model.",
            "network.",
        ]:

            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        cleaned[new_key] = value

    return cleaned


def load_checkpoint(model):
    """
    Load Part 2.16 Epoch-5 checkpoint.
    """

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    state_dict = clean_state_dict(state_dict)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"Checkpoint loaded: {CHECKPOINT_PATH}"
    )

    print(
        f"Missing keys: {len(missing)}"
    )

    print(
        f"Unexpected keys: {len(unexpected)}"
    )

    if len(missing) > 0:
        print("WARNING: missing model keys detected.")

    if len(unexpected) > 0:
        print("WARNING: unexpected model keys detected.")

    return model


# ---------------------------------------------------------------------
# VALIDATION SERIES SELECTION
# ---------------------------------------------------------------------

def load_fixed_validation_series():
    """
    Reconstruct the same deterministic validation pool.

    We use study-disjoint splitting:
        80% train
        20% validation

    Then select the first deterministic 25 validation studies
    using the same seed.
    """

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    df["study_id"] = df["study_id"].astype(str)
    df["series_id"] = df["series_id"].astype(str)

    studies = sorted(
        df["study_id"].unique()
    )

    rng = random.Random(SEED)

    studies = studies.copy()
    rng.shuffle(studies)

    split_index = int(
        len(studies) * 0.80
    )

    train_studies = set(
        studies[:split_index]
    )

    validation_studies = set(
        studies[split_index:]
    )

    validation_df = df[
        df["study_id"].isin(validation_studies)
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
            ["study_id", "series_id"]
        )
    )

    validation_series = validation_series.head(
        NUM_VALIDATION_CASES
    )

    print(
        f"Validation studies available: "
        f"{len(validation_studies)}"
    )

    print(
        f"Validation series available: "
        f"{len(validation_series)}"
    )

    return (
        df,
        train_studies,
        validation_studies,
        validation_series,
    )


# ---------------------------------------------------------------------
# IMAGE RESIZING
# ---------------------------------------------------------------------

def resize_volume_nearest(volume, target_shape):
    """
    Nearest-neighbour 3D resize for visualization geometry.
    """

    from scipy.ndimage import zoom

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    factors = [
        target_shape[i] / volume.shape[i]
        for i in range(3)
    ]

    return zoom(
        volume,
        factors,
        order=1,
    )


# ---------------------------------------------------------------------
# IMAGE NORMALIZATION
# ---------------------------------------------------------------------

def normalize_volume(volume):
    """
    Percentile normalization.
    """

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    finite = np.isfinite(volume)

    if not finite.any():
        return np.zeros_like(volume)

    values = volume[finite]

    p1 = np.percentile(values, 1)
    p99 = np.percentile(values, 99)

    if p99 <= p1:
        return np.zeros_like(volume)

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


# ---------------------------------------------------------------------
# SERIES LOADING
# ---------------------------------------------------------------------

def load_series_tensor(study_id, series_id):
    """
    Load one native DICOM series and convert to the model grid.
    """

    row = pd.Series(
        {
            "study_id": study_id,
            "series_id": series_id,
        }
    )

    series_dir = resolve_series_dir(row)

    image_native, dicom_records, info = (
        read_dicom_series_robust(
            series_dir
        )
    )

    image_native = np.asarray(
        image_native,
        dtype=np.float32,
    )

    image_native = normalize_volume(
        image_native
    )

    image_model = resize_volume_nearest(
        image_native,
        MODEL_SHAPE,
    )

    tensor = torch.from_numpy(
        image_model
    ).float()

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return (
        tensor,
        image_native,
        dicom_records,
        info,
    )


# ---------------------------------------------------------------------
# POINT EXTRACTION
# ---------------------------------------------------------------------

def get_case_points(
    manifest_df,
    study_id,
    series_id,
):
    """
    Return all annotation points for one series.
    """

    rows = manifest_df[
        (manifest_df["study_id"] == str(study_id))
        &
        (manifest_df["series_id"] == str(series_id))
    ].copy()

    points = []

    for _, row in rows.iterrows():

        points.append(
            {
                "class_id": int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "level": str(row["level"]),
                "z": float(row["model_z_float"]),
                "y": float(row["model_y_float"]),
                "x": float(row["model_x_float"]),
            }
        )

    return points


# ---------------------------------------------------------------------
# POINT PREDICTION
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_volume(model, image_tensor):
    """
    Full-volume inference.

    No segmentation ground truth is assumed.
    """

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.amp.autocast(
        "cuda",
        enabled=DEVICE.type == "cuda",
    ):

        logits = model(
            image_tensor
        )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    return (
        logits.detach().cpu(),
        probabilities.detach().cpu(),
    )


# ---------------------------------------------------------------------
# POINT METRICS
# ---------------------------------------------------------------------

def evaluate_points(
    probabilities,
    points,
):
    """
    Evaluate class prediction at every annotation point.
    """

    probability_volume = probabilities[
        0
    ].numpy()

    rows = []

    for point in points:

        class_id = int(
            point["class_id"]
        )

        z = int(
            np.clip(
                round(point["z"]),
                0,
                MODEL_SHAPE[0] - 1,
            )
        )

        y = int(
            np.clip(
                round(point["y"]),
                0,
                MODEL_SHAPE[1] - 1,
            )
        )

        x = int(
            np.clip(
                round(point["x"]),
                0,
                MODEL_SHAPE[2] - 1,
            )
        )

        point_probabilities = (
            probability_volume[
                :,
                z,
                y,
                x,
            ]
        )

        predicted_class = int(
            np.argmax(
                point_probabilities
            )
        )

        correct_probability = float(
            point_probabilities[
                class_id
            ]
        )

        hit = (
            correct_probability >= 0.50
        )

        correct = (
            predicted_class == class_id
        )

        rows.append(
            {
                "true_class_id": class_id,
                "true_class_name": CLASS_NAMES[
                    class_id
                ],
                "predicted_class_id": predicted_class,
                "predicted_class_name": CLASS_NAMES[
                    predicted_class
                ],
                "level": point["level"],
                "z": z,
                "y": y,
                "x": x,
                "correct_probability": correct_probability,
                "correct": int(correct),
                "hit_at_0_50": int(hit),
            }
        )

    return rows


# ---------------------------------------------------------------------
# LOCALIZATION METRIC
# ---------------------------------------------------------------------

def calculate_localization_distance(
    probabilities,
    point,
):
    """
    Distance from annotation point to the nearest predicted
    voxel of the annotated disease class.

    This is a localization metric, NOT voxel-wise segmentation
    accuracy.
    """

    class_id = int(
        point["class_id"]
    )

    probability_volume = probabilities[
        0,
        class_id,
    ].numpy()

    predicted_mask = (
        probability_volume >= 0.50
    )

    if not predicted_mask.any():
        return math.nan

    coordinates = np.argwhere(
        predicted_mask
    )

    target = np.array(
        [
            point["z"],
            point["y"],
            point["x"],
        ],
        dtype=np.float32,
    )

    distances = np.sqrt(
        np.sum(
            (
                coordinates.astype(
                    np.float32
                )
                - target
            )
            ** 2,
            axis=1,
        )
    )

    return float(
        distances.min()
    )


# ---------------------------------------------------------------------
# QUALITATIVE VISUALIZATION
# ---------------------------------------------------------------------

def save_case_visualization(
    image_tensor,
    probabilities,
    points,
    study_id,
    series_id,
):
    """
    Save central-slice qualitative visualization.

    Red/yellow visualization is probability-based and is NOT
    described as a confirmed disease segmentation.
    """

    image = image_tensor[
        0,
        0,
    ].numpy()

    probability_volume = probabilities[
        0
    ].numpy()

    center_z = image.shape[0] // 2

    image_slice = image[
        center_z
    ]

    foreground_probability = (
        1.0
        - probability_volume[
            0,
            center_z,
        ]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    axes[0].imshow(
        image_slice,
        cmap="gray",
    )

    axes[0].set_title(
        f"MRI\nStudy {study_id}"
    )

    axes[0].axis("off")

    axes[1].imshow(
        image_slice,
        cmap="gray",
    )

    axes[1].imshow(
        foreground_probability,
        cmap="hot",
        alpha=0.35,
        vmin=0,
        vmax=1,
    )

    for point in points:

        z = int(
            round(point["z"])
        )

        if z != center_z:
            continue

        axes[1].scatter(
            point["x"],
            point["y"],
            s=20,
            marker="x",
        )

    axes[1].set_title(
        "Foreground Probability + Points"
    )

    axes[1].axis("off")

    plt.tight_layout()

    filename = (
        f"study_{study_id}_"
        f"series_{series_id}.png"
    )

    plt.savefig(
        VIS_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("PART 2.17")
    print("EPOCH-5 DISEASE-WISE POINT-SUPERVISED VALIDATION")
    print("=" * 80)

    print()
    print(f"Project root: {ROOT}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------
    # LOAD MANIFEST / VALIDATION SPLIT
    # -------------------------------------------------------------

    (
        manifest_df,
        train_studies,
        validation_studies,
        validation_series,
    ) = load_fixed_validation_series()

    print(
        f"Manifest rows: {len(manifest_df)}"
    )

    print(
        f"Annotated series: "
        f"{manifest_df[['study_id','series_id']].drop_duplicates().shape[0]}"
    )

    print(
        f"Validation cases selected: "
        f"{len(validation_series)}"
    )

    overlap = (
        set(validation_series["study_id"])
        &
        set(train_studies)
    )

    print(
        f"Study overlap with training: "
        f"{len(overlap)}"
    )

    if len(overlap) != 0:
        raise RuntimeError(
            "Study leakage detected."
        )

    # -------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------

    model = build_model()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    model = load_checkpoint(
        model
    )

    model.to(DEVICE)
    model.eval()

    # -------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------

    all_point_rows = []
    all_case_rows = []

    processed_cases = 0

    for case_index, (_, series_row) in enumerate(
        validation_series.iterrows(),
        start=1,
    ):

        study_id = str(
            series_row["study_id"]
        )

        series_id = str(
            series_row["series_id"]
        )

        print()
        print(
            f"[{case_index}/{len(validation_series)}] "
            f"Study={study_id} "
            f"Series={series_id}"
        )

        try:

            image_tensor, image_native, records, info = (
                load_series_tensor(
                    study_id,
                    series_id,
                )
            )

            points = get_case_points(
                manifest_df,
                study_id,
                series_id,
            )

            if len(points) == 0:
                print(
                    "  No annotation points. Skipping."
                )
                continue

            logits, probabilities = (
                predict_volume(
                    model,
                    image_tensor,
                )
            )

            point_rows = evaluate_points(
                probabilities,
                points,
            )

            for row, point in zip(
                point_rows,
                points,
            ):

                distance = (
                    calculate_localization_distance(
                        probabilities,
                        point,
                    )
                )

                row["study_id"] = study_id
                row["series_id"] = series_id
                row["localization_distance"] = (
                    distance
                )

                all_point_rows.append(
                    row
                )

            # -----------------------------------------------------
            # CASE-LEVEL FOREGROUND VOLUME
            # -----------------------------------------------------

            probability_array = (
                probabilities[
                    0
                ].numpy()
            )

            predicted_labels = (
                np.argmax(
                    probability_array,
                    axis=0,
                )
            )

            foreground_voxels = int(
                np.count_nonzero(
                    predicted_labels > 0
                )
            )

            total_voxels = int(
                np.prod(
                    MODEL_SHAPE
                )
            )

            foreground_ratio = (
                foreground_voxels
                /
                total_voxels
            )

            all_case_rows.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "num_points": len(points),
                    "predicted_foreground_voxels":
                        foreground_voxels,
                    "total_model_voxels":
                        total_voxels,
                    "predicted_foreground_ratio":
                        foreground_ratio,
                }
            )

            save_case_visualization(
                image_tensor=image_tensor,
                probabilities=probabilities,
                points=points,
                study_id=study_id,
                series_id=series_id,
            )

            processed_cases += 1

            print(
                f"  Points: {len(points)}"
            )

            print(
                f"  Predicted foreground voxels: "
                f"{foreground_voxels:,}"
            )

            print(
                f"  Foreground ratio: "
                f"{foreground_ratio:.6f}"
            )

        except Exception as exc:

            print(
                f"  ERROR: {type(exc).__name__}: {exc}"
            )

    # -------------------------------------------------------------
    # DATAFRAME
    # -------------------------------------------------------------

    points_df = pd.DataFrame(
        all_point_rows
    )

    case_df = pd.DataFrame(
        all_case_rows
    )

    if points_df.empty:
        raise RuntimeError(
            "No point predictions were produced."
        )

    # -------------------------------------------------------------
    # OVERALL METRICS
    # -------------------------------------------------------------

    overall_accuracy = float(
        points_df["correct"].mean()
    )

    overall_probability = float(
        points_df[
            "correct_probability"
        ].mean()
    )

    hit_rate = float(
        points_df[
            "hit_at_0_50"
        ].mean()
    )

    finite_distances = points_df[
        "localization_distance"
    ].dropna()

    if len(finite_distances) > 0:

        mean_distance = float(
            finite_distances.mean()
        )

        median_distance = float(
            finite_distances.median()
        )

    else:

        mean_distance = None
        median_distance = None

    # -------------------------------------------------------------
    # DISEASE METRICS
    # -------------------------------------------------------------

    disease_rows = []

    for class_id in DISEASE_CLASS_IDS:

        subset = points_df[
            points_df["true_class_id"]
            == class_id
        ]

        if len(subset) == 0:
            continue

        disease_rows.append(
            {
                "class_id": class_id,
                "disease": CLASS_NAMES[
                    class_id
                ],
                "points": len(subset),
                "accuracy": float(
                    subset["correct"].mean()
                ),
                "mean_correct_probability":
                    float(
                        subset[
                            "correct_probability"
                        ].mean()
                    ),
                "hit_rate_at_0_50":
                    float(
                        subset[
                            "hit_at_0_50"
                        ].mean()
                    ),
                "mean_localization_distance":
                    float(
                        subset[
                            "localization_distance"
                        ].dropna().mean()
                    )
                    if subset[
                        "localization_distance"
                    ].notna().any()
                    else None,
            }
        )

    disease_df = pd.DataFrame(
        disease_rows
    )

    macro_disease_accuracy = float(
        disease_df["accuracy"].mean()
    )

    # -------------------------------------------------------------
    # LEVEL METRICS
    # -------------------------------------------------------------

    level_df = (
        points_df
        .groupby("level")
        .agg(
            points=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_probability=(
                "correct_probability",
                "mean",
            ),
            hit_rate_at_0_50=(
                "hit_at_0_50",
                "mean",
            ),
        )
        .reset_index()
    )

    # -------------------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------------------

    confusion = pd.crosstab(
        points_df["true_class_name"],
        points_df["predicted_class_name"],
        rownames=["True"],
        colnames=["Predicted"],
        dropna=False,
    )

    for class_name in CLASS_NAMES.values():

        if class_name not in confusion.columns:
            confusion[class_name] = 0

    confusion = confusion.reindex(
        index=list(CLASS_NAMES.values()),
        columns=list(CLASS_NAMES.values()),
        fill_value=0,
    )

    # -------------------------------------------------------------
    # FOREGROUND METRICS
    # -------------------------------------------------------------

    mean_foreground_ratio = float(
        case_df[
            "predicted_foreground_ratio"
        ].mean()
    )

    median_foreground_ratio = float(
        case_df[
            "predicted_foreground_ratio"
        ].median()
    )

    mean_foreground_voxels = float(
        case_df[
            "predicted_foreground_voxels"
        ].mean()
    )

    # -------------------------------------------------------------
    # SAVE CSV FILES
    # -------------------------------------------------------------

    points_df.to_csv(
        OUTPUT_DIR
        / "part217_point_results.csv",
        index=False,
    )

    disease_df.to_csv(
        OUTPUT_DIR
        / "part217_disease_metrics.csv",
        index=False,
    )

    level_df.to_csv(
        OUTPUT_DIR
        / "part217_level_metrics.csv",
        index=False,
    )

    confusion.to_csv(
        OUTPUT_DIR
        / "part217_confusion_matrix.csv"
    )

    case_df.to_csv(
        OUTPUT_DIR
        / "part217_case_metrics.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # SUMMARY JSON
    # -------------------------------------------------------------

    summary = {
        "part": "2.17",
        "status": "COMPLETE",
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "validation_cases_requested":
            NUM_VALIDATION_CASES,
        "validation_cases_processed":
            processed_cases,
        "study_disjoint":
            len(overlap) == 0,
        "model_shape":
            list(MODEL_SHAPE),
        "overall_point_accuracy":
            overall_accuracy,
        "macro_disease_accuracy":
            macro_disease_accuracy,
        "mean_correct_class_probability":
            overall_probability,
        "hit_rate_at_0_50":
            hit_rate,
        "mean_localization_distance":
            mean_distance,
        "median_localization_distance":
            median_distance,
        "mean_predicted_foreground_voxels":
            mean_foreground_voxels,
        "mean_predicted_foreground_ratio":
            mean_foreground_ratio,
        "median_predicted_foreground_ratio":
            median_foreground_ratio,
        "manual_voxel_ground_truth":
            False,
        "voxelwise_dice_reported":
            False,
        "dashboard_integrated":
            False,
        "scientific_limitation":
            "RSNA annotations are point/localization "
            "annotations rather than manual voxel-wise "
            "segmentation masks.",
    }

    with open(
        OUTPUT_DIR / "part217_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # TEXT REPORT
    # -------------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "part217_epoch5_validation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PART 2.17 - EPOCH-5 DISEASE-WISE VALIDATION\n"
        )
        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Checkpoint:\n{CHECKPOINT_PATH}\n\n"
        )

        f.write(
            f"Validation cases processed: "
            f"{processed_cases}\n"
        )

        f.write(
            f"Study-disjoint: {len(overlap) == 0}\n\n"
        )

        f.write(
            "OVERALL METRICS\n"
        )
        f.write(
            "-" * 50 + "\n"
        )

        f.write(
            f"Point accuracy: "
            f"{overall_accuracy:.6f}\n"
        )

        f.write(
            f"Macro disease accuracy: "
            f"{macro_disease_accuracy:.6f}\n"
        )

        f.write(
            f"Mean correct-class probability: "
            f"{overall_probability:.6f}\n"
        )

        f.write(
            f"Hit rate >= 0.50: "
            f"{hit_rate:.6f}\n"
        )

        f.write(
            f"Mean localization distance: "
            f"{mean_distance}\n"
        )

        f.write(
            f"Median localization distance: "
            f"{median_distance}\n\n"
        )

        f.write(
            "DISEASE-WISE RESULTS\n"
        )
        f.write(
            "-" * 50 + "\n"
        )

        f.write(
            disease_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "LEVEL-WISE RESULTS\n"
        )
        f.write(
            "-" * 50 + "\n"
        )

        f.write(
            level_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "CONFUSION MATRIX\n"
        )
        f.write(
            "-" * 50 + "\n"
        )

        f.write(
            confusion.to_string()
        )

        f.write("\n\n")

        f.write(
            "FOREGROUND VOLUME\n"
        )
        f.write(
            "-" * 50 + "\n"
        )

        f.write(
            f"Mean foreground voxels: "
            f"{mean_foreground_voxels:.2f}\n"
        )

        f.write(
            f"Mean foreground ratio: "
            f"{mean_foreground_ratio:.6f}\n"
        )

        f.write(
            f"Median foreground ratio: "
            f"{median_foreground_ratio:.6f}\n\n"
        )

        f.write(
            "SCIENTIFIC LIMITATION\n"
        )
        f.write(
            "-" * 50 + "\n"
        )

        f.write(
            "The RSNA dataset provides point/localization "
            "annotations rather than manual voxel-wise "
            "segmentation masks. Therefore these results "
            "must not be interpreted as clinical voxel-wise "
            "segmentation accuracy or Dice performance.\n"
        )

        f.write(
            "The dashboard was not modified or integrated "
            "during Part 2.17.\n"
        )

    # -------------------------------------------------------------
    # FINAL TERMINAL OUTPUT
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("PART 2.17 COMPLETE")
    print("=" * 80)

    print(
        f"Processed cases: {processed_cases}"
    )

    print(
        f"Point accuracy: "
        f"{overall_accuracy:.6f}"
    )

    print(
        f"Macro disease accuracy: "
        f"{macro_disease_accuracy:.6f}"
    )

    print(
        f"Mean correct probability: "
        f"{overall_probability:.6f}"
    )

    print(
        f"Hit rate >= 0.50: "
        f"{hit_rate:.6f}"
    )

    print(
        f"Mean localization distance: "
        f"{mean_distance}"
    )

    print()
    print("DISEASE-WISE RESULTS")
    print(
        disease_df.to_string(
            index=False
        )
    )

    print()
    print("LEVEL-WISE RESULTS")
    print(
        level_df.to_string(
            index=False
        )
    )

    print()
    print("CONFUSION MATRIX")
    print(
        confusion.to_string()
    )

    print()
    print(
        f"Mean foreground ratio: "
        f"{mean_foreground_ratio:.6f}"
    )

    print()
    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()