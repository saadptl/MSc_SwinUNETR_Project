"""
PART 2.30 — LFNN/RFNN LOCAL ANATOMICAL CONTEXT AUDIT

Analysis-only experiment.

Uses:
    - Part 2.20B geometry-corrected loading
    - Part 2.27 best checkpoint
    - Same 25 validation cases
    - Same 90 LFNN/RFNN points

Measures:
    - local MRI intensity statistics
    - local gradient magnitude
    - model class probabilities
    - foreground ratio
    - physical/canonical coordinates

Does NOT:
    - train
    - modify checkpoints
    - modify Part 2.20B
    - modify Part104
    - modify dashboard
    - create voxel ground truth
"""

from pathlib import Path
import sys
import json
import math

import numpy as np
import pandas as pd
import torch


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src import (
    segmentation_rsna_part220b_geometry_corrected_training as p220b
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part230_local_context_audit"
)

REPORT_DIR = OUTPUT_DIR / "reports"
TABLE_DIR = OUTPUT_DIR / "tables"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
    / "part227_best_macro_disease.pth"
)


LFNN = 2
RFNN = 3

CLASS_NAMES = p220b.CLASS_NAMES


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def load_checkpoint(model):

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=p220b.DEVICE,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print(
        f"Checkpoint missing keys: {len(missing)}"
    )

    print(
        f"Checkpoint unexpected keys: {len(unexpected)}"
    )


def build_validation_groups(manifest):

    validation_series = (
        p220b.select_validation_series(
            manifest
        )
    )

    if isinstance(
        validation_series,
        pd.DataFrame,
    ):

        validation_ids = (
            validation_series[
                [
                    "study_id",
                    "series_id",
                ]
            ]
            .drop_duplicates()
            .copy()
        )

    else:

        validation_ids = []

        for item in validation_series:

            if isinstance(
                item,
                dict,
            ):

                validation_ids.append(
                    (
                        str(
                            item["study_id"]
                        ),
                        str(
                            item["series_id"]
                        ),
                    )
                )

            elif (
                isinstance(
                    item,
                    (
                        tuple,
                        list,
                    ),
                )
                and len(item) >= 2
            ):

                validation_ids.append(
                    (
                        str(item[0]),
                        str(item[1]),
                    )
                )

        validation_ids = pd.DataFrame(
            validation_ids,
            columns=[
                "study_id",
                "series_id",
            ],
        )

    manifest_copy = manifest.copy()

    for column in [
        "study_id",
        "series_id",
    ]:

        manifest_copy[column] = (
            manifest_copy[column]
            .astype(str)
        )

        validation_ids[column] = (
            validation_ids[column]
            .astype(str)
        )

    validation_manifest = (
        manifest_copy.merge(
            validation_ids[
                [
                    "study_id",
                    "series_id",
                ]
            ].drop_duplicates(),
            on=[
                "study_id",
                "series_id",
            ],
            how="inner",
        )
    )

    required = [
        "study_id",
        "series_id",
        "native_z",
        "native_y",
        "native_x",
        "class_id",
        "class_name",
        "level",
    ]

    missing = [
        c
        for c in required
        if c not in validation_manifest.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {c}"
                for c in missing
            )
        )

    groups = []

    for (
        study_id,
        series_id,
    ), group in validation_manifest.groupby(
        [
            "study_id",
            "series_id",
        ],
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


@torch.no_grad()
def predict_volume(
    model,
    image,
):

    model.eval()

    tensor = torch.from_numpy(
        image.astype(np.float32)
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(
            p220b.DEVICE,
            non_blocking=True,
        )
    )

    with torch.amp.autocast(
        "cuda",
        enabled=torch.cuda.is_available(),
    ):

        logits = model(
            tensor
        )

    if isinstance(
        logits,
        (tuple, list),
    ):
        logits = logits[0]

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    probabilities = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    prediction = np.argmax(
        probabilities,
        axis=0,
    )

    return (
        probabilities,
        prediction,
    )


# ============================================================
# LOCAL STATISTICS
# ============================================================

def local_statistics(
    image,
    probabilities,
    prediction,
    point,
    radius,
):

    depth, height, width = (
        image.shape
    )

    z = int(
        np.clip(
            round(
                float(
                    point["z"]
                )
            ),
            0,
            depth - 1,
        )
    )

    y = int(
        np.clip(
            round(
                float(
                    point["y"]
                )
            ),
            0,
            height - 1,
        )
    )

    x = int(
        np.clip(
            round(
                float(
                    point["x"]
                )
            ),
            0,
            width - 1,
        )
    )

    z0 = max(
        0,
        z - radius,
    )

    z1 = min(
        depth,
        z + radius + 1,
    )

    y0 = max(
        0,
        y - radius,
    )

    y1 = min(
        height,
        y + radius + 1,
    )

    x0 = max(
        0,
        x - radius,
    )

    x1 = min(
        width,
        x + radius + 1,
    )

    local_image = image[
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    local_prediction = prediction[
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    local_probabilities = probabilities[
        :,
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    # --------------------------------------------------------
    # Gradient magnitude
    # --------------------------------------------------------

    gradients = np.gradient(
        local_image.astype(
            np.float32
        )
    )

    gradient_magnitude = np.sqrt(
        gradients[0] ** 2
        + gradients[1] ** 2
        + gradients[2] ** 2
    )

    # --------------------------------------------------------
    # Intensity
    # --------------------------------------------------------

    intensity_mean = float(
        np.mean(
            local_image
        )
    )

    intensity_std = float(
        np.std(
            local_image
        )
    )

    intensity_min = float(
        np.min(
            local_image
        )
    )

    intensity_max = float(
        np.max(
            local_image
        )
    )

    intensity_median = float(
        np.median(
            local_image
        )
    )

    intensity_range = (
        intensity_max
        - intensity_min
    )

    # --------------------------------------------------------
    # Model probabilities
    # --------------------------------------------------------

    mean_probabilities = (
        np.mean(
            local_probabilities,
            axis=(1, 2, 3),
        )
    )

    max_probabilities = (
        np.max(
            local_probabilities,
            axis=(1, 2, 3),
        )
    )

    true_class = int(
        point["class_id"]
    )

    point_probabilities = (
        probabilities[
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

    return {

        "z": z,
        "y": y,
        "x": x,

        "intensity_mean":
            intensity_mean,

        "intensity_std":
            intensity_std,

        "intensity_min":
            intensity_min,

        "intensity_max":
            intensity_max,

        "intensity_median":
            intensity_median,

        "intensity_range":
            intensity_range,

        "gradient_mean":
            float(
                np.mean(
                    gradient_magnitude
                )
            ),

        "gradient_std":
            float(
                np.std(
                    gradient_magnitude
                )
            ),

        "foreground_ratio":
            float(
                np.mean(
                    local_prediction != 0
                )
            ),

        "background_ratio":
            float(
                np.mean(
                    local_prediction == 0
                )
            ),

        "point_true_probability":
            float(
                point_probabilities[
                    true_class
                ]
            ),

        "point_predicted_probability":
            float(
                point_probabilities[
                    predicted_class
                ]
            ),

        "point_background_probability":
            float(
                point_probabilities[0]
            ),

        "point_lfnn_probability":
            float(
                point_probabilities[LFNN]
            ),

        "point_rfnn_probability":
            float(
                point_probabilities[RFNN]
            ),

        "mean_background_probability":
            float(
                mean_probabilities[0]
            ),

        "mean_lfnn_probability":
            float(
                mean_probabilities[LFNN]
            ),

        "mean_rfnn_probability":
            float(
                mean_probabilities[RFNN]
            ),

        "max_background_probability":
            float(
                max_probabilities[0]
            ),

        "max_lfnn_probability":
            float(
                max_probabilities[LFNN]
            ),

        "max_rfnn_probability":
            float(
                max_probabilities[RFNN]
            ),

        "predicted_class":
            predicted_class,

        "predicted_class_name":
            CLASS_NAMES[
                predicted_class
            ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "PART 2.30 — LFNN/RFNN LOCAL "
        "ANATOMICAL CONTEXT AUDIT"
    )
    print("=" * 72)

    print(
        "GPU:",
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
    )

    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print()

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = p220b.load_manifest()

    print(
        f"Manifest rows: {len(manifest)}"
    )

    validation_groups = (
        build_validation_groups(
            manifest
        )
    )

    print(
        f"Validation cases: "
        f"{len(validation_groups)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = p220b.build_model()

    load_checkpoint(
        model
    )

    model.to(
        p220b.DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Records
    # --------------------------------------------------------

    records = []

    processed = 0

    # --------------------------------------------------------
    # Cases
    # --------------------------------------------------------

    for (
        study_id,
        series_id,
        point_df,
    ) in validation_groups:

        image, points, geometry = (
            p220b.load_case(
                study_id,
                series_id,
                point_df,
            )
        )

        probabilities, prediction = (
            predict_volume(
                model,
                image,
            )
        )

        for point in points:

            class_id = int(
                point["class_id"]
            )

            if class_id not in {
                LFNN,
                RFNN,
            }:
                continue

            base = {

                "study_id":
                    study_id,

                "series_id":
                    series_id,

                "level":
                    str(
                        point["level"]
                    ),

                "true_class":
                    class_id,

                "true_class_name":
                    CLASS_NAMES[
                        class_id
                    ],

                "patient_x":
                    safe_float(
                        point["patient_x"]
                    ),

                "patient_y":
                    safe_float(
                        point["patient_y"]
                    ),

                "patient_z":
                    safe_float(
                        point["patient_z"]
                    ),

                "canonical_x":
                    safe_float(
                        point["x"]
                    ),

                "canonical_y":
                    safe_float(
                        point["y"]
                    ),

                "canonical_z":
                    safe_float(
                        point["z"]
                    ),
            }

            for radius in (
                2,
                4,
                6,
            ):

                stats = local_statistics(
                    image,
                    probabilities,
                    prediction,
                    point,
                    radius,
                )

                record = base.copy()

                record["radius"] = radius

                record.update(
                    stats
                )

                records.append(
                    record
                )

        processed += 1

        print(
            f"Processed {processed:02d}/"
            f"{len(validation_groups):02d} "
            f"| study={study_id} "
            f"| series={series_id}"
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        records
    )

    if df.empty:
        raise RuntimeError(
            "No LFNN/RFNN records generated."
        )

    point_table = (
        TABLE_DIR
        / "part230_lfnn_rfnn_local_context.csv"
    )

    df.to_csv(
        point_table,
        index=False,
    )

    # ========================================================
    # SUMMARY BY DISEASE + RADIUS
    # ========================================================

    numeric_columns = [

        "intensity_mean",
        "intensity_std",
        "intensity_range",

        "gradient_mean",
        "gradient_std",

        "foreground_ratio",
        "background_ratio",

        "point_true_probability",
        "point_background_probability",

        "point_lfnn_probability",
        "point_rfnn_probability",

        "mean_background_probability",
        "mean_lfnn_probability",
        "mean_rfnn_probability",

        "max_background_probability",
        "max_lfnn_probability",
        "max_rfnn_probability",
    ]

    summary = (
        df
        .groupby(
            [
                "true_class_name",
                "radius",
            ]
        )[numeric_columns]
        .mean()
        .reset_index()
    )

    summary.to_csv(
        TABLE_DIR
        / "part230_local_context_summary.csv",
        index=False,
    )

    # ========================================================
    # LFNN VS RFNN DIFFERENCE
    # ========================================================

    difference_rows = []

    for radius in (
        2,
        4,
        6,
    ):

        current = df[
            df["radius"]
            == radius
        ]

        left = current[
            current["true_class"]
            == LFNN
        ]

        right = current[
            current["true_class"]
            == RFNN
        ]

        row = {
            "radius": radius,
            "lfnn_points": len(left),
            "rfnn_points": len(right),
        }

        for column in numeric_columns:

            lfnn_mean = (
                left[column].mean()
            )

            rfnn_mean = (
                right[column].mean()
            )

            row[
                f"lfnn_{column}"
            ] = lfnn_mean

            row[
                f"rfnn_{column}"
            ] = rfnn_mean

            row[
                f"rfnn_minus_lfnn_{column}"
            ] = (
                rfnn_mean
                - lfnn_mean
            )

        difference_rows.append(
            row
        )

    difference_df = pd.DataFrame(
        difference_rows
    )

    difference_df.to_csv(
        TABLE_DIR
        / "part230_lfnn_vs_rfnn_difference.csv",
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    lines = []

    lines.append(
        "PART 2.30 — LFNN/RFNN LOCAL "
        "ANATOMICAL CONTEXT AUDIT"
    )

    lines.append("")

    lines.append(
        f"Validation cases: "
        f"{processed}/{len(validation_groups)}"
    )

    lines.append(
        "LFNN/RFNN points: 90"
    )

    lines.append("")

    lines.append(
        "SUMMARY BY DISEASE AND RADIUS"
    )

    lines.append(
        summary.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "LFNN VS RFNN DIFFERENCES"
    )

    lines.append(
        difference_df.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "Training performed: NO"
    )

    lines.append(
        "Checkpoint modified: NO"
    )

    lines.append(
        "Part 2.20B modified: NO"
    )

    lines.append(
        "Part104 modified: NO"
    )

    lines.append(
        "Dashboard modified: NO"
    )

    lines.append(
        "Voxel ground truth fabricated: NO"
    )

    lines.append("")

    lines.append(
        "RSNA coordinates remain "
        "point/localization annotations."
    )

    lines.append(
        "This analysis does not establish "
        "voxel-wise segmentation accuracy."
    )

    report_path = (
        REPORT_DIR
        / "part230_report.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "PART 2.30 COMPLETE"
    )
    print("=" * 72)

    print(
        f"Processed cases: "
        f"{processed}/{len(validation_groups)}"
    )

    print(
        f"LFNN/RFNN records: "
        f"{len(df)}"
    )

    print(
        f"Outputs: {OUTPUT_DIR}"
    )

    print()
    print(
        "Training performed: NO"
    )

    print(
        "Checkpoint modified: NO"
    )

    print(
        "Dashboard modified: NO"
    )


if __name__ == "__main__":
    main()