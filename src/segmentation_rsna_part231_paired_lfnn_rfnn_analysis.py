"""
PART 2.31
Paired LFNN/RFNN Anatomical Context Analysis

Purpose
-------
Compare LFNN and RFNN points that belong to the SAME:

    study_id
    series_id
    spinal level

This removes much of the patient-to-patient variability.

Analysis only:
    - no training
    - no checkpoint modification
    - no dashboard modification
    - no fabricated voxel ground truth
"""

from pathlib import Path
import sys
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
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part231_paired_lfnn_rfnn_analysis"
)

TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
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


# ============================================================
# HELPERS
# ============================================================

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
        f"Checkpoint missing keys: "
        f"{len(missing)}"
    )

    print(
        f"Checkpoint unexpected keys: "
        f"{len(unexpected)}"
    )


def safe_float(x):

    try:

        x = float(x)

        if math.isfinite(x):
            return x

    except Exception:
        pass

    return np.nan


def build_validation_manifest(manifest):

    validation = (
        p220b.select_validation_series(
            manifest
        )
    )

    if isinstance(
        validation,
        pd.DataFrame,
    ):

        ids = (
            validation[
                [
                    "study_id",
                    "series_id",
                ]
            ]
            .drop_duplicates()
            .copy()
        )

    else:

        rows = []

        for item in validation:

            if isinstance(
                item,
                dict,
            ):

                rows.append(
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
                        list,
                        tuple,
                    ),
                )
                and len(item) >= 2
            ):

                rows.append(
                    (
                        str(item[0]),
                        str(item[1]),
                    )
                )

        ids = pd.DataFrame(
            rows,
            columns=[
                "study_id",
                "series_id",
            ],
        )

    m = manifest.copy()

    m["study_id"] = (
        m["study_id"]
        .astype(str)
    )

    m["series_id"] = (
        m["series_id"]
        .astype(str)
    )

    ids["study_id"] = (
        ids["study_id"]
        .astype(str)
    )

    ids["series_id"] = (
        ids["series_id"]
        .astype(str)
    )

    result = m.merge(
        ids[
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

    return result


@torch.no_grad()
def predict(
    model,
    image,
):

    model.eval()

    x = torch.from_numpy(
        image.astype(
            np.float32
        )
    )

    x = (
        x
        .unsqueeze(0)
        .unsqueeze(0)
        .to(
            p220b.DEVICE
        )
    )

    with torch.amp.autocast(
        "cuda",
        enabled=torch.cuda.is_available(),
    ):

        logits = model(x)

    if isinstance(
        logits,
        (tuple, list),
    ):

        logits = logits[0]

    probs = torch.softmax(
        logits,
        dim=1,
    )[0]

    probs = (
        probs
        .detach()
        .cpu()
        .numpy()
    )

    pred = np.argmax(
        probs,
        axis=0,
    )

    return probs, pred


def point_model_values(
    probabilities,
    prediction,
    point,
):

    z = int(
        np.clip(
            round(
                float(
                    point["z"]
                )
            ),
            0,
            probabilities.shape[1] - 1,
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
            probabilities.shape[2] - 1,
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
            probabilities.shape[3] - 1,
        )
    )

    p = probabilities[
        :,
        z,
        y,
        x,
    ]

    predicted_class = int(
        np.argmax(p)
    )

    return {
        "z": z,
        "y": y,
        "x": x,
        "predicted_class":
            predicted_class,
        "background_probability":
            float(p[0]),
        "lfnn_probability":
            float(p[LFNN]),
        "rfnn_probability":
            float(p[RFNN]),
        "true_probability":
            float(
                p[
                    int(
                        point[
                            "class_id"
                        ]
                    )
                ]
            ),
        "foreground_probability":
            float(
                1.0 - p[0]
            ),
    }


def local_stats(
    image,
    probabilities,
    prediction,
    point,
    radius,
):

    z = int(
        np.clip(
            round(
                float(
                    point["z"]
                )
            ),
            0,
            image.shape[0] - 1,
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
            image.shape[1] - 1,
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
            image.shape[2] - 1,
        )
    )

    z0 = max(
        0,
        z - radius,
    )

    z1 = min(
        image.shape[0],
        z + radius + 1,
    )

    y0 = max(
        0,
        y - radius,
    )

    y1 = min(
        image.shape[1],
        y + radius + 1,
    )

    x0 = max(
        0,
        x - radius,
    )

    x1 = min(
        image.shape[2],
        x + radius + 1,
    )

    patch = image[
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    patch_prediction = prediction[
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    patch_prob = probabilities[
        :,
        z0:z1,
        y0:y1,
        x0:x1,
    ]

    gradients = np.gradient(
        patch.astype(
            np.float32
        )
    )

    gradient = np.sqrt(
        gradients[0] ** 2
        + gradients[1] ** 2
        + gradients[2] ** 2
    )

    return {

        "intensity_mean":
            float(np.mean(patch)),

        "intensity_std":
            float(np.std(patch)),

        "intensity_range":
            float(
                np.max(patch)
                - np.min(patch)
            ),

        "gradient_mean":
            float(np.mean(gradient)),

        "gradient_std":
            float(np.std(gradient)),

        "foreground_ratio":
            float(
                np.mean(
                    patch_prediction != 0
                )
            ),

        "background_ratio":
            float(
                np.mean(
                    patch_prediction == 0
                )
            ),

        "mean_background_probability":
            float(
                np.mean(
                    patch_prob[0]
                )
            ),

        "mean_lfnn_probability":
            float(
                np.mean(
                    patch_prob[LFNN]
                )
            ),

        "mean_rfnn_probability":
            float(
                np.mean(
                    patch_prob[RFNN]
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "PART 2.31 — PAIRED LFNN/RFNN "
        "ANATOMICAL CONTEXT ANALYSIS"
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

    manifest = (
        p220b.load_manifest()
    )

    validation_manifest = (
        build_validation_manifest(
            manifest
        )
    )

    print(
        f"Validation cases: "
        f"{validation_manifest[['study_id','series_id']].drop_duplicates().shape[0]}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        p220b.build_model()
    )

    load_checkpoint(
        model
    )

    model.to(
        p220b.DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Process cases
    # --------------------------------------------------------

    paired_records = []

    case_count = 0

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

        # Important:
        # retain native coordinates from manifest.
        image, points, geometry = (
            p220b.load_case(
                str(study_id),
                str(series_id),
                group.copy(),
            )
        )

        probabilities, prediction = (
            predict(
                model,
                image,
            )
        )

        # ----------------------------------------------------
        # Convert returned point list into lookup
        # ----------------------------------------------------

        point_lookup = {}

        for point in points:

            class_id = int(
                point["class_id"]
            )

            if class_id not in {
                LFNN,
                RFNN,
            }:
                continue

            key = (
                str(
                    point["level"]
                ),
                class_id,
            )

            point_lookup[key] = point

        # ----------------------------------------------------
        # Same level LFNN/RFNN pairing
        # ----------------------------------------------------

        levels = sorted(
            {
                level
                for (
                    level,
                    class_id
                ) in point_lookup.keys()
            }
        )

        for level in levels:

            left = point_lookup.get(
                (
                    level,
                    LFNN,
                )
            )

            right = point_lookup.get(
                (
                    level,
                    RFNN,
                )
            )

            if left is None or right is None:
                continue

            # ----------------------------------------------
            # Physical-space distance
            # ----------------------------------------------

            physical_distance = float(
                np.sqrt(
                    (
                        left["patient_x"]
                        - right["patient_x"]
                    ) ** 2
                    +
                    (
                        left["patient_y"]
                        - right["patient_y"]
                    ) ** 2
                    +
                    (
                        left["patient_z"]
                        - right["patient_z"]
                    ) ** 2
                )
            )

            canonical_distance = float(
                np.sqrt(
                    (
                        left["x"]
                        - right["x"]
                    ) ** 2
                    +
                    (
                        left["y"]
                        - right["y"]
                    ) ** 2
                    +
                    (
                        left["z"]
                        - right["z"]
                    ) ** 2
                )
            )

            left_model = (
                point_model_values(
                    probabilities,
                    prediction,
                    left,
                )
            )

            right_model = (
                point_model_values(
                    probabilities,
                    prediction,
                    right,
                )
            )

            record = {

                "study_id":
                    str(study_id),

                "series_id":
                    str(series_id),

                "level":
                    str(level),

                "physical_distance":
                    physical_distance,

                "canonical_distance":
                    canonical_distance,

                # ----------------------------
                # Coordinates
                # ----------------------------

                "lfnn_patient_x":
                    safe_float(
                        left["patient_x"]
                    ),

                "rfnn_patient_x":
                    safe_float(
                        right["patient_x"]
                    ),

                "lfnn_patient_y":
                    safe_float(
                        left["patient_y"]
                    ),

                "rfnn_patient_y":
                    safe_float(
                        right["patient_y"]
                    ),

                "lfnn_patient_z":
                    safe_float(
                        left["patient_z"]
                    ),

                "rfnn_patient_z":
                    safe_float(
                        right["patient_z"]
                    ),

                # ----------------------------
                # Model point response
                # ----------------------------

                "lfnn_true_probability":
                    left_model[
                        "true_probability"
                    ],

                "rfnn_true_probability":
                    right_model[
                        "true_probability"
                    ],

                "lfnn_background_probability":
                    left_model[
                        "background_probability"
                    ],

                "rfnn_background_probability":
                    right_model[
                        "background_probability"
                    ],

                "lfnn_lfnn_probability":
                    left_model[
                        "lfnn_probability"
                    ],

                "rfnn_lfnn_probability":
                    right_model[
                        "lfnn_probability"
                    ],

                "lfnn_rfnn_probability":
                    left_model[
                        "rfnn_probability"
                    ],

                "rfnn_rfnn_probability":
                    right_model[
                        "rfnn_probability"
                    ],

                "lfnn_foreground_probability":
                    left_model[
                        "foreground_probability"
                    ],

                "rfnn_foreground_probability":
                    right_model[
                        "foreground_probability"
                    ],

                "lfnn_predicted_class":
                    left_model[
                        "predicted_class"
                    ],

                "rfnn_predicted_class":
                    right_model[
                        "predicted_class"
                    ],
            }

            # ------------------------------------------------
            # Local context radii
            # ------------------------------------------------

            for radius in (
                2,
                4,
                6,
            ):

                left_stats = (
                    local_stats(
                        image,
                        probabilities,
                        prediction,
                        left,
                        radius,
                    )
                )

                right_stats = (
                    local_stats(
                        image,
                        probabilities,
                        prediction,
                        right,
                        radius,
                    )
                )

                for key, value in (
                    left_stats.items()
                ):

                    record[
                        f"lfnn_r{radius}_{key}"
                    ] = value

                for key, value in (
                    right_stats.items()
                ):

                    record[
                        f"rfnn_r{radius}_{key}"
                    ] = value

                # ------------------------------------------------
                # Paired difference
                # RFNN - LFNN
                # ------------------------------------------------

                for key in left_stats:

                    record[
                        f"rfnn_minus_lfnn_r{radius}_{key}"
                    ] = (
                        right_stats[key]
                        - left_stats[key]
                    )

            paired_records.append(
                record
            )

        case_count += 1

        print(
            f"Processed "
            f"{case_count:02d}/"
            f"{validation_manifest[['study_id','series_id']].drop_duplicates().shape[0]:02d}"
            f" | study={study_id}"
            f" | series={series_id}"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    df = pd.DataFrame(
        paired_records
    )

    if df.empty:
        raise RuntimeError(
            "No paired LFNN/RFNN observations found."
        )

    pair_path = (
        TABLE_DIR
        / "part231_paired_observations.csv"
    )

    df.to_csv(
        pair_path,
        index=False,
    )

    # --------------------------------------------------------
    # Paired summary
    # --------------------------------------------------------

    summary_rows = []

    for radius in (
        2,
        4,
        6,
    ):

        row = {
            "radius": radius,
            "paired_observations": len(df),
        }

        metrics = [

            "intensity_mean",
            "intensity_std",
            "intensity_range",
            "gradient_mean",
            "gradient_std",
            "foreground_ratio",
            "background_ratio",
            "mean_background_probability",
            "mean_lfnn_probability",
            "mean_rfnn_probability",
        ]

        for metric in metrics:

            left_col = (
                f"lfnn_r{radius}_{metric}"
            )

            right_col = (
                f"rfnn_r{radius}_{metric}"
            )

            diff_col = (
                f"rfnn_minus_lfnn_r{radius}_{metric}"
            )

            row[
                f"lfnn_{metric}"
            ] = df[left_col].mean()

            row[
                f"rfnn_{metric}"
            ] = df[right_col].mean()

            row[
                f"rfnn_minus_lfnn_{metric}"
            ] = df[diff_col].mean()

            row[
                f"absolute_difference_{metric}"
            ] = df[diff_col].abs().mean()

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        TABLE_DIR
        / "part231_paired_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Level-wise analysis
    # --------------------------------------------------------

    level_rows = []

    for level, group in df.groupby(
        "level",
        sort=True,
    ):

        level_rows.append({

            "level":
                level,

            "paired_observations":
                len(group),

            "mean_physical_distance":
                group[
                    "physical_distance"
                ].mean(),

            "mean_canonical_distance":
                group[
                    "canonical_distance"
                ].mean(),

            "r2_foreground_difference":
                group[
                    "rfnn_minus_lfnn_r2_foreground_ratio"
                ].mean(),

            "r4_foreground_difference":
                group[
                    "rfnn_minus_lfnn_r4_foreground_ratio"
                ].mean(),

            "r6_foreground_difference":
                group[
                    "rfnn_minus_lfnn_r6_foreground_ratio"
                ].mean(),

            "r2_true_probability_difference":
                (
                    group[
                        "rfnn_true_probability"
                    ]
                    - group[
                        "lfnn_true_probability"
                    ]
                ).mean(),

        })

    level_df = pd.DataFrame(
        level_rows
    )

    level_df.to_csv(
        TABLE_DIR
        / "part231_level_analysis.csv",
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "PART 2.31 — PAIRED LFNN/RFNN "
        "ANATOMICAL CONTEXT ANALYSIS"
    )

    report.append("")

    report.append(
        f"Validation cases processed: "
        f"{case_count}"
    )

    report.append(
        f"Paired LFNN/RFNN observations: "
        f"{len(df)}"
    )

    report.append("")

    report.append(
        "PAIRING DEFINITION:"
    )

    report.append(
        "Same study_id + same series_id + same spinal level."
    )

    report.append("")

    report.append(
        "PAIRED SUMMARY:"
    )

    report.append(
        summary.to_string(
            index=False
        )
    )

    report.append("")

    report.append(
        "LEVEL-WISE ANALYSIS:"
    )

    report.append(
        level_df.to_string(
            index=False
        )
    )

    report.append("")

    report.append(
        "IMPORTANT:"
    )

    report.append(
        "This is an analysis-only experiment."
    )

    report.append(
        "No model training was performed."
    )

    report.append(
        "No checkpoint was modified."
    )

    report.append(
        "No dashboard was modified."
    )

    report.append(
        "No voxel-wise ground truth was fabricated."
    )

    report.append("")

    report.append(
        "RSNA coordinates remain point/localization "
        "annotations rather than manual voxel masks."
    )

    report_path = (
        REPORT_DIR
        / "part231_report.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 72)
    print(
        "PART 2.31 COMPLETE"
    )
    print("=" * 72)

    print(
        f"Validation cases: "
        f"{case_count}"
    )

    print(
        f"Paired observations: "
        f"{len(df)}"
    )

    print(
        f"Output directory:"
    )

    print(
        OUTPUT_DIR
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