"""
PART 2.28 — RFNN/LFNN FAILURE ANALYSIS

Purpose
-------
Analysis-only investigation of the persistent foraminal failure observed
in Part 2.27.

This script:
    1. Reuses the exact Part 2.20B geometry-corrected data pipeline.
    2. Uses the Part 2.27 best checkpoint.
    3. Evaluates the fixed 25-case validation cohort.
    4. Separates LFNN and RFNN points.
    5. Records predicted class/probability.
    6. Measures local neighborhood statistics.
    7. Reports LFNN/RFNN confusion and background failures.
    8. Does NOT train.
    9. Does NOT modify any checkpoint.
   10. Does NOT modify the dashboard.
   11. Does NOT fabricate voxel ground truth.

RSNA coordinates remain point/localization annotations.
"""

from pathlib import Path
import sys
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT EXACT PART 2.20B PIPELINE
# ============================================================

from src import segmentation_rsna_part220b_geometry_corrected_training as p220b


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part228_rfnn_lfnn_failure_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
    / "part227_best_macro_disease.pth"
)


# ============================================================
# CLASS CONTRACT
# ============================================================

CLASS_NAMES = p220b.CLASS_NAMES

LFNN_CLASS = 2
RFNN_CLASS = 3


# ============================================================
# UTILITY
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
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Part 2.27 checkpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            f"Check the checkpoint filename in:\n"
            f"{CHECKPOINT_PATH.parent}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=p220b.DEVICE,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"Checkpoint missing keys: {len(missing)}"
    )

    print(
        f"Checkpoint unexpected keys: {len(unexpected)}"
    )

    if missing:
        print("Missing keys:")
        for key in missing[:10]:
            print("  ", key)

    if unexpected:
        print("Unexpected keys:")
        for key in unexpected[:10]:
            print("  ", key)


# ============================================================
# MODEL PREDICTION
# ============================================================

@torch.no_grad()
def predict_volume(model, image):
    """
    Run the model using the same model-grid image returned by
    Part 2.20B load_case().
    """

    model.eval()

    tensor = torch.from_numpy(
        image.astype(np.float32)
    )

    if tensor.ndim != 3:
        raise ValueError(
            f"Expected 3D image, received shape={tensor.shape}"
        )

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(
        p220b.DEVICE,
        non_blocking=True,
    )

    with torch.amp.autocast(
        "cuda",
        enabled=torch.cuda.is_available(),
    ):
        logits = model(tensor)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    probabilities = probabilities[0]

    prediction = torch.argmax(
        probabilities,
        dim=0,
    )

    return (
        probabilities.detach().cpu().numpy(),
        prediction.detach().cpu().numpy(),
    )


# ============================================================
# LOCAL NEIGHBORHOOD
# ============================================================

def extract_local_statistics(
    probabilities,
    prediction,
    point,
):
    """
    Extract local statistics around one annotated point.

    Point coordinates are the exact canonical model coordinates
    returned by Part 2.20B load_case():
        z, y, x
    """

    z = int(round(float(point["z"])))
    y = int(round(float(point["y"])))
    x = int(round(float(point["x"])))

    depth = probabilities.shape[1]
    height = probabilities.shape[2]
    width = probabilities.shape[3]

    z = int(np.clip(z, 0, depth - 1))
    y = int(np.clip(y, 0, height - 1))
    x = int(np.clip(x, 0, width - 1))

    result = {
        "z": z,
        "y": y,
        "x": x,
    }

    # --------------------------------------------------------
    # Point prediction
    # --------------------------------------------------------

    point_probs = probabilities[:, z, y, x]

    point_pred = int(
        np.argmax(point_probs)
    )

    result["predicted_class"] = point_pred
    result["predicted_class_name"] = CLASS_NAMES[
        point_pred
    ]

    result["true_probability"] = safe_float(
        point_probs[
            int(point["class_id"])
        ]
    )

    result["predicted_probability"] = safe_float(
        point_probs[point_pred]
    )

    # --------------------------------------------------------
    # Local neighborhoods
    # --------------------------------------------------------

    for radius in (1, 2, 4):

        z0 = max(0, z - radius)
        z1 = min(depth, z + radius + 1)

        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)

        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)

        local_probs = probabilities[
            :,
            z0:z1,
            y0:y1,
            x0:x1,
        ]

        local_prediction = prediction[
            z0:z1,
            y0:y1,
            x0:x1,
        ]

        local_true_probability = local_probs[
            int(point["class_id"])
        ]

        local_max_class = int(
            np.argmax(
                local_probs.reshape(
                    local_probs.shape[0],
                    -1,
                ).mean(axis=1)
            )
        )

        result[
            f"radius{radius}_true_mean_probability"
        ] = safe_float(
            np.mean(
                local_true_probability
            )
        )

        result[
            f"radius{radius}_true_max_probability"
        ] = safe_float(
            np.max(
                local_true_probability
            )
        )

        result[
            f"radius{radius}_predicted_class"
        ] = local_max_class

        result[
            f"radius{radius}_predicted_class_name"
        ] = CLASS_NAMES[
            local_max_class
        ]

        result[
            f"radius{radius}_foreground_ratio"
        ] = safe_float(
            np.mean(
                local_prediction != 0
            )
        )

    return result


# ============================================================
# BUILD VALIDATION CASES
# ============================================================

def build_validation_groups(manifest):
    """
    Build validation groups while preserving the COMPLETE
    Part 2.13 manifest rows.

    Important:
        Part 2.20B load_case() requires:
            native_z
            native_y
            native_x

    Therefore, validation selection is used only to identify
    the validation study/series pairs. The original manifest
    is then used to recover the complete point rows.
    """

    validation_series = (
        p220b.select_validation_series(
            manifest
        )
    )

    # --------------------------------------------------------
    # Extract validation study/series identifiers
    # --------------------------------------------------------

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

        # If the helper returns identifiers rather than a
        # DataFrame, handle the possible structures explicitly.

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

            elif isinstance(
                item,
                (tuple, list),
            ) and len(item) >= 2:

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

    # --------------------------------------------------------
    # Normalize identifiers
    # --------------------------------------------------------

    manifest_copy = manifest.copy()

    manifest_copy[
        "study_id"
    ] = manifest_copy[
        "study_id"
    ].astype(str)

    manifest_copy[
        "series_id"
    ] = manifest_copy[
        "series_id"
    ].astype(str)

    validation_ids[
        "study_id"
    ] = validation_ids[
        "study_id"
    ].astype(str)

    validation_ids[
        "series_id"
    ] = validation_ids[
        "series_id"
    ].astype(str)

    # --------------------------------------------------------
    # Keep only validation study/series pairs
    # while retaining ALL original manifest columns.
    # --------------------------------------------------------

    validation_manifest = manifest_copy.merge(
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

    # --------------------------------------------------------
    # Verify required Part 2.20B coordinates
    # --------------------------------------------------------

    required_columns = [
        "native_z",
        "native_y",
        "native_x",
        "class_id",
        "class_name",
        "level",
        "study_id",
        "series_id",
    ]

    missing = [
        column
        for column in required_columns
        if column not in validation_manifest.columns
    ]

    if missing:
        raise RuntimeError(
            "Part 2.28 validation manifest is missing "
            "required Part 2.20B columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # --------------------------------------------------------
    # Build groups
    # --------------------------------------------------------

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


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 72)
    print(
        "PART 2.28 — RFNN/LFNN FAILURE ANALYSIS"
    )
    print("=" * 72)

    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    print(
        f"Checkpoint:\n{CHECKPOINT_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = p220b.load_manifest()

    print(
        f"Manifest rows: {len(manifest)}"
    )

    # --------------------------------------------------------
    # Validation cases
    # --------------------------------------------------------

    validation_groups = (
        build_validation_groups(
            manifest
        )
    )

    print(
        f"Validation cases: {len(validation_groups)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = p220b.build_model()

    load_checkpoint(model)

    model.to(
        p220b.DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Analysis records
    # --------------------------------------------------------

    records = []

    processed_cases = 0

    # --------------------------------------------------------
    # Case loop
    # --------------------------------------------------------

    for (
        study_id,
        series_id,
        point_df,
    ) in validation_groups:

        try:

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

            # Only LFNN and RFNN
            target_points = [
    point
    for point in points
    if int(point["class_id"])
    in {
        LFNN_CLASS,
        RFNN_CLASS,
    }
]

            for point in target_points:

                stats = (
                    extract_local_statistics(
                        probabilities,
                        prediction,
                        point,
                    )
                )

                record = {
    "study_id": study_id,
    "series_id": series_id,
    "condition": point[
        "class_name"
    ],
    "series_description": "Part220B canonical validation series",
                    "true_class": int(
                        point["class_id"]
                    ),
                    "true_class_name": CLASS_NAMES[
                        int(point["class_id"])
                    ],
                    "level": point["level"],
                    "patient_x": safe_float(
                        point.get(
                            "patient_x",
                            np.nan,
                        )
                    ),
                    "patient_y": safe_float(
                        point.get(
                            "patient_y",
                            np.nan,
                        )
                    ),
                    "patient_z": safe_float(
                        point.get(
                            "patient_z",
                            np.nan,
                        )
                    ),
                    "model_z": safe_float(
                        point["z"]
                    ),
                    "model_y": safe_float(
                        point["y"]
                    ),
                    "model_x": safe_float(
                        point["x"]
                    ),
                }

                record.update(
                    stats
                )

                record["correct"] = int(
                    record[
                        "predicted_class"
                    ]
                    == record[
                        "true_class"
                    ]
                )

                record[
                    "rfnn_or_lfnn_pair"
                ] = (
                    "LFNN"
                    if record[
                        "true_class"
                    ] == LFNN_CLASS
                    else "RFNN"
                )

                records.append(
                    record
                )

            processed_cases += 1

            print(
                f"Processed {processed_cases:02d}/"
                f"{len(validation_groups):02d} "
                f"| study={study_id} "
                f"| series={series_id}"
            )

        except Exception as exc:

            print(
                f"[WARNING] Failed case "
                f"{study_id}/{series_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    results = pd.DataFrame(
        records
    )

    if results.empty:
        raise RuntimeError(
            "No LFNN/RFNN analysis records were generated."
        )

    results_path = (
        TABLE_DIR
        / "part228_rfnn_lfnn_point_analysis.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # Confusion
    # --------------------------------------------------------

    confusion = pd.crosstab(
        results["true_class_name"],
        results["predicted_class_name"],
        dropna=False,
    )

    confusion_path = (
        TABLE_DIR
        / "part228_rfnn_lfnn_confusion.csv"
    )

    confusion.to_csv(
        confusion_path
    )

    # --------------------------------------------------------
    # Disease summary
    # --------------------------------------------------------

    disease_summary = (
        results
        .groupby(
            "true_class_name"
        )
        .agg(
            points=(
                "true_class_name",
                "size",
            ),
            accuracy=(
                "correct",
                "mean",
            ),
            mean_true_probability=(
                "true_probability",
                "mean",
            ),
            median_true_probability=(
                "true_probability",
                "median",
            ),
            mean_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            mean_foreground_radius2=(
                "radius2_foreground_ratio",
                "mean",
            ),
            mean_foreground_radius4=(
                "radius4_foreground_ratio",
                "mean",
            ),
        )
        .reset_index()
    )

    disease_path = (
        TABLE_DIR
        / "part228_disease_summary.csv"
    )

    disease_summary.to_csv(
        disease_path,
        index=False,
    )

    # --------------------------------------------------------
    # RFNN predicted classes
    # --------------------------------------------------------

    rfnn_results = results[
        results["true_class"]
        == RFNN_CLASS
    ]

    lfnn_results = results[
        results["true_class"]
        == LFNN_CLASS
    ]

    rfnn_confusion = (
        rfnn_results[
            "predicted_class_name"
        ]
        .value_counts()
        .rename_axis(
            "predicted_class_name"
        )
        .reset_index(
            name="count"
        )
    )

    lfnn_confusion = (
        lfnn_results[
            "predicted_class_name"
        ]
        .value_counts()
        .rename_axis(
            "predicted_class_name"
        )
        .reset_index(
            name="count"
        )
    )

    rfnn_confusion.to_csv(
        TABLE_DIR
        / "part228_rfnn_predictions.csv",
        index=False,
    )

    lfnn_confusion.to_csv(
        TABLE_DIR
        / "part228_lfnn_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Pair comparison
    # --------------------------------------------------------

    pair_summary = {
        "LFNN_points": int(
            len(lfnn_results)
        ),
        "RFNN_points": int(
            len(rfnn_results)
        ),
        "LFNN_accuracy": safe_float(
            lfnn_results[
                "correct"
            ].mean()
        ),
        "RFNN_accuracy": safe_float(
            rfnn_results[
                "correct"
            ].mean()
        ),
        "LFNN_mean_true_probability":
            safe_float(
                lfnn_results[
                    "true_probability"
                ].mean()
            ),
        "RFNN_mean_true_probability":
            safe_float(
                rfnn_results[
                    "true_probability"
                ].mean()
            ),
        "LFNN_mean_foreground_radius2":
            safe_float(
                lfnn_results[
                    "radius2_foreground_ratio"
                ].mean()
            ),
        "RFNN_mean_foreground_radius2":
            safe_float(
                rfnn_results[
                    "radius2_foreground_ratio"
                ].mean()
            ),
        "LFNN_mean_foreground_radius4":
            safe_float(
                lfnn_results[
                    "radius4_foreground_ratio"
                ].mean()
            ),
        "RFNN_mean_foreground_radius4":
            safe_float(
                rfnn_results[
                    "radius4_foreground_ratio"
                ].mean()
            ),
    }

    with open(
        REPORT_DIR
        / "part228_pair_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            pair_summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    report_lines = []

    report_lines.append(
        "PART 2.28 — RFNN/LFNN FAILURE ANALYSIS"
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        f"Validation cases processed: "
        f"{processed_cases}/{len(validation_groups)}"
    )

    report_lines.append(
        f"LFNN/RFNN points analyzed: "
        f"{len(results)}"
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        "DISEASE SUMMARY"
    )

    report_lines.append(
        disease_summary.to_string(
            index=False
        )
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        "RFNN PREDICTION DISTRIBUTION"
    )

    report_lines.append(
        rfnn_confusion.to_string(
            index=False
        )
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        "LFNN PREDICTION DISTRIBUTION"
    )

    report_lines.append(
        lfnn_confusion.to_string(
            index=False
        )
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        "LFNN vs RFNN SUMMARY"
    )

    for key, value in pair_summary.items():

        report_lines.append(
            f"{key}: {value}"
        )

    report_lines.append(
        ""
    )

    report_lines.append(
        "Training performed: NO"
    )

    report_lines.append(
        "Checkpoint modified: NO"
    )

    report_lines.append(
        "Part 2.20B modified: NO"
    )

    report_lines.append(
        "Part104 modified: NO"
    )

    report_lines.append(
        "Dashboard modified: NO"
    )

    report_lines.append(
        "Voxel ground truth fabricated: NO"
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        "RSNA coordinates remain "
        "point/localization annotations."
    )

    report_lines.append(
        "This analysis does not establish "
        "voxel-wise segmentation accuracy."
    )

    report_path = (
        REPORT_DIR
        / "part228_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final console
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "PART 2.28 COMPLETE"
    )
    print("=" * 72)

    print(
        f"Processed cases: "
        f"{processed_cases}/{len(validation_groups)}"
    )

    print(
        f"LFNN/RFNN points analyzed: "
        f"{len(results)}"
    )

    print()
    print(
        "LFNN accuracy:",
        f"{pair_summary['LFNN_accuracy']:.6f}",
    )

    print(
        "RFNN accuracy:",
        f"{pair_summary['RFNN_accuracy']:.6f}",
    )

    print(
        "LFNN mean true probability:",
        f"{pair_summary['LFNN_mean_true_probability']:.6f}",
    )

    print(
        "RFNN mean true probability:",
        f"{pair_summary['RFNN_mean_true_probability']:.6f}",
    )

    print()
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

    print(
        "Voxel ground truth fabricated: NO"
    )


if __name__ == "__main__":
    main()